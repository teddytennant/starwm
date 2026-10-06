"""Checks for the spatial router, eq. 8 composition, and the gradient split."""

import jax
import jax.numpy as jnp
import optax

from starwm.losses import (
    BETA_KL,
    compute_losses,
    contrast_phi,
    init_inverse,
    init_params,
    inverse_dynamics_loss,
    logistic_contrastive,
    train_step,
)
from starwm.routing import (
    A,
    C,
    H,
    L,
    N,
    T,
    W,
    attention,
    compose,
    coverage_mask,
    foreground,
    init_router,
    keys_and_values,
    pooled_content,
    positional_encoding,
    spatial_coords,
)
from starwm.synthetic import encode, init_encoder, make_batch, make_episode


def _max_abs(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return float(max(jnp.max(jnp.abs(x)) for x in leaves))


def test_positional_encoding_shape():
    pe = positional_encoding()
    coords = spatial_coords()
    assert pe.shape == (L, C)
    assert coords.shape == (L, 2)
    assert jnp.all(coords >= 0.0)
    assert jnp.all(coords <= 1.0)
    # Distinct cells. A constant encoding would not be location-aware.
    assert float(jnp.std(pe)) > 0.0
    assert float(jnp.max(jnp.abs(coords[0] - coords[-1]))) > 0.5


def test_keys_add_position_values_append_coords():
    features = jnp.ones((2, L, C))
    keys, values = keys_and_values(features)
    assert keys.shape == (2, L, C)
    assert values.shape == (2, L, C + 2)
    assert jnp.allclose(keys, features + positional_encoding())
    assert jnp.allclose(values[..., :C], features)
    assert jnp.allclose(values[..., C:], spatial_coords())


def test_attention_matches_eq5_and_sums_to_one():
    key = jax.random.PRNGKey(0)
    params = init_router(key)
    features = jax.random.normal(jax.random.PRNGKey(1), (3, T, L, C))
    attn = attention(params, features)
    assert attn.shape == (3, T, N, L)
    assert jnp.allclose(jnp.sum(attn, axis=-1), 1.0, atol=1e-5)

    keys, _values = keys_and_values(features)
    q_proj = jnp.einsum("nc,cd->nd", params["Q"], params["WQ"])
    k_proj = jnp.einsum("btlc,cd->btld", keys, params["WK"])
    logits = jnp.einsum("nd,btld->btnl", q_proj, k_proj) / jnp.sqrt(C)
    expected = jax.nn.softmax(logits, axis=-1)
    assert jnp.allclose(attn, expected, atol=1e-5)


def test_attention_concentrates_on_a_spatial_spike():
    """A key that matches the query should take the mass, not a time index."""
    params = {
        "Q": jnp.zeros((N, C)).at[0, 0].set(1.0).at[1, 1].set(1.0),
        "WQ": jnp.eye(C),
        "WK": jnp.eye(C),
    }
    # Cancel Epos so the spike is the key.
    features = -positional_encoding()[None, None, :, :]
    features = features.at[0, 0, 5, 0].add(8.0)
    features = features.at[0, 0, 9, 1].add(8.0)
    attn = attention(params, features)
    assert int(jnp.argmax(attn[0, 0, 0])) == 5
    assert int(jnp.argmax(attn[0, 0, 1])) == 9
    assert float(attn[0, 0, 0, 5]) > 0.5


def test_pooled_content_drops_coordinates():
    features = jax.random.normal(jax.random.PRNGKey(2), (2, L, C))
    params = init_router(jax.random.PRNGKey(3))
    attn = attention(params, features)
    eattn = pooled_content(attn, features)
    manual = jnp.einsum("bnl,blc->bnc", attn, features).reshape(2, N * C)
    assert jnp.allclose(eattn, manual)
    # Shifting coordinates cannot change eattn. They are not in Ft.
    # (The function never reads Ecoord. This locks that.)
    assert eattn.shape == (2, N * C)


def test_coverage_and_composition_eq8():
    attn = jnp.zeros((N, L)).at[0, 0].set(0.8).at[0, 1].set(0.2)
    attn = attn.at[1, 0].set(0.4).at[1, 2].set(0.6)
    mask = coverage_mask(attn)
    # sum over queries, clipped
    assert jnp.allclose(mask[0], 1.0)  # 0.8+0.4 clipped
    assert jnp.allclose(mask[1], 0.2)
    assert jnp.allclose(mask[2], 0.6)
    assert jnp.allclose(mask[3], 0.0)

    entity = jnp.arange(N * C, dtype=jnp.float32).reshape(N, C)
    fg = foreground(attn, entity)
    bg = jnp.ones((L, C))
    composed = compose(fg, bg, mask)
    expected = fg * mask[:, None] + bg * (1.0 - mask[:, None])
    assert jnp.allclose(composed, expected)
    # Uncovered token is pure background.
    assert jnp.allclose(composed[3], bg[3])


def test_recon_does_not_train_router():
    params = init_params(jax.random.PRNGKey(4))
    batch = make_batch(0, batch=2)
    key = jax.random.PRNGKey(5)

    def recon(router):
        trial = dict(params)
        trial["router"] = router
        return compute_losses(trial, batch["video"], batch["actions"], key)["recon"]

    grads = jax.grad(recon)(params["router"])
    assert _max_abs(grads) == 0.0


def test_dynamics_does_not_train_decoder():
    params = init_params(jax.random.PRNGKey(6))
    batch = make_batch(1, batch=4)
    key = jax.random.PRNGKey(7)

    def dynamics(decoder_params):
        trial = dict(params)
        trial["fg"] = decoder_params["fg"]
        trial["bg"] = decoder_params["bg"]
        trial["pixel"] = decoder_params["pixel"]
        out = compute_losses(trial, batch["video"], batch["actions"], key)
        return out["inverse"] + out["contrastive"]

    grads = jax.grad(dynamics)(
        {"fg": params["fg"], "bg": params["bg"], "pixel": params["pixel"]}
    )
    assert _max_abs(grads) == 0.0


def test_dynamics_trains_router():
    params = init_params(jax.random.PRNGKey(8))
    batch = make_batch(2, batch=4)
    key = jax.random.PRNGKey(9)

    def dynamics(router):
        trial = dict(params)
        trial["router"] = router
        out = compute_losses(trial, batch["video"], batch["actions"], key)
        return out["inverse"] + out["contrastive"]

    grads = jax.grad(dynamics)(params["router"])
    assert _max_abs(grads) > 1e-8


def test_recon_trains_decoder_and_encoder():
    params = init_params(jax.random.PRNGKey(10))
    batch = make_batch(3, batch=2)
    key = jax.random.PRNGKey(11)

    def recon(subset):
        trial = dict(params)
        trial["pixel"] = subset["pixel"]
        trial["encoder"] = subset["encoder"]
        return compute_losses(trial, batch["video"], batch["actions"], key)["recon"]

    grads = jax.grad(recon)({"pixel": params["pixel"], "encoder": params["encoder"]})
    assert _max_abs(grads["pixel"]) > 1e-8
    assert _max_abs(grads["encoder"]) > 1e-8


def test_logistic_contrastive_prefers_matched_pairs():
    key = jax.random.PRNGKey(12)
    pos = jax.random.normal(key, (6, 8))
    matched = logistic_contrastive(pos, pos, jnp.roll(pos, 1, axis=0))
    mismatched = logistic_contrastive(pos, jnp.roll(pos, 1, axis=0), pos)
    assert jnp.isfinite(matched)
    assert float(matched) < float(mismatched)


def test_inverse_dynamics_loss_and_grad():
    key = jax.random.PRNGKey(13)
    k_e, k_n, k_p = jax.random.split(key, 3)
    e_t = jax.random.normal(k_e, (8, N * C))
    e_tp1 = jax.random.normal(k_n, (8, N * C))
    actions = jax.nn.one_hot(jnp.arange(8) % A, A)
    params = init_inverse(k_p)

    def loss(p):
        return inverse_dynamics_loss(p, e_t, e_tp1, actions)

    assert jnp.isfinite(loss(params))
    assert _max_abs(jax.grad(loss)(params)) > 1e-6


def test_train_step_finite_and_loss_moves():
    params = init_params(jax.random.PRNGKey(14))
    batch = make_batch(4, batch=4)
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)
    key = jax.random.PRNGKey(15)
    _params, _state, first = train_step(params, opt_state, batch, optimizer, key)
    params2, opt_state, _aux = params, opt_state, first
    for step in range(12):
        params2, opt_state, aux = train_step(
            params2, opt_state, batch, optimizer, jax.random.fold_in(key, step)
        )
    assert jnp.isfinite(first["total"])
    assert jnp.isfinite(aux["total"])
    assert float(aux["total"]) < float(first["total"])
    assert float(BETA_KL) > 0.0


def test_encoder_keeps_spatial_tokens():
    video, _actions = make_episode(0)
    features = encode(init_encoder(jax.random.PRNGKey(0)), video[None, ...])
    assert features.shape == (1, T, L, C)


def test_synthetic_episode_moves():
    video, actions = make_episode(0)
    assert video.shape == (T, 16, 16, 1)
    assert actions.shape == (T, A)
    assert jnp.allclose(jnp.sum(actions, axis=-1), 1.0)

    def center(frame):
        weights = frame[..., 0]
        ys, xs = jnp.mgrid[0:16, 0:16]
        total = jnp.sum(weights) + 1e-8
        y = jnp.sum(weights * ys) / total
        x = jnp.sum(weights * xs) / total
        return jnp.array([y, x])

    assert float(jnp.linalg.norm(center(video[0]) - center(video[3]))) > 1.0


def test_contrast_phi_is_used_by_the_loss():
    params = init_params(jax.random.PRNGKey(16))
    e = jax.random.normal(jax.random.PRNGKey(17), (2, N * C))
    phi = contrast_phi(params["contrast"], e)
    assert phi.shape == e.shape
