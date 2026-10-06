"""CPU tests for routing math, stop-gradient isolation, and synthetic video."""

import jax
import jax.numpy as jnp
import optax

from starwm.losses import (
    info_nce,
    init_params,
    inverse_dynamics_loss,
    losses_from_latents,
    reconstruction_loss,
    train_step,
)
from starwm.routing import A, D, T, init_inverse, route
from starwm.synthetic import make_batch, make_episode


def _max_abs(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return max(float(jnp.max(jnp.abs(leaf))) for leaf in leaves)


def _all_zero(tree):
    return all(bool(jnp.all(leaf == 0)) for leaf in jax.tree_util.tree_leaves(tree))


def _latent_batch(seed=0, batch=4):
    key = jax.random.PRNGKey(seed)
    k_z, k_a, k_y = jax.random.split(key, 3)
    z = jax.random.normal(k_z, (batch, T, D))
    labels = jax.random.randint(k_a, (batch, T), 0, A)
    actions = jax.nn.one_hot(labels, A)
    target = jax.random.normal(k_y, (batch, 8, 8, 1))
    return z, actions, target


def test_alpha_sums_to_one_and_concentrates():
    d, t, b = D, T, 2
    t_star = 3
    params = {
        "q": jnp.zeros((d,)).at[0].set(1.0),
        "W_q": jnp.eye(d),
        "W_k": jnp.eye(d),
        "W_v": jnp.eye(d),
    }
    z = jnp.zeros((b, t, d)).at[:, t_star, 0].set(20.0)
    _z_att, alpha = route(params, z)
    assert alpha.shape == (b, t)
    assert jnp.allclose(jnp.sum(alpha, axis=-1), 1.0, atol=1e-5)
    assert float(jnp.min(alpha[:, t_star])) > 0.5
    assert jnp.all(jnp.argmax(alpha, axis=-1) == t_star)


def test_recon_grad_router_is_zero():
    params = init_params(jax.random.PRNGKey(0))
    z, actions, target = _latent_batch(1)

    def recon(router_params):
        trial = dict(params)
        trial["router"] = router_params
        return losses_from_latents(trial, z, actions, target)["recon"]

    grads = jax.grad(recon)(params["router"])
    assert _all_zero(grads)


def test_dynamics_grad_decoder_is_zero():
    params = init_params(jax.random.PRNGKey(1))
    z, actions, target = _latent_batch(2)

    def dynamics(decoder_params):
        trial = dict(params)
        trial["decoder"] = decoder_params
        out = losses_from_latents(trial, z, actions, target)
        return out["inverse"] + out["contrastive"]

    grads = jax.grad(dynamics)(params["decoder"])
    assert _all_zero(grads)


def test_dynamics_grad_router_nonzero():
    params = init_params(jax.random.PRNGKey(2))
    z, actions, target = _latent_batch(3)

    def dynamics(router_params):
        trial = dict(params)
        trial["router"] = router_params
        out = losses_from_latents(trial, z, actions, target)
        return out["inverse"] + out["contrastive"]

    grads = jax.grad(dynamics)(params["router"])
    assert _max_abs(grads) > 1e-8


def test_recon_grad_decoder_nonzero():
    params = init_params(jax.random.PRNGKey(3))
    z, actions, target = _latent_batch(4)

    def recon(decoder_params):
        trial = dict(params)
        trial["decoder"] = decoder_params
        return losses_from_latents(trial, z, actions, target)["recon"]

    grads = jax.grad(recon)(params["decoder"])
    assert _max_abs(grads) > 1e-8


def test_infonce_lower_for_matched_pairs():
    key = jax.random.PRNGKey(5)
    target = jax.random.normal(key, (8, D)) + jnp.eye(8, D) * 5.0
    matched = info_nce(target, target)
    mismatched = info_nce(target, jnp.roll(target, 1, axis=0))
    assert jnp.isfinite(matched)
    assert jnp.isfinite(mismatched)
    assert float(matched) < float(mismatched)


def test_inverse_dynamics_loss_finite():
    key = jax.random.PRNGKey(6)
    k_z, k_n, k_p = jax.random.split(key, 3)
    z_t = jax.random.normal(k_z, (8, D))
    z_tp1 = jax.random.normal(k_n, (8, D))
    actions = jax.nn.one_hot(jnp.arange(8) % A, A)
    params = init_inverse(k_p)
    loss = inverse_dynamics_loss(params, z_t, z_tp1, actions)
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_inverse_dynamics_head_grad_nonzero():
    # Fixed batch closed over by the loss (a batch the head can memorize).
    key = jax.random.PRNGKey(7)
    k_z, k_n, k_p = jax.random.split(key, 3)
    z_t = jax.random.normal(k_z, (8, D))
    z_tp1 = jax.random.normal(k_n, (8, D))
    actions = jax.nn.one_hot(jnp.arange(8) % A, A)
    params = init_inverse(k_p)

    def loss(p):
        return inverse_dynamics_loss(p, z_t, z_tp1, actions)

    grads = jax.grad(loss)(params)
    assert jnp.isfinite(loss(params))
    assert _max_abs(grads) > 1e-6


def test_train_step_finite():
    params = init_params(jax.random.PRNGKey(8))
    batch = make_batch(0, batch=4)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    _params, _state, aux = train_step(params, opt_state, batch, optimizer)
    assert jnp.isfinite(aux["total"])
    assert jnp.isfinite(aux["recon"])
    assert jnp.isfinite(aux["inverse"])
    assert jnp.isfinite(aux["contrastive"])


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
