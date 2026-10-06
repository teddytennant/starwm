"""Self-supervised losses with the paper's stop-gradient split.

Inverse dynamics and the contrastive loss read flatten(At Ft), so they train
Q, WQ, and WK. Reconstruction composes the two streams with A_bar = sg(At),
so it does not train those parameters. The pixel decoder and the residual
bottleneck are not inputs to the dynamics losses.

Eq. 7 is the logistic form in the paper, not a full-batch InfoNCE:

    -log sigmoid( phi(e_t) · phi(e_{t+1}) / tau )
    -log sigmoid( -phi(e_t) · phi(e_neg) / tau )
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from starwm.routing import (
    A,
    C,
    H,
    L,
    N,
    W,
    attention,
    compose,
    coverage_mask,
    dynamics_tokens,
    foreground,
    init_router,
    keys_and_values,
    pooled_content,
)
from starwm.synthetic import encode, init_encoder

TEMPERATURE = 0.1
BETA_KL = 1e-3
BOTTLENECK = 4


def init_fg(key, c: int = C):
    k_w, k_b = jax.random.split(key)
    return {
        "W": jax.random.normal(k_w, (c + 2, c)) * 0.1,
        "b": jax.random.normal(k_b, (c,)) * 0.1,
    }


def init_bg(key, c: int = C, width: int = BOTTLENECK):
    keys = jax.random.split(key, 6)
    return {
        "W_mu": jax.random.normal(keys[0], (c, width)) * 0.1,
        "b_mu": jax.random.normal(keys[1], (width,)) * 0.1,
        "W_lv": jax.random.normal(keys[2], (c, width)) * 0.1,
        "b_lv": jax.random.normal(keys[3], (width,)) * 0.1,
        "W_out": jax.random.normal(keys[4], (width, c)) * 0.1,
        "b_out": jax.random.normal(keys[5], (c,)) * 0.1,
    }


def init_pixel(key, c: int = C):
    """Non-overlapping stride-4 patch decoder: each token maps to a 4x4 patch."""
    k_w, k_b = jax.random.split(key)
    return {
        "W": jax.random.normal(k_w, (c, 16)) * 0.1,
        "b": jax.random.normal(k_b, (16,)) * 0.1,
    }


def init_inverse(key, c: int = C, n: int = N, a: int = A):
    k_w, k_b = jax.random.split(key)
    width = n * c
    return {
        "W": jax.random.normal(k_w, (2 * width, a)) * 0.1,
        "b": jax.random.normal(k_b, (a,)) * 0.1,
    }


def init_contrast(key, c: int = C, n: int = N):
    k_w, k_b = jax.random.split(key)
    width = n * c
    return {
        "W": jax.random.normal(k_w, (width, width)) * 0.1,
        "b": jax.random.normal(k_b, (width,)) * 0.1,
    }


def init_params(key):
    keys = jax.random.split(key, 6)
    return {
        "encoder": init_encoder(keys[0]),
        "router": init_router(keys[1]),
        "fg": init_fg(keys[2]),
        "bg": init_bg(keys[3]),
        "pixel": init_pixel(keys[4]),
        "inverse": init_inverse(keys[5]),
        "contrast": init_contrast(keys[0]),
    }


def project_entity(params, edyn):
    """phi_fg: dynamics tokens (..., N, C+2) -> entity features (..., N, C)."""
    return jnp.einsum("...nd,dc->...nc", edyn, params["W"]) + params["b"]


def residual_stream(params, features, key):
    """Gaussian bottleneck on Ft. Returns F_bg (..., L, C) and a mean KL."""
    mu = jnp.einsum("...lc,cw->...lw", features, params["W_mu"]) + params["b_mu"]
    logvar = jnp.einsum("...lc,cw->...lw", features, params["W_lv"]) + params["b_lv"]
    logvar = jnp.clip(logvar, -8.0, 8.0)
    eps = jax.random.normal(key, mu.shape)
    z = mu + eps * jnp.exp(0.5 * logvar)
    bg = jnp.einsum("...lw,wc->...lc", z, params["W_out"]) + params["b_out"]
    kl = -0.5 * jnp.mean(1.0 + logvar - jnp.square(mu) - jnp.exp(logvar))
    return bg, kl


def pixel_decode(params, features):
    """Stitch non-overlapping 4x4 patches into (..., 16, 16, 1)."""
    patches = jnp.einsum("...lc,cp->...lp", features, params["W"]) + params["b"]
    lead = patches.shape[:-2]
    patches = patches.reshape(lead + (H, W, 4, 4))
    image = jnp.transpose(patches, (*range(len(lead)), len(lead), len(lead) + 2, len(lead) + 1, len(lead) + 3))
    image = image.reshape(lead + (H * 4, W * 4, 1))
    return image


def reconstruction_loss(pred, target):
    return jnp.mean(jnp.square(pred - target))


def cross_entropy(logits, target_onehot):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(target_onehot * log_probs, axis=-1))


def inverse_logits(params, e_t, e_tp1):
    x = jnp.concatenate([e_t, e_tp1], axis=-1)
    return jnp.einsum("...d,da->...a", x, params["W"]) + params["b"]


def inverse_dynamics_loss(params, e_t, e_tp1, actions):
    """Cross-entropy of phi_inv(eattn_t, eattn_{t+1}) against action_t."""
    return cross_entropy(inverse_logits(params, e_t, e_tp1), actions)


def contrast_phi(params, eattn):
    return jnp.einsum("...d,dh->...h", eattn, params["W"]) + params["b"]


def logistic_contrastive(phi_t, phi_tp1, phi_neg, temperature: float = TEMPERATURE):
    """Eq. 7. phi_* are (..., D)."""
    pos = jnp.sum(phi_t * phi_tp1, axis=-1) / temperature
    neg = jnp.sum(phi_t * phi_neg, axis=-1) / temperature
    return jnp.mean(-jax.nn.log_sigmoid(pos) - jax.nn.log_sigmoid(-neg))


def compose_features(router, fg_params, features):
    """Stopped attention, foreground broadcast, coverage mask. No residual yet."""
    attn = attention(router, features)
    attn_bar = jax.lax.stop_gradient(attn)
    _keys, values = keys_and_values(features)
    edyn = dynamics_tokens(attn_bar, values)
    entity = project_entity(fg_params, edyn)
    fg = foreground(attn_bar, entity)
    mask = coverage_mask(attn_bar)
    return attn, fg, mask


def compute_losses(params, video, actions, key):
    """Encode frames, route, and apply the barred losses.

    video: (B, T, 16, 16, 1). actions: (B, T, A).
    """
    features = encode(params["encoder"], video)
    attn = attention(params["router"], features)
    eattn = pooled_content(attn, features)
    attn_bar = jax.lax.stop_gradient(attn)
    _keys, values = keys_and_values(features)
    edyn = dynamics_tokens(attn_bar, values)
    entity = project_entity(params["fg"], edyn)
    fg = foreground(attn_bar, entity)
    mask = coverage_mask(attn_bar)
    bg, kl = residual_stream(params["bg"], features, key)
    composed = compose(fg, bg, mask)
    decoded = pixel_decode(params["pixel"], composed)
    recon = reconstruction_loss(decoded, video)

    e_t = eattn[:, :-1]
    e_tp1 = eattn[:, 1:]
    action_t = actions[:, :-1]
    inverse = inverse_dynamics_loss(params["inverse"], e_t, e_tp1, action_t)

    phi = contrast_phi(params["contrast"], eattn)
    phi_t = phi[:, :-1]
    phi_tp1 = phi[:, 1:]
    # A negative from another batch row. Roll is a no-op only when B == 1.
    phi_neg = jnp.roll(phi_tp1, 1, axis=0)
    contrastive = logistic_contrastive(phi_t, phi_tp1, phi_neg)

    total = recon + inverse + contrastive + BETA_KL * kl
    return {
        "total": total,
        "recon": recon,
        "inverse": inverse,
        "contrastive": contrastive,
        "kl": kl,
        "attn": attn,
        "mask": mask,
        "composed": composed,
    }


def train_step(params, opt_state, batch, optimizer, key):
    """One gradient step. key draws the residual bottleneck noise."""

    def loss_fn(p):
        out = compute_losses(p, batch["video"], batch["actions"], key)
        return out["total"], out

    (_total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, aux
