"""Cross-attention router and dual-stream heads.

Latent sequence z has shape (B, T, D). A learned query q in R^D scores each
timestep, and the routed latent is the attention-weighted value projection:

    scores_t = (W_q q) · (W_k z_t) / sqrt(D)
    alpha = softmax(scores) over t
    z_att = sum_t alpha_t * (W_v z_t)

The content stream decodes a patch from stop_gradient(z_att). The dynamics
stream keeps per-step value latents so inverse dynamics and next-latent
prediction can read z_att_t. This is not a policy, actor, or critic.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

D = 16
T = 8
A = 4


def init_router(key, d: int = D):
    kq, k_wq, k_wk, k_wv = jax.random.split(key, 4)
    scale = 0.2
    return {
        "q": jax.random.normal(kq, (d,)) * scale,
        "W_q": jax.random.normal(k_wq, (d, d)) * scale,
        "W_k": jax.random.normal(k_wk, (d, d)) * scale,
        "W_v": jax.random.normal(k_wv, (d, d)) * scale,
    }


def init_decoder(key, d: int = D, patch: int = 64):
    k_w, k_b = jax.random.split(key)
    return {
        "W": jax.random.normal(k_w, (d, patch)) * 0.1,
        "b": jax.random.normal(k_b, (patch,)) * 0.1,
    }


def init_predict(key, d: int = D, a: int = A):
    k_w, k_b = jax.random.split(key)
    return {
        "W": jax.random.normal(k_w, (d + a, d)) * 0.1,
        "b": jax.random.normal(k_b, (d,)) * 0.1,
    }


def init_inverse(key, d: int = D, a: int = A):
    k_w, k_b = jax.random.split(key)
    return {
        "W": jax.random.normal(k_w, (2 * d, a)) * 0.1,
        "b": jax.random.normal(k_b, (a,)) * 0.1,
    }


def route(params, z):
    """Cross-attend over time with a learned query.

    Args:
        params: q (D,), W_q, W_k, W_v each (D, D).
        z: latent sequence (B, T, D).

    Returns:
        z_att: (B, D) attention-pooled values.
        alpha: (B, T) softmax weights, summing to 1 over time.
    """
    z_att, alpha, _values = route_details(params, z)
    return z_att, alpha


def route_details(params, z):
    """Same as route, plus per-timestep value projections W_v z_t."""
    d = z.shape[-1]
    q_proj = params["W_q"] @ params["q"]
    keys = jnp.einsum("ij,btj->bti", params["W_k"], z)
    values = jnp.einsum("ij,btj->bti", params["W_v"], z)
    scale = jnp.sqrt(jnp.asarray(d, dtype=z.dtype))
    scores = jnp.einsum("i,bti->bt", q_proj, keys) / scale
    alpha = jax.nn.softmax(scores, axis=-1)
    z_att = jnp.einsum("bt,bti->bi", alpha, values)
    return z_att, alpha, values


def temporal_latents(z_att, values):
    """Per-step dynamics latents that still depend on pooled z_att.

    z_att_t = W_v z_t + z_att. Inverse dynamics and predict_next read these.
    Adding the pooled vector makes alpha, W_q, W_k, and q visible to the
    dynamics loss, not only W_v.
    """
    return values + z_att[:, None, :]


def decoder_input(z_att, z):
    """Input to the content decoder.

    The decoder reads stop_gradient(z_att), so reconstruction does not train
    the router. A straight-through residual from the mean latent lets the
    same reconstruction loss train the encoder. Forward value equals z_att
    plus a zero residual.
    """
    pooled = jnp.mean(z, axis=1)
    return jax.lax.stop_gradient(z_att) + pooled - jax.lax.stop_gradient(pooled)


def content_decode(params, z_att):
    """Map a routed latent (B, D) to a patch (B, 8, 8, 1)."""
    flat = z_att @ params["W"] + params["b"]
    return flat.reshape(z_att.shape[0], 8, 8, 1)


def predict_next(params, z_att_t, action_t):
    """Latent prediction head. Not a pixel decoder.

    z_att_t: (..., D), action_t: (..., A) -> z_{t+1} hat (..., D).
    """
    x = jnp.concatenate([z_att_t, action_t], axis=-1)
    return x @ params["W"] + params["b"]


def inverse_logits(params, z_t, z_tp1):
    """Predict action logits from a consecutive latent pair."""
    x = jnp.concatenate([z_t, z_tp1], axis=-1)
    return x @ params["W"] + params["b"]
