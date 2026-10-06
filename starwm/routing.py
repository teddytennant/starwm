"""Spatial cross-attention router from StarWM section 3.

Features Ft have shape (..., L, C) with L = H*W. Keys add a fixed 2D
sinusoidal positional encoding, values append normalized coordinates:

    Kt = Ft + Epos
    Vt = [Ft ; Ecoord]

N learnable queries attend over space (eq. 5):

    At = softmax( (Q WQ) (Kt WK)^T / sqrt(C) )   over the L tokens

Coordinate-free pooled content is flatten(At Ft). The dynamics stream and
the dual-stream decoder read stop_gradient(At), so reconstruction does not
train Q, WQ, or WK. Inverse dynamics and the contrastive loss read At Ft,
so those losses do train the router.

Not multi-head. Not an RSSM. Not a policy.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

H = 4
W = 4
L = H * W
C = 8
N = 2
T = 8
A = 4
# Alias kept so older imports that meant "latent width" still resolve.
D = C


def positional_encoding(h: int = H, w: int = W, c: int = C):
    """2D sinusoidal encoding, shape (L, C). Even channels are y, odd are x."""
    if c % 2 != 0:
        raise ValueError("C must be even so y and x each get c/2 channels")
    half = c // 2
    ys = jnp.arange(h, dtype=jnp.float32)
    xs = jnp.arange(w, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    yy = yy.reshape(-1)
    xx = xx.reshape(-1)
    freq_dim = half // 2
    denom = jnp.power(10000.0, jnp.arange(freq_dim, dtype=jnp.float32) / freq_dim)
    y_ang = yy[:, None] / denom[None, :]
    x_ang = xx[:, None] / denom[None, :]
    y_pe = jnp.concatenate([jnp.sin(y_ang), jnp.cos(y_ang)], axis=-1)
    x_pe = jnp.concatenate([jnp.sin(x_ang), jnp.cos(x_ang)], axis=-1)
    pe = jnp.stack([y_pe, x_pe], axis=-1).reshape(h * w, c)
    return pe


def spatial_coords(h: int = H, w: int = W):
    """Normalized (y, x) in [0, 1], shape (L, 2)."""
    ys = jnp.linspace(0.0, 1.0, h, dtype=jnp.float32)
    xs = jnp.linspace(0.0, 1.0, w, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    return jnp.stack([yy, xx], axis=-1).reshape(h * w, 2)


def init_router(key, c: int = C, n: int = N):
    kq, kwq, kwk = jax.random.split(key, 3)
    scale = 0.2
    return {
        "Q": jax.random.normal(kq, (n, c)) * scale,
        "WQ": jax.random.normal(kwq, (c, c)) * scale,
        "WK": jax.random.normal(kwk, (c, c)) * scale,
    }


def keys_and_values(features):
    """Kt = Ft + Epos, Vt = [Ft; Ecoord]. features is (..., L, C)."""
    epos = positional_encoding(H, W, features.shape[-1])
    ecoord = spatial_coords(H, W)
    keys = features + epos
    values = jnp.concatenate([features, jnp.broadcast_to(ecoord, features.shape[:-1] + (2,))], axis=-1)
    return keys, values


def attention(params, features):
    """Eq. 5. Returns At with shape (..., N, L), softmax over space."""
    keys, _values = keys_and_values(features)
    q_proj = jnp.einsum("nc,cd->nd", params["Q"], params["WQ"])
    k_proj = jnp.einsum("...lc,cd->...ld", keys, params["WK"])
    scale = jnp.sqrt(jnp.asarray(features.shape[-1], dtype=features.dtype))
    logits = jnp.einsum("nd,...ld->...nl", q_proj, k_proj) / scale
    return jax.nn.softmax(logits, axis=-1)


def pooled_content(attn, features):
    """flatten(At Ft). Coordinates are not in this vector. Shape (..., N*C)."""
    pooled = jnp.einsum("...nl,...lc->...nc", attn, features)
    n = pooled.shape[-2]
    c = pooled.shape[-1]
    return pooled.reshape(pooled.shape[:-2] + (n * c,))


def dynamics_tokens(attn_bar, values):
    """edyn = A_bar Vt. Shape (..., N, C+2)."""
    return jnp.einsum("...nl,...ld->...nd", attn_bar, values)


def coverage_mask(attn_bar):
    """Mt = clamp(sum over queries of A_bar, 0, 1). Shape (..., L)."""
    return jnp.clip(jnp.sum(attn_bar, axis=-2), 0.0, 1.0)


def foreground(attn_bar, entity):
    """F_fg = A_bar^T e_hat. attn (..., N, L), entity (..., N, C) -> (..., L, C)."""
    return jnp.einsum("...nl,...nc->...lc", attn_bar, entity)


def compose(fg, bg, mask):
    """Eq. 8. F_hat = F_fg * M + F_bg * (1 - M)."""
    m = mask[..., None]
    return fg * m + bg * (1.0 - m)


def route(params, features):
    """Attention and coordinate-free pooled content. No stop-gradient here.

    features: (..., L, C)
    returns attn (..., N, L), eattn (..., N*C)
    """
    attn = attention(params, features)
    return attn, pooled_content(attn, features)
