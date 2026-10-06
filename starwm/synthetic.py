"""Tiny synthetic video and a linear spatial encoder.

The encoder is a strided convolution. It keeps the 4x4 feature map as L spatial
tokens. It is not a Dreamer encoder.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from starwm.routing import A, C, H, L, T, W


def init_encoder(key, c: int = C):
    k_conv = jax.random.split(key, 1)[0]
    # 16x16, kernel 4, stride 4 -> 4x4 with C channels. No dense collapse.
    return {"kernel": jax.random.normal(k_conv, (4, 4, 1, c)) * 0.05}


def encode(params, video):
    """Map frames (B, T, 16, 16, 1) to spatial tokens Ft (B, T, L, C)."""
    b, t, h, w, ch = video.shape
    x = video.reshape(b * t, h, w, ch)
    y = jax.lax.conv_general_dilated(
        x,
        params["kernel"],
        window_strides=(4, 4),
        padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    if y.shape[1] != H or y.shape[2] != W or y.shape[3] != C:
        raise ValueError(f"encoder map {y.shape} is not {(H, W, C)}")
    return y.reshape(b, t, L, C)


def make_episode(seed: int, length: int = T, size: int = 16, square: int = 3):
    """Return a video (T, 16, 16, 1) and one-hot actions (T, A).

    A bright square moves by the discrete action. Actions are up, right, down,
    left. The first three transitions are forced right so the square at t=0
    and t=3 are in different positions. The last action is unused padding so
    the action tensor lines up with the T frames.
    """
    key = jax.random.PRNGKey(int(seed))
    moves = jax.random.randint(key, (length - 1,), 0, A)
    moves = moves.at[:3].set(1)
    labels = jnp.concatenate([moves, jnp.zeros((1,), dtype=jnp.int32)])
    actions = jax.nn.one_hot(labels, A)

    dirs = ((-2, 0), (0, 2), (2, 0), (0, -2))
    frames = jnp.zeros((length, size, size, 1), dtype=jnp.float32)
    y, x = 2, 2
    for t in range(length):
        frames = frames.at[t, y : y + square, x : x + square, 0].set(1.0)
        if t < length - 1:
            dy, dx = dirs[int(moves[t])]
            y = min(max(y + dy, 0), size - square)
            x = min(max(x + dx, 0), size - square)
    return frames, actions


def make_batch(seed: int, batch: int = 4, length: int = T):
    videos = []
    actions = []
    for i in range(batch):
        video, action = make_episode(seed + i, length=length)
        videos.append(video)
        actions.append(action)
    video = jnp.stack(videos, axis=0)
    action = jnp.stack(actions, axis=0)
    return {"video": video, "actions": action}
