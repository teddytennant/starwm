"""Tiny synthetic video and a linear conv encoder.

The encoder is a strided convolution plus a dense map, with no nonlinearity.
It is trained by the reconstruction loss. It is not a Dreamer encoder.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from starwm.routing import A, D, T


def init_encoder(key, d: int = D):
    k_conv, k_w, k_b = jax.random.split(key, 3)
    # 16x16, kernel 4, stride 4 -> 4x4 with 8 channels = 128, then linear to D.
    return {
        "kernel": jax.random.normal(k_conv, (4, 4, 1, 8)) * 0.05,
        "W": jax.random.normal(k_w, (128, d)) * 0.05,
        "b": jax.random.normal(k_b, (d,)) * 0.05,
    }


def encode(params, video):
    """Map frames (B, T, 16, 16, 1) to latents (B, T, D).

    Linear encoder: strided conv and a dense layer, no activation.
    Not a Dreamer encoder.
    """
    b, t, h, w, c = video.shape
    x = video.reshape(b * t, h, w, c)
    y = jax.lax.conv_general_dilated(
        x,
        params["kernel"],
        window_strides=(4, 4),
        padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    flat = y.reshape(b * t, -1)
    z = flat @ params["W"] + params["b"]
    return z.reshape(b, t, -1)


def downsample_video(video):
    """Mean frame, then 2x2 average pool, to a provided recon target (B, 8, 8, 1).

    The target is a function of the video, not of the latent, so reconstruction
    cannot leak a shortcut through z.
    """
    mean = jnp.mean(video, axis=1)
    b = mean.shape[0]
    return mean.reshape(b, 8, 2, 8, 2, 1).mean(axis=(2, 4))


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

    # up, right, down, left. Step of 2 keeps the square on the 16x16 grid.
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
    return {
        "video": video,
        "actions": action,
        "target": downsample_video(video),
    }
