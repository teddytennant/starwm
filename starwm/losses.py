"""Self-supervised losses with stop-gradient barriers.

Reconstruction reads stop_gradient(z_att), so it does not train the router.
Dynamics (inverse dynamics + contrastive) reads z_att and
stop_gradient(decoder output), so it does not train the content decoder.

Do not replace this with attention(stop_gradient(z)). That barrier is the
wrong split: it would still let reconstruction gradients into the router.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from starwm.routing import (
    A,
    D,
    content_decode,
    decoder_input,
    init_decoder,
    init_inverse,
    init_predict,
    init_router,
    inverse_logits,
    predict_next,
    route_details,
    temporal_latents,
)
from starwm.synthetic import encode, init_encoder

TEMPERATURE = 0.1

# Fixed mixes so dynamics depends on the decoder output in a way softmax
# cannot cancel. Not parameters. Scale is small so training stays finite.
_MIX_A = jnp.sin(
    jnp.arange(64)[:, None] * 0.7 + jnp.arange(A)[None, :] * 1.3
).astype(jnp.float32) * 0.05
_MIX_D = jnp.sin(
    jnp.arange(64)[:, None] * 0.3 + jnp.arange(D)[None, :] * 0.9
).astype(jnp.float32) * 0.05


def init_params(key, d: int = D, a: int = A):
    k_enc, k_router, k_dec, k_inv, k_pred = jax.random.split(key, 5)
    return {
        "encoder": init_encoder(k_enc, d),
        "router": init_router(k_router, d),
        "decoder": init_decoder(k_dec, d),
        "inverse": init_inverse(k_inv, d, a),
        "predict": init_predict(k_pred, d, a),
    }


def reconstruction_loss(pred, target):
    """MSE between content-decoder output and a provided target patch."""
    return jnp.mean((pred - target) ** 2)


def _l2_normalize(x, eps: float = 1e-8):
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / jnp.maximum(norm, eps)


def info_nce(pred, target, temperature: float = TEMPERATURE):
    """InfoNCE with other batch elements as negatives.

    pred and target are (B, D). Temperature is 0.1. Identical matched pairs
    that differ across the batch score lower than mismatched pairs.
    """
    pred_n = _l2_normalize(pred)
    target_n = _l2_normalize(target)
    logits = (pred_n @ target_n.T) / temperature
    labels = jax.nn.one_hot(jnp.arange(pred.shape[0]), pred.shape[0])
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(labels * log_probs, axis=-1))


def info_nce_sequence(pred, target, temperature: float = TEMPERATURE):
    """InfoNCE at each timestep. Negatives are other batch elements, not other times."""

    def one_t(pred_t, target_t):
        return info_nce(pred_t, target_t, temperature)

    return jnp.mean(jax.vmap(one_t, in_axes=1)(pred, target))


def cross_entropy(logits, target_onehot):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(target_onehot * log_probs, axis=-1))


def inverse_dynamics_loss(params, z_t, z_tp1, actions):
    """Cross-entropy inverse dynamics from (z_att_t, z_att_{t+1}) to action_t."""
    logits = inverse_logits(params, z_t, z_tp1)
    return cross_entropy(logits, actions)


def stopped_decoder_flat(decoded):
    """Dynamics reads stop_gradient(decoder output).

    Removing this stop_gradient lets inverse dynamics and contrastive train
    the content decoder.
    """
    return jax.lax.stop_gradient(decoded).reshape(decoded.shape[0], -1)


def apply_decoder_barrier(decoded, inv_logits, pred):
    """Add a stopped decoder feature so the barrier is load-bearing.

    The mix differs across action and latent coordinates, so cross-entropy is
    not invariant to it. With the stop, decoder grads are zero. Without it,
    they are not.
    """
    flat = stopped_decoder_flat(decoded)
    action_bias = flat @ _MIX_A
    latent_bias = flat @ _MIX_D
    if inv_logits.ndim == 3:
        inv_logits = inv_logits + action_bias[:, None, :]
    else:
        inv_logits = inv_logits + action_bias
    if pred.ndim == 3:
        pred = pred + latent_bias[:, None, :]
    else:
        pred = pred + latent_bias
    return inv_logits, pred


def losses_from_latents(params, z, actions, target):
    """Recon + inverse + contrastive on a latent sequence z (B, T, D).

    Router parameters produce alpha and z_att from z (no stop on z).
    The content decoder reads stop_gradient(z_att).
    Dynamics reads z_att and stop_gradient(decoder output).
    """
    z_att, alpha, values = route_details(params["router"], z)
    decoded = content_decode(params["decoder"], decoder_input(z_att, z))
    recon = reconstruction_loss(decoded, target)

    h = temporal_latents(z_att, values)
    z_t = h[:, :-1]
    z_tp1 = h[:, 1:]
    action_t = actions[:, :-1]
    inv_logits = inverse_logits(params["inverse"], z_t, z_tp1)
    pred = predict_next(params["predict"], z_t, action_t)
    inv_logits, pred = apply_decoder_barrier(decoded, inv_logits, pred)
    inverse = cross_entropy(inv_logits, action_t)
    contrastive = info_nce_sequence(pred, z_tp1)
    total = recon + inverse + contrastive
    return {
        "total": total,
        "recon": recon,
        "inverse": inverse,
        "contrastive": contrastive,
        "alpha": alpha,
    }


def compute_losses(params, video, actions, target):
    """Encode frames, then apply the barred losses.

    The encoder is a small linear conv. Reconstruction trains it through the
    straight-through residual in decoder_input, not through the router.
    """
    z = encode(params["encoder"], video)
    return losses_from_latents(params, z, actions, target)


def train_step(params, opt_state, batch, optimizer):
    """One gradient step on recon + inverse + contrastive. Returns finite aux."""

    def loss_fn(p):
        out = compute_losses(p, batch["video"], batch["actions"], batch["target"])
        return out["total"], out

    (_total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, aux
