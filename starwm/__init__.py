"""Attention routing with stop-gradient barriers for a self-supervised world-model latent."""

from starwm.losses import (
    compute_losses,
    info_nce,
    inverse_dynamics_loss,
    reconstruction_loss,
    train_step,
)
from starwm.routing import content_decode, predict_next, route
from starwm.synthetic import make_episode

__all__ = [
    "compute_losses",
    "content_decode",
    "info_nce",
    "inverse_dynamics_loss",
    "make_episode",
    "predict_next",
    "reconstruction_loss",
    "route",
    "train_step",
]
