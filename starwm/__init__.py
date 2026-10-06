"""Spatial cross-attention routing with stop-gradient barriers."""

from starwm.losses import compute_losses, logistic_contrastive, train_step
from starwm.routing import attention, compose, route
from starwm.synthetic import make_episode

__all__ = [
    "attention",
    "compose",
    "compute_losses",
    "logistic_contrastive",
    "make_episode",
    "route",
    "train_step",
]
