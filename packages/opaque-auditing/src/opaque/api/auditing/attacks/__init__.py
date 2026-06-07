"""Membership inference attack scoring functions."""

from opaque.api.auditing.attacks._gradient import gradient_scores
from opaque.api.auditing.attacks._loss import loss_scores

__all__ = ["gradient_scores", "loss_scores"]
