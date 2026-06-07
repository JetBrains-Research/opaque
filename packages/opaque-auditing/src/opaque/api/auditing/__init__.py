"""Empirical privacy auditing impl — canary-based with pluggable attacks."""

from opaque.api.auditing._coin_flip import coin_flip
from opaque.api.auditing.attacks import gradient_scores, loss_scores
from opaque.api.auditing.one_run import one_run

__all__ = ["coin_flip", "gradient_scores", "loss_scores", "one_run"]
