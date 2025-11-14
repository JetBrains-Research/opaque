"""Adaptive clipping utilities for DP optimization.

This package provides infrastructure for adaptive gradient clipping as described in:
    Zuo et al., "DP-Adam-AC: Privacy-preserving Fine-Tuning of Localizable
    Language Models Using Adam Optimization with Adaptive Clipping"
    https://arxiv.org/abs/2510.05288

Key components:
    - clip_buffer: Functional API for tracking gradient norms (create, update, get_adaptive_clip_norm, etc.)
    - clip_rate_based_lr_adjustment: Dynamic learning rate adjustment based on clipping frequency
    - compute_clip_rate_thresholds: Helper to compute acceptable clip rate range
"""

from opaque.optimizers.adaptive import clip_buffer
from opaque.optimizers.adaptive.lr_scheduler import (
    clip_rate_based_lr_adjustment,
    compute_clip_rate_thresholds,
)

__all__ = [
    "clip_buffer",
    "clip_rate_based_lr_adjustment",
    "compute_clip_rate_thresholds",
]
