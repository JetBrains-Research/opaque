"""Adaptive clipping utilities for DP optimization.

This package provides infrastructure for adaptive gradient clipping as described in:
    Zuo et al., "DP-Adam-AC: Privacy-preserving Fine-Tuning of Localizable
    Language Models Using Adam Optimization with Adaptive Clipping"
    https://arxiv.org/abs/2510.05288

Key components:
    - ClipNormBuffer: Efficient tracking of gradient norms for percentile computation
    - clip_rate_based_lr_adjustment: Dynamic learning rate adjustment based on clipping frequency
    - compute_clip_rate_thresholds: Helper to compute acceptable clip rate range
"""

from opaque.adaptive.clip_buffer import ClipNormBuffer
from opaque.adaptive.lr_scheduler import (
    clip_rate_based_lr_adjustment,
    compute_clip_rate_thresholds,
)

__all__ = [
    "ClipNormBuffer",
    "clip_rate_based_lr_adjustment",
    "compute_clip_rate_thresholds",
]
