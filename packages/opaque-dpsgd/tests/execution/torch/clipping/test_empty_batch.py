"""Torch-only tests for the private ``_compute_clipping_stats`` helper.

Public empty-batch behavior for ``clipped_grad`` / ``adaptive_clipped_grad``
(zero grads, empty aux tensors, preserved adaptive clipping_norm, paired
second-moment output, and single-process sync passthrough) is covered by the
provider-neutral matrix suites under ``execution/portable``.
"""

import torch

from opaque.api.dpsgd.clipping._adaptive import _compute_clipping_stats


class TestComputeClippingStats:
    def test_empty_tensor_returns_zeros(self):
        num_clipped, total, clipping_rate = _compute_clipping_stats(
            torch.empty(0), clipping_norm=1.0
        )
        assert num_clipped == 0.0
        assert total == 0.0
        assert clipping_rate == 0.0

    def test_nonempty_tensor_reports_honest_total(self):
        norms = torch.tensor([0.5, 1.5, 2.5])
        num_clipped, total, rate = _compute_clipping_stats(norms, clipping_norm=1.0)
        assert total == 3.0
        assert num_clipped == 2.0
        assert abs(rate - 2.0 / 3.0) < 1e-6

    def test_all_below_clip_norm(self):
        norms = torch.tensor([0.1, 0.2, 0.3])
        num_clipped, _total, rate = _compute_clipping_stats(norms, clipping_norm=1.0)
        assert num_clipped == 0.0
        assert rate == 0.0
