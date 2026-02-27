"""Fused Linear + Cross-Entropy Loss with vmap support.

Computes CE(hidden_states @ weight.T, labels) without materializing the full
logit matrix. Uses Apple's cut_cross_entropy Triton kernels (ICLR 2025) for
tiled matmul + streaming LSE in SRAM.

Mathematical decomposition:
    CE(e, c, t) = -e·c[t] + log(Σ_v exp(e·c[v]))
                = neg_dot(e, c[t]) + LSE(e @ c.T)

Memory savings: O(B) + O(V) intermediate storage instead of O(B * V).
For LLaMA-3 (128K vocab), this avoids materializing ~1 GB of logits per sample.
"""

from __future__ import annotations

import torch

from cut_cross_entropy.cce_lse_forward import cce_lse_forward_kernel
from cut_cross_entropy.cce_backward import cce_backward_kernel
from cut_cross_entropy.indexed_dot import indexed_neg_dot_forward_kernel
from cut_cross_entropy.utils import _build_flat_valids, _handle_eps


# =============================================================================
# Shared forward helper
# =============================================================================

def _linear_ce_forward(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    softcap: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute per-token NLL via fused linear CE (no reduction).

    Args:
        hidden_states: (..., D) embeddings
        weight: (V, D) classifier weights
        labels: (...,) target indices
        ignore_index: label value to ignore
        softcap: Gemma2 softcap value (None = disabled)

    Returns:
        (nll, valids) — per-valid-token NLL values, valids index tensor
    """
    e = hidden_states.contiguous().flatten(0, -2)  # (N, D)
    targets = labels.contiguous().flatten()          # (N,)

    valids = _build_flat_valids(targets, ignore_index, shift=True)

    # Streaming LSE: log(Σ exp(e @ c[v].T)) — tiled in SRAM, never writes full logits
    lse = cce_lse_forward_kernel(e, weight, valids, softcap=softcap)

    # Neg dot: -e·c[target] — per-token dot product with correct class only
    neg_dot = indexed_neg_dot_forward_kernel(
        e, weight, targets, shift=True,
        valids=valids, softcap=softcap, out_dtype=lse.dtype,
    )

    # NLL = neg_dot + lse = -e·c[t] + log(Σ exp(e·c[v]))
    nll = neg_dot.add_(lse)

    return nll, valids


# =============================================================================
# Backward wrapper (autograd.Function for vmap dispatch)
# =============================================================================

class _LinearCEBackward(torch.autograd.Function):
    """Backward pass wrapped as autograd.Function for vmap(grad()) support.

    Recomputes valids and LSE from raw inputs (activation checkpointing style).
    This avoids saving large intermediate tensors and handles vmap correctly.
    """

    @staticmethod
    def forward(
        grad_out, hidden_states, weight, labels,
        softcap, filter_eps, ignore_index,
    ):
        e = hidden_states.contiguous().flatten(0, -2)
        targets = labels.contiguous().flatten()
        valids = _build_flat_valids(targets, ignore_index, shift=True)
        lse = cce_lse_forward_kernel(e, weight, valids, softcap=softcap)

        resolved_eps = _handle_eps(filter_eps, e.dtype)

        # Ensure CCE backward computes weight gradient even when weight.requires_grad
        # is False — functorch's grad(f, argnums=1) may need it via vmap dispatch.
        weight_bwd = weight if weight.requires_grad else weight.detach().requires_grad_(True)

        de, dc = cce_backward_kernel(
            grad_out, e, weight_bwd, lse, valids,
            softcap, resolved_eps,
            targets=targets, shift=True, grad_scale=1.0,
        )

        if dc is None:
            dc = torch.zeros_like(weight)

        return de, dc

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Double backward not supported for LinearCrossEntropyLoss")

    @staticmethod
    def vmap(info, in_dims, grad_out, hidden_states, weight, labels,
             softcap, filter_eps, ignore_index):
        (grad_bdim, h_bdim, w_bdim, lab_bdim,
         sc_bdim, fe_bdim, ii_bdim) = in_dims

        assert w_bdim is None, "weight should not be batched"
        assert sc_bdim is None, "softcap should not be batched"
        assert fe_bdim is None, "filter_eps should not be batched"
        assert ii_bdim is None, "ignore_index should not be batched"

        B_vmap = hidden_states.shape[0]
        de_list = []
        dc_list = []

        # Ensure CCE backward computes weight gradient for vmap(grad(f, argnums=1))
        weight_bwd = weight if weight.requires_grad else weight.detach().requires_grad_(True)

        for i in range(B_vmap):
            h_i = hidden_states[i].contiguous().flatten(0, -2)
            t_i = labels[i].contiguous().flatten()
            g_i = grad_out[i] if grad_bdim is not None else grad_out

            valids_i = _build_flat_valids(t_i, ignore_index, shift=True)
            lse_i = cce_lse_forward_kernel(h_i, weight_bwd, valids_i, softcap=softcap)

            resolved_eps = _handle_eps(filter_eps, h_i.dtype)

            de_i, dc_i = cce_backward_kernel(
                g_i, h_i, weight_bwd, lse_i, valids_i,
                softcap, resolved_eps,
                targets=t_i, shift=True, grad_scale=1.0,
            )
            de_list.append(de_i)
            dc_list.append(dc_i if dc_i is not None else torch.zeros_like(weight))

        de = torch.stack(de_list)
        dc = torch.stack(dc_list)
        return (de, dc), (0, 0)


# =============================================================================
# Main autograd.Function
# =============================================================================

class Opaque_LinearCrossEntropyLoss(torch.autograd.Function):
    """Fused linear projection + cross-entropy loss with vmap support.

    Computes the NLL sum:
        nll_sum = Σ_valid_tokens CE(hidden_states @ weight.T, labels)

    Without materializing the full (batch*seq, vocab) logit matrix.
    Uses cut_cross_entropy Triton kernels for tiled computation in SRAM.

    Returns unreduced nll_sum — caller handles reduction (mean, num_items_in_batch).
    """

    @staticmethod
    def forward(
        hidden_states, weight, labels,
        ignore_index=-100, logit_softcapping=0, logit_scaling=0,
    ):
        """Forward pass.

        Args:
            hidden_states: (..., hidden_dim) embeddings from backbone
            weight: (vocab_size, hidden_dim) lm_head weight
            labels: (...,) target token IDs (-100 = ignore)
            ignore_index: label value to ignore (default -100)
            logit_softcapping: Gemma2 softcap value (0 = disabled)
            logit_scaling: Granite scaling divisor (0 = disabled)

        Returns:
            nll_sum: scalar tensor — sum of per-valid-token NLL (unreduced)
        """
        # Handle Granite logit scaling via weight pre-division:
        # (h @ w.T) / s = h @ (w/s).T  — mathematically equivalent
        if logit_scaling != 0:
            weight = weight / logit_scaling

        softcap = logit_softcapping if logit_softcapping != 0 else None

        nll, valids = _linear_ce_forward(
            hidden_states, weight, labels, ignore_index, softcap,
        )

        return nll.sum()

    @staticmethod
    def setup_context(ctx, inputs, output):
        (hidden_states, weight, labels,
         ignore_index, logit_softcapping, logit_scaling) = inputs

        # Handle scaling for backward (same pre-division as forward)
        if logit_scaling != 0:
            weight = weight / logit_scaling

        # Save raw tensors — _LinearCEBackward recomputes valids and lse
        ctx.save_for_backward(hidden_states, weight, labels)
        ctx.softcap = logit_softcapping if logit_softcapping != 0 else None
        ctx.filter_eps = "auto"
        ctx.ignore_index = ignore_index

    @staticmethod
    def backward(ctx, grad_loss):
        hidden_states, weight, labels = ctx.saved_tensors

        de, dc = _LinearCEBackward.apply(
            grad_loss, hidden_states, weight, labels,
            ctx.softcap, ctx.filter_eps, ctx.ignore_index,
        )

        de = de.reshape(hidden_states.shape)
        # Return grads for: hidden_states, weight, labels, ignore_idx, softcap, scaling
        return de, dc, None, None, None, None

    @staticmethod
    def vmap(info, in_dims, hidden_states, weight, labels,
             ignore_index, logit_softcapping, logit_scaling):
        """Custom vmap rule for DP-SGD.

        Loops over vmap batch dimension, computing per-sample NLL sums
        independently to avoid cross-batch label shift contamination.
        """
        (h_bdim, w_bdim, lab_bdim,
         ii_bdim, sc_bdim, ls_bdim) = in_dims

        if h_bdim != 0:
            raise ValueError(f"hidden_states should be batched at dim 0, got {h_bdim}")
        if lab_bdim != 0:
            raise ValueError(f"labels should be batched at dim 0, got {lab_bdim}")
        assert w_bdim is None, "weight should not be batched"
        assert ii_bdim is None, "ignore_index should not be batched"
        assert sc_bdim is None, "logit_softcapping should not be batched"
        assert ls_bdim is None, "logit_scaling should not be batched"

        # Handle Granite scaling
        if logit_scaling != 0:
            weight = weight / logit_scaling
        softcap = logit_softcapping if logit_softcapping != 0 else None

        B_vmap = hidden_states.shape[0]
        nll_sums = []

        for i in range(B_vmap):
            nll_i, _ = _linear_ce_forward(
                hidden_states[i], weight, labels[i], ignore_index, softcap,
            )
            nll_sums.append(nll_i.sum())

        return torch.stack(nll_sums), 0


def opaque_linear_cross_entropy_loss(
    hidden_states, weight, labels,
    num_items_in_batch=None, ignore_index=-100,
    logit_softcapping=0, logit_scaling=0,
):
    """Convenience wrapper for fused linear + cross-entropy loss.

    The kernel returns nll_sum (unreduced). This wrapper divides by
    num_items_in_batch (if given) or count of valid tokens.

    Args:
        hidden_states: (..., hidden_dim) embeddings from backbone
        weight: (vocab_size, hidden_dim) lm_head weight
        labels: (...,) target token IDs (-100 = ignore)
        num_items_in_batch: optional denominator for loss averaging
        ignore_index: label value to ignore
        logit_softcapping: Gemma2 softcap value (0 = disabled)
        logit_scaling: Granite scaling divisor (0 = disabled)

    Returns:
        loss: scalar tensor
    """
    nll_sum = Opaque_LinearCrossEntropyLoss.apply(
        hidden_states, weight, labels,
        ignore_index, logit_softcapping, logit_scaling,
    )

    if num_items_in_batch is not None:
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(nll_sum.device)
        return nll_sum / num_items_in_batch

    shifted_labels = labels[..., 1:].contiguous().flatten()
    n_valid = (shifted_labels != ignore_index).sum().float().clamp(min=1)
    return nll_sum / n_valid
