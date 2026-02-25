"""Cross-entropy loss kernel with vmap support for DP-SGD."""
import triton
import triton.language as tl
import torch
from .utils import calculate_settings, torch_gpu_device


@triton.jit
def _cross_entropy_forward(
    logits_ptr,
    logits_row_stride,
    loss_ptr,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    logits_ptr += row_idx * logits_row_stride
    loss_ptr += row_idx
    logsumexp_ptr += row_idx
    labels_ptr += row_idx

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)

    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if label_idx != -100:
        x = tl.load(logits_ptr + label_idx).to(tl.float32)
        loss = logsumexp - x
    else:
        loss = 0.0

    tl.store(loss_ptr, loss)
    tl.store(logsumexp_ptr, logsumexp)


@triton.jit
def _cross_entropy_backward(
    logits_ptr,
    logits_row_stride,
    dlosses_ptr,
    dlosses_row_stride,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    logits_ptr += row_idx * logits_row_stride
    logsumexp_ptr += row_idx
    labels_ptr += row_idx
    dlosses_ptr += row_idx * dlosses_row_stride

    col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    dloss = tl.load(dlosses_ptr)
    logsumexp = tl.load(logsumexp_ptr)
    label_idx = tl.load(labels_ptr).to(tl.int32)

    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    probs = tl.exp(logits - logsumexp)

    dlogits = dloss * probs
    label_mask = (col_offsets == label_idx) & (label_idx != -100)
    dlogits = tl.where(label_mask, dlogits - dloss, dlogits)

    tl.store(logits_ptr + col_offsets, dlogits.to(logits_ptr.dtype.element_ty), mask=mask)


class NewStyleCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(logits, labels):
        """New-style API forward without ctx parameter."""
        original_shape = logits.shape[:-1]
        vocab_size = logits.shape[-1]

        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)
        n_rows = logits_flat.shape[0]
        device = logits.device

        losses = torch.empty(n_rows, dtype=torch.float32, device=device)
        logsumexp = torch.empty(n_rows, dtype=torch.float32, device=device)

        BLOCK_SIZE, num_warps = calculate_settings(vocab_size)

        grid = (n_rows,)
        _cross_entropy_forward[grid](
            logits_flat,
            logits_flat.stride(0),
            losses,
            logsumexp,
            labels_flat,
            VOCAB_SIZE=vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        losses = losses.reshape(original_shape)
        logsumexp = logsumexp.reshape(original_shape)

        return losses, logsumexp

    @staticmethod
    def setup_context(ctx, inputs, output):
        logits, labels = inputs
        losses, logsumexp = output
        ctx.save_for_backward(logits, logsumexp, labels)
        ctx.vocab_size = logits.shape[-1]
        ctx.original_shape = logits.shape[:-1]

    @staticmethod
    def backward(ctx, grad_losses, grad_logsumexp):
        logits, logsumexp, labels = ctx.saved_tensors
        vocab_size = ctx.vocab_size
        original_shape = ctx.original_shape

        logits_flat = logits.reshape(-1, vocab_size)
        logsumexp_flat = logsumexp.reshape(-1)
        labels_flat = labels.reshape(-1)
        grad_losses_flat = grad_losses.reshape(-1)
        n_rows = logits_flat.shape[0]

        BLOCK_SIZE = 4096
        div, mod = divmod(vocab_size, BLOCK_SIZE)
        n_blocks = div + (mod != 0)

        grid = (n_rows, n_blocks)
        _cross_entropy_backward[grid](
            logits_flat,
            logits_flat.stride(0),
            grad_losses_flat,
            grad_losses_flat.stride(0) if grad_losses_flat.dim() > 0 else 0,
            logsumexp_flat,
            labels_flat,
            VOCAB_SIZE=vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
        )

        grad_logits = logits_flat.reshape(logits.shape)

        return grad_logits, None

    @staticmethod
    def vmap(info, in_dims, logits, labels):
        """Custom vmap rule for DP-SGD."""
        logits_bdim, labels_bdim = in_dims

        if logits_bdim != 0:
            raise ValueError(f"logits should be batched at dim 0, got {logits_bdim}")
        if labels_bdim != 0:
            raise ValueError(f"labels should be batched at dim 0, got {labels_bdim}")

        output = NewStyleCrossEntropy.apply(logits, labels)
        return output, (logits_bdim, logits_bdim)


def cross_entropy_vmap(logits, labels):
    """Convenience wrapper."""
    losses, _ = NewStyleCrossEntropy.apply(logits, labels)
    return losses
