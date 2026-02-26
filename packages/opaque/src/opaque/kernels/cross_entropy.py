"""Cross-entropy loss kernel with vmap support for DP-SGD."""
import triton
import triton.language as tl
import torch
from .utils import MAX_FUSED_SIZE, calculate_settings, torch_gpu_device, triton_cast


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
    logits_ptr += row_idx * triton_cast(logits_row_stride, tl.int64)
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
def _chunked_cross_entropy_forward(
    logits_ptr,
    logits_row_stride,
    loss_ptr,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Chunked forward for vocab > MAX_FUSED_SIZE.

    Each program handles one (row, chunk). Computes per-chunk logsumexp.
    Chunk 0 also stores -x[label] as the initial loss value.
    Python-side then does: loss = logsumexp(per_chunk_lse) + initial_loss
    """
    row_idx = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    logits_ptr += row_idx * triton_cast(logits_row_stride, tl.int64)
    loss_ptr += row_idx
    logsumexp_ptr += row_idx * N_CHUNKS + chunk_idx
    labels_ptr += row_idx

    col_offsets = chunk_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)

    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    # Chunk 0 stores the -x[label] part of the loss
    if chunk_idx == 0:
        if label_idx != -100:
            x = tl.load(logits_ptr + label_idx).to(tl.float32)
            loss = -1.0 * x
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

    logits_ptr += row_idx * triton_cast(logits_row_stride, tl.int64)
    logsumexp_ptr += row_idx
    labels_ptr += row_idx
    dlosses_ptr += row_idx * dlosses_row_stride

    col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    dloss = tl.load(dlosses_ptr)
    label_idx = tl.load(labels_ptr).to(tl.int32)

    # Zero gradient for masked positions (-100 = ignore_index)
    if label_idx == -100:
        dloss = 0.0

    logsumexp = tl.load(logsumexp_ptr)

    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
    probs = tl.exp(logits - logsumexp)

    dlogits = dloss * probs
    label_mask = (col_offsets == label_idx) & (label_idx != -100)
    dlogits = tl.where(label_mask, dlogits - dloss, dlogits)

    tl.store(logits_ptr + col_offsets, dlogits.to(logits_ptr.dtype.element_ty), mask=mask)


def _ce_forward_impl(logits_flat, labels_flat, n_rows, vocab_size, device):
    """Shared forward implementation for both standard and vmap paths.

    Returns (losses, logsumexp) both of shape (n_rows,).
    """
    div, mod = divmod(vocab_size, MAX_FUSED_SIZE)
    n_chunks = div + (mod != 0)

    losses = torch.empty(n_rows, dtype=torch.float32, device=device)

    if n_chunks == 1:
        # Small vocab (<= 65536): single-chunk path
        logsumexp = torch.empty(n_rows, dtype=torch.float32, device=device)
        BLOCK_SIZE, num_warps = calculate_settings(vocab_size)

        with torch_gpu_device(device):
            _cross_entropy_forward[(n_rows,)](
                logits_flat,
                logits_flat.stride(0),
                losses,
                logsumexp,
                labels_flat,
                VOCAB_SIZE=vocab_size,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
    else:
        # Large vocab (> 65536): chunked path
        logsumexp = torch.empty((n_rows, n_chunks), dtype=torch.float32, device=device)

        with torch_gpu_device(device):
            _chunked_cross_entropy_forward[(n_rows, n_chunks)](
                logits_flat,
                logits_flat.stride(0),
                losses,
                logsumexp,
                labels_flat,
                VOCAB_SIZE=vocab_size,
                N_CHUNKS=n_chunks,
                BLOCK_SIZE=MAX_FUSED_SIZE,
                num_warps=32,
            )

        # Reduce per-chunk logsumexp to global logsumexp
        logsumexp = torch.logsumexp(logsumexp, dim=1)
        # loss = -x[label] + logsumexp (chunk 0 stored -x, now add logsumexp)
        losses += logsumexp
        losses.masked_fill_(labels_flat == -100, 0)

    return losses, logsumexp


class _CrossEntropyBackward(torch.autograd.Function):
    """Backward pass wrapped as autograd.Function for vmap(grad()) support."""

    @staticmethod
    def forward(grad_losses, logits, logsumexp, labels, vocab_size):
        original_shape = logits.shape
        # In-place: kernel writes dlogits directly into logits buffer
        logits_flat = logits.reshape(-1, vocab_size).contiguous()
        logsumexp_flat = logsumexp.reshape(-1)
        labels_flat = labels.reshape(-1)
        grad_losses_flat = grad_losses.reshape(-1)
        n_rows = logits_flat.shape[0]

        BLOCK_SIZE = 4096
        div, mod = divmod(vocab_size, BLOCK_SIZE)
        n_blocks = div + (mod != 0)

        grid = (n_rows, n_blocks)
        with torch_gpu_device(logits.device):
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

        return logits_flat.reshape(original_shape)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Double backward not supported for CrossEntropy")

    @staticmethod
    def vmap(info, in_dims, grad_losses, logits, logsumexp, labels, vocab_size):
        grad_bdim, logits_bdim, lse_bdim, labels_bdim, vs_bdim = in_dims

        assert vs_bdim is None, "vocab_size should not be batched"

        original_shape = logits.shape
        # Merge vmap batch into rows, in-place (kernel writes dlogits into logits buffer)
        logits_flat = logits.reshape(-1, vocab_size).contiguous()
        logsumexp_flat = logsumexp.reshape(-1)
        labels_flat = labels.reshape(-1)
        grad_losses_flat = grad_losses.reshape(-1)
        n_rows = logits_flat.shape[0]

        BLOCK_SIZE = 4096
        div, mod = divmod(vocab_size, BLOCK_SIZE)
        n_blocks = div + (mod != 0)

        grid = (n_rows, n_blocks)
        with torch_gpu_device(logits.device):
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

        return logits_flat.reshape(original_shape), logits_bdim


class Opaque_CrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(logits, labels):
        """New-style API forward without ctx parameter."""
        original_shape = logits.shape[:-1]
        vocab_size = logits.shape[-1]

        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)
        n_rows = logits_flat.shape[0]
        device = logits.device

        losses, logsumexp = _ce_forward_impl(
            logits_flat, labels_flat, n_rows, vocab_size, device
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
        grad_logits = _CrossEntropyBackward.apply(
            grad_losses, logits, logsumexp, labels, ctx.vocab_size
        )
        return grad_logits, None

    @staticmethod
    def vmap(info, in_dims, logits, labels):
        """Custom vmap rule for DP-SGD.

        Calls Triton forward kernel directly with merged batch dims.
        """
        logits_bdim, labels_bdim = in_dims

        if logits_bdim != 0:
            raise ValueError(f"logits should be batched at dim 0, got {logits_bdim}")
        if labels_bdim != 0:
            raise ValueError(f"labels should be batched at dim 0, got {labels_bdim}")

        batched_shape = logits.shape[:-1]
        vocab_size = logits.shape[-1]

        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)
        n_rows = logits_flat.shape[0]
        device = logits.device

        losses, logsumexp = _ce_forward_impl(
            logits_flat, labels_flat, n_rows, vocab_size, device
        )

        losses = losses.reshape(batched_shape)
        logsumexp = logsumexp.reshape(batched_shape)

        return (losses, logsumexp), (logits_bdim, logits_bdim)


def opaque_cross_entropy(logits, labels):
    """Convenience wrapper."""
    losses, _ = Opaque_CrossEntropy.apply(logits, labels)
    return losses
