#!/usr/bin/env python3
"""
Validation script for all kernels.

Generates a comparison table:
PyTorch (grad) | Unsloth (grad) | PyTorch (vmap) | Opaque (vmap)

For each kernel, reports:
- Forward time (ms)
- Backward time (ms)
- Peak memory (MB)
- Correctness (✓/✗)
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path

# Add kernels parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "opaque"))
sys.path.insert(0, str(Path(__file__).parent))
# Add unsloth to path
sys.path.insert(0, "/mnt/jetbrains/work/unsloth")

# Import kernels directly - avoid full opaque module init by importing from kernels submodule
from kernels.cross_entropy import NewStyleCrossEntropy
from kernels.swiglu import NewStyleSwiGLU
from kernels.geglu import NewStyleGeGLUExact
from kernels.layernorm import NewStyleLayerNorm
from kernels.rms_layernorm import RMSLayerNorm
from kernels.rope_embedding import NewStyleRoPEEmbedding
from kernels.lora import NewStyleLoRAW, NewStyleLoRAQKV, NewStyleLoRAMLP

# Import unsloth kernels for comparison
try:
    from unsloth.kernels.cross_entropy_loss import Fast_CrossEntropyLoss
    from unsloth.kernels.swiglu import swiglu_fg_kernel
    from unsloth.kernels.geglu import geglu_exact_forward_kernel
    from unsloth.kernels.layernorm import Fast_Layernorm
    from unsloth.kernels.rms_layernorm import Fast_RMS_Layernorm
    from unsloth.kernels.rope_embedding import Fast_RoPE_Embedding
    from unsloth.kernels.fast_lora import LoRA_W
    UNSLOTH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Unsloth kernels not available: {e}")
    UNSLOTH_AVAILABLE = False

from kernel_validation_framework import (
    benchmark_forward_backward,
    validate_implementations,
)


def format_time(ms):
    """Format time in milliseconds."""
    if ms < 0.001:
        return "<0.001"
    return f"{ms:.3f}"


def format_memory(mb):
    """Format memory in MB."""
    return f"{mb:.1f}"


def check_mark(validation_result):
    """Return ✓ if validation passes, ✗ otherwise."""
    if validation_result is None:
        return "N/A"
    return "✓" if (validation_result.forward_matches and validation_result.backward_matches) else "✗"


# ============================================================================
# Kernel Definitions
# ============================================================================

class CrossEntropyValidator:
    name = "CrossEntropy"

    @staticmethod
    def create_inputs(batch_size=4, seq_len=128, vocab_size=32000, vmap_batch=None, **kwargs):
        if vmap_batch:
            logits = torch.randn(vmap_batch, batch_size, seq_len, vocab_size, device="cuda", dtype=torch.float32, requires_grad=True)
            labels = torch.randint(0, vocab_size, (vmap_batch, batch_size, seq_len), device="cuda")
        else:
            logits = torch.randn(batch_size, seq_len, vocab_size, device="cuda", dtype=torch.float32, requires_grad=True)
            labels = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
        return [logits, labels], [0]

    @staticmethod
    def get_vmap_in_dims():
        """Both logits and labels are batched."""
        return (0, 0)

    @staticmethod
    def pytorch_impl(logits, labels):
        batch_seq = logits.shape[:-1]
        vocab_size = logits.shape[-1]
        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)
        return F.cross_entropy(logits_flat, labels_flat, reduction="mean")

    @staticmethod
    def opaque_impl(logits, labels):
        losses, _ = NewStyleCrossEntropy.apply(logits, labels)
        mask = (labels != -100).float()
        n_valid = mask.sum()
        masked_losses = losses * mask
        return masked_losses.sum() / torch.clamp(n_valid, min=1.0)

    @staticmethod
    def unsloth_impl(logits, labels):
        if UNSLOTH_AVAILABLE:
            # Unsloth requires 2D logits: (batch*seq, vocab)
            batch_seq = logits.shape[:-1]
            vocab_size = logits.shape[-1]
            logits_flat = logits.reshape(-1, vocab_size)
            labels_flat = labels.reshape(-1)
            losses = Fast_CrossEntropyLoss.apply(logits_flat, labels_flat)
            return losses.mean()
        return None


class SwiGLUValidator:
    name = "SwiGLU"

    @staticmethod
    def create_inputs(batch_size=4, seq_len=128, intermediate_dim=None, hidden_dim=None, vmap_batch=None, **kwargs):
        # Use intermediate_dim for FFN, fallback to hidden_dim
        dim = intermediate_dim if intermediate_dim else (hidden_dim if hidden_dim else 4096)
        if vmap_batch:
            gate = torch.randn(vmap_batch, batch_size, seq_len, dim, device="cuda", requires_grad=True)
            up = torch.randn(vmap_batch, batch_size, seq_len, dim, device="cuda", requires_grad=True)
        else:
            gate = torch.randn(batch_size, seq_len, dim, device="cuda", requires_grad=True)
            up = torch.randn(batch_size, seq_len, dim, device="cuda", requires_grad=True)
        return [gate, up], [0, 1]

    @staticmethod
    def get_vmap_in_dims():
        return (0, 0)  # Both gate and up are batched

    @staticmethod
    def pytorch_impl(gate, up):
        return F.silu(gate) * up

    @staticmethod
    def opaque_impl(gate, up):
        result = NewStyleSwiGLU.apply(gate, up)
        # SwiGLU returns (output, gate, up) tuple, extract only output
        return result[0] if isinstance(result, tuple) else result

    @staticmethod
    def unsloth_impl(gate, up):
        if UNSLOTH_AVAILABLE:
            return swiglu_fg_kernel(gate, up)
        return None


class GeGLUExactValidator:
    name = "GeGLU-Exact"

    @staticmethod
    def create_inputs(batch_size=4, seq_len=128, hidden_dim=4096, vmap_batch=None, **kwargs):
        if vmap_batch:
            gate = torch.randn(vmap_batch, batch_size, seq_len, hidden_dim, device="cuda", requires_grad=True)
            up = torch.randn(vmap_batch, batch_size, seq_len, hidden_dim, device="cuda", requires_grad=True)
        else:
            gate = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", requires_grad=True)
            up = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", requires_grad=True)
        return [gate, up], [0, 1]

    @staticmethod
    def get_vmap_in_dims():
        return (0, 0)  # Both gate and up are batched

    @staticmethod
    def pytorch_impl(gate, up):
        return F.gelu(gate, approximate="none") * up

    @staticmethod
    def opaque_impl(gate, up):
        result = NewStyleGeGLUExact.apply(gate, up)
        # GeGLU returns (output, gate, up) tuple, extract only output
        return result[0] if isinstance(result, tuple) else result

    @staticmethod
    def unsloth_impl(gate, up):
        if UNSLOTH_AVAILABLE:
            return geglu_exact_forward_kernel(gate, up)
        return None


class LayerNormValidator:
    name = "LayerNorm"

    @staticmethod
    def create_inputs(batch_size=4, seq_len=128, hidden_dim=4096, vmap_batch=None, **kwargs):
        if vmap_batch:
            x = torch.randn(vmap_batch, batch_size, seq_len, hidden_dim, device="cuda", requires_grad=True)
        else:
            x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", requires_grad=True)
        # Weight and bias are NOT batched - shared across vmap examples
        weight = torch.randn(hidden_dim, device="cuda", requires_grad=True)
        bias = torch.randn(hidden_dim, device="cuda", requires_grad=True)
        return [x, weight, bias], [0, 1, 2]

    @staticmethod
    def get_vmap_in_dims():
        return (0, None, None)  # Only x is batched

    @staticmethod
    def pytorch_impl(x, weight, bias):
        return F.layer_norm(x, (x.shape[-1],), weight, bias, eps=1e-5)

    @staticmethod
    def opaque_impl(x, weight, bias):
        # NewStyleLayerNorm returns (out, mean, var), but we only need out
        out = NewStyleLayerNorm.apply(x, weight, bias, 1e-5)
        if isinstance(out, tuple):
            return out[0]
        return out

    @staticmethod
    def unsloth_impl(x, weight, bias):
        if UNSLOTH_AVAILABLE:
            return Fast_Layernorm.apply(x, weight, bias, 1e-5)
        return None


class RMSLayerNormValidator:
    name = "RMSLayerNorm"

    @staticmethod
    def create_inputs(batch_size=4, seq_len=128, hidden_dim=4096, vmap_batch=None, **kwargs):
        if vmap_batch:
            x = torch.randn(vmap_batch, batch_size, seq_len, hidden_dim, device="cuda", requires_grad=True)
        else:
            x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", requires_grad=True)
        # Weight is NOT batched - shared across vmap examples
        weight = torch.randn(hidden_dim, device="cuda", requires_grad=True)
        return [x, weight], [0, 1]

    @staticmethod
    def get_vmap_in_dims():
        return (0, None)  # Only x is batched

    @staticmethod
    def pytorch_impl(x, weight):
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + 1e-5)
        return weight * x_normed

    @staticmethod
    def opaque_impl(x, weight):
        return RMSLayerNorm.apply(x, weight, 1e-5)

    @staticmethod
    def unsloth_impl(x, weight):
        if UNSLOTH_AVAILABLE:
            return Fast_RMS_Layernorm.apply(x, weight, 1e-5)
        return None


class RoPEValidator:
    name = "RoPE"

    @staticmethod
    def create_inputs(batch_size=4, seq_len=128, n_heads=32, head_dim=128, vmap_batch=None, **kwargs):
        if vmap_batch:
            Q = torch.randn(vmap_batch, batch_size, seq_len, n_heads, head_dim, device="cuda", requires_grad=True)
        else:
            Q = torch.randn(batch_size, seq_len, n_heads, head_dim, device="cuda", requires_grad=True)
        # Cos/sin are NOT batched - shared across vmap examples
        cos = torch.randn(1, seq_len, 1, head_dim, device="cuda")
        sin = torch.randn(1, seq_len, 1, head_dim, device="cuda")
        return [Q, cos, sin], [0]

    @staticmethod
    def get_vmap_in_dims():
        return (0, None, None)  # Only Q is batched

    @staticmethod
    def pytorch_impl(Q, cos, sin):
        # PyTorch RoPE implementation (even/odd interleaving - matches Triton kernels)
        # RoPE applies rotation: [x_even, x_odd] -> [x_even*cos - x_odd*sin, x_even*sin + x_odd*cos]
        shape = Q.shape
        seq_len = cos.shape[1]
        Q_reshaped = Q.view(*shape[:-1], -1, 2)

        # Get even and odd positions
        Q_even = Q_reshaped[..., 0]  # even indices [..., 0, 2, 4, ...]
        Q_odd = Q_reshaped[..., 1]   # odd indices  [..., 1, 3, 5, ...]

        # Reshape cos/sin similarly
        cos_reshaped = cos.view(1, seq_len, 1, -1, 2)
        sin_reshaped = sin.view(1, seq_len, 1, -1, 2)
        cos_even = cos_reshaped[..., 0]
        sin_even = sin_reshaped[..., 0]

        # Apply rotation
        Q_even_out = Q_even * cos_even - Q_odd * sin_even
        Q_odd_out = Q_even * sin_even + Q_odd * cos_even

        # Recombine
        Q_out = torch.stack([Q_even_out, Q_odd_out], dim=-1)
        return Q_out.view(shape)

    @staticmethod
    def opaque_impl(Q, cos, sin):
        result = NewStyleRoPEEmbedding.apply(Q, cos, sin)
        # RoPE returns (Q_out, cos, sin, BLOCK_SIZE, num_warps, n_groups)
        # We only need Q_out for validation
        if isinstance(result, tuple):
            return result[0]
        return result

    @staticmethod
    def unsloth_impl(Q, cos, sin):
        if UNSLOTH_AVAILABLE:
            return Fast_RoPE_Embedding.apply(Q, cos, sin)
        return None


class LoRAWValidator:
    name = "LoRA-W"

    @staticmethod
    def create_inputs(batch_size=4, seq_len=128, hidden_dim=3072, rank=64, vmap_batch=None, **kwargs):
        in_dim = out_dim = hidden_dim
        if vmap_batch:
            x = torch.randn(vmap_batch, batch_size, seq_len, in_dim, device="cuda", requires_grad=True)
        else:
            x = torch.randn(batch_size, seq_len, in_dim, device="cuda", requires_grad=True)
        # W, A, B, scaling are NOT batched - shared across vmap examples
        W = torch.randn(out_dim, in_dim, device="cuda")
        # Opaque LoRA: X @ A @ B where A is (in_dim, rank), B is (out_dim, rank)
        A_opaque = torch.randn(in_dim, rank, device="cuda", requires_grad=True)
        B_opaque = torch.randn(out_dim, rank, device="cuda", requires_grad=True)
        # Unsloth LoRA: X @ A.T @ B.T where A is (rank, in_dim), B is (out_dim, rank)
        A_unsloth = A_opaque.t().contiguous()  # (rank, in_dim)
        scaling = 1.0
        return [x, W, A_opaque, A_unsloth, B_opaque, scaling], [0, 2, 4]

    @staticmethod
    def get_vmap_in_dims():
        return (0, None, None, None, None, None)  # Only x is batched

    @staticmethod
    def pytorch_impl(x, W, A_opaque, A_unsloth, B, scaling):
        # Use opaque-style computation for PyTorch reference
        out = F.linear(x, W)
        if A_opaque is not None and B is not None:
            lora_out = (x @ A_opaque) @ B.t() * scaling
            out = out + lora_out
        return out

    @staticmethod
    def opaque_impl(x, W, A_opaque, A_unsloth, B, scaling):
        return NewStyleLoRAW.apply(x, W, A_opaque, B, scaling)

    @staticmethod
    def unsloth_impl(x, W, A_opaque, A_unsloth, B, scaling):
        if UNSLOTH_AVAILABLE:
            # Unsloth LoRA signature: (X, W, W_quant, A, B, S)
            # A_unsloth is (rank, in_dim)
            return LoRA_W.apply(x, W, None, A_unsloth, B, scaling)
        return None


class LoRAQKVValidator:
    name = "LoRA-QKV"

    @staticmethod
    def create_inputs(batch_size=4, seq_len=128, hidden_dim=3072, rank=64, vmap_batch=None, **kwargs):
        if vmap_batch:
            x = torch.randn(vmap_batch, batch_size, seq_len, hidden_dim, device="cuda", requires_grad=True)
        else:
            x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", requires_grad=True)

        # Weights (not batched)
        Wq = torch.randn(hidden_dim, hidden_dim, device="cuda")
        Wk = torch.randn(hidden_dim, hidden_dim, device="cuda")
        Wv = torch.randn(hidden_dim, hidden_dim, device="cuda")

        # LoRA weights (not batched)
        Aq = torch.randn(hidden_dim, rank, device="cuda", requires_grad=True)
        Bq = torch.randn(hidden_dim, rank, device="cuda", requires_grad=True)
        Ak = torch.randn(hidden_dim, rank, device="cuda", requires_grad=True)
        Bk = torch.randn(hidden_dim, rank, device="cuda", requires_grad=True)
        Av = torch.randn(hidden_dim, rank, device="cuda", requires_grad=True)
        Bv = torch.randn(hidden_dim, rank, device="cuda", requires_grad=True)

        Sq = Sk = Sv = 1.0

        return [x, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv], [0, 2, 4, 6, 8, 10, 12]

    @staticmethod
    def get_vmap_in_dims():
        # Only x is batched, all weights shared
        return (0, None, None, None, None, None, None, None, None, None, None, None, None)

    @staticmethod
    def pytorch_impl(x, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv):
        Q = F.linear(x, Wq)
        if Aq is not None and Bq is not None:
            Q = Q + (x @ Aq) @ Bq.t() * Sq

        K = F.linear(x, Wk)
        if Ak is not None and Bk is not None:
            K = K + (x @ Ak) @ Bk.t() * Sk

        V = F.linear(x, Wv)
        if Av is not None and Bv is not None:
            V = V + (x @ Av) @ Bv.t() * Sv

        # Return combined output for simpler validation
        return torch.cat([Q, K, V], dim=-1)

    @staticmethod
    def opaque_impl(x, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv):
        Q, K, V = NewStyleLoRAQKV.apply(x, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv)
        return torch.cat([Q, K, V], dim=-1)

    @staticmethod
    def unsloth_impl(x, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv):
        # Unsloth doesn't have fused QKV, would need 3 separate calls
        return None


class LoRAMLPValidator:
    name = "LoRA-MLP"

    @staticmethod
    def create_inputs(batch_size=4, seq_len=128, hidden_dim=3072, intermediate_dim=8256, rank=64, vmap_batch=None, **kwargs):
        if vmap_batch:
            x = torch.randn(vmap_batch, batch_size, seq_len, hidden_dim, device="cuda", requires_grad=True)
        else:
            x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", requires_grad=True)

        # Weights (not batched)
        Wg = torch.randn(intermediate_dim, hidden_dim, device="cuda")
        Wu = torch.randn(intermediate_dim, hidden_dim, device="cuda")
        Wd = torch.randn(hidden_dim, intermediate_dim, device="cuda")

        # LoRA weights (not batched)
        Ag = torch.randn(hidden_dim, rank, device="cuda", requires_grad=True)
        Bg = torch.randn(intermediate_dim, rank, device="cuda", requires_grad=True)
        Au = torch.randn(hidden_dim, rank, device="cuda", requires_grad=True)
        Bu = torch.randn(intermediate_dim, rank, device="cuda", requires_grad=True)
        Ad = torch.randn(intermediate_dim, rank, device="cuda", requires_grad=True)
        Bd = torch.randn(hidden_dim, rank, device="cuda", requires_grad=True)

        Sg = Su = Sd = 1.0

        return [x, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd], [0, 2, 4, 6, 8, 10, 12]

    @staticmethod
    def get_vmap_in_dims():
        # Only x is batched, all weights shared
        return (0, None, None, None, None, None, None, None, None, None, None, None, None)

    @staticmethod
    def pytorch_impl(x, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd):
        gate = F.linear(x, Wg)
        if Ag is not None and Bg is not None:
            gate = gate + (x @ Ag) @ Bg.t() * Sg

        up = F.linear(x, Wu)
        if Au is not None and Bu is not None:
            up = up + (x @ Au) @ Bu.t() * Su

        h = F.silu(gate) * up

        out = F.linear(h, Wd)
        if Ad is not None and Bd is not None:
            out = out + (h @ Ad) @ Bd.t() * Sd

        return out

    @staticmethod
    def opaque_impl(x, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd):
        out, _, _, _ = NewStyleLoRAMLP.apply(x, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd)
        return out

    @staticmethod
    def unsloth_impl(x, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd):
        # Unsloth doesn't have fused MLP with LoRA
        return None

    # Note: No custom tolerance needed! Using relative-only error which is magnitude-independent


# ============================================================================
# Validation Runner
# ============================================================================

def validate_kernel(validator_class, config):
    """Run validation for a single kernel."""
    validator = validator_class()

    # Standard (non-vmap) inputs
    inputs, grad_idx = validator.create_inputs(**config)

    # Vmap inputs
    vmap_config = {**config, "vmap_batch": 4}
    vmap_inputs, _ = validator.create_inputs(**vmap_config)

    results = {
        "name": validator.name,
        "pytorch": None,
        "opaque": None,
        "unsloth": None,
        "pytorch_vmap": None,
        "opaque_vmap": None,
        "validation_pytorch": None,
        "validation_unsloth": None,
        "validation_vmap": None,
    }

    # Benchmark parameters: proper warmup and multiple runs
    N_WARMUP = 10
    N_RUNS = 50

    # Get custom tolerances if available
    # Default: rtol=1e-5 (much stricter), relative-only comparison
    tolerances = validator.get_tolerances() if hasattr(validator, 'get_tolerances') else {"rtol": 1e-5, "atol": 1e-5, "use_relative_only": True}
    rtol = tolerances.get("rtol", 1e-5)  # Stricter: 1e-5 instead of 1e-3
    atol = tolerances.get("atol", 1e-5)  # Much stricter: 1e-5 instead of 1e-3
    use_relative_only = tolerances.get("use_relative_only", True)  # Default to relative-only (magnitude-independent)

    try:
        # Benchmark PyTorch (standard grad)
        results["pytorch"] = benchmark_forward_backward(
            validator.pytorch_impl, inputs, grad_idx, n_warmup=N_WARMUP, n_runs=N_RUNS, name="PyTorch"
        )
    except Exception as e:
        print(f"PyTorch benchmark failed: {e}")

    try:
        # Benchmark Opaque (standard grad)
        results["opaque"] = benchmark_forward_backward(
            validator.opaque_impl, inputs, grad_idx, n_warmup=N_WARMUP, n_runs=N_RUNS, name="Opaque"
        )
        # Validate Opaque vs PyTorch (standard grad)
        if results["pytorch"] is not None:
            results["validation_pytorch"] = validate_implementations(
                validator.opaque_impl,
                validator.pytorch_impl,
                inputs, grad_idx,
                name="Opaque vs PyTorch",
                rtol=rtol, atol=atol, use_relative_only=use_relative_only
            )
    except Exception as e:
        print(f"Opaque benchmark/validation failed: {e}")

    # Benchmark Unsloth (standard grad - if available)
    if validator.unsloth_impl is not None:
        try:
            results["unsloth"] = benchmark_forward_backward(
                validator.unsloth_impl, inputs, grad_idx, n_warmup=N_WARMUP, n_runs=N_RUNS, name="Unsloth"
            )
            # Validate Opaque vs Unsloth (standard grad)
            results["validation_unsloth"] = validate_implementations(
                validator.opaque_impl,
                validator.unsloth_impl,
                inputs, grad_idx,
                name="Opaque vs Unsloth",
                rtol=1e-4, atol=1e-4
            )
        except Exception as e:
            print(f"Unsloth benchmark/validation failed: {e}")

    try:
        # Benchmark PyTorch vmap
        # Get in_dims from validator
        in_dims = validator.get_vmap_in_dims()

        def pytorch_vmap_fn(*args):
            return torch.vmap(validator.pytorch_impl, in_dims=in_dims)(*args)

        results["pytorch_vmap"] = benchmark_forward_backward(
            pytorch_vmap_fn, vmap_inputs, grad_idx, n_warmup=N_WARMUP, n_runs=N_RUNS, name="PyTorch vmap"
        )
    except Exception as e:
        print(f"PyTorch vmap benchmark failed: {e}")

    try:
        # Benchmark Opaque vmap
        # Get in_dims from validator
        in_dims = validator.get_vmap_in_dims()

        def opaque_vmap_fn(*args):
            return torch.vmap(validator.opaque_impl, in_dims=in_dims)(*args)

        results["opaque_vmap"] = benchmark_forward_backward(
            opaque_vmap_fn, vmap_inputs, grad_idx, n_warmup=N_WARMUP, n_runs=N_RUNS, name="Opaque vmap"
        )

        # Validate Opaque vmap vs PyTorch vmap
        if results["pytorch_vmap"] is not None:
            results["validation_vmap"] = validate_implementations(
                opaque_vmap_fn,
                pytorch_vmap_fn,
                vmap_inputs, grad_idx,
                name="Opaque vmap vs PyTorch vmap",
                rtol=rtol, atol=atol, use_relative_only=use_relative_only
            )
    except Exception as e:
        print(f"Opaque vmap benchmark/validation failed: {e}")

    return results


# ============================================================================
# Table Formatter
# ============================================================================

def print_validation_table(all_results):
    """Print a formatted table with all validation results."""

    print("\n" + "="*160)
    print("KERNEL VALIDATION RESULTS")
    print("="*160)
    print()
    print(f"{'Kernel':<16} | {'PyTorch (grad)':<20} | {'Opaque (grad)':<20} | {'Unsloth (grad)':<20} | {'PyTorch (vmap)':<20} | {'Opaque (vmap)':<20}")
    print(f"{'':16} | {'Time | Mem':<20} | {'Time | Mem | ✓':<20} | {'Time | Mem | ✓':<20} | {'Time | Mem':<20} | {'Time | Mem | ✓':<20}")
    print("-"*160)

    for result in all_results:
        name = result["name"]

        # PyTorch (grad)
        pt = result["pytorch"]
        pt_str = f"{format_time(pt.total_time_ms):>5} | {format_memory(pt.memory_peak_mb):>7}" if pt else "N/A".center(20)

        # Opaque (grad)
        opaque = result["opaque"]
        val_pt = result["validation_pytorch"]
        check_pt = check_mark(val_pt)
        opaque_str = f"{format_time(opaque.total_time_ms):>5} | {format_memory(opaque.memory_peak_mb):>7} | {check_pt:^3}" if opaque else "N/A".center(20)

        # Unsloth (grad)
        unsloth = result["unsloth"]
        val_un = result["validation_unsloth"]
        check_un = check_mark(val_un)
        unsloth_str = f"{format_time(unsloth.total_time_ms):>5} | {format_memory(unsloth.memory_peak_mb):>7} | {check_un:^3}" if unsloth else "N/A".center(20)

        # PyTorch vmap
        pt_vmap = result["pytorch_vmap"]
        pt_vmap_str = f"{format_time(pt_vmap.total_time_ms):>5} | {format_memory(pt_vmap.memory_peak_mb):>7}" if pt_vmap else "N/A".center(20)

        # Opaque vmap
        opaque_vmap = result["opaque_vmap"]
        validation = result["validation_vmap"]
        check = check_mark(validation)
        opaque_vmap_str = f"{format_time(opaque_vmap.total_time_ms):>5} | {format_memory(opaque_vmap.memory_peak_mb):>7} | {check:^3}" if opaque_vmap else "N/A".center(20)

        print(f"{name:<16} | {pt_str:<20} | {opaque_str:<20} | {unsloth_str:<20} | {pt_vmap_str:<20} | {opaque_vmap_str:<20}")

    print("="*160)
    print("\nLegend:")
    print("  Time: Forward + Backward pass (milliseconds)")
    print("  Mem: Peak memory usage (MB)")
    print("  ✓/✗: Correctness validation (✓ = passes, ✗ = fails)")
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        sys.exit(1)

    # Mellum model configuration (JetBrains/Mellum-4b-base)
    # 4B parameters, 30 layers, 24 attention heads
    # Note: vocab_size reduced to 32K due to Triton kernel limitations
    config = {
        "batch_size": 8,          # DP-SGD batch size
        "seq_len": 512,           # Sequence length (context: 8192)
        "vocab_size": 32000,      # Reduced from 98304 due to Triton blocksize limit
        "hidden_dim": 3072,       # Hidden size / embedding length
        "intermediate_dim": 8256, # FFN intermediate size
        "n_heads": 24,            # Attention heads
        "head_dim": 128,          # RoPE dimension
        "in_dim": 3072,
        "out_dim": 3072,
        "rank": 64,               # LoRA rank
    }

    validators = [
        CrossEntropyValidator,
        SwiGLUValidator,
        GeGLUExactValidator,
        LayerNormValidator,
        RMSLayerNormValidator,
        RoPEValidator,
        LoRAWValidator,
        LoRAQKVValidator,
        LoRAMLPValidator,
    ]

    all_results = []

    for validator_class in validators:
        print(f"\n{'#'*120}")
        print(f"# Validating: {validator_class.name}")
        print(f"{'#'*120}")

        try:
            result = validate_kernel(validator_class, config)
            all_results.append(result)
        except Exception as e:
            print(f"Validation failed for {validator_class.name}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "name": validator_class.name,
                "pytorch": None,
                "unsloth": None,
                "pytorch_vmap": None,
                "opaque_vmap": None,
                "validation_pytorch": None,
                "validation_vmap": None,
            })

    # Print final table
    print_validation_table(all_results)


if __name__ == "__main__":
    main()
