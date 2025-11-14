"""Integration tests for gradient equivalence across different computation methods.

Tests full end-to-end scenarios with realistic models (both custom and HuggingFace),
validating that clipped_grad produces identical gradients to standard approaches.

Parametrized across multiple model architectures to ensure broad compatibility.
"""

import pytest
import torch

from opaque.clipping import clipped_grad
from opaque.utils import global_norm, make_functional
# Import shared fixtures
from tests.integration.conftest import (
    compute_causal_lm_loss,
    create_custom_llama,
    create_huggingface_model,
)

# Optional: Import transformers if available
try:
    import transformers

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# ============================================================================
# Test Parametrization
# ============================================================================

# Define test cases
# Models tested:
# - Custom LLaMA: Minimal custom implementation for fast testing
# - GPT-2 variants: Classic transformer decoder (2019)
# - Qwen 2/2.5: Modern Chinese LLM from Alibaba (2024)
# - Gemma: Google's open model family (2024)
TEST_MODELS = [
    pytest.param("custom_llama", id="custom_llama"),
    # GPT-2 variants (older architecture)
    pytest.param(
        "gpt2",
        marks=[
            pytest.mark.skipif(not HAS_TRANSFORMERS, reason="requires transformers"),
        ],
        id="gpt2",
    ),
    pytest.param(
        "Qwen/Qwen2.5-0.5B",
        marks=[
            pytest.mark.skipif(not HAS_TRANSFORMERS, reason="requires transformers"),
            pytest.mark.slow(),
        ],
        id="qwen2.5-0.5b",
    ),
    pytest.param(
        "google/gemma-3-270m",
        marks=[
            pytest.mark.skipif(not HAS_TRANSFORMERS, reason="requires transformers"),
            pytest.mark.slow(),
        ],
        id="gemma-3",
    ),
]


# ============================================================================
# Helper Functions
# ============================================================================


def compute_gradients_with_backward(model, tokens):
    """Method 1: Standard .backward() on stateful model."""
    model.zero_grad()
    logits = model(tokens)
    loss = compute_causal_lm_loss(logits, tokens)
    loss.backward()

    grads = {
        name: param.grad.clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }

    # Compute gradient norm using global_norm
    grad_norm = global_norm(grads).item()

    return grads, loss.item(), grad_norm


def convert_to_functional(model):
    """Convert model to functional form and return param names."""
    fmodel, params = make_functional(model, disable_autograd_tracking=True)
    param_names = [name for name, _ in model.named_parameters()]
    return fmodel, params, param_names


def compute_gradients_with_functional_batch(fmodel, params, param_names, tokens):
    """Method 2: Functional + torch.func.grad (batch)."""

    def batch_loss_fn(params_tuple, tokens_batch):
        logits = fmodel(params_tuple, tokens_batch)
        return compute_causal_lm_loss(logits, tokens_batch)

    # Compute gradients
    grad_fn = torch.func.grad(batch_loss_fn)
    grads_tuple = grad_fn(params, tokens)

    # Convert tuple to dict
    grads = {name: grad for name, grad in zip(param_names, grads_tuple)}

    loss = batch_loss_fn(params, tokens)

    # Compute gradient norm using global_norm
    grad_norm = global_norm(grads).item()

    return grads, loss.item(), grad_norm


def compute_gradients_with_manual_iteration(fmodel, params, param_names, tokens):
    """Method 3: Functional + manual per-example iteration."""
    batch_size = tokens.shape[0]

    def per_example_loss_fn(params_tuple, tokens_single):
        # Add batch dimension
        tokens_batch = tokens_single.unsqueeze(0)
        logits = fmodel(params_tuple, tokens_batch)
        return compute_causal_lm_loss(logits, tokens_batch)

    # Accumulate gradients manually
    grad_fn_single = torch.func.grad(per_example_loss_fn)

    accumulated_grads = None
    per_example_losses = []
    per_example_norms = []

    for i in range(batch_size):
        tokens_single = tokens[i]

        # Compute per-example gradient
        grads_single = grad_fn_single(params, tokens_single)

        # Compute per-example gradient norm using global_norm
        grad_norm = global_norm(grads_single).item()
        per_example_norms.append(grad_norm)

        # Compute loss
        loss_single = per_example_loss_fn(params, tokens_single)
        per_example_losses.append(loss_single.item())

        # Accumulate
        if accumulated_grads is None:
            accumulated_grads = grads_single
        else:
            accumulated_grads = tuple(
                acc + grad for acc, grad in zip(accumulated_grads, grads_single)
            )

    # Average
    grads_tuple = tuple(grad / batch_size for grad in accumulated_grads)
    grads = {name: grad for name, grad in zip(param_names, grads_tuple)}

    loss_avg = sum(per_example_losses) / len(per_example_losses)

    # Compute norm of the averaged gradient (not average of per-example norms)
    grad_norm = global_norm(grads).item()

    return grads, loss_avg, grad_norm


def compute_gradients_with_clipped_grad(fmodel, params, param_names, tokens):
    """Method 4: Functional + clipped_grad (large clip norm)."""
    batch_size = tokens.shape[0]

    def per_example_loss_fn(params_tuple, tokens_single):
        # Add batch dimension
        tokens_batch = tokens_single.unsqueeze(0)
        logits = fmodel(params_tuple, tokens_batch)
        return compute_causal_lm_loss(logits, tokens_batch)

    # Use very large clip norm (no actual clipping)
    large_clip_norm = 1e6

    clipped_grad_fn = clipped_grad(
        per_example_loss_fn,
        argnums=0,
        batch_argnums=(1,),
        l2_clip_norm=large_clip_norm,
        keep_batch_dim=False,
        return_grad_norms=True,
        return_values=True,
    )

    grads_tuple, aux = clipped_grad_fn(params, tokens)

    # Average
    grads_tuple_avg = tuple(grad / batch_size for grad in grads_tuple)
    grads = {name: grad for name, grad in zip(param_names, grads_tuple_avg)}

    loss_avg = aux.values.mean().item()

    # Compute norm of the averaged gradient (not average of per-example norms)
    grad_norm = global_norm(grads).item()

    return grads, loss_avg, grad_norm


# ============================================================================
# Main Test
# ============================================================================


@pytest.mark.integration
@pytest.mark.parametrize("model_id", TEST_MODELS)
def test_gradient_equivalence(model_id):
    """
    Integration test: Gradient equivalence verification across models.

    Tests that make_functional + clipped_grad work correctly by comparing
    4 gradient computation methods across different model architectures.

    Args:
        model_id: Model identifier (custom_llama or HuggingFace model name)
    """
    # =========================================================================
    # Setup model and data
    # =========================================================================
    if model_id == "custom_llama":
        model, tokens = create_custom_llama()
    else:
        model, tokens = create_huggingface_model(model_id)

    batch_size, seq_len = tokens.shape

    print("\n" + "=" * 80)
    print(f"Testing: {model_id}")
    print(f"Batch size: {batch_size}, Seq len: {seq_len}")
    print("=" * 80)

    # =========================================================================
    # Compute gradients using all 4 methods
    # =========================================================================

    # Method 1: Standard .backward()
    grads_standard, loss_standard, norm_standard = compute_gradients_with_backward(model, tokens)
    print(f"\n[1/4] Standard .backward(): loss={loss_standard:.6f}, norm={norm_standard:.2f}")

    # Convert to functional (shared by methods 2-4)
    fmodel, params, param_names = convert_to_functional(model)

    # Method 2: Functional + torch.func.grad (batch)
    grads_functional_batch, loss_functional_batch, norm_functional_batch = (
        compute_gradients_with_functional_batch(fmodel, params, param_names, tokens)
    )
    print(
        f"[2/4] Functional batch: loss={loss_functional_batch:.6f}, norm={norm_functional_batch:.2f}"
    )

    # Method 3: Functional + manual per-example iteration
    grads_manual, loss_manual, norm_manual = compute_gradients_with_manual_iteration(
        fmodel, params, param_names, tokens
    )
    print(f"[3/4] Manual iteration: loss={loss_manual:.6f}, norm={norm_manual:.2f}")

    # Method 4: Functional + clipped_grad
    grads_clipped, loss_clipped, norm_clipped = compute_gradients_with_clipped_grad(
        fmodel, params, param_names, tokens
    )
    print(f"[4/4] Clipped grad: loss={loss_clipped:.6f}, norm={norm_clipped:.2f}")

    # =========================================================================
    # Compare all methods
    # =========================================================================
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    # Loss comparison
    loss_tol = 1e-5
    assert abs(loss_standard - loss_functional_batch) < loss_tol, (
        f"Functional batch loss mismatch: {loss_functional_batch} vs {loss_standard}"
    )
    assert abs(loss_standard - loss_manual) < loss_tol, (
        f"Manual iteration loss mismatch: {loss_manual} vs {loss_standard}"
    )
    assert abs(loss_standard - loss_clipped) < loss_tol, (
        f"Clipped grad loss mismatch: {loss_clipped} vs {loss_standard}"
    )

    print("✓ All losses match within tolerance")

    # Gradient comparison
    # Use more lenient tolerance for large models (100M+ params)
    num_params = sum(p.numel() for p in model.parameters())
    if num_params > 10_000_000:
        atol, rtol = 5e-4, 5e-4
    else:
        atol, rtol = 5e-5, 5e-5

    mismatches = []
    for name in grads_standard.keys():
        grad_std = grads_standard[name]
        grad_func = grads_functional_batch[name]
        grad_man = grads_manual[name]
        grad_clip = grads_clipped[name]

        close_func = torch.allclose(grad_std, grad_func, atol=atol, rtol=rtol)
        close_man = torch.allclose(grad_std, grad_man, atol=atol, rtol=rtol)
        close_clip = torch.allclose(grad_std, grad_clip, atol=atol, rtol=rtol)

        if not (close_func and close_man and close_clip):
            norm = grad_std.norm().item()
            err_func = (grad_std - grad_func).norm().item()
            err_man = (grad_std - grad_man).norm().item()
            err_clip = (grad_std - grad_clip).norm().item()

            pct_func = (err_func / norm * 100) if norm > 0 else 0
            pct_man = (err_man / norm * 100) if norm > 0 else 0
            pct_clip = (err_clip / norm * 100) if norm > 0 else 0

            mismatches.append(
                (name, close_func, close_man, close_clip, pct_func, pct_man, pct_clip)
            )

    if mismatches:
        print(f"\n✗ {len(mismatches)} parameters have gradient mismatches:")
        print(f"\nGradient norms comparison:")
        print(f"  Standard:  {norm_standard:.6f}")
        print(f"  Functional: {norm_functional_batch:.6f}")
        print(f"  Manual:     {norm_manual:.6f}")
        print(f"  Clipped:    {norm_clipped:.6f}")
        print(f"\nTolerance: atol={atol:.2e}, rtol={rtol:.2e}")

        for name, cf, cm, cc, pf, pm, pc in mismatches[:5]:
            grad_std = grads_standard[name]
            grad_func = grads_functional_batch[name]
            grad_man = grads_manual[name]
            grad_clip = grads_clipped[name]

            norm_std = grad_std.norm().item()
            err_func_abs = (grad_std - grad_func).abs().max().item()
            err_man_abs = (grad_std - grad_man).abs().max().item()
            err_clip_abs = (grad_std - grad_clip).abs().max().item()

            print(f"\n  - {name}:")
            print(f"      Param norm: {norm_std:.6e}")
            print(
                f"      Functional: {'✓' if cf else '✗'} (rel: {pf:.4f}%, max_abs_diff: {err_func_abs:.6e})"
            )
            print(
                f"      Manual:     {'✓' if cm else '✗'} (rel: {pm:.4f}%, max_abs_diff: {err_man_abs:.6e})"
            )
            print(
                f"      Clipped:    {'✓' if cc else '✗'} (rel: {pc:.4f}%, max_abs_diff: {err_clip_abs:.6e})"
            )
        if len(mismatches) > 5:
            print(f"\n  ... and {len(mismatches) - 5} more")

        pytest.fail(f"{len(mismatches)} parameters have gradient mismatches")

    print("✓ All gradients match within tolerance")
    print("✓ Test PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    # Run with: python -m pytest tests/integration/test_gradient_equivalence.py -v -s
    pytest.main([__file__, "-v", "-s"])
