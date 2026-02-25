"""
Kernel Validation Framework for Opaque

This framework validates ported kernels through a 3-phase comparison:

1. Baseline comparison: Unsloth kernel vs PyTorch (standard autograd)
   - Verifies Unsloth kernel correctness
   - Documents memory/speed characteristics

2. API migration: Opaque kernel (new API) vs Unsloth kernel
   - Should match Unsloth's performance/memory exactly
   - Proves we didn't break anything in the rewrite

3. Vmap support: Opaque vmap vs torch.vmap(pytorch_implementation)
   - Validates vmap correctness for DP-SGD
   - Reports memory overhead from vmap
"""

import time
import gc
from dataclasses import dataclass, field
from typing import Callable, Optional, Any
from contextlib import contextmanager

import torch
import torch.nn.functional as F


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    forward_time_ms: float
    backward_time_ms: float
    total_time_ms: float
    memory_allocated_mb: float
    memory_peak_mb: float
    output_shape: tuple
    grad_shapes: dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of comparing two implementations."""
    name: str
    forward_max_diff: float
    backward_max_diff: dict
    forward_matches: bool
    backward_matches: bool
    rtol: float = 1e-7
    atol: float = 1e-7


@contextmanager
def cuda_memory_tracker():
    """Context manager to track CUDA memory usage."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    start_mem = torch.cuda.memory_allocated()
    yield

    torch.cuda.synchronize()
    end_mem = torch.cuda.memory_allocated()
    peak_mem = torch.cuda.max_memory_allocated()


def benchmark_forward_backward(
    fn: Callable,
    inputs: list,
    grad_inputs_idx: list[int],
    n_warmup: int = 3,
    n_runs: int = 10,
    name: str = "benchmark"
) -> BenchmarkResult:
    """Benchmark forward and backward pass of a function.

    Args:
        fn: Function to benchmark
        inputs: List of input tensors
        grad_inputs_idx: Indices of inputs that need gradients
        n_warmup: Number of warmup iterations
        n_runs: Number of timed iterations
        name: Name for the benchmark

    Returns:
        BenchmarkResult with timing and memory stats
    """
    device = inputs[0].device

    # Prepare inputs with gradients
    def prepare_inputs():
        prepared = []
        for i, inp in enumerate(inputs):
            if i in grad_inputs_idx and isinstance(inp, torch.Tensor):
                prepared.append(inp.detach().clone().requires_grad_(True))
            elif isinstance(inp, torch.Tensor):
                prepared.append(inp.detach().clone())
            else:
                prepared.append(inp)
        return prepared

    # Warmup
    for _ in range(n_warmup):
        prep_inputs = prepare_inputs()
        out = fn(*prep_inputs)
        if isinstance(out, tuple):
            out = out[0]
        if out.requires_grad:
            out.sum().backward()
        torch.cuda.synchronize()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Timed runs
    forward_times = []
    backward_times = []

    for _ in range(n_runs):
        prep_inputs = prepare_inputs()

        torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn(*prep_inputs)
        torch.cuda.synchronize()
        forward_time = time.perf_counter() - start
        forward_times.append(forward_time)

        if isinstance(out, tuple):
            main_out = out[0]
        else:
            main_out = out

        if main_out.requires_grad:
            torch.cuda.synchronize()
            start = time.perf_counter()
            main_out.sum().backward()
            torch.cuda.synchronize()
            backward_time = time.perf_counter() - start
            backward_times.append(backward_time)

    # Memory stats
    memory_allocated = torch.cuda.memory_allocated() / 1024**2
    memory_peak = torch.cuda.max_memory_allocated() / 1024**2

    # Get output and grad shapes
    prep_inputs = prepare_inputs()
    out = fn(*prep_inputs)
    if isinstance(out, tuple):
        output_shape = out[0].shape
    else:
        output_shape = out.shape

    grad_shapes = {}
    if isinstance(out, tuple):
        main_out = out[0]
    else:
        main_out = out
    if main_out.requires_grad:
        main_out.sum().backward()
        for i in grad_inputs_idx:
            if hasattr(prep_inputs[i], 'grad') and prep_inputs[i].grad is not None:
                grad_shapes[f"input_{i}"] = prep_inputs[i].grad.shape

    avg_forward = sum(forward_times) / len(forward_times) * 1000
    avg_backward = sum(backward_times) / len(backward_times) * 1000 if backward_times else 0

    return BenchmarkResult(
        name=name,
        forward_time_ms=avg_forward,
        backward_time_ms=avg_backward,
        total_time_ms=avg_forward + avg_backward,
        memory_allocated_mb=memory_allocated,
        memory_peak_mb=memory_peak,
        output_shape=output_shape,
        grad_shapes=grad_shapes
    )


def compare_outputs(
    out1: torch.Tensor,
    out2: torch.Tensor,
    name: str = "comparison",
    rtol: float = 1e-7,
    atol: float = 1e-7,
    use_relative_only: bool = True
) -> tuple[float, bool]:
    """Compare two tensors and return max diff and match status.

    Args:
        use_relative_only: If True, use only relative error (magnitude-independent)
    """
    if isinstance(out1, tuple):
        out1 = out1[0]
    if isinstance(out2, tuple):
        out2 = out2[0]

    max_diff = (out1 - out2).abs().max().item()

    if use_relative_only:
        # Pure relative error comparison (magnitude-independent)
        # relative_error = |a - b| / |b|
        rel_error = (out1 - out2).abs() / (out2.abs() + 1e-10)  # Add epsilon to avoid division by zero
        max_rel_error = rel_error.max().item()
        matches = max_rel_error < rtol
        return max_rel_error, matches
    else:
        # Standard torch.allclose (combines relative and absolute)
        matches = torch.allclose(out1, out2, rtol=rtol, atol=atol)
        return max_diff, matches


def validate_implementations(
    impl1_fn: Callable,
    impl2_fn: Callable,
    inputs: list,
    grad_inputs_idx: list[int],
    name: str = "validation",
    rtol: float = 1e-7,
    atol: float = 1e-7,
    use_relative_only: bool = True
) -> ValidationResult:
    """Validate that two implementations produce the same results.

    Args:
        impl1_fn: First implementation
        impl2_fn: Second implementation
        inputs: List of input tensors
        grad_inputs_idx: Indices of inputs that need gradients
        name: Name for the validation
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        ValidationResult with comparison details
    """
    # Prepare inputs for both implementations
    def prepare_inputs():
        prepared = []
        for i, inp in enumerate(inputs):
            if i in grad_inputs_idx and isinstance(inp, torch.Tensor):
                prepared.append(inp.detach().clone().requires_grad_(True))
            elif isinstance(inp, torch.Tensor):
                prepared.append(inp.detach().clone())
            else:
                prepared.append(inp)
        return prepared

    inputs1 = prepare_inputs()
    inputs2 = prepare_inputs()

    # Forward pass
    out1 = impl1_fn(*inputs1)
    out2 = impl2_fn(*inputs2)

    forward_max_diff, forward_matches = compare_outputs(out1, out2, rtol=rtol, atol=atol, use_relative_only=use_relative_only)

    # Backward pass
    backward_max_diff = {}
    backward_matches = True

    main_out1 = out1[0] if isinstance(out1, tuple) else out1
    main_out2 = out2[0] if isinstance(out2, tuple) else out2

    if main_out1.requires_grad and main_out2.requires_grad:
        main_out1.sum().backward()
        main_out2.sum().backward()

        for i in grad_inputs_idx:
            if hasattr(inputs1[i], 'grad') and inputs1[i].grad is not None:
                if hasattr(inputs2[i], 'grad') and inputs2[i].grad is not None:
                    if use_relative_only:
                        # Relative error only
                        rel_error = (inputs1[i].grad - inputs2[i].grad).abs() / (inputs2[i].grad.abs() + 1e-10)
                        max_rel = rel_error.max().item()
                        backward_max_diff[f"input_{i}"] = max_rel
                        if max_rel >= rtol:
                            backward_matches = False
                    else:
                        # Standard comparison
                        diff = (inputs1[i].grad - inputs2[i].grad).abs().max().item()
                        backward_max_diff[f"input_{i}"] = diff
                        if not torch.allclose(inputs1[i].grad, inputs2[i].grad, rtol=rtol, atol=atol):
                            backward_matches = False

    return ValidationResult(
        name=name,
        forward_max_diff=forward_max_diff,
        backward_max_diff=backward_max_diff,
        forward_matches=forward_matches,
        backward_matches=backward_matches,
        rtol=rtol,
        atol=atol
    )


def print_benchmark_result(result: BenchmarkResult):
    """Pretty print a benchmark result."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {result.name}")
    print(f"{'='*60}")
    print(f"  Forward time:  {result.forward_time_ms:8.3f} ms")
    print(f"  Backward time: {result.backward_time_ms:8.3f} ms")
    print(f"  Total time:    {result.total_time_ms:8.3f} ms")
    print(f"  Memory (alloc): {result.memory_allocated_mb:8.2f} MB")
    print(f"  Memory (peak):  {result.memory_peak_mb:8.2f} MB")
    print(f"  Output shape:   {result.output_shape}")
    if result.grad_shapes:
        print(f"  Grad shapes:    {result.grad_shapes}")


def print_validation_result(result: ValidationResult):
    """Pretty print a validation result."""
    status = "PASS" if result.forward_matches and result.backward_matches else "FAIL"
    print(f"\n{'='*60}")
    print(f"Validation: {result.name} [{status}]")
    print(f"{'='*60}")
    print(f"  Forward max diff:  {result.forward_max_diff:.2e} (tol: rtol={result.rtol}, atol={result.atol})")
    print(f"  Forward matches:   {result.forward_matches}")
    if result.backward_max_diff:
        print(f"  Backward max diff:")
        for k, v in result.backward_max_diff.items():
            print(f"    {k}: {v:.2e}")
    print(f"  Backward matches:  {result.backward_matches}")


def print_comparison_table(results: list[BenchmarkResult], title: str = "Performance Comparison"):
    """Print a comparison table of benchmark results."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    print(f"{'Implementation':<25} {'Forward (ms)':<15} {'Backward (ms)':<15} {'Peak Mem (MB)':<15}")
    print(f"{'-'*80}")
    for r in results:
        print(f"{r.name:<25} {r.forward_time_ms:<15.3f} {r.backward_time_ms:<15.3f} {r.memory_peak_mb:<15.2f}")
    print(f"{'='*80}")


class KernelValidator:
    """Base class for kernel validation with 3-phase comparison."""

    def __init__(self, name: str):
        self.name = name
        self.results = {}

    def create_inputs(self, batch_size: int, seq_len: int, **kwargs) -> dict:
        """Create input tensors for testing. Override in subclass."""
        raise NotImplementedError

    def pytorch_reference(self, **inputs) -> torch.Tensor:
        """PyTorch reference implementation. Override in subclass."""
        raise NotImplementedError

    def unsloth_kernel(self, **inputs) -> torch.Tensor:
        """Unsloth kernel implementation. Override in subclass."""
        raise NotImplementedError

    def opaque_kernel(self, **inputs) -> torch.Tensor:
        """Opaque kernel implementation. Override in subclass."""
        raise NotImplementedError

    def get_grad_inputs_idx(self) -> list[int]:
        """Return indices of inputs that need gradients. Override in subclass."""
        return [0]

    def run_validation(
        self,
        batch_size: int = 4,
        seq_len: int = 128,
        vmap_batch: int = 4,
        **kwargs
    ) -> dict:
        """Run the full 3-phase validation.

        Phase 1: Unsloth vs PyTorch
        Phase 2: Opaque vs Unsloth
        Phase 3: Opaque vmap vs torch.vmap(PyTorch)
        """
        print(f"\n{'#'*80}")
        print(f"# Validating: {self.name}")
        print(f"# batch_size={batch_size}, seq_len={seq_len}, vmap_batch={vmap_batch}")
        print(f"{'#'*80}")

        # Create inputs
        inputs = self.create_inputs(batch_size, seq_len, **kwargs)
        input_list = list(inputs.values())
        grad_idx = self.get_grad_inputs_idx()

        # Phase 1: Unsloth vs PyTorch
        print("\n--- Phase 1: Unsloth vs PyTorch (Baseline) ---")
        try:
            val1 = validate_implementations(
                lambda *args: self.unsloth_kernel(**dict(zip(inputs.keys(), args))),
                lambda *args: self.pytorch_reference(**dict(zip(inputs.keys(), args))),
                input_list, grad_idx,
                name="Unsloth vs PyTorch"
            )
            print_validation_result(val1)
            self.results["phase1_validation"] = val1

            # Benchmark both
            bench_unsloth = benchmark_forward_backward(
                lambda *args: self.unsloth_kernel(**dict(zip(inputs.keys(), args))),
                input_list, grad_idx, name="Unsloth"
            )
            bench_pytorch = benchmark_forward_backward(
                lambda *args: self.pytorch_reference(**dict(zip(inputs.keys(), args))),
                input_list, grad_idx, name="PyTorch"
            )
            print_comparison_table([bench_unsloth, bench_pytorch], "Phase 1: Unsloth vs PyTorch")
            self.results["phase1_unsloth"] = bench_unsloth
            self.results["phase1_pytorch"] = bench_pytorch
        except Exception as e:
            print(f"Phase 1 failed: {e}")
            self.results["phase1_error"] = str(e)

        # Phase 2: Opaque vs Unsloth
        print("\n--- Phase 2: Opaque vs Unsloth (API Migration) ---")
        try:
            val2 = validate_implementations(
                lambda *args: self.opaque_kernel(**dict(zip(inputs.keys(), args))),
                lambda *args: self.unsloth_kernel(**dict(zip(inputs.keys(), args))),
                input_list, grad_idx,
                name="Opaque vs Unsloth"
            )
            print_validation_result(val2)
            self.results["phase2_validation"] = val2

            bench_opaque = benchmark_forward_backward(
                lambda *args: self.opaque_kernel(**dict(zip(inputs.keys(), args))),
                input_list, grad_idx, name="Opaque"
            )
            print_comparison_table([bench_opaque, bench_unsloth], "Phase 2: Opaque vs Unsloth")
            self.results["phase2_opaque"] = bench_opaque
        except Exception as e:
            print(f"Phase 2 failed: {e}")
            self.results["phase2_error"] = str(e)

        # Phase 3: Opaque vmap vs torch.vmap(PyTorch)
        print("\n--- Phase 3: Opaque vmap vs torch.vmap(PyTorch) ---")
        try:
            # Create vmap inputs (add batch dimension)
            vmap_inputs = self.create_inputs(batch_size, seq_len, vmap_batch=vmap_batch, **kwargs)
            vmap_input_list = list(vmap_inputs.values())

            # Define vmapped functions
            def opaque_vmapped(*args):
                fn = lambda *a: self.opaque_kernel(**dict(zip(vmap_inputs.keys(), a)))
                return torch.vmap(fn, in_dims=self.get_vmap_in_dims())(*args)

            def pytorch_vmapped(*args):
                fn = lambda *a: self.pytorch_reference(**dict(zip(vmap_inputs.keys(), a)))
                return torch.vmap(fn, in_dims=self.get_vmap_in_dims())(*args)

            val3 = validate_implementations(
                opaque_vmapped,
                pytorch_vmapped,
                vmap_input_list, grad_idx,
                name="Opaque vmap vs torch.vmap(PyTorch)"
            )
            print_validation_result(val3)
            self.results["phase3_validation"] = val3

            bench_opaque_vmap = benchmark_forward_backward(
                opaque_vmapped, vmap_input_list, grad_idx, name="Opaque vmap"
            )
            bench_pytorch_vmap = benchmark_forward_backward(
                pytorch_vmapped, vmap_input_list, grad_idx, name="PyTorch vmap"
            )
            print_comparison_table([bench_opaque_vmap, bench_pytorch_vmap], "Phase 3: vmap Comparison")
            self.results["phase3_opaque_vmap"] = bench_opaque_vmap
            self.results["phase3_pytorch_vmap"] = bench_pytorch_vmap
        except Exception as e:
            print(f"Phase 3 failed: {e}")
            import traceback
            traceback.print_exc()
            self.results["phase3_error"] = str(e)

        return self.results

    def get_vmap_in_dims(self) -> tuple:
        """Return in_dims for vmap. Override in subclass if needed."""
        return (0,) * len(self.create_inputs(1, 1))
