# Opaque benchmarks

This directory owns the measurements behind Opaque's published performance and
resource claims. The harness uses native Python/PyTorch operations and a native
Rust executable for accounting FFTs; it does not scrape prose, replay stored
numbers, or shell out to `--help` as a proxy for work.

## Commands

```bash
uv run python -m benchmarks list
uv run python -m benchmarks run optimizers.state --preset reference --device cpu
uv run python -m benchmarks validate
uv run python -m benchmarks render
uv run python -m benchmarks check
```

`run` writes a JSON artifact containing the full command, case configuration,
Git state, hardware, Python/Rust/package versions, raw timing samples, summary
statistics, correctness errors, and SHA-256 hashes for the benchmark and
implementation sources. Use `--set key=JSON` for an explicit configuration
override and `--output` to preserve multiple configurations of one case.

## Evidence policy

- `benchmarks/claims.toml` records whether an audited claim is supported,
  contradicted, corrected, derived, illustrative, under investigation, or
  withdrawn.
- A supported benchmark claim must name a registered case and have committed
  result data. Exact numbers live in generated tables, not duplicated prose.
- `docs/benchmarks.md` is generated from the claim ledger and result JSON. The
  repository check fails if it is hand-edited or stale.
- Result validation fails when any declared implementation source or dependency
  lock changes. Source declarations may be files or whole directories.
- CUDA-only claims remain withdrawn until the H200 workflow produces an artifact
  for the exact case. A result from another device cannot stand in for it.

## Measurement semantics

Wall time uses `perf_counter_ns` with device synchronization before and after
each sample. Reports retain every sample and use the median as the displayed
statistic. CUDA memory is peak allocated memory. MPS exposes no resettable
allocated-memory peak, so the harness samples driver allocation every 0.5 ms and
labels the result accordingly. CPU cases do not pretend to provide allocator
peak memory.

The native FFT case compares the production RealFFT implementation with a
matched full-complex RustFFT implementation; both cache their FFT planners. The
optimizer case counts unique tensor storage and excludes Python object overhead.
Correctness metrics accompany alternative implementations so a faster but
different result cannot silently become evidence.