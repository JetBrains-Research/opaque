# Probe-derived per-layer rank allocations (Qwen2.5-Coder-7B / KStack, ε=3)

Computed from the **noised** core spectra of `util3-probe-eps3` (LoRA-XSe, 60
steps, 196 layers) via `examples/compute_rank_allocation.py` logic, at the
uniform-r=16 budget (Σr_ℓ² = 196·256).

- `probe_alloc_ainf_r16.*` — Rényi α=∞ (stable rank, noise-robust). Ranks 9–23.
- `probe_alloc_a1_r16.*`   — Rényi α=1 (Shannon, noise-naive). Ranks 3–23.

`.b64` files are ready to pass inline:
`--lora-xs-rank-pattern-b64 "$(cat docs/alloc/probe_alloc_ainf_r16.b64)"`

NOTE: runs using these predate the `rank_pattern_fix_scaling` fix (submodule
8bb1d4e) and are therefore confounded by per-layer scaling drift — see
`../renyi-campaign-results.md`. Re-run with image ≥ f297d89 for a valid test.
