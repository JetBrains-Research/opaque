# Liger-style kernels: porting plan for Opaque

This document is a **roadmap** for adapting ideas and Triton approaches from
[Liger Kernel](https://github.com/linkedin/Liger-Kernel) (BSD-2-Clause) into
Opaque, with **vmap(`grad(...)`)** correctness and **benchmark-gated** auto-patching.

It is **not** a commitment to copy Liger verbatim: reference implementation,
numerics, and block sizes can be reused where license-compliant; autograd
packaging must follow Opaque patterns.

## Goals

- Extend **fused kernels** in `opaque.patches`
  where they **win** on wall time **or** peak memory.
- Extend **model support** in lockstep:
  - `opaque.transformers` — vmap-safe compat
    (`patches/`, causal mask, attention, cache).
  - `opaque.patches.transformers` —
    optional Triton patches (env-gated; see `OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES`).
- **Default auto-patch only** if a kernel passes acceptance (below) on **this
  machine’s** reference GPU (H100-class here): forward, backward, and
  **vmap(`grad`) backward**.

Non-goals for default patching: operations that only help standard Trainer
loops without vmap; those may still ship as **manual** APIs under
`opaque.patches.kernels`.

## Licensing

- Liger Kernel: **BSD-2-Clause** (Copyright LinkedIn).
- Opaque: **Apache-2.0**.

Substantially derived files or blocks must **retain BSD-2 copyright and
license notices** (file header, `NOTICE`, or `third_party/` — follow repo
conventions once defined). Clean-room implementations that do not copy
non-trivial expressive code remain Opaque-contributed under Apache-2.0.

Do not imply LinkedIn endorsement of Opaque. **This is not legal advice.**

## Technical constraints

### Autograd API

Opaque uses the **new-style** `torch.autograd.Function` with `setup_context`,
not legacy `forward(ctx, …)`. Required for **functorch / vmap** integration.

Overhead per op is real: prefer **larger fused ops** (e.g. residual + RMSNorm)
so kernel time dominates.

### Kernel pattern

Match existing ops: main `Opaque_*` class plus `_Opaque*Backward` (or
equivalent) with `vmap` support where the main backward dispatches. See:

- `packages/opaque-patches/src/opaque/patches/kernels/swiglu.py`
- `packages/opaque-patches/src/opaque/patches/kernels/rope_embedding.py`
- `packages/opaque-patches/src/opaque/patches/kernels/lora.py`

AGENTS.md notes incompatibility with `@torch.amp.custom_fwd` / `custom_bwd`.

### HuggingFace split

| Layer | Package | Role |
|-------|---------|------|
| Correctness under vmap | `opaque.transformers` | Patches in `patches/_shared.py`, `_standard_models.py`, `_gemma2.py`, `_phi3.py`, … |
| Fused kernels | `opaque.patches` | `opaque.patches.transformers` wires Triton into Transformers classes |

New architectures (e.g. VL) need **both** tracks before claiming support in
[`docs/user-guide/huggingface.md`](../user-guide/huggingface.md).

### HuggingFace version pin

Parity in this roadmap is evaluated against `transformers==4.57.1` (locked in
`uv.lock`; see root `pyproject.toml` and `packages/opaque-transformers/pyproject.toml`).

## Auto-patch acceptance criteria

Before enabling **default** patching for a new op, record results in this doc or
a linked benchmark log (table or committed JSON optional).

### Must pass

1. **Correctness**: HF-faithful numerics within agreed tolerance; kernel + HF
   smoke tests pass.
2. **vmap**: `vmap(grad(...))` tests pass (extend
  `packages/opaque-patches/tests/kernels/` and HF tests as needed).
3. **Performance or memory**: On **at least two** shape buckets from the H100
   matrix below:
   - **≥10%** wall-time improvement **or** **≥10%** peak CUDA memory reduction
     vs baseline (PyTorch eager or current Opaque path), for **forward**,
     **backward**, and **vmap(`grad`) backward**; and
   - **≤5%** regression on the remaining buckets, unless explicitly waived
     (document why).

If an op wins without vmap but fails the vmap path, ship **kernels only** or
**opt-in** patch (document env flag).

### Baselines

- **Norms / activations**: HF module forward + `torch.autograd.grad` vs
  `Opaque_*`.
- **Loss**: Materialized logits + CE vs fused linear CE (where applicable).

## H100 benchmarking protocol

Hardware here: **NVIDIA H100** (or note exact SKU / driver / PyTorch / CUDA in
the log).

### Environment

```bash
# Example: sync dev deps + Triton (adjust to your workflow)
uv sync --group dev --all-packages --extra all

# Optional: isolate clock / power for repeatable numbers (machine-specific)
# sudo nvidia-smi -pm 1
```

### Shape matrix (minimum)

Use **bf16** as primary; add fp16 if a kernel is half-specific.

| Bucket | `batch` or vmap `N` | `seq` | `hidden` | Notes |
|--------|---------------------|-------|----------|--------|
| Small | 1–4 | 512 | 1024–2048 | Autograd overhead sensitive |
| Medium | 8 | 2048 | 4096 | Typical finetune |
| Large | 1–2 | 8192+ | 4096–8192 | Memory stress |

For **vmap**, set microbatch `N` (e.g. 4–16) matching DP-SGD use cases.

### Metrics

- Wall time: median of ≥10 runs after warmup (`torch.cuda.synchronize()`).
- Memory: `torch.cuda.max_memory_allocated()` reset per configuration, or
  equivalent profiler.

### Implementation options

1. **Ad-hoc scripts** under `benchmarks/` (create if needed; keep optional /
   non-CI unless stabilized).
2. **pytest** `@pytest.mark.slow` + `cuda` microbench modules mirroring CI
  patterns in `AGENTS.md`.

Record: PyTorch version, Triton version, GPU name, and command line in the
benchmark log.

## Phased roadmap

### Phase 0 — Foundations

- Finalize shape matrix and pass/fail thresholds for this repo.
- Add or extend a small benchmark harness (script or slow tests).
- License / attribution convention for any Liger-derived Triton (header text,
  NOTICE).

**Exit:** Harness runs on H100; no new default auto-patches required.

### Phase 1 — RMSNorm and fused Add+RMSNorm

**Rationale:** Largest gap vs Liger for decoder-only LMs; fuses multiple small
ops and reduces torch→Triton launch overhead.

**Work:**

- New modules under `opaque.patches.kernels` (names TBD: e.g.
  `rms_norm.py`, `fused_add_rms_norm.py`).
- `Opaque_*` + backward split with `setup_context` and vmap tests.
- Wire into `opaque.patches.transformers`
  for families already in vmap standard patches (LLaMA, Mistral, Qwen2/3,
  Phi-3, Gemma, Gemma2, Granite, Cohere).
- Gemma/Gemma2 casting and offset behavior must match HF and existing
  `packages/opaque-patches/src/opaque/patches/transformers/models/gemma2.py` semantics.

**Exit:** Default patch only if acceptance criteria pass; extend
`OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES` with granular tokens (e.g.
`rmsnorm`, `fused_add_rms`).

### Phase 2 — Cross-entropy / LM-head parity

**Rationale:** Liger CE supports label smoothing, z-loss, aux metrics; Opaque
already has fused linear CE with Gemma2 softcapping and Cohere/Granite scaling.

**Work:** Port **recipe-driven** knobs only if H100 benchmarks show net benefit
under vmap; avoid feature creep without measurements.

**Status (2026-04-30):** Implemented and benchmark-gated for the current
`opaque-patches` test geometry.

- Added fused linear+CE **label smoothing** support in
  `packages/opaque-patches/src/opaque/patches/kernels/linear_cross_entropy.py`
  (forward + backward + vmap pathways).
- Updated HF fused ForCausalLM wiring in
  `opaque.patches.transformers`
  so non-zero `label_smoothing` can stay on the fused path when other loss
  knobs are compatible.
- Added coverage:
  - kernel-level gating tests:
    `packages/opaque-patches/tests/transformers/components/test_fused_ce_gating.py`
  - kernel-level smoothing parity:
    `packages/opaque-patches/tests/kernels/test_linear_cross_entropy.py`
  - HF fused-forward integration for `label_smoothing > 0`:
    `packages/opaque-patches/tests/transformers/components/test_fused_ce_label_smoothing.py`

Current perf baseline remains in the same envelope as pre-change runs for
`label_smoothing=0` (differences are run-to-run noise); enabling
`label_smoothing=0.1` showed negligible overhead in local spot checks.

### Phase 3 — RoPE and extended **text** model families

**Rationale (current):** expand **decoder-only text** coverage (compat +
kernels) where the contract matches existing primitives (half-split RoPE,
SwiGLU, RMSNorm, fused CE). VL / M-RoPE remains **out of scope** for the
default matrix until explicitly scheduled.

**Work:**

- `opaque.transformers`: vmap-safe masks and eager attention for each new text
  family added to the patch list.
- `opaque.patches`: register the same kernel hooks on the family’s
  `modeling_*` module paths.
- Update [`docs/user-guide/huggingface.md`](../user-guide/huggingface.md) table
  when a family lands.

**Coverage (incremental):** Module-level `apply_rotary_pos_emb` Triton swap now
includes **Cohere** and **Cohere2** (same half-split contract as Llama). Tests:
`packages/opaque-patches/tests/transformers/components/test_rope.py`.

**Text-first scope (product default):** kernel and compat expansion targets
**decoder-only text** models (``ForCausalLM`` and shared text modules). Example
addition: **OLMo2** — SwiGLU / RMSNorm / RoPE / fused CE wiring in
`opaque.patches.transformers`
plus vmap eager-attention patches in `opaque.transformers`. OLMo2 uses a
different residual layout than Llama (no ``LlamaDecoderLayer``-style fused
add+RMS block); fused post-attention kernels stay **off** until a dedicated
factory exists.

### Phase 4 — MoE and wide FFN

**Rationale:** Liger `fused_moe`, tiled MLP — high complexity; vmap through
routing is non-trivial.

**Work:** Schedule only if product commits to MoE model families; treat
auto-patch as **opt-in** until gated.

### Phase 5 — Chunked alignment / preference losses

**Rationale:** Large memory wins for DPO/ORPO/… in standard training; vmap
story is harder (chunk loops, ref model).

**Work:** Prefer **manual** `chunked_*` API first; default HF patch only if
vmap acceptance is met or scope is explicitly non-vmap.

## Documentation and tests to update per phase

- [`docs/user-guide/huggingface.md`](../user-guide/huggingface.md) — kernel /
  model table.
- `packages/opaque-patches/tests/kernels/` —
  vmap + numerical tests.
- `packages/opaque-patches/tests/transformers/` —
  integration / smoke.
- `AGENTS.md` — only if new tooling or markers are added.

## Implementation checklist (per kernel)

- [ ] **Triton:** forward + backward kernels; dtype / alignment / block sizes
      tuned on H100.
- [ ] **Autograd:** `Opaque_*`, `_Backward`, `setup_context`; no
      `custom_fwd`/`custom_bwd`.
- [ ] **Tests:** correctness, fallback (CPU / no Triton), `vmap(grad(...))`.
- [ ] **Benchmarks:** forward / backward / vmap-backward vs baseline; memory.
- [ ] **Patching:** HF class list + skip env; doc + changelog entry for merges
      to `main`.
- [ ] **License:** attribution for BSD-2 derived material.

## Next step

Run **Phase 0** harness on this H100, then implement **Phase 1** behind
benchmarks; iterate until default patch eligibility is demonstrated or the
kernel ships as opt-in only.
