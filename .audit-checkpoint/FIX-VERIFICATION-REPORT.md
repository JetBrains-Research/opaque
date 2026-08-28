# Fix verification report — Opaque remediation

**Range:** `79c916e3..4b13d82` (202 commits) · **Scope:** the 95 closed audit issues
**Method:** 12 subsystem verifiers (142 issue checks, several issues checked by two or three of them) plus 4 adversarial sweeps over cross-cutting surfaces (horizon prefixes, Monte-Carlo bounds, canary pools, MPS/backend). Verifiers read the delta commits, read the current tree, and — wherever the environment allowed — executed the code: full package suites, `cargo test --lib`, live 2- and 3-rank Gloo runs, numerical re-derivations against the primary literature, and hand-built repros.

---

## 1. Headline verdict

The remediation is real work, not paperwork. Sixty-two percent of the closed issues are fixed and independently reproduced; almost every remaining one has a genuine fix underneath a residual. Only two issues were closed with nothing in the tree at all.

### Merged per-issue status (95 issues)

| Status | Count | Meaning |
| --- | ---: | --- |
| fixed-verified | 59 | The titled defect is gone in the current tree and a verifier reproduced the fixed behavior (execution, numerical check, or built artifact) — not just read the diff. |
| fixed-with-caveats | 30 | The titled defect is genuinely fixed, but verification turned up a residual: a sibling code path with the same bug, a stale doc, a test that does not cover the fixed branch, or a second-order consequence the fix introduced. |
| partial | 3 | Part of what the issue asked for landed; a named, material half did not. |
| not-fixed | 2 | No commit in the window changes the behavior the issue titles. |
| cannot-verify | 1 | The fix is outside the checking agent's installable environment; another agent covered it (see §4). |

### Raw per-check status (142 checks)

100 fixed-verified, 34 fixed-with-caveats, 4 partial, 3 not-fixed, 1 cannot-verify. The merged table is worse than the raw one because 13 issues were graded differently by two agents and the worse grade wins (§2 records both views).

### What verification cost, in defects

Fifty distinct remediation-era defects were confirmed adversarially (57 reports, 7 of which are the same defect found twice from different areas). By adjudicated severity: **1 high, 12 medium, 37 low**. Forty-eight are CONFIRMED (reproduced or traced to exact lines); two are PLAUSIBLE (code-traced but not runtime-reproduced — the asymmetric-PLD mass discrepancy and the Triton-only fused-activation in-place write).

Nine of the fifty touch epsilon:

1. b-min-sep Monte Carlo epsilon varies with the host's Rayon thread count at a fixed seed (`packages/opaque-accounting/src/amplification/b_min_sep/mc.rs:174-215`).
2. `lr_schedule` is `compare=False`, so the accountant's structural merge collapses MF processes with different schedules (demonstrated: true epsilons 3.52 and 3.68 merged into one).
3. Per-group adaptive-clip DDP sync permutes group counts across ranks, so thresholds silently diverge (`packages/opaque-dpsgd/src/opaque/api/dpsgd/clipping/_distributed.py:64-66`).
4. `AdaClip` accounting has no `__post_init__`; `num_groups=0` prices adaptive clipping as free.
5. lambda-CGD's noise stream is still keyed by bare integer folds and can collide with a caller's key space.
6. Loss-scale backoff/growth is an unaccounted data-dependent adaptive choice that the precision docs claim is fully accounted.
7. Stop-at-epsilon still fires only at logging boundaries for the Monte-Carlo accountant modes.
8. `powu(count as u32)` truncation on the budgeted circular FFT path (Rust-API-reachable only).
9. Asymmetric PLD beta still renormalizes negative-infinity mass away (~1e-15 under defaults).

### What to do first

1. **`packages/opaque-patches/.../runtime/masking.py:77-82, 226-227`** — the only high-severity finding: SDPA with `attention_mask=None` silently drops the sliding-window constraint (0.39 logit deviation measured against upstream). Sliding-window models train on the wrong attention pattern with no error.
2. **Per-group clipping under DDP** — `#368`'s presence check plus the empty-batch short-circuit's hardcoded `clipping_rate=0.0` means the first partially-empty Poisson round raises `RuntimeError` on every rank. Per-group DP-SGD + DDP + Poisson is currently unrunnable.
3. **The three epsilon-relevant runtime defects** — per-group adaptive-clip sync order, lambda-CGD RNG rooting, and the schedule-blind MF merge. All three are small, local fixes.
4. **b-min-sep MC determinism** — apply `#331`'s fixed-shard pattern verbatim to the sibling driver in the same crate.
5. **The two CI fail-open gates** — the dropped CUDA-availability assert and `allow-empty-test-selection` on the distributed lane. Both make a *blocking* lane capable of validating nothing.

---

## 2. Full status table (worst first)

Issue numbers are the closed audit issues. Where two verifiers disagreed, the worse status is shown and both views are named in the note.

| # | Title | Status | Note |
| --- | --- | --- | --- |
| 338 | Reconcile negative-infinity mass in asymmetric PLD metrics | **not-fixed** | Both accounting agents agree: `pld/metrics.rs` is byte-identical in the window; symmetric `pmf_beta` still floors with `negative_infinity_mass` (:212-216) while `pmf_beta_asymmetric` renormalizes it away (:305, :314). |
| 373 | Reject GDP grids that erase detectable leakage | **not-fixed** | `_MIN_GRID_SIZE = 16` and the `gdp()` floor are byte-identical at both ends of the range. Measured: m=1000/u=350 gives eps=2.30 at grid_size=10000 and eps=0.0 at 16 and 64. |
| 362 | Apply LR schedules on the MF step axis and expose weighted Grams | **cannot-verify** | accounting-py could not execute the torch-side fix; accounting-rs verified the Rust half (`*_gram_matrix_lr`) as fixed-verified; dpftrl verified the runtime numerically as fixed-with-caveats (the `compare=False` merge collapse). Net: fixed, but no single agent saw both halves. |
| 335 | Add nightly privacy-regression vectors across mechanisms | **partial** | Deterministic epsilon vectors at rel 1e-9 do run unmarked in every PR lane (packaging-ci calls that stronger than nightly), but **no scheduled workflow exists anywhere** and MC mechanisms have no vectors — tracked by open #666. Both accounting agents say partial. |
| 400 | Apply or reject router dropout and batchify consistently | **partial** | `6e26cc97` (#493) made the unregistered-family path fail closed including via `compat=True`; `2b309916` (#495, titled as a CI fix) weakened it one day later to explicit kwargs only. `DPTrainer` passes neither, so the default trainer path silently skips both again. |
| 416 | Execute documentation examples and validate factory signatures | **partial** | Notebooks re-executed with committed outputs; but nothing in CI executes docs, the factory-signature contract tests added by #480 were deleted by #587, and the exact OPQ-141/OPQ-144 examples the issue named are still broken. |
| 328 | Make distributed DP correctness blocking and schema-driven | fixed-with-caveats | Umbrella of #366/#368; both halves landed and ran green on live 2-rank Gloo. engine-dist says caveats (inherits both children's residuals); transformers and packaging-ci say verified. |
| 332 | Key PLD caches by mechanism and discretization fingerprints | fixed-with-caveats | Caches key on the resolved `DiscretizationConfig` + structural fingerprint + `n_steps` and are schedule-aware; accounting-rs says verified, accounting-py and dpftrl keep the `compare=False` merge residual. |
| 334 | Serialize and compare deep process compositions iteratively | fixed-with-caveats | Explicit-stack eq/hash/repr/codec verified at depth 3000 (accounting-rs: depth 10000, verified). Residual: the trainer's `json.dump(opaque_state_dict(accountant))` still recurses and dies past ~1000. |
| 336 | Normalize Poisson and parallel-Poisson validation and cross-checks | fixed-with-caveats | Validation moved into `__post_init__` on both; cross-checks are behavioral now. dpsgd keeps two caveats (accounting rejects `sample_rate=1.0` the sampler still supports; sibling `AdaClip` never got the pattern); both accounting agents say verified. |
| 350 | Define privacy accounting for loss-scaler skipped steps | fixed-with-caveats | The skip is gone and every attempted step composes (test asserts `_step_counter==2` across a finite and an overflowed step). Caveat: the scale state machine itself is an unaccounted data-dependent channel the docs claim is covered. |
| 351 | Avoid materializing discarded per-example gradients in adaptive clipping | fixed-with-caveats | `return_stats=not inner_return_aux`, `_force_grad_norms` gone, stats-only microbatch accumulator added; engine-core says verified. dpsgd found the stats path stores per-group counts in a different order than the sync reduces them. |
| 352 | Detect non-finite values through clipped-gradient wrappers | fixed-with-caveats | `all_finite` now recurses through `ClippedPytree`/`NoisedPytree`/`SecondMomentClippingOutput`, measured pre-sanitization. Siblings `unscale_grads` and `global_norm` are still wrapper-blind (silent no-op / silent 0.0). |
| 360 | Apply the full BISR strategy operator at runtime | fixed-with-caveats | Runtime operator now equals the exact banded-Toeplitz `C^{-1}` (max err 0.0 at n=12, p=3). Caveat: it does so with dense length-n coefficients — 2.04 GB and 4.27 s/step at n=512, d=1e6 — while the docs promise p-1 vectors via PRNG replay. |
| 365 | Preserve integer and float64 exactness in scalar reductions | fixed-with-caveats | Int64 collectives, `atol=0/rtol=0` float64 compares, seeds compared as strings; live 2-rank exactness test passes. Caveat: the optimizer-state drift audit computes float64 stats but marshals them through the float32 default with rtol=1e-5. |
| 366 | Add a blocking two-rank CPU distributed correctness lane | fixed-with-caveats | Lane exists, is a hard member of the required gate, and its 67 tests pass locally. engine-dist keeps two caveats (empty-selection fail-open; the shipped per-group empty-batch test uses a hand-built float aux that misses the real crash); transformers and packaging-ci say verified. |
| 368 | Derive distributed synchronization from schemas, not rank-local values | fixed-with-caveats | Aux sync is genuinely schema-driven with a fixed collective schedule, validated before any collective. Caveat: under per-group clipping the new presence check itself fires and raises on every rank on a partially-empty Poisson round (reproduced live). |
| 380 | Validate alignment losses with locality and independent parity tests | fixed-with-caveats | NaN-poison row-locality over the full public `__all__` plus real-TRL parity; 79 passed / 1 xfailed. Both agents agree the caveat is that the suite *found* the WPO divergence and xfailed it instead of fixing it. |
| 387 | Identify assistant spans correctly in Gemma chat templates | fixed-with-caveats | The silent whole-template-wrap fallback is gone; every strategy must pass a generation-block probe or raise. Real Gemma-2 and Qwen2.5 verified correct; Gemma-3, Llama-3-Instruct and canonical ChatML fail closed. Caveat: comments overstate coverage and the marker guards are exact-string. |
| 392 | Enforce stop-at-epsilon on every accounted step | fixed-with-caveats | Per-step enforcement is real for deterministic accountants (`predict_stop_step` + post-composition check), cadence-independence tested. Caveats: MC modes are excluded and keep the pre-fix log-boundary check; `k_out_of_t`'s flat prefix makes the predicted stop step k0+1. |
| 393 | Implement private SquareChiPO or keep it explicitly non-private | fixed-with-caveats | The non-private head was deleted and unknown `loss_type` fails at construction. Three docs pages still advertise it as implemented. |
| 397 | Preserve sliding-window attention semantics | fixed-with-caveats | Dense mask path matches HF's boundary exactly (bit-identical logits vs upstream, eager and SDPA, when a mask is passed). Residual is the **high-severity** SDPA `attention_mask=None` window drop; no shipped test covers that branch. |
| 398 | Preserve Gemma2 attention logit softcapping | fixed-with-caveats | Softcap applied in upstream's exact order; SDPA routes softcapped calls to eager so the 50.0 default is never dropped; non-vacuous component test. Caveat: the shim is installed into the process-global `ALL_ATTENTION_FUNCTIONS`, rerouting SDPA for every family. |
| 401 | Avoid mutating saved activations in fused LoRA backward | fixed-with-caveats | X-storage clobbering removed at all four sites with a parallel-residual regression test. Caveat (PLAUSIBLE, Triton path not runnable here): the fused activation backward still overwrites saved gate/up buffers in place. |
| 404 | Compare patched model forwards and gradients with upstream | fixed-with-caveats | A real 492-line parity suite landed (forward, backward, vmap(grad), dtype tolerances). Caveats: the harness restores only class forwards, so module-global rebinds contaminate the "upstream" reference after the first test per family; and the sliding-window variant is vacuous. |
| 405 | Replace vacuous optimizer regressions with behavior-sensitive tests | fixed-with-caveats | Closed-form oracles pin exact updates and corrected/uncorrected gains; round-trips compare at rtol=atol=0 with a meta-assertion that polices vacuity. Caveat: the same vacuity pattern survives in `test_adadelta.py:306-338`, and adadelta kept the dead `bc_floor` the commit removed from adagrad. |
| 406 | Align Adafactor and Adadelta memory behavior with docs | fixed-with-caveats | Adadelta BC state is allocated only when BC is on and the false 1.5x claim is gone; unverifiable Adafactor benchmark claims removed. optimizers keeps the caveat that the Adafactor docstring still misstates rank<2 behavior and the BC default; packaging-ci's lighter check says verified. |
| 408 | Provide the documented schedule-free evaluation accessor | fixed-with-caveats | Resolved doc-side: the phantom `get_eval_params()` is gone repo-wide and docs teach `opt_state.x`, with a Polyak-average test. Caveat: `DPTrainer` accepts `optim='schedule_free'` and never reads `.x`, so trainer eval and checkpoints use the wrong iterate. |
| 412 | Make contributor, workflow, and export-contract docs accurate | fixed-with-caveats | OPQ-140 fixed and WORKFLOWS.md rewritten against reality. Caveat: the docs re-drifted *inside* the window — stale advisory-minimum claims and a stale distributed-test path. |
| 418 | Audit and repair scholarly references | fixed-with-caveats | Eight OPQ citation defects verified repaired at every site including Rust. Caveat: OPQ-136's exact defect survives — `_schedule_free.py:10` still credits "Defazio, Yaida, Cutkosky". No mechanical citation check exists. |
| 419 | Align transformer reference tables and defaults with dataclass fields | fixed-with-caveats | `chunked_nll` and the per-class `use_performance_kernels` defaults are correct in `docs/reference/transformers.md`. Both agents keep the same caveat: OPQ-142's second cited site, `docs/alignment/trainers.md:109-110`, still lists only nll/dft. |
| 422 | Apply workspace Rust lints to opaque-accounting | fixed-with-caveats | Crate-level `[lints.rust] unsafe_code = "warn"` closes the dead-config gap and `cargo clippy -D warnings` escalates it; accounting-rs and packaging-ci say verified. accounting-py keeps the caveat that it duplicates the lint instead of `[lints] workspace = true`, so future workspace lints still reach zero members. |
| 424 | Make Triton kernel validation blocking | fixed-with-caveats | `continue-on-error` removed and CUDA lanes are in the required gate; patches says verified from config. packaging-ci found that #600's rewrite dropped #476's "Assert CUDA available" preflight, so a broken GPU runner auto-skips every cuda test and the blocking lane goes green. |
| 426 | Verify noise-stream continuity across checkpoint resume | fixed-with-caveats | Bit-identical uninterrupted-vs-resumed draws through the real save/load payload plus a negative counter-reset check; transformers says verified. dpftrl independently verified all 7 state shapes by hand but notes CI covers 2, and the trainer resume test still asserts only step count and epsilon. |
| 428 | Test the supported Python 3.12 runtime | fixed-with-caveats | Regressed inside the window: #600 added 3.12 lanes, #665 retargeted them to 3.13. `grep '3.12'` over `.github/` now returns nothing while every wheel still advertises it. |
| 431 | Install TRL and enforce zero-noise parity in CI | fixed-with-caveats | TRL is in `[all]`, CI installs `--extra all`, and 132 parity + 14 trainer numeric-parity tests pass against real trl 1.12.0; alignment and packaging-ci say verified. transformers keeps the caveat that WPO parity is a strict xfail covering a genuine divergence. |
| 331 | Make balls-in-bins Monte Carlo reproducible across thread counts | fixed-verified | Fixed 1024-sample shards with shard-index seeds, `par_chunks_mut`, bitwise-equal PMFs at 1 vs 8 threads (Rust + subprocess Python tests). Both agents flag the untouched sibling driver separately. |
| 333 | Derive calibration search direction from the calibrated parameter | fixed-verified | Direction probed from both bracket ends, convergence accepted only from the proven-safe endpoint, fail-closed on flat/NaN/non-bracketing. All four quadrants executed live; 8/8 tests pass. |
| 337 | Return conservative privacy guarantees from Monte Carlo amplification | fixed-verified | MC PLDs are simultaneous one-sided confidence bands; the Chernoff-KL construction, union bookkeeping, ceil bucketing and residual-to-infinity-mass were re-derived from first principles and cross-checked against exact Beta-inverse and deterministic oracles. |
| 339 | Remove latent sorted-sensitivity assumptions from MixtureConstants | fixed-verified | max/min/sampling_prob are order-independent folds; #592 pins deltas for an explicitly unsorted non-unit sensitivity vector against direct quadrature at 1e-10 in the blocking cargo lane. |
| 340 | Make the linear-FFT safety fallback effective for zero-tail PMFs | fixed-verified | Circular fallback is now gated on a positive tail budget; a zero-budget oversized composition returns `SelfCompositionTooLarge` instead of aliasing. Both branches tested. |
| 341 | Reject invalid and overflowing PLD composition counts | fixed-verified | Count validated in [1, u32::MAX] at the Python boundary; overflow-safe `pow_usize`. 0/-3 → ValueError, 2^32/2^63 → OverflowError, verified live. (One Rust-API-only residual, §3.) |
| 342 | Serialize every implementation of the public Budget protocol | fixed-verified | Registry codec with a collision-guarded extension point; fail-closed in both directions. All five builtins JSON round-trip with type+value equality. |
| 343 | Preserve clipping numerics across microbatching and bfloat16 | fixed-verified | fp32/fp64 promoted norm reductions, a computed roundoff guard that makes `norm(output) <= C` hold on stored values, and bit-exact microbatch-vs-full parity in bf16 (max abs diff 0.0) for both streams. |
| 347 | Make PerGroup state deeply immutable | fixed-verified | `MappingProxyType` over private copies; caller-dict mutation does not leak, writes raise TypeError. Verified independently by dpftrl and engine-core. |
| 349 | Reject shape-mismatched tensor checkpoint values | fixed-verified | Exact-shape equality (broadcast-compatible mismatches rejected too), `nn.Parameter` subclass preserved by its own handler, unrecognized leaves fail closed. |
| 354 | Tighten Poisson sampler statistical regression tests | fixed-verified | Exact scipy binomial bands with a 1e-9 false-failure budget, a tightness precondition before containment, Bonferroni-split per-record inclusion, and an exhaustiveness test at q=1.0. |
| 355 | Compute MF row norms without eager quadratic work | fixed-verified | Closed forms per family, validated against generic probing and materialized norms (max err 4.4e-16); band_mf bands=16 n=5000 row norms in 0.4 ms. |
| 357 | Recreate DP-FTRL samplers across configured epochs | fixed-verified | One sampler spans the stream in trainer and examples; every sampler keeps a `_consumed` cursor so `__iter__` resumes rather than restarting. |
| 358 | Preserve empty balls-in-bins slots in the runtime sampler | fixed-verified | Empty bins are yielded unconditionally for all `n_steps` slots and the trainer executes them as pure-noise steps, matching the Lemma 3.2 accounting. |
| 361 | Reuse column-keyed Gaussian draws across MF noise steps | fixed-verified | Column keys derived from a namespaced fold; measured cross-step correlations 0.708 and 0.593 against theory 0.707 and 0.577 (pre-fix ~0). Covariance tests exist for both engines. |
| 363 | Do not retry collectives after callback TypeError | fixed-verified | The retry is gone, field callables are invoked exactly once, arity validated before any collective; the 2-rank repro asserting exactly one reduction passes. |
| 364 | Shard reference-logprob precomputation across ranks | fixed-verified | Contiguous shards re-concatenated in rank order through one gather; size/fingerprint asserted up front and the cache-hit decision min-reduced unconditionally. 3-rank Gloo test passes. |
| 367 | Reduce paired private-second-moment gradients across ranks | fixed-verified | Explicit recursion into both children of the paired wrappers, non-tensor leaves now raise, and the previously missing `SecondMomentMFNoiseState` sync handler registered. Single-vs-two-rank parity verified live. |
| 369 | Gather uneven distributed evaluation shards safely | fixed-verified | One symmetric `all_gather_object` per call with `None` payloads still participating, post-collective structural validation, four unconditional gathers in the accumulator. Empty and uneven shard tests pass on 2 ranks. |
| 370 | Make perfect-separation ROC tests non-vacuous | fixed-verified | The `if len(idx) > 0` escape is gone; the corner is asserted to exist exactly once plus exact full-array equality. |
| 371 | Make score-to-membership ordering explicit and verifiable | fixed-verified | `CanaryScores` identifiers are mandatory; bare arrays raise; joins by identifier reject unexpected/duplicate/missing ids. Tutorial and all four examples migrated. |
| 372 | Validate sorted batch_argnums | fixed-verified | Empty, negative, duplicate, unsorted and out-of-range all rejected, from both scorers. |
| 374 | Domain-separate auditing and mechanism RNG streams | fixed-verified | Selection and coin streams carry distinct string domains and #728 namespaced every mechanism root; the pre-fix collision (coin flips == DP-SGD step-1 key) is impossible for Opaque's own tags. |
| 375 | Export gradient_scores consistently from the auditing facade | fixed-verified | Facade export works and is documented. Note: the `__all__`-parity contract test added by the fix was later deleted per ARC-007's review-based enforcement model. |
| 377 | Provide explicit canary construction for coin-flip audits | fixed-verified | Both acceptable remedies landed: README/doc wording corrected, and `coin_flip(candidate_indices=)` with full validation and a fixed-pool requirement that preserves the one-run independence assumption. |
| 378 | Compute raw ROC metrics with correct infinite-score denominators | fixed-verified | Raw ROC by default with terminal counts pinned to true totals, hull opt-in, `tpr_at_given_fpr` guarded; raw AUC unbiased under the null while hull is provably biased high. |
| 379 | Include reference-model identity in DPO cache fingerprints | fixed-verified | Required `cache_identity`, versioned canonical-JSON SHA-256 (v2 invalidates pre-fix caches), trainer supplies a full state-dict digest including adapter mode. |
| 381 | Preserve policy adapters when a PEFT reference model is supplied | fixed-verified | Any non-None ref model is treated as separate; `ref_model is model` rejected before mutation. Regression tests run against real PEFT LoRA models. |
| 382 | Honor fractional num_train_epochs | fixed-verified | `ceil(epochs * steps_per_epoch)` flows to sampler, calibration, accounting and scheduler alike; tests pin step count, epsilon at the fractional horizon and the 0<f<1 case. |
| 384 | Evaluate SFT DFT runs with the training objective | fixed-verified | `prediction_step` routes dft through the vmapped per-example closure in every mode including `prediction_loss_only=True`; eval loss compared against an independent DFT reference. |
| 385 | Test the actual Hugging Face evaluation cadence | fixed-verified | The unused pure helper is deleted; cadence comes only from HF's real `DefaultFlowCallback`, with 204 lines of end-to-end cadence tests including resume. |
| 386 | Align best-model selection with save and evaluation boundaries | fixed-verified | Improving evals force a save, the callback is auto-injected for `load_best_model_at_end`, best checkpoint registered by folder lookup so rotation cannot delete it, alignment validated in `__post_init__`. |
| 388 | Wire or remove inert privacy and logging arguments | fixed-verified | `privacy_noise_radius` removed tree-wide (only the raising regression test remains); `log_level`/`log_level_replica` wired with HF parity and validation. |
| 389 | Translate or reject fused Hugging Face optimizer settings | fixed-verified | `adamw_torch_fused` → `adamw` + RuntimeWarning and no `optim_args`; apex/8-bit/paged still raise; the TRL config path shares the converter. (One stale doc line, §3.) |
| 390 | Match fused and eager token-count dtypes | fixed-verified | Exact integer counts via bool-cast sums and ≥fp32 accumulation; fused paths divide by the same exact shifted-label count. Verified against upstream on the CPU chunked kernel. |
| 391 | Secure or disable the TR-DPO reference-logprob temp cache | fixed-verified | TR-DPO seed pass uses `use_cache=False` and per-step logps stay in memory; the general cache is chmod 0700/0600 on read and write with fail-closed `PermissionError`. |
| 394 | Correct BCO, MPO, f-divergence, and LD-DPO loss contracts | fixed-verified | Each verified against executing TRL 1.12.0 including gradients; LD-DPO now splits on completion-relative positions and matches TRL's `ld_alpha` pipeline end to end. |
| 395 | Clamp DiscoPOP inside the active dtype range | fixed-verified | Dtype-aware exponent clamp (11.0 fp16, 80.0 otherwise) with fp16/bf16/fp32 value+gradient finiteness sweeps and NaN-locality checks. |
| 396 | Scope checkpoint differentiation overrides to Opaque transforms | fixed-verified | The process-wide `create_graph=False` rebind is conditioned on captured grad mode; the master-list repro now returns [6,12,18] on torch 2.13 where it previously returned zeros. Gating regimes unit-tested. |
| 399 | Validate LoRA bias, projection shape, and vmap dimensions | fixed-verified | `_validate_vmap_dims` called in all 9 custom vmap rules with correct batched-index sets; the Lite-backward dX reshape bug fixed; MLP fusion gated on no-bias. |
| 402 | Stop cross-entropy backward from overwriting model logits | fixed-verified | Both the forward and vmap backward write into `logits.clone()`, with a regression test; sibling kernels scanned and clean apart from the #401 gate/up caveat. |
| 403 | Correct unsupported grouped-GEMM claims in fused MoE docs | fixed-verified | The three false capability claims are gone from both kernels and the model-patches guide; replacement text describes the real 2D×2D offset-grouped layout. Doc-only. |
| 409 | Give update_rms_clip one model-wide or per-leaf meaning | fixed-verified | Everything routes through one param-count-weighted global-RMS helper, including Adafactor (with the chain-level clip disabled to avoid double clipping); docs say model-wide explicitly. |
| 410 | Restore per-group noise-variance EMA from checkpoints | fixed-verified | `phi` keyed by optree `ParamPath` round-trips through real on-disk save/load with distinct per-path values (4.49e-4 vs 1.25e-3) for adamw, adafactor and adadelta; tests made non-vacuous. |
| 411 | Make Adafactor epsilon floors scale-aware | fixed-verified | All three `eps_root` sites are scale-relative; update norms identical to 6 decimals across gradient scales 1.0 → 1e-12. |
| 413 | Express Gaussian per-step privacy in terms of the noise multiplier | fixed-verified | `dp-concepts.md` now defines σ = σ_abs/C as the controlling quantity and reattributes advanced composition to Dwork-Rothblum-Vadhan 2010. |
| 414 | Document the current alignment reference-cache format | fixed-verified | The safetensors cache, its fingerprint tuple, `use_cache=False` for TR-DPO, owner-only permissions and the node-local default are documented and match the implementation. |
| 415 | Correct adaptive clipping distributed-use documentation | fixed-verified | Docs and docstring now state local-only semantics and the required `sync(clip_state)`; the documented call shape was executed against the real API. (The per-group *code* defect is separate, §3.) |
| 420 | Pin umbrella-package dependencies during builds | fixed-verified | Sentinel rewriting with fail-closed validation plus a METADATA pin checker in both build jobs; validate-distributions installs `opaque[all]` from built wheels only. |
| 421 | Produce portable and reproducible accounting artifacts | fixed-verified | Bytecode exclusion declared in both packaging layers and enforced by a blocking post-build policy script; a wheel built on this dirty host contains zero `__pycache__`/`.pyc` entries. |
| 423 | Ship licenses and correct third-party attribution paths | fixed-verified | All NOTICE paths resolve, per-package NOTICE + LICENSE symlink; wheels built locally contain `dist-info/licenses/{LICENSE,NOTICE}`. |
| 425 | Cover workspace manifests and lockfiles with dependency updates | fixed-verified | `Cargo.lock` committed; dependabot covers github-actions, cargo and uv (lockfile-only by design). One point unverifiable without the GitHub API (§4). |
| 430 | Align opaque-transformers metadata, extras, and Triton policy | fixed-verified | README no longer advertises nonexistent extras or `opaque-core`; the sole extra is `trl`; the Triton comment states the intentional platform-gated transitive dependency. |
| 432 | Name domain-significant numeric comparisons | fixed-verified | PLR2004 selected for src and ignored only under tests; ruff clean, so remaining magic values are named or justified. |
| 433 | Reduce public API width and oversized implementations | fixed-verified | PLR0913/0915/0917 selected with calibrated limits and six Triton-kernel exemptions; ruff clean. The enforcement mechanism is real even though the semantic reduction is judgment. |
| 435 | Add Google-style docstrings to public surfaces | fixed-verified | D100/D101/D103/D104/D417 with google convention, enforced by the autoformat lint gate; `ruff check packages/ tests/` exits clean. |
| 437 | Replace or justify PLD method caches | fixed-verified | Per-method caches replaced by a weak-identity cache keyed on config + structural key + n_steps, with a byte-budgeted native handle registry cleared in `calibrate`'s finally. No `functools` cache remains. |
| 438 | Annotate public facades and callable factories | fixed-verified | Public factories carry typed callable/state returns; ruff gate clean on `packages/`. |
| 442 | MC random-allocation accounting for the DP-SGD Gaussian prefix | fixed-verified | Implemented **deterministically** rather than by Monte Carlo: full-epoch self-composition plus an exact partial-epoch prefix transform, every rounding direction traced conservative, prefix epsilon monotone over K=1..640, one-sided cross-validation against the paper authors' package. Verified by three agents. |

---

## 3. Caveats and regressions in detail

Fifty confirmed defects, grouped by what they cost. Each is tied to the closed issue whose verification surfaced it.

### 3.1 Privacy-relevant

**b-min-sep MC transcripts are sharded by live Rayon thread count** (medium, CONFIRMED) — `packages/opaque-accounting/src/amplification/b_min_sep/mc.rs:174-215, 255-270`. `bandmf_b_min_sep_prepare_transcripts` and `..._pld_from_transcripts` partition `num_samples` by `rayon::current_num_threads()` and seed each chunk `seed + tid` / `seed + 1000 + tid`. Reproduced: `RAYON_NUM_THREADS` 1/4/8 give three different deltas at a fixed seed. This is exactly the pattern #331 fixed for balls-in-bins in the same crate (fixed 1024-sample shards, `par_chunks_mut`); the sibling driver was never converted. Attaches to #331.

**Schedule-blind strategy equality collapses distinct-LR MF processes** (medium, CONFIRMED) — `_band_mf.py:110`, `_bisr.py:160`, `_blt.py:132`, `_lambda_cgd.py:119`. All four strategies declare `lr_schedule: Schedule | None = field(default=None, compare=False)`, so `MfGaussian.__eq__/__hash__` are schedule-blind while the schedule genuinely changes coefficients and priced sensitivity since #529. `DpProcess.__or__`'s same-leaf merge (`core/_base.py:474-477`) then folds two different mechanisms into `Repeated(first, 2)` — demonstrated with true epsilons 3.52 and 3.68. The #332 fix keyed the *caches* correctly; the *equality* half was left. Attaches to #332 and #362.

**Per-group adaptive clip sync permutes group counts across ranks** (medium, CONFIRMED) — `packages/opaque-dpsgd/src/opaque/api/dpsgd/clipping/_distributed.py:64-66`. `sync_adaptive_clip_state` reduces per-group counts positionally by iterating `state._num_clipped.items()`, but the dict's key order is code-path dependent: the non-empty stats path (the `return_aux=False` default introduced by the #351 fix) stores counts in leaf-traversal group order, the empty-batch path in `PerGroup.values` construction order. Confirmed in a live 2-rank Gloo repro: counts land on the wrong groups and thresholds diverge across ranks. Attaches to #351.

**AdaClip accounting accepts invalid parameters on direct construction** (low, CONFIRMED) — `packages/opaque-dpsgd/src/opaque/api/accounting/dpsgd/mechanisms/_adaclip.py:23-55`. #336/#516 moved validation into `__post_init__` for Poisson, ParallelPoisson, RandomAllocation and KOutOfT so neither direct construction nor deserialization can carry invalid state into the native call. The sibling `AdaClip` dataclass has no `__post_init__` at all: `num_groups=0` prices adaptive clipping as free, negative `sigma_b` is accepted. Attaches to #336.

**lambda-CGD noise stream is not RNG-namespaced** (medium, CONFIRMED) — `packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/_lambda_cgd.py:280, 292`. #728 gave every noise mechanism a namespaced string root (`MF_GAUSSIAN_STREAM_FOLD='opaque.dpftrl.mf_gaussian'`, `GAUSSIAN_STREAM_FOLD` in dpsgd); lambda-CGD's PRNG-replay path still derives per-step generators from bare integer folds `rng_fold_in(st._rng_key, step)`. `docs/reference/rng.md:208-212` states the convention exists precisely because "`fold_in(key, step)` is what every mechanism writes first". Collision reproduced. Attaches to #374/#728.

**Loss-scale backoff/growth is an unaccounted data-dependent adaptive choice** (medium, CONFIRMED) — `packages/opaque-engine/src/opaque/api/engine/precision/_loss_scaler.py:185-195`. The #350 fix removed the data-dependent *skip*, but the replacement feeds the raw un-noised private statistic `stats.all_finite` into `LossScaler.update`, so the scale at step t+1 is a function of private data at step t that was never released or accounted. The scale determines which per-example gradients overflow and are zeroed. `docs/user-guide/precision.md:99` claims the loop is fully composed into the accountant. Attaches to #350.

**Stop-at-epsilon still only fires at log boundaries for MC accountant modes** (low, CONFIRMED) — `_dp_trainer.py:1175, 3970`. The #392 fix excludes `b_min_sep` and `balls_in_bins` from the per-step predicted stop, and those are the canonical samplers for every `mf_*` mechanism except `mf_identity`. For them the only stop mechanism remains the fallback inside `if ctrl.should_log:` — the exact pre-fix behavior, which never fires with logging disabled. Attaches to #392.

**self-convolution exponent truncated to u32 on the budgeted circular path** (low, CONFIRMED) — `packages/opaque-accounting/src/numerics/fft.rs:316-318`. `*c = c.powu(count as u32)` wraps modulo 2^32. The linear strategy always rejects such counts, but when tail budgets are set the circular fallback is taken and the spectrum is raised to the wrapped power, silently composing `count mod 2^32` steps. Reachable only through the Rust crate's own API — the Python boundary caps at u32::MAX (#341). Attaches to #341.

**Asymmetric PLD metrics still renormalize negative-infinity mass away** (low, PLAUSIBLE) — `packages/opaque-accounting/src/pld/metrics.rs:305, 314` vs `:212-216`. See §4; magnitude is ~1e-15 under default tail budgets, and MC PLDs fail closed, which is presumably why the issue was closed.

### 3.2 Distributed correctness and crashes

**Per-group clipping crashes DDP on any partially-empty Poisson round** (medium, CONFIRMED) — `_clipped_grad.py:238`, with `_clipped_fun.py:852-856`, `engine/clipping/_distributed.py:224-230`, `dpsgd/clipping/_adaptive.py:423`. The #368 fix made the aux sync schema-driven and added a presence check — but left the aux *schema itself* data-dependent under per-group clipping: the empty-batch short-circuit hardcodes `clipping_rate=0.0` while the non-empty per-group path emits `clipping_rate=None`. One empty rank plus one non-empty rank raises `RuntimeError` on every rank. Reproduced live. The shipped test `_worker_sync_aux_empty_vs_per_group` hand-builds a float 0.5 on the non-empty rank and therefore misses it. Attaches to #368 and #366.

**Empty-batch short-circuit ignores `dtype=` and `pre_clipping_transform` structure** (low, CONFIRMED) — `_clipped_grad.py:216-256`. `zero_grads_like` copies the parameter pytree's structure and dtype, while non-empty steps return the configured output dtype and whatever structure the transform produces. Step-shape stability under Poisson sampling breaks. Attaches to #368.

**Optimizer-state drift audit marshals float64 fingerprints through float32 collectives** (low, CONFIRMED) — `packages/opaque-optimizers/src/opaque/api/optimizers/distributed.py:98-103`. `_assert_tensor_fingerprint_equal` computes (sum, sumsq, min, max) in double precision but calls `assert_scalar_equal` without `compute_dtype`, so the collective runs at the float32 default with rtol=1e-5. Cross-rank drift below ~1e-5 relative passes an audit whose stated purpose is detecting exactly that. Attaches to #365.

### 3.3 Verification gaps — tests and CI that can pass while validating nothing

**GPU lanes fail open** (medium, CONFIRMED) — `.github/workflows/python-tests.yml:97-128`. #424's fix added an "Assert CUDA available" preflight; #600's rewrite of the reusable workflow deleted it. `grep 'is_available|Assert CUDA'` over `.github/` returns nothing. All three CUDA lanes — the only place Triton kernels and patched-model parity execute — auto-skip on a runner with broken CUDA and report success. Attaches to #424.

**Distributed lane passes silently if its selection collapses to zero** (low, CONFIRMED) — `.github/workflows/pr.yml:80` with `run_python_test_package.sh:40-43`. The blocking two-rank lane sets `allow-empty-test-selection: true` and the runner converts pytest exit code 5 into success per package. Renaming or unmarking the `distributed` tests turns the desync-regression lane green with zero tests executed. Attaches to #366/#328.

**Parity suite's "upstream" reference is contaminated** (medium, CONFIRMED) — `packages/opaque-patches/tests/transformers/models/_test_utils.py:256-272`. `parity_model_patches` snapshots and restores class-level forwards only; the family apply functions also rebind `mod.eager_attention_forward`, `mod.create_causal_mask`/`create_sliding_window_causal_mask` and `ALL_ATTENTION_FUNCTIONS['sdpa']`, none of which are restored. Verified: `modeling_llama.eager_attention_forward` stays patched after context exit, so the reference is genuinely upstream only for the first test per family per process. Attaches to #404.

**`test_sliding_window_parity` never exercises a binding window** (low, CONFIRMED) — `test_parity.py:295-313` sets `sliding_window=64` while inputs are shape (2, 10), so every position is inside the window and the feature under test contributes nothing. The harness also always passes all-ones masks, which is why it cannot see the high-severity SDPA defect either. Attaches to #404 and #397.

**Noise-stream continuity covers 2 of 7 state shapes** (low, CONFIRMED) — `test_checkpoint_helpers.py:283-300`. The #426 fix tests gaussian and band_mf; BLT, BISR, BSR, lambda-CGD, identity and the paired second-moment state have no continuity regression, and the end-to-end resume test still asserts only `global_step` and epsilon — the exact criticism OPQ-157 made. (The dpftrl verifier confirmed all 7 shapes are in fact bit-exact by hand; the gap is coverage, not behavior.) Attaches to #426.

**`test_mode_switch_does_not_double_correct` promises a comparison it never performs** (low, CONFIRMED) — `packages/opaque-optimizers/tests/test_adadelta.py:306-338`. The docstring says the post-switch update must match a phi-forced-to-zero reference; the only assertion is `torch.isfinite(...).all()`. The #405 vacuity pattern, in a sibling file the fix did not touch. Attaches to #405.

**Ruff enforcement for `tests/` is a no-op** (low, CONFIRMED) — `pyproject.toml:231`. #706 added `tests/` to the ruff commands in `autoformat.yml`, but `[tool.ruff.lint].exclude` still contains `"tests/**"`. Verified empirically: a probe file with an unused import and an F821 undefined name inside `tests/` passes the check.

**Python 3.12 executes zero tests** (low, CONFIRMED) — `.github/workflows/pr.yml:87-111`. #600 added 3.12 lanes; `182b80b1` (#665) retargeted them to 3.13. No test lane, no validate-distributions leg, no release matrix runs 3.12, while every `pyproject` declares the classifier and `requires-python` admits it. A regression inside the remediation window. Attaches to #428.

**`#495` quietly reverted `#493`'s fail-closed contract** (low, CONFIRMED) — `packages/opaque-patches/.../transformers/_router.py:116-135`. Commit `2b309916`, titled "fix(ci): separate Python and Rust test gates", changed the unregistered-family guard to raise only when dropout/batchify are explicitly passed. `DPTrainer` calls `apply_model_patches(model, compat=True)` with neither, so the default trainer path silently skips both. Attaches to #400.

### 3.4 Runtime behavior and design residuals

**Sliding-window constraint silently dropped under SDPA** (**high**, CONFIRMED) — `packages/opaque-patches/.../transformers/runtime/masking.py:77-82, 226-227`. `vmap_create_causal_mask` returns `None` (SDPA `is_causal` skip) whenever `attention_mask is None`, `attn_impl != 'eager'` and no cache is populated, with no sliding-window awareness; `vmap_create_sliding_window_causal_mask` propagates the `None`. SDPA then runs plain causal attention and the look-back limit is discarded. Measured 0.39 logit deviation against upstream. Every unit test added by the #397 fix uses eager. Attaches to #397.

**BISR keeps n_steps-1 gradient-sized buffers** (medium, CONFIRMED) — `_bisr.py:198-211`. The #360 fix made the runtime operator mathematically exact by handing dense length-`n_steps` coefficients to `inverse_as_streaming_matrix`, whose forward-substitution state is `(bands-1, *grad_shape)` with `bands = n_steps`. Measured 2.04 GB and 4.27 s/step at n_steps=512, d=1e6, while `docs/mechanisms/dp-ftrl/bisr.md:10, 96, 109` promises p-1 noise vectors via PRNG replay. Correct, but not the documented design. Attaches to #360.

**`DPTrainer` never consults `ScheduleFreeState.x`** (medium, CONFIRMED) — `trainer/_optim.py:55, 94`. The trainer accepts `optim='schedule_free'` and the HF alias `schedule_free_radam`, but nothing in `opaque-transformers` reads `.x`; it evaluates, checkpoints and saves the y_t forward-pass iterate that the wrapper's own docstring says defeats schedule-free averaging. `docs/reference/transformers.md:233-234` lists the optimizer with no caveat. Attaches to #408.

**`unscale_grads` and `global_norm` are still wrapper-blind** (low, CONFIRMED) — `_loss_scaler.py:172-183` and `opaque.pytree`. #352 was fixed for `all_finite` only; `ClippedPytree`/`NoisedPytree` are unregistered dataclasses, so `unscale_grads` returns the wrapper unchanged without unscaling anything and `global_norm` returns 0.0. Empirically confirmed. Attaches to #352.

**`wpo_weights` omits the WPO Eq.(2) alignment term** (medium, CONFIRMED, found twice) — `packages/opaque-alignment/.../dpo/loss/_wpo.py:33-67`. Opaque computes `exp(mean masked per-token logp)`; the paper (arXiv:2406.11827, Eq. 2) and TRL 1.12 both subtract the per-token `log(sum_v p(v)^2)` term first (verified in the installed TRL source: `lse1 = logsumexp(shift_logits); lse2 = logsumexp(2.0*shift_logits); log_denom = lse2 - 2.0*lse1`). `use_weighting=True` runs therefore optimize a different objective than the cited paper. The #380/#431 parity suite found this and xfailed it. Attaches to #380 and #431.

**Gemma2 SDPA shim installed process-globally** (low, CONFIRMED) — `_family.py:172-176`. `mod.ALL_ATTENTION_FUNCTIONS['sdpa'] = ...` writes the singleton `GeneralInterface`'s `_local_mapping`, so applying Gemma2 family patches reroutes SDPA for every model in the process. Verified: llama's sdpa entry becomes the gemma2 shim. Attaches to #398.

**Fused activation backward mutates saved gate/up activations** (low, PLAUSIBLE) — `kernels/lora.py:757-759` with `swiglu.py:145-160`. #482 removed the dX/X-storage reuse, but `_lora_mlp_backward_impl` still passes reshape views of `ctx.saved_tensors` as both input and output to `act_backward_fused`, so `retain_graph=True` re-backward produces wrong gradients. Code-traced; the Triton path needs CUDA to reproduce. Attaches to #401.

**Sampler RNG streams exempt from the #728 convention** (low, CONFIRMED) — `sampling/_random_allocation.py:125`, `_poisson.py:108`, `_k_out_of_t.py:58`. Samplers seed numpy directly from `key.seed` with no domain root, so two sampler streams can collide byte-identically. Attaches to #374/#728.

**`fold_in`'s int/str domain disjointness is not structural** (low, CONFIRMED) — `engine/random/_engine.py:27-35, 64-67`. `_stable_hash64` encodes an int as its 16-byte signed little-endian form plus `|`, and a str as UTF-8 plus `|`, with no type tag — so a 16-byte string collides with an integer fold. The docstring guarantee the #374/#728 convention rests on is a convention, not a structure. Attaches to #374.

**Accounting rejects `sample_rate=1.0` the sampler still supports** (low, CONFIRMED) — `accounting/dpsgd/amplification/_poisson.py:38-41`. #336 tightened the accounting bound to strictly (0,1); `PoissonSampler` still validates `0 < q <= 1` and `0ac7b008` added a test pinning q=1.0 as a supported exhaustive mode. A full-batch run has a supported sampler and no constructible accountant. Fail-closed, but an asymmetry and a behavior break for old serialized state. Attaches to #336.

**`_mu_at` hang fix is symptom-level** (medium, CONFIRMED) — `one_run/_gdp.py:40, 80-91`. `611faf51` caps bracket doublings at 60 and raises — verified terminating in under a second on the pathological inputs — but the underlying invariant is untouched (truncated ranks use `v_k=0.5` regardless of mu at `:333`, so the Chernoff floor is `n_trunc/2`). Strong attacks with >2000 canary guesses now raise `RuntimeError`, including the default `epsilon_at` surface at the documented 10000-canary scale. Attaches to #299 (verified) / #373 area.

**`AdaClip`/adaptive-clipping thresholds, `bc_floor`, docstrings** (low, CONFIRMED, three findings) — `_adadelta.py:121, 377, 382` keeps the dead `bc_floor` the same cleanup removed from adagrad; `_adafactor.py:6-7, 28, 30-31` misstates the BC default (says True, code default is False), the rank<2 memory behavior, and the phi-EMA floor semantics. Attach to #405 and #406.

### 3.5 Documentation defects that survived documentation issues

Two closed issues promised to eliminate this class and did not.

From **#416** ("Execute documentation examples and validate public factory signatures"): `docs/reference/rng.md:309` still calls `gaussian_noise(1.1, key=noise_key)` although every parameter is keyword-only, and lines 380/401/412/415/419 bind the `(noise_fn, state)` 2-tuple to one name while :307/:332 do it correctly on the same page (OPQ-141, CONFIRMED twice). `docs/user-guide/dp-sgd.md:143-159`'s only complete training loop iterates the `PoissonSampler` directly (it yields index lists, not data), builds the sampler without `n_steps` so iteration never terminates, and never composes the accountant (OPQ-144, CONFIRMED). `docs/user-guide/precision.md:97` binds `optimizer.update`'s result as `(opt_state, params)` — reversed — where 11 other doc sites do it correctly. `docs/user-guide/clipping.md:236-249` and `:558-568` call `dpsgd_acc.poisson` / `.gaussian` / `.adaclip` while importing only `opaque.accounting as acc`; `dpsgd_acc` is never defined on the page.

From **#418** ("Audit and repair scholarly references"): `_schedule_free.py:10` still credits "The Road Less Scheduled" (arXiv:2405.15682) to "Defazio, Yaida, Cutkosky" — the real authors are Defazio, Yang, Mehta, Mishchenko, Khaled, Cutkosky, and Sho Yaida is not among them. This is OPQ-136's exact defect, reported independently by two agents. Separately, `_discopop.py:5-7` attributes DiscoPOP to a nonexistent "Azar, M. G., et al. (2024)" citation (the paper is Lu et al.), and the `sequence_logp` docstring denies the `ld_alpha` feature it implements.

Other stale docs, each tied to a fix that changed behavior inside the window: `docs/user-guide/huggingface/training-arguments.md` still says stop-at-ε "halts at the first logging boundary" and that `logging_steps=0` disables it silently (false for every deterministic accountant after `b7073f37`), and still lists `adamw_torch_fused → adamw + optim_args={"fused": True}` (removed by `6cc02681`). `docs/alignment/trainers.md:18` and `docs/alignment/dpo.md:46` still list the deleted SquareChiPO head; `docs/alignment/trainers.md:109-110` still omits the supported `chunked_nll` SFT loss — OPQ-142's second cited site, reported by three agents. `AGENTS.md:268` and `CONTRIBUTING.md:163` still call the minimum-dependencies lane advisory after `a31629f2` (#723) made every lane blocking, and `.github/WORKFLOWS.md:50-52` describes caller-supplied package matrices and `Build / <distribution>` check names that no longer exist. `packages/opaque-alignment/.../data/_chat_template.py:229, 376` claims coverage ("Handles ChatML, Llama-3, Phi-3", "Gemma 2/3 templates") that the strategies do not have — those templates fail closed with `ValueError`, which is safe but not what the comments say. `packages/opaque-auditing/tests/auditing/test_integration.py:42-44` still names a Bonferroni correction that does not exist anywhere in the code (OPQ-130 residual).

One packaging residual: `packages/opaque-accounting/tests/test_smoke.py:145-161` imports `opaque.dpftrl.accounting` and `opaque.dpftrl.noise` from inside `opaque-accounting`'s wheel-local test directory, while `opaque-accounting` declares only `opaque-base` and `opaque-dpftrl` requires torch — an ARC-006 violation and the single failure in an otherwise-green 96/97 standalone accounting suite. Reported by both accounting agents. And `packages/opaque-accounting/Cargo.toml:39-40` duplicates `unsafe_code = "warn"` locally instead of `[lints] workspace = true`, so the root `[workspace.lints.rust]` table still applies to zero members and future additions will not propagate.

---

## 4. Not fixed, partial, cannot-verify — and what would close them

### #338 — Reconcile negative-infinity mass in asymmetric PLD metrics · **not-fixed**

`git diff 79c916e3..4b13d82 -- src/pld/metrics.rs` shows only the `infinity_mass()` accessor (#466) and test-signature churn; `git log -S` over the file confirms nothing touches the asymmetric path. The symmetric `pmf_beta` seeds `cdf_y[0]` with `negative_infinity_mass` (`:212-216`) to keep beta conservative; `pmf_beta_asymmetric` and `pmf_beta_symmetrized` still drop it through `clean_probs` renormalization, carrying the exact comments the audit quoted (`:297, :306`). Asymmetric PLDs with nonzero negative-infinity mass are reachable (`random_allocation.rs:157-185, 250`; `connect_the_dots.rs:278`).

Both accounting agents independently rate the practical impact tiny: MC PLDs fail closed (`beta_at`/`risk_at` return 0), and for non-MC asymmetric PLDs the reachable mass is bounded by the Chernoff left-tail budget (5e-16 by default), so the discrepancy is ~1e-15. Closure may have been a deliberate withdrawal, which neither agent could confirm without GitHub API access.

**To close it:** either (a) land the one-line floor in `pmf_beta_asymmetric` mirroring the symmetric path, plus the beta/delta-correspondence test on an amplified PLD that the issue asked for; or (b) reopen-and-reclose with an explicit "won't fix, bounded by tail budget" rationale in the issue and a comment at `metrics.rs:305`.

### #373 — Reject GDP grids that erase detectable leakage · **not-fixed**

`_MIN_GRID_SIZE = 16` and the `gdp()` floor at `_estimate.py:37, 176-177` are byte-identical between the range endpoints; `git log -S grid_size 79c916e3..4b13d82 -- packages/opaque-auditing` hits only the #299 hang fix, which does not touch grid handling. No rejection logic, warning, test, or doc caveat was added. Measured this session: at m=1000, u=350 the audit reports eps=2.30 at `grid_size=10000` and **eps=0.0** at 16 and 64 — a coarse grid silently converts detected leakage into "no leakage found", and the docs describe `grid_size` as merely a tunable with no lower-bound guidance.

**To close it:** a data-dependent floor (reject or warn when the grid's resolution is coarse relative to the observed `(r, u)` separation), or at minimum a documented lower bound plus a regression test asserting that the m=1000/u=350 case cannot return 0.0.

### #335 — Nightly privacy-regression vectors · **partial** (packaging-ci dissents: fixed-verified)

What landed: committed deterministic epsilon vectors asserted at rel 1e-9 for dpsgd (poisson, truncated, parallel-poisson, adaclip, random-allocation) and dpftrl amplifier×mechanism pairs, running **unmarked in every PR lane**; plus dp-accounting/riskcal/random-allocation cross-validation in the dev group so those suites are blocking on every leg. packaging-ci reasonably calls per-PR stronger than nightly.

What did not: there is **no scheduled workflow anywhere** in `.github/workflows` (zero `schedule:`/cron triggers — confirmed by all three agents), and there are no vectors for the Monte Carlo mechanisms (balls-in-bins, b-min-sep), which is the part that most needs a long-running lane. That residual is explicitly tracked by open #666, so it was not re-filed as a finding.

**To close it:** either add the cron lane with MC vectors, or amend the issue to say the per-PR deterministic vectors are the accepted delivery and let #666 carry the MC half.

### #400 — Router dropout and batchify · **partial**

Documented in §3.3. `6e26cc97` fixed it; `2b309916` — a commit whose title says it separates Python and Rust test gates — narrowed the guard to explicitly-passed kwargs one day later. The explicit-kwarg path is fixed and tested; the default trainer path is not, and the trainer docstring still claims info-level logging where the code logs at debug.

**To close it:** restore the #493 guard for the `compat=True` default path, and add a test that calls `apply_model_patches(model, compat=True)` on an unregistered family and asserts it raises.

### #416 — Execute documentation examples and validate factory signatures · **partial**

What landed: all tutorial notebooks re-executed with committed outputs (except the torchrun-only distributed one) and the DP-FTRL signature docs corrected.

What did not: (a) nothing in CI executes notebooks or doc examples — mkdocs runs with `execute: false` and there is no nbmake; (b) the factory-signature contract tests #480 added (`tests/contracts/test_mf_strategy_signatures.py`) were deleted by #587 along with the whole contracts suite; (c) the specific broken examples the issue named (OPQ-141 in `rng.md`, OPQ-144 in `dp-sgd.md`) plus `precision.md`'s reversed `optimizer.update` binding are still broken — see §3.5.

**To close it:** an nbmake or `pytest --codeblocks`-style lane over `docs/` would make this mechanically verifiable and would have caught all four surviving examples. Absent that, the issue is a one-shot sweep that has already re-drifted, and the four named sites need fixing regardless.

### #362 — LR schedules on the MF step axis · **cannot-verify** (accounting-rs: fixed-verified; dpftrl: fixed-with-caveats)

Not a defect in the fix — a coverage seam between agents. accounting-py could not install torch and therefore could not execute the half of `0baaba7a` that lives in `packages/opaque-dpftrl`; it verified structurally only (the `bisr_gram_matrix_lr` / `lambda_cgd_gram_matrix_lr` natives exist and are exported, schedules materialize into coefficients and cache keys). accounting-rs verified the Rust builders in depth. dpftrl verified the runtime numerically: band-MF now applies lr as per-row `query_weights` in the optimizer objective rather than the old wrong-axis coefficient product, and `gram_matrix` output demonstrably changes when a schedule is supplied.

Between the three views the fix is verified end to end; only the `compare=False` merge residual (§3.1) is outstanding.

### Also unverifiable in this environment, by construction

Two claims could not be checked at all and should be treated as open questions rather than verified facts: whether Dependabot's cargo updater actually refreshes the root workspace `Cargo.lock` from a member-directory entry (#425 — needs the GitHub API), and the CUDA/Triton kernel behavior behind #424, #390's Triton branch, and the #401 in-place write (needs a GPU). CI gate behavior throughout was verified from workflow configuration, not from executed runs.

---

## 5. Delta coverage beyond the issues

The verifiers did not stop at the closed issues; each read the full commit delta for its area and ran what it could. This is where the confidence in the "fixed-verified" column comes from, and it is also where several of the §3 findings originated.

### Per-area coverage

| Area | Issues | Delta reviewed | Executed |
| --- | ---: | --- | --- |
| accounting-py | 17 | all 31 commits touching the package, including non-fix features #307/#459/#460/#468 and #748 | full package suite in a torch-free venv (96 pass, 1 ARC-006 failure); re-derived and executed the `#290` CachedProcess/repeated_pld fix on five metrics; calibration in all four quadrants; depth-3000 serialization; b-min-sep thread-count repro |
| accounting-rs | 17 | all 30 commits touching the Rust crate | `cargo test --lib` (360 pass, 7 slow-ignored); re-derived the MC confidence band, the Feldman-Shenfeld random-allocation transform, every `Rounding::Up/Down` direction, and the OPQ-180/181/182/183/184/185 originals from the archived master list |
| dpsgd | 5 | all 55 commits touching the package | 334 sampling/clipping/noise + 372 accounting/functional/rng + 5 Gloo tests; two custom repros (AdaClip validation gap, 2-rank per-group sync corruption); confirmed open #344/#345 were not made worse |
| dpftrl | 9 | all 47 commits, plus the dpftrl trainer surface | full 526-test suite; BISR operator exactness and memory measurement; closed-form vs probing row norms; past-horizon raises for all six strategies; serialization continuity for all strategy states; cross-schedule merge-collapse repro |
| engine-core | 6 | 37 commits (distributed-only ones excluded) | 474 engine tests + 93 dpsgd clipping + the loss-scale integration test; bf16 microbatch-vs-full bit parity; wrapper no-op reproductions; empty-batch dtype divergence |
| engine-dist | 8 | every distributed commit across six packages plus the CI lane | the whole `distributed and not cuda` selection (67 tests, green) plus a fresh 2-rank Gloo repro of the per-group presence crash the shipped tests miss |
| optimizers | 7 | all 13 commits touching the package plus the trainer converter | 230 tests; Adafactor scale-invariance sweep 1.0 → 1e-12; on-disk per-group phi round-trips compared against a `79c916e3` worktree |
| alignment | 12 | all 32 commits plus the TRL trainer surfaces | 468 pass / 1 strict xfail; 2-rank shard test; 14 zero-noise trainer parity tests; real Gemma-2/Gemma-3/Llama-3/ChatML/Qwen2.5 template probes |
| transformers | 16 | the full 56-commit delta, including #747, #729, #728, #721, #720, #604, #274, #461, #298, #507/#548, #759 | predict_stop_step, noise continuity, logging wiring, HF compat, ignore_data_skip, TRL parity, row locality — all green; checked that no new un-noised training statistic entered the logging surface |
| patches | 11 | all 30 commits including the non-fix kernel/KV-cache/annotation churn | 303 CPU tests; reproduced the higher-order silent-zeros defect as fixed on torch 2.13; built the wheel to confirm license shipping; numerical sliding-window/softcap probes against transformers 5.16.1 — which is how the high-severity SDPA defect was found |
| auditing | 8 | all 24 commits plus the cross-cutting RNG commit | 222 tests; `_mu_at` termination and the r=2000 boundary; raw-vs-hull AUC bias; GDP grid-size sweep; `fold_in` collision demo |
| packaging-ci | 26 | ~45 `.github/**` commits, all 11 pyprojects, uv.lock, Cargo manifests, NOTICE wiring, and every release script read in full | built opaque-base and opaque-accounting wheels and inspected dist-info; full ruff gate; probed the `tests/` lint exclusion (fail-open confirmed); verified every release tag v0.2.0–v0.15.3 resolves to an ancestor of main |

### Cross-cutting sweeps

Four adversarial sweeps went after surfaces no single issue owns. Each produced a written clean-verdict inventory as well as findings.

**Horizon and prefix accounting.** Re-derived `random_allocation_gaussian_prefix_pld` by hand — the `P_r = (r/t)·P̄_r + ((t-r)/t)·Q^r` decomposition, the neg-dual identity, the add-direction symmetry and the `add_constant(t-r)` terms all match the exact prefix law, and every rounding direction is conservative. Verdict: exact and safe. Epsilon monotonicity in prefix length was measured for RandomAllocation (K=1..20), KOutOfT k=1 (K=1..16), BallsInBins identity (K=1..12) and CyclicPoisson BandMF (K=1..8); the correlated families are constant in K by construction. No non-monotone prefix bound exists, so the early-stopping exploit (report a lower epsilon by stopping at a dip) is closed for every shipped family. `per_step`/`Repeated`/`Cached` wiring was verified by execution, including that composing past a declared horizon raises.

**Monte-Carlo bounds.** Re-derived the Chernoff-Hoeffding order-statistic band powering `#337`, checked it is applied to the right functional and one-sided in the safe direction for balls-in-bins (Choquette-Choo et al. Lemma 3.2 after Gram reduction), verified the union bookkeeping (`failure/(2n)` per rank across both adjacency directions), the `resolved_num_mc_samples` algebra, the fail-closed paths, and the confidence-metadata algebra under composition and self-composition. Numerical oracles: the identity-Gram BnB bound sits above the exact random-allocation PLD (1.477 vs 1.109 at δ=1e-2), the bands=1 b-min-sep bound above the analytic Poisson self-composition. The MC guarantee threading is sound; the sweep's own finding is the b-min-sep thread-count gap.

**Canary pools.** Executed `coin_flip` at the tip with duplicate, negative, out-of-range, float, bool, 2-D and set pools — all fail closed before a partition exists. Verified pool-order irrelevance by running ascending vs descending pools, the domain separation of selection and coin streams, that no code path assumes canaries are a uniform sample of the dataset (validity per Steinke et al. Thm 5.2 holds for any fixed canary set given fair coins), and that the newly cited Dagréou & Bellet paper exists with exactly the claimed title. Content-level duplicate canaries are undetectable by any index-based API but can only deflate epsilon-hat, and the guide covers it.

**MPS and backend paths.** Brute-forced the fp32 squared-norm accumulator that MPS forces: 300 adversarial cases (up to 4M-element bf16 leaves, subnormals, norms within 1e-7 of C) with `norm(output) <= C` holding with zero excess every time, and the guard math shown conservative for any summation order, so inductor reassociation under `torch.compile` cannot break it. Verified DP-SGD noise is device-independent by construction (CPU generator, then move), that the MF generator fallback matches the real ATen error string in the installed torch, that fp16 training is rejected and MPS bf16 admitted only behind an empirical probe, and that no DP randomness can fall back to a global or device RNG. Also confirmed there was no backend-split refactor in the window — the MPS stabilization predates the range.

### What no one covered

CUDA and Triton kernel behavior (no GPU in any session — the cuda-marked tests never ran, only their CPU fallbacks); NCCL variants of the DDP tests; actual CI run execution, as opposed to workflow configuration; notebook and doc-example execution (nothing in CI does it either, per #416); GitHub issue bodies and closure rationales, which were unavailable, so issue-to-fix mapping was reconstructed from commit content and the archived OPQ master list; and trainer end-to-end runs in the dpftrl area, where transformers was not installed. Three of the fifty findings (the fused-activation in-place write, the Dependabot cargo-lockfile question, and the asymmetric-PLD mass discrepancy) rest on code tracing rather than execution and are labeled accordingly.