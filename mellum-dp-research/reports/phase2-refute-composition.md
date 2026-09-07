# Phase 2 — refute-composition: composition, amplification, adaptivity, DP-FTRL, side channels

Agent: `refute-composition`. Target: `phase2-final-design.md` (read in full). Inputs read in full: BRIEF, PHASE1-DIGEST,
`.junie/differential-privacy-review.md`, phase1-math, phase1-primitives, phase1-critic, phase2-judge-dp. Repo
`/home/user/opaque` @ branch `claude/mellum-dp-representation-r6slaz`, nothing tracked modified.

Evidence tags: **VERIFIED** = I read the cited lines or ran the cited script this session; **VERIFIED (phase-1 X / judge-Y)**
= established by that agent's executed script or primary-source extraction, not repeated by me; **PLAUSIBLE** = derived, not
executed. Repo paths relative to `/home/user/opaque`; HF = `.venv/lib/python3.11/site-packages/transformers/`. My scripts and
raw outputs: `scratchpad/research/refute-composition/{t5_recorder_ckpt.py,.out, corr_coins.py,.out, probe_optimizer.py,.out}`
(CPU, each < 60 s). I cite no theorem beyond those phase-1 `critic` R6 extracted from the primary PDFs.

Verdict: **sound-with-fixes**. The core mechanism (one joint per-group Gaussian / MF release on the same sampled batch, accountant
unchanged, lagged use free) survives every attack I could mount, and I turned the design's PLAUSIBLE recorder-under-checkpointing
claim into VERIFIED at toy scale. What fails is narrower: one inherited "as run" gap under DDP + resume that the design's DDP rows
declare closed, three side-channel / test-statement errors in the hygiene table, and three specification slips (per-group
pattern collision, the v1.1 second-moment plan, the v2 noise-key statement).

---

## 1. Refuted claims

### R1 (major, inherited) — "DDP: nothing to add"; "provably DP … as run" does not hold under DDP + resume

**Claim** (§0.2 row (a); §8 DDP column "rank-identical (shared noise key)"; §9.2 "DDP: nothing to add — `sum_gradients_`
all-reduces every leaf …, noise is added per rank with the shared key"). The design's privacy statement (§8) is unconditional.

**Why it fails.** The trainer's own code comment states that after a DDP resume "the sampler snapshot is self-contained
(carries its own key) and is written once on rank 0, so resuming a DDP run currently restores rank 0's per-rank key on every
rank, re-introducing the cross-rank correlation after the resume point … the per-record marginal stays Bernoulli(q) either way,
so the privacy accounting is unaffected" (`packages/opaque-transformers/src/opaque/api/transformers/trainer/_dp_trainer.py:3750-3757`,
VERIFIED). The restore path installs the saved state with no rank re-fold (`:1786-1800`: `ctx.current_sampler =
from_state_dict(ctx.current_sampler, saved_sampler_state)`, VERIFIED); the rank fold exists only on fresh construction (`:3803`,
VERIFIED); shards are contiguous per rank (`opaque-engine/.../distributed/_shard.py:23-35`, VERIFIED). Hence after a resume,
record `x` on rank 1 and record `y` on rank 0 that share a local shard index have **identical inclusion coins at every
remaining step** (Poisson and b-min-sep alike — both draw from the restored stream key).

The "marginal stays Bernoulli(q), accounting unaffected" argument is wrong for the *tight* subsampled PLD the accountant runs.
The Poisson-amplification pair the accountant evaluates (Feldman–Shenfeld Lemma 3.2 / ZDW Thm 11, VERIFIED critic R6;
`src/amplification/poisson.rs:15-42` exact-Gaussian path, VERIFIED phase-1 primitives) is `(P, (1−q)P + qQ)`: the "x out"
component must equal `M(D)`, which requires x's coin to be independent of the other records' coins. With a shared coin between
x and y the add/remove pair becomes `((1−q)A + qB₀, (1−q)A + qB₁)` with `A = M(D | y out) ≠ B₀ = M(D | y in)`. I computed the
hockey-stick divergence of both pairs numerically in 1-D (`corr_coins.py`, VERIFIED; sensitivity 1, y's aligned contribution
`u ∈ {0, 0.5, 1}`):

| σ, q | ε | δ standard (independent coins) | δ correlated, u = 0.5 | δ correlated, u = 1 |
|---|---|---|---|---|
| 0.5622, 5.12e-4 (preset regime, one step) | 1 | 1.85e-8 | 1.17e-5 | 1.01e-4 |
| 0.5622, 5.12e-4 | 3 | 3.16e-11 | 1.02e-6 | 2.76e-5 |
| 1.0, 0.01 | 2 | 1.73e-12 | 3.66e-7 | 2.36e-5 |

At `u = 0` (no correlation, or an uncorrelated y) the two coincide exactly, confirming the computation; with a shared coin the
per-step δ at ε = 3 is **six orders of magnitude** above what `poisson(gaussian(0.5622), q)` charges. DP is a worst case over
datasets, so y's aligned contribution is admissible. The joint (gradient + probe) release does not create this problem — the
sampler does — but the design *inherits* it in exactly the configuration its §8/§9.2 rows call closed, and its privacy
statement does not exclude it. The same holds for the pre-existing gradient release, i.e. this is not specific to the new leaf.

**Fix (concrete).** (a) At the resume site (`_dp_trainer.py:1799-1800`) re-derive the per-rank stream key after restoring the
cursor — e.g. restore the cursor/position but set the stream key to `fold_in(restored_key, self._ddp.rank)` (the same
derivation `:3803` uses on fresh construction), or write per-rank sampler snapshots; add a 2-rank `distributed` test asserting
the two ranks' post-resume inclusion masks are not identical. (b) Until then, the design's privacy statement must read
"single-process, or DDP without checkpoint resume", and §9.2's "DDP: nothing to add" must cite the trainer caveat. Severity:
major for the "as run" claim, limited to DDP + resume; pre-existing.

### R2 (minor) — §8 row 1: "per-example aux value … never leaves `clipped_grad`; this design adds nothing to the un-noised loss mean"

**Why it fails.** The per-example loss the design differentiates is `ℓ_x = CE_x + α·E·Σ_e (f̃_e − k/E)·P_e(x)` (§1.1), and T10
explicitly requires "loss value equals `CE + α·surrogate` bit-for-bit". `grad_and_value` returns that value,
`ClippedGradAux.loss_values` carries it, and the trainer logs `aux.loss_values.mean()` un-noised
(`_dp_trainer.py:2249`, VERIFIED). So with the feature on, the logged train loss becomes
`mean CE + α·E·⟨f̃_t − k/E, P̄(B_t)⟩` — a new, un-noised, un-accounted function of the batch's mean router probabilities. It is
α-scaled (≤ 1e-3·64·‖f̃ − k/E‖_∞) and sits next to the pre-existing un-noised CE mean (F11), but the table's "adds nothing" and
"never leaves `clipped_grad`" are false, and T17 as written ("no logged key is a function of un-noised `h`/`f(B)`") would fail
on the `loss` key (it is a function of `P(x)`, the differentiable twin of `h`).

**Fix.** Make the surrogate value-neutral: `loss = CE_x + α·(S − S.detach()) + ⟨z, λ d(x).detach()⟩`. The gradient is
unchanged (`∇(S − S.detach()) = ∇S`), the reported `loss_values` is exactly `CE_x`, and the pre-existing posture is not widened.
Rewrite T10 to "loss value equals `CE_x` bit-for-bit; gradient equals `∇CE + α∇S`". Alternatively keep the value and list the
term explicitly in the privacy statement under the F11 caveat.

### R3 (minor) — §8 row 3: "the probe group is skipped in [the `group_metrics`] loop" closes the probe-norm channel

**Why it fails.** Two other logged keys carry the probe's un-noised norm: `metrics["grad_norm"] = aux.grad_norms.mean()`
(`_dp_trainer.py:2251`) and `metrics["clipped_grad_norm"] = aux.clipped_grad_norms.mean()` (`:2257`), both VERIFIED.
`aux.grad_norms` is the per-example **total** norm over *all* leaves (`_clipped_fun.py:673` `"norms": norm.norm`,
`_pytree.py:473` `orig_norm = sqrt(_sq_norm(leaves, …))` over the whole pytree, VERIFIED), i.e. `√(‖g_x‖² + ‖λ d(x)‖²)`.
Skipping the probe in the per-group loop does not remove it from these aggregates. The information content is negligible
(≤ (λΔ_h)²/2 ≈ 1.6e-4 in quadrature at ρ = 0.02), but the table's classification "excluded" is incomplete and T17 would fail
on `grad_norm` if implemented literally.

**Fix.** With the feature on, compute the logged `grad_norm` / `clipped_grad_norm` from the non-probe `group_norms`
(`√Σ_{g≠probe} group_norms_g²`), and scope T17 to assert exactly that. (`clip_rate` is already recomputed from the surviving
groups, `:2283`, VERIFIED — fine.)

### R4 (minor, implementability) — §9.2 "a dict `clipping_norm` gets the `router_load_probe` key"; §4.7 `per_group(trainable, gate=C_r, experts=C_e, …)`

**Why it fails.** `per_group` matches by *substring* of the dotted path and raises `ConfigurationError` when a leaf matches two
or more patterns (`opaque-engine/.../clipping/_per_group.py:143-158`, VERIFIED). Any user pattern that is a substring of
`router_load_probe` collides with the probe leaf at setup — and `"router"` is exactly the pattern a user training Mellum's
`mlp.router.weight` (§4.7/§7 scope) would write. The design's own §4.7 example collides independently: `experts.gate_up_proj`
matches both `gate` and `experts`. Hard error, no leak.

**Fix.** Build the user `PerGroup` over `trainable` *without* the probe, then add the probe by direct construction
(`PerGroup(groups={**pg.groups, (probe_path,): "router_load_probe"}, values={**pg.values, "router_load_probe": C_h})`);
in §4.7 use non-overlapping patterns (`mlp.router`, `experts.`, `self_attn`). Add a test that a user `{"router": …}` group
coexists with the probe.

### R5 (minor) — T10: "`α = 0` / `off` runs are bit-identical to today's path"

**Why it fails.** Only `off` is bit-identical. With the feature on and α = 0 (`monitor`), the probe leaf (i) inflates σ_g by
√(1+ρ) by construction and (ii) changes the noise realisation of every gradient leaf, because both noise engines draw the
leaves sequentially from one step generator (`opaque-dpsgd/.../noise/_gaussian.py:419` step key then per-leaf draws;
`opaque-dpftrl/.../noise/_engine.py:412-418` likewise, VERIFIED; `probe_optimizer.out` shows the probe draw is distinct from
the gradient draw). Reproducibility statement only; privacy unaffected.

**Fix.** T10: "`off` runs are bit-identical; `monitor` runs differ from `off` only by the √(1+ρ) σ inflation and the noise
stream". Document in the mechanism page.

### R6 (minor, v1.1 spec) — §9.2 second-moment exclusion "pass a `0.0` second-moment bound for the probe group so `paired_noise_stddevs` allocates σ² = 0"

**Why it fails.** (a) `per_group` rejects non-positive values ("Per-group value must be positive", `_per_group.py:127-129`,
VERIFIED), so the 0.0 bound cannot be built through the public factory. (b) More importantly, a zero *noise* allocation on a
stream whose zeroing is a separate step is a latent un-noised release: if the zeroing is ever skipped or reordered, the squared
probe stream `(λ d(x))²` (per-expert usage squared, summed over the batch) is emitted with σ = 0. Sensitivity 0 is only
legitimate if the *clip* enforces it.

**Fix.** Keep v1's `ConfigurationError`. For v1.1, add a per-group `return_zero`-style option to the engine's clip so the
probe's squared stream is structurally zero inside the vmapped clip (then the 0 bound is justified by the clip, not by a
post-hoc zeroing), and test T20 on the *clipped* squared stream.

### R7 (minor, v2 spec) — §2.7 condition (i): the second sampler key is rank-domain-separated; the second noise "from a separately rooted key"

**Why it is incomplete.** Under DDP, noise is added on every rank *after* `reduce_pytree_` with a key that must be identical
across ranks (the gradient noise key is shared, `_dp_trainer.py:1478`, VERIFIED) — otherwise ranks diverge in `f̃`. §2.7 states
the rank fold for the sampler key but not the *no*-rank-fold requirement for the second noise key; an implementer following
"separately rooted" + "rank-domain-separated" literally could rank-fold both.

**Fix.** State: sampler key `fold_in(key, "opaque.moe.load_release", rank)`; noise key `fold_in(key,
"opaque.moe.load_release.noise")` shared across ranks, added after the all-reduce; T24 asserts rank-identical `f̃`.

---

## 2. Upheld — what I tried and could not refute

U1. **Same-batch joint release is one Gaussian at `nm` under Poisson; accountant unchanged.** Per-group allocation
`σ_i = nm·√(B_i ΣB_j)` satisfies `Σ B_i²/σ_i² = 1/nm²` with equality (VERIFIED judge-dp `check.py`; I re-derived); whitening is
a bijection; both releases share the sampling coin, so Feldman–Shenfeld Lemma 3.2 applies once to the joint whitened Gaussian.
`_build_mechanism` consults `num_groups` only for adaclip (design VERIFIED; adaptive mode is rejected). Truncated Poisson keeps
a Gaussian inner (`dpsgd/amplification/_poisson.py:95-104`, VERIFIED phase-1) — unchanged by the extra group.

U2. **Adaptivity: `f̃_t` is a function of previous outputs and public constants only.** `_augment_inputs` runs before the grad
fn (`_dp_trainer.py:2080`); the callback reads the *noised* pytree at `:2199` after `:2182` and before `opt.update` `:2210`
(VERIFIED); no eval/log/save hook touches the state. The per-record bounds (`C_g` by clipping, `C_h` structural) hold for every
prefix, so ZDW Thm 10 / Denisov Thm 2.1 (adaptive rows) apply, including the `monitor_then_surrogate` switch and the `stop`
decision (`D_t` public). Shrinkage uses `s_t` from public `nm`, `λ`, `φ_t`.

U3. **Empty Poisson batch.** `clipped_grad` short-circuits with zero grads carrying the `PerGroup` max_norm
(`_clipped_grad.py:220-277`, VERIFIED); the trainer still noises, fires the callback and steps the optimizer, skipping only the
metrics (`:2182-2214`, `:2234`, VERIFIED). The probe release is pure noise, a valid output; the MF noise column index stays
contiguous with the sampler's step index (no skipped column ⇒ `min_sep` preserved).

U4. **DP-FTRL per-group MF.** Whitened Frobenius bound with a participation pattern shared across groups (same example, same
step) gives `sens(C)²/nm²` — the scalar `mf_gaussian_pld(nm, sens)`. Per-leaf iid noise is drawn independently per leaf
(`_engine.py:87, 412-418`, VERIFIED), the constant two-group `PerGroup` passes the latch (VERIFIED phase-1 primitives E2),
realised σ = base·‖row_t(C⁻¹)‖ (`_mf_gaussian_noise.py:186-192`). The probe's rows are near-identical across an example's
participations — that is the worst case the MF sensitivity already assumes, not a violation.

U5. **b-min-sep expected batch.** I attacked "E|B_t| ≈ B̄ constant" via the first `b` steps (all examples eligible ⇒ 3.3 %
larger); the sampler uses the paper's warm start (Alg. 2 initial cooldown, `_b_min_sep.py:1-11, 119`, VERIFIED), so the
attack fails. The divisor is the public `expected_batch_size` regardless.

U6. **RNG key reuse.** Sampler streams root at `fold_in(key, [rank], B_MIN_SEP_STREAM_FOLD | POISSON_STREAM_FOLD)`; Gaussian
noise at `fold_in(key, GAUSSIAN_STREAM_FOLD, step)` (`_gaussian.py:419`); MF at `fold_in(key, MF_GAUSSIAN_STREAM_FOLD,
"mf_gaussian_column", step)` (`_engine.py:412`); string roots are unreachable from integer folds
(`opaque-engine/tests/rng/test_stream_root_namespacing.py`, VERIFIED read). The shared `quantile_noise_key = gradient_noise_key`
(`:1478`) is split only in adaptive mode, which the design rejects. No reuse between sampling coins and noise.

U7. **Probe leaf through the optimizer.** After zeroing the noised probe entry, AdamW-BC (`_adam.py:239-240` falls back to
`v_raw` when `v − φ ≤ 0`), plain AdamW (decoupled WD on a zero param) and SGD-momentum all produce update exactly 0 and finite
parameters over 3 steps (`probe_optimizer.out`, VERIFIED); `resolve_noise_variance` is per path
(`_bias_correction.py:108-123`), so σ_h does not leak into other leaves' correction. The `on_pre_optimizer_step` seam receives
the same `NoisedPytree` object that `opt.update` consumes, so in-place zeroing is effective.

U8. **Hooks under gradient checkpointing (design's PLAUSIBLE T5) — now VERIFIED at toy scale.** `t5_recorder_ckpt.py`
(tiny Mellum, 2 layers, E = 4, k = 2, ragged mask, opaque patches, backbone `output_router_logits=True` under
`clipped_grad`'s `vmap(grad)`, `PerGroup` probe group): with `gradient_checkpointing_enable()` (forced non-reentrant,
`patches/torch/checkpoint/huggingface.py:35-52`) the recorder returns exactly L = 2 tensors while the router hook fires 2L
times (forward + recompute) — the recompute appends nothing because `capture_outputs` resets the `ContextVar` collector in its
`finally` (`output_capturing.py:105-107, 258-265`, VERIFIED) — and every gradient leaf including the probe equals the
non-checkpointed run to rel-L2 **0.00e+00**; the probe leaf equals `(λ/B)Σ_x d(x)` (`allclose` True), sums to 0 (−1.9e-9), and
`‖λ d(x)‖ ≤ C_h` for every example. The released vector is not changed by recompute.

U9. **NaN / fully-masked example.** `clip_pytree` sanitises NaN/Inf to 0 per example before clipping (`_pytree.py:489-527`,
VERIFIED), so an example with `T_x = 0` contributes 0 to the probe and to the gradient — no NaN reaches the release. (A
`T_x.clamp_min(1)` in `router_load_and_probs` is still advisable: bias hygiene, not privacy.)

U10. **DDP on a fresh run.** Per-rank sampler keys (`:3803`), contiguous shards, all-reduce of every leaf including the probe
(`sum_gradients_` asserts identical `max_norm`), shared-key noise after the reduction ⇒ `ŷ_t` and hence `f̃` rank-identical.

U11. **No gradient accumulation in the DP path.** `TrainingArguments.gradient_accumulation_steps` is hard-wired to 1
(`_training_arguments.py:1314-1316`, VERIFIED), so the HF-Trainer per-microbatch-`f` artefact (critic Exp A) cannot arise.

U12. **Post-processing chain.** Sum-zero projection, EMA/window, exact-variance recursion, James–Stein shrinkage, clamp, `D_t`,
trip: functions of noised outputs and public constants only. `aux.batch_size` is never used as a divisor.

U13. **Checkpoint / resume of the mechanism state.** The sidecar holds public state; the MF noise state carries the probe's
columns; a resume with a different leaf set trips the constant-max_norm latch (`_engine.py:473-517`) ⇒ `ConfigurationError`,
not a silent mismatch.

U14. **DPO reference forward.** Runs outside vmap in `_augment_inputs`; `capture_outputs` scopes its collector per call, so
reference router logits cannot reach the policy's probe.

U15. **Independent-draw v2 accounting** `poisson(g(nm),q)*T | poisson(g(c·nm),q₂)*(T/m)`: fresh coins and fresh noise per
factor, ZDW Thm 10 across the interleaving, Feldman–Shenfeld per factor; `Poisson` accepts any `DpProcess`
(`dpsgd/amplification/_poisson.py:25-45`, VERIFIED phase-1); `ConfigurationError` under MF is the right call (no theorem
verified for concurrent composition with b-min-sep).

Pre-existing side channels outside the design's scope but relevant to its privacy statement (per the review protocol's "all
releases in the privacy statement"): un-noised `loss`, `grad_norm`, `clip_rate` means (F11) **and** the realised `batch_size`
(`_dp_trainer.py:2250`), which under add/remove adjacency is `Binomial(N, q)` vs `Binomial(N+1, q)` — minuscule but
unaccounted. The design lists F11; it should add `batch_size`.

---

## 3. Evidence appendix

- `t5_recorder_ckpt.py / .out` — recorder under `vmap(grad)` with and without non-reentrant checkpointing; probe leaf identity,
  sum-zero, structural bound. Output: `ckpt=False: 2 tensors, hook fired 2×; ckpt=True: 2 tensors, hook fired 4×; rel-L2 0.00e+00
  on all five leaves; probe == (LAM/B)·Σd_x: True`.
- `corr_coins.py / .out` — hockey-stick δ(ε) for the independent-coin subsampled Gaussian pair vs the shared-coin pair (table in
  R1); `u = 0` reproduces the standard value exactly.
- `probe_optimizer.py / .out` — zeroed probe through AdamW-BC / AdamW / SGD-momentum with `PerGroup` `noise_stddev`; probe
  noise independent of gradient noise.

Sources relied on (all VERIFIED from primary text by phase-1 `critic` R6; not re-fetched): Feldman–Shenfeld Lemma 3.2 / Thm 3.3
(https://arxiv.org/abs/2602.17284); Zhu–Dong–Wang Def. 7 / Thm 10 / Thm 11 (https://arxiv.org/abs/2106.08567); Denisov et al.
Thm 2.1 (https://arxiv.org/abs/2202.08312); Dong–Roth–Su Thm 2.7 (https://arxiv.org/abs/1905.02383); Dong & Ganesh b-min-sep
Alg. 2 as cited by `_b_min_sep.py` (https://arxiv.org/abs/2602.09338, PLAUSIBLE for the algorithm number — I read the repo's
implementation, not the paper).
