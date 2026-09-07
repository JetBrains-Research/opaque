# Phase 1 — `primitives`: inventory of Opaque building blocks for a DP-MoE (Mellum2) mechanism

Agent: `primitives`. Scope: every Opaque primitive a designer could compose to (a) compute a per-example
router-load vector inside the vmapped loss, (b) clip+noise it as its own group, (c) feed the noisy aggregate
into the next step, (d) account for it — without reading the whole repo.

Conventions: `file:line` = repo path relative to `/home/user/opaque`. Status tags: **VERIFIED** (read the code
or ran it in this session), **PLAUSIBLE** (consistent with what was read, not executed / theorem number not
re-fetched), **REFUTED**. Scripts + raw outputs: `scratchpad/research/primitives/e{1,2,3}_*.{py,out,json}`.

---

## 0. Executive summary (what a designer needs to know)

1. **A per-example non-gradient statistic can ride along in the clipped pytree as its own clipping group with
   zero changes to opaque-engine.** `clipped_grad` differentiates any `argnums` (`_clipped_grad.py:85-101`),
   so a zero "probe" parameter `z ∈ R^E` with `loss += <z, f_x>` yields `∂/∂z = f_x` (the load vector) as a
   leaf; `PerGroup` clipping (`_pytree.py:439-477`) clips it to its own bound λ; `gaussian_noise` and
   `mf_gaussian_noise` allocate per-group σ (`noise_allocation.py:44-110`, `_gaussian.py:323-331`,
   `_mf_gaussian_noise.py:163-167`). **VERIFIED end to end (E2, E3).**
2. **Accounting of that joint release is unchanged: `gaussian(nm)`.** The per-group allocation
   σ_i = nm·√(C_i·ΣC_j) satisfies the Mahalanobis constraint Σ(C_i/σ_i)² = 1/nm² with equality
   (`noise_allocation.py:50-61`; numerically checked E1(f)). The price is paid in *utility*, not ε:
   σ_grad rises by √(1+λ/C) (E1(f): ×1.049 at λ/C=0.1, ×1.118 at 0.25, ×1.225 at 0.5, ×1.414 at 1).
3. **Heterogeneous composition inside Poisson IS supported** (`Poisson.__post_init__` accepts any `DpProcess`,
   `_poisson.py:25-45`; Rust `poisson_pld` applies Feldman–Shenfeld Thm 3.3/Alg 8-9 to an arbitrary base PLD,
   `src/amplification/poisson.rs:15-42`). `poisson(gaussian(1)|gaussian(σ_h), q)` reproduces the analytic
   Gaussian collapse nm_eff=(1+1/σ_h²)^{-1/2} to within ~1–3 % (upper bound side) — E1. At the preset regime
   (ε=3, q=256/5e5, T=15625, δ=1e-6 ⇒ nm=0.562) an extra same-batch release with σ_h = 2×nm raises ε from 3.0
   to 4.81, or costs ×1.165 gradient noise when ε-matched; σ_h = 4×nm costs ×1.041; σ_h = 8×nm costs ×1.016.
4. **An independently subsampled second release is much cheaper in ε than a same-batch one** (E1(h) vs (b)):
   at nm=1, σ_h=1 same-batch ε=1.00 vs independent-draw ε=0.489 (baseline 0.342). Design lever, if the load
   statistic can be computed on its own Poisson draw (forward-only pass).
5. **DPTrainer seams exist for every step of the recipe:** `compute_per_example_loss(_and_metrics)`
   (`_dp_trainer.py:2314-2467`), `_augment_inputs` (`:2295-2312`, runs outside vmap once per step),
   `on_pre_optimizer_step` callback receives the **noised** pytree keyed by param name (`:2218-2226`),
   `_create_grad_fn` (`:4230-4300`) hard-codes `argnums=0` — so the probe should live inside
   `trainable_params` (Route B) rather than as an extra argnum (Route A).
6. **HF's own `output_router_logits=True` path does NOT work under `vmap(grad)`** (in-place `scatter_add_`
   / `bincount` in `load_balancing_loss_func`, `modeling_mellum.py:583-604`; the Opaque fused/chunked-CE
   patch also *falls back to the original HF forward* in that case, `components/cross_entropy.py:212-230`).
   A forward hook on `MellumTopKRouter` capturing `router_logits` works under `vmap(grad)` with the patches
   applied and lets you compute f_x (top-k counts) and P_x (mean prob) per example (**VERIFIED**, E3).
7. **DP-FTRL constraints**: `mf_gaussian_noise` latches the first-call `max_norm` and rejects any change
   (`_engine.py:473-517`) — a `PerGroup` including the load group is constant, so it passes (E2, identity and
   band-MF). Per-step realized σ on the load leaf is base σ·‖row_t(C⁻¹)‖ (`_mf_gaussian_noise.py:186-192`),
   larger per step than DP-SGD (E2: 0.1716 vs 0.1443 at step 0 with bands=4) but anti-correlated across steps.

---

## 1. Clipping (`opaque.api.engine.clipping`, façades `opaque.dpsgd.clipping` / `opaque.dpftrl.clipping`)

### 1.1 Types that carry privacy metadata — `packages/opaque-engine/src/opaque/api/engine/types.py`

| Symbol | Where | What it carries |
| --- | --- | --- |
| `PerGroup(groups: Mapping[ParamPath|str,str], values: Mapping[str,float])` | `types.py:71-167` | leaf-path→group map + per-group scalar; `.effective = sqrt(Σv²)` (`:111-114`); `.num_groups`; arithmetic `*`, `/`, `PerGroup*PerGroup` (`:121-157`); `.for_path(path)` (`:159-166`) |
| `ClippedPytree(pytree, max_norm: float|PerGroup)` | `types.py:259-393` | `.sensitivity` = scalar effective L2 (`:272-278`); `.noise_stddev_for(noise_multiplier, allocation="optimal"|"isotropic")` (`:280-338`); scalar `*`/`/`/neg only, `+`/`-` raise (`:347-378`) |
| `NoisedPytree(ClippedPytree)` + `noise_stddev: float|PerGroup|None` | `types.py:396-411` | realized σ metadata; `max_norm` remains the record bound |
| `SecondMomentClippingOutput(grads, squared_grads)` / `SecondMomentNoiseOutput` | `types.py:421-460` | paired first/second-moment streams |
| `clipped(pytree, *, max_norm)`, `noised(pytree, *, max_norm, noise_stddev)` | `types.py:465-479` | manual wrappers for values you privatised yourself |
| `ClipState` (marker ABC), `NoiseState(_step_counter, _rng_key)` | `types.py:171-203` | |

**Metadata carried = only `max_norm` (and `noise_stddev`).** There is **no count / batch-size field** on
`ClippedPytree`; the realized batch size travels in `ClippedGradAux.batch_size` / `ClippingStats.batch_size`
(diagnostic, `_clipped_fun.py:37-70`). Sensitivity after `normalize_by=n` is `max_norm = C/n`
(`_clipped_fun.py:611-615`, `_clipped_grad.py:215-218`). **VERIFIED.**

Public façade: `opaque.types` = `packages/opaque-engine/src/opaque/types.py` (re-exports the above).

### 1.2 Fixed clipping — `clipped_grad` (`_clipped_grad.py:85-362`)

```python
clipped_grad(loss_fn, argnums=0, has_aux=False, *, clipping_norm: float|PerGroup, normalize_by=1.0,
             batch_argnums=1, return_aux=False, return_stats=False, second_moment=False,
             pre_clipping_transform=lambda x: x, microbatch_size=None, dtype=None, compute_dtype=None,
             _scale_fn=None) -> (grad_fn, FixedClipState)
grad_fn(*args, state=clip_state, **kwargs) -> (ClippedPytree | (ClippedPytree, ClippedGradAux), state)
```
- Built on `torch.func.grad_and_value(loss_fn, argnums, has_aux=True)` (`:280`) then `clipped_fun` (`:300-313`).
- **`argnums` may be a tuple** and the arguments must not carry a batch axis (`:63-82`) → an extra
  non-batched probe argument is legal (Route A, E2 verified: pytree becomes `(dict, tensor)` with paths
  `(0,'w')…,(1,)`).
- **`pre_clipping_transform(grad_pytree) -> pytree` "possibly with different structure"** (`:172-177`) runs per
  example *before* clipping but sees only the gradient (not the loss aux) → it cannot inject a statistic that
  is not already in the gradient; useful for re-rooting / re-scaling leaves.
- `has_aux=True`: per-example `loss_aux` returned in `ClippedGradAux.loss_aux` — **no sensitivity guarantee**
  (`:153-156`, `:141-145`), NOT noised; "must not be fed back into private computation" (`:37-39`).
- Empty batch short-circuit `:220-277` (zero grads + empty aux; DDP collectives still run).
- Per-group path: `_clip_pytree_per_group` (`_pytree.py:439-477`): each group scaled by
  `min(1, C_g/‖g_g‖)` (`:461-465`), ULP-guarded so the *stored* values respect the bound (`:111-147`);
  leaf-path set must equal `PerGroup.groups` keys exactly (`:56-74`).
- `ClipPytreeAux(norm, group_norms)` (`_pytree.py:28-35`), `ClippedGradAux(loss_values, grad_norms,
  clipped_grad_norms, loss_aux, clipping_rate, batch_size, group_norms)` (`_clipped_grad.py:33-60`).

### 1.3 Arbitrary per-example function clipping — `clipped_fun` (`_clipped_fun.py:492-861`)

```python
clipped_fun(fun, has_aux=False, *, batch_argnums=0, clipping_norm: float|PerGroup=1.0, normalize_by=1.0,
            return_aux=False, return_stats=False, second_moment=False, microbatch_size=None,
            dtype=None, compute_dtype=None, _scale_fn=None) -> (clip_fn, FixedClipState)
```
"clip its output and sum across a batch" for **any** per-example function output pytree (`:497-503`);
formal guarantee = L2 sensitivity `clipping_norm` under add/remove, `2·clipping_norm` under replace-one
(`:520-530`). `vmap(..., randomness="same")` (`:733-738`, microbatched `:255-260`). This is the primitive to
clip a router-load vector **as a separate release** (second vmap pass, separate mechanism); the gradient-graft
route (§1.2) folds it into the same release instead. Exposed via `opaque.api.engine.clipping.fun`
(`fun.py:20-29`) and mirrored at `opaque.dpftrl.clipping.fun` (`opaque-dpftrl/.../clipping/fun.py:14-28`).

### 1.4 AUTO-S — `auto_clipped_grad` (`_auto.py:203-364`), `auto_scale_pytree` (`_pytree.py:350-436`)

`R: float|PerGroup`, `gamma=0.01`; scale `R/(‖g‖+γ)` with **no `min(1,·)`** (`_pytree.py:424-430`, per-group
`:314-347`). Per-record bound constant ⇒ compatible with MF latch (`_auto.py:15-29`, `:283-296`). Caveat for a
load group: AUTO-S rescales *every* example's load vector to norm ≈λ (direction only) — a different estimator
than fixed clipping (which leaves small vectors untouched). **VERIFIED (read).**

### 1.5 Adaptive clipping — `adaptive_clipped_grad` (`opaque-dpsgd/.../clipping/_adaptive.py:164-215`)

`initial_clipping_norm: float|PerGroup`, `target_quantile`, `learning_rate`, `clipping_norm_min/max`,
`fraction_noise_std`, `key`; threshold drifts per step ⇒ **rejected by MF noise** (`_engine.py:473-517`);
accounted by `dpsgd_acc.adaclip(gaussian(nm), expected_batch_size, num_groups)` (`_adaclip.py:131`,
trainer `_dp_trainer.py:4333-4341`).

### 1.6 `per_group(params, patterns=None, *, fallback=None, allow_unused_patterns=False, **kw) -> PerGroup`
(`_per_group.py:44-186`): substring match on dotted leaf path, each leaf exactly one pattern, `fallback`
catch-all group named `"fallback"`. A probe leaf `router_load_probe` gets its own group with
`per_group(trainable, router_load_probe=λ, fallback=C)` (E2 Route B). **VERIFIED.**

---

## 2. Noise

### 2.1 Per-group / paired allocation — `opaque/api/engine/noise_allocation.py` (read fully)

- `per_group_noise_stddev(max_norm: PerGroup, noise_multiplier) -> PerGroup` (`:44-110`):
  σ_i = nm·√(B_i·Σ_j B_j) (`:103-110`); short-circuit nm=0 (`:97-102`). Docstring: minimizes Σ d_i σ_i²
  subject to Σ(C_i/n)²/σ_i² ≤ 1/nm²; accounting `gaussian(nm)` "with no composition penalty regardless of the
  number of groups" (`:50-61`). Numerically: Σ(C_i/σ_i)² = ΣC_i/(nm²·S) = 1/nm² exactly (E1(f)). **VERIFIED.**
  The underlying fact — an anisotropic Gaussian with block sensitivities Δ_i and stddevs σ_i is dominated by
  the 1-D Gaussian with Δ/σ = √(ΣΔ_i²/σ_i²) (whitening) — is the standard Gaussian-mechanism statement of
  Dong, Roth, Su (2019) https://arxiv.org/abs/1905.02383 (GDP of the Gaussian mechanism, μ = Δ/σ; theorem
  number not re-fetched here: **PLAUSIBLE** for the numbering, the fact itself is standard).
- `paired_noise_stddevs(noise_multiplier, *, first, second)` (`:153-255`): S = Σ_g(Δ¹_g+Δ²_g),
  σ¹_g = nm√(Δ¹_g S), σ²_g = nm√(Δ²_g S); joint PLD = single sensitivity-1 Gaussian at nm (`:169-174`);
  DP-FTRL translation nm/‖C₁‖ (`:183-186`). Polymorphic float/PerGroup but both streams same kind (`:219-226`).
- `resolve_paired_clipped(SecondMomentClippingOutput, *, noise_multiplier)` (`:258-287`).
- RNG stream tags `PAIRED_FIRST_STREAM_FOLD="opaque.paired.first"`, `..second` (`:40-41`); rationale for
  string tags vs ints (`:29-39`).
- **What is NOT there:** no API for *two different noise multipliers* on two groups under one accountant other
  than via the Mahalanobis allocation (i.e., you cannot say "grad at nm₁, load at nm₂" and account
  `gaussian(nm₁)`; you either (i) use the joint allocation and account `gaussian(nm)`, or (ii) treat the load
  as a second mechanism and compose, §3).

### 2.2 `gaussian_noise` — `opaque-dpsgd/src/opaque/api/dpsgd/noise/_gaussian.py:162-460`

```python
gaussian_noise(*, noise_multiplier, key: RngKey, bound=None, compute_dtype=torch.float32)
    -> (noise_fn, GaussianNoiseState);  noise_fn(clipped: ClippedPytree|SecondMomentClippingOutput, state)
    -> (NoisedPytree|SecondMomentNoiseOutput, state)
```
- σ = nm·max_norm (scalar) (`:326-331`) or `per_group_noise_stddev(max_norm, nm)` for `PerGroup` (`:323-325`);
  per-leaf σ looked up by `PerGroup.for_path` (`:295-315`). Inverse-CDF sampling from a CPU generator
  (`:239-272`). Step key = `fold_in(_rng_key, GAUSSIAN_STREAM_FOLD, step_counter)` (`:434`), paired streams
  fold `PAIRED_*` tags (`:346-357`). Rejects `NoisedPytree` inputs (`:417-424`). **VERIFIED** (E2: σ values
  match the closed form exactly).
- Bounded Gaussian (`bound=`) is experimental and **not covered by the accountant** (`:22-28`).

### 2.3 DDP — `opaque/api/engine/distributed/{gradients.py, collectives.py, _state.py}`

- `sum_gradients_(clipped)` / `reduce_pytree_(pytree, op="sum")` all-reduce every tensor leaf **including a
  probe leaf**, asserting `max_norm` (and `noise_stddev`) identical across ranks (`gradients.py:150-200`);
  out-of-place `reduce_pytree` updates metadata: sum keeps `max_norm`, mean divides by world size; noised sum
  scales σ by √W (`gradients.py:1-15`, `:117-136`). `sync(*states)` type-dispatched (`_state.py:581-620`),
  `register_sync_type(type, fn)` (`:537-546`); `ClippedGradAux` sync gathers per-example tensor fields with
  `all_gather_object` and size-weights `clipping_rate` (`clipping/_distributed.py:214-286`).
- Trainer passes the **same** noise key on every rank (`_dp_trainer.py:1478`, `:1611-1614`;
  comment `:2170-2173`), so after `sum_gradients_` the noised pytree — including a noised load leaf — is
  bit-identical on all ranks; no extra sync of an EMA state is needed. **VERIFIED (read).**

### 2.4 RNG — `opaque/api/engine/random/_engine.py`
`key(seed)` (`:55`), `fold_in(key, *data: int|str)` (`:62-116`; ints for steps/ranks, a unique **string** tag
roots each mechanism), `split(key, n)` = integer folds (`:119-131`), `generator_from_key` (`:134`). A new
mechanism must root itself with its own string tag (e.g. `fold_in(key, "opaque.moe.load_ema")`).

---

## 3. Accounting (`opaque.accounting`, `opaque.dpsgd.accounting`, `opaque.dpftrl.accounting`)

### 3.1 Algebra — `opaque-accounting/src/opaque/api/accounting/core/_base.py`
`DpProcess` (`:112-`) with `pld()`, `epsilon_at(δ)`, `delta_at`, `advantage`, `beta_at`, `risk_at`;
`proc * k` (`:454-477`), `a | b` heterogeneous compose (`:483-`; `composition/__init__.py:50-72`),
`cached`, `Accountant`, `calibrate(budget, process: nm->DpProcess, param_min, param_max, tolerance, ...)
-> CalibrateResult(param, achieved, target, iterations, converged, mc_failure_probability)`
(`calibration.py:110-160`, `:71-95`). Generic mechanisms: `identity()`, `nonprivate()`, `eps_delta(ε,δ)`
(`mechanisms/_{identity,nonprivate,eps_delta}.py`). Native surface (`opaque_accounting.pyi`): `gaussian_pld(nm)`
(sensitivity 1, `:236-253`), `poisson_pld(base: Pld, rate)` (`:302-314`), `mf_gaussian_pld(nm, sensitivity)`
(`:485-508`), plus b-min-sep / balls-in-bins MC PLDs. **There is no `gaussian(sensitivity=…)`** — sensitivity
is always folded into the noise multiplier (σ/Δ), consistent with the runtime where `max_norm` sets σ.

### 3.2 DP-SGD factories — `opaque-dpsgd/src/opaque/api/accounting/dpsgd/`
`gaussian(nm)` (`mechanisms/_gaussian.py:59`), `poisson(inner: DpProcess, sample_rate, *,
truncated_batch_size=None, dataset_size=None)` (`amplification/_poisson.py:176-230`): **"Plain Poisson accepts
any Opaque DpProcess"** (`:25-30`); truncated Poisson requires Gaussian/AdaClip/NonPrivate inner (`:95-104`).
`adaclip`, `parallel_poisson`, `k_out_of_t`.

**Can two Gaussians with different σ per step be composed under Poisson?** Yes:
`poisson(gaussian(nm) | gaussian(σ_h), q)` builds `Composed` → `poisson_pld(composed_pld, q)`. Rust
`src/amplification/poisson.rs:15-42`: Feldman & Shenfeld, "Efficient privacy loss accounting for subsampling
and random allocation", arXiv:2602.17284, Theorem 3.3 / Algorithms 8–9; "The input PLD may be a conservative
approximation of a mechanism; the pointwise transform preserves that domination"; asymmetric add/remove PMFs
handled (`:31-41`). Exact Gaussian fast path when the base is a pure Gaussian (`:27-29`). **VERIFIED (read +
numerics below).** (The arXiv id and theorem numbers are quoted from the Rust doc comment, not re-fetched.)

### 3.3 Numerical results (E1, `e1_accounting.out`; q = 256/500000 = 5.12e-4, T = 15625, δ = 1e-6)

Baseline `poisson(gaussian(1.0), q)*T`: **ε = 0.3415**.

| σ_h (extra release nm) | same-batch `poisson(g(1)|g(σ_h))*T` | analytic collapse `g(nm_eff)`, nm_eff=(1+σ_h⁻²)^-½ | independent 2nd draw `(p(g1)|p(g σ_h))*T` | ε-matched grad nm' |
| --- | --- | --- | --- | --- |
| 1 | 1.0003 | 0.9724 (nm_eff .7071) | 0.4894 | unreachable |
| 2 | 0.4289 | 0.4250 (.8944) | 0.3686 | 1.1613 |
| 4 | 0.3643 | 0.3611 (.9701) | 0.3482 | 1.0380 |
| 8 | 0.3494 | 0.3463 (.9923) | 0.3436 | 1.0129 |

Context (E1(g)): ε(σ)= 4.79/2.25/0.97/0.57/0.34/0.18 at σ = .5/.6/.7071/.8/1/1.5 — the collapse to σ/√2 is
expensive because subsampled-Gaussian ε is strongly convex in 1/σ below σ≈1.

Preset regime (ε=3 target ⇒ nm = **0.5622**): same-batch extra release at σ_h = k·nm:
k=1 → ε 12.19 (ε-match unreachable); k=2 → ε 4.81 / nm' 0.6548 (×1.165); k=4 → ε 3.51 / ×1.041;
k=8 → ε 3.21 / ×1.016.

Joint per-group release (accounting unchanged, E1(f)): σ_grad/(nm·C) = √(1+λ/C) = 1.049 / 1.118 / 1.225 /
1.414 for λ/C = .1/.25/.5/1; σ_load/(nm·λ) = √(1+C/λ) = 3.32 / 2.24 / 1.73 / 1.41; isotropic alternative
σ/(nm·C) = √(1+(λ/C)²) = 1.005 / 1.031 / 1.118 / 1.414.

**Interpretation.** (i) The same-batch heterogeneous composition and the joint per-group release are *the same
mechanism family* (both are one Gaussian release on the sampled batch with Mahalanobis budget 1/nm²); the
per-group allocation just picks the MSE-optimal point, and the composed-PLD route is a ~1–3 % looser (safe)
numerical bound. (ii) Choosing between "pay in ε" and "pay in gradient noise" is therefore only a
reparameterisation; the only genuinely cheaper option is an **independently subsampled** load release (E1(h)),
because two small-q draws amplify better than one draw of a σ/√2 mechanism. (iii) With λ/C ≲ 0.25 the joint
route costs ≤ 12 % extra gradient σ, which is the regime to aim for (the load vector is bounded by
construction: with the HF definition f_x ∈ [0,1]^E, ‖f_x‖₂ ≤ √k·… see §6).

### 3.4 DP-FTRL accounting — `opaque-dpftrl/src/opaque/api/accounting/dpftrl/`
`mf_gaussian(nm, strategy, *, n_steps=1, min_sep=1, max_participations=None)` (`mechanisms/_mf_gaussian.py:129-`):
PLD = `mf_gaussian_pld(nm, strategy.sensitivity(n_steps, min_sep, max_participations))` (`:118-128`) — **one
Gaussian for the whole horizon**, effective multiplier nm/‖C‖_sens. Amplifiers: `poisson(inner: MfGaussian,
sample_rate, *, n_steps, truncated_batch_size=None, dataset_size=None) -> CyclicPoisson` (`amplification/
_poisson.py:225-300`; BandMF ⇒ ceil(n_steps/bands) groups), `b_min_sep(inner, *, n_steps, p0) -> BMinSep`
(MC PLD, `_b_min_sep/__init__.py:199-240`), `balls_in_bins`. **They require `MfGaussian` inner** (`:281-284`,
`:219-222`) — you cannot put a `Composed` inside them. Hence an extra per-step release under DP-FTRL must either
(a) be folded into the same clipped pytree as a group (sensitivity S = C+λ enters via `PerGroup` max_norm and
`per_group_noise_stddev`; accountant unchanged at `mf_gaussian(nm, strategy)`), or (b) be a separate mechanism
composed at the process level: `Accountant(prefix=cached(horizon)) | other_process` (`_dp_trainer.py:270-283`,
`DpProcess.__or__`) — valid only if that other mechanism's participation model is accounted on its own (e.g. its
own Poisson draws), not the b-min-sep schedule.

### 3.5 Trainer wiring
`_build_mechanism` (`_dp_trainer.py:4310-4420`): Gaussian branch builds `nm -> poisson(gaussian(nm), q)`
(adaclip wrapper when adaptive; `num_groups = clip_norm.num_groups`, `:4330-4341`); DP-FTRL branch returns the
amplifier factory (`_dpftrl.py:116-178`). `_initialize_accounting` (`:270-283`): horizon processes are installed
once as `Accountant(prefix=…)`, per-step processes composed each step (`_account_independent_step`, `:286-289`).
Calibration via `acc.calibrate` (`_calibrate_noise`, `:4425-`).

---

## 4. DPTrainer seams — `packages/opaque-transformers/src/opaque/api/transformers/trainer/_dp_trainer.py`

### 4.1 Setup (`:1340-1500`)
`make_functional(model, disable_autograd_tracking=True, partition_trainable=True)` (`:1357-1361`);
`batch_keys = _discover_batch_keys()` (`:1369`, `:3651-`; tensor keys emitted by the collator on one example —
placeholder columns must be seeded at construction, cf. TR-DPO `_dpo_trainer.py:721-748`);
`wants_metrics = _overrides_metrics_seam()` ⇒ `has_aux=True` (`:1383-1389`, `:1491`);
`_build_per_example_loss` closes over `frozen_params` and calls
`compute_per_example_loss[_and_metrics](fmodel, {**frozen,**trainable}, inputs)` (`:3523-3600`);
`_create_grad_fn` (`:4230-4300`) dispatches `adaptive_clipped_grad` / `auto_clipped_grad` / `clipped_grad`
with **`argnums=0`, `return_aux=True`, `normalize_by=expected_batch_size`, no `pre_clipping_transform`**;
noise: `gaussian_noise(noise_multiplier, key=gradient_noise_key)` (`:1596-1614`) or
`mf_gaussian_noise(trainable_params, mf.strategy, n_steps, min_sep, max_participations, nm, key=key(a.seed))`
(`:1628-1636`). `_TrainingContext` fields (`:238-283`): `grad_fn, clip_state, noise_fn, noise_state, opt,
opt_state, accounting, mechanism, step_process, batch_keys, clip_norm, mechanism_kind, horizon_process, mf, …`.

### 4.2 `training_step` (`:2053-2293`) — order of operations
1. `_prepare_input` → **`_augment_inputs(inputs)`** (`:2072-2078`; hook `:2295-2312`: "Runs once per step …
   *outside* vmap … keys it writes must already exist in `ctx.batch_keys`").
2. `batch_args = (inputs[k] for k in ctx.batch_keys)` (`:2082-2083`).
3. `(grads, aux), clip_state = ctx.grad_fn(ctx.trainable_params, *batch_args, state=…)` under autocast/offload
   (`:2126-2131`); OOM made collective under DDP (`:2137-2154`).
4. DDP: `sum_gradients_(grads)`; `clip_state, aux = sync(clip_state, aux)` (`:2172-2175`).
5. `noisy_grads, noise_state = ctx.noise_fn(grads, noise_state)` (`:2183`).
6. **`on_pre_optimizer_step` callback with `grads=noisy_grads` (NoisedPytree keyed by param name) and
   `trainable_params`** (`:2210-2218`) — the natural seam to read a noised probe leaf.
7. `updates, opt_state = ctx.opt.update(noisy_grads, opt_state, params=trainable_params)`;
   `trainable_params = torchopt.apply_updates(...)` (`:2233-2238`) — expects the *same dict structure* as
   `trainable_params`; a tuple-rooted pytree (Route A) would break here.
8. `on_optimizer_step` (`:2243-2249`); metrics from `aux` (`:2258-2293`): `loss` = un-noised mean of per-example
   losses, `loss_aux` = per-example telemetry meaned ("Same un-noised diagnostic posture", `:2266-2270`).

### 4.3 Loss hooks
`compute_per_example_loss(fmodel, params, inputs, *, return_logits=False)` (`:2314-2434`): default
`fmodel(params, **inputs)["loss"]`, `compute_loss_func` escape hatch, label smoothing rebuild.
`compute_per_example_loss_and_metrics(fmodel, params, inputs) -> (loss, dict[str, Tensor])` (`:2436-2467`):
telemetry dict rides `ClippedGradAux.loss_aux`, DDP-gathered, meaned per logged step and in eval. **Aux is
private (not noised, not accounted)** — it is *logged* un-noised (existing behaviour; flag for the DP review:
`differential-privacy-review.md` "Composition and accounting — all releases … included in the privacy
statement") and must never feed back into training. Anything fed back must go through clip+noise (§6).

### 4.4 Persisting cross-step state (e.g. a noisy load EMA)
- Runtime checkpoint bundle `RuntimeCheckpoint` (`_checkpoint.py:201-`), written by `save_dp_runtime_state(path,
  *, clip_state, noise_state, sampler_state, sample_rate, target_delta, noise_multiplier, …,
  horizon_process_state, mf_*, lr_*)` (`:311-381`) — **fixed signature, no extension slot**; restored by
  `_apply_runtime_state` (`_dp_trainer.py:5335-5356`) via `opaque.serialization.from_state_dict`.
  Call sites: `_save_checkpoint` (`:4910`, bundle at `:5091-5120`), resume at `:1176`, `:5317`.
- Serialization registry (`opaque-base/.../serialization/_dispatch.py:83-108`, `_registry.py:27-58`,
  `_structural.py:50-114`): any frozen dataclass with tensor/primitive fields round-trips with
  `state_dict`/`from_state_dict` out of the box (E2: `LoadEmaState(ema: Tensor, step: int, beta: float)`
  → keys `['beta','ema','step']`, restored exactly). Tensor handlers registered in
  `opaque-engine/.../serialization/_structural.py:52-111` (template-driven dtype/device).
- `DPTrainerState` (`_state.py:38-124`) is JSON (`to_json`/`from_json`, unknown keys filtered) — fine for a few
  floats, not for tensors.
- **Practical route:** a subclass overrides `_save_checkpoint` to write a sidecar (`torch.save(state_dict(ema))`)
  and `_apply_runtime_state` to restore it; under DDP the EMA is rank-identical by construction (§2.3), so no
  `register_sync_type` handler is required (register one anyway if you want `sync()` to assert equality:
  `assert_pytree_equal`, `_state.py:90`).

---

## 5. Functional / LoRA layer

- `make_functional(mod, disable_autograd_tracking=False, partition_trainable=False)` (`functional/__init__.py:15-186`):
  `partition_trainable=True` → `(fmodel(params_dict, *a, **kw), trainable: dict, frozen: dict)` split on
  `requires_grad` (`:152-171`), backed by `torch.func.functional_call`. `with_batch_dim` (`:219-360`),
  `merge`/`partition` (`pytree.py:181-340`).
- **LoRA on stacked expert `nn.Parameter (E, 2I, H)`**: nothing in `opaque-patches` targets experts
  (`grep experts|moe peft/` → no hits; `peft/_router.py:52-130` fuses only `q/k/v_proj` and `gate/up/down_proj`
  `nn.Linear` LoRA). The MoE component patch only swaps `MellumExperts.forward` onto `opaque_moe(x, gate_up_proj,
  down_proj, idx, weights, grouped)` (`components/moe.py:22-45`, kernel `kernels/moe.py:578-`); expert weights are
  read straight off the module (`self.gate_up_proj`), so a PEFT `ParamWrapper` (installed peft 0.20.0 supports
  `LoraConfig(target_parameters=["…experts.gate_up_proj", "…experts.down_proj"])`, `peft/tuners/lora/config.py:521-531`,
  `layer.py:2193-2236`) would have to be validated under `functional_call`+vmap — **untested; PLAUSIBLE at best**.
  `opaque_moe`'s backward "frozen experts skip those gradients entirely" (`kernels/moe.py:600-603`): training
  experts turns on the dense per-example expert-weight gradient path (memory ∝ B·E·2I·H per microbatch).
- Fused LoRA QKV: `_opaque_fused_lora_qkv` / `_resolve_fused_qkv_forward_factory` (`peft/components/qkv*.py`),
  gated on all three of q/k/v having LoRA, frozen bias, no dropout (`peft/_router.py:74-97`).
- Preset `mellum2-kstack` (`examples/train_dpftrl.py:873-896`): LoRA r=16/α=32 on `q,k,v,o_proj` only, batch 256,
  8 epochs over 500k samples, ε=3, bands=64, `b_min_sep`, microbatch 8, seq 1024.

---

## 6. Mellum router facts relevant to the recipe (VERIFIED, E3)

- `MellumTopKRouter.forward(hidden) -> (router_logits, router_scores, router_indices)` (`modeling_mellum.py:323-341`;
  softmax in fp32, top-k, renorm). `MellumSparseMoeBlock.forward` discards `router_logits` (`:350-355`);
  `MellumForCausalLM.forward` computes `aux_loss = load_balancing_loss_func(outputs.router_logits, …)` whenever
  `output_router_logits` and adds `router_aux_loss_coef * aux_loss` to `loss` when labels are given (`:691-704`).
  `load_balancing_loss_func` (`:540-607`): f_e = (Σ_layers Σ_tokens 1[e∈topk]·mask)/(L·Σmask) via `bincount`
  (no mask) or in-place `scatter_add_` (mask), P_e = mean masked softmax prob, loss = E·Σ_e f_e P_e.
- Under `vmap(grad)`: `output_router_logits=True` **fails** (`RuntimeError: vmap: scatter_add_ … Tensor other …
  vmapped over but self is not`), and would in any case bypass the chunked-CE patch (`cross_entropy.py:212-230`,
  falls back to full-logit HF forward — at vocab 98304×seq 1024 that is the memory path PR #978 removed).
- **Hook route works**: `register_forward_hook` on every `MellumTopKRouter` capturing `out[0]`; inside the
  per-example loss, `logits = stack(captured)` `(L,T,E)`, `p = softmax(fp32)`, `top = topk(p,k).indices`,
  `f_x = (top[...,None]==arange(E)).sum((0,1,2))/(L·T)`, `P_x = p.mean((0,1))` — `F.one_hot` is NOT vmap-safe
  (`.item()` in its bounds check), the broadcast-compare is. Eager per-example f matches the vmapped one; grafted
  probe leaf equals the mean of clipped f_x; surrogate loss `CE_x + coef·E·<f̃, P_x>` differentiates fine under
  `vmap(grad)` (E3 (ii),(iii)). Batch-level f (HF) equals the token-weighted mean of per-example f_x by definition
  (`:583-604`) — exactly equal for equal-length unpadded examples.
- Caveats: hooks fire again during checkpoint recompute (`opaque.patches.torch.checkpoint`); clear the list at the
  start of every per-example call (done in E3) and keep the stack before backward; `torch.compile` of the grad
  transform (`_grad_compiler`, `:4200-4228`) will likely graph-break on the Python list — the fullgraph fallback
  handles it. A cleaner long-term alternative: monkeypatch `transformers.models.mellum.modeling_mellum.
  load_balancing_loss_func` (module-level lookup at `:693`) with a vmap-safe per-example version that accepts a
  constant f̃ — but that still goes through the CE-patch fallback; extending the chunked-CE causal-LM patch to
  request/return router logits is the memory-safe fix.
- Bound on the load vector for choosing λ: with the HF definition, f_x ∈ [0,1]^E, Σ_e f_x(e) = k (=8), so
  ‖f_x‖₂ ≤ √k·max_e f… more precisely ‖f_x‖₂ ≤ √(Σ f_e²) ≤ √k when at most k experts saturate; for balanced
  routing ‖f_x‖₂ ≈ k/√E (8/8 = 1.0 for E=64). Choosing λ ≈ 1 (clip rarely) vs C≈1 gives λ/C≈1 ⇒ ×1.41 grad σ;
  rescale the statistic (e.g. release f_x/√k or the *deviation* f_x − k/E) to push λ/C to ≲0.25 (×≤1.12).

---

## 7. DP-FTRL-specific constraints a solution must respect (VERIFIED, read)

1. **Constant per-step sensitivity**: `_validate_constant_max_norm` (`dpftrl/noise/_engine.py:473-517`) latches
   `max_norm` (float or `PerGroup`, compared by equality) at the first call; any later change raises
   `ConfigurationError` (E2 confirmed for identity and band-MF). So λ, C, `normalize_by` and the group map must be
   fixed for the run; adaptive clipping is out; AUTO-S and fixed clipping are in (`_auto.py:15-29`).
2. **Single release per step per example**: the MF proof treats the whole clipped pytree as one vector with
   ‖·‖ ≤ effective bound; adding the load group changes the per-record bound from C to √(C²+λ²)
   (isotropic) / S=C+λ (allocation) *inside* the same `PerGroup` — the accountant `mf_gaussian(nm, strategy)` is
   unchanged because the per-group σ scale is applied before the streaming matrix (`_mf_gaussian_noise.py:163-179`).
3. **Participation schedule**: `BMinSepSampler(data_source, bands, sampling_prob, n_steps, *, key)`
   (`dpftrl/sampling/_b_min_sep.py:30-`; per-iteration p vs p₀ conversion `:6-11`, `:45-49`) fixes which
   examples appear in which step; the load statistic must be computed from the *same* batch (no second draw),
   otherwise it is a different mechanism with its own participation model and must be accounted separately (§3.4).
4. **Realized per-step σ on any leaf is base σ·‖row_t(C⁻¹)‖** (`_mf_gaussian_noise.py:186-192`), correlated
   across steps; a per-step consumer (the load EMA) sees more noise per step than under DP-SGD (E2: 0.1716 vs
   0.1443), but an EMA is a linear filter of the stream, which is the workload MF strategies are optimised for
   (`band_mf_strategy(bands, momentum, lr_schedule)`, `_band_mf.py:142-171`) — **PLAUSIBLE** design note, not
   quantified here.
5. Whole-horizon accounting: never `* num_steps` a DP-FTRL process (`_poisson.py:236-238`,
   `_dp_trainer.py:4313-4318`); horizon must equal `total_steps` (`:1621-1627`).

---

## 8. Recipe sketch — exact functions/classes to touch

**(a) Per-example router-load vector inside the vmapped loss.**
Subclass `DPTrainer` (or `DPSFTTrainer`); at construction register forward hooks on every
`transformers.models.mellum.modeling_mellum.MellumTopKRouter` (`self._router_logits: list`). Override
`compute_per_example_loss(fmodel, params, inputs)`: clear the list, `out = fmodel(params, **inputs)` (keeps the
chunked-CE path), `logits = torch.stack(self._router_logits)` `(L,T,E)`, `p = softmax(logits.float(), -1)`,
`f_x = ((topk(p,k).indices[...,None] == arange(E)).sum((0,1,2)) / (L·T_valid))` (mask padding with
`inputs["attention_mask"]` before the sums), `P_x = p.mean((0,1))`. Return
`out["loss"] + coef·E·<f̃, P_x> + <params["router_load_probe"], f_x.detach()>` (surrogate H2 with f̃ constant +
probe graft). Because `params["router_load_probe"] == 0`, the probe term is exactly 0 in value and contributes
`f_x` only to `∂/∂probe` (E2 "other-param grads unaffected": True).

**(b) Clip + noise it as its own group.** Add `router_load_probe = zeros(E)` to the trainable set (a registered
`nn.Parameter` with `requires_grad=True` on the model, so `make_functional(partition_trainable=True)` picks it
up; it is never used by the forward). Set `TrainingArguments.clipping_norm` to a dict / build
`per_group(trainable_params, router_load_probe=λ, fallback=C)` (`_per_group.py:44`; the trainer resolves dict
`clipping_norm` into a `PerGroup`, `_training_arguments.py:288`). `_create_grad_fn` then produces a
`ClippedPytree` whose `max_norm = PerGroup({'fallback': C/n, 'router_load_probe': λ/n})`;
`gaussian_noise`/`mf_gaussian_noise` allocate σ_grad = nm·√(C(C+λ))/n, σ_load = nm·√(λ(C+λ))/n
(`noise_allocation.py:103-110`; E2 numbers). Keep `clipping_mode="fixed"` or `"auto"` (never `"adaptive"` with MF).
Do **not** enable `second_moment` unless you want the load's squared stream to consume budget too
(`paired_noise_stddevs` sums Δ¹+Δ² over *all* groups, `:252`).

**(c) Feed the noisy aggregate into the next step.** Read the noised probe leaf in an `on_pre_optimizer_step`
callback (`grads.pytree["router_load_probe"]`, `_dp_trainer.py:2210-2218`; identical on all ranks, §2.3) or by
wrapping `ctx.noise_fn` after setup; update `self._load_ema = β·ema + (1−β)·noisy_f` (a frozen dataclass
`LoadEmaState(ema, step, beta)`, serialisable as-is, E2). Set `f̃` for the next step either (i) as a closure
tensor on the trainer read inside `compute_per_example_loss` (unbatched under vmap; update it in place in
`_augment_inputs`, `:2295-2312`), or (ii) as a seeded batch column `load_target` of shape `(B,E)` that
`_augment_inputs` overwrites each step (TR-DPO pattern, `_dpo_trainer.py:807-841`, columns seeded at
construction `:721-748` so `_discover_batch_keys` includes them). Reset the probe to zero each step: in
`_augment_inputs`, `self._ctx.trainable_params["router_load_probe"].zero_()` (the optimizer will otherwise
integrate the noisy load into it; alternatively exclude the leaf from the optimizer with a masked torchopt
chain). Checkpoint the EMA via a sidecar in `_save_checkpoint` / restore in `_apply_runtime_state` (§4.4).

**(d) Account for it.** DP-SGD: nothing changes — `poisson(gaussian(nm), q) * T` (`_build_mechanism`
`:4380-4384`), because the joint per-group release is one Gaussian at nm (§2.1, §3.3). DP-FTRL: nothing changes
— `b_min_sep(mf_gaussian(nm, band_mf_strategy(bands)), n_steps, p0)` (`_dpftrl.py:152-159`), the latch accepts
the constant `PerGroup` (E2). What changes is the *utility* budget: gradient σ ×√(1+λ/C). If instead the load is
released as a *separate* mechanism (second vmap via `clipped_fun` + its own `gaussian_noise` with its own
`fold_in(key,"…load")` root), account `poisson(gaussian(nm) | gaussian(σ_h), q) * T` (same batch; E1 table) or
`(poisson(gaussian(nm),q) | poisson(gaussian(σ_h),q)) * T` (independent draw; E1(h)) — the latter is not
available under b-min-sep without a separate participation analysis (§3.4, §7.3).

**Where the privacy claim rests.** (1) `clipped_grad` sensitivity contract (`_clipped_grad.py:135-145`,
add/remove; ×2 for replace-one) applied to the joint (grad, f_x) vector, per-group bounds C and λ per record;
(2) Gaussian mechanism with Mahalanobis budget 1/nm² (`noise_allocation.py:50-61`; Dong–Roth–Su 2019);
(3) Poisson amplification of that single Gaussian (`poisson.rs:27-29`, exact Gaussian path) or MF sensitivity
(`mf_gaussian_pld`, `_mf_gaussian.py:118-128`); (4) f̃ entering the loss is a *post-processed* DP output
(the only data-dependence of the surrogate on other examples is through the noised probe), so per-example
separability holds and no further cost accrues (post-processing; Dwork–Roth). Adjacency/unit: example-level,
add/remove, as everywhere in Opaque (`differential-privacy-review.md`, "Adjacency and protected unit").

---

## 9. Things that could NOT be verified / open questions

- PEFT `target_parameters` LoRA on stacked experts under `functional_call`+`vmap` — not executed.
- Whether `torch.compile` of the grad transform tolerates the hook list (fullgraph fallback exists,
  `_dp_trainer.py:4200-4228`) — not executed.
- Theorem numbering for the anisotropic-Gaussian/GDP fact (Dong–Roth–Su 2019) and for Feldman–Shenfeld
  (arXiv:2602.17284) quoted from the Rust doc comment — not re-fetched (alphaXiv MCP was unavailable this session).
- The ~3 % gap between composed-PLD Poisson and the exact Gaussian path (E1) is on the safe (upper-bound) side;
  its origin (double discretisation) inferred, not traced in Rust.
- How much the per-step MF noise on the load leaf is mitigated by EMA filtering (§7.4) — not quantified.
- The existing trainer logs un-noised `loss` / `loss_aux` means (`_dp_trainer.py:2258-2270`) — a pre-existing
  telemetry release outside the accountant; flagged, not investigated.
