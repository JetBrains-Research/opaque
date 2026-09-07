# Mellum2 under Opaque DP — final design specification (phase 2 synthesis)

Agent: `synthesizer`. Repo `/home/user/opaque` @ `ef1abc5` (branch `claude/mellum-dp-representation-r6slaz`), nothing
tracked modified. This document is self-contained: it takes the judges' winner (`design-minimal`: winner under the DP-correctness
and implementability lenses) as the spine, grafts everything the three judges listed from `faithful`, `optimal` and `skeptic`,
and corrects every error they found. Where the four designs disagreed and no judge settled it, §0.3 records the decision and
the reason.

Evidence tags. **VERIFIED** = I read the cited lines with `sed -n` or ran the cited script in this session; **VERIFIED (phase-1
X / judge-Y)** = established by the named earlier agent's executed script or primary-source read, which I did not repeat;
**PLAUSIBLE** = derived or read, not executed end to end. Repo paths are relative to `/home/user/opaque`; HF =
`.venv/lib/python3.11/site-packages/transformers/` (transformers 5.16.1). My own scripts and outputs:
`scratchpad/research/synthesizer/{final_table.py, nm_mf_probe.py, nm_mf_probe.out}`; I also re-ran
`scratchpad/research/judge-dp/check.py` and `scratchpad/research/judge-impl/mf_rownorm.py` (CPU, < 10 s each). Theorem
citations are those phase-1 `critic` extracted from the primary PDFs (critic R6); I did not re-fetch papers and cite nothing
beyond what phase 1 read.

---

## 0. The design in one page

### 0.1 Result

Mellum2's HF training objective has exactly one term that is not per-example separable: the Switch-style load-balancing loss
`L_aux(B) = E·Σ_e f_e(B)·P_e(B)` (VERIFIED, HF `modeling_mellum.py:540-606, 692-700`; Switch Transformer eqs. (4)–(6),
https://arxiv.org/abs/2101.03961 §2.2, VERIFIED phase-1 math). Because the load fraction `f` is argmax-derived, its gradient is
zero almost everywhere and

```
∇_θ L_aux(B) = Σ_{x∈B} ∇_θ S(x; f̃)|_{f̃ = f(B)},     S(x; f̃) = E · w_x · Σ_e f̃_e · P_e(x)        (exact; F3)
```

(VERIFIED phase-1 math §1.2 to 1e-17 in float64; phase-1 empirical E4 to 2–3e-7 in fp32 on a real forward). So the *only* thing
the DP path needs that a per-example pipeline does not have is a public constant `f̃_t` close to the batch load vector. The
design obtains it as a **DP release of the per-example centred load vector `d(x) = h(x) − k/E`**, carried as a second
`PerGroup` group of the *same* clipped pytree via a zero "probe" parameter, noised by the *same* Gaussian / matrix mechanism as
the gradient. Opaque's per-group allocation makes the joint release exactly one sensitivity-`1/nm` Gaussian, so **the
accountant call is literally unchanged under both stacks** (`poisson(gaussian(nm), q)*T` and
`b_min_sep(mf_gaussian(nm, band_mf_strategy(64, 0.95)), n_steps, p0)`), and the whole price is a `√(1+ρ)` inflation of the
gradient noise, where `ρ = C_h/C_g` is the share of the clipping budget given to the load group. At the default **ρ = 0.02 that
is ×1.010 gradient noise, ε unchanged**. The noised leaf is post-processed (sum-zero projection, EMA β = 0.99 with an exactly
known noise std, positive-part James–Stein shrinkage, clamp) into `f̃_{t+1}`; under band-MF the anti-correlated noise makes the
smoothed estimate ≈ 2.8× more accurate than under DP-SGD at equal base σ (factor 0.0249 vs 0.0709, VERIFIED). The same leaf
gives a free public **imbalance monitor** `D_t` with a decision rule, so the surrogate can be switched on mid-run with no
accountant or MF-latch change. Since `Σ_e P_e(x) = 1`, the surrogate gradient is `α·E·Σ_e (f̃_e − k/E)·∇P_e(x)`: it is
proportional to the *imbalance* and vanishes at balance — exactly what the real objective does — and the shrinkage guarantees
the design degrades to that zero, never to a random regulariser, when the true imbalance is below the noise floor.

### 0.2 What the four requirements of the brief get

| requirement | how it is met | status |
|---|---|---|
| (a) provably DP, accounting matches what runs | one joint Gaussian / MF release per step with per-group bounds `(C_g, C_h)`, structural (never active) bound on the load group, Mahalanobis allocation `Σ(C_i/σ_i)² = 1/nm²` (VERIFIED with the real allocator), Poisson subsampling of the *one* joint mechanism (shared coin), adaptive use of `f̃_t` covered by adaptive composition; accountant unchanged; hygiene table §8 closes every side channel the judges found | VERIFIED (mechanism), PLAUSIBLE only for "as run" until tests T7–T22 pass |
| (b) faithful to the real objective incl. the batch term | target = HF logical-batch formula (Fact A), attention mask, executed routes, α = `model.config.router_aux_loss_coef` by default with presets pinned to Mellum2's own SFT value 1e-4; batch mean of `∇ℓ_x` equals HF's `∇[CE + α·L_aux]` exactly at `f̃ = f(B)` and equal lengths | VERIFIED (identity), PLAUSIBLE (magnitude of `f̃ − f(B)` on real data, G3) |
| (c) usable clipping norm / noise | load leaf is its own group with a structural bound — `C_g` is chosen exactly as for any dense LoRA fine-tune; ρ = 0.02 costs ×1.010; the aux term moves per-example norms < 0.1 % and is 0 at balance | VERIFIED (toy), PLAUSIBLE (trained checkpoint, §4/§10) |
| (d) numerically stable vs the non-DP HF path | precision-matched oracle; vmap-vs-eager route flips are 0 at equal precision (VERIFIED toy, phase-1 F8 and skeptic's re-run); residual bf16 drift is accumulation order inside HF's own batched-vs-loop spread; fp32 router is an opt-in *pretraining-faithfulness* choice, not a drift fix | VERIFIED (toy), acceptance criteria §10 |

### 0.3 Decisions where the designs disagreed and the judges did not settle it

| question | minimal | faithful | optimal | skeptic | **decision** and why |
|---|---|---|---|---|---|
| released statistic | uncentred `s·h`, renormalise by noisy sum | centred `λ·d` | centred | uncentred + renormalise | **centred `λ·d(x)`** (all three judges graft it): 6.5 % smaller bound (2.646 vs 2.828), `Σ_e d = 0` structurally so no renormalisation and no `k·|B_t|/B̄` scale issue; skeptic's renormalisation-denominator noise is 4.4 % of k per step, not 0.55 % (judge error E3) |
| budget share ρ | 0.10 | 0.05 | 0.02 | 0.02 (monitor) | **0.02** default (×1.010); **0.10** recommended when the router or experts are trainable (×1.049). The cost table shows 0.02 already gives 0.83 %·(nm_MF/0.5622) smoothed error under band-MF and 2.35 % under DP-SGD, and the shrinkage handles the noise floor. |
| filter | EMA .95 (SGD) / window 256 (MF) | EMA β := workload momentum (0.95) | EMA .99 | EMA .99 / window 256 | **EMA β = 0.99 under both stacks**, window mean selectable. Under the preset's actual strategy (momentum 0.95) EMA .99 = 0.0249 and window-256 = 0.0198 — equivalent within 20 % at comparable lag (100 vs 128 steps); the EMA has a scalar exact-variance recursion (needed by the shrinkage), 64 floats of state instead of a 64 KB ring, and one code path for both stacks. faithful's β := 0.95 costs 3.3× accuracy for no privacy gain (judge-utility E7). |
| α default | `None` → config (1e-3); presets 1e-4 | config; presets 1e-4 | 1e-4 | 0 (regime A) | **`None` → `model.config.router_aux_loss_coef`** (1e-3, what the HF artefact declares) in `surrogate` mode; **presets pinned to 1e-4** (Mellum2's own SFT coefficient, TR §5.1.2, VERIFIED phase-1 literature F.2); α = 0 only in `monitor` mode. A silent 1e-4 default would hide a utility prior inside a faithfulness default (judge-utility graft 1); α = 0 does not meet the brief. |
| feature default | opt-in flag | on when α > 0 | on (in_stream) | off / monitor | **trainer default `off`** (existing runs bit-identical); **TRL converter maps `router_aux_loss_coef > 0` → `surrogate`** on a supported family (so HF-config users get the faithful objective without a new flag); **presets set `surrogate`** (SFT) / **`monitor`** (DPO, §7). |
| fp32 router | default on | default on | default on | opt-in | **opt-in, default off.** Skeptic's experiment (VERIFIED by judge-utility's re-run, and by my read of `a_out.json`) refutes the flip-removal rationale; what survives is pretraining faithfulness and tie-robustness, which is an informed user choice, not a default that moves the fine-tune away from the bf16 serving router. |
| AUTO-S with the load leaf | `ConfigurationError` | "simplest: reject" | engine `fixed_groups` | reject | **v1: `ConfigurationError` for `clipping_mode ∈ {auto, adaptive}`**; **v2: `fixed_groups` engine change** (§9.4). Fixed clipping stays the preset default (judge-utility). |
| `f̃` delivery into vmap | closure tensor | seeded batch column | seeded column | seeded column | **closure tensor** (minimal): the seeded column is pruned by `_remove_unused_columns` under plain `DPTrainer` with HF's default `remove_unused_columns=True` (VERIFIED `_dp_trainer.py:3446-3470`; the SFT/DPO configs only work because they set it `False`, `_sft_config.py:113`, `_dpo_config.py:139`). |
| checkpoint | sidecar | `extra_state` slot in `save_dp_runtime_state` | sidecar | `RuntimeCheckpoint` field | **sidecar `router_load_state.pt`** (fixed 23-parameter `save_dp_runtime_state` signature left alone, VERIFIED `_checkpoint.py:311-335`; the "single edit" claim for `RuntimeCheckpoint` is three edits, judge-impl E11). |
| statistics path | chunked-CE kwarg + HF `OutputRecorder` | same | same (extra output fields) | forward hooks | **chunked-CE kwarg + HF recorder** (memory-safe, recompute-safe by the ContextVar mechanism, VERIFIED `output_capturing.py:104-108, 259-272`); skeptic's hook route with overwrite-by-layer is the documented fallback. |
| DPO | out | in | in | monitor only | **mechanism in scope** (pooling rule §7, same helper called twice); **`mellum2-codesec` preset ships `monitor`** in v1 (TRL never added the aux to DPO; no DPO validation exists), `surrogate` is a flag. |
| second-moment streams | silent | exclude probe (T6) | silent | disallow | **v1: `ConfigurationError` with `second_moment=True`**; v1.1: exclusion by a zero second-moment bound on the probe group (§9.2). |
| independent forward-only Poisson draw | not considered | rejected for uniformity | opt-in optimum | rejected | **v2 opt-in, DP-SGD only**, with the serialization / rank-independence requirements written down (§2.7); `ConfigurationError` under any MF mechanism. |
| C-calibration pass | on training data | on training data | on training data | "is a private query" | **public proxy (KStack is public) or the accounted quantile release**; under MF prefer the public proxy (skeptic §4.2, grafted by all judges). |

---

## 1. Objective (G1): the exact per-example loss

### 1.1 Definitions

Example `x` (one collated row; right-padded — both repo collators right-pad, VERIFIED phase-1 critic R10), attention mask
`m_{x,t}`, `T_x = Σ_t m_{x,t}`, layers `l = 1..L` (L = 28), experts `e = 1..E` (E = 64), `k = 8`. For every layer the router
computes `z^l_{x,t} = W_l · h^l_{x,t}` (bf16 `F.linear` upstream), `p^l_{x,t} = softmax_fp32(z^l_{x,t})` over all `E` experts,
executed set `S^l_{x,t} = topk_k(p^l_{x,t})` (VERIFIED `modeling_mellum.py:332-341`).

```
P_e(x; θ) = (1/(L·T_x)) Σ_l Σ_t m_{x,t} · p^l_{x,t,e}              per-example mean router prob     (differentiable)
h_e(x; θ) = (1/(L·T_x)) Σ_l Σ_t m_{x,t} · 1{e ∈ S^l_{x,t}}         per-example load fraction        (piecewise constant, ∇ = 0 a.e.)
d(x)      = h(x) − (k/E)·1                                        centred load;  Σ_e d_e(x) = 0

ℓ_x(θ; f̃_t) = CE_x(θ) + α · E · Σ_e ( f̃_{t,e} − k/E ) · P_e(x; θ)   [ + ζ·Z_x(θ) ]   [ + ⟨z, λ·d(x).detach()⟩ ]
```

- `CE_x` = HF's per-example token-mean causal-LM CE over label-valid tokens, exactly what `DPTrainer.compute_per_example_loss`
  reads from `fmodel(params, **inputs)["loss"]` (`_dp_trainer.py:2314`, VERIFIED) through the chunked LM-head path
  (`chunked_linear_cross_entropy=2048`, `models/mellum.py:44`, VERIFIED).
- The bracketed probe term is the release carrier (§2): `z ∈ R^E` is a zero-valued parameter, so the term is identically 0 in
  value and contributes only `∂ℓ_x/∂z = λ·d(x)` (VERIFIED phase-1 primitives E2: other leaves' gradients unaffected).
- `Z_x = (1/(L·T_x)) Σ_{l,t} m_{x,t}·(logsumexp_e z^l_{x,t,e})²` is the router z-loss (ST-MoE §3.1 eq. (5),
  https://arxiv.org/abs/2202.08906, VERIFIED phase-1 literature C2), opt-in with `ζ = 0` by default (§7).
- Centred vs uncentred surrogate: `Σ_e P_e(x) = 1` ⇒ `Σ_e (k/E)·∇P_e = 0`, so `E·Σ_e f̃_e P_e` and `E·Σ_e (f̃_e − k/E) P_e` have
  **identical gradients** (VERIFIED phase-1 math §1.1/empirical E4: uniform `f̃` gives `‖∇‖ ≈ 3e-8`); the centred form differs
  from HF's loss *value* by the constant `α·k` (0.0008 nats at α = 1e-4) and makes "the signal is the imbalance" visible in code.

### 1.2 Faithfulness claim (exact)

With `f̃_t = f(B_t)` and equal lengths (`T_x ≡ T`, true for the packed T = 1024 presets),

```
(1/B) Σ_{x∈B_t} ∇ℓ_x  =  ∇[ CE_tokenmean(B_t) + α · L_aux^HF(B_t) ]        (L_aux^HF = load_balancing_loss_func, modeling_mellum.py:540-606)
```

VERIFIED phase-1 math §1.2 (float64: 8e-18 full mask) and empirical E4 (fp32 real forward: 2.4e-7, incl. ragged lengths with
`w_x = T_x/T_tot` reweighting). With a lagged DP estimate `f̃_t` the aux gradient is the exact gradient of the same functional form
at a nearby load vector; §2.6 prices "nearby".

### 1.3 Target of faithfulness and why (the four candidates of critic G1)

| candidate | verdict |
|---|---|
| **(a) HF logical-batch pooled formula** — one denominator `L·T_tot` across all layers, `Σ_e f_e = k`, attention mask (VERIFIED `modeling_mellum.py:575-603, 694-697`) | **TARGET.** It is the objective the released checkpoint declares (`router_aux_loss_coef = 0.001`, `configuration_mellum.py:103`, VERIFIED), the only one reproducible from the HF artefact, and the cheapest to release (64-dim, bound 2.646). |
| (b) HF-Trainer-realised: per-*microbatch* `f`, effective coefficient `G·α` (VERIFIED phase-1 critic Exp A, rel-L2 0.0) | rejected — an artefact of `trainer.py:1961-1963` (the aux is added per forward call and never divided by the accumulation count when the model accepts loss kwargs); at the preset (B = 256, mb = 8) it would mean `f` over 8 sequences and `α_eff = 0.032`. The DP path has no accumulation (microbatches are vmap chunks of one logical batch), so it is *more* faithful to (a) than HF Trainer itself. |
| (c) Megatron per-layer running-average `f` (what pretraining saw; Mellum2 TR §3.6, arXiv:2605.31268, VERIFIED phase-1 literature F.2) | rejected as *target*: not reproducible from HF (mean of per-layer products vs product of pooled means) and a per-layer release costs ×√L = ×5.29 relative noise at equal privacy (bound `√(kL(1−k/E)) = 14.0` vs 2.646; VERIFIED phase-1 math §3). Its **lagged running-average character is adopted**: `f̃_t` is a filtered stream of releases, the same kind of estimate pretraining used. Per-layer is an opt-in row of the cost table. |
| (d) per-sequence aux (`f(x)` from the example's own tokens; DeepSeek-V2/V3, Wang et al.) | rejected as a stand-in: a materially different regulariser (cos 0.26–0.39 to the batch gradient at balance, anti-specialisation closed form `E·P̄_{S_x}`; VERIFIED phase-1 math §2 / empirical E4). Offered as an explicit zero-cost option flag, never labelled "faithful". |

### 1.4 The remaining G1 sub-decisions

| item | decision | reason |
|---|---|---|
| mask for `h`, `P` | attention mask | HF passes `attention_mask` to the aux (`modeling_mellum.py:694-697`, VERIFIED); prompt tokens with label −100 were routed and count. CE keeps the label mask. |
| CE weighting | equal example weights | Opaque's convention (pre-clip division by the example's own token count, `opaque-alignment/.../sft/loss/_nll.py:15-24`, VERIFIED phase-1 math); coincides with HF token weighting iff `T_x ≡ T` (F4). A public-`N̄` token-weighted variant is representable (phase-1 divergence §4) but not built. The released statistic is therefore the *example-mean* load `(1/B̄)Σ_x d(x)`. |
| coefficient | `router_aux_loss_coef: float \| None = None` → `model.config.router_aux_loss_coef` (1e-3) in `surrogate` mode; both Mellum2 presets pin **1e-4** | see §0.3; value at balance `α·k` = 0.0008 nats (1e-4) / 0.008 (1e-3). |
| `f` from executed routes | yes: `h(x)` is computed from the *same* fp32 softmax tensor and the *same* `topk` op the router ran (`modeling_mellum.py:335-336`) | HF's aux recomputes top-k from a softmax in the *logits dtype* (bf16 for a bf16 model, `:584`), so 0.03–1.7 % of tokens get a different aux top-k than the executed one (VERIFIED phase-1 critic R9); the deviation from HF's *number* is ≤ 1e-3·k/E, 10× below the smallest noise row of §2.6. "The load that ran" is the only definition consistent with the sensitivity bound `Σ_e h_e = k` exactly. |
| pooled vs per-layer | pooled `E`-vector (HF) | per-layer opt-in priced ×5.29 (§2.6). |
| initial value | `d̃_0 = 0` ⇒ `f̃_0 = k/E` | aux gradient identically zero at step 0 — exactly the real objective at balance — and switches on as releases arrive. |
| z-loss | opt-in `ζ = 0` | not part of HF's objective; per-token separable, zero DP cost; part of Mellum2's pretraining objective (1e-3); only meaningful with a trainable router. |

---

## 2. Mechanism and accountant (G4)

### 2.1 Adjacency, protected unit, sensitivities

Protected unit = one training example (one packed sequence; for DPO one preference pair). Adjacency = **add/remove**, the repo
default (`.junie/differential-privacy-review.md` "Adjacency"; `clipped_grad` contract `_clipped_grad.py:135-145`, VERIFIED:
"sensitivity … is guaranteed to be `clipping_norm` … Under replace-one DP, the sensitivity is doubled"). Divisor = the public
expected batch size `B̄ = a.train_batch_size` (`normalize_by=expected_batch_size`, `_dp_trainer.py:1394, 4266-4290`, VERIFIED),
never the realised batch.

Constants: gradient bound `C_g` (preset 0.9, `examples/train_dpftrl.py:611-612`, VERIFIED), ratio `ρ = C_h/C_g` (default 0.02),
structural load bound `Δ_h = √(k(1−k/E)) = 2.6458`, probe scale `λ = ρ·C_g/Δ_h` (= 0.006803 at the preset), group bound
`C_h = λ·Δ_h·(1+1e-6)` (= 0.0180; the `(1+1e-6)` guard keeps the bound used by accounting slightly *above* the true sup so the
ULP-guarded clip never fires on round-off), noise multiplier `nm`.

Per-record L2 bounds (both hold for **every** prefix of previous outputs because neither depends on `f̃_t` — the condition
adaptive composition needs):

| group | bound | why |
|---|---|---|
| gradient leaves (`fallback`) | `C_g` | clipping `min(1, C_g/‖g‖)` per group (`_pytree.py:439-477`, VERIFIED) |
| load leaf `λ·d(x)` (`router_load_probe`) | `C_h = λΔ_h` | `0 ≤ h_e ≤ 1`, `Σ_e h_e = k` ⇒ `‖h‖² ≤ max_e h_e·Σ_e h_e ≤ k`; centred `‖h − k/E‖² = ‖h‖² − k²/E ≤ k(1−k/E) = 7` (VERIFIED phase-1 math §3; attained when every token in every layer routes to the same 8 experts). **Structural — the `PerGroup` entry is a bound, never an active clip — the release is unbiased** and `clip_rate` on that group is 0 by construction. |

Replace-one: the gradient bound doubles to `2C_g`. For the load group the tight replace-one bound is `λ·√(2k) = 4.0·λ` (both
histograms in `[0,1]^E` with sum `k` ⇒ `Σ(h_e−h'_e)² ≤ max|·|·Σ|·| ≤ 2k`; centring cancels), tighter than the generic doubling
`2Δ_h = 5.29` (judge-dp, optional tightening; VERIFIED arithmetic). Nothing in code changes either way — Opaque documents the
add/remove bound and the doubling.

### 2.2 Per-step mechanism — DP-SGD / Poisson

Public state entering step `t`: `θ_t`, `d̃_t ∈ R^E` (smoothed imbalance), `s_t` (its known noise std), `f̃_t = clamp(k/E + shrink(d̃_t), 0, 1)`.

1. **Sample** `B_t ~ Poisson(q)`, `q = B̄/N` (existing sampler).
2. **Augment** (once per step, outside vmap, `_augment_inputs`, `_dp_trainer.py:2302-2313`, VERIFIED): copy `f̃_t` into the loss
   closure's device tensor; assert `ctx.trainable_params["router_load_probe"]` is zero.
3. **Per example inside `vmap(grad)`**: forward with `opaque_router_logits=True` (§9.1) → `CE_x` and the tuple of 28 fp32 router
   logits `(T, E)`; `p`, executed `S`, `P(x)`, `h(x)`, `d(x)`; loss `ℓ_x` as in §1.1 with `f̃_t` and the probe term.
   `∂ℓ_x/∂z = λ·d(x)`; `∂ℓ_x/∂θ` = CE + surrogate (+ z-loss).
4. **Per-group clip** (`per_group(trainable, router_load_probe=C_h, fallback=C_g)`, `_per_group.py:44-75`, VERIFIED): gradient
   group scaled by `min(1, C_g/‖g_x‖)`; probe group never rescaled (structural). Sum over `B_t`, divide by `B̄`
   (`clipped_grad(..., normalize_by=B̄)`; stored bounds `C_g/B̄`, `C_h/B̄`).
5. **Noise** (`gaussian_noise` → `per_group_noise_stddev` for a `PerGroup` max_norm, `_gaussian.py:320-322` and
   `noise_allocation.py:103-110`, VERIFIED): `σ_i = nm·√(B_i·Σ_j B_j)`, i.e. with `S = C_g + C_h`

   ```
   σ_g = nm·√(C_g·S)/B̄   on every gradient leaf          (= nm·C_g·√(1+ρ)/B̄)
   σ_h = nm·√(C_h·S)/B̄   on the probe leaf               (= nm·C_h·√(1+1/ρ)/B̄)
   Mahalanobis check:  (C_g/B̄)²/σ_g² + (C_h/B̄)²/σ_h² = 1/nm²   with equality
   ```
   VERIFIED by running the real function this session (`check.py` re-run: `chk = 1.0000` at ρ = 0.02/0.05/0.1;
   `σ_g/(nm·C_g) = 1.0100 / 1.0247 / 1.0488`).
6. **Release** = the noised pytree `(ĝ_t, ŷ_t)` — the only DP output of the step. Under DDP the probe leaf is all-reduced
   pre-noise with every other leaf (`sum_gradients_`, `_dp_trainer.py:2170-2173`) and the noise key is shared across ranks
   (`_dp_trainer.py:1478`, VERIFIED), so `ŷ_t` is bit-identical on all ranks.
7. **Post-processing** (§2.4) → `d̃_{t+1}, s_{t+1}, f̃_{t+1}, D_{t+1}`; the probe entry of the noised pytree is zeroed in place after
   reading so the optimizer's update for `z` is exactly 0 (zero moments, zero weight decay on a zero parameter).
8. **Accountant: unchanged** — `poisson(gaussian(nm), q) * T` exactly as `_build_mechanism` builds it today
   (`_dp_trainer.py:4308-4392`, VERIFIED; `num_groups` is consulted only by the adaclip wrapper, which this design excludes).

**Why step 8 is correct (the chain).** The step releases one Gaussian on the concatenation `[Σ_x clip g_x ; λ Σ_x d(x)]` with
diagonal covariance. Whitening by `Σ^{-1/2}` is a bijection, so the mechanism and its whitened form have identical privacy; the
whitened statistic has add/remove L2 sensitivity `√(C_g²/σ_g² + C_h²/σ_h²) = 1/nm` (attained by an example saturating both
bounds), i.e. it is a sensitivity-1 Gaussian at multiplier `nm` — Dong–Roth–Su Thm 2.7 (https://arxiv.org/abs/1905.02383) for the
Gaussian, Zhu–Dong–Wang Def. 7 (https://arxiv.org/abs/2106.08567) for its dominating pair (both VERIFIED phase-1 critic R6 from
the PDFs). Precedent: Andrew et al. 2021 Thm 1 (https://arxiv.org/abs/1905.03871, VERIFIED phase-1 literature B1) — the
clipped-count bit of adaptive clipping is this same joint Gaussian with `z_Δ = (z⁻² − (2σ_b)⁻²)^{-1/2}`. Poisson subsampling then
applies to the **one** joint mechanism, because both releases share the sampling coin: Feldman–Shenfeld Lemma 3.2 / Thm 3.3
(https://arxiv.org/abs/2602.17284) as implemented in `src/amplification/poisson.rs:15-42` (VERIFIED read: doc comment cites
Thm 3.3 / Alg. 8–9; exact Gaussian fast path at `:27-29` when the base is a pure Gaussian). "Compose two subsampled Gaussians"
would be the *wrong* description (the second release's subsampling is not fresh); the joint-vector view is the right one
(phase-1 math §4(c)). `f̃_t = φ(y_{<t})` is a function of previous outputs — the same kind of dependence `ℓ_x` already has on
`θ_t` — and Zhu–Dong–Wang Thm 10 (adaptive composition of dominating pairs, VERIFIED critic R6) charges nothing for it. The naive
"σ·C_g on the gradient, σ·C_h on the load" allocation is `gaussian(nm/√2)`, **not** `gaussian(nm)` (phase-1 math 4(b) REFUTED row);
the design never uses it.

Empty Poisson batch: `clipped_grad` short-circuits with zero grads (`_clipped_grad.py:220-277`, VERIFIED phase-1 primitives);
the probe leaf is then `0 + noise`, a valid release of the empty sum; post-processing consumes it unchanged.

### 2.3 Per-step mechanism — DP-FTRL / band-MF / b-min-sep (the actual `mellum2-kstack` preset)

Steps 1–4 as above with `B_t` from `BMinSepSampler` (`dpftrl/sampling/_b_min_sep.py:30-`; per-iteration `p = p₀/(1−p₀(b−1))` so
`E|B_t| ≈ B̄` is constant, `:6-11`, VERIFIED read) — the load statistic is computed on the **same** batch (no second draw). Then:

5'. **Noise** (`mf_gaussian_noise`, `_mf_gaussian_noise.py:163-192`, VERIFIED): the constant-max_norm latch
   `_validate_constant_max_norm` (`_engine.py:473-517`, VERIFIED: compares the whole `max_norm`, `PerGroup` included, by
   equality) accepts the constant two-group `PerGroup`; `base_stddev = per_group_noise_stddev(max_norm, nm)` (`:166-167`); the
   correlated noise `C⁻¹Z` is applied leaf-wise to the whole pytree, probe leaf included; realised per-step σ on every leaf
   `= base·‖row_t(C⁻¹)‖` (`:186-188`). Under the preset strategy `band_mf_strategy(bands=64, momentum=0.95)` the single-step
   factor is **1.431**, stationary from `t ≈ 7` (VERIFIED, `mf_rownorm.py` re-run: 1.4309 at n−1, 1.2833 at t = 0).
6'. **Release** = the noised stream row `(ĝ_t, ŷ_t)`.
7'. **Post-processing** as §2.4, with the filter factors computed from the instantiated strategy (§6).
8'. **Accountant: unchanged** — `b_min_sep(mf_gaussian(nm, band_mf_strategy(bands=64, momentum=0.95, lr_schedule)), n_steps=T, p0=q)`
   as `build_amplifier_factory` constructs it (`trainer/_dpftrl.py:150-158`, VERIFIED; `_mf_gaussian.py:118-128` folds
   `strategy.sensitivity(n_steps, min_sep, max_participations)` into the multiplier, VERIFIED).

Correctness: Denisov et al. 2022 Thm 2.1 (https://arxiv.org/abs/2202.08312, VERIFIED critic R6: "… satisfies the same DP
guarantee … even when the rows of the input are chosen adaptively") applied to the per-group-whitened stream. The
participation-pattern sensitivity is homogeneous of degree 1 in the row bound, so for a shared participation pattern π
`Σ_g ‖C(G_g−H_g)‖²_F/σ_g² ≤ s(π)²·Σ_g C_g²/(nm²·C_g·S) = s(π)²/nm²`, and the sup over π is `sens(C)²/nm²` — the same scalar PLD
(phase-1 math §5(i)). The argument needs the pattern to be shared across groups, which it is (same example, same step), so the
gradient's `min_sep` / `max_participations` apply to the probe group. The trainer calibrates `nm` from the target ε
(`_calibrate_noise`), so ε is *held* automatically and the entire price shows up as `×√(1+ρ)` on the gradient noise.

Excluded under MF: adaptive clipping (drifting bound; rejected by the latch anyway); an independently sampled load release (a
different participation model — it would need concurrent composition, Vadhan–Wang / Vadhan–Zhang, theorem numbers **not
verified** in phase 1, PLAUSIBLE only; §2.7 keeps it DP-SGD-only). AUTO-S on the gradient group is MF-admissible but v1 rejects
it with the feature because `_auto_scale_per_group` applies `R/(‖·‖+γ)` to *every* group with `clamp_to_one=False`
(`_pytree.py:313-340`, VERIFIED), which would rescale every `d(x)` to norm ≈ `C_h` — a mean *unit direction*, a different
statistic (privacy-valid, biased).

### 2.4 Post-processing (all public, free) — producing `f̃_{t+1}` and the monitor

Given the noised probe leaf `ŷ_t = noisy_grads.pytree["router_load_probe"]` (already `/B̄`, rank-identical):

```
1. d̂_t   = ŷ_t / λ                                  # (1/B̄) Σ_{x∈B_t} d(x) + N(0, (σ_h/λ)² I)
2. d̂_t  ← d̂_t − mean_e(d̂_t)                         # sum-zero projection (signal already sums to 0; noise variance ×63/64)
3. d̃_{t+1} = β·d̃_t + (1−β)·d̂_t,   β = 0.99          # EMA (window mean W selectable, §6)
   s_{t+1} = (σ_h/λ) · φ_{t+1}                        # known noise std of d̃: φ from the exact variance recursion (DP-SGD:
                                                      #   φ² ← β²φ² + (1−β)²·63/64, stationary √((1−β)/(1+β)) = 0.0709) or
                                                      #   from ‖row_t(F·C⁻¹)‖ of the instantiated MF strategy (stationary 0.0249)
4. d̃⁺     = d̃_{t+1} · max(0, 1 − (E−1)·s_{t+1}² / ‖d̃_{t+1}‖²)   # positive-part James–Stein shrinkage toward balance (optional, default on)
5. f̃_{t+1} = clamp(k/E + d̃⁺, 0, 1)                   # clamp inactive unless |d̃⁺_e| > k/E, i.e. ≥ 40σ of the smoothed noise
6. D_{t+1} = max_e |d̃_{t+1,e}| / (k/E)               # public imbalance monitor (pre-shrinkage), logged every step
```

- Why the sum-zero projection: `Σ_e d(x) = 0` for every example, so the signal lives in the sum-zero subspace and the noise's
  mean component is pure noise (variance ×63/64, VERIFIED arithmetic; trivial but free). This replaces the noisy-sum
  renormalisation of minimal/skeptic — whose denominator noise is `√E·σ_h/(λ·B̄)` = 0.355 = **4.4 % of k per step** at ρ = 0.02
  (judge error E3; not 0.55 %) — and the `k·|B_t|/B̄` realised-batch scale issue (critic M11): `E[Σ_x d(x)] = B̄·E_x[d(x)]` under
  Poisson and `E|B_t| ≈ B̄` under b-min-sep, so `d̂_t` is unbiased for the population centred load and the realised `|B_t|/B̄`
  (rel. sd 6 %) multiplies the *signal* only.
- Why shrinkage: the surrogate gradient is `α·E·Σ_e (f̃_e − k/E)·∇P_e(x)`; without shrinkage a noise-dominated `d̃` injects a
  *random* 64-dim regularisation direction of norm `≈ s√E` every step — small in absolute terms at α = 1e-4 but a systematic
  deviation from the real objective's behaviour at balance (gradient exactly 0). The positive-part James–Stein rule is the
  identity when `‖d̃‖ ≫ s√(E−1)` and zero when the smoothed estimate is inside its own noise ball. **Threshold, stated in the
  right units** (judge-utility/impl E3 corrected optimal's √E slip): it engages when the *per-coordinate RMS* imbalance
  `δ·k/E` satisfies `δ < s·√((E−1)/E)/(k/E)`, i.e. at ρ = 0.02 with EMA .99: **δ < 0.023 (DP-SGD) / 0.008 (band-MF at
  nm = 0.5622)** — a well-balanced checkpoint (δ ≈ 0.1) is far above it, so the rule is conservative (VERIFIED arithmetic,
  `final_table.py`). Optional: `router_load_shrink=False` gives the plain estimate.
- Why no renormalisation to `Σ = k` on the loss path: `f̃` enters linearly; `Σ_e f̃_e = k` holds up to the clamp automatically.
- Why clamp: a negative `f̃_e` is physically meaningless; the clamp is inactive unless an expert is dead/hot (`|d̃_e| > 0.125`,
  ≥ 40σ of the ρ = 0.02 smoothed noise), so it introduces no bias in practice (VERIFIED arithmetic).
- **Monitor and decision rule** (skeptic §2.3, grafted by all judges): trip when `D_t > τ = 0.5` (some expert carries ≥ 1.5× or
  ≤ 0.5× its share) on two consecutive logged evaluations. At ρ = 0.02 the smoothed per-entry noise is 2.35 % of `k/E` (DP-SGD)
  / 0.83 % (band-MF, at nm = 0.5622), so τ = 0.5 is **21σ / 60σ** (VERIFIED arithmetic): false-alarm and miss probabilities are
  negligible (skeptic's tail arithmetic, VERIFIED by judge-dp; its garbled rationale sentence is dropped — τ is chosen for
  detectability, validation §10 tunes it). Responses on trip, in order of simplicity: (i) log and continue (`monitor`),
  (ii) switch α from 0 to the configured value (`monitor_then_surrogate`) — **without touching clipping, noise, the accountant or
  the MF latch**, since the leaf, its bound and ρ are already in place and changing the loss is an adaptive-row change
  (Denisov Thm 2.1 / ZDW Thm 10), (iii) stop. `D_t`, `f̃` min/max/entropy, the shrink factor and the trip flag are logged as
  public curves.
- Lagged use is free: adaptive composition (ZDW Thm 10, VERIFIED critic R6); the dependence of `ℓ_x` on `f̃_t` is the same kind as
  on `θ_t` (phase-1 math §4(d)). A same-step variant (forward-only routing pass, release, then gradient pass) is a second
  mechanism per step and doubles forward cost; not adopted.

### 2.5 Modes and defaults

`router_load_release: Literal["off", "monitor", "surrogate", "monitor_then_surrogate"] = "off"`; `router_load_ratio = 0.02` (ρ);
`router_aux_loss_coef = None` (→ config in surrogate modes; 0 in `monitor`); `router_load_filter = {"kind": "ema", "beta": 0.99}`
(or `{"kind": "window", "steps": 256}`); `router_load_shrink = True`; `router_load_trip = 0.5`; `router_load_per_layer = False`;
`router_z_loss_coef = 0.0`; `router_fp32 = False`. Presets: `mellum2-kstack` → `surrogate`, α = 1e-4, ρ = 0.02;
`mellum2-codesec` (DPO) → `monitor`, ρ = 0.02 (§7). Constraints enforced at setup with `ConfigurationError`: any non-`off` mode
requires `clipping_mode == "fixed"`, `second_moment == False`, and a family whose backbone can record `router_logits`.

### 2.6 THE cost table — preset regime `nm = 0.5622, B̄ = 256, k = 8, E = 64, C_g = 0.9, L = 28, q = 256/5e5, T = 15625, δ = 1e-6`

All ρ rows computed this session with the real `per_group_noise_stddev` and the real accountant (`final_table.py`; baseline
`poisson(gaussian(0.5622), q)*T` → **ε = 3.0004**). Single-release per-entry noise on `d̂` in units of `k/E = 0.125`:
`r₁ = nm·Δ_h·√(1+1/ρ)/(B̄·k/E) = 0.04648·√(1+1/ρ)`. Filter factors (noise std of the filtered stream / noise std of one
release): DP-SGD EMA `√((1−β)/(1+β))` = 0.1601 (β = .95) / **0.0709** (β = .99), window-256 `1/√256` = 0.0625; band-MF
(`bands=64, momentum=0.95`, exact `‖row_t(F·C⁻¹)‖`, VERIFIED `mf_rownorm.py` re-run) single step **1.431**, EMA .95 0.0824,
EMA .99 **0.0249**, window-256 0.0198. ε column = `poisson(gaussian(nm) | gaussian(c·nm), q)*T`, `epsilon_at(1e-6)`.

| ρ = C_h/C_g | c = √(1+1/ρ) | **gradient-noise inflation √(1+ρ)** (ε held at 3.00, accountant unchanged) | ε **if nm were held** instead (same family, "pay in ε") | r₁ single release | DP-SGD after EMA .99 / window 256 | band-MF single step (×1.431) | band-MF after EMA .99 / window 256 † | per-layer opt-in (×5.29) after EMA .99, SGD / MF † | shrinkage engages below per-coordinate δ, SGD / MF † |
|---|---|---|---|---|---|---|---|---|---|
| 0.50 | 1.732 | ×1.225 | 5.443 | 8.1 % | 0.57 % / 0.50 % | 11.5 % | 0.20 % / 0.16 % | 3.0 % / 1.1 % | 0.006 / 0.002 |
| 0.20 | 2.449 | ×1.095 | 4.219 | 11.4 % | 0.81 % / 0.71 % | 16.3 % | 0.28 % / 0.23 % | 4.3 % / 1.5 % | 0.008 / 0.003 |
| **0.10** (router/experts trainable) | 3.317 | **×1.049** | 3.703 | 15.4 % | 1.09 % / 0.96 % | 22.1 % | **0.38 %** / 0.31 % | 5.8 % / 2.0 % | 0.011 / 0.004 |
| 0.05 | 4.583 | ×1.025 | 3.417 | 21.3 % | 1.51 % / 1.33 % | 30.5 % | 0.53 % / 0.42 % | 8.0 % / 2.8 % | 0.015 / 0.005 |
| **0.02 (default)** | 7.141 | **×1.010** | 3.234 | 33.2 % | **2.35 %** / 2.07 % | 47.5 % | **0.83 %** / 0.66 % | 12.5 % / 4.4 % | **0.023 / 0.008** |
| 0.01 | 10.05 | ×1.005 | 3.172 | 46.7 % | 3.31 % / 2.92 % | 66.8 % | 1.16 % / 0.92 % | 17.5 % / 6.2 % | 0.033 / 0.012 |
| *v2 opt-in, DP-SGD only: independent forward-only draw, c = 2, every m = 4 steps* (§2.7) | — | ×1.0002 (ε-matched) | **3.003** | 9.3 % | 2.1 % (EMA .9 over releases, lag 40 steps) | n/a | n/a | 11 % | — |
| *independent draw, c = 1, m = 4* | — | ×1.013 | 3.128 | 4.6 % | 1.1 % | n/a | n/a | 5.6 % | — |
| *independent draw, c = 2, m = 16* | — | ×1.0001 | 3.001 | 9.3 % | 2.1 % (lag 160) | n/a | n/a | 11 % | — |

† **nm_MF caveat (minimal §2.5, grafted by every judge).** The band-MF columns are evaluated at `nm = 0.5622`, the DP-SGD/Poisson
ε = 3 calibration. The preset calibrates its own `nm_MF` for `b_min_sep(mf_gaussian(·, band_mf(64, 0.95)))` at ε = 3, which is
**not** 0.5622; every absolute band-MF load-error number scales by `nm_MF/0.5622` (PLAUSIBLE: several × larger — with up to
8 participations the un-amplified MF sensitivity exceeds 1). Nobody in phase 2 computed `nm_MF` (one MC-PLD evaluation at
n = 15625 exceeded 170 s CPU in minimal's run; my own bounded probe `nm_mf_probe.py` is reported in §6.4). The *ratio* claims
— `×√(1+ρ)` gradient price, the 2.8× filter advantage of band-MF over DP-SGD at equal base σ, accountant unchanged — are
unaffected. The two rightmost DP-SGD columns are the *same mechanism family* re-parametrised (critic C7): a reader may choose
"pay in gradient σ" (shipped) or "pay in ε".

**Reading the table against the objective.** The surrogate gradient is `α·E·Σ_e (f̃_e − k/E)·∇P_e(x)`, so the error that matters
is `‖noise‖/‖d_true‖`, not `r` against `k/E`. With an unknown real per-coordinate RMS imbalance `δ·k/E`, the relative aux-gradient
error at the default row is ≈ `0.0083/δ` (band-MF, nm = 0.5622) or `0.0235/δ` (DP-SGD): 8 % / 24 % at δ = 0.1 (a well-balanced
checkpoint), 3 % / 8 % at δ = 0.3. Below δ ≈ 0.008 / 0.023 the shrinkage zeroes the term — exactly where the true objective's
gradient is negligible too. Signal-to-noise on real data (G3, §10) decides whether the term carries signal; if it does not, the
design degrades to what the real objective does at balance, never to a different regulariser. (Phase-1 empirical E4:
relative error of a noised `f̃` grows linearly with `‖noise‖/‖f(B)−k/E‖`, 3.3× more tolerant under induced imbalance.)

Rejected levers, for the record (optimal §2.5, VERIFIED arithmetic where numbers are given): randomised response on `sign(d)` —
one example can flip every near-zero coordinate, per-step ε = `64·ln((1−p)/p)` = 12.8 at p = .45 / 70.3 at p = .25, dominated;
independent draw with a 4× batch (`q₂ = 4q, c = 1`) → ε 5.30, amplification lost; amortised same-batch release every m steps —
same budget in bursts, no gain over small ρ every step; per-sequence aux — different regulariser; public-data `f̃` (Davody et
al. 2020, https://arxiv.org/abs/2006.10919; Ponomareva et al. "DP-fy" §5(b) option (a), both VERIFIED phase-1 literature B2/B3) —
zero cost but estimates the public distribution's imbalance; opt-in when a public corpus exists, never labelled faithful.

### 2.7 Appendix (v2, opt-in, DP-SGD only): the independent forward-only Poisson release

Every `m` steps draw a **second, independent** Poisson sample `B'_t ~ Poisson(q₂)` over the dataset, run a forward-only vmap of
the routing statistic through `clipped_fun` (`_clipped_fun.py:492`, `clipping_norm = Δ_h`, `normalize_by = q₂N`), add
`N(0, (c·nm·Δ_h/(q₂N))²)` from a separately rooted key, and post-process as §2.4 with the EMA running over releases. The
gradient mechanism is untouched (`C_g`, `nm`). Accountant:

```
dpsgd.poisson(dpsgd.gaussian(nm), q) * T  |  dpsgd.poisson(dpsgd.gaussian(c·nm), q₂) * (T/m)
```

Validity: two mechanisms with **fresh** sampling coins and fresh noise, adaptively interleaved — ZDW Thm 10 with each factor
dominated via Feldman–Shenfeld Lemma 3.2; `Poisson` accepts any `DpProcess` inner (`dpsgd/amplification/_poisson.py:25-45`,
VERIFIED) and `|` composes at the process level. Numbers VERIFIED this session (`check.py` re-run): c = 2, m = 4 → **ε 3.0030**;
c = 1, m = 4 → 3.1284; c = 2, m = 16 → 3.0011. Why it is cheaper: two small-`q` draws compose almost additively in the GDP regime,
whereas one draw at `σ_eff = nm/√(1+1/c²)` sits on the strongly convex part of `ε(1/σ)` (phase-1 primitives E1(g)). Cost: one
forward over `q₂N ≈ 256` examples every 4 steps ≈ +8 % step time with grouped MoE (PLAUSIBLE ratio). **Conditions before
"provably DP as run" holds for this variant (judge-dp E7):** (i) the second `PoissonSampler`'s key is rank-domain-separated
(`fold_in(key, "opaque.moe.load_release", rank)`) so per-example inclusion coins are independent across ranks — the mixture
decomposition in the amplification proof conditions on the sampled subset; (ii) the sampler's key/position and the second noise
key's counter are serialised in the checkpoint sidecar; (iii) each rank samples its shard at `q₂` with `N` the global size and
`reduce_pytree_` sums before noise; (iv) `ConfigurationError` when `mechanism_kind != "gaussian"` (no second draw exists under
b-min-sep / balls-in-bins, and the interleaving would need concurrent composition). Not part of v1.

---

## 3. Router precision and routing (G5)

**Decision: routes are computed per example inside vmap from the current model (no pinning); `f` is defined from those executed
routes; the router runs in the stock HF precision by default; an fp32-logit router patch is shipped as an opt-in documented as
pretraining-faithful and tie-robust — NOT as a drift or flip fix.**

- **The refutation that settles it** (skeptic `a_fp32_router_flips.py`; output `a_out.json` read by me, script re-run by
  judge-utility with identical numbers; VERIFIED at toy scale): on a bf16 tiny Mellum vs the fp32 model, the stock bf16 router
  flips 46/54 (random) and 46/47 (structured) top-8 sets per 1024 rows per layer; the fp32-logit router (`F.linear(h.float(),
  W.float())`, fp32 softmax) flips 51/56 and 39/43. The flips originate in the **bf16 hidden states** entering the router (bf16
  has fixed *relative* resolution, so ≈ 1 % of tokens per layer sit inside its rounding of a top-k tie regardless of router
  sharpness — phase-1 empirical E2), not in the rounding of the logits. Phase-1 E1b's 12.9 %/11.1 % → 1.7 %/1.5 % router/expert
  gradient-error reduction came from pinning to the fp32 *forward's* routes, which no in-vmap router precision change can
  reproduce. And the pair that matters for requirement (d) — Opaque bf16 `vmap(grad)` vs HF bf16 eager at equal precision — has
  **0/1024 flips** in both layers (skeptic re-run) and 0/256 and 0/2048 in phase-1 E1/E1b (F8). Therefore "fp32 router removes
  the bf16 tie flips and the 8× excess gradient error" (faithful §3, minimal §3, optimal §3) is withdrawn.
- **What the fp32 router does buy** (opt-in `router_fp32=True`): it is the router Mellum2 was pretrained with (TR appendix
  "router precision FP32", all aux computations FP32; VERIFIED phase-1 literature F.2); exact bf16 ties in `router_logits`
  disappear (ties resolved at fp32 resolution given the same bf16 input); HF's bf16-softmax aux top-k and the executed top-k
  coincide (critic R9). Cost: one `64×2304` fp32 GEMM per token per layer = 147 k MAC vs 49.5 M MAC of routed expert compute —
  **0.30 %** (VERIFIED arithmetic; skeptic's 0.15 % is a 2× slip). Patch: `MellumTopKRouter.forward` replacement returning
  `(logits_fp32, scores.to(h.dtype), indices)`; downstream consumers use only `scores`/`indices`
  (`MellumSparseMoeBlock.forward`, `modeling_mellum.py:350-355`, VERIFIED phase-1 primitives). Default **off**: for an
  attention-only LoRA served through the HF bf16 router, training under bf16 routes adapts the adapter to the serving routes;
  the Mellum2 authors document train/inference route disagreement on this checkpoint either way (TR §5.2).
- **Pinning**: routes are already per-example constants of `(x, θ_t)`; no cross-example dependence, no DP consequence (phase-1
  math §6: clipping bounds the gradient for any routing decision; continuity is irrelevant to privacy). Frozen-base pinning is
  rejected — it freezes training-time routing while inference routing (whose inputs move through attention LoRA) drifts.
  Pinning from an fp32 routing-only forward remains a *validated-later* option for router/expert training (E1b), costing a
  full fp32 forward per step; not built.
- **`f` from executed routes**: the statistics code calls `topk(softmax(router_logits))` on the same tensor the router used —
  a deterministic op on identical input gives identical indices — and the parity test (T2) compares against the router's own
  returned `indices`.
- **Oracle consequence (G2)**: because flips are a property of precision, not of vmap, the oracle must be **precision-matched**:
  same patched module, same router forward (patch on both sides or on neither), same dtype (§10).

---

## 4. Clipping norm and per-example gradient norms (the procedure)

What the design does so that a reasonable clipping norm exists, and how `C_g` is chosen:

1. **The load release never competes with the gradient for the clipping budget.** It is its own `PerGroup` group with a
   structural bound; the gradient group keeps exactly today's `C_g`. The joint single-vector clip (phase-1 math 4(a)) is
   rejected because it would shrink the gradient's admissible norm to `C√(1−ρ²)` and bias the histogram. At ρ = 0.02 the whole
   price is ×1.010 gradient noise.
2. **The aux term does not move the per-example norm.** At α ∈ [1e-4, 1e-3] the per-example norm changes by < 0.1 % (VERIFIED
   phase-1 empirical E3: 1.9409 → 1.9413 at coefficient 1e-3; the *unscaled* aux gradient is of CE size, so the bound is
   `α × O(1)`), and the surrogate gradient is identically 0 at balance (F3). `C_g` is governed by CE. The *per-example* aux
   variant (`E·P̄_{S_x}`) would add a heavier-tailed component (max/median 1.6–2.0, E3) — one more reason not to use it.
3. **Scale of the problem** (VERIFIED arithmetic): LoRA r = 16 on q/k/v/o (`q: 2304→4096, k/v: 2304→512, o: 4096→2304`) is
   294 912 parameters per layer, `d = 8 257 536`, `√d = 2874`; the per-step noise vector has norm `nm·C_g·√d/B̄ = 5.68 = 6.3·C_g`
   at C_g = 0.9 — 6× the largest possible clipped per-example contribution. In this regime the per-step signal-to-noise is set by
   the *coherence* of clipped per-example directions across the batch, and `C_g` acts mostly as a learning-rate scale once most
   examples are clipped (optimal §4.2).
4. **The calibration pass is itself a private query** (skeptic §4.2, grafted by all judges). The `clipped_grad(...,
   clipping_norm=1e9, return_aux=True)` pass that reads `aux.grad_norms` to pick `C_g` must run **on a public proxy** (KStack is
   public; `JetBrains/KStack` is the preset's own dataset) or be **accounted** as one Gaussian release of the norm quantile
   (`adaptive_clipped_grad` already provides the accounted version, `opaque-dpsgd/.../clipping/_adaptive.py:164-215`, accounted
   by `dpsgd_acc.adaclip`, VERIFIED phase-1 primitives §1.5). Under MF adaptive clipping is barred by the latch, so prefer the
   public proxy. Numbers from that pass are design-time inputs and are never logged by a DP run.
5. **How to pick `C_g`** (the G3 script, §10.2): one pass over ≈ 256 public-proxy examples under the preset partition
   (LoRA r = 16 q/k/v/o); report p10/p50/p90/p99/max of `aux.grad_norms` (toy distributions are tight: max/median 1.24–1.33,
   E3; expect p90/p50 ≈ 1.5–2 on real code, heterogeneity from language/file kind and FIM vs plain — PLAUSIBLE). Choose `C_g` by
   the **bias²+noise² curve**: for candidate `C`, bias = `‖mean(clip_C g_x) − mean(g_x)‖`, noise = `nm·C·√d/B̄`; minimise the
   sum — with `d = 8.3 M` this lands at or below the median (clip rate ≥ 50 %). Sanity check: clip rate 40–60 % in the first
   100 steps of the DP run (the same operating point the trainer's adaptive mode targets, `target_clipping_rate = 0.5`,
   `_dp_trainer.py:4251`, VERIFIED). The preset pins the measured value (0.9 today).
6. **Clipping mode: fixed** (both presets, `examples/train_dpftrl.py:614-618`, VERIFIED). With the feature on, `clipping_mode ∈
   {"auto", "adaptive"}` raises `ConfigurationError` in v1 (§2.3 for why AUTO-S biases the release; adaptive is barred under MF
   and, under DP-SGD, would drift the probe's bound and route the release through the adaclip accountant wrapper). v2 adds
   `fixed_groups=("router_load_probe",)` to `auto_clipped_grad` (§9.4) so AUTO-S users can opt in; AUTO-S's utility claim
   (Bu et al. 2023, https://arxiv.org/abs/2206.07136) is PLAUSIBLE, theorem numbers not re-fetched — fixed stays the preset default.
7. **Partition-aware groups when experts/router are trained** (§7): `per_group(trainable, gate=C_r, experts=C_e, fallback=C_a,
   router_load_probe=C_h)`; bounds at each group's per-example median from the (public-proxy) calibration pass; the optimal
   allocation inflates each group's σ by `√(S/C_g)`, `S = Σ_g C_g` — keep the number of groups ≤ 4. Expect the router group to
   need the smallest bound (E3: router aux-only norms are the heaviest-tailed). Recommended ρ = 0.1 there.
8. **Numerical stability vs the non-DP path** (requirement (d)): fp32 accumulation in `Opaque_MoE` (`kernels/moe.py:83, 103`,
   VERIFIED phase-1), upstream RMSNorm retained (`models/mellum.py:41`, VERIFIED), SDPA causal fast-path parity (PR #980) — the
   remaining bf16 drift is accumulation order, 0.45–0.5 % rel-L2 with **zero** route flips (VERIFIED phase-1 E1/E1b toy), inside
   HF's own batched-vs-loop spread.

---

## 5. Performance (G6)

### 5.1 Dense vs grouped MoE default

Facts (VERIFIED): `DPTrainer` passes `kernels=bool(args.use_performance_kernels)` (`_dp_trainer.py:841-847`), default `False`
(`_training_arguments.py:436`); the factory sets `grouped_moe = kwargs.get("grouped_moe", kernels)` (`_factory.py:316-322`);
`grouped=False` selects the dense every-token-through-every-expert `Opaque_MoE` on every host including CUDA
(`kernels/moe.py:578-635`); the flag is captured by the **first** class-level patch per process (`_router.py:59-92`:
`__opaque_patched__` guard); the `use_performance_kernels` docstring (`_training_arguments.py:429-435`) lists
rope/rms_norm/activation/cross_entropy and never mentions MoE; neither preset sets `grouped_moe` (VERIFIED phase-1 critic R3 /
skeptic grep).

FLOP arithmetic from the public config (VERIFIED arithmetic; timing PLAUSIBLE, G6 measurement §10): one expert =
`H·2I + I·H = 6.19 M` MAC/token/layer; routed (k = 8) = 49.5 M MAC/token/layer; dense (E = 64) = 396 M — **8× the expert
FLOPs**; over 28 layers 2.77 vs 22.2 GFLOP/token forward; with attention ≈ 1.7 and LM head 0.45 GFLOP/token the dense default is
≈ **5× the total forward FLOPs** (24.4 vs 4.9). With frozen experts the backward skips expert weight grads (`moe.py:536-539`,
`needs_input_grad`), so the ratio is similar in backward. Per 256-example step at T = 1024 this is roughly 12–17 PFLOP dense vs
1.5–2 PFLOP grouped — on one H100 at ~200 TFLOP/s effective 60–90 s/step vs 8–10 s/step, i.e. 11–16 days vs ≈ 2 days for the
15 625-step preset (minimal §5.1, PLAUSIBLE). The dense default is the single largest practical blocker for the presets.

**Decision (all four designs and all judges agree): decouple `grouped_moe` from `use_performance_kernels`.** In `_factory.py`
resolve `grouped_moe = kwargs.get("grouped_moe", kernels or _grouped_route_available())`, where the helper (next to
`kernels/moe.py:575-635`) is true on CUDA + Triton (fused kernel) or wherever `torch._grouped_mm` exists with ≥ 16 experts
(`_SPARSE_MOE_MIN_EXPERTS`, `moe.py:575`); `opaque_moe` already gates the grouped route on a workspace estimate and falls back to
dense (`moe.py:614-632`, VERIFIED). Justification: the docstring states every path is numerically equivalent within the dtype
floor (`moe.py:610-612`); phase-1 divergence measured grouped-vs-HF bf16 4.63e-3 vs dense 4.98e-3 (both inside HF's own spread;
fp32 both 3.5e-7, VERIFIED). Log the chosen path once; document the first-patch capture ("first `apply_model_patches` in a process
wins") in the Mellum patch docstring and add the MoE entry to the `use_performance_kernels` docstring. Dense stays the compat
fallback (`grouped_moe=False`). The Triton fused path's accumulation dtype is PLAUSIBLE (not verified here); the §10 flip counter
dense-vs-grouped is the acceptance test.

### 5.2 Cost and memory of the mechanism itself (per microbatch of 8, T = 1024, L = 28, E = 64, k = 8)

| buffer | shape / dtype | size | note |
|---|---|---|---|
| router logits tuple (recorder) | 28 × (T, E) references | 0 extra | already live in the autograd graph (softmax → top-k weights); the recorder stores references, not copies (`output_capturing.py:104-115`, VERIFIED) |
| probabilities for `P(x)` | (T, E) fp32 per layer, reduced immediately | 0.26 MB transient | layer-by-layer loop over the 28 tensors into an (E,) accumulator |
| one-hot counts | (T, k, E) bool per layer, reduced immediately | 0.5 MB transient | broadcast compare `idx[..., None] == arange(E)` (vmap-safe; `F.one_hot` is not, F7) |
| fp32 router logits (opt-in fp32 router) | (T, E) fp32 instead of bf16 per layer | + 7 MB/example | kept for backward through the softmax |
| probe leaf | (E,) fp32 | 256 B | + its per-example gradient (B, E) inside `clipped_grad`; 128 floats of inert optimizer state |
| filter state | (E,) EMA + scalars (or (W, E) ring for the window filter) | 256 B (64 KB) | public |
| `f̃` closure tensor | (E,) | 256 B | updated in `_augment_inputs` |

Compute: one `(L·T)×E` softmax per example (already computed by the router; reused), one masked mean, one top-k indicator
reduction — `O(L·T·E) = 1.8 M` elements per example; negligible against expert GEMMs. Chunked CE is preserved (§9.1), so peak
memory stays that of PR #978 rather than the full-vocab `98304×1024×4 B = 400 MB`-per-example logits of the current
`output_router_logits` fallback (`cross_entropy.py:212-233`, VERIFIED).

---

## 6. Smoothing under MF (G7)

### 6.1 The preset strategy, and the error the two losing designs made

The `mellum2-kstack` preset runs `band_mf_strategy(bands=64, momentum=0.95, lr_schedule=…)`: `--optimizer` default `sgd`
(`examples/train_dpftrl.py:447-460`), `--momentum` default 0.95 "per BandMF paper" (`:495-498`), `_workload_momentum()` →
`args.momentum` for SGD / `beta1` for Adam and `_make_strategy` passes it (`:1563-1581`) — all VERIFIED. The *library* default is
`momentum=1.0` (prefix sums; `_band_mf.py:111, 145`, VERIFIED). minimal §6 and skeptic §6 computed their filter tables for the
library default, which is **not the preset**; under momentum 1.0 the single-step row norm grows with the horizon (2.26 at n = 1024
→ 3.80 at n = 15625) and EMA .95 only reaches parity with iid (0.185 vs 0.160). Under the preset's momentum 0.95 the row norm is
**stationary from t ≈ 7 and n-independent** (VERIFIED, `mf_rownorm.py` re-run, both momenta at n = 1024):

| filter | band-MF(64, **0.95**) factor (stationary) | band-MF(64, 1.0) factor at n = 1024 (NOT the preset) | DP-SGD (iid) factor | MF(0.95)/SGD | lag (steps) |
|---|---|---|---|---|---|
| single step `‖row_t(C⁻¹)‖` | **1.431** | 2.260 (3.80 at n = 15625) | 1.000 | 1.43× worse | 0 |
| EMA β = 0.95 (= strategy momentum) | **0.0824** | 0.1157 | 0.1601 | 0.51 | 20 |
| **EMA β = 0.99 (default)** | **0.0249** | 0.0253 | 0.0709 | 0.35 | 100 |
| window mean W = 256 (selectable) | **0.0198** | 0.0157 | 0.0625 | 0.32 | 128 |

(Optimal's n = 2048 run adds EMA .90 = 0.1482, EMA .995 = 0.0162, window 16/64 = 0.1432/0.0479 at momentum 0.95, VERIFIED by
judge-utility's independent recomputation; not re-run by me.)

### 6.2 Which filter, and why

- The strategy is optimised for the momentum workload `A = Toeplitz(β^i)`, β = 0.95 (`_band_mf.py:120-138`, phase-1 primitives
  §7.4), and the EMA with `β_f = β` is exactly `(1−β)·A` — it inherits the strategy's optimality statement scaled by `(1−β)`
  (faithful §6.1, algebraically true). **But** the noise variance of *any* fixed linear filter `F` of the released stream
  `x̂ = d + σ_h·(C⁻¹Z)` is the deterministic quantity `σ_h·‖row_t(F·C⁻¹)‖` — the table is exact and needs no theorem — and every
  low-pass filter benefits 2–3.5× from band-MF's anti-correlated noise, the benefit growing with the filter's memory. The filter
  is therefore chosen by **lag tolerance**, not by the strategy's momentum (optimal §6, endorsed by judge-utility/impl):
  `β_f = 0.99` (lag 100 steps = 0.64 % of the 15 625-step horizon; imbalance drifts on the scale of the router's adaptation) gives
  0.0249, **0.83 %·(nm_MF/0.5622) of k/E at ρ = 0.02**. Fixing `β_f := 0.95` "to inherit the guarantee" (faithful) costs 3.3×
  accuracy for no privacy gain.
- EMA .99 vs window 256 at the preset: 0.0249 vs 0.0198 (20 %), lag 100 vs 128 — a wash. EMA is the default under **both**
  stacks (one code path; scalar exact-variance recursion for the shrinkage; 64 floats of state); `{"kind": "window", "steps": W}`
  is selectable, and the boxcar also removes the short `t < 7` transient. Under DP-SGD the same EMA has factor 0.0709.
- **Compute the factors from the instantiated strategy at setup — never hand constants, never the factory-default momentum
  (judge-dp graft 2 / judge-impl E8).** At `_setup_training`, when `ctx.mf` is present, take `ctx.mf.strategy` (which already
  carries the trainer's `lr_schedule`, `trainer/_dpftrl.py:100-113`) and its `coefficients(n_steps)`; obtain the lower-triangular
  Toeplitz *inverse* coefficients for lags `0..N_φ` (`N_φ = 2048`) by the triangular recursion (`O(N_φ·bands)`; the same object
  `inverse_as_streaming_matrix` builds, `_band_mf.py:135-136` → `_toeplitz.py:177`, VERIFIED), form `φ_t = ‖row_t(F·C⁻¹)‖` for
  `t < N_φ` (`O(N_φ²)` — 4 M flops), assert it has stabilised, and store `φ_t` (and `φ_∞`) in `RouterLoadState`. **Never a dense
  `n×n` solve**: at n = 15 625 that is 1.95 GB and `O(n³)`. Under DP-SGD `φ_t` is the closed-form EMA recursion. The `lr_schedule`
  (query weights) changes `A`, hence `C` and `C⁻¹` slightly; computing from the actual strategy makes the stored `s_t` exact for
  the run (faithful risk 6 closed).

### 6.3 Per-step realised σ and diagnostics

The trainer already reads the realised per-step σ from `NoisedPytree.noise_stddev` (`PerGroup` × `row_l2`,
`_mf_gaussian_noise.py:186-192`); the EMA consumer uses its own *filtered* factor `φ_t` for `s_t` and for the logged
`router_load/noise_std`, never the per-step one — a single-step reading under band-MF is 1.43× the DP-SGD value and must never be
consumed directly.

### 6.4 The `nm_MF` probe (this session, best effort)

`scratchpad/research/synthesizer/nm_mf_probe.py` evaluates `b_min_sep(mf_gaussian(nm, band_mf_strategy(64, 0.95)), n_steps=15625,
p0=256/5e5).epsilon_at(1e-6)` for nm ∈ {0.5622, 1.5, 3.0} under a 570 s wall-clock bound. **Result (VERIFIED,
`nm_mf_probe.out`): the first evaluation did not finish — the accountant warns that Monte Carlo accounting at
`mc_resolution = 5e-7`, `mc_failure_probability = 1e-6` needs 64 997 003 samples per adjacency direction
(`opaque-accounting/.../core/_pld_cache.py:206`), and the process was killed at the bound (exit 124).** So `nm_MF` remains
uncomputed by every phase-2 agent (minimal's single evaluation exceeded 170 s; mine exceeded 570 s on 4 CPU cores), the caveat in
§2.6 stands as PLAUSIBLE, and the GPU-side validation plan (§10.2, last row) records `nm_MF` from the trainer's own
`_calibrate_noise` — which the preset runs at every start anyway, so the number is free to record there.

---

## 7. Scope (G8)

| item | v1 | what it needs / why |
|---|---|---|
| Causal-LM SFT via `DPTrainer` default path and `DPSFTTrainer` (`nll` / `chunked_nll`) — the `mellum2-kstack` preset | **in** (`surrogate`, α = 1e-4, ρ = 0.02) | §9 |
| DP-FTRL (band-MF, BLT, BSR, BiSR, λ-CGD) with b-min-sep / Poisson / balls-in-bins | **in** | nothing beyond §6: all strategies go through `mf_gaussian_noise`'s `PerGroup` path; the latch accepts the constant `PerGroup` (VERIFIED phase-1 primitives E2 for identity and band-MF) |
| DP-DPO mechanism (`DPDPOTrainer`, `_dpo_trainer.py:1065-1195`) | **in** | protected unit = the preference pair; `P(x)`, `h(x)` pool the **chosen + rejected policy forwards** with denominator `L·(T_c + T_r)` (same structural bound — fractions); the **reference forward contributes no aux** (HF adds the aux only to the policy loss, `modeling_mellum.py:692-700`) and is never called with `opaque_router_logits=True` — whether precomputed or the TR-DPO EMA reference refreshed in `_augment_inputs` (`_dpo_trainer.py:807-841`, VERIFIED); its stats must never reach the probe (test T21). The two policy forwards already exist in `compute_per_example_loss_and_metrics`, so it is the same helper called twice plus the converter mapping. |
| `mellum2-codesec` preset (DP-DPO) | **`monitor`** in v1 (ρ = 0.02, α = 0) | TRL never added the aux to DPO/KTO (phase-1 literature C13, VERIFIED) and no DPO validation has run; `surrogate` with α = 1e-4 is one flag away once §10's DPO row passes |
| Router z-loss `Z_x` (ST-MoE eq. 5) | **opt-in, ζ = 0** | per-token separable ⇒ zero privacy cost (inside the clipped per-example gradient); Mellum2 pretraining used 1e-3; absent from HF's objective; useful only with a trainable router |
| Router / experts trainable | **in for the mechanism, out for presets** until critic M5 runs | nothing in §2 depends on which leaves exist; `opaque_moe` emits per-example expert-weight grads when `needs_input_grad` is set (`moe.py:536-539`, VERIFIED). Blockers: (i) PEFT `target_parameters` LoRA on stacked `experts.gate_up_proj/down_proj` under `functional_call`+vmap is **untested** (phase-1 primitives §5); (ii) per-group `C` by parameter class (§4.7); (iii) full expert training needs `B·E·2I·H` per-example buffers per layer (≈ 0.53 GB/example/layer bf16; 4.2 GB per layer per microbatch of 8) — LoRA-on-experts / ESFT-style subsets are the practical forms, and ESFT selection from private data is itself a query to account (phase-1 literature D.2); (iv) route pinning to an fp32 forward is the validated-later lever for the E1b tail. Recommended ρ = 0.1 (×1.049; MF 0.38 %) or, under DP-SGD, the independent draw with c = 1. |
| Per-layer `f̃^l` (Megatron pooling) | **opt-in** `router_load_per_layer=True` | probe leaf `R^{L×E}`, bound `√(kL(1−k/E)) = 14.0`, ×5.29 relative noise (§2.6 column); usable under MF at ρ ≥ 0.2, never free |
| Per-sequence aux (option (d)) | **opt-in flag** `router_aux="per_sequence"` | zero privacy cost; documented as a *different* regulariser (anti-specialisation; phase-1 math §2), never "faithful" |
| Independent forward-only Poisson release | **v2 opt-in, DP-SGD only** | §2.7 conditions |
| Public-data `f̃` (`router_load_source="public"`) | opt-in | forward-only on a public batch every `m` steps, zero release, no probe group; the hybrid (public prior + private release) has the same sensitivity as the private release and buys nothing |
| Loss-free balancing (Wang et al. 2024 Alg. 1, https://arxiv.org/abs/2408.15664; DeepSeek-V3 eq. (16), https://arxiv.org/abs/2412.19437; both VERIFIED phase-1 literature C4/C5) | **out** (documented) | `b_{t+1} = b_t − u·sign(d̂_t)` (or a dead-zone / soft-sign variant) is **post-processing of the same release — free**; RR on the true sign is dominated (§2.6). But the checkpoint's `MellumTopKRouter` has no bias tensor and `nn.Linear(bias=False)` has no constant input channel, so `b` cannot be folded into `W`: it is an architecture extension that must ship with the adapter and a serving patch; training-only use (discard `b` at export) is inadmissible — it creates train/inference routing mismatch by construction (the TR §5.2 defect). The authors say LFB is their next step (TR A.2). A separate design if wanted. |
| HF-Trainer-realised objective (per-microbatch `f`, `G·α`) | out | artefact (§1.3) |
| Token-weighted CE with public `N̄` | out | one-line loss change plus a clip-norm rethink (phase-1 divergence §4) |
| STE / DenseMixer through top-k | out | changes the real model's backward; noted as a future utility lever (phase-1 literature) |
| Users passing `output_router_logits=True` under DP | **rejected with a clear error** | fails under vmap with a mask (in-place `scatter_add_`, `modeling_mellum.py:598`, VERIFIED phase-1 empirical 0.1) and bypasses chunked CE |

---

## 8. Privacy hygiene (G9)

Rule: everything computed from private examples inside the grad transform is private-internal until it has passed clip → noise;
only noised aggregates and their post-processing are public. Adjacency/unit as in §2.1.

| tensor / state | where it lives | class | logged? | checkpointed? | DDP |
|---|---|---|---|---|---|
| router logits, fp32 probs, executed routes `S`, `h(x)`, `P(x)`, `d(x)`, `Z_x`, per-example aux value | inside the vmapped loss closure | **private-internal** — never leave `clipped_grad`; **not** added to `loss_aux` (the existing un-noised `loss`/`loss_aux` means at `_dp_trainer.py:2286-2293` are a pre-existing posture, F11 — this design adds nothing to them) | never | never | n/a |
| per-example probe gradient `λ·d(x)` | inside `clipped_grad` before summation | private-internal | never | never | n/a |
| `ClippedGradAux.group_norms["router_load_probe"]` (per-example `‖λ d(x)‖`) | existing telemetry channel | **private, un-noised** — the trainer's `group_metrics` loop logs `group_norms.mean()` and a `clip_rate` for **every** `PerGroup` group (`_dp_trainer.py:2259-2275`, VERIFIED); with a probe group that would be an unaccounted release of the batch-mean routing norm (faithful §8 / optimal §8 error E5). **The probe group is skipped in that loop** (one `continue`; test T17). The clip-rate entry would be 0 by structure anyway. | **excluded** | no | gathered in-process by `sync(aux)` (existing) |
| clipped-summed probe leaf, pre-noise (`grads.pytree["router_load_probe"]`) | `ClippedPytree` in `training_step` | private — same status as the un-noised gradient sum; only `noise_fn` may consume it | no | no | all-reduced by `sum_gradients_` (private sum, like every gradient leaf) |
| noised probe leaf `ŷ_t` (`noisy_grads.pytree["router_load_probe"]`) | `NoisedPytree` handed to `on_pre_optimizer_step` | **public** (DP output; the only release added by this design) | may be logged (`router_load/raw_*`) | via filter state | rank-identical (shared noise key, `_dp_trainer.py:1478`) |
| `d̂_t`, `d̃_t`, `s_t`, `f̃_t`, `D_t`, trip flag, `φ_t` | `RouterLoadState` on the built-in callback | **public post-processing** | yes: `router_load/D`, `/f_min`, `/f_max`, `/entropy`, `/shrink`, `/noise_std = s_t`, `/tripped` | **yes** — sidecar `router_load_state.pt` (registry `state_dict`); reproducibility, not privacy | rank-identical by construction; optional `register_sync_type(RouterLoadState, assert_equal)` (`distributed/_state.py:537`, VERIFIED) |
| probe parameter `z` | `ctx.trainable_params["router_load_probe"]` | public constant **0** (callback zeros its noised entry; `_augment_inputs` asserts) | no | as a zero vector in the model state (harmless) | identical |
| `f̃_t` closure tensor | trainer attribute read by the loss | public | — | derived | identical |
| `aux.batch_size` (realised `\|B_t\|`) | existing | private, pre-existing — **never** used as a divisor or renormaliser for `d̂` | (pre-existing) | no | summed (existing) |
| second-moment squared stream of the probe | — | would consume budget (`paired_noise_stddevs` sums `Δ¹+Δ²` over all groups, `noise_allocation.py:153-175`, VERIFIED) | — | — | **v1: `ConfigurationError` with `second_moment=True`**; v1.1: exclusion (§9.2) |
| `C_g` calibration pass (`aux.grad_norms` quantiles) | design-time script | **a private query if run on training data** — run on a public proxy or account it (§4.4) | design-time only, never in a DP run | no | — |
| independent-draw sampler key/position, second noise-key counter (v2) | `RouterLoadReleaser` | public RNG state that the privacy argument depends on (fresh coins) | no | **yes** (sidecar) | rank-domain-separated key |
| pinned routes (if ever enabled) | inside vmap | private-internal | never | never | no |
| `α, ρ, λ, C_h, β, W, E, k, τ, ζ` | args | public hyperparameters | yes | yes (args) | — |
| pre-existing un-noised `loss`, `grad_norm`, `clip_rate` means (`_dp_trainer.py:2249-2293`) | existing | private, unaccounted — outside this task; flagged per F11 and the review protocol's "all releases in the privacy statement" | yes (pre-existing) | — | — |

**Privacy statement addition** (for `docs/mechanisms/dp-sgd/…`, the DP-FTRL mechanism page and the trainer docstring): "When
`router_load_release` is `monitor`, `surrogate` or `monitor_then_surrogate`, each step releases one Gaussian (or matrix)
mechanism on the concatenation of the clipped per-example gradients and the per-example centred router-load vectors
`λ·(h(x) − k/E)`, with per-record bounds `C_g` and `C_h = λ·√(k(1−k/E))`; the two are one mechanism under Opaque's per-group
allocation and the accountant is unchanged (`gaussian(nm)` per step under the stated sampler / `mf_gaussian(nm, strategy)` for
the horizon). The gradient noise is inflated by `√(1+ρ)`. The load estimate `f̃_t` consumed by the loss and the monitor `D_t`
are post-processing of previous releases. No other quantity derived from private routing is released; the per-example
group norms of the probe group are excluded from telemetry." [v2 adds: "every `m` steps an independent Poisson-subsampled
Gaussian release of the centred router-load vector, sensitivity `Δ_h`, multiplier `c·nm`, composed with `|`."]

---

## 9. Implementation plan (G10) and tests

Engine (`opaque-engine`), `opaque-dpsgd`, `opaque-dpftrl`, `opaque-accounting`: **no change in v1.** Reused as-is:
`per_group` (`_per_group.py:44`), `PerGroup`, `clipped_grad` (`_clipped_grad.py:85`), `per_group_noise_stddev`
(`noise_allocation.py:44`), `gaussian_noise` (`_gaussian.py:320-322`), `mf_gaussian_noise` (`_mf_gaussian_noise.py:163-192`),
the serialization registry (`opaque-base` `_dispatch.py`, `opaque-engine` `_structural.py`), `sum_gradients_`, the streaming
Toeplitz inverse (`_toeplitz.py:177`).

### 9.1 `opaque-patches`

| file | change |
|---|---|
| `src/opaque/api/patches/transformers/components/moe_stats.py` (**new**) | `router_load_and_probs(router_logits: Sequence[Tensor], attention_mask: Tensor \| None, *, top_k: int) -> (h, P)`: layer-loop, out-of-place, broadcast-compare one-hot (no `F.one_hot`, no `scatter_add_`, no `bincount`), returns `(E,)` each; `centred_load(h, top_k) = h − k/E`; `load_balancing_surrogate(P, f_tilde, *, num_experts, top_k) = E·⟨f̃ − k/E, P⟩`; `router_z_loss(router_logits, attention_mask)`; optional `per_layer=True` returning `(L, E)`. |
| `src/opaque/api/patches/transformers/components/router.py` (**new**) | `make_fp32_router_forward(original)` for `MellumTopKRouter` (§3): `logits = F.linear(h.float(), W.float())`, fp32 softmax, `topk`, renormalise on the fresh top-k tensor (already vmap-safe upstream), `scores.to(h.dtype)`, return `(logits_fp32, scores, indices)`. **Opt-in.** |
| `src/opaque/api/patches/transformers/components/cross_entropy.py` | new **named** kwarg `opaque_router_logits: bool = False` on the fused/chunked causal-LM `forward` (`:176-192`; named so it is not swallowed by `**kwargs`, which the backbone call forwards verbatim, `:252-264`, VERIFIED): when true, call the backbone with `output_router_logits=True` **without** taking the `output_router_logits` fallback branch (`:212-233`, kept for HF's aux-loss contract and its existing test `test_fused_ce_preserves_router_auxiliary_loss_contract`); keep chunked CE; the existing `hasattr(outputs, "router_logits")` return path (`:359-371`, VERIFIED) already builds `MoeCausalLMOutputWithPast(loss, logits=None on the loss-only path, router_logits=outputs.router_logits, aux_loss=None)` — no HF batch aux is added, which is the whole point. |
| `src/opaque/api/patches/transformers/_factory.py` | (a) `classes["router"]` role + `router_kind` recipe arg, gated in the `compat` bucket by `kwargs.get("router_fp32", False)` (opt-in); (b) `grouped_moe = kwargs.get("grouped_moe", kernels or _grouped_route_available())` (§5.1) with a one-time log of the chosen path. |
| `src/opaque/api/patches/kernels/moe.py` | `_grouped_route_available()` helper next to `:575-635`. |
| `src/opaque/api/patches/transformers/models/mellum.py` | `classes["router"] = "MellumTopKRouter"`; docstring: opt-in fp32 router (pretraining-faithful, not a drift fix), grouped default, first-patch capture. |
| `docs/mechanisms/dp-sgd/moe-load-balancing.md` (+ a DP-FTRL section) and `docs/user-guide/huggingface.md` | loss, mechanism, cost table with the `nm_MF` caveat, privacy statement, primary sources (Switch eqs. 4–6, Andrew Thm 1, Dong–Roth–Su Thm 2.7, ZDW Thm 10, Feldman–Shenfeld Lemma 3.2/Thm 3.3, Denisov Thm 2.1, Mellum2 TR §3.6/§5.1.2, ST-MoE eq. 5). Diary-free per AGENTS.md. |

**How the statistics reach the loss, and why it composes (G10 — the correct mechanism, minimal §9.1, VERIFIED by me):**
`MellumModel.forward` is decorated `@capture_outputs` (`modeling_mellum.py:474`) and declares
`_can_record_outputs["router_logits"] = OutputRecorder(MellumTopKRouter, index=0)` (`:432`). The recorder hooks are installed
**once** and persist (`output_capturing.py:255-258`); each hook appends only while a `ContextVar` collector is active
(`:104-108`: returns immediately when `_active_collector.get()` is `None`), and `capture_outputs` sets that collector immediately
before the backbone forward and resets it in a `finally` right after (`:259-272`). Hence:

- **gradient checkpointing**: the non-reentrant recompute during backward (`_force_non_reentrant`,
  `patches/torch/checkpoint/huggingface.py:31, 35`, VERIFIED) runs the router hook with the collector reset → **nothing is
  appended, no double capture**. (faithful's "hooks registered for the duration of one forward call" and optimal's "fires again
  … nothing reads it" describe the mechanism wrongly; the conclusion is right — judge-impl E6.) Whether the captured logits carry
  gradient through the checkpoint region under vmap is PLAUSIBLE (HF trains MoE aux under checkpointing this way in eager) —
  test T5 makes it VERIFIED. Fallback if T5 fails: skeptic's forward-hook capture with *overwrite by layer index* and reset per
  call (VERIFIED phase-1 primitives E3 at toy scale), or returning the logits functionally from a patched `MellumSparseMoeBlock`.
- **microbatch chunks**: each chunk is a separate vmapped call (`_clipped_fun.py:273-276`, VERIFIED) → separate collector; the
  probe leaf accumulates across chunks like every gradient leaf.
- **`torch.compile`**: HF uses a `CompileableContextVar` (`output_capturing.py:97`, VERIFIED); if dynamo still breaks,
  `_compile_with_fullgraph_fallback` (`_dp_trainer.py:4191-4228`) downgrades to `fullgraph=False` — the same fallback the
  adaptive/auto paths rely on today; the closure tensor `f̃` is lifted as a graph input (PLAUSIBLE, T18).
- **batchify**: `_squeeze_output` squeezes only top-level tensor values (`functional/__init__.py:189-214`, VERIFIED judge-impl);
  `router_logits` is a tuple of `(T, E)` tensors (the router reshapes to `(-1, H)`, `modeling_mellum.py:333`) — no shim.
- **MF latch**: constant two-group `PerGroup`; `ρ, λ, C_g`, group map fixed at construction and asserted on resume.

### 9.2 `opaque-transformers`

| file | change |
|---|---|
| `trainer/_training_arguments.py` | fields of §2.5 (`router_load_release`, `router_aux_loss_coef`, `router_load_ratio`, `router_load_filter`, `router_load_shrink`, `router_load_trip`, `router_load_per_layer`, `router_aux`, `router_z_loss_coef`, `router_fp32`). Validation: any non-`off` mode requires `clipping_mode == "fixed"`, `second_moment == False`, and a model whose backbone can record `router_logits` (checked at setup, else `ConfigurationError`). |
| `trainer/_router_load.py` (**new**) | frozen dataclass `RouterLoadState(d_ema: Tensor, noise_std: float, f_tilde: Tensor, phi: Tensor, step: int, tripped: bool, alpha_active: float, kind: str, beta: float, window: int, rho: float, top_k: int, num_experts: int)` (registry-serialisable, VERIFIED phase-1 primitives §4.4 `LoadEmaState` round-trip); pure functions `initial_state(...)`, `filter_factors(strategy \| None, n_steps, kind, beta/window) -> phi` (§6.2), `update(state, noised_leaf, lam) -> state` (§2.4 steps 1–6), `summary(state) -> dict[str, float]`. Built-in `RouterLoadCallback(TrainerCallback)` with `on_pre_optimizer_step(self, args, state, control, *, grads, trainable_params, **kw)`: reads `grads.pytree["router_load_probe"]`, calls `update`, zeros that leaf in place, sets the α for the next step in `monitor_then_surrogate`. Registered automatically when the feature is on — reuses the existing `call_event("on_pre_optimizer_step", …, grads=noisy_grads, trainable_params=…)` seam (`_dp_trainer.py:2198-2206`, VERIFIED) with **no new seam in `training_step`**. |
| `trainer/_dp_trainer.py` `_setup_training` | if enabled: (1) `self._model.register_parameter("router_load_probe", nn.Parameter(torch.zeros(E [or L·E], dtype=float32, device)))` **before** `make_functional(partition_trainable=True)` (`:1357-1361`, VERIFIED) so the leaf lands in `trainable_params` under path `("router_load_probe",)`; (2) extend the clip-norm block (`:1455-1470`, VERIFIED): a scalar `clipping_norm=C_g` becomes `per_group(trainable, router_load_probe=C_h, fallback=C_g)`, a dict gets the `router_load_probe` key; (3) compute `phi` from `ctx.mf.strategy` (or the EMA closed form) and build `RouterLoadState`; (4) instantiate `RouterLoadCallback`; (5) store `λ, α, E, k` on the trainer; (6) `ConfigurationError` checks. |
| `_dp_trainer.py` `_augment_inputs` (`:2302-2313`) | copy `callback.state.f_tilde` into `self._router_load_target` (a device tensor read by the loss closure — unbatched under vmap, a graph input under compile); assert `ctx.trainable_params["router_load_probe"]` is zero. No batch column ⇒ no `_remove_unused_columns` issue (`:3446-3470`, VERIFIED). |
| `_dp_trainer.py` `compute_per_example_loss` (`:2314-`) and `trl/_sft_trainer.py` `compute_per_example_loss` (`:554-615`) | shared helper `self._apply_router_load_terms(loss, outputs, params, inputs)`: `h, P = router_load_and_probs(outputs["router_logits"], inputs.get("attention_mask"), top_k=k)`; `loss = loss + α_active·load_balancing_surrogate(P, self._router_load_target, …) + (params["router_load_probe"] * (λ·(h − k/E)).detach()).sum() [+ ζ·router_z_loss]`. The forward is called with `opaque_router_logits=True` (marker detected by signature exactly like `_fused_forward_uses_marker`, `_sft_trainer.py:287-291`, VERIFIED); the SFT `chunked_nll` path (`:576-590`) keeps `opaque_fused_loss_only=True`. |
| `trl/_dpo_trainer.py` `compute_per_example_loss_and_metrics` (`:1065-`) | call the helper on both policy forwards (chosen, rejected), pool with `L·(T_c+T_r)` weighting; the reference forward (`:807-841`) is never called with the kwarg; assert its outputs carry no `router_logits` (T21). |
| `_dp_trainer.py` metrics (`:2256-2293`) | skip the probe group in the `group_metrics` loop; add `summary(state)` keys under `router_load/*`. |
| `_dp_trainer.py` `_save_checkpoint` (`:4910`, bundle at `:5091-5122`) / `_apply_runtime_state` (`:5335-5358`) / resume | write/read the sidecar `router_load_state.pt = opaque_state_dict(state)` next to `DP_STATE_NAME`; on resume assert `rho, beta/window, E, k, C_g` match (else `CheckpointError`). `save_dp_runtime_state`'s fixed signature (`_checkpoint.py:311-335`, VERIFIED) is left alone. |
| `trl/_convert.py` `_drop_router_aux_loss` (`:69-76`, VERIFIED) and the SFT/DPO converters | SFT on a family with router-recorder support: `router_aux_loss_coef > 0` → `router_load_release="surrogate", router_aux_loss_coef=value` (info log instead of the warning). DPO: → `monitor` with a message that the surrogate term for DPO is available by flag. Unsupported families keep the warning. |
| second-moment exclusion (v1.1) | when `second_moment=True`, zero the probe leaf of `squared_grads` before the release and pass a `0.0` second-moment bound for the probe group so `paired_noise_stddevs` allocates it `σ² = nm·√(0·S) = 0` (sensitivity 0 is legitimate for a discarded stream; PLAUSIBLE — needs a look at how the engine derives the second-moment `PerGroup`); test T20. v1 raises. |
| `examples/train_dpftrl.py` `mellum2-kstack` (`:873-896`), `examples/train_dpo.py` `mellum2-codesec` (`:1300-1318`) | kstack: `router_load_release="surrogate"`, `router_aux_loss_coef=1e-4`, `router_load_ratio=0.02`; codesec: `router_load_release="monitor"`, ρ = 0.02. Both rely on the new grouped default (or set `performance_kernels_config={"grouped_moe": True}` explicitly). |

DDP: nothing to add — `sum_gradients_` all-reduces every leaf including the probe (`distributed/gradients.py:150-`, VERIFIED
phase-1 primitives §2.3), noise is added per rank with the shared key after the reduction (`_dp_trainer.py:2172-2183`), so
`ŷ_t`, the filter state and `f̃` are bit-identical on all ranks.

### 9.3 Test plan (placement per ARC-006; markers per AGENTS.md; behaviour only — no docstring pinning)

`packages/opaque-patches/tests/transformers/models/test_mellum.py` + new `test_moe_router_stats.py` (tiny random-init models via
`build_moe_model`):

- **T1** fp32 router (opt-in): `router_logits.dtype == float32`, scores dtype = hidden dtype, indices equal to the unpatched
  router on an fp32 model; works under `vmap` and `vmap(grad)`; unpatched contract untouched when `router_fp32=False`.
- **T2** `router_load_and_probs`: vmap-safe; `Σ_e h = k`, `0 ≤ h ≤ 1`, `‖h − k/E‖₂ ≤ Δ_h`, `Σ_e P = 1`; equals an eager per-example
  loop; padded tokens excluded; `h` from the logits equals `h` from the router's returned `indices`; per-layer variant `(L, E)`.
- **T3** surrogate identity (float64, equal lengths): `Σ_x ∇ℓ_x|_{f̃=f(B)} == ∇ load_balancing_loss_func` to 1e-12 relative;
  ragged lengths with the `T_x/T_tot` reweighting; centred and uncentred surrogates give identical gradients.
- **T4** chunked-CE forward with `opaque_router_logits=True`: returns `L` `(T, E)` router logits under vmap, `logits is None`,
  loss equals the same forward without the kwarg; with `output_router_logits=True` the HF-aux fallback is still taken (existing
  test stays green).
- **T5** gradient checkpointing: `gradient_checkpointing_enable()` → `(h, P, grads)` equal to the non-checkpointed run to fp32
  tolerance; `len(router_logits) == L` (no double append).
- **T6** grouped vs dense `opaque_moe` with the statistics enabled: identical `h`, grads within the parity-harness tolerance;
  `_grouped_route_available()` default resolution.

`packages/opaque-transformers/tests/opaque_transformers/test_router_load_balancing.py`:

- **T7** setup: probe leaf in `trainable_params`; `clip_norm` is a two-group `PerGroup` with `C_h = ρ·C_g·(1+1e-6)`; scalar
  `clipping_norm` converted; σ values equal `per_group_noise_stddev` closed form; Mahalanobis identity holds.
- **T8** pre-noise probe leaf equals `(λ/B̄) Σ_x (h_x − k/E)` exactly and `group_norms["router_load_probe"] ≤ C_h` for adversarial
  single-expert routing (clipping never triggers; `clip_rate` 0).
- **T9** post-processing: with `noise_fn` monkey-patched to inject known noise, `d̂`, projection, EMA/window, `s_t` recursion,
  shrinkage (identity when `‖d̃‖ ≫ s√63`, zero below), clamp and `f̃` match closed form; `f̃_0 = k/E`; `Σ_e f̃ = k` up to clamp.
- **T10** probe hygiene: after `n` steps `trainable_params["router_load_probe"] == 0` exactly; loss value equals
  `CE + α·surrogate` bit-for-bit; optimizer state for the probe is zero; `α = 0` / `off` runs are bit-identical to today's path.
- **T11** checkpoint round-trip: sidecar restores `RouterLoadState` exactly; resuming reproduces the next `f̃` bit-identically;
  mismatched `ρ` raises.
- **T12** (`distributed` marker, 2 Gloo ranks): identical `f̃` and filter state on both ranks after 3 steps; probe leaf
  all-reduced.
- **T13** `microbatch_size=2` chunks == single chunk (grads, `h`, `d̂`).
- **T14** DP-FTRL: `mf_gaussian_noise` + `band_mf_strategy(bands=4, momentum=0.95)` accepts the `PerGroup` latch across steps;
  realised `noise_stddev.values["router_load_probe"] == base·row_l2(t)`; `phi` computed from the strategy matches a dense
  small-n reference; Monte-Carlo over the noise key reproduces the filtered noise factor within 10 %.
- **T15** accounting invariance: `epsilon_at(δ)` **identical** with and without the feature under both stacks (the
  machine-checkable form of "accountant unchanged"); `_build_mechanism` returns the same factory.
- **T16** converter: `router_aux_loss_coef` maps to `surrogate` for SFT on Mellum, to `monitor` for DPO, warns elsewhere.
- **T17** hygiene: no logged key is a function of un-noised `h`/`f(B)`; `router_load/*` values equal `summary(state)`; the
  probe group is absent from `group_metrics`; `loss_aux` carries nothing new.
- **T18** (`slow`) `torch_compile=True` one step == eager on CPU inductor (fullgraph fallback exercised).
- **T19** `ConfigurationError` for `clipping_mode ∈ {"auto", "adaptive"}`, `second_moment=True`, and unsupported families.
- **T20** (v1.1) second-moment exclusion: the probe's squared stream is zero and consumes no budget (paired σ for the probe = 0).
- **T21** DPO pooling: `h`, `P` pooled over chosen + rejected with `L·(T_c+T_r)`; reference forward contributes nothing; the
  structural bound holds for the pair.
- **T22** decision rule on synthetic `d̂` streams: no false alarm under exact balance at ρ = 0.02, W/EMA defaults; detection of a
  deviation 1.0 within one window; `monitor_then_surrogate` switches α without changing `max_norm` or the accountant.
- **T23** (engine, v2) `auto_clipped_grad(fixed_groups=("router_load_probe",))`: fixed group uses `min(1, C/‖·‖)`, AUTO-S groups
  unchanged, `max_norm` constant, MF latch accepts it.
- **T24** (v2) independent draw: accounting equals `poisson(g(nm),q)*T | poisson(g(c·nm),q₂)*(T/m)`; realised release counts
  ~ `Poisson(q₂N)` per rank shard; sampler key/position round-trips; `mechanism_kind="band_mf"` + `"independent"` raises.

Rust: no change. Docs build: the new mechanism page.

### 9.4 v2 engine change (deferred, written down)

`auto_clipped_grad(..., fixed_groups: tuple[str, ...] = ())` (`_auto.py:203`) → `auto_scale_pytree` / `_auto_scale_per_group`
(`_pytree.py:313-347`, currently a single `clamp_to_one=False` for all groups, VERIFIED): named groups use `min(1, C/‖·‖)`
(fixed semantics) while the rest use AUTO-S `R/(‖·‖+γ)`. `PerGroup` max_norm unchanged (constant ⇒ MF latch OK); per-record
bounds `R` for AUTO-S groups and `C_h` for the fixed one ⇒ privacy unchanged. `_create_grad_fn` passes
`fixed_groups=("router_load_probe",)` in `auto` mode. Lifts the v1 `ConfigurationError`.

---

## 10. Validation plan on the real checkpoint (GPU) — G2 / G3

Script `examples/validate_mellum_dp.py` (one 80 GB GPU; `JetBrains/Mellum2-12B-A2.5B-Base` bf16 weights as in the presets,
`JetBrains/KStack` (public), T = 1024 right-padded, PEFT LoRA r = 16/α = 32 on q/k/v/o; ≤ 30 min for the oracle + statistics,
≈ 30 min more for the 200-step DP run with grouped MoE). Every number below is a design-time measurement on **public** data;
nothing here is logged by a DP run.

### 10.1 Oracle definition and drift metric (G2)

**Oracle O0 (the "non-DP HF path", precision-matched):** same process, same patched module (`apply_model_patches` with the run's
`grouped_moe`, `router_fp32` and dtype — patch on both sides or on neither), HF eager forward on one microbatch (B = 8) with
`output_router_logits=False`, `loss.backward()` per example in a Python loop ("HF loop") and once batched ("HF batched", the
non-DP Trainer's computation). **O1 (floor):** the same loop in fp32 weights (48 GB; sequential, one model at a time). **DP
side:** Opaque `clipped_grad(..., clipping_norm=1e9, return_aux=True)` internals — bf16 `vmap(grad)` per-example vectors with the
full Mellum patch set (grouped and dense variants; chunked CE; gradient checkpointing on/off; one padded microbatch too).

**Metrics, always reported side by side, never as one number:** (m1) per-example rel-L2 of the whole LoRA gradient, vmap vs
HF loop and vs O1, median/max; (m2) **route-flip counter**: per `(token, layer)`, symmetric difference of the executed top-8 sets
(captured on both sides from the same router; fp32 softmax) vmap vs HF loop and each vs O1, as flips/token/layer, plus the
fraction of tokens whose margin `p_(k) − p_(k+1)` is below 1e-6 (true near-ties) and the exact-bf16-tie fraction; (m3)
per-parameter-group rel-L2 (q, k, v, o adapters); (m4) HF batched vs HF loop on the same batch (upstream's own bf16 spread);
(m5) loss absolute difference; (m6) the same with `router_fp32=True` on both sides (the deliberate pretraining-faithful
deviation, reported separately) and the bf16-vs-fp32-forward flip rate per layer.

**Acceptance (requirement (d)):** (A1) vmap-vs-HF-loop flips = **0** at equal precision (or every flip's margin < 1e-6); if not,
bisect — SDPA kernel choice via `all_valid_attention` (`runtime/masking.py:195-216`), MoE path (dense vs grouped vs Triton),
RMSNorm — before touching the DP design (pinning would mask, not fix); (A2) rel-L2(vmap, HF loop) ≤ 2 × rel-L2(HF batched, HF
loop) on the same batch, with flip-free examples showing the same drift as flip examples (accumulation order only); (A3) fp32:
rel-L2 ≤ 1e-5 and 0 flips on every configuration; (A4) PR #980's "≈ 1.3 %" figure reproduced to ±0.5 pp under this written-down
definition (its own script is untraceable, critic G2); (A5) surrogate identity on the checkpoint in fp32: batch mean of
per-example surrogate gradients with `f̃ = f(B)` vs HF `output_router_logits=True` gradient, rel-L2 ≤ 1e-4.

### 10.2 Statistics to collect (G3) — 256 KStack examples under the preset partition

| statistic | how | decides |
|---|---|---|
| per-example gradient-norm quantiles p10/p50/p90/p99/max | `clipped_grad(…, clipping_norm=1e9, return_aux=True).grad_norms` | `C_g` via the bias²+noise² curve (§4.5); report the clip rate at the chosen `C_g` |
| per-coordinate RMS imbalance `δ = ‖f(B) − k/E‖₂/(k/E)/√E` and `‖·‖_∞` for 32 batches of 256, plus batch-to-batch drift over 100 consecutive LoRA steps | router recorder, eager | whether the term carries signal (need `r_smoothed ≲ 0.3·δ`, §2.6 reading), the filter lag (lag bias ≤ noise), τ for the monitor |
| per-example `‖d(x)‖₂` distribution and fraction of experts with `h_e = 0` per example, pooled and per layer | same pass | tightness of the structural bound 2.646 (expect ≈ 1.0 balanced); H5 on real code at T = 1024 (phase-1 E5 was an artefact, critic R8); whether per-layer is ever worth ×5.29 |
| bf16-vs-fp32-forward router flip rate per layer on real code; logit scale (median `\|z\|`, top-8/9 margin) | m2/m6 | documents the opt-in fp32 router; predicts flip rates |
| dense vs grouped MoE step time and peak memory, microbatch 8 (frozen experts); dense-vs-grouped flip counter | `torch.cuda.max_memory_allocated`, wall clock | G6 default; acceptance: grouped ≥ 3× faster, flips 0 |
| aux/CE gradient-norm ratio at α = 1e-4 and 1e-3, on attention LoRA (and on router weights with the router unfrozen) | T3 machinery | H4 at scale; compare against the per-step DP noise norm 5.68 and the projected noise `nm·C_g/B̄ = 1.98e-3` in the aux direction |
| surrogate tracking error `‖f̃_t − f(B_t)‖/‖f(B_t) − k/E‖` over a 512-step dry run (noise from the mechanism, `f(B_t)` from the run's own batches — lab-only) | monitor mode | the lag/noise trade-off and shrinkage engagement |
| the preset's calibrated `nm_MF` at ε = 3 (the trainer computes it at start) | `_calibrate_noise` | re-scales every band-MF column of §2.6 by `nm_MF/0.5622` |

### 10.3 Mechanism acceptance on a 200-step DP run (ε = 3 calibrated, `surrogate`, ρ = 0.02, both stacks; DP-SGD/Poisson and
band-MF/b-min-sep)

(a) after warm-up (200 steps ≥ 2 EMA time constants) `f̃_t` tracks the validation-only `f(B_t)` with per-entry error ≤ 10 % of
`k/E`, and `‖d̃⁺_t − d_true‖/‖d_true‖ ≤ 0.25` whenever `‖d_true‖ ≥ s√63` (shrinkage engages only below that); (b) cosine between the
DP surrogate aux gradient and the exact batch aux gradient ≥ 0.8 whenever `δ ≥ 0.1`; (c) eval loss within run-to-run noise of the
feature-off run (the term must not hurt at 1e-4); expert-usage entropy on eval data not below the base model's; no expert with
usage < 0.25·k/E after training; (d) reported ε identical with and without the feature (in-stream); (e)
`group_norms["router_load_probe"]` max ≤ `C_h` (never clipped) — read from the aux inside the harness only; (f) throughput with
the grouped default within 20 % of the feature-off run; (g) `D_t` never trips under exact balance; the `monitor_then_surrogate`
switch changes neither `max_norm` nor `epsilon_at`; (h) per-example gradient rel-L2 vs the precision-matched oracle tracked every
100 steps stays ≤ 1.5 % with zero flips.

---

## 11. Risks, and what would falsify the design

1. **No usable signal on real data.** If §10.2 finds `δ ≲ 0.02–0.03` on KStack (a well-balanced checkpoint on in-distribution
   code), the smoothed noise at ρ = 0.02 (2.35 % DP-SGD / 0.83 %·(nm_MF/0.5622) MF) is comparable to the imbalance, the shrinkage
   zeroes the term and the release buys nothing. This does not falsify faithfulness (the true aux gradient is ∝ the same imbalance
   and equally silent) but it means acceptance (b) never triggers. Mitigation: ρ = 0.1 (×1.049), a longer filter, the independent
   draw (DP-SGD), or `monitor` only (OLMoE §4.3 and Tholoniat et al. 2024 evidence that dropping the aux in fine-tuning is benign,
   https://arxiv.org/abs/2409.02060 (with its noted internal inconsistency), https://arxiv.org/abs/2402.07334 — VERIFIED phase-1
   literature; transfer PLAUSIBLE).
2. **Lag bias.** If `f(B)` drifts faster than the filter's time constant (100 steps), `f̃` chases a stale target. Falsifier:
   batch-to-batch drift over 100 steps larger than `r_smoothed` (§10.2 row 2). Mitigation: shorter filter at a higher ρ.
3. **The surrogate is inert even when needed.** With the router trainable, if the aux/CE ratio (§10.2) is below the projected
   per-step noise and the `D_t` trajectory with α = 1e-4 does not differ from α = 0, the honest recommendation is Tholoniat's
   (drop the aux, freeze the router) and the surrogate is documentation-only. The design stays correct; its utility claim would be
   false.
4. **Recorder under checkpointing / compile.** The claim that captured router logits inside a non-reentrant checkpoint region
   carry gradient and are not double-appended is PLAUSIBLE until T5/T18 pass. Fallback: the forward-hook variant with
   overwrite-by-layer (VERIFIED toy) — graph-breaks under compile (fullgraph fallback exists).
5. **Flip-free vmap does not hold at 28 layers** (toy depth only). Falsifier: A1 fails. Then the drift is a kernel-selection or
   accumulation issue to bisect (SDPA path, MoE path, Triton accumulation dtype — PLAUSIBLE), not a DP-design issue.
6. **Grouped-MoE default changes numerics of existing runs** beyond the documented floor, and the process-wide first-patch
   capture can surprise multi-model processes. Falsifier: any `test_parity_harness.py` parity moving outside tolerance, or a
   dense-vs-grouped flip count > 0 on the real model → keep dense default, fix the kernel.
7. **The `nm_MF` caveat.** Every absolute band-MF number in §2.6/§6 is at nm = 0.5622; the preset's calibrated `nm_MF` is
   PLAUSIBLY several × larger, so MF load errors are correspondingly larger. Ratio claims are unaffected. §10.2 records `nm_MF`.
8. **Probe leaf and the optimizer.** If a future optimizer treats a zero gradient differently (e.g. decoupled weight decay on a
   nonzero parameter), the probe could drift. T10 guards it; the `_augment_inputs` assertion turns silent drift into a hard error.
9. **AUTO-S users lose the feature in v1**; falsifier of the restriction's necessity is the `fixed_groups` engine change (§9.4),
   deferred, not blocking. AUTO-S's own utility claim is PLAUSIBLE (Bu et al. 2023, not re-fetched).
10. **fp32 router opt-in moves the fine-tune away from the HF-bf16 serving router.** By design and off by default; the oracle
    carries the same setting. Falsifier of the *default*: eval with bf16 routing after an fp32-router fine-tune degrades measurably
    vs bf16-router training — then document the trade-off, do not flip the default.
11. **Telemetry leakage.** Any future PR that adds `h(x)`, `P(x)`, per-example aux or the probe group's norms to
    `loss_aux`/`group_metrics` silently releases un-noised statistics. Guard: T17, and a review-checklist line under the DP
    protocol's "Composition and accounting". The pre-existing un-noised `loss`/`grad_norm`/`clip_rate` logging (F11) sits next to
    this design and should be documented or noised — outside this task.
12. **Second-moment streams.** Forgetting the exclusion would silently spend budget on a squared load stream — a utility bug, not
    a privacy bug (the allocation stays `gaussian(nm)`). v1 raises; T20 guards v1.1.
13. **Mahalanobis allocation is MSE-optimal only for equal group dimensions** (`noise_allocation.py:55-58`); privacy is
    unaffected (equality holds regardless, VERIFIED), but the split between a 64-dim group and an 8 M-dim group is chosen by ρ,
    not by the allocator. Utility only.
14. **Faithfulness is to Fact A, not to HF Trainer.** A reviewer comparing against an HF-Trainer-with-accumulation run will see a
    different aux (per-microbatch `f`, `G·α`, critic Exp A). Stated explicitly; if the product decision were "match HF Trainer's
    realised objective", set `α_eff = G·α` and accept microbatch-level `f`, which the same mechanism can emulate only approximately.
15. **Concurrent releases under DP-FTRL.** The design deliberately avoids a separate per-step load mechanism under MF (would need
    concurrent composition, Vadhan–Wang / Vadhan–Zhang, theorem numbers not verified). Falsifier: someone later adds an
    independently sampled load release under b-min-sep — a different participation model that must be accounted separately.
16. **DPO pair pooling** has not been exercised; a wrong per-pair divisor would change the aux weighting but not privacy (the
    structural bound holds for any masked mean). T21 and the `monitor` preset default contain it.
17. **Experts-trainable variant (M5)**: PEFT `target_parameters` under `functional_call`+vmap may not work; the mechanism is
    unaffected, the preset is.
18. **Adjacency/normalisation under b-min-sep**: the design assumes `normalize_by = expected_batch_size` equals the sampler's
    per-step expected batch (critic M11; `_b_min_sep.py:6-11` constructs it so). Falsifier: a unit test comparing
    `BMinSepSampler`'s per-step expectation with `ctx.expected_batch_size` — a pre-existing trainer question; the centred release's
    unbiasedness does not depend on it beyond the scale factor.
19. **Unverified at scale.** Every magnitude except the accounting table, the allocator identity and the MF filter factors comes
    from random-init toys; requirement (c) is settled only by §10.2, and requirement (d) only by §10.1.

**Sources relied on** (all VERIFIED from primary text by phase-1 `critic` R6 / `literature` unless marked; I did not re-fetch):
Switch Transformer eqs. (4)–(6) https://arxiv.org/abs/2101.03961 §2.2; Andrew et al. 2021 Thm 1 https://arxiv.org/abs/1905.03871;
Zhu–Dong–Wang Def. 7 / Thm 10 https://arxiv.org/abs/2106.08567; Feldman–Shenfeld Lemma 3.2 / Thm 3.3 https://arxiv.org/abs/2602.17284
(as cited by `src/amplification/poisson.rs`); Denisov et al. Thm 2.1 https://arxiv.org/abs/2202.08312; Dong–Roth–Su Thm 2.7
https://arxiv.org/abs/1905.02383; Dong & Ganesh b-min-sep https://arxiv.org/abs/2602.09338 (as cited by `_b_min_sep.py`); ST-MoE
§3.1 eq. (5) https://arxiv.org/abs/2202.08906; Mellum 2 Technical Report https://arxiv.org/abs/2605.31268 §3.6 (running-average f),
§5.1.2 (SFT α = 1e-4), §5.2 (train/inference route disagreement), appendix (FP32 router); Tholoniat et al.
https://arxiv.org/abs/2402.07334; OLMoE https://arxiv.org/abs/2409.02060; Wang et al. 2024 https://arxiv.org/abs/2408.15664;
DeepSeek-V3 https://arxiv.org/abs/2412.19437; Davody et al. 2020 https://arxiv.org/abs/2006.10919; Ponomareva et al. "How to DP-fy
ML" §5 (VERIFIED phase-1 literature B2; URL not re-fetched here); Bu et al. 2023 AUTO-S https://arxiv.org/abs/2206.07136
(PLAUSIBLE, theorem numbers not re-fetched). Megatron's per-layer averaging (§1.3 row (c)) is PLAUSIBLE — Megatron source not read.
