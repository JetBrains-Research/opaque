# Mellum2 under Opaque DP — final design specification v2 (post-refutation revision)

Agent: `reviser`. Repo `/home/user/opaque` @ `ef1abc5` (branch `claude/mellum-dp-representation-r6slaz`), nothing tracked
modified. Supersedes `phase2-final-design.md` (v1). This document is self-contained: it restates the whole design with every
refutation from `phase2-refute-{sensitivity,composition,feasibility}.md` applied, rejected with evidence, or narrowed, and it
ends with §12, a log of every refutation and its disposition. Where v1 made a claim that fell, the claim is removed or
rewritten *here*, never silently kept.

Evidence tags. **VERIFIED** = I read the cited lines with `sed -n`/`grep` or ran the cited script in this session (my scripts:
`scratchpad/research/reviser/checks.py` → `checks.out`; re-runs of `refute-composition/corr_coins.py` and
`refute-feasibility/nm_bracket.py`); **VERIFIED (agent)** = established by that earlier agent's executed script or read, not
repeated by me; **PLAUSIBLE** = derived or read, not executed end to end. Repo paths relative to `/home/user/opaque`; HF =
`.venv/lib/python3.11/site-packages/transformers/` (5.16.1). Theorem citations are those phase-1 `critic` R6 extracted from the
primary PDFs; I fetched no paper and cite nothing beyond what phase 1 read.

What changed from v1 in one paragraph. Three major refutations were accepted: (i) the `mellum2-kstack` preset is **ragged,
not packed**, so the aux term now carries a public-constant token weight `w_x = T_x/T̄` (exact HF direction under any lengths;
v1's "equal weights" identity is restricted to packed data); (ii) the two presets are **manual functional loops**, so the
mechanism is factored into a trainer-independent helper with four seams and the loops call it; (iii) the chunked forward that
carries `opaque_router_logits` is installed only when `fused_linear_cross_entropy=True` is passed — v1's marker detection would
have been swallowed by HF's `**kwargs`; detection is now by named parameter only and the trainer sets the flag itself; plus
(iv) a pre-existing **DDP + checkpoint-resume** sampler-key bug that voids the tight Poisson/b-min-sep amplification is now a
stated precondition of the "as run" claim (privacy statement narrowed until the trainer fix lands). Fifteen minor refutations
were accepted (telemetry rows, value-neutral surrogate, per-layer carrier, dead-zone shrinkage, EMA bias correction, guard
size, mask binarisation, capture-count normalisation, calibration split, per-group construction, memory numbers, `nm_MF`
bracket, process-boundary comparisons, dense fp32 on CUDA, v2 key statement), one was narrowed (all_valid_attention: caveat
added, kernel choice made public), none rejected. The core mechanism — one joint per-group Gaussian / matrix release on the
same sampled batch, accountant literally unchanged, `√(1+ρ)` gradient-noise price — survived every attack and is unchanged.

> **Errata (added after the phase-3 validation; the body below is the reviser's v2 text unchanged).**
> 1. Section 8.1 cites `_dp_trainer.py:3750-3757` for both quotes; the "privacy accounting is unaffected" sentence is at `:3741-3744`, the resume caveat at `:3750-3756`, and the `DPTrainer.__init__` docstring at `:953-968` makes the same wrong claim. The P0 fix must apply the rank fold *after* `from_state_dict` (the restore overwrites the template key at `_poisson.py:261` / `_b_min_sep.py:259`). The impact is under-stated: per-step `epsilon` at the accountant's `delta` is 10.5 instead of 3, and a rigorous whole-run lower bound is `epsilon(1e-6) >= 6.0` (`scripts/phase3/verifier/k1_run_bounds.py`).
> 2. Section 2.6, the "pay in epsilon" column: the printed values (3.234 at rho = 0.02, 3.703 at rho = 0.1, ...) come from `poisson(gaussian(nm) | gaussian(nm_h), q) * T`, which the accountant routes through its generic discretised Poisson path; the tight values for the same-coin joint Gaussian `poisson(gaussian(nm_eff), q) * T`, `nm_eff = nm sqrt((1 + rho)/(1 + 2 rho))`, are 3.126 / 3.590 (5.305, 4.100, 3.306, 3.064 at rho = 0.5, 0.2, 0.05, 0.01). The shipped route (epsilon held, gradient noise x sqrt(1 + rho)) is unaffected (`scripts/phase3/verifier/k2_composed_inner_v2.py`).
> 3. Section 2.1, replace-one: the general per-layer bound is `sqrt(2 min(k, E - k) L)`, equal to `sqrt(2 k L)` for `E >= 2k` (the stated form, tight); `sqrt(2 k L (1 - k/E))` is not a bound (`scripts/phase3/verifier/k3_structural_bound.py`).
> 4. `rho = 0.02` is a preset-regime default; at other `(B_bar, nm, beta, k/E)` use the rule "smallest rho whose smoothed load noise is below about 30 % of the expected imbalance" and print `s_inf/(k/E)` and the dead-zone threshold at setup (`scripts/phase3/prototype/`). The dead-zone false-pass rate `P(chi^2_{E-1} > 2E)` is `E`-dependent (4.2e-6 at E = 64, about 5e-2 at E = 8).
> 5. Citation drifts: `modeling_mellum.py:474` is `@merge_with_config_defaults`, `@capture_outputs` is `:475`; `kernels/moe.py:614-632` is the non-CUDA `else` branch, the CUDA-fp32 dense fall-through is `:606-607, 633`; the manual-loop seam "between `noise_fn` and `opt.update`" is `train_dpftrl.py:2061-2076`, not `:1735-1760` (the noise-factory construction); T8's "adversarial single-expert routing" means routing every token to the same `k` experts.


---

## 0. The design in one page

### 0.1 Result

Mellum2's HF training objective has exactly one term that is not per-example separable: the Switch-style load-balancing loss
`L_aux(B) = E·Σ_e f_e(B)·P_e(B)` (VERIFIED (synthesizer) HF `modeling_mellum.py:540-606, 692-700`; Switch Transformer eqs. (4)–(6),
https://arxiv.org/abs/2101.03961 §2.2, VERIFIED phase-1 math). `f` is argmax-derived, so its gradient is zero a.e. and

```
∇_θ L_aux(B) = Σ_{x∈B} ∇_θ S(x; f̃)|_{f̃ = f(B)},    S(x; f̃) = E · (T_x/T_tot) · Σ_e f̃_e · P_e(x)      (exact; F3)
```

(VERIFIED phase-1 math §1.2 in float64; VERIFIED (refute-feasibility) check F on the *patched* model with ragged lengths:
gradient rel-L2 2.2e-7). The only thing the DP path needs that a per-example pipeline lacks is a public constant `f̃_t` close to
the token-weighted batch load `f(B)`, and a public stand-in for the batch token count `T_tot`. The design obtains `f̃_t` as a **DP
release of the per-example, token-weighted, centred load vector `w_x·d(x)`, `d(x) = h(x) − k/E`, `w_x = T_x/T̄`** (`T̄` a public
constant, default the row length `T_max`), carried as a second `PerGroup` group of the *same* clipped pytree through a zero
"probe" parameter, noised by the *same* Gaussian / matrix mechanism as the gradient. Opaque's per-group allocation makes the
joint release exactly one sensitivity-`1/nm` Gaussian, so **the accountant call is literally unchanged under both stacks**
(`poisson(gaussian(nm), q)*T` and `b_min_sep(mf_gaussian(nm, band_mf_strategy(64, 0.95)), n_steps, p0)`), and the whole price
is a `√(1+ρ)` inflation of the gradient noise, `ρ = C_h/C_g` the share of the clipping budget given to the load group. At the
default **ρ = 0.02 that is ×1.010 gradient noise, ε unchanged**. The carrier is the per-layer `(L, E)` vector (same privacy, same
gradient price, and the pooled estimate has *identical* noise to a pooled release — VERIFIED `checks.py`: 0.04149 vs 0.04149);
the noised leaf is post-processed (layer pooling, sum-zero projection, bias-corrected EMA β = 0.99 with an exactly known noise
std, a dead zone at `2·E·s²` followed by positive-part James–Stein shrinkage, clamp) into `f̃_{t+1}`. Under band-MF the
anti-correlated noise makes the smoothed estimate ≈ 2.8× more accurate than under DP-SGD at equal base σ (0.0249 vs 0.0709,
VERIFIED (synthesizer, refute-sensitivity)). The same leaf gives a free public **imbalance monitor** `D_t` (pooled and per
layer) with a decision rule, so the surrogate can be switched on mid-run with no accountant or MF-latch change. Since
`Σ_e P_e(x) = 1`, the surrogate gradient is `α·E·w_x·Σ_e (f̃_e − k/E)·∇P_e(x)`: proportional to the *imbalance*, vanishing at
balance — exactly what the real objective does. The dead zone makes "degrades to zero when the true imbalance is below the
noise floor" true with per-step false-pass probability 4.2e-6 (χ²₆₃ tail at 2·63; expected 0.07 false passes over the
15 625-step horizon, VERIFIED `checks.py`) — v1's unqualified "never" is withdrawn (§12 S5).

### 0.2 What the four requirements of the brief get

| requirement | how it is met | status |
|---|---|---|
| (a) provably DP, accounting matches what runs | one joint Gaussian / MF release per step with per-group bounds `(C_g, C_h)`, structural (never active) bound on the load group, Mahalanobis allocation `Σ(C_i/σ_i)² = 1/nm²` (VERIFIED with the real allocator by three agents), Poisson / b-min-sep subsampling of the *one* joint mechanism (shared coin), adaptive use of `f̃_t` covered by adaptive composition; accountant unchanged; hygiene table §8 | VERIFIED (mechanism). **"As run" holds for single-process runs and for DDP runs that are never resumed from a checkpoint** until the pre-existing resume-key bug (§8.1, §9.0 P0) is fixed; the residual floating-point kernel-selection effect is named in the privacy statement (§8.2) |
| (b) faithful to the real objective incl. the batch term | target = HF logical-batch formula (Fact A): token-weighted pooling over all layers, attention mask, executed routes, α = `model.config.router_aux_loss_coef` by default, presets pinned to Mellum2's own SFT value 1e-4. The batch mean of `∇ℓ_x` equals `(T_tot/(B̄T̄))·∇[α·L_aux^HF]` **in direction exactly** for *any* length pattern (cosine 1.000000000000, VERIFIED `checks.py`), with the realised scale `T_tot/(B̄T̄)` (= 1 under packing; mean 1 when `T̄` is the true mean length). CE keeps Opaque's example-mean convention (pre-existing, F4) | VERIFIED (identity, direction); the scale factor and the CE convention are stated, not hidden (§1.2) |
| (c) usable clipping norm / noise | load leaf is its own group with a structural bound — `C_g` is chosen exactly as for any dense LoRA fine-tune; ρ = 0.02 costs ×1.010; the aux term moves per-example norms < 0.1 % and is 0 at balance | VERIFIED (toy), PLAUSIBLE (trained checkpoint, §4/§10) |
| (d) numerically stable vs the non-DP HF path | precision-matched oracle; vmap vs eager vs module backward rel-L2 **0.0** at equal precision on the patched model (VERIFIED (refute-feasibility) check B); residual bf16 drift is accumulation order inside HF's own batched-vs-loop spread; fp32 router is an opt-in *pretraining-faithfulness* choice; attention-kernel selection is made a public property so per-example gradients do not depend on microbatch-mates' padding (§8.2) | VERIFIED (toy), acceptance criteria §10 |

### 0.3 Decisions where the designs disagreed and the judges did not settle it (v1 table, updated where refutations moved it)

| question | **decision** and why |
|---|---|
| released statistic | **centred, token-weighted `λ·w_x·d(x)`**, `w_x = T_x/T̄`: centring gives the 6.5 % smaller bound (2.646 vs 2.828) and `Σ_e = 0` structurally (no renormalisation); the token weight is what makes the surrogate collinear with HF's aux gradient under ragged rows (§1.2; refute-feasibility R1 accepted) |
| carrier shape | **per-layer `(L, E)` probe** with pooling in post-processing (refute-sensitivity R7 accepted): pooled accuracy identical to a pooled release, per-layer entries free at ×√L noise; state 1792 floats |
| budget share ρ | **0.02** default (×1.010); **0.10** when the router or experts are trainable (×1.049) |
| filter | **bias-corrected EMA β = 0.99** under both stacks (refute-sensitivity R6 accepted), window mean selectable |
| α default | `None` → `model.config.router_aux_loss_coef` (1e-3) in `surrogate`; presets pinned to 1e-4 (Mellum2 SFT, TR §5.1.2); α = 0 in `monitor` |
| feature default | trainer default `off` (bit-identical); TRL converter maps `router_aux_loss_coef > 0` → `surrogate` on a supported family; presets `surrogate` (SFT) / `monitor` (DPO) |
| fp32 router | opt-in, default off (pretraining-faithful, tie-robust; not a drift fix) |
| AUTO-S / adaptive with the load leaf | v1: `ConfigurationError`; v2: engine `fixed_groups` (§9.4) |
| `f̃` delivery into vmap | closure tensor (`_remove_unused_columns` prunes seeded columns under plain `DPTrainer`, VERIFIED (synthesizer) `_dp_trainer.py:3446-3470`) |
| checkpoint | sidecar `router_load_state.pt` |
| statistics path | chunked-CE **named** kwarg + HF `OutputRecorder` (recorder under non-reentrant checkpointing now VERIFIED at toy scale by two agents: no double append, gradient carried exactly) |
| DPO | mechanism in scope (§7); `mellum2-codesec` ships `monitor` |
| second-moment streams | v1: `ConfigurationError`; v1.1: engine-side structural zero (refute-composition R6 accepted; v1's "0.0 bound" plan withdrawn) |
| independent forward-only Poisson draw | v2 opt-in, DP-SGD only, with the key statement corrected (§2.7) |
| C-calibration pass | **a held-out split disjoint from the protected set**, or the accounted quantile release (refute-sensitivity R8 accepted; "KStack is public" wording withdrawn) |
| where the code lives | **trainer-independent helper** (`opaque.api.patches.transformers.components.moe_stats` for statistics; `opaque.api.transformers.moe_load` for probe group / state / update / filter factors) consumed by `DPTrainer` *and* by the two manual example loops (refute-feasibility R2 accepted) |

---

## 1. Objective (G1): the exact per-example loss

### 1.1 Definitions

Example `x` (one collated row; right-padded — both repo collators right-pad, VERIFIED phase-1 critic R10; the `mellum2-kstack`
preset tokenises with `truncation=True, max_length=1024` and **no packing**, `examples/train_dpftrl.py:1090-1094`, and the
collator pads, `:1130-1135`, VERIFIED — rows are ragged). Binary mask `m_{x,t} = 1{attention_mask_{x,t} ≠ 0}` (binarised inside the
helper; §12 S4), `T_x = Σ_t m_{x,t}`, layers `l = 1..L` (L = 28), experts `e = 1..E` (E = 64), `k = 8`, public row length
`T_max` (1024 at the presets), public mean-length constant `T̄` (`router_load_mean_tokens`, default `T_max`). For every layer the
router computes `z^l_{x,t} = W_l · h^l_{x,t}` (bf16 `F.linear` upstream), `p^l_{x,t} = softmax_fp32(z^l_{x,t})`, executed set
`S^l_{x,t} = topk_k(p^l_{x,t})` (VERIFIED (synthesizer) `modeling_mellum.py:332-341`).

```
P_e(x; θ)   = (1/(L·T_x)) Σ_l Σ_t m_{x,t} · p^l_{x,t,e}               per-example mean router prob (differentiable)
h^l_e(x; θ) = (1/T_x) Σ_t m_{x,t} · 1{e ∈ S^l_{x,t}}                  per-layer load fraction (∇ = 0 a.e.)
h_e(x)      = (1/L) Σ_l h^l_e(x)                                    pooled load fraction
d^l(x)      = h^l(x) − (k/E)·1,   d(x) = (1/L) Σ_l d^l(x)             centred; Σ_e d^l_e(x) = Σ_e d_e(x) = 0
w_x         = T_x / T̄                                              public-constant token weight, 0 ≤ w_x ≤ T_max/T̄
S_x         = E · w_x · Σ_e ( f̃_{t,e} − k/E ) · P_e(x; θ)              surrogate (differentiable in θ)

ℓ_x(θ; f̃_t) = CE_x(θ) + α · ( S_x − sg[S_x] ) [ + ζ·( Z_x − sg[Z_x] ) ] + ⟨ z, λ · w_x · d^{(L,E)}(x) ⟩_{sg}
```

`sg[·]` = `detach`. The bracketed terms are **value-neutral**: `S_x − sg[S_x]` is exactly 0 in value (`a − a = 0` exactly in
floating point for finite `a`) and has gradient `∇S_x`; hence `ClippedGradAux.loss_values` carries exactly `CE_x` and the
trainer's un-noised logged `loss` mean (`_dp_trainer.py:2249`, VERIFIED) is not widened by this design (refute-composition R2
accepted). `z ∈ R^{L×E}` is a zero-valued probe parameter; its term is identically 0 in value and contributes only
`∂ℓ_x/∂z = λ·w_x·d^{(L,E)}(x)` (VERIFIED phase-1 primitives E2 and refute-feasibility check A: other leaves unaffected,
`∂ℓ/∂z = λd(x)` to 1e-7). If `T_x = 0` (fully masked row) the helper returns `d^{(L,E)}(x) = 0` and `P(x) = 0`
**explicitly** (a `where(T_x > 0, ·, 0)` on the centred quantities — note that a bare `T_x.clamp(min=1)` would give
`h = 0` and hence `d = −k/E·1`, a spurious non-zero contribution): the example contributes nothing to the release or the
surrogate, which is also what the engine's NaN sanitiser would produce for the 0/0 form (`_pytree.py:489-527`, VERIFIED
(refute-sensitivity `edge2.py` / refute-composition U9)); T2 covers it so the property does not depend on the sanitiser.

- `CE_x` = HF's per-example token-mean causal-LM CE over label-valid tokens, exactly what `DPTrainer.compute_per_example_loss`
  reads from `fmodel(params, **inputs)["loss"]` (`_dp_trainer.py:2314`, VERIFIED (synthesizer)) — through the chunked LM-head
  path **when it is installed** (§9.1: it is installed only if `fused_linear_cross_entropy=True` reaches `apply_model_patches`,
  `_factory.py:378-392`, VERIFIED; the trainer sets it when the feature is on).
- `Z_x = (1/(L·T_x)) Σ_{l,t} m_{x,t}·(logsumexp_e z^l_{x,t,e})²` is the router z-loss (ST-MoE §3.1 eq. (5),
  https://arxiv.org/abs/2202.08906, VERIFIED phase-1 literature C2), opt-in with `ζ = 0` by default (§7), value-neutral like `S_x`.
- Centred vs uncentred surrogate: `Σ_e P_e(x) = 1` ⇒ identical gradients (VERIFIED phase-1 math §1.1 / empirical E4).

### 1.2 Faithfulness claim (exact, now for any length pattern)

Let `T_tot = Σ_{x∈B} T_x`. HF's aux pools `f` and `P` over tokens with one denominator `L·T_tot` (`modeling_mellum.py:575-603`,
VERIFIED (refute-feasibility)), i.e. `f(B) = Σ_x (T_x/T_tot) h(x)`, `P(B) = Σ_x (T_x/T_tot) P(x)`. Then

```
(1/B̄) Σ_{x∈B_t} ∇ℓ_x  =  (1/B̄) Σ_x ∇CE_x  +  α · (T_tot/(B̄·T̄)) · ∇L_aux^HF(B_t)|_{f = f̃_t}
```

exactly (linearity in `w_x`; VERIFIED `checks.py`: cosine 1.000000000000, scale `T_tot/(B̄T̄)` to 6 digits; the `T_x/T_tot`
form is VERIFIED (refute-feasibility) check F on the patched model at 2.2e-7). Consequences, stated rather than hidden:

- **Direction**: the aux gradient direction is HF's for every batch, ragged or packed. v1's "equal example weights" surrogate is
  off by 15.6 % rel-L2 at a 16/12/10/16 length spread (VERIFIED (refute-feasibility) check F) and is withdrawn (§12 F1).
- **Scale**: the realised factor `T_tot/(B̄T̄)` has mean `T̄_true/T̄` and relative sd ≈ 6 % (the same kind of factor Poisson's
  `|B_t|/B̄` already puts on the whole gradient). With `T̄ = T̄_true` it is 1 in expectation; with the default `T̄ = T_max` it is
  `≤ 1` (the aux is down-weighted by the mean length fraction — within the 1e-4…1e-3 uncertainty of α itself). Under packing
  (`T_x ≡ T_max`) it is exactly 1.
- **`f̃_t` scale**: the released `d̂_t = (1/B̄)Σ_x w_x d(x)` is unbiased for `(T̄_true/T̄)·d_tw`, `d_tw = f(B) − k/E` HF's
  token-weighted centred load (VERIFIED arithmetic). So the total aux weight relative to HF is `(T̄_true/T̄)²`; presets that
  measure `T̄` on the public held-out split (§4.4) get 1; presets that leave `T̄ = T_max` get a documented `(T̄_true/T_max)²`.
- **CE weighting**: Opaque's per-example token-mean CE with equal example weights (`opaque-alignment/.../sft/loss/_nll.py:15-24`,
  VERIFIED phase-1 math) differs from HF's token-weighted CE unless `T_x ≡ T` — pre-existing (F4), unchanged by this design, and
  stated in the mechanism page. A public-`N̄` token-weighted CE variant is representable (phase-1 divergence §4) but changes the
  per-example norm distribution (∝ `T_x`) and is out of scope.
- With a lagged DP estimate `f̃_t` the aux gradient is the exact gradient of the same functional form at a nearby load vector;
  §2.6 prices "nearby".

### 1.3 Target of faithfulness and why (the four candidates of critic G1)

| candidate | verdict |
|---|---|
| **(a) HF logical-batch pooled formula** — one denominator `L·T_tot` across all layers, `Σ_e f_e = k`, attention mask (VERIFIED (synthesizer) `modeling_mellum.py:575-603, 694-697`) | **TARGET.** The objective the released checkpoint declares (`router_aux_loss_coef = 0.001`, `configuration_mellum.py:103`), the only one reproducible from the HF artefact. |
| (b) HF-Trainer-realised: per-*microbatch* `f`, effective coefficient `G·α` (VERIFIED phase-1 critic Exp A) | rejected — an artefact of `trainer.py:1961-1963`; the DP path has no accumulation (`gradient_accumulation_steps` hard-wired to 1, `_training_arguments.py:1314-1316`, VERIFIED (refute-composition)), so it is *more* faithful to (a) than HF Trainer itself. |
| (c) Megatron per-layer running-average `f` (Mellum2 TR §3.6, arXiv:2605.31268, VERIFIED phase-1 literature F.2) | rejected as *target* **on faithfulness grounds only**: mean of per-layer products vs HF's product of pooled means — not reproducible from the HF artefact. **Not** on cost grounds (v1's "×5.29 relative noise at equal privacy" was wrong for the release, §12 S7): the per-layer `(L, E)` vector *is* the carrier, per-layer entries `f̃^l` are free at ×√L per-entry noise, and a per-layer surrogate is an opt-in (`router_aux="per_layer"`) priced in §2.6. Its lagged running-average character is adopted. |
| (d) per-sequence aux (`f(x)` from the example's own tokens; DeepSeek-V2/V3, Wang et al.) | rejected as a stand-in: a materially different regulariser (cos 0.26–0.39 to the batch gradient at balance; VERIFIED phase-1 math §2 / empirical E4). Opt-in flag, never labelled "faithful". |

### 1.4 The remaining G1 sub-decisions

| item | decision | reason |
|---|---|---|
| mask for `h`, `P` | the collator's attention mask, **binarised** (`m = attention_mask != 0`, 2-D asserted) | HF multiplies the float-cast mask (`modeling_mellum.py:581`, VERIFIED (refute-sensitivity)); any non-negative weighting keeps the bound but a same-sign additive mask would score *padding* and a mixed-sign mask breaks the bound (`‖d‖ = 3.18 > 2.646`, VERIFIED (refute-sensitivity) `edge3.py`). Binarising makes the contract explicit. **`attention_mask=None`** (what the kstack manual loop passes today, `train_dpftrl.py:1377`, VERIFIED) means every position counts — HF-faithful for that call but different from the collator path; the manual loop is changed to thread the mask (§9.2), T2 covers `None`. |
| example weighting of the aux | `w_x = T_x/T̄`, `T̄` public (default `T_max`) | §1.2 |
| CE weighting | equal example weights (Opaque convention, pre-existing F4) | §1.2 |
| coefficient | `router_aux_loss_coef: float \| None = None` → `model.config.router_aux_loss_coef` (1e-3) in `surrogate`; both presets pin **1e-4** | Mellum2's own SFT coefficient (TR §5.1.2) |
| `f` from executed routes | `h(x)` from the same fp32 softmax tensor and the same `topk` op the router ran (`modeling_mellum.py:333-336`) | identical input ⇒ identical indices (VERIFIED (refute-feasibility) U12); HF's bf16-softmax aux top-k differs on 0.03–1.7 % of tokens (phase-1 critic R9) — deviation ≤ 1e-3·k/E |
| pooled vs per-layer | carrier `(L, E)`; **surrogate uses the pooled `f̃`** (HF); per-layer surrogate opt-in | §1.3(c), §2.6 |
| initial value | `d̃_0 = 0` ⇒ `f̃_0 = k/E` | aux gradient identically zero at step 0; bias-corrected EMA (§2.4) makes `f̃_1` the first release itself |
| z-loss | opt-in `ζ = 0` | not in HF's objective; per-token separable; zero DP cost |
| capture count | `h` normalised by `len(router_logits)`·`T_x`, and `assert len(router_logits) == config.num_hidden_layers` | a duplicate capture (2L tensors) would otherwise make `Σ_e h = 2k`, the probe clip fire at 2.035·C_h and the release biased 2× (VERIFIED (refute-sensitivity) A8); with the `(L, E)` carrier a duplicate capture also fails the reshape loudly |

---

## 2. Mechanism and accountant (G4)

### 2.1 Adjacency, protected unit, sensitivities

Protected unit = one training example (one collated row = one truncated file at the kstack preset; for DPO one preference
pair). **Packing would change the protected unit to a 1024-token row that may contain pieces of several files and split a long
file across rows — a weaker unit for long files; it is therefore *not* the preset default and, if chosen, must be stated in the
privacy statement.** Adjacency = **add/remove**, the repo default (`.junie/differential-privacy-review.md` "Adjacency";
`clipped_grad` contract `_clipped_grad.py:135-145`, VERIFIED (synthesizer)). Divisor = the public expected batch size
`B̄ = a.train_batch_size` (`normalize_by=expected_batch_size`, `_dp_trainer.py:1394, 4266-4290`, VERIFIED (synthesizer, refute-sensitivity)),
never the realised batch.

Constants: gradient bound `C_g` (preset 0.9, `examples/train_dpftrl.py:611-612`), ratio `ρ = C_h/C_g` (default 0.02), public
`T̄` and `T_max`, structural per-layer load bound `Δ_L = (T_max/T̄)·√(k·L·(1−k/E))` (= 14.0 at `T̄ = T_max`, L = 28; pooled
`Δ_h = (T_max/T̄)·√(k(1−k/E))` = 2.6458), probe scale `λ = ρ·C_g/Δ_L`, group bound

```
C_h = λ·Δ_L·(1 + g),   g = 1e-3   (device-independent guard; v1's 1e-6 is withdrawn, §12 S3)
```

The guard exists because `clip_pytree` shrinks every ratio by `2·(u_store + norm_roundoff)` before `min(1, ·)`
(`_pytree.py:111-131, 135-148`, VERIFIED) and `norm_roundoff` scales with the leaf count and the widest leaf of the whole pytree
and with the sum-of-squares accumulator dtype (`_pytree.py:214-224`, VERIFIED): 1.8e-7 on CPU/CUDA (fp64 accumulator) but
**1.3e-4 on MPS** (fp32 accumulator) for the preset partition (VERIFIED (refute-sensitivity) A10). `g = 1e-3` is ≥ 7× the largest
measured shrink, moves ρ by 0.1 % (σ_h by 0.05 %), and T8 asserts on every CI device that the sup example is released with scale
exactly 1. The implementation should additionally assert at setup `g > 2·(u_store + norm_roundoff)` using the engine's own
round-off bound if it is exposed.

Per-record L2 bounds (both hold for **every** prefix of previous outputs because neither depends on `f̃_t` — the condition
adaptive composition needs):

| group | bound | why |
|---|---|---|
| gradient leaves (`fallback`) | `C_g` | clipping `min(1, C_g/‖g‖)` per group (`_pytree.py:439-477`, VERIFIED) |
| load leaf `λ·w_x·d^{(L,E)}(x)` (`router_load_probe`) | `C_h/(1+g) = λΔ_L` | per layer `0 ≤ h^l_e ≤ 1`, `Σ_e h^l_e = k` ⇒ `‖h^l‖² ≤ k`, `‖d^l‖² ≤ k(1−k/E) = 7`; over L layers `‖d^{(L,E)}‖² ≤ 7L`; `w_x ≤ T_max/T̄` (VERIFIED phase-1 math §3; A1–A5 attacks through the real clipper VERIFIED (refute-sensitivity): sup attained at 0.99999896·C_h, never exceeded, any non-negative mask). **Structural — the `PerGroup` entry is a bound, never an active clip — the release is unbiased** and `clip_rate` on that group is 0 by construction (given the binarised mask, the capture-count assert and the guard `g`). |

Replace-one: the gradient bound doubles to `2C_g`; for the load group the tight replace-one bound is `λ·(T_max/T̄)·√(2kL)`
(attained, VERIFIED (refute-sensitivity) A4 for the pooled case: `√(2k) = 4.000`), tighter than the generic doubling. Nothing in
code changes either way — Opaque documents the add/remove bound and the doubling.

### 2.2 Per-step mechanism — DP-SGD / Poisson

Public state entering step `t`: `θ_t`, `d̃_t ∈ R^E` (smoothed pooled imbalance), `d̃^{(L,E)}_t` (per-layer, monitor only),
`s_t` (known noise std of `d̃_t`), `f̃_t = clamp(k/E + shrink(d̃_t), 0, 1)`.

1. **Sample** `B_t ~ Poisson(q)`, `q = B̄/N` (existing sampler; under DDP each rank draws its shard with a rank-folded key,
   `_dp_trainer.py:3802-3803`, VERIFIED — see §8.1 for the resume caveat).
2. **Augment** (once per step, outside vmap, `_augment_inputs`, `_dp_trainer.py:2302-2313`, VERIFIED (synthesizer)): copy `f̃_t`
   into the loss closure's device tensor; assert the probe parameter is zero.
3. **Per example inside `vmap(grad_and_value)`**: forward with `opaque_router_logits=True` (§9.1) → `CE_x` and the tuple of L fp32
   router logits `(T, E)`; `p`, executed `S`, `P(x)`, `h^{(L,E)}(x)`, `d^{(L,E)}(x)`, `w_x`; loss `ℓ_x` as in §1.1.
   `∂ℓ_x/∂z = λ·w_x·d^{(L,E)}(x)`; `∂ℓ_x/∂θ` = CE + surrogate (+ z-loss); value = `CE_x` exactly.
4. **Per-group clip**: `PerGroup` built by direct construction (§9.2 — **not** by a `per_group(..., router_load_probe=…)` pattern,
   whose substring matching would collide with any user pattern such as `"router"` and raise, `_per_group.py:143-158`, VERIFIED;
   §12 C4): gradient groups scaled by `min(1, C_g/‖g_x‖)`; probe group never rescaled (structural). Sum over `B_t`, divide by
   `B̄` (`clipped_grad(..., normalize_by=B̄)`; stored bounds `C_g/B̄`, `C_h/B̄`).
5. **Noise** (`gaussian_noise` → `per_group_noise_stddev` for a `PerGroup` max_norm, `_gaussian.py:320-322`,
   `noise_allocation.py:103-110`, VERIFIED (synthesizer, refute-sensitivity, refute-feasibility)): `σ_i = nm·√(B_i·Σ_j B_j)`, with
   `S = C_g + C_h`:

   ```
   σ_g = nm·√(C_g·S)/B̄ = nm·C_g·√(1+ρ)/B̄        on every gradient leaf
   σ_h = nm·√(C_h·S)/B̄ = nm·C_h·√(1+1/ρ)/B̄      on the probe leaf (each of its L·E entries)
   Mahalanobis:  (C_g/B̄)²/σ_g² + (C_h/B̄)²/σ_h² = 1/nm²   with equality  (printed 1.000000 at six ρ, three agents)
   ```
6. **Release** = the noised pytree `(ĝ_t, ŷ_t)` — the only DP output of the step. Under DDP the probe leaf is all-reduced pre-noise
   with every other leaf (`sum_gradients_`, `_dp_trainer.py:2170-2173`) and the noise key is shared across ranks
   (`_dp_trainer.py:1478`, VERIFIED), so `ŷ_t` is bit-identical on all ranks.
7. **Post-processing** (§2.4) → `d̃_{t+1}, s_{t+1}, f̃_{t+1}, D_{t+1}`; the probe entry of the noised pytree is zeroed in place after
   reading so the optimizer's update for `z` is exactly 0 (VERIFIED (refute-composition) U7 for AdamW-BC / AdamW / SGD-momentum).
8. **Accountant: unchanged** — `poisson(gaussian(nm), q) * T` exactly as `_build_mechanism` builds it today
   (`_dp_trainer.py:4308-4392`, VERIFIED (synthesizer); `num_groups` is consulted only by the adaclip wrapper, excluded here).

**Why step 8 is correct (the chain).** The step releases one Gaussian on the concatenation `[Σ_x clip g_x ; λ Σ_x w_x d^{(L,E)}(x)]`
with diagonal covariance and independent per-leaf draws (`_gaussian.py:311-313`, VERIFIED (refute-sensitivity)). Whitening by
`Σ^{-1/2}` is a bijection, so the mechanism and its whitened form have identical privacy; the whitened statistic has add/remove
L2 sensitivity `√(C_g²/σ_g² + C_h²/σ_h²) = 1/nm` (attained by an example saturating both bounds — A1 shows the load half is
attainable), i.e. it is a sensitivity-1 Gaussian at multiplier `nm` — Dong–Roth–Su Thm 2.7 (https://arxiv.org/abs/1905.02383),
Zhu–Dong–Wang Def. 7 (https://arxiv.org/abs/2106.08567) for its dominating pair (both VERIFIED phase-1 critic R6). Precedent:
Andrew et al. 2021 Thm 1 (https://arxiv.org/abs/1905.03871, VERIFIED phase-1 literature B1). Poisson subsampling then applies to
the **one** joint mechanism, because both releases share the sampling coin: Feldman–Shenfeld Lemma 3.2 / Thm 3.3
(https://arxiv.org/abs/2602.17284) as implemented in `src/amplification/poisson.rs:15-42` (VERIFIED phase-1 primitives). **That
lemma needs the added record's coin to be independent of every other record's coin** — the property §8.1 shows is violated after
a DDP checkpoint resume today. `f̃_t = φ(y_{<t})` is a function of previous outputs — the same dependence `ℓ_x` already has on
`θ_t` — and Zhu–Dong–Wang Thm 10 (adaptive composition of dominating pairs) charges nothing for it. The naive "σ·C_g on the
gradient, σ·C_h on the load" allocation is `gaussian(nm/√2)`, not `gaussian(nm)` (phase-1 math 4(b)); the design never uses it.

Empty Poisson batch: `clipped_grad` short-circuits with zero grads carrying the `PerGroup` max_norm (`_clipped_grad.py:220-277`,
VERIFIED phase-1 primitives / refute-composition U3); the probe leaf is `0 + noise`, a valid release of the empty sum;
post-processing consumes it unchanged; the MF column index stays contiguous.

### 2.3 Per-step mechanism — DP-FTRL / band-MF / b-min-sep (the actual `mellum2-kstack` preset)

Steps 1–4 as above with `B_t` from `BMinSepSampler` (`dpftrl/sampling/_b_min_sep.py:6-11, 119-129`: per-iteration
`p = p₀/(1−p₀(b−1))` and the paper's warm start, so `E|B_t| = B̄` from step 0, VERIFIED (refute-sensitivity, refute-composition)) —
the load statistic is computed on the **same** batch (no second draw). Then:

5'. **Noise** (`mf_gaussian_noise`, `_mf_gaussian_noise.py:163-192`, VERIFIED (synthesizer, refute-feasibility check D)): the
   constant-max_norm latch `_validate_constant_max_norm` (`_engine.py:473-517`) accepts the constant two-group `PerGroup`;
   `base_stddev = per_group_noise_stddev(max_norm, nm)`; the correlated noise `C⁻¹Z` is applied leaf-wise, probe leaf included;
   realised per-step σ on every leaf `= base·‖row_t(C⁻¹)‖`. Under `band_mf_strategy(bands=64, momentum=0.95)` the single-step
   factor is **1.431**, stationary from `t ≈ 7` (VERIFIED (synthesizer, refute-sensitivity) `mf_rownorm.py` / `filters.py`).
6'. **Release** = the noised stream row `(ĝ_t, ŷ_t)`.
7'. **Post-processing** as §2.4, with the filter factors computed from the instantiated strategy (§6).
8'. **Accountant: unchanged** — `b_min_sep(mf_gaussian(nm, band_mf_strategy(bands=64, momentum=0.95, lr_schedule)), n_steps=T, p0=q)`
   as `build_amplifier_factory` constructs it (`trainer/_dpftrl.py:150-158`; `_mf_gaussian.py:118-128`, VERIFIED (synthesizer)).

Correctness: Denisov et al. 2022 Thm 2.1 (https://arxiv.org/abs/2202.08312, VERIFIED critic R6) applied to the per-group-whitened
stream. The participation-pattern sensitivity is homogeneous of degree 1 in the row bound, so for a shared participation pattern
π `Σ_g ‖C(G_g−H_g)‖²_F/σ_g² ≤ s(π)²/nm²`, and the sup over π is `sens(C)²/nm²` — the same scalar PLD (phase-1 math §5(i)). The
pattern is shared across groups (same example, same step), so the gradient's `min_sep` / `max_participations` apply to the probe
group. The probe's rows being near-identical across an example's participations is the worst case the MF sensitivity already
assumes (refute-composition U4). The trainer calibrates `nm` from the target ε, so ε is *held* and the entire price shows up as
`×√(1+ρ)` on the gradient noise.

Excluded under MF: adaptive clipping (drifting bound; rejected by the latch anyway); an independently sampled load release (a
different participation model — would need concurrent composition, theorem numbers **not verified**, PLAUSIBLE only; §2.7 keeps
it DP-SGD-only). AUTO-S on the gradient group is MF-admissible but v1 rejects it with the feature because `_auto_scale_per_group`
applies `R/(‖·‖+γ)` to *every* group with `clamp_to_one=False` (`_pytree.py:313-340`, VERIFIED (synthesizer)), which would
rescale every `d(x)` to norm ≈ `C_h` — a mean *unit direction*, a different statistic (privacy-valid, biased).

### 2.4 Post-processing (all public, free) — producing `f̃_{t+1}` and the monitor

Given the noised probe leaf `ŷ_t = noisy_grads.pytree["router_load_probe"] ∈ R^{L×E}` (already `/B̄`, rank-identical):

```
1. d̂^{(L,E)}_t = ŷ_t / λ                                  # (1/B̄) Σ_{x∈B_t} w_x d^{(L,E)}(x) + N(0, (σ_h/λ)² I)  per entry
2. d̂_t        = mean_l d̂^{(L,E)}_t                          # pooled: noise per entry (σ_h/λ)/√L = the pooled-release value (VERIFIED)
3. d̂_t       ← d̂_t − mean_e(d̂_t)                            # sum-zero projection (signal already sums to 0; noise variance ×(E−1)/E)
4. m_{t+1}    = β·m_t + (1−β)·d̂_t,   β = 0.99;   d̃_{t+1} = m_{t+1} / (1 − β^{t+1})       # bias-corrected EMA (window mean W selectable, §6)
   s_{t+1}    = (σ_h/(λ√L)) · φ_{t+1} / (1 − β^{t+1})       # known noise std of d̃: φ from the exact variance recursion (DP-SGD:
                                                          #   φ² ← β²φ² + (1−β)²·(E−1)/E, stationary √((1−β)/(1+β))·√(63/64)) or
                                                          #   from ‖row_t(F·C⁻¹)‖ of the instantiated MF strategy (stationary 0.0249)
5. d̃⁺        = 0                                  if ‖d̃_{t+1}‖² < c·E·s²_{t+1}     (dead zone, c = 2)
              = d̃_{t+1} · (1 − E·s²_{t+1}/‖d̃_{t+1}‖²)  otherwise                     (positive-part James–Stein toward balance)
6. f̃_{t+1}   = clamp(k/E + d̃⁺, 0, 1)                      # clamp inactive unless |d̃⁺_e| > k/E (≥ 40σ of the smoothed noise)
7. D_{t+1}    = max_e |d̃_{t+1,e}| / (k/E);  D^l_{t+1} likewise from the bias-corrected per-layer EMA   # public monitors
```

- **Why the per-layer carrier and pooling (§12 S7).** With `λ_L = ρC_g/√(7L)` the per-entry noise of `d̂^{(L,E)}` is `√L` larger
  than a pooled release at the same ρ, and averaging over L layers divides it by exactly `√L`: pooled `d̂_t` has **identical**
  noise to a direct pooled release (0.04149 vs 0.04149, VERIFIED `checks.py` and refute-sensitivity `filters.py`), at the same
  `C_h`, the same σ_g inflation, the same accountant. The per-layer entries are a free diagnostic at ×√L = ×5.29 per-entry noise.
- Why the sum-zero projection: `Σ_e d^l(x) = 0` for every example and layer, so the signal lives in the sum-zero subspace; the
  noise's mean component is pure noise. This replaces the noisy-sum renormalisation of the minimal/skeptic designs (denominator
  noise 4.4 % of k per step at ρ = 0.02) and the `k·|B_t|/B̄` realised-batch scale issue: the realised `|B_t|/B̄` and
  `T_tot/(B̄T̄)` multiply the *signal* only.
- **Why bias correction (§12 S6).** The uncorrected EMA from `m_0 = 0` has mean `(1−β^t)·d_true`: 0.634 / 0.866 / 0.951 / 0.990
  at t = 100 / 200 / 300 / 460 (VERIFIED `checks.py`), i.e. a surrogate coefficient 37 % / 13 % too small for the first 200–300
  steps. Dividing both the estimate and its noise std by `(1−β^t)` removes the bias at every t (the corrected std at t = 100 is
  0.1032 vs the stationary 0.0710 — the early estimate is honestly noisier, and the dead zone accounts for it). Under MF the EMA's
  *signal* gain is still `(1−β^t)` (a deterministic linear filter), so the same correction applies to the filtered stream; the
  window mean is unbiased by construction (divide by `min(t, W)`).
- **Why a dead zone, then shrinkage (§12 S5).** The surrogate gradient is `α·E·w_x·Σ_e (f̃_e − k/E)·∇P_e(x)`; without a dead
  zone a noise-dominated `d̃` injects a *random* regularisation direction. Plain JS+ (v1) zeroes only when `‖d̃‖² < (E−1)s²`, which
  pure noise (`‖d̃‖²/s² ~ χ²₆₃ · E/(E−1)`) exceeds ≈ 48 % of the time (VERIFIED (refute-sensitivity): 52.3 % zeroed at δ = 0), so
  v1's "never a random regulariser" was an overclaim. With the dead zone at `c·E·s²`: `c = 1.5` → false-pass 6.3e-3 per step
  (≈ 98 over the horizon); **`c = 2` (default) → 4.2e-6 per step, 0.07 expected false passes over 15 625 steps** (VERIFIED
  `checks.py`, exact χ²₆₃ tails). Above the dead zone the JS+ factor `1 − E·s²/‖d̃‖²` (dof corrected: after the projection the
  noise's total variance is `E·s²`, not `(E−1)s²` — a 1.6 % correction, §12 S5) is the identity when `‖d̃‖ ≫ s√E` and
  continuous at the boundary (factor ≥ 1/2 at `c = 2`). **Threshold in signal units**: the dead zone engages when the
  per-coordinate RMS imbalance `δ·k/E` satisfies `δ < √c · s/(k/E)`: at ρ = 0.02 with EMA .99, **δ < 0.033 (DP-SGD) / 0.012
  (band-MF at nm = 0.5622) / ≤ 0.032 (band-MF at the `nm_MF` bracket 1.544, §2.6)**. A well-balanced checkpoint (δ ≈ 0.1) is
  3–8× above it. The precise statement replacing v1's: *below the dead zone the surrogate is identically the real objective's
  behaviour at balance (zero); a pure-noise step passes the dead zone with probability 4.2e-6.* `router_load_shrink=False`
  gives the plain estimate.
- Why clamp: a negative `f̃_e` is physically meaningless; the clamp is inactive unless an expert is dead/hot (`|d̃_e| > 0.125`,
  ≥ 40σ of the ρ = 0.02 smoothed noise).
- **Monitor and decision rule**: trip when `D_t > τ = 0.5` (some expert carries ≥ 1.5× or ≤ 0.5× its share) on two consecutive
  logged evaluations. At ρ = 0.02 the smoothed per-entry noise is 2.35 % of `k/E` (DP-SGD) / 0.83–2.28 % (band-MF, nm 0.5622 …
  1.544), so τ = 0.5 is **21σ / 22–60σ** (VERIFIED arithmetic). Responses on trip: (i) log and continue (`monitor`), (ii) switch α
  from 0 to the configured value (`monitor_then_surrogate`) — without touching clipping, noise, the accountant or the MF latch
  (an adaptive-row change, Denisov Thm 2.1 / ZDW Thm 10), (iii) stop. `D_t`, `D^l_t`, `f̃` min/max/entropy, the shrink factor,
  the dead-zone flag and the trip flag are logged as public curves. Per-layer `D^l_t` at ×5.29 noise (12.5 % of k/E under
  DP-SGD at ρ = 0.02) is still 4σ at τ = 0.5 — usable for locating a collapsing layer, not for fine decisions.
- Lagged use is free: adaptive composition (ZDW Thm 10); the ordering in `training_step` (augment → grad → clip → reduce → noise →
  callback → optimizer) is VERIFIED (refute-composition U2).

### 2.5 Modes and defaults

`router_load_release: Literal["off", "monitor", "surrogate", "monitor_then_surrogate"] = "off"`; `router_load_ratio = 0.02` (ρ);
`router_aux_loss_coef = None` (→ config in surrogate modes; 0 in `monitor`); `router_load_mean_tokens: int | None = None` (`T̄`;
`None` → `T_max` = the collator's row length); `router_load_filter = {"kind": "ema", "beta": 0.99}` (or `{"kind": "window",
"steps": 256}`); `router_load_shrink = True`; `router_load_dead_zone = 2.0` (c); `router_load_trip = 0.5`; `router_aux:
Literal["pooled", "per_layer", "per_sequence"] = "pooled"`; `router_z_loss_coef = 0.0`; `router_fp32 = False`. Presets:
`mellum2-kstack` → `surrogate`, α = 1e-4, ρ = 0.02, `T̄` = measured mean length of the public held-out split (§4.4);
`mellum2-codesec` (DPO) → `monitor`, ρ = 0.02 (§7). Constraints enforced at setup with `ConfigurationError`: any non-`off` mode
requires `clipping_mode == "fixed"`, `second_moment == False`, a family whose backbone records `router_logits`, and the chunked
causal-LM forward installed (§9.1).

### 2.6 THE cost table — preset regime `B̄ = 256, k = 8, E = 64, C_g = 0.9, L = 28, q = 256/5e5, T = 15625, δ = 1e-6`, `T̄ = T_max`

All ρ rows reproduced with the real `per_group_noise_stddev` and the real accountant by two agents (`final_table.py`,
`costtable.py`; baseline `poisson(gaussian(0.5622), q)*T` → ε = 3.0004, VERIFIED). Single-release per-entry noise on the pooled
`d̂` in units of `k/E = 0.125`: `r₁ = nm·Δ_h·√(1+1/ρ)/(B̄·k/E)`; multiply by `T_max/T̄` if `T̄ < T_max`. Filter factors: DP-SGD EMA
.99 **0.0709**, window-256 0.0625; band-MF(64, 0.95) single step **1.431**, EMA .99 **0.0249**, window-256 0.0198 (VERIFIED by
two agents). **Band-MF columns are given at two multipliers**: nm = 0.5622 (the DP-SGD/Poisson ε = 3 calibration, a lower
bound for `nm_MF`) and **nm = 1.544, the deterministic un-amplified band-MF bound at ε = 3** (`mf_gaussian(nm, band_mf(64, 0.95),
n_steps=15625, min_sep=64)`, `strategy.sensitivity = 1.0000`, VERIFIED `nm_bracket.py` re-run, 24 s). Amplification cannot
require a larger multiplier than the un-amplified bound (PLAUSIBLE only insofar as the MC accountant's upper bound could in
principle be looser than the deterministic one), so `0.5622 ≤ nm_MF ≤ 1.544` and `nm_MF/0.5622 ≤ 2.746`; v1's "PLAUSIBLY several ×
larger" is replaced by this bracket (§12 F4).

| ρ = C_h/C_g | c = √(1+1/ρ) | **gradient-noise inflation √(1+ρ)** (ε held, accountant unchanged) | ε if nm were held instead ("pay in ε") | r₁ single release | DP-SGD after EMA .99 / window 256 | band-MF after EMA .99 at nm 0.5622 / **1.544** | per-layer *entries* (×5.29) after EMA .99, SGD / MF@1.544 | dead zone (c = 2) engages below δ, SGD / MF@0.5622 / MF@1.544 |
|---|---|---|---|---|---|---|---|---|
| 0.50 | 1.732 | ×1.225 | 5.443 | 8.1 % | 0.57 % / 0.50 % | 0.20 % / **0.55 %** | 3.0 % / 2.9 % | 0.008 / 0.003 / 0.008 |
| 0.20 | 2.449 | ×1.095 | 4.219 | 11.4 % | 0.81 % / 0.71 % | 0.28 % / **0.77 %** | 4.3 % / 4.1 % | 0.011 / 0.004 / 0.011 |
| **0.10** (router/experts trainable) | 3.317 | **×1.049** | 3.703 | 15.4 % | 1.09 % / 0.96 % | 0.38 % / **1.04 %** | 5.8 % / 5.5 % | 0.015 / 0.005 / 0.015 |
| 0.05 | 4.583 | ×1.025 | 3.417 | 21.3 % | 1.51 % / 1.33 % | 0.53 % / **1.46 %** | 8.0 % / 7.7 % | 0.021 / 0.007 / 0.020 |
| **0.02 (default)** | 7.141 | **×1.010** | 3.234 | 33.2 % | **2.35 %** / 2.07 % | 0.83 % / **2.28 %** | 12.5 % / 12.1 % | **0.033 / 0.012 / 0.032** |
| 0.01 | 10.05 | ×1.005 | 3.172 | 46.7 % | 3.31 % / 2.92 % | 1.16 % / **3.19 %** | 17.5 % / 16.9 % | 0.047 / 0.016 / 0.045 |
| *v2 opt-in, DP-SGD only: independent forward-only draw, c = 2, every m = 4 steps* (§2.7) | — | ×1.0002 (ε-matched) | **3.003** | 9.3 % | 2.1 % (EMA .9 over releases, lag 40) | n/a | 11 % | — |
| *independent draw, c = 1, m = 4* | — | ×1.013 | 3.128 | 4.6 % | 1.1 % | n/a | 5.6 % | — |
| *independent draw, c = 2, m = 16* | — | ×1.0001 | 3.001 | 9.3 % | 2.1 % (lag 160) | n/a | 11 % | — |

Reading: at the worst admissible `nm_MF` the band-MF smoothed error (2.28 %) equals the DP-SGD one (2.35 %) — the 2.8× filter
advantage of anti-correlated noise is what pays for the un-amplified multiplier. The two rightmost DP-SGD columns are the same
mechanism family re-parametrised (critic C7): "pay in gradient σ" (shipped) or "pay in ε". **The per-layer column now prices the
accuracy of per-layer entries `f̃^l` (diagnostics, or the opt-in per-layer surrogate), not the release**: the pooled surrogate
pays nothing for the `(L, E)` carrier (§2.4).

**Reading the table against the objective.** The error that matters is `‖noise‖/‖d_true‖`. With per-coordinate RMS imbalance
`δ·k/E`, the relative aux-gradient error at the default row is ≈ `0.0083/δ … 0.0228/δ` (band-MF, nm 0.5622 … 1.544) or
`0.0235/δ` (DP-SGD): 8–24 % at δ = 0.1 (a well-balanced checkpoint), 3–8 % at δ = 0.3 (VERIFIED (refute-sensitivity): 0.231 /
0.078 at δ = 0.1 / 0.3 reproduce). Below the dead zone (δ ≈ 0.012–0.033) the term is exactly zero — where the true objective's
gradient is negligible too. Signal-to-noise on real data (G3, §10) decides whether the term carries signal.

Rejected levers, for the record (VERIFIED arithmetic where numbers are given): randomised response on `sign(d)` — per-step
ε = `64·ln((1−p)/p)` = 12.8 at p = .45, dominated; independent draw with a 4× batch (`q₂ = 4q, c = 1`) → ε 5.30; amortised
same-batch release every m steps — same budget in bursts; per-sequence aux — different regulariser; public-data `f̃` (Davody et
al. 2020, https://arxiv.org/abs/2006.10919; Ponomareva et al. "DP-fy" §5(b)) — zero cost but estimates the public
distribution's imbalance; opt-in when a public corpus exists.

### 2.7 Appendix (v2, opt-in, DP-SGD only): the independent forward-only Poisson release

Every `m` steps draw a **second, independent** Poisson sample `B'_t ~ Poisson(q₂)` over the dataset, run a forward-only vmap of
the routing statistic through `clipped_fun` (`_clipped_fun.py:492`, `clipping_norm = Δ_L`, `normalize_by = q₂N`), add
`N(0, (c·nm·Δ_L/(q₂N))²)` and post-process as §2.4 with the EMA running over releases. Accountant:

```
dpsgd.poisson(dpsgd.gaussian(nm), q) * T  |  dpsgd.poisson(dpsgd.gaussian(c·nm), q₂) * (T/m)
```

Validity: two mechanisms with **fresh** sampling coins and fresh noise, adaptively interleaved — ZDW Thm 10 with each factor
dominated via Feldman–Shenfeld Lemma 3.2; `Poisson` accepts any `DpProcess` inner (`dpsgd/amplification/_poisson.py:25-45`);
`|` composes add/remove direction-wise (`pld/mod.rs:461-490`, VERIFIED (refute-sensitivity)). Numbers VERIFIED by two agents:
c = 2, m = 4 → **ε 3.0030**; c = 1, m = 4 → 3.1284; c = 2, m = 16 → 3.0011. **Conditions before "provably DP as run" holds for
this variant** (judge-dp E7; refute-composition R7 accepted, §12 C7): (i) the second sampler's stream key is
**rank-domain-separated** — `fold_in(key, "opaque.moe.load_release", rank)` — so per-example inclusion coins are independent
across ranks *and across the primary sampler*; (ii) the second **noise** key is `fold_in(key, "opaque.moe.load_release.noise",
step)`, **shared across ranks (no rank fold)**, and the noise is added *after* `reduce_pytree_` — otherwise `f̃` diverges across
ranks; T24 asserts rank-identical `f̃`; (iii) the sampler's key/position and the noise-key counter are serialised in the sidecar
and **re-folded by rank on resume** (the primary sampler's resume bug, §8.1, must not be replicated); (iv) each rank samples its
shard at `q₂` with `N` the global size; (v) `ConfigurationError` when `mechanism_kind != "gaussian"`. Not part of v1.

---

## 3. Router precision and routing (G5)

**Decision: routes are computed per example inside vmap from the current model (no pinning); `f` is defined from those executed
routes; the router runs in the stock HF precision by default; an fp32-logit router patch is shipped as an opt-in documented as
pretraining-faithful and tie-robust — NOT as a drift or flip fix.** Unchanged from v1; refutations touched only executability
(§12 F7).

- **The refutation that settles it** (skeptic `a_fp32_router_flips.py`, re-run by judge-utility; VERIFIED at toy scale): on a
  bf16 tiny Mellum vs the fp32 model, the stock bf16 router flips 46/54 (random) and 46/47 (structured) top-8 sets per 1024
  rows per layer; the fp32-logit router flips 51/56 and 39/43. The flips originate in the **bf16 hidden states** entering the
  router (fixed *relative* resolution ⇒ ≈ 1 % of tokens per layer sit inside a rounding tie regardless of router sharpness —
  phase-1 E2), not in the logits' rounding. And the pair that matters for requirement (d) — Opaque bf16 `vmap(grad)` vs HF
  eager at equal precision — has **0 flips** and **0.0 rel-L2** on the patched model (VERIFIED (refute-feasibility) check B;
  phase-1 F8).
- **What the fp32 router buys** (opt-in `router_fp32=True`): the router Mellum2 was pretrained with (TR appendix, VERIFIED
  phase-1 literature F.2); exact bf16 ties disappear; HF's bf16-softmax aux top-k and the executed top-k coincide. Cost 0.30 %
  of routed expert MACs (VERIFIED arithmetic). It changes the *executed routing function* on ≈ 1 %/layer of tokens, not the
  weights; adapters served through stock HF run bf16 routes, so default **off**.
- **Installation (§12 F7).** A class-level `_patch_forward` replacement is guarded by `__opaque_patched__` (`_router.py:59-92`,
  VERIFIED (refute-feasibility)) and cannot be toggled off in-process. The fp32 router is therefore installed as a **removable
  instance-level swap** (`types.MethodType` on the router instances, recorded so `apply_model_patches(..., router_fp32=False)`
  can undo it), and §10's bf16-vs-fp32-router comparisons are specified to run **in separate processes** regardless.
- **Pinning**: routes are per-example constants of `(x, θ_t)`; no cross-example dependence, no DP consequence. Frozen-base
  pinning is rejected (freezes training-time routing while inference routing drifts). Pinning from an fp32 routing-only forward
  remains a validated-later option for router/expert training; not built.
- **Oracle consequence (G2)**: flips are a property of precision, not of vmap, so the oracle must be **precision-matched**.

---

## 4. Clipping norm and per-example gradient norms (the procedure)

1. **The load release never competes with the gradient for the clipping budget.** Own `PerGroup` group with a structural bound;
   the gradient group keeps today's `C_g`. The joint single-vector clip is rejected (shrinks the admissible gradient norm to
   `C√(1−ρ²)` and biases the histogram). At ρ = 0.02 the whole price is ×1.010 gradient noise.
2. **The aux term does not move the per-example norm.** At α ∈ [1e-4, 1e-3] the per-example norm changes by < 0.1 % (VERIFIED
   phase-1 empirical E3), and the surrogate gradient is identically 0 at balance and inside the dead zone. `C_g` is governed by CE.
3. **Scale of the problem** (VERIFIED arithmetic): LoRA r = 16 on q/k/v/o is 294 912 parameters per layer, `d = 8 257 536`,
   `√d = 2874`; the per-step noise vector has norm `nm·C_g·√d/B̄ = 5.68 = 6.3·C_g` at C_g = 0.9. `C_g` acts mostly as a
   learning-rate scale once most examples are clipped.
4. **The calibration pass is itself a private query (§12 S8).** The `clipped_grad(..., clipping_norm=1e9, return_aux=True)` pass
   that reads `aux.grad_norms` to pick `C_g` (and the `T̄`, `δ`, `‖d(x)‖` statistics of §10.2) must run on a **calibration split
   disjoint from the protected training set** — for the presets, a held-out KStack shard that is never fed to the DP run
   (KStack being publicly downloadable does not make the *protected rows* public: "hyperparameter searches … included in the
   privacy statement", review protocol) — or be **accounted** as one Gaussian release of the norm quantile
   (`adaptive_clipped_grad` + `dpsgd_acc.adaclip`, `opaque-dpsgd/.../clipping/_adaptive.py:164-215`, VERIFIED phase-1
   primitives §1.5). Under MF adaptive clipping is barred by the latch, so use the disjoint split. §8 and §10.2 say so.
5. **How to pick `C_g`** (the G3 script, §10.2): one pass over ≈ 256 held-out examples under the preset partition; report
   p10/p50/p90/p99/max of `aux.grad_norms` (expect p90/p50 ≈ 1.5–2 on real code, PLAUSIBLE). Choose `C_g` by the bias²+noise²
   curve; sanity check clip rate 40–60 % in the first 100 DP steps (the trainer's adaptive target, `_dp_trainer.py:4251`).
   The preset pins the measured value (0.9 today).
6. **Clipping mode: fixed** (both presets). With the feature on, `clipping_mode ∈ {"auto", "adaptive"}` raises
   `ConfigurationError` in v1; v2 adds `fixed_groups=("router_load_probe",)` to `auto_clipped_grad` (§9.4).
7. **Partition-aware groups when experts/router are trained** (§7): patterns must be **non-overlapping substrings** of the
   dotted paths — `mlp.router`, `experts.`, `self_attn` (v1's `gate=…, experts=…` example collided: `experts.gate_up_proj`
   matches both, §12 C4) — and the probe group is added by direct construction, never by pattern. Bounds at each group's
   per-example median from the held-out calibration pass; keep the number of groups ≤ 4; recommended ρ = 0.1 there.
8. **Numerical stability vs the non-DP path** (requirement (d)): fp32 accumulation in `Opaque_MoE`, upstream RMSNorm retained,
   SDPA causal fast-path parity (PR #980) — with the kernel choice made a public property (§8.2). Remaining bf16 drift is
   accumulation order, 0.45–0.5 % rel-L2 with zero route flips (VERIFIED phase-1 E1/E1b toy), inside HF's own batched-vs-loop
   spread.

---

## 5. Performance (G6)

### 5.1 Dense vs grouped MoE default

Facts (VERIFIED): `DPTrainer` passes `kernels=bool(args.use_performance_kernels)` and `**performance_kernels_config`
(`_dp_trainer.py:841-847`), default `False`; the factory sets `grouped_moe = kwargs.get("grouped_moe", kernels)`
(`_factory.py:316-322`, VERIFIED (synthesizer)); `grouped=False` selects the dense every-token-through-every-expert `Opaque_MoE`
on every host including CUDA (`kernels/moe.py:578-635`); the flag is captured by the **first** class-level patch per process
(`_router.py:59-92`). FLOP arithmetic (VERIFIED arithmetic; timing PLAUSIBLE): dense = 8× the routed expert FLOPs, ≈ 5× total
forward FLOPs (24.4 vs 4.9 GFLOP/token); per 256-example step at T = 1024 roughly 12–17 PFLOP dense vs 1.5–2 PFLOP grouped —
11–16 days vs ≈ 2 days for the 15 625-step preset on one H100. The dense default is the single largest practical blocker.

**Decision (all designs and judges agree): decouple `grouped_moe` from `use_performance_kernels`** — `grouped_moe =
kwargs.get("grouped_moe", kernels or _grouped_route_available())`, true on CUDA + Triton or wherever `torch._grouped_mm` exists
with ≥ 16 experts (`_SPARSE_MOE_MIN_EXPERTS`, `moe.py:575`); `opaque_moe` already gates the grouped route on a workspace estimate
(`moe.py:614-632`). Two facts added by the refutations: (i) `_grouped_route_available()` is true on the **CPU** build too
(`torch._grouped_mm` exists, VERIFIED (refute-feasibility) U11), so the entire CPU MoE test surface flips to the grouped route
for E ≥ 16 — the parity tests already cover `vmap(grad)` with frozen experts (`tests/kernels/test_grouped_moe.py:59-72,
113-116, 145-148`); risk 6 names CPU CI explicitly; (ii) on CUDA the fused Triton route takes only bf16/fp16 — **CUDA fp32 falls
to the dense `Opaque_MoE`** (`kernels/moe.py:614-632` is inside `if _TRITON_AVAILABLE and x.is_cuda`, VERIFIED (refute-feasibility)
R8), which matters for the fp32 floor O1 in §10 (budgeted dense there). Log the chosen path once; document first-patch capture.

### 5.2 Cost and memory of the mechanism itself (per microbatch of 8, T = 1024, L = 28, E = 64, k = 8) — numbers corrected (§12 F5)

| buffer | shape / dtype | size per microbatch of 8 | note |
|---|---|---|---|
| router logits tuple (recorder) | 28 × (T, E) bf16 references | **29.4 MB** (3.67 MB/example) | v1's "0 extra — already live in the graph" was wrong: `softmax` saves its *output* and `F.linear` its *inputs*; the logits tensor is kept alive only by the recorder's references (VERIFIED (refute-feasibility) R5; retention through the checkpoint region VERIFIED check E) |
| fp32 softmax saved for the surrogate backward | 28 × (T, E) fp32 | **58.7 MB** (7.34 MB/example) | the `(L, T, E)` fp32 stash the brief asked about |
| fp32 router logits (opt-in fp32 router) | 28 × (T, E) fp32 instead of bf16 | 58.7 MB | replaces the bf16 row above |
| one-hot counts | (T, k, E) bool per layer, reduced immediately | 0.5 MB transient | broadcast compare `idx[..., None] == arange(E)` (vmap-safe; `F.one_hot` is not) |
| probe leaf | (L, E) fp32 | 7 KB | + its per-example gradient (B, L, E) inside `clipped_grad` (57 KB); inert optimizer state |
| filter state | (L, E) EMA + (E,) pooled + scalars (or a (W, E) ring) | 7 KB (64 KB) | public |
| `f̃` closure tensor | (E,) | 256 B | updated in `_augment_inputs` |

Total < 0.15 GB per microbatch against ≈ 24 GB of bf16 weights — not a constraint. Compute: one `(L·T)×E` softmax per example
(reused), one masked mean, one top-k indicator reduction — `O(L·T·E) = 1.8 M` elements per example. Chunked CE is preserved
**when installed** (§9.1), so peak memory stays that of PR #978 rather than the full-vocab `98304×1024×4 B = 403 MB`-per-example
logits (3.2 GB per microbatch of 8) of the unpatched HF forward (VERIFIED (refute-feasibility) R3).

---

## 6. Smoothing under MF (G7)

### 6.1 The preset strategy

The `mellum2-kstack` preset runs `band_mf_strategy(bands=64, momentum=0.95, lr_schedule=…)` (`examples/train_dpftrl.py:447-460,
495-498, 1563-1581`, VERIFIED (synthesizer)); the *library* default is `momentum=1.0` (`_band_mf.py:111, 145`) — filter tables
computed for the library default do not describe the preset. Under momentum 0.95 the row norm is stationary from t ≈ 7 and
n-independent (VERIFIED by two agents at n = 1024):

| filter | band-MF(64, **0.95**) factor | band-MF(64, 1.0) at n = 1024 (not the preset) | DP-SGD (iid) | MF(0.95)/SGD | lag (steps) |
|---|---|---|---|---|---|
| single step `‖row_t(C⁻¹)‖` | **1.431** | 2.260 (3.80 at n = 15625) | 1.000 | 1.43× worse | 0 |
| EMA β = 0.95 | 0.0824 | 0.1157 | 0.1601 | 0.51 | 20 |
| **EMA β = 0.99 (default, bias-corrected)** | **0.0249** | 0.0253 | 0.0709 | 0.35 | 100 |
| window mean W = 256 | 0.0198 | 0.0157 | 0.0625 | 0.32 | 128 |

### 6.2 Which filter, and why

- The noise variance of *any* fixed linear filter `F` of the released stream `x̂ = d + σ_h·(C⁻¹Z)` is the deterministic quantity
  `σ_h·‖row_t(F·C⁻¹)‖` — exact, no theorem needed — and every low-pass filter benefits 2–3.5× from band-MF's anti-correlated
  noise. The filter is chosen by **lag tolerance**: `β_f = 0.99` (lag 100 steps = 0.64 % of the horizon) gives 0.0249, i.e.
  **0.83 %·(nm_MF/0.5622) ≤ 2.28 % of k/E at ρ = 0.02**. Fixing `β_f := 0.95` "to inherit the strategy's guarantee" costs 3.3×
  accuracy for no privacy gain.
- **Bias correction under MF (§12 S6):** the EMA's signal gain `(1−β^t)` is a property of the filter, not of the noise; the
  corrected estimate is `m_t/(1−β^t)` and its noise std `(σ_h/(λ√L))·φ_t/(1−β^t)` with `φ_t = ‖row_t(F·C⁻¹)‖` from the
  instantiated strategy. The window mean divides by `min(t, W)`.
- **Compute the factors from the instantiated strategy at setup — never hand constants.** At setup, when an MF mechanism is
  present, take its strategy (which carries the trainer's `lr_schedule`) and its `coefficients(n_steps)`; obtain the
  lower-triangular Toeplitz *inverse* coefficients for lags `0..N_φ` (`N_φ = 2048`) by the triangular recursion (the same object
  `inverse_as_streaming_matrix` builds, `_band_mf.py:135-136` → `_toeplitz.py:177`, VERIFIED (synthesizer)), form
  `φ_t = ‖row_t(F·C⁻¹)‖` for `t < N_φ` (4 M flops), assert stabilisation, store `φ_t` and `φ_∞` in `RouterLoadState`. Never a
  dense `n×n` solve. Under DP-SGD `φ_t` is the closed-form recursion. In the manual loop the strategy object is the one the loop
  built (`_make_strategy`, `train_dpftrl.py:1563-1581`) — passed to the helper explicitly (§9.2).

### 6.3 Per-step realised σ and diagnostics

The trainer reads the realised per-step σ from `NoisedPytree.noise_stddev` (`PerGroup` × `row_l2`, `_mf_gaussian_noise.py:186-192`);
the EMA consumer uses its own *filtered* factor `φ_t` for `s_t` and for the logged `router_load/noise_std`, never the per-step one.

### 6.4 `nm_MF`: bracket, and why "free to record" was withdrawn (§12 F4)

No phase-2 agent completed one `b_min_sep(mf_gaussian(·, band_mf(64, 0.95)))` evaluation on 4 CPU cores — not at n = 15625
(synthesizer 570 s, refute-feasibility 280 s) and **not even at n = 256** (150 s). Cause (VERIFIED (refute-feasibility)):
`epsilon_at` clamps `mc_resolution` to `min(configured, δ/2) = 5e-7` (`opaque-accounting/.../core/_base.py:216-221`) regardless
of the example's `--mc-resolution` default 1e-5 (`train_dpftrl.py:704-707`, VERIFIED), which the accountant reports as
64 997 003 transcripts per adjacency direction; the transcript corpus is reused across σ (`registry.rs:38-60, 75-110`) but
nothing bounds the wall-clock. Consequences: (i) the deterministic bracket `nm_MF ∈ [0.5622, 1.544]` of §2.6 is what the design
relies on; (ii) the validation plan budgets the preset calibration explicitly — run once offline on a many-core host, record
its wall-clock, and pass `--noise-multiplier` to every validation run — or uses **`nm = 1.544` as a conservative fixed
multiplier**, which is a *valid* privacy guarantee on its own (the un-amplified `mf_gaussian` accountant is a legitimate, looser
bound); (iii) a 200-step run "calibrated at ε = 3" under band-MF at `n_steps = 200` has a different `nm` than the preset's and
must say so; (iv) recommended follow-up (outside this design): make the CLI `--mc-resolution` honour or warn about the δ/2 clamp.

---

## 7. Scope (G8)

| item | v1 | what it needs / why |
|---|---|---|
| Causal-LM SFT via `DPTrainer` default path and `DPSFTTrainer` (`nll` / `chunked_nll`) | **in** (`surrogate`, α = config, ρ = 0.02) | §9; the chunked forward must be installed (§9.1) |
| **`mellum2-kstack` preset — a manual functional loop, not a `DPTrainer` run** (`examples/train_dpftrl.py`: own `per_example_loss_fn` `:1369-1378`, own clip/noise/calibration; no `DPTrainer` reference, VERIFIED grep) | **in** via the trainer-independent helper (§9.2); `surrogate`, α = 1e-4, ρ = 0.02, `T̄` from the held-out split | v1's claim that the trainer seams reach this preset was false (§12 F2) |
| DP-FTRL (band-MF, BLT, BSR, BiSR, λ-CGD) with b-min-sep / Poisson / balls-in-bins | **in** | all strategies go through `mf_gaussian_noise`'s `PerGroup` path; the latch accepts the constant `PerGroup` (VERIFIED by three agents) |
| DP-DPO mechanism (`DPDPOTrainer`, `_dpo_trainer.py:1065-1195`) | **in** | protected unit = the preference pair; `P(x)`, `h(x)` pool the **chosen + rejected policy forwards** with denominator `L·(T_c + T_r)`, `w_x = (T_c+T_r)/T̄_pair` (`T̄_pair` public, default `2·T_max`); the reference forward contributes no aux and is never called with `opaque_router_logits=True` (`_dpo_trainer.py:807-841`; `_fused_logp` calls the backbone functionally, `:998-1063`, VERIFIED (refute-feasibility) U10 — a one-line addition); T21 |
| **`mellum2-codesec` preset (DP-DPO) — also a manual loop** (`examples/train_dpo.py:544-562`, VERIFIED) | **`monitor`** via the helper (ρ = 0.02, α = 0) | TRL never added the aux to DPO/KTO; `surrogate` is one flag away once §10's DPO row passes; the loop already passes attention masks |
| Router z-loss `Z_x` | **opt-in, ζ = 0**, value-neutral | per-token separable ⇒ zero privacy cost; only meaningful with a trainable router |
| Router / experts trainable | **in for the mechanism, out for presets** | nothing in §2 depends on which leaves exist; blockers: PEFT `target_parameters` on stacked experts under `functional_call`+vmap untested; per-group `C` by parameter class (§4.7, non-overlapping patterns); full expert training memory (0.53 GB/example/layer bf16); ESFT selection from private data is a query to account. ρ = 0.1 |
| Per-layer `f̃^l` | **carrier always `(L, E)`; per-layer entries logged free**; per-layer *surrogate* (`router_aux="per_layer"`, Megatron-style mean of per-layer products) opt-in, ×5.29 per-entry noise, usable under MF at ρ ≥ 0.2, documented as a different objective than HF's | §12 S7 |
| Per-sequence aux (`router_aux="per_sequence"`) | opt-in | zero privacy cost; documented as a *different* regulariser |
| Independent forward-only Poisson release | v2 opt-in, DP-SGD only | §2.7 conditions |
| Public-data `f̃` (`router_load_source="public"`) | opt-in | forward-only on a public batch; zero release; the hybrid buys nothing |
| Packing the preset to fixed 1024-token rows | **option, off** | makes both identities exact and `T̄ = T_max`, but changes the protected unit to a packed row (§2.1) — must be declared if used |
| Loss-free balancing (Wang et al. 2024 Alg. 1, https://arxiv.org/abs/2408.15664; DeepSeek-V3 eq. (16), https://arxiv.org/abs/2412.19437; VERIFIED phase-1 literature) | **out** (documented) | `b_{t+1} = b_t − u·sign(d̂_t)` is post-processing of the same release — free — but the checkpoint's router has no bias tensor, so `b` is an architecture extension needing a serving patch; training-only use creates a train/inference routing mismatch by construction |
| HF-Trainer-realised objective (per-microbatch `f`, `G·α`) | out | artefact (§1.3) |
| Token-weighted CE with public `N̄` | out | pre-existing convention F4; clip-norm rethink |
| STE / DenseMixer through top-k | out | changes the real model's backward |
| Users passing `output_router_logits=True` under DP | **rejected with a clear error** | fails under vmap with a mask (in-place `scatter_add_`, reproduced VERIFIED (refute-feasibility) check G) and bypasses chunked CE |

---

## 8. Privacy hygiene (G9)

Rule: everything computed from private examples inside the grad transform is private-internal until it has passed clip → noise;
only noised aggregates and their post-processing are public. Adjacency/unit as in §2.1.

### 8.1 Pre-existing "as run" gap: DDP + checkpoint resume (refute-composition R1, accepted — §12 C1)

The trainer's own comment records that after a DDP resume "the sampler snapshot … is written once on rank 0, so resuming a DDP
run currently restores rank 0's per-rank key on every rank, re-introducing the cross-rank correlation after the resume point"
and asserts "the privacy accounting is unaffected" (`_dp_trainer.py:3750-3757`, VERIFIED); the restore path installs the saved
sampler with no rank re-fold (`:1794-1800`, VERIFIED) while the rank fold exists only on fresh construction (`:3802-3803`,
VERIFIED). Records sharing a local shard index on different ranks then have **identical inclusion coins** for the rest of the
run (Poisson and b-min-sep alike). **The "accounting is unaffected" comment is wrong.** The tight subsampled-Gaussian PLD the
accountant runs (Feldman–Shenfeld Lemma 3.2 / ZDW Thm 11) needs the added record's coin independent of the others'; with a shared
coin the add/remove pair is `((1−q)A + qB₀, (1−q)A + qB₁)` with `A ≠ B₀`, which keeps δ-amplification but **loses
ε-amplification**. 1-D hockey-stick computation (`corr_coins.py`, re-run VERIFIED): at σ = 0.5622, q = 5.12e-4, ε = 3 the per-step
δ is **3.16e-11 with independent coins and 2.76e-5 with a shared coin** and an aligned neighbour contribution (`u = 0`
reproduces the standard value exactly); the analytic estimate `q·δ_Gauss(ε=3, σ=0.5622) ≈ 5.7e-5` agrees in order (VERIFIED
arithmetic). Six orders of magnitude, on every step after the resume, for the gradient release and the probe alike. Not caused
by this design; inherited in exactly the configuration v1's DDP rows called closed.

**Disposition.** (a) **P0 prerequisite change** (§9.0): at the resume site re-derive the per-rank stream key after restoring
the cursor — `fold_in(restored_key, self._ddp.rank)`, the same derivation `:3803` uses on fresh construction — or write
per-rank sampler snapshots; correct the comment; add a 2-rank `distributed` test (T25) asserting post-resume inclusion masks
differ across ranks and that a single-process resume is bit-identical to today. (b) **Until that lands, the privacy statement
(§8.2) reads "single-process, or DDP without checkpoint resume"** and §9.2's DDP row cites the caveat. The same narrowing
applies to every existing DP run of the trainer, not only to this feature.

### 8.2 Residual non-per-example effects, named (refute-sensitivity R9, narrowed — §12 S9)

`runtime/masking.py:195-206` selects the attention path (None-mask `is_causal` SDPA fast path vs a materialised mask) from
`physical_mask.all()` over the **whole physical microbatch** (functorch-unwrapped; VERIFIED). Under add/remove, one padded
example in an otherwise all-valid microbatch changes the kernel for every microbatch-mate, so `g_y` for a mate `y` is a function
of the microbatch composition, not of `y` alone — by bf16 accumulation-order differences (0.45–0.5 % rel-L2, F8) or, if a route
flips on `y`, by up to `2C_g` after clipping. Each contribution is still clipped, but the sum's sensitivity is `C_g` only up to
this *systematic* floating-point term. Packed all-ones presets never flip the switch; the ragged kstack preset does, per
microbatch. **Disposition — narrow, not reject**: privacy proofs are stated for real arithmetic and floating-point deviations
are the standard caveat (the protocol's "distributed and microbatched paths must be equivalent to the analyzed single-process
mechanism" is where this belongs); but a *data-dependent kernel switch* is avoidable, so under DP training the path is derived
from a **public property**: `all_valid_attention := args.packed_sequences` (collator guarantee, set once) and otherwise the mask
is always materialised — i.e. the fast path is taken only when it is publicly known that no row is padded. This keeps every
`g_x` a function of `(x, θ_t)` up to the kernel's own non-determinism. The privacy statement names floating-point kernel
selection and accumulation order as the residual non-per-example effect. Change in `opaque-patches` (`masking.py`) gated on a
flag the DP trainer and the manual loops set; PR #980's fast-path parity is unaffected for packed data.

### 8.3 Tensor / state classification

| tensor / state | where it lives | class | logged? | checkpointed? | DDP |
|---|---|---|---|---|---|
| router logits, fp32 probs, executed routes `S`, `h^{(L,E)}(x)`, `P(x)`, `d(x)`, `w_x`, `Z_x`, surrogate value `S_x` | inside the vmapped loss closure | **private-internal** — never leave `clipped_grad`; not added to `loss_aux`; **the loss value is `CE_x` exactly** because the surrogate and z-loss enter value-neutrally (§1.1) — v1's "adds nothing to the un-noised loss mean" is now *true by construction* rather than false (§12 C2) | never | never | n/a |
| per-example probe gradient `λ·w_x·d^{(L,E)}(x)` | inside `clipped_grad` before summation | private-internal | never | never | n/a |
| `ClippedGradAux.group_norms["router_load_probe"]` (per-example `‖λ w_x d(x)‖`) | existing telemetry channel | **private, un-noised** — the trainer's `group_metrics` loop logs `group_norms.mean()` and a `clip_rate` for every group (`_dp_trainer.py:2259-2275`, VERIFIED) | **excluded** (`continue` on the probe group) | no | gathered in-process by `sync(aux)` |
| **`aux.grad_norms`, `aux.clipped_grad_norms` (per-example TOTAL norm over ALL leaves, probe included)** — `metrics["grad_norm"]`, `metrics["clipped_grad_norm"]` (`_dp_trainer.py:2251, 2257`, VERIFIED; `_clipped_fun.py:673-675` `"norms": norm.norm`; `_pytree.py:472` `orig_norm = sqrt(_sq_norm(leaves…))` over the whole pytree, VERIFIED) | existing telemetry channel | **private, un-noised, a function of `h(x)`** through `√(‖g_x‖² + ‖λ w_x d(x)‖²)` (0.30054 vs 0.30001 for the sup vs a random example, VERIFIED (refute-sensitivity) A1; information content ≤ `C_h²/(2‖g‖)` ≈ 1.8e-4 absolute). v1 missed this row (§12 S1/C3). | **with the feature on, `grad_norm` / `clipped_grad_norm` are computed as `√Σ_{g≠probe} group_norms_g²` from the per-group aux** (the pre-existing un-noised posture, F11, is not widened); `clip_rate` is already recomputed from surviving groups (`:2283`) | no | in-process |
| clipped-summed probe leaf, pre-noise | `ClippedPytree` in `training_step` | private — same status as the un-noised gradient sum | no | no | all-reduced by `sum_gradients_` |
| noised probe leaf `ŷ_t` | `NoisedPytree` handed to `on_pre_optimizer_step` / the loop's seam 3 | **public** (DP output; the only release added) | may be logged | via filter state | rank-identical (shared noise key) |
| `d̂_t`, `d̃_t`, `d̃^{(L,E)}_t`, `s_t`, `f̃_t`, `D_t`, `D^l_t`, dead-zone/trip flags, `φ_t` | `RouterLoadState` | **public post-processing** | yes: `router_load/D`, `/D_layer_max`, `/f_min`, `/f_max`, `/entropy`, `/shrink`, `/dead_zone`, `/noise_std = s_t`, `/tripped` | **yes** — sidecar `router_load_state.pt` | rank-identical; optional `register_sync_type(RouterLoadState, assert_equal)` |
| probe parameter `z` | `trainable_params["router_load_probe"]` | public constant **0** (zeroed each step; asserted) | no | as a zero tensor (harmless) | identical |
| `f̃_t` closure tensor | trainer / loop attribute | public | — | derived | identical |
| `aux.batch_size` (realised `\|B_t\|`) | existing | private, pre-existing; `Binomial(N,q)` vs `Binomial(N+1,q)` under add/remove — logged un-noised today (`_dp_trainer.py:2250`, VERIFIED); **never** used as a divisor for `d̂` | (pre-existing; added to the F11 list) | no | summed |
| second-moment squared stream of the probe | — | would consume budget (`paired_noise_stddevs` sums over all groups, `noise_allocation.py:153-175`) | — | — | **v1: `ConfigurationError` with `second_moment=True`**; v1.1: engine-side structural zero (§9.2; v1's "0.0 bound" plan withdrawn, §12 C6) |
| `C_g` / `T̄` / `δ` calibration pass | design-time script | **a private query if run on protected rows** — run on a disjoint held-out split or account it (§4.4) | design-time only | no | — |
| independent-draw sampler key/position, second noise-key counter (v2) | `RouterLoadReleaser` | public RNG state the privacy argument depends on | no | **yes** (sidecar; rank re-fold on resume) | sampler key rank-folded; noise key shared |
| `α, ρ, λ, C_h, g, T̄, T_max, β, W, c, E, k, L, τ, ζ` | args | public hyperparameters | yes | yes | — |
| pre-existing un-noised `loss`, `grad_norm`, `clip_rate`, `batch_size` means | existing | private, unaccounted — outside this task; flagged per F11 | yes (pre-existing) | — | — |

### 8.4 Privacy statement (for the mechanism page, the DP-FTRL page and the trainer docstring)

"When `router_load_release` is `monitor`, `surrogate` or `monitor_then_surrogate`, each step releases one Gaussian (or matrix)
mechanism on the concatenation of the clipped per-example gradients and the per-example token-weighted centred router-load
vectors `λ·(T_x/T̄)·(h^{(L,E)}(x) − k/E)`, with per-record bounds `C_g` and `C_h = λ·(T_max/T̄)·√(k·L·(1−k/E))·(1+g)`; the two are
one mechanism under Opaque's per-group allocation and the accountant is unchanged (`gaussian(nm)` per step under the stated
sampler / `mf_gaussian(nm, strategy)` for the horizon). The gradient noise is inflated by `√(1+ρ)`. The load estimate `f̃_t`
consumed by the loss and the monitors `D_t`, `D^l_t` are post-processing of previous releases. No other quantity derived from
private routing is released: the per-example probe-group norms are excluded from telemetry and the logged gradient norms are
computed over the non-probe groups; the per-example loss value carries no router-derived term. **Guarantee as run: single-process,
or DDP without checkpoint resume** (a DDP resume today restores one sampler key on every rank, which invalidates the
subsampling amplification for every release of the trainer; see §8.1). Residual non-per-example effects are floating-point
only: attention-kernel selection is derived from the public packing flag, not from the batch, and per-example gradients differ
from real arithmetic by accumulation order. The pre-existing un-noised logging of the batch-mean loss, gradient norms, clip rate
and realised batch size (F11) is unchanged by this feature and is outside its accounting." [v2 adds the independent-draw
sentence of §2.7.] [If packing is enabled: "the protected unit is one packed 1024-token row."]

---

## 9. Implementation plan (G10) and tests

### 9.0 P0 prerequisite (pre-existing trainer bug, §8.1)

`_dp_trainer.py:1794-1800`: after `from_state_dict(ctx.current_sampler, saved_sampler_state)` under `world_size > 1`, re-fold the
restored stream key by rank exactly as `:3802-3803` does on fresh construction (or write per-rank sampler snapshots); fix the
comment at `:3750-3757`; T25. Engine, `opaque-dpsgd`, `opaque-dpftrl`, `opaque-accounting`: **no other change in v1.** Reused as-is:
`PerGroup` (direct construction, `types.py:72-110`, VERIFIED: `groups`/`values` mappings), `clipped_grad`, `per_group_noise_stddev`,
`gaussian_noise`, `mf_gaussian_noise`, the serialization registry, `sum_gradients_`, the streaming Toeplitz inverse.

### 9.1 `opaque-patches`

| file | change |
|---|---|
| `src/opaque/api/patches/transformers/components/moe_stats.py` (**new**) | `router_load_and_probs(router_logits: Sequence[Tensor], attention_mask: Tensor \| None, *, top_k: int, num_layers: int) -> (h_layers: (L,E), P: (E,), T_x: scalar)`: `assert len(router_logits) == num_layers`; `m = (attention_mask != 0)` if given (2-D asserted) else all-ones; layer loop, out-of-place, `softmax(z.float())`, `topk`, broadcast-compare one-hot (no `F.one_hot`, no `scatter_add_`, no `bincount`); `T_x = m.sum()`; `h^l = (m @ onehot) / T_x.clamp(min=1)` with an explicit zero result when `T_x = 0`; `P` likewise with the common denominator `L·T_x`. `centred_load(h_layers, top_k)`; `load_balancing_surrogate(P, f_tilde, w_x, *, num_experts, top_k) = E·w_x·⟨f̃ − k/E, P⟩`; `router_z_loss(router_logits, mask)`. |
| `src/opaque/api/patches/transformers/components/router.py` (**new**) | `install_fp32_router(model) -> undo` / `remove_fp32_router(model)`: **instance-level** `types.MethodType` swap on every `MellumTopKRouter` (`logits = F.linear(h.float(), W.float())`, fp32 softmax, `topk`, renormalise, `scores.to(h.dtype)`, return `(logits_fp32, scores, indices)`); removable (§3). Opt-in. |
| `src/opaque/api/patches/transformers/components/cross_entropy.py` | new **named** kwarg `opaque_router_logits: bool = False` on the fused/chunked causal-LM `forward` (`:176-192`): when true, call the backbone with `output_router_logits=True` **without** taking the HF-aux fallback branch (`:212-233`, kept for HF's contract and its existing test); keep chunked CE; the existing `hasattr(outputs, "router_logits")` return path (`:359-371`) builds `MoeCausalLMOutputWithPast(loss, logits=None, router_logits=…, aux_loss=None)`. **The forward is installed only when `fused_linear_cross_entropy=True` reaches `apply_model_patches` (`_factory.py:378-392`, VERIFIED)** — callers that want the feature must pass it (§9.2). |
| `src/opaque/api/patches/transformers/runtime/masking.py` | `all_valid_attention` derived from a public flag (`packed_sequences`) under DP; the batch-content probe (`:195-206`) is used only when the flag is unset *and* the caller is not a DP grad transform (§8.2). |
| `src/opaque/api/patches/transformers/_factory.py` | (a) `grouped_moe = kwargs.get("grouped_moe", kernels or _grouped_route_available())` (§5.1) with a one-time log; (b) `router_fp32` opt-in routed to the instance-level installer. |
| `src/opaque/api/patches/kernels/moe.py` | `_grouped_route_available()` helper next to `:575-635`. |
| `src/opaque/api/patches/transformers/models/mellum.py` | docstring: opt-in fp32 router (pretraining-faithful, not a drift fix), grouped default, first-patch capture, **chunked CE requires the `fused_linear_cross_entropy=True` call flag**. |
| `docs/mechanisms/dp-sgd/moe-load-balancing.md` (+ DP-FTRL section) and `docs/user-guide/huggingface.md` | loss (with `w_x`), mechanism, cost table with the `nm_MF` bracket, privacy statement §8.4, primary sources. Diary-free. |

**How the statistics reach the loss, and why it composes (G10)** — now VERIFIED at toy scale by two independent agents
(refute-composition `t5_recorder_ckpt.py`; refute-feasibility checks A–E): `MellumModel.forward` is `@capture_outputs`
(`modeling_mellum.py:474`) with `_can_record_outputs["router_logits"] = OutputRecorder(MellumTopKRouter, index=0)` (`:431-433`);
hooks are installed once and append only while a `ContextVar` collector is active (`output_capturing.py:104-108, 259-272`).
Hence: **gradient checkpointing** (non-reentrant, `patches/torch/checkpoint/huggingface.py:31-56`) — the router hook fires 2L
times but the recorder returns exactly L tensors, every gradient leaf including the probe equals the non-checkpointed run to
rel-L2 0.0, and the surrogate gradient through logits captured *inside* the checkpoint region is carried exactly (VERIFIED,
both agents); v1's PLAUSIBLE is upgraded, the hook-with-overwrite fallback is no longer needed. **Microbatch chunks**: separate
vmapped calls (`_clipped_fun.py:273-296`) → separate collectors; chunk-invariance VERIFIED (check C, 4.7e-8). **`torch.compile`**:
HF's `CompileableContextVar`; if dynamo breaks, `_compile_with_fullgraph_fallback` (`_dp_trainer.py:4191-4228`) downgrades —
PLAUSIBLE, T18. **batchify**: `router_logits` is a tuple of `(T, E)` tensors — no shim. **MF latch**: constant two-group
`PerGroup`; `ρ, λ, C_g, T̄`, group map fixed at construction and asserted on resume.

### 9.2 `opaque-transformers` — trainer-independent helper first, trainer second (§12 F2)

| file | change |
|---|---|
| `src/opaque/api/transformers/moe_load.py` (**new**, trainer-independent; `opaque-transformers` already depends on engine/patches/dpsgd/dpftrl, `pyproject.toml:33-45`, VERIFIED) | `attach_probe(model, *, num_layers, num_experts, name="router_load_probe") -> str` (registers the zero `(L,E)` fp32 parameter — must run **before** `make_functional` / `partition_trainable` so it lands in the trainable pytree); `probe_bounds(clipping_norm: float \| PerGroup, trainable, *, ratio, num_layers, num_experts, top_k, mean_tokens, max_tokens, guard=1e-3) -> (PerGroup, lam)`: builds the user's `PerGroup` **over `trainable` minus the probe** (scalar → `fallback`) and adds the probe by direct construction `PerGroup(groups={**pg.groups, (name,): "router_load_probe"}, values={**pg.values, "router_load_probe": C_h})` — never by substring pattern (§12 C4); frozen dataclass `RouterLoadState(m_layers: Tensor(L,E), m: Tensor(E), noise_std: float, f_tilde: Tensor(E), phi: Tensor, step: int, tripped: bool, alpha_active: float, kind, beta, window, rho, dead_zone, top_k, num_experts, num_layers, mean_tokens, max_tokens)` (registry-serialisable; `LoadEmaState` round-trip VERIFIED phase-1 primitives §4.4); `filter_factors(strategy \| None, n_steps, kind, beta/window) -> phi` (§6.2; the caller passes its strategy object); `update(state, noised_leaf, lam) -> state` (§2.4 steps 1–7, bias-corrected); `summary(state) -> dict`; `telemetry_without_probe(aux, probe_name) -> (grad_norm, clipped_grad_norm)` (§8.3). |
| `trainer/_router_load.py` (**new**) | `RouterLoadCallback(TrainerCallback)` with `on_pre_optimizer_step(self, args, state, control, *, grads, trainable_params, **kw)`: reads `grads.pytree["router_load_probe"]`, calls `update`, zeros that leaf in place, sets α for the next step in `monitor_then_surrogate`. Registered automatically when on — reuses the existing `call_event("on_pre_optimizer_step", …, grads=noisy_grads, …)` seam (`_dp_trainer.py:2198-2206`) with **no new seam in `training_step`**. |
| `trainer/_training_arguments.py` | fields of §2.5. Validation: any non-`off` mode requires `clipping_mode == "fixed"`, `second_moment == False`, a recorder-capable family, and the chunked forward installed. |
| `trainer/_dp_trainer.py` `_apply_model_patches` (`:841-847`) | when `router_load_release != "off"`, pass `fused_linear_cross_entropy=True` unless `performance_kernels_config` sets it explicitly `False` (then `ConfigurationError` naming the flag). After patching, **require `hasattr(type(model).forward, "__opaque_patched__")`** and a **named** `opaque_router_logits` parameter in `inspect.signature(model.forward)` — **never accept `VAR_KEYWORD`** (v1's "exactly like `_fused_forward_uses_marker`" copied a rule that accepts `**kwargs`, `_sft_trainer.py:287-291`, VERIFIED; HF's forward has `**kwargs`, so the marker would be swallowed silently and `router_logits` would be `None` — §12 F3). |
| `_dp_trainer.py` `_setup_training` | if enabled: (1) `attach_probe` **before** `make_functional(partition_trainable=True)` (`:1357-1361`); (2) replace the clip-norm block (`:1455-1470`) result by `probe_bounds(...)`; (3) `phi = filter_factors(ctx.mf.strategy if ctx.mf else None, …)`; build `RouterLoadState`; (4) instantiate the callback; (5) store `λ, α, T̄, T_max, E, k, L` on the trainer; (6) `ConfigurationError` checks; (7) set the public `packed_sequences` flag for `masking.py` from the collator (§8.2). |
| `_dp_trainer.py` `_augment_inputs` (`:2302-2313`) | copy `callback.state.f_tilde` into `self._router_load_target`; assert the probe parameter is zero. No batch column. |
| `_dp_trainer.py` `compute_per_example_loss` (`:2314-`) and `trl/_sft_trainer.py` `compute_per_example_loss` (`:554-615`) | shared helper `self._apply_router_load_terms(loss, outputs, params, inputs)`: `h_layers, P, T_x = router_load_and_probs(outputs["router_logits"], inputs.get("attention_mask"), top_k=k, num_layers=L)`; `w = T_x / T̄`; `S = load_balancing_surrogate(P, self._router_load_target, w, …)`; `loss = loss + α_active·(S − S.detach()) + (params[probe] * (λ·w·(h_layers − k/E)).detach()).sum() [+ ζ·(Z − Z.detach())]`. Forward called with `opaque_router_logits=True`; the SFT `chunked_nll` path keeps `opaque_fused_loss_only=True`. |
| `trl/_dpo_trainer.py` `compute_per_example_loss_and_metrics` (`:1065-`) | helper on both policy forwards, pooled with `L·(T_c+T_r)` and `w = (T_c+T_r)/T̄_pair`; reference forward never receives the kwarg; assert no `router_logits` on it (T21). |
| `_dp_trainer.py` metrics (`:2249-2293`) | `grad_norm`/`clipped_grad_norm` from `telemetry_without_probe` when on; skip the probe group in the `group_metrics` loop; add `summary(state)` under `router_load/*`. |
| `_dp_trainer.py` `_save_checkpoint` / `_apply_runtime_state` / resume | sidecar `router_load_state.pt = opaque_state_dict(state)`; on resume assert `rho, beta/window, dead_zone, E, k, L, C_g, T̄` match (`CheckpointError`). `save_dp_runtime_state`'s fixed signature left alone. |
| `trl/_convert.py` `_drop_router_aux_loss` (`:69-76`) and the SFT/DPO converters | SFT on a recorder-capable family: `router_aux_loss_coef > 0` → `router_load_release="surrogate", router_aux_loss_coef=value` (info log); DPO → `monitor` with a message; unsupported families keep the warning. |
| second-moment exclusion (v1.1) | **engine** change: a per-group `zero_groups=("router_load_probe",)` option on the squared-stream clip so the probe's squared per-example stream is **structurally zero inside the vmapped clip**; then a zero second-moment bound is justified by the clip. v1's plan (zero after the fact + pass a `0.0` bound) is withdrawn: `per_group` rejects non-positive values (`_per_group.py:127-129`, VERIFIED) and a zero *noise* allocation on a stream zeroed in a separate step is a latent un-noised release if the zeroing is ever skipped (§12 C6). T20 asserts on the clipped squared stream. v1 raises. |
| **`examples/train_dpftrl.py` `mellum2-kstack`** (`:873-896`) and **`examples/train_dpo.py` `mellum2-codesec`** (`:1300-1318`) — **manual loops** | call the helper at four seams: (1) `attach_probe` before `make_functional`; (2) `probe_bounds(...)` where the loop builds `per_group` / the scalar clip (`train_dpftrl.py:1427-1454`); (3) between `noise_fn` and `opt.update` (`:1735-1760`): read the noised probe leaf, `update`, zero the leaf in place, refresh the closure `f̃`; (4) checkpoint save/restore of `RouterLoadState`. **Thread the collator's `attention_mask` into `per_example_loss_fn`** (`train_dpftrl.py:1369-1378` passes none today; the collate at `:1133-1135` returns `input_ids` only) — also correct for SDPA under right padding; pass `fused_linear_cross_entropy=True` (already done at `:1068`) and `opaque_router_logits=True`; new CLI args `--router-load-release`, `--router-aux-loss-coef`, `--router-load-ratio`, `--router-load-mean-tokens`; presets: kstack `surrogate`, 1e-4, 0.02, `T̄` measured on the held-out split; codesec `monitor`, 0.02. Both rely on the grouped default (or set `grouped_moe=True`). Under DDP the loop must (a) all-reduce the probe with `sum_gradients_`, (b) use a shared noise key, (c) not reuse the trainer's resume-key bug. |

DDP (`DPTrainer`): `sum_gradients_` all-reduces every leaf including the probe (`distributed/gradients.py:150-`), noise is added
per rank with the shared key after the reduction (`_dp_trainer.py:2172-2183`), so `ŷ_t`, the filter state and `f̃` are
bit-identical on all ranks — **for fresh runs; after a checkpoint resume the sampler-key caveat of §8.1 applies until P0 lands.**

### 9.3 Test plan (placement per ARC-006; markers per AGENTS.md; behaviour only)

`packages/opaque-patches/tests/transformers/models/test_mellum.py` + new `test_moe_router_stats.py` (tiny random-init models via
`build_moe_model`):

- **T1** fp32 router (opt-in): `router_logits.dtype == float32`, scores dtype = hidden dtype, indices equal to the unpatched
  router on an fp32 model; works under `vmap` and `vmap(grad)`; **installable and removable in-process**; unpatched contract
  untouched when off.
- **T2** `router_load_and_probs`: vmap-safe; per layer `Σ_e h^l = k`, `0 ≤ h^l ≤ 1`, `‖d^{(L,E)}‖₂ ≤ √(7L)`, `Σ_e P = 1`; equals
  an eager per-example loop; padded tokens excluded; `h` from the logits equals `h` from the router's returned `indices`;
  **`attention_mask=None` counts every position; a non-binary mask (additive 0/−1e9, mixed sign) is binarised to the same
  result as its 0/1 form; `T_x = 0` gives `h = 0, P = 0, d = 0`; a duplicated logits tuple (2L) raises.**
- **T3** surrogate identity (float64): `Σ_x ∇ℓ_x|_{f̃ = f(B)}` with `w_x = T_x/T̄` is collinear with `∇ load_balancing_loss_func`
  (cosine 1 − 1e-12) with scale `T_tot/(B̄T̄)`, **on ragged lengths**; equal lengths give equality; centred and uncentred
  surrogates give identical gradients; the value-neutral form has loss value `CE_x` exactly and the same gradient.
- **T4** chunked-CE forward with `opaque_router_logits=True`: returns `L` `(T, E)` router logits under vmap, `logits is None`,
  loss equals the same forward without the kwarg; with `output_router_logits=True` the HF-aux fallback is still taken;
  **without `fused_linear_cross_entropy=True` the forward has no named `opaque_router_logits` parameter and the trainer-side
  check raises `ConfigurationError`.**
- **T5** gradient checkpointing: `(h, P, grads)` equal to the non-checkpointed run to fp32 tolerance; `len(router_logits) == L`;
  the surrogate gradient through captured logits is carried (VERIFIED at toy scale already; the test pins it).
- **T6** grouped vs dense `opaque_moe` with the statistics enabled: identical `h`, grads within the parity-harness tolerance;
  `_grouped_route_available()` default resolution on CPU and CUDA.
- **T6b** masking: with `packed_sequences=True` the mask path is `None` for every microbatch; with it unset under the DP grad
  transform the mask is materialised regardless of batch content; per-example gradients of an all-valid example are identical
  whether or not a padded example is in its microbatch (fp32).

`packages/opaque-transformers/tests/opaque_transformers/test_router_load_balancing.py` (+ `test_moe_load_helper.py` for the
trainer-independent API, exercised through a minimal manual loop):

- **T7** setup: probe leaf `(L, E)` in `trainable_params`; `clip_norm` is a `PerGroup` with `C_h = ρ·C_g·(1+1e-3)`; scalar and
  dict `clipping_norm` converted; **a user pattern `{"router": …}` coexists with the probe** (no `ConfigurationError`); σ values
  equal `per_group_noise_stddev` closed form; Mahalanobis identity holds.
- **T8** pre-noise probe leaf equals `(λ/B̄) Σ_x w_x (h^{(L,E)}_x − k/E)` exactly and `group_norms["router_load_probe"] ≤ C_h`
  for adversarial single-expert routing at `T_x = T_max` (clipping never triggers, scale exactly 1, `clip_rate` 0) **on every CI
  device including MPS**.
- **T9** post-processing with injected known noise: `d̂`, layer pooling, projection, bias-corrected EMA/window, `s_t` recursion,
  dead zone (zero below `c·E·s²`, JS+ above with factor ≥ 1/2), clamp and `f̃` match closed form; `f̃_0 = k/E`; `f̃_1` equals the
  first (projected, shrunk) release; `Σ_e f̃ = k` up to clamp; per-layer `D^l` from the per-layer EMA.
- **T10** probe hygiene: after `n` steps `trainable_params["router_load_probe"] == 0` exactly; **loss value equals `CE_x`
  bit-for-bit; gradient equals `∇CE + α∇S`**; optimizer state for the probe is zero; **`off` runs are bit-identical to today's
  path; `monitor` runs differ from `off` only by the `√(1+ρ)` σ inflation and the noise stream** (both noise engines draw
  leaves sequentially from one per-step generator, `_gaussian.py:419`, `_engine.py:412-418`, VERIFIED (refute-composition)) —
  v1's "α = 0 runs bit-identical" is withdrawn (§12 C5).
- **T11** checkpoint round-trip: sidecar restores `RouterLoadState` exactly; resuming reproduces the next `f̃` bit-identically;
  mismatched `ρ`/`T̄` raises.
- **T12** (`distributed`, 2 Gloo ranks): identical `f̃` and filter state on both ranks after 3 steps; probe leaf all-reduced.
- **T13** `microbatch_size=2` chunks == single chunk (grads, `h`, `d̂`).
- **T14** DP-FTRL: `mf_gaussian_noise` + `band_mf_strategy(bands=4, momentum=0.95)` accepts the `PerGroup` latch across steps;
  realised `noise_stddev.values["router_load_probe"] == base·row_l2(t)`; `phi` from the strategy matches a dense small-n
  reference; Monte-Carlo over the noise key reproduces the filtered (bias-corrected) noise factor within 10 %.
- **T15** accounting invariance: `epsilon_at(δ)` **identical** with and without the feature under both stacks;
  `_build_mechanism` returns the same factory.
- **T16** converter: `router_aux_loss_coef` maps to `surrogate` for SFT on Mellum, to `monitor` for DPO, warns elsewhere.
- **T17** hygiene, scoped precisely: with the feature on, `metrics["grad_norm"]` and `metrics["clipped_grad_norm"]` equal
  `√Σ_{g≠probe} group_norms_g²` (mean over examples) and are **invariant to the routing of the examples** (two batches with
  identical gradients and different routes give identical values); the probe group is absent from `group_metrics`; `loss`
  equals the mean `CE_x`; `router_load/*` equals `summary(state)`; `loss_aux` carries nothing new.
- **T18** (`slow`) `torch_compile=True` one step == eager on CPU inductor (fullgraph fallback exercised).
- **T19** `ConfigurationError` for `clipping_mode ∈ {"auto", "adaptive"}`, `second_moment=True`, unsupported families, the
  chunked forward absent, and `performance_kernels_config={"fused_linear_cross_entropy": False}` with the feature on.
- **T20** (v1.1) second-moment exclusion: the probe's **clipped** squared stream is zero and consumes no budget.
- **T21** DPO pooling: `h`, `P` pooled over chosen + rejected with `L·(T_c+T_r)` and `w = (T_c+T_r)/T̄_pair`; reference forward
  contributes nothing; structural bound for the pair.
- **T22** decision rule on synthetic `d̂` streams: no false alarm under exact balance at ρ = 0.02 (dead zone: zero surrogate in
  ≥ 99.99 % of pure-noise steps at `c = 2`); detection of a deviation 1.0 within one window; `monitor_then_surrogate` switches
  α without changing `max_norm` or the accountant.
- **T23** (engine, v2) `auto_clipped_grad(fixed_groups=("router_load_probe",))`.
- **T24** (v2) independent draw: accounting equals `poisson(g(nm),q)*T | poisson(g(c·nm),q₂)*(T/m)`; realised release counts
  ~ `Poisson(q₂N)` per rank shard; **sampler key rank-folded, noise key shared, `f̃` rank-identical**; keys round-trip and re-fold
  on resume; MF + `"independent"` raises.
- **T25** (P0, `distributed`, 2 ranks): after `save → resume`, the two ranks' inclusion masks over the next 20 steps are **not**
  identical (and are each Bernoulli(q) in expectation); a single-process resume is bit-identical to the pre-fix behaviour.
- **T26** (examples, `slow`): the manual-loop helper round-trips through a 3-step tiny-Mellum DP-FTRL loop shaped like
  `train_dpftrl.py` (seams 1–4) and reproduces T8–T10 invariants.

Rust: no change. Docs build: the new mechanism page.

### 9.4 v2 engine change (deferred, written down)

`auto_clipped_grad(..., fixed_groups: tuple[str, ...] = ())` (`_auto.py:203`) → `_auto_scale_per_group` (`_pytree.py:313-347`):
named groups use `min(1, C/‖·‖)` while the rest use AUTO-S `R/(‖·‖+γ)`. `PerGroup` max_norm unchanged (constant ⇒ MF latch OK);
per-record bounds `R` for AUTO-S groups and `C_h` for the fixed one ⇒ privacy unchanged. Lifts the v1 `ConfigurationError`.
Same PR: the `zero_groups` option for the squared stream (§9.2 v1.1).

---

## 10. Validation plan on the real checkpoint (GPU) — G2 / G3

Script `examples/validate_mellum_dp.py` (one 80 GB GPU; `JetBrains/Mellum2-12B-A2.5B-Base` bf16 weights, `JetBrains/KStack`
**held-out shard disjoint from the preset's training rows** (§4.4), T = 1024 right-padded **ragged** rows, PEFT LoRA r = 16/α = 32
on q/k/v/o). **Budget (revised, §12 F4/F8):** oracle O0 + statistics ≈ 30 min; **O1 (fp32 floor) runs the dense `Opaque_MoE`
on CUDA** (fp32 is not routed to the fused kernel, §5.1) — budget ≈ 8× the bf16 expert cost for its 8-example microbatches,
≈ 20–40 min, or force the `_grouped_mm` route for fp32 in the script; the **preset's band-MF calibration is not part of this
budget**: run it once offline on a many-core host with its wall-clock recorded, and pass `--noise-multiplier` (or the bracket
1.544) to every validation run; the 200-step DP runs ≈ 30 min each with grouped MoE. Comparisons that need the fp32 router on
one side and not the other run in **separate processes** (or use the removable instance-level installer, §3). Every number
below is a design-time measurement on held-out data; nothing here is logged by a DP run.

### 10.1 Oracle definition and drift metric (G2)

**Oracle O0 (the "non-DP HF path", precision-matched):** same process, same patched module (`apply_model_patches` with the run's
`grouped_moe`, `router_fp32` and dtype), HF eager forward on one microbatch (B = 8) with `output_router_logits=False`,
`loss.backward()` per example in a Python loop ("HF loop") and once batched ("HF batched"). **O1 (floor):** the same loop in fp32
weights (48 GB; dense MoE on CUDA). **DP side:** `clipped_grad(..., clipping_norm=1e9, return_aux=True)` internals — bf16
`vmap(grad)` per-example vectors with the full Mellum patch set (grouped and dense; chunked CE; gradient checkpointing on/off;
**one ragged microbatch with the public-flag mask path**).

**Metrics, side by side:** (m1) per-example rel-L2 of the whole LoRA gradient, vmap vs HF loop and vs O1, median/max; (m2)
**route-flip counter** per `(token, layer)` (symmetric difference of executed top-8 sets from the same fp32 softmax), plus the
fraction of tokens with margin `p_(k) − p_(k+1) < 1e-6` and the exact-bf16-tie fraction; (m3) per-parameter-group rel-L2; (m4)
HF batched vs HF loop (upstream's own bf16 spread); (m5) loss absolute difference; (m6) the same with `router_fp32=True` on both
sides — **separate process** — and the bf16-vs-fp32-forward flip rate per layer.

**Acceptance (requirement (d)):** (A1) vmap-vs-HF-loop flips = **0** at equal precision (or every flip's margin < 1e-6); if
not, bisect — SDPA path (now public-flag driven), MoE path, RMSNorm — before touching the DP design; (A2) rel-L2(vmap, HF loop)
≤ 2 × rel-L2(HF batched, HF loop); (A3) fp32: rel-L2 ≤ 1e-5 and 0 flips; (A4) PR #980's "≈ 1.3 %" reproduced to ±0.5 pp under
this definition; (A5) **surrogate identity on the checkpoint in fp32 on a ragged microbatch**: batch mean of per-example
surrogate gradients with `f̃ = f(B)`, `w_x = T_x/T̄`, vs HF `output_router_logits=True` gradient — cosine ≥ 1 − 1e-6 and scale
`T_tot/(B̄T̄)` to 1e-4; also on a packed microbatch (equality).

### 10.2 Statistics to collect (G3) — 256 held-out KStack examples under the preset partition

| statistic | how | decides |
|---|---|---|
| per-example gradient-norm quantiles p10/p50/p90/p99/max | `clipped_grad(…, clipping_norm=1e9, return_aux=True).grad_norms` | `C_g` via the bias²+noise² curve; clip rate at the chosen `C_g` |
| **length distribution `T_x`: mean `T̄`, quantiles, fraction at `T_max`** | tokenizer | the preset's public `T̄` (§1.2); the `T_max/T̄` noise factor; whether packing is worth its unit change |
| per-coordinate RMS imbalance `δ = ‖f(B) − k/E‖₂/(k/E)/√E` (token-weighted, HF definition) and `‖·‖_∞`, for 32 batches of 256; batch-to-batch drift over 100 LoRA steps; **per-layer `δ^l`** | router recorder, eager | whether the term carries signal (need `r_smoothed ≲ 0.3·δ`), filter lag, τ, whether per-layer entries are informative at ×5.29 |
| per-example `‖d(x)‖₂` and `‖d^{(L,E)}(x)‖₂` distributions; fraction of experts with `h_e = 0` per example | same pass | tightness of the structural bounds (expect ≈ 1.0 / 5.3 balanced); H5 at T = 1024 |
| bf16-vs-fp32-forward router flip rate per layer; logit scale; top-8/9 margin | m2/m6 | documents the opt-in fp32 router |
| dense vs grouped step time and peak memory, microbatch 8 (frozen experts); dense-vs-grouped flip counter | `max_memory_allocated`, wall clock | G6 default; acceptance grouped ≥ 3× faster, flips 0 |
| aux/CE gradient-norm ratio at α = 1e-4 and 1e-3, on attention LoRA (and on router weights with the router unfrozen) | T3 machinery | H4 at scale |
| surrogate tracking error `‖f̃_t − f(B_t)‖/‖f(B_t) − k/E‖` over a 512-step dry run (lab-only `f(B_t)`) | monitor mode | lag/noise trade-off, dead-zone engagement rate |
| the preset's calibrated `nm_MF` at ε = 3 | offline `_calibrate_noise` / `cal.calibrate` on a many-core host, wall-clock recorded | replaces the bracket `[0.5622, 1.544]` by the value; every band-MF column of §2.6 rescales |

### 10.3 Mechanism acceptance on a 200-step DP run (`surrogate`, ρ = 0.02, both stacks)

DP-SGD/Poisson calibrated at ε = 3 for n = 200; band-MF/b-min-sep at **`nm = 1.544` (fixed, conservative)** unless the offline
calibration is available. (a) after warm-up (200 steps ≥ 2 EMA time constants, now bias-corrected so the steady-state gain is 1
from step 1) `f̃_t` tracks the lab-only token-weighted `f(B_t)` with per-entry error ≤ 10 % of `k/E`, and `‖d̃⁺_t −
d_true‖/‖d_true‖ ≤ 0.25` whenever `‖d_true‖² ≥ c·E·s²` (dead zone engages only below); (b) cosine between the DP surrogate aux
gradient and the exact batch aux gradient ≥ 0.8 whenever `δ ≥ 0.1`; (c) eval loss within run-to-run noise of the feature-off run;
expert-usage entropy on eval data not below the base model's; no expert with usage < 0.25·k/E; (d) reported ε identical with and
without the feature; (e) `group_norms["router_load_probe"]` max ≤ `C_h` (never clipped) — read inside the harness only; (f)
throughput with the grouped default within 20 % of the feature-off run; (g) `D_t` never trips under exact balance and the
dead zone zeroes the surrogate in ≥ 99.99 % of balanced steps; the `monitor_then_surrogate` switch changes neither `max_norm`
nor `epsilon_at`; (h) per-example gradient rel-L2 vs the precision-matched oracle every 100 steps ≤ 1.5 % with zero flips; (i)
`grad_norm` logged with the feature on equals the non-probe group norm (T17 on the real run).

---

## 11. Risks, and what would falsify the design

1. **No usable signal on real data.** If §10.2 finds `δ ≲ 0.03` on KStack, the smoothed noise at ρ = 0.02 (2.35 % DP-SGD /
   0.83–2.28 % MF) sits at the dead zone and the release buys nothing beyond the monitor. Not a faithfulness failure (the true
   aux gradient is equally silent). Mitigation: ρ = 0.1, a longer filter, the independent draw (DP-SGD), or `monitor` only
   (OLMoE §4.3, Tholoniat et al. 2024 — dropping the aux in fine-tuning is benign; transfer PLAUSIBLE).
2. **Lag bias.** If `f(B)` drifts faster than 100 steps, `f̃` chases a stale target. Falsifier: drift over 100 steps larger than
   `r_smoothed` (§10.2). Mitigation: shorter filter at higher ρ.
3. **The surrogate is inert even when needed** (router trainable, aux/CE ratio below the noise). Then the honest recommendation
   is Tholoniat's (drop the aux, freeze the router); the design stays correct, its utility claim would be false.
4. **Recorder under compile.** Checkpointing is now VERIFIED at toy scale; `torch.compile` remains PLAUSIBLE (T18).
5. **Flip-free vmap does not hold at 28 layers.** Falsifier: A1 fails → bisect kernels, not the DP design.
6. **Grouped-MoE default changes numerics** beyond the documented floor — **including the CPU test surface**, which flips to the
   grouped route for E ≥ 16 (VERIFIED (refute-feasibility) U11). Falsifier: any parity-harness tolerance moving, or a
   dense-vs-grouped flip count > 0 on the real model → keep dense default, fix the kernel.
7. **`nm_MF` bracket looseness.** `[0.5622, 1.544]` assumes the MC amplified bound is never looser than the deterministic
   un-amplified one (PLAUSIBLE). If the offline calibration lands above 1.544, use the calibrated value (privacy holds either
   way; only the MF load-error column moves).
8. **Probe leaf and the optimizer.** A future optimizer treating a zero gradient differently could drift the probe; T10 guards
   it; the `_augment_inputs` assertion makes silent drift a hard error.
9. **AUTO-S users lose the feature in v1**; `fixed_groups` (§9.4) is the deferred lift.
10. **fp32 router opt-in moves the fine-tune away from the HF-bf16 serving router.** Off by default; comparisons run in separate
    processes or with the removable installer. Falsifier of the default: bf16-routed eval after an fp32-router fine-tune degrades
    measurably → document, do not flip the default.
11. **Telemetry leakage.** Any future PR that adds `h(x)`, `P(x)`, `S_x` or the probe's norms to `loss_aux`/`group_metrics`,
    or reverts `grad_norm` to the all-leaf total, silently releases un-noised statistics. Guard: T17 (routing-invariance form).
    The pre-existing un-noised `loss`/`grad_norm`/`clip_rate`/`batch_size` logging (F11) sits next to this design and should be
    documented or noised — outside this task.
12. **DDP + resume (P0).** Until §9.0 lands, any resumed DDP run — with or without this feature — does not have the accounted
    guarantee; the privacy statement says so. Falsifier of the fix: T25.
13. **Data-dependent kernel selection elsewhere.** §8.2 fixes the one switch found; a second batch-content-dependent branch in
    the forward would be the same class of issue. Guard: T6b's "all-valid example's gradient is invariant to a padded
    microbatch-mate" test extended to any new kernel switch.
14. **Faithfulness is to Fact A, not to HF Trainer** (per-microbatch `f`, `G·α` under accumulation); stated.
15. **Concurrent releases under DP-FTRL** (would need concurrent composition, theorem numbers not verified) — deliberately avoided.
16. **DPO pair pooling** not exercised; T21 and the `monitor` preset contain it.
17. **Experts-trainable variant**: PEFT `target_parameters` under `functional_call`+vmap may not work; mechanism unaffected.
18. **Adjacency/normalisation under b-min-sep**: `E|B_t| = B̄` from step 0 via the warm start (VERIFIED by two agents); the
    centred release's unbiasedness does not depend on it beyond scale.
19. **Ragged-length scale factor.** With `T̄ = T_max` the aux is down-weighted by `(T̄_true/T_max)²`; with a measured `T̄` the
    release noise grows by `T_max/T̄`. Both are stated; neither affects privacy. Falsifier of the default: §10.2's length
    distribution shows `T̄_true/T_max < 0.5` on KStack — then set `T̄` in the preset (ρ = 0.05 keeps the smoothed error ≤ 2.4 %).
20. **Unverified at scale.** Every magnitude except the accounting table, the allocator identity, the MF filter factors and the
    hockey-stick computation comes from random-init toys; requirement (c) is settled only by §10.2, and (d) only by §10.1.

**Sources relied on** (all VERIFIED from primary text by phase-1 `critic` R6 / `literature` unless marked; not re-fetched):
Switch Transformer eqs. (4)–(6) https://arxiv.org/abs/2101.03961 §2.2; Andrew et al. 2021 Thm 1 https://arxiv.org/abs/1905.03871;
Zhu–Dong–Wang Def. 7 / Thm 10 / Thm 11 https://arxiv.org/abs/2106.08567; Feldman–Shenfeld Lemma 3.2 / Thm 3.3
https://arxiv.org/abs/2602.17284 (as cited by `src/amplification/poisson.rs`); Denisov et al. Thm 2.1 https://arxiv.org/abs/2202.08312;
Dong–Roth–Su Thm 2.7 https://arxiv.org/abs/1905.02383; Dong & Ganesh b-min-sep Alg. 2 https://arxiv.org/abs/2602.09338 (as cited
by `_b_min_sep.py`; PLAUSIBLE for the algorithm number); ST-MoE §3.1 eq. (5) https://arxiv.org/abs/2202.08906; Mellum 2 Technical
Report https://arxiv.org/abs/2605.31268 §3.6, §5.1.2, §5.2, appendix (FP32 router); Tholoniat et al. https://arxiv.org/abs/2402.07334;
OLMoE https://arxiv.org/abs/2409.02060; Wang et al. 2024 https://arxiv.org/abs/2408.15664; DeepSeek-V3 https://arxiv.org/abs/2412.19437;
Davody et al. 2020 https://arxiv.org/abs/2006.10919; Ponomareva et al. "How to DP-fy ML" §5; Bu et al. 2023 AUTO-S
https://arxiv.org/abs/2206.07136 (PLAUSIBLE, theorem numbers not re-fetched). The James–Stein / χ² dead-zone arithmetic is
elementary (χ²₆₃ tails computed with `scipy.stats.chi2`, VERIFIED `checks.py`), no external source claimed.

---

## 12. Refutation log — every refutation, its disposition, and the evidence

Legend: **accept** = the refutation is right and the fix is applied above; **narrow** = right in part, the surviving part is
applied and the rest explained; **reject** = wrong, with evidence. Severity is the refuter's. "Where" points to the v2 section
that carries the change. Evidence tags: VERIFIED = I re-read the cited lines or re-ran the cited script this session.

### Sensitivity lens (`phase2-refute-sensitivity.md`)

| id | claim refuted (v1) | severity | decision | evidence and what changed |
|---|---|---|---|---|
| **S1** | T17 / §8: "no logged key is a function of un-noised `h`"; only `group_norms["router_load_probe"]` needs excluding | minor | **accept** | `metrics["grad_norm"] = aux.grad_norms.mean()` and `clipped_grad_norm` (`_dp_trainer.py:2251, 2257`, VERIFIED) are the per-example *total* norm over all leaves (`_clipped_fun.py:673-675`, `_pytree.py:472`, VERIFIED), probe included; numerically 0.30054 vs 0.30001 (VERIFIED (refuter) A1). v2 §8.3 adds the row; with the feature on the two metrics are computed as `√Σ_{g≠probe} group_norms_g²` (`telemetry_without_probe`, §9.2); T17 restated as a routing-invariance test. Magnitude ≤ 1.8e-4 absolute — a hygiene, not an ε, issue. |
| **S2** | Load-leaf bound "structural / never an active clip" with `h` normalised by the configured `L` | minor | **accept** | A8 (VERIFIED (refuter)): 2L captured tensors → `Σ_e h = 2k`, probe norm 2.035·C_h, clip fires, release biased 2×. v2 §1.4: normalise by `len(router_logits)·T_x` **and** `assert len(router_logits) == num_layers`; with the `(L, E)` carrier a duplicate capture also fails the reshape loudly. Privacy was never at risk (the clip bounds it); the unbiasedness claim now holds by construction. |
| **S3** | `C_h = λΔ_h(1+1e-6)` keeps the ULP-guarded clip from firing on round-off | minor | **accept** | `_guard_scale` shrinks by `2(u_store + norm_roundoff)` (`_pytree.py:111-131, 135-148`, VERIFIED); `norm_roundoff` scales with leaf count, widest leaf and the accumulator dtype (`:214-224`, VERIFIED): 1.8e-7 CPU/CUDA but 1.3e-4 on MPS (VERIFIED (refuter) A10) > 1e-6. v2 §2.1: flat `g = 1e-3` (≥ 7× the largest measured shrink; moves ρ by 0.1 %), setup assertion against the engine's round-off bound where exposed, T8 on every CI device incl. MPS. |
| **S4** | `h`, `P` weighted by "the attention mask"; bound structural | minor | **accept** | Bound holds for any non-negative weighting (A5, VERIFIED (refuter)); a same-sign additive mask keeps the bound but scores padding; a mixed-sign mask gives `‖d‖ = 3.18 > 2.646` (`edge3.py`, VERIFIED (refuter)); HF float-casts and multiplies (`modeling_mellum.py:581`). v2 §1.1/§1.4/§9.1: `m = (attention_mask != 0)`, 2-D asserted; T2 covers additive, mixed-sign and `None` masks. |
| **S5** | "Shrinkage guarantees the design degrades to zero, never to a random regulariser" | minor | **accept** | JS+ alone zeroes only when `‖d̃‖² < (E−1)s²`, which pure noise exceeds ≈ 48 % of the time (52.3 % zeroed at δ = 0, VERIFIED (refuter)); dof nit (threshold should be `E·s²`) also right. v2 §2.4: dead zone `‖d̃‖² < c·E·s²` with **`c = 2`** (per-step false pass 4.2e-6, 0.07 expected over the horizon; `c = 1.5` gives 6.3e-3 — VERIFIED `checks.py`, exact χ²₆₃ tails; the refuter's "0.3 %" for c = 1.5 was a Gaussian approximation), JS+ with `E·s²` above it; the claim is restated quantitatively (§0.1, §2.4); threshold column of §2.6 recomputed (×√2). |
| **S6** | EMA β = 0.99 from `d̃_0 = 0` tracks the imbalance; acceptance 10.3(a) | minor | **accept** | Uncorrected gain `1−β^t` = 0.634 / 0.866 / 0.951 / 0.990 at t = 100 / 200 / 300 / 460 (VERIFIED `checks.py`). v2 §2.4/§6.2: Adam-style correction of both the estimate and its noise std (`m_t/(1−β^t)`, `s_t/(1−β^t)`; corrected std 0.1032 at t = 100 vs 0.0710 stationary); under MF the same factor applies because the EMA's signal gain is a property of the filter, not of the noise; window mean divides by `min(t, W)`; 10.3(a) restated. |
| **S7** | Per-layer release costs ×5.29 relative noise, "never free", "usable under MF only at ρ ≥ 0.2" | minor | **accept** | At the same ρ, `λ_L = ρC_g/√(7L)` is `√L` smaller and the layer mean divides the noise by exactly `√L`: pooled per-entry noise 0.04149 either way (VERIFIED `checks.py`: 0.04150 vs 0.04152 MC, closed forms equal; refuter `filters.py` ratio 1.0000). v2: the `(L, E)` vector **is** the carrier (§0.1, §1.1, §2.2, §2.4), pooled surrogate default, per-layer entries free at ×5.29 for diagnostics; candidate (c) re-argued on faithfulness grounds only (§1.3); §2.6 column relabelled; §7 row rewritten. Extra state 1792 floats (§5.2). |
| **S8** | `C_g` calibration pass "on a public proxy (KStack is public; the preset's own dataset)" | minor | **accept** | A non-DP pass over protected rows is a query on the protected set (review protocol "hyperparameter searches … included in the privacy statement"); v1's own §8 row said so and §4.4's parenthetical contradicted it. v2 §4.4, §8.3, §10: the calibration split (and every §10.2 statistic incl. `T̄`) must be a **held-out shard disjoint from the protected set**, or the accounted quantile release; "KStack is public" wording withdrawn. |
| **S9** | "everything else in the forward is per-example, so vmap is exact up to floating point"; `all_valid_attention` relied on without a privacy caveat | minor (pre-existing) | **narrow** | `masking.py:195-206` selects the kernel from `physical_mask.all()` over the whole physical microbatch (VERIFIED); under add/remove one padded example changes mates' kernels — systematic, not random, floating-point dependence (rel-L2 0.45–0.5 %, or up to `2C_g` on a flip). *Accepted*: the caveat is added to the privacy statement (§8.4) and the switch is derived from a **public** packing flag under DP (§8.2, §9.1, T6b), so `g_x` is a function of `(x, θ_t)` up to the kernel's own non-determinism. *Not accepted as a defect of the mechanism*: privacy proofs are stated for real arithmetic; the bound `C_g` on each clipped contribution is intact; the ragged preset is the one that exercised the switch, and it now materialises the mask consistently. |

Also from the sensitivity report, not in the JSON list: **`T_x = 0`** rows are safe via the engine's NaN sanitiser (VERIFIED
(refuter) `edge.py`/`edge2.py`); v2 makes the zero contribution explicit in the helper (§1.1, T2) so it does not depend on the
sanitiser. The cost table (§2.6) was independently reproduced entry-for-entry by that agent (VERIFIED (refuter) `costtable.py`).

### Composition lens (`phase2-refute-composition.md`)

| id | claim refuted (v1) | severity | decision | evidence and what changed |
|---|---|---|---|---|
| **C1** | "DDP: nothing to add"; "provably DP … as run" unconditional under DDP | **major** (pre-existing) | **accept** | Resume restores rank 0's sampler key on every rank with no rank re-fold (`_dp_trainer.py:3750-3757` comment; `:1794-1800` restore; fold only at `:3802-3803`, all VERIFIED). Shared coins across ranks break the independence Feldman–Shenfeld Lemma 3.2 / ZDW Thm 11 need: hockey-stick δ at ε = 3, σ = 0.5622, q = 5.12e-4 is 3.16e-11 (independent) vs **2.76e-5** (shared coin, aligned neighbour); `u = 0` reproduces the standard value (VERIFIED, `corr_coins.py` re-run); analytic order check `q·δ_Gauss ≈ 5.7e-5` agrees. The trainer's "accounting is unaffected" comment is wrong. v2: §8.1 (analysis), §9.0 (P0 prerequisite fix at the resume site + T25), **privacy statement narrowed to "single-process, or DDP without checkpoint resume"** (§8.4) until P0 lands; §9.2 DDP row cites it; §2.7(iii) forbids replicating the bug for the v2 sampler. Not caused by the probe leaf; inherited by every DP run of the trainer. |
| **C2** | §8 row 1: per-example aux value "never leaves `clipped_grad`"; the design "adds nothing" to the un-noised logged loss mean | minor | **accept** | `grad_and_value` returns `CE_x + α·S_x`, `ClippedGradAux.loss_values` carries it, `metrics["loss"] = aux.loss_values.mean()` (`_dp_trainer.py:2249`, VERIFIED) — an un-noised function of `P̄(B_t)`; v1's T10 even required it. v2 §1.1: **value-neutral** surrogate and z-loss `α·(S − sg[S])` — value exactly 0, gradient `α∇S`; `loss_values = CE_x` exactly; T10 rewritten; §8.3 row 1 now true by construction. |
| **C3** | §8 row 3: skipping the probe group in `group_metrics` excludes its un-noised norm | minor | **accept** | Same fact as S1 (`grad_norm`/`clipped_grad_norm` are all-leaf totals; `clip_rate` is already recomputed from surviving groups, `:2283`, VERIFIED). Fix as S1. |
| **C4** | §9.2 "a scalar `clipping_norm` becomes `per_group(trainable, router_load_probe=C_h, fallback=C_g)`, a dict gets the key"; §4.7 example `per_group(…, gate=…, experts=…)` | minor | **accept** | `per_group` matches by substring and raises on ≥ 2 matches (`_per_group.py:143-158`, VERIFIED); `"router"` (the natural pattern for `mlp.router.weight`) is a substring of `router_load_probe`; `experts.gate_up_proj` matches both `gate` and `experts`. v2 §2.2 step 4, §4.7, §9.2: build the user `PerGroup` over `trainable` minus the probe, add the probe by direct construction (`PerGroup(groups=…, values=…)`, `types.py:72-110`, VERIFIED fields); non-overlapping patterns `mlp.router`, `experts.`, `self_attn`; T7 asserts a user `{"router": …}` group coexists with the probe. |
| **C5** | T10: "α = 0 / `off` runs are bit-identical to today's path" | minor | **accept** | Only `off` is: with the feature on, σ_g is inflated by `√(1+ρ)` and every leaf's noise realisation changes because both engines draw leaves sequentially from one per-step generator (`_gaussian.py:419`, `_engine.py:412-418`; `probe_optimizer.out`, VERIFIED (refuter)). v2 T10 restated (§9.3); documented on the mechanism page. Reproducibility only. |
| **C6** | v1.1 second-moment exclusion via "zero the probe leaf of `squared_grads` and pass a `0.0` second-moment bound" | minor | **accept** | (a) `per_group` rejects non-positive values (`_per_group.py:127-129`, VERIFIED); (b) a zero *noise* allocation on a stream zeroed in a separate step is a latent un-noised release of `Σ(λd(x))²` if the zeroing is skipped/reordered — sensitivity 0 is legitimate only when the clip enforces it. v2 §9.2/§9.4: v1 keeps `ConfigurationError`; v1.1 adds an engine-side `zero_groups` option so the squared stream is structurally zero inside the vmapped clip; T20 asserts on the clipped stream. |
| **C7** | §2.7(i): second sampler key rank-domain-separated; second noise "from a separately rooted key" | minor | **accept** | Under DDP the noise key must be identical across ranks and applied after `reduce_pytree_` (gradient key shared, `_dp_trainer.py:1478`, VERIFIED); v1 never said the second noise key must *not* be rank-folded. v2 §2.7: sampler key `fold_in(key, "opaque.moe.load_release", rank)`; noise key `fold_in(key, "opaque.moe.load_release.noise", step)` shared; noise after the all-reduce; T24 asserts rank-identical `f̃`; resume re-folds the sampler key by rank. |

Also from the composition report: the realised `batch_size` (`_dp_trainer.py:2250`, VERIFIED) is an un-noised `Binomial(N,q)`
vs `Binomial(N+1,q)` side channel — added to the F11 list (§8.3, §8.4, §11.11). That agent also turned v1's PLAUSIBLE
recorder-under-checkpointing claim into VERIFIED at toy scale (`t5_recorder_ckpt.py`: L tensors, hook fires 2L times, rel-L2
0.0 on every leaf incl. the probe) — v2 §9.1 records it and drops the hook-with-overwrite fallback.

### Feasibility lens (`phase2-refute-feasibility.md`)

| id | claim refuted (v1) | severity | decision | evidence and what changed |
|---|---|---|---|---|
| **F1** | Identity exact at the preset because lengths are equal ("packed T = 1024 presets"); equal example weights; example-mean load released | **major** | **accept** | The preset tokenises with `truncation=True, max_length=1024` and no packing (`train_dpftrl.py:1090-1094`, VERIFIED), collator pads (`:1130-1135`), labels masked (`:1376`): ragged rows. Under ragged lengths HF pools with one denominator `L·T_tot` (`modeling_mellum.py:575-603`); the `T_x/T_tot` surrogate reproduces HF to 2.2e-7, the equal-weight one is **15.6 % rel-L2 off** (VERIFIED (refuter) check F); my `checks.py` confirms cosine 1.000000000000 with the public-constant weight and scale exactly `T_tot/(B̄T̄)`. v2 §1.1–1.2: `w_x = T_x/T̄` with public `T̄` (default `T_max`) on **both** the surrogate and the probe term; bound `(T_max/T̄)·√(7L)` structural; released statistic and its `(T̄_true/T̄)` scale stated; CE convention F4 stated as pre-existing; §0.2(b) relabelled; A5 runs on a ragged microbatch. The refuter's option (i) *packing* is offered as an option, not the default, because it changes the **protected unit** to a packed row (§2.1, §7) — a privacy-statement change the refuter did not weigh. Option (iii) (a third scalar group for `Σ T_x/T_max`) is not adopted: the realised scalar has ≈ 6 % relative sd, so a public `T̄` measured on the held-out split is as good without a ratio-of-noisy estimator. |
| **F2** | §9.2 preset rows: the presets get the feature through `DPTrainer` seams | **major** | **accept** | Neither example references `DPTrainer`/`DPDPOTrainer` (grep VERIFIED: only two comments mention `DPTrainer`); `train_dpftrl.py` builds its own `per_example_loss_fn` (`:1369-1378`), clip fn, noise fn, calibration; `train_dpo.py` likewise (`:544-562`). v2 §0.3, §7, §9.2: the mechanism is a **trainer-independent helper** (`opaque.api.transformers.moe_load`: `attach_probe`, `probe_bounds`, `RouterLoadState`, `filter_factors`, `update`, `summary`, `telemetry_without_probe`; statistics in `opaque-patches` `moe_stats`) called from both manual loops at four seams; new CLI args; `attention_mask` threaded into the kstack loss (which passes none today, `:1377`, VERIFIED); T26. `DPTrainer` consumes the same helper. Migrating the presets onto `DPSFTTrainer`/`DPDPOTrainer` is noted as the alternative, not chosen (larger, separate change). |
| **F3** | `DPTrainer`'s default path reads the loss through the chunked LM-head forward; the kwarg is detected "exactly like `_fused_forward_uses_marker`"; peak memory stays that of PR #978 | **major** | **accept** | The chunked/fused forward is installed only when the caller passes `fused_linear_cross_entropy=True` (`_factory.py:378-392`, VERIFIED); `DPTrainer` passes only `performance_kernels_config` (`_dp_trainer.py:841-847`, VERIFIED); `_fused_forward_uses_marker` accepts `VAR_KEYWORD` (`_sft_trainer.py:287-291`, VERIFIED) and HF's forward has `**kwargs`, so the marker would be swallowed and `router_logits` would be `None` (VERIFIED (refuter): `causal_lm_forward_is_opaque_patched: false` on the tiny build). v2 §1.1, §9.1, §9.2: detect the **named** parameter only and require `__opaque_patched__` at setup else `ConfigurationError` naming the flag; `DPTrainer._apply_model_patches` passes `fused_linear_cross_entropy=True` itself when the feature is on (explicit `False` → error); T4/T19 cases; §5.2 memory claim conditioned on installation. The example already passes the flag (`train_dpftrl.py:1068`, VERIFIED). |
| **F4** | `nm_MF` "free to record" from the trainer's start-up calibration; ≈ 30 min budget; "several × larger" | **major** (validation plan) | **accept** | `epsilon_at` clamps `mc_resolution` to `δ/2 = 5e-7` (`_base.py:216-221`, VERIFIED (refuter)) regardless of `--mc-resolution` (`train_dpftrl.py:704-707`, VERIFIED) → 64 997 003 transcripts per direction; no phase-2 evaluation finished on 4 cores, not even at n = 256. The deterministic un-amplified bound gives **`nm = 1.5439` at ε = 3** with `sensitivity = 1.0000` (VERIFIED, `nm_bracket.py` re-run, 24 s). v2 §2.6 carries both `nm = 0.5622` and `1.544` columns (band-MF smoothed error at the default row **≤ 2.28 % of k/E**, dead zone ≤ 0.032), §6.4 explains the clamp and budgets the calibration offline, §10 uses `nm = 1.544` as a conservative fixed multiplier (a valid guarantee in its own right) for the 200-step MF run; "several ×" withdrawn. The bracket's "amplification cannot raise the required nm" step is PLAUSIBLE w.r.t. MC-bound looseness, stated (risk 7). |
| **F5** | §5.2: router-logits tuple costs "0 extra — already live in the autograd graph" | minor | **accept** | `softmax` saves its output and `F.linear` its inputs; the logits are kept alive only by the recorder's references: 29.4 MB bf16 per microbatch of 8, + 58.7 MB fp32 softmax for the surrogate backward, 58.7 MB with the fp32 router (VERIFIED arithmetic; retention VERIFIED (refuter) check E). v2 §5.2 table rewritten with the numbers; conclusion (negligible) unchanged. |
| **F6** | §1.4: mask for `h`, `P` is the attention mask because "HF passes `attention_mask` to the aux" | minor | **accept** | HF passes whatever the caller passed; the kstack loop passes none (`train_dpftrl.py:1377`, VERIFIED), so HF-faithful behaviour there is `None` (pad rows routed and counted). v2 §1.4 documents both cases; the loop threads the collator's mask (§9.2); T2 covers `None`. |
| **F7** | §11.10 / §10.1 (m6): bf16-routing eval after an fp32-router fine-tune, and the fp32-router pair "in the same harness" | minor | **accept** | Class-level `_patch_forward` is guarded by `__opaque_patched__` (`_router.py:59-92`, VERIFIED (refuter)) and cannot be toggled off in-process. v2 §3, §9.1: the fp32 router ships as a **removable instance-level** `types.MethodType` swap; §10/§11 specify separate processes for the comparisons regardless. Faithfulness reading (executed-routing change, no weight change, opt-in default off) upheld. |
| **F8** | §10 budget "≤ 30 min for the oracle + statistics" including the fp32 floor O1 | minor | **accept** | `opaque_moe` routes only bf16/fp16 CUDA to the fused kernel; the `else` at `kernels/moe.py:614-632` sits inside `if _TRITON_AVAILABLE and x.is_cuda`, so CUDA fp32 falls to the dense `Opaque_MoE` (VERIFIED (refuter) read). v2 §5.1 notes it; §10 budgets O1 at dense cost (≈ 20–40 min) or forces the `_grouped_mm` route. |

Also from the feasibility report: checks A–E (recorder under `vmap(grad)`, vmap = eager = module backward at 0.0 rel-L2,
`clipped_grad` + `PerGroup` + microbatching exact, allocator and MF latch, checkpointing) are VERIFIED at toy scale on the
*patched* model and are cited in v2 §0.2(d), §2.2, §9.1; the `scatter_add_` failure of HF's `output_router_logits=True` under
vmap was reproduced (§7 last row).

### Tally and what did not change

23 JSON refutations: **22 accepted, 1 narrowed (S9), 0 rejected.** Four major (C1, F1, F2, F3) plus F4 (major for the
validation plan). Privacy claims changed visibly: (i) the "as run" guarantee is now conditioned on single-process or
never-resumed DDP until the P0 sampler-key fix lands (§8.1, §8.4); (ii) floating-point kernel selection is named as the
residual non-per-example effect and made data-independent (§8.2); (iii) two un-noised telemetry channels the design would have
touched (`grad_norm` totals, the logged loss value) are closed by construction rather than by a table entry (§8.3); (iv) the
faithfulness claim is now exact in direction for any length pattern with a stated scale factor, instead of exact only under a
packing premise the preset does not satisfy (§1.2). Unchanged, because every attack upheld it: the joint per-group release
(one `gaussian(nm)` / `mf_gaussian(nm, strategy)` per step, Mahalanobis identity 1.000000 at six ρ with the real allocator, three
agents), the structural load bound (attained, never exceeded, real clipper), the public divisor `B̄`, b-min-sep's `E|B_t| = B̄`
from step 0, adaptive use of `f̃_t`, the empty-batch and fully-masked-row paths, RNG domain separation, the probe through
three optimizers, the `√(1+ρ)` price, the band-MF filter factors, the v2 `|` accounting numbers, and the accountant-unchanged
claim.
