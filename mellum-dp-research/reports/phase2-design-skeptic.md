# Phase 2 — design-skeptic: the simplest provably-correct DP objective for Mellum2, and exactly when the surrogate release is needed

Agent: `design-skeptic`. Lens: SKEPTIC / SIMPLEST-CORRECT. Repo `/home/user/opaque` @ `ef1abc5`
(branch `claude/mellum-dp-representation-r6slaz`). Inputs: BRIEF.md, PHASE1-DIGEST.md (F1–F11, G1–G10),
the five phase-1 reports, the critic report, `.junie/differential-privacy-review.md`.
Two new CPU checks for this note (each < 10 s, `uv run python`):
`scratchpad/research/design-skeptic/a_fp32_router_flips.py` (→ `a_out.json`) and
`scratchpad/research/design-skeptic/b_cost_table.py` (stdout quoted inline).
Tags: **VERIFIED** = I read the cited lines / ran the cited script, or a named phase-1 script executed it;
**PLAUSIBLE** = derived or read, not executed end to end. Repo paths relative to `/home/user/opaque`;
HF = `.venv/lib/python3.11/site-packages/transformers/…` (5.16.1). Theorem citations are the ones the critic
re-verified from primary PDFs (critic R6); I did not re-fetch papers (the alphaXiv MCP failed to connect this
session) and I cite nothing beyond what phase 1 read.

---

## 0. Position in one paragraph

Mellum2's HF objective has exactly one batch-coupled term, the Switch-style load-balancing loss, and it is a
**regulariser on the router** (F1, F3). In every Opaque Mellum2 preset the router and the experts are frozen and
only attention LoRA (r = 16, q/k/v/o) trains (F9). For that regime the simplest correct DP objective is the
per-example causal-LM cross-entropy **with the balancing coefficient at 0** — which is literally what the
checkpoint's HF class computes by default (`output_router_logits=False`, `configuration_mellum.py:102`;
`modeling_mellum.py:692-700`, VERIFIED), what the only DP-MoE paper did (Tholoniat et al. 2024), and what
OLMoE did for SFT/DPO with no collapse (F10). "Batch statistics make it impossible" is wrong for a precise
reason: the batch statistic `f(B)` enters the gradient only as a **constant coefficient vector** (F3), so per-example
gradients exist for any constant `f̃`, clipping bounds them regardless of routing discontinuities, and the load vector
itself needs no clipping (‖h_x‖₂ ≤ √k structurally, F5). "The clipping norm cannot be reasonable" is also wrong:
the aux term at α ≤ 1e-3 moves per-example gradient norms by < 0.1 % (E3), route flips add ~0.2 pp to attention-only
gradient error (E1b), and the load leaf is a separate per-group bound that never touches the gradient's clip
threshold. What is *not* free is the assumption "the router stays balanced": the design therefore adds a cheap
**DP-released imbalance monitor** (a second `PerGroup` leaf at ρ = 0.02 → ×1.010 gradient noise, accountant
unchanged) with an explicit decision rule, and — because the monitor leaf *is* the load release — the exact
released-load surrogate (F3/F6/F7) can be switched on mid-run with **no change to the accountant or to the MF
sensitivity latch** if the rule trips or the router/experts are trainable. Everything else (fp32 routing, dense-MoE
default, MF smoothing, hygiene, implementation) follows.

**Two regimes, one mechanism family.**

| regime | trainable | objective coefficient α | load leaf | accountant |
|---|---|---|---|---|
| **A (presets)** | attention LoRA only; router + experts frozen | **0** (default) | `monitor` (ρ = 0.02) recommended; `off` allowed | unchanged |
| **B (conditional)** | router and/or experts trainable, **or** monitor tripped | 1e-4 default (Mellum2 SFT value), 1e-3 = HF-Trainer parity | `surrogate` (same leaf; ρ = 0.02–0.10) | unchanged |

---

## 1. Objective (G1)

### 1.1 The per-example loss (both regimes; α = 0 collapses it to CE)

For example `x` (right-padded; attention mask `m_{x,t}`, `T_x = Σ_t m_{x,t}`), layers `l = 1..L` (28), experts
`e = 1..E` (64), `k = 8`, router logits `z^l_{x,t}` (stock: bf16 `F.linear`, `modeling_mellum.py:334`),
`p^l_{x,t} = softmax_fp32(z^l_{x,t})` (`:335`), executed top-k set `S^l_{x,t} = topk_k(p^l_{x,t})` (`:336`):

```
P_e(x; θ) = (1/(L·T_x)) Σ_l Σ_t m_{x,t} · p^l_{x,t,e}                 differentiable
h_e(x; θ) = (1/(L·T_x)) Σ_l Σ_t m_{x,t} · 1{e ∈ S^l_{x,t}}             piecewise constant (∇ = 0 a.e.)

ℓ_x(θ; f̃_t) = CE_x(θ) + α · E · Σ_e ( f̃_{t,e} − k/E ) · P_e(x; θ)      (+ ⟨z, s·h(x)⟩, value 0 — the release carrier, §2)
```

- `CE_x` = HF's per-example token-mean causal-LM CE over label-valid tokens, exactly what
  `DPTrainer.compute_per_example_loss` reads from `fmodel(params, **inputs)["loss"]`
  (`packages/opaque-transformers/src/opaque/api/transformers/trainer/_dp_trainer.py:2314-2434`, VERIFIED) through
  the chunked LM-head path (`mellum.py:44`, `chunked_linear_cross_entropy=2048`).
- Centred form: `Σ_e P_e(x) = 1` ⇒ `Σ_e (k/E) ∇P_e = 0`, so the centred and HF-uncentred surrogates have
  **identical gradients** (math §1.1/F3, VERIFIED to 1e-17); the centred form makes "the signal is the imbalance"
  explicit and gives `α·E·Σ_e f̃_e P_e = α·k` + centred at balance.
- Exactness (VERIFIED, math §1.2, empirical E4 rel-err 2–3e-7): with `f̃_t = f(B_t)` and equal lengths,
  `(1/B) Σ_x ∇ℓ_x = ∇[ CE_tokenmean(B_t) + α · L_aux^HF(B_t) ]`, `L_aux^HF = load_balancing_loss_func`
  (`modeling_mellum.py:540-606`).

### 1.2 Target of faithfulness and why

The four candidates (critic G1) and the skeptic's verdict:

| candidate | verdict |
|---|---|
| (a) HF logical-batch pooled formula (Fact A) | **the target of the surrogate (Regime B)** — it is the documented loss of the checkpoint's HF class and F3 makes it exactly per-example-separable. |
| (b) HF-Trainer-realised (per-microbatch f, coefficient G·α; critic Exp A, VERIFIED rel-L2 0.0) | rejected: an artefact of `trainer.py:1961-1963`; at the preset it would mean f over 8 sequences and α_eff = 0.032. The DP path is *more* faithful to (a) than HF Trainer is. |
| (c) Megatron per-layer running average (pretraining; TR §3.6, literature F.2 VERIFIED) | rejected as target (not reproducible from HF; per-layer release costs ×√L = 5.3 noise, F5). Its *lagged running-average* character is what the smoothed `f̃_t` of §2/§6 reproduces. |
| (d) per-sequence aux | rejected as a stand-in (different regulariser, cos 0.26–0.39 to the batch gradient, E4; anti-specialisation closed form, math §2). Available as an explicit, zero-cost option flag, never labelled "faithful". |

**But in Regime A the target is (a) with α = 0, and that is a faithfulness claim, not a shortcut:**

1. It is the objective the HF artefact computes by default (`MellumForCausalLM.forward` adds the aux only under
   `output_router_logits`, `modeling_mellum.py:692-700`, VERIFIED); HF `Trainer` users get it unless they opt in
   (HF issue #44242, literature C13). The TRL converters already document that the coefficient is dropped
   (`packages/opaque-transformers/src/opaque/api/transformers/trl/_convert.py:69-76`, VERIFIED).
2. The aux loss trains the **router** (Switch §2.2: only `P` is differentiable; the gradient hits router logits,
   literature C2). With the router frozen its designed lever is gone; the only remaining path is through hidden
   states into attention LoRA, where it is (toy) 2.4e-4 of the CE gradient at balance and ~1.2e-3 with induced
   imbalance (divergence §5.3, empirical E4b; both VERIFIED at toy scale, magnitude PLAUSIBLE at 12B).
3. The regulariser's gradient is **identically zero at balance** (F3), so for a checkpoint whose router is
   balanced (Mellum2 TR: SFT coefficient reduced 10× "since the router is already well-balanced", literature F.2
   VERIFIED) the term is inert until the fine-tune *creates* imbalance — which is exactly what the monitor
   detects (§2.3).
4. Precedent: Tholoniat et al. 2024 (arXiv:2402.07334, literature A.2 VERIFIED) drop the loss under DP;
   OLMoE §4.3 (arXiv:2409.02060, literature C8 VERIFIED, with the noted internal inconsistency) drop it during
   SFT/DPO and measured the balance loss *decrease*.

So Regime A is faithful to the HF default objective, to the DP-MoE literature, and to the model authors' own
observation; the residual assumption (balance persists) is turned into a measured, DP-released quantity.

### 1.3 The other G1 sub-decisions

| item | decision | reason |
|---|---|---|
| mask | attention mask (prompt tokens count) | HF passes `attention_mask` to the aux (`modeling_mellum.py:694-697`, VERIFIED); label-masked prompt tokens are still routed. |
| CE weighting | equal example weights (Opaque's convention, `opaque-alignment/.../sft/loss/_nll.py:15-24, 86-87`) | coincides with HF token weighting iff `T_x ≡ T` (F4; true for the packed T = 1024 presets). The public-`N̄` token-weighted variant (divergence §4) is representable but not built (YAGNI). |
| coefficient default | Regime A: **0**. Regime B: **1e-4** (Mellum2's own SFT value, TR §5.1.2). `1e-3` = `model.config.router_aux_loss_coef` opt-in for "HF Trainer with `output_router_logits=True`" parity. | §1.2. The trainer never silently reads the config coefficient — it is an explicit argument (`moe_router_aux_coef`). |
| `f` from executed routes? | **yes**, from the fp32-softmax top-k that ran (`modeling_mellum.py:335-336`), not HF's aux-side bf16 recomputation (critic R9: 0.03–1.7 % of tokens differ) | "the load that ran" is the only definition consistent with the executed forward and with the sensitivity bound (Σ_e h_e = k exactly). |
| per-layer vs pooled | pooled `E`-vector (HF) | per-layer `L×E` release costs ×√L noise (F5) and is not what HF computes. Per-layer diagnostic only in the validation script (§10), never released in training. |
| z-loss | out of "faithful to HF" (HF does not implement it for Mellum); opt-in separable term (§7) | — |

---

## 2. Mechanism and accountant (G4)

### 2.1 Adjacency, unit, sensitivities

Protected unit = one training example (sequence); adjacency = **add/remove** (repo default,
`opaque-engine/.../clipping/_clipped_grad.py:135-145`; `.junie/differential-privacy-review.md` "Adjacency");
replace-one doubles every sensitivity below. Divisor is the public expected batch size `B̄` (`normalize_by=expected_batch_size`,
`_dp_trainer.py:4256-4290`, VERIFIED) — never the realised Poisson/b-min-sep batch size.

Per-example released vector (Regime A `monitor`, Regime B `surrogate`; Regime A `off` = first block only):

```
v_x = [ clip_C(∇_θ ℓ_x) ;  s · h(x) ],     s = ρ·C/√k,     PerGroup bounds  {fallback: C,  router_load_probe: ρ·C}
```

- gradient group: ‖clip_C(∇ℓ_x)‖₂ ≤ C by clipping (`_pytree.py:439-477`).
- load group: ‖s·h(x)‖₂ ≤ s·√k = ρ·C **structurally** (0 ≤ h_e ≤ 1, Σ_e h_e = k ⇒ ‖h‖₂² ≤ max_e h_e · Σ_e h_e ≤ k; F5,
  math §3 VERIFIED numerically). The `PerGroup` entry is a *bound*, not a clip: the release is unbiased and
  `clip_rate` on that group is 0 by construction (critic C6).
- With `normalize_by = B̄` the stored bounds are `C/B̄` and `ρC/B̄` (`_clipped_fun.py:611-615`).

### 2.2 Per-step mechanism, DP-SGD / Poisson

Step `t`, Poisson batch `B_t` (rate `q = B̄/N`):

```
G_t  = Σ_{x∈B_t} clip_C ∇ℓ_x(θ_t; f̃_t) / B̄  +  N(0, σ_g² I),     σ_g = nm · C · √(1+ρ) / B̄
H_t  = Σ_{x∈B_t} s·h(x) / B̄                +  N(0, σ_h² I_E),    σ_h = nm · ρC · √(1+1/ρ) / B̄
```

exactly what `per_group_noise_stddev` produces for the two-group `PerGroup`
(`opaque-engine/.../noise_allocation.py:103-110`: `σ_i = nm·√(B_i·Σ_j B_j)`; VERIFIED by running it in
`b_cost_table.py`: Mahalanobis `Σ (C_i/σ_i)² · nm² = 1.000` for every ρ). Privacy: the whitened concatenation has
add/remove L2 sensitivity `√(C²/σ_g² + (ρC)²/σ_h²) = 1/nm` (math §4(b); Andrew et al. 2021 Thm 1 form,
arXiv:1905.03871, critic R6 VERIFIED; Dong–Roth–Su Thm 2.7 for the Gaussian, critic R6 VERIFIED), so the per-step
mechanism is one sensitivity-1 Gaussian at `nm`, subsampled once by the batch coin
(Feldman–Shenfeld Lemma 3.2 / Thm 3.3, exact-Gaussian path in `poisson.rs:27-29`, critic R6 VERIFIED). Dependence
of `ℓ_x` on `f̃_t` (a function of previous outputs) is adaptive composition (Zhu–Dong–Wang Thm 10, critic R6
VERIFIED) — the same dependence DP-SGD already has on `θ_t`. **Accountant call: unchanged**
`poisson(gaussian(nm), q) * T` (`_dp_trainer.py:4380-4384`).

Post-processing (public, free), producing `f̃_{t+1}`:

1. `f̂_t = H_t / s` — unbiased for `(|B_t|/B̄)·(1/|B_t|)Σ_x h(x)` (Poisson scale factor `|B_t|/B̄`; F5/critic M11).
2. **Renormalise**: `f̂_t ← f̂_t · k / Σ_e f̂_{t,e}` (removes the realised-batch scale; the denominator's noise std is
   `√E·σ_h/s` = 0.044 at ρ = 0.02 against `k = 8`, i.e. 0.55 %).
3. **Smooth**: DP-SGD → EMA `f̄_t = β f̄_{t−1} + (1−β) f̂_t`, β = 0.99 (noise variance ×(1−β)/(1+β) = 0.005;
   lag ≈ 100 steps); DP-FTRL → window mean, §6.
4. **Clamp** to `[0, 1]`, renormalise to `Σ_e = k` again, `f̃_{t+1} := result`. `f̃_1 = k/E·1` (zero aux gradient
   during the first window — by design, the term is inert until there is evidence of imbalance).
5. Regime A: additionally evaluate the decision rule (§2.3). `f̃` is held fixed inside a step.

### 2.3 The decision rule (Regime A → Regime B)

Statistic: `D_t = max_e |f̄_{t,e} − k/E| / (k/E)` from the smoothed public `f̄_t` (window mean, W = 256 steps, or
EMA β = 0.99). Rule: **trip when `D_t > τ = 0.5` (some expert carries ≥ 1.5× or ≤ 0.5× its balanced share) on
two consecutive evaluations.** With ρ = 0.02 the smoothed per-entry noise is 2.2 % (DP-SGD, W = 256) / 0.8 %
(band-MF, §6) of `k/E`, so (`b_cost_table.py`, Gaussian tail arithmetic, VERIFIED):
`P(false alarm under exact balance) < 1e-14`, `P(miss | true deviation 1.0)` < 1e-15; even at ρ = 0.01, W = 64,
τ = 0.25 the false-alarm probability is 4e-3. The rule is a function of a DP output, hence free; the run log records
`D_t` as a public curve. Reference threshold for τ: the pretraining balance loss at `f ≡ k/E` is `k`; `τ = 0.5`
corresponds to a relative excess of the *un*-centred aux value of order `(k/E)·τ²·E/k` ≈ 0.25·… — chosen for
detectability, not from a utility model; validation (§10) tunes it.

On trip, three admissible responses, in order of simplicity: (i) log and continue (the operator judges); (ii) switch
`α` from 0 to 1e-4 — the surrogate — **without touching clipping, noise or the accountant** (the leaf, its bound and
ρ are already in place; changing the loss is an adaptive-row change, Denisov Thm 2.1 / ZDW Thm 10); (iii) stop.
Default: (i) for `monitor`, (ii) for `monitor_then_surrogate`.

Public-proxy alternative (Ponomareva §5(b) option (a) / Davody, literature B2/B3 VERIFIED): if a public sample of the
fine-tuning distribution exists, evaluate `f` on it with the current model every W steps at zero DP cost and apply
the same rule. Offered as an option; the DP monitor is the general rule.

### 2.4 Per-step mechanism, DP-FTRL / band-MF / b-min-sep

Row `t` of the stream is the same two-group vector `[G̃_t ; H̃_t]` (before noise), per-group per-row bound
`(C/B̄, ρC/B̄)` **constant for the whole run** — required by `_validate_constant_max_norm`
(`opaque-dpftrl/.../noise/_engine.py:473-517`, VERIFIED; a `PerGroup` is latched by equality, primitives E2 VERIFIED
for identity and band-MF). Noise: `mf_gaussian_noise` computes the base per-group σ with the same
`per_group_noise_stddev` and applies the strategy's streaming inverse; realised per-step σ on each leaf is
`base σ · ‖row_t(C⁻¹)‖` (`_mf_gaussian_noise.py:163-192`, VERIFIED). Privacy: Denisov et al. Thm 2.1 (adaptive rows,
critic R6 VERIFIED) with per-group whitening — the participation-pattern sensitivity is homogeneous of degree 1 in
the row bound, so `Σ_g ‖C(G_g−H_g)‖_F²/σ_g² ≤ sens(C)²/nm²` (math §5(i)). The b-min-sep schedule
(`BMinSepSampler`, `dpftrl/sampling/_b_min_sep.py:30-`) is shared by both groups because they come from the same
example at the same step. **Accountant call: unchanged** `b_min_sep(mf_gaussian(nm, band_mf_strategy(64)),
n_steps, p0)` (`_dpftrl.py:152-159`); the calibrated `nm` is whatever the trainer's `calibrate` returns and is not
changed by this design. What *does* change is utility: gradient σ ×√(1+ρ). Smoothing: §6.

Rejected alternatives (simplicity): an *independently subsampled* second release (cheaper in ε, F6) needs a second
forward and a second Poisson draw and is unavailable under b-min-sep; a separate composed mechanism
`poisson(gaussian(nm)|gaussian(σ_h), q)` is the same Mahalanobis family expressed as "pay in ε" (F6) — one
parametrisation is enough, and the joint route leaves every accountant call and every test of the accounting stack
untouched.

### 2.5 Cost table — preset regime (nm = 0.5622, B̄ = 256, k = 8, E = 64, q = 256/5e5, T = 15625, δ = 1e-6)

Computed by `b_cost_table.py` (DP-SGD columns, VERIFIED; the band-MF factors at n = 15625 are read from
`scratchpad/research/design-minimal/cost_table.out` (VERIFIED by reading; window-256 factor 0.0216, EMA-0.99 factor
0.0381, `‖row_t(C⁻¹)‖ → 3.80`) — my own n = 1024 run gives 0.0157 / 0.0253 / 2.26, same direction). "ε if nm
held" is the ε of the *equivalent* separate same-batch release `poisson(gaussian(nm)|gaussian(nm√(1+1/ρ)), q)*T`
if one refused to inflate the gradient noise; in the joint route used here **ε stays 3.00** for every row.

| ρ = C_h/C | role | gradient noise × | load rel. error, 1 step | after W = 256 window (DP-SGD / band-MF) | after EMA β=.99 (DP-SGD / band-MF) | ε (joint route) | ε if nm held instead |
|---|---|---|---|---|---|---|---|
| 0 | Regime A, `off` | 1.000 | — | — | — | 3.000 | 3.000 |
| 0.01 | monitor (minimum) | 1.005 | 49.9 % | 3.1 % / 1.1 % | 3.5 % / 1.9 % | 3.000 | 3.172 |
| **0.02** | **monitor default; surrogate-capable** | **1.010** | 35.5 % | **2.2 % / 0.8 %** | 2.5 % / 1.4 % | **3.000** | 3.234 |
| 0.05 | surrogate, DP-SGD default | 1.025 | 22.8 % | 1.4 % / 0.5 % | 1.6 % / 0.9 % | 3.000 | 3.417 |
| 0.10 | surrogate, fast-tracking | 1.049 | 16.5 % | 1.0 % / 0.4 % | 1.2 % / 0.6 % | 3.000 | 3.703 |
| 0.25 | (not recommended) | 1.118 | 11.1 % | 0.7 % / 0.2 % | 0.8 % / 0.4 % | 3.000 | 4.453 |

Reading. The single-step release is useless at any affordable ρ (11–50 % error against a signal that is itself the
*deviation* from `k/E`), which is why every design that uses the load "in the same step" is wrong-headed; a lagged,
smoothed `f̃` costs nothing extra and lands at 1–3 % — the same object Megatron's running average was (literature
F.2). At ρ = 0.02 the gradient pays 1 % more noise for a load estimate good to ~2 % after 256 steps. Under band-MF
the anti-correlated noise makes the window mean ~3× *better* than under DP-SGD (§6). The per-step realised σ on the
load leaf under band-MF is 3.8× the DP-SGD value (`‖row_t(C⁻¹)‖`), so single-step readings must never be consumed.

---

## 3. Router precision and routing (G5)

**Claim tested here (VERIFIED, `a_fp32_router_flips.py`, tiny Mellum E = 64/top-8, 2 layers, 16×64 tokens, random
init):** replacing the router's bf16 `F.linear` by an fp32 GEMM (`hidden.float() @ W.float()`, softmax fp32, top-k)
does **not** materially reduce bf16-vs-fp32 route flips:

| data | layer | stock bf16 router flips / 1024 rows | fp32-logit router flips / 1024 rows |
|---|---|---|---|
| random | L0 / L1 | 46 / 54 | 51 / 56 |
| structured | L0 / L1 | 46 / 47 | 39 / 43 |

The flips come from the bf16 **hidden states** entering the router, not from rounding the logits (same conclusion as
E2: fixed relative resolution). Hence "compute routes in fp32 inside vmap" is *not* a lever against flips relative to
an fp32 reference — the only way to get fp32 routes is an fp32 forward, which the presets do not run. Further
VERIFIED in the same script: the fp32-logit router runs under `clipped_grad` (vmap(grad)) with the patched model
(finite per-example gradients, `batch_argnums=(1,2,3)`), and **vmap-vs-eager on the same bf16 model gives 0/1024
flips in both layers** — consistent with F8 (0 flips Opaque-bf16 vs HF-bf16 at toy depth).

Decisions:

1. **Stock router by default.** No new router patch in the compat set; the executed routes are the HF ones, so the
   DP path matches the bf16 *inference* router (`MellumTopKRouter.forward`, `modeling_mellum.py:332-341`) — the
   HF-faithful choice; pretraining's FP32 router (TR appendix) is a training-time detail that HF inference already
   departs from.
2. **Routes are per-example constants of `(x, θ_t)`** — no pinning pass, no frozen-base routes. The top-k is a
   deterministic function of the example and the current parameters; the per-example gradient of the resulting
   piecewise-smooth loss is what `vmap(grad)` returns and clipping bounds it (math §6: continuity is irrelevant to
   privacy). Pinning from a frozen base would create train/inference routing mismatch (math §6) for no DP gain.
3. **`f` from executed routes** (§1.3), computed inside the vmapped loss from the same `p^l_{x,t}` the forward used.
4. Opt-in `router_logits_fp32=True` (a `MellumTopKRouter.forward` replacement, 64×2304 GEMM/token ≈ 0.15 % of the
   routed-expert FLOPs) for users who want the pretraining router or a full-fp32 oracle comparison — documented as
   *not* a drift fix.
5. Route pinning from an fp32 routing-only forward remains a **Regime B option to be validated** (E1b: pinning to
   fp32 routes cut router/expert gradient error 12.9 %/11.1 % → 1.7 %/1.5 % in the toy; +0.2 pp only for attention
   params). It costs a full fp32 forward per step and is not built in v1.

**Oracle consequence (G2).** Because flips are a property of precision, not of vmap, the oracle must be
precision-matched: same patched module (same MoE path, same router forward, same dtype), HF eager batched forward,
`output_router_logits=False`. Against that oracle the expected flip count is 0 (VERIFIED at toy depth; to be
confirmed at 28 layers, §10) and the residual drift is accumulation order (F8: 0.45–0.5 % at toy depth).

---

## 4. Clipping norm and per-example gradient norms

### 4.1 Why a reasonable clipping norm exists — confronting the claim directly

The claim "batch statistics make it impossible or the clipping norm cannot be reasonable" conflates three things:

1. *Separability.* `∇_θ L_aux(B) = Σ_x ∇_θ S(x; f(B))` **exactly** (F3; math §1.2 VERIFIED to 1e-17; empirical E4
   2–3e-7 including ragged lengths). The batch statistic is a coefficient, not a coupling of gradients. The only
   inadmissible per-example loss is the token-weighted HF-match `B·w_x·(…)` whose weights contain `T_tot` (math §1.3)
   — and Opaque's equal-example-weight convention avoids it (F4).
2. *Boundedness.* Clipping bounds `∇ℓ_x` for **any** constant `f̃` and any routing decision; the top-k
   discontinuity changes *which* bounded vector is clipped, never the bound (math §6). The load vector is bounded
   without clipping (‖h‖₂ ≤ √k). Nothing about MoE enlarges the per-example sensitivity.
3. *Reasonableness of C.* The per-example CE gradient-norm distribution for attention-only parameters is tight
   (max/median 1.24–1.33 at toy scale, E3 VERIFIED); the aux term at α = 1e-3 shifts norms by < 0.1 % (E3); route
   flips add ~0.2 pp to attention-only gradient error (E1b). So C is chosen exactly as for any dense LoRA fine-tune.
   The preset C = 0.9 (`examples/train_dpftrl.py:611-612`) is a fixed clip; the load leaf's bound `ρC` is a
   *separate* group and does not shrink the gradient's admissible norm (unlike joint single-vector clipping, math
   §4(a), which is why that variant is rejected).

Residual, real risks (not impossibilities): (a) the per-example norm distribution of the *trained* checkpoint under
LoRA r = 16 is unmeasured (G3) — C must be set from data; (b) in Regime B, expert gradients are sparse across
experts for real documents (E5 caveat, critic R8: open) and would dominate a single-group norm — handled by
per-group clipping below; (c) AUTO-S must not be applied to the load group (it rescales every load vector to the
bound, a different estimator; primitives §1.4).

### 4.2 What the design does

- **Partition (Regime A):** one gradient group (`fallback`: all LoRA leaves) + the load group. Fixed clipping
  (preset default). AUTO-S on the gradient group would be admissible under MF (constant bound) but the current
  engine applies AUTO-S to *every* group of a `PerGroup` (`_auto.py:203-364`, `_pytree.py:314-347`); v1 therefore
  **requires `clipping_mode="fixed"` whenever the load leaf is enabled** (`ConfigurationError` otherwise). Adaptive
  clipping is excluded under MF anyway (`_engine.py:473-517`).
- **Partition (Regime B):** `per_group(trainable, attention=C_a, router=C_r, experts=C_e, router_load_probe=ρ·C_a)`
  (`opaque-engine/.../clipping/_per_group.py:44-`), bounds at each group's per-example median from the calibration
  pass; optimal allocation inflates each group's σ by `√(S/C_g)`, `S = Σ_g C_g` — keep the number of groups ≤ 4.
- **How to pick C:** one `clipped_grad(..., clipping_norm=∞, return_aux=True)` pass over ~256 examples (§10
  script), read `aux.grad_norms` (and `aux.group_norms`), set `C` (or `C_g`) at the **median** (clip rate ≈ 0.5,
  the same target the trainer's adaptive mode defaults to, `_dp_trainer.py:4252` `target_clipping_rate=0.5`).
  This pass is on training data and is itself a private query if its output is used to set C; do it on a public
  proxy sample, or account it as one Gaussian release of the norm quantile (`adaptive_clipped_grad` already
  provides the accounted version, `_adaclip.py:131`) — under MF prefer the public proxy.
- **Load leaf scale:** `s = ρC/√k` with ρ = 0.02 (monitor) or 0.05 (surrogate under DP-SGD); the group bound is
  exactly `ρC`, so `clip_rate` on it is 0 (assert in tests).

---

## 5. Performance (G6)

### 5.1 Dense-MoE default

VERIFIED (critic R3; `_training_arguments.py:436`, `_dp_trainer.py:846`, `_factory.py:319`, `kernels/moe.py:602-633`):
`DPTrainer` defaults `use_performance_kernels=False` ⇒ `grouped_moe=False` ⇒ dense `Opaque_MoE` on every host,
and the docstring at `_training_arguments.py:429-435` does not mention MoE. Neither preset sets `grouped_moe`
(grep over `examples/train_dpftrl.py`, `examples/train_dpo.py`: no hits, VERIFIED).

FLOP arithmetic from the public config (H = 2304, I = 896, E = 64, k = 8, L = 28, 32 heads/128, 4 KV heads, vocab
98304; VERIFIED arithmetic, timing PLAUSIBLE): one expert = `H·2I + I·H` = 6.19 M MAC/token; routed (k = 8) =
49.5 M MAC/token/layer → 2.77 GFLOP/token forward over 28 layers; dense (E = 64) = 22.2 GFLOP/token. Attention ≈
1.7 GFLOP/token (T = 1024), LM head 0.45. So the dense default is ≈ **5× the forward FLOPs** of the routed model
(24.4 vs 4.9 GFLOP/token) and, with frozen experts, a similar ratio in the backward (only `dx` is needed through
the experts; `dW` is skipped, `moe.py:536-539`). Memory is bounded (every route chunks temporaries to the internal
budget, `moe.py:596-598`) but the dense intermediate is `(T, E, 2I)` per chunk.

Decision: **decouple `grouped_moe` from `use_performance_kernels`.** In `_factory.py:319` resolve
`grouped_moe = kwargs.get("grouped_moe", kernels or _grouped_route_available())`, where the helper is true on CUDA
with Triton or wherever `torch._grouped_mm` exists; `opaque_moe` already gates the grouped route on a memory
estimate and falls back to dense (`moe.py:614-632`). Keep the class-level first-patch capture (critic R3) but log
the chosen path once. Document in the `use_performance_kernels` docstring that MoE routing is a compat patch whose
*path* is chosen by `grouped_moe`. Precision: both grouped paths are "numerically equivalent within the documented
dtype floor" (`moe.py:610-612`); the Triton fused path's accumulation dtype is not verified here (PLAUSIBLE) —
the §10 flip counter is the acceptance test.

### 5.2 New buffers (monitor/surrogate only)

| buffer | shape / size | where | lifetime |
|---|---|---|---|
| router probabilities captured per example | `(L, T, E)` fp32 = 28·1024·64·4 B = 7.3 MB/example; ×8 microbatch = 59 MB (+ the same for the executed indices, int64 `(L,T,k)` = 1.8 MB/example) | inside the vmapped loss | one microbatch chunk |
| `P_x`, `h_x` | 2 × 64 floats/example | inside vmap | transient |
| probe parameter `router_load_probe` | 64 floats (always 0) | model / trainable dict | run |
| window buffer (band-MF) or EMA (DP-SGD) | `W×E` = 256·64 fp32 = 64 KB, or 64 floats | trainer state | run, checkpointed |
| `f̃_t` broadcast input | `E` floats (or `(B,E)` column) | `_augment_inputs` | step |

Compute: `softmax` over `(L,T,E)` per example is already done in the forward; the capture adds two reductions.
Negligible against the expert GEMMs. With `router_logits_fp32=True`: +0.15 % FLOPs.

---

## 6. Smoothing under MF (G7)

The MF stream releases `x̂_t = x_t + σ_base·(C⁻¹Z)_t` on the load leaf (`_mf_gaussian_noise.py:186-188`). Any
linear filter `F` of the stream is post-processing (free). The strategy is optimised for the workload
`A = prefix sums` (momentum 1.0 is the preset default, `band_mf_strategy(bands=64)`, `_band_mf.py:142-171`), so
the filter whose noise the strategy *bounds* is a prefix sum; a **window mean** `(1/W)(S_t − S_{t−W})` is the
difference of two prefix-sum rows and inherits `error ≤ 2·max_t ‖row_t(AC⁻¹)‖σ/W`. An EMA is not workload-matched
(still valid, just not what the strategy optimised). Numbers (`b_cost_table.py`, n = 1024, VERIFIED; n = 15625 from
`design-minimal/cost_table.out`, VERIFIED by reading):

| filter | noise factor, band-MF bands=64 (n=1024 / n=15625) | iid (DP-SGD) factor | ratio MF/iid |
|---|---|---|---|
| single step | 2.26 / 3.80 (`‖row_t(C⁻¹)‖`) | 1 | 2.3–3.8× worse |
| window mean W = 64 | 0.0496 / 0.0666 | 0.125 | 0.4–0.5× |
| **window mean W = 256** | **0.0157 / 0.0216** | 0.0625 | **0.25–0.35×** |
| EMA β = 0.95 | 0.116 / 0.185 | 0.160 | 0.7–1.2× |
| EMA β = 0.99 | 0.0253 / 0.0381 | 0.0709 | 0.36–0.54× |

Decision: under DP-FTRL use the **window mean with W = 4·bands = 256** (lag 128 steps = 0.8 % of the 15625-step
preset; ≤ 4 participations of any example inside a window under b-min-sep — no privacy relevance, the schedule
is unchanged). The window buffer is public state (§8). Under DP-SGD use EMA β = 0.99 (equivalent lag). Both are
exposed as `moe_router_load_filter={"window": W} | {"ema": β}` with the stack-specific default.

---

## 7. Scope (G8)

| item | v1 status | what it needs |
|---|---|---|
| DP-SFT presets (`mellum2-kstack`), attention-only LoRA | **in**: Regime A, `monitor` recommended | nothing beyond §9 |
| DP-DPO preset (`mellum2-codesec`, `examples/train_dpo.py:1300-1306`, attention-only LoRA) | **in**: Regime A with α = 0. Consistent with TRL, which never added the Mixtral-style aux to DPO/KTO (literature C13 VERIFIED); the reference forward contributes no gradient and no load release. Monitor over the *policy* forwards of chosen + rejected (h_x pooled over both sequences of the pair = one protected unit). | monitor only |
| DP-DPO Regime B (surrogate over chosen + rejected) | **out** (v1); design is the same surrogate with `P` pooled over both policy sequences | validation of §10 on DPO first |
| router z-loss (Mellum2 pretraining 1e-3; per-token, separable) | **opt-in** `moe_router_z_coef` (default 0): `ζ·(1/(L T_x)) Σ_{l,t} m_{x,t}·(logsumexp_e z^l_{x,t,e})²` added per example — zero privacy cost, uses the captured logits | only meaningful when the router is trainable |
| router-trainable (64×2304 per layer, 4.1 M params) | **in as Regime B** (`surrogate` default, per-group `router` clip group) | §10 validation of the aux/noise ratio on router params |
| experts-trainable via PEFT `target_parameters` on stacked `gate_up_proj`/`down_proj` | **out** until critic M5 is executed (untested under `functional_call`+vmap). Note dense per-example expert grads are `(B,E,2I,H)` bf16 = 8·64·1792·2304·2 B ≈ 4.2 GB per layer per microbatch of 8 — full-expert training is out regardless; shared-A MoE-LoRA is the lever | a 10-line smoke test, then memory design |
| per-sequence aux (option (d)) | **opt-in flag** `moe_router_aux="per_sequence"`, zero privacy cost, documented as a different regulariser | — |
| loss-free balancing (expert bias from previous-batch counts, DeepSeek-V3 / Wang et al.) | **out**; would reuse the same released `f̂` (sign of deviation) but changes the model's routing function; the checkpoint has no bias tensor | — |
| `output_router_logits=True` passed by users | **rejected with a clear error** under DP: it fails under vmap with a mask (F7) and bypasses chunked CE | — |

---

## 8. Privacy hygiene (G9)

Rule: everything computed from private examples inside the grad transform is private-internal until it has passed
clip → noise; only noised aggregates and their post-processing are public. Adjacency/unit as in §2.1.

| tensor / state | class | logged? | checkpointed? | DDP-synced? | notes |
|---|---|---|---|---|---|
| router logits/probs `p^l_{x,t}`, indices `S^l_{x,t}` (inside vmap) | private-internal | never | never | no | consumed only in `ℓ_x`; not placed in `loss_aux` |
| `P_x`, `h_x` (per example) | private-internal | never | never | no | **must not** be added to `compute_per_example_loss_and_metrics` telemetry (`_dp_trainer.py:2258-2272` logs `loss_aux` un-noised) |
| `aux.group_norms["router_load_probe"]` (per-example ‖s·h_x‖) | private-internal | **suppressed**: the trainer logs per-group `grad_norm` means un-noised (`_dp_trainer.py:2261-2272`); the probe group is skipped in `group_metrics` | no | gathered by `sync(aux)` — remains in-process only |
| probe parameter `router_load_probe` (value 0) | public/inert | no | as zeros (or omitted) | trivially | zeroed after every optimizer step; excluded from weight decay/momentum by masking or zeroing |
| noised probe leaf = `H_t` → `f̂_t` | **public (DP output)** | yes | no (derived) | identical on all ranks by construction (shared noise key, `_dp_trainer.py:1478`) | the *only* release added by this design |
| window buffer / EMA state, `f̃_t`, `D_t`, trip flag | public post-processing | yes | **yes** (`RuntimeCheckpoint` field `moe_load_state`, `_checkpoint.py:201-208`) | equal on all ranks; `register_sync_type` handler asserts equality | reproducibility, not privacy |
| pinned routes (if ever enabled) | private-internal | never | never | no | — |
| un-noised `loss`, `grad_norm`, `clip_rate` means (pre-existing, `_dp_trainer.py:2249-2255`) | private, unaccounted | yes (pre-existing) | — | — | outside this task; flagged per F11 |

Privacy statement addition (docs `dp-sgd.md` / `dp-ftrl.md` Mellum section): "When `moe_router_load` is `monitor`
or `surrogate`, each step releases, in addition to the clipped gradient sum, the per-example router-load fraction
vector `h_x ∈ [0,1]^64` (Σ = 8, ‖h_x‖₂ ≤ √8) as a second per-group leaf of the same Gaussian / matrix mechanism with
bound ρ·C. The accountant is unchanged; the gradient noise is inflated by √(1+ρ)."

---

## 9. Implementation plan in Opaque (G10)

Engine and accounting packages: **no changes**. dpftrl: **no changes** (the latch accepts a constant `PerGroup`,
primitives E2). The work is in `opaque-patches` (one component + one factory default) and `opaque-transformers`
(trainer arguments, one helper module, checkpoint field, TRL message).

### 9.1 `opaque-patches`

1. `transformers/_factory.py:319` — `grouped_moe` default decoupled from `kernels` (§5.1); helper
   `_grouped_route_available()` next to `kernels/moe.py:575-633`; docstring update at
   `opaque-transformers/.../_training_arguments.py:429-435`.
2. New `transformers/components/router_stats.py`:
   - `install_router_capture(model) -> RouterCapture` — registers forward hooks on every module of class
     `MellumTopKRouter` (resolved by name through the family registry so Qwen3-MoE-style routers can reuse it),
     storing `(probs_fp32, indices)` per layer index into a dict (**overwrite by layer index**, so a gradient-
     checkpointing recompute cannot duplicate entries); `capture.reset()` at the start of every per-example call;
     `capture.stack()` returns `(L,T,E)` / `(L,T,k)`. F.one_hot is not vmap-safe; use the broadcast compare
     `(indices[..., None] == arange(E)).sum(...)` (F7).
   - `per_example_load_and_prob(probs, indices, attention_mask) -> (h_x, P_x)` — the formulas of §1.1.
   - `fp32_router_forward` factory behind `router_logits_fp32=True` (opt-in, §3).
3. `transformers/models/mellum.py` — register the capture hook installer and the optional fp32 router under the
   family's `classes` (adds `"router": "MellumTopKRouter"`); default behaviour unchanged.

### 9.2 `opaque-transformers`

1. `_training_arguments.py` — new fields: `moe_router_load: Literal["off","monitor","monitor_then_surrogate","surrogate"] = "off"`,
   `moe_router_load_fraction: float = 0.02` (ρ), `moe_router_aux_coef: float = 1e-4` (used only in surrogate mode),
   `moe_router_aux: Literal["batch","per_sequence"] = "batch"`, `moe_router_load_filter: dict | None = None`
   (stack default §6), `moe_router_load_trip: float = 0.5`, `moe_router_z_coef: float = 0.0`,
   `router_logits_fp32: bool = False`. Validation: any non-`off` mode requires `clipping_mode == "fixed"` (§4.2)
   and a MoE family with a registered router class.
2. New `trainer/_moe_load.py`: frozen dataclass `MoeLoadState(f_tilde: Tensor, buffer: Tensor|None, ema: Tensor|None,
   step: int, tripped: bool, d_history: Tensor)` (serialisable via the registry, primitives §4.4);
   `update(state, noised_leaf, s, k, filter) -> MoeLoadState` implementing §2.2 steps 1–5 and §2.3.
3. `DPTrainer` seams (all existing):
   - construction: if enabled, `model.register_parameter("router_load_probe", nn.Parameter(zeros(E)))` **before**
     `make_functional(partition_trainable=True)` (`_dp_trainer.py:1357-1361`) so it lands in `trainable_params`;
     add the group to the resolved `PerGroup` (`clipping_norm` dict → `PerGroup`, `_training_arguments.py:288,
     1044`; `per_group(trainable, router_load_probe=ρ·C, fallback=C)`); seed a batch column `router_load_target`
     `(E,)` so `_discover_batch_keys` (`_dp_trainer.py:3651-3690`) includes it (TR-DPO pattern).
   - `_augment_inputs` (`:2295-2312`): write `f̃_t` into `router_load_target` (broadcast); zero the probe parameter
     in `ctx.trainable_params` (this is the "reset each step").
   - `compute_per_example_loss` (`:2314-2434`): `capture.reset()`; `out = fmodel(params, **inputs)`; compute
     `h_x, P_x`; return `out["loss"] + α·E·⟨f̃ − k/E, P_x⟩ [+ ζ·zloss_x] + ⟨params["router_load_probe"], s·h_x.detach()⟩`.
     The chunked-CE path is untouched because `output_router_logits` stays `False` (VERIFIED pattern, primitives E3).
   - after noise (`:2183`) and before the optimizer (`:2197-2218`): read `noisy_grads.pytree["router_load_probe"]`
     (already `/B̄` and noised; rank-identical), call `MoeLoadState.update`; in surrogate/trip modes set the α used at
     the next step. Implemented as an internal step, not a user callback, so `on_pre_optimizer_step` keeps its
     signature.
   - telemetry (`:2258-2272`): skip the probe group in `group_metrics`; log `moe/D_t`, `moe/f_tilde_minmax`,
     `moe/tripped` (public).
   - checkpoint: add `moe_load_state` to `RuntimeCheckpoint` (`_checkpoint.py:201-208`: "adding a new field is a
     single edit"); restore in `_apply_runtime_state` (`:5335-5356`); `register_sync_type(MoeLoadState, assert-equal)`.
4. `trl/_convert.py:69-76` — message: the coefficient is dropped unless `moe_router_load="surrogate"`, in which
   case `moe_router_aux_coef` is honoured.
5. Presets: `examples/train_dpftrl.py` `mellum2-kstack` and `examples/train_dpo.py` `mellum2-codesec` add
   `--moe-router-load monitor` (ρ = 0.02) and `--grouped-moe` (or rely on the new default).

### 9.3 Composition with the existing machinery (G10)

| feature | interaction | status |
|---|---|---|
| chunked CE (`cross_entropy.py:212-234` fallback only on `output_router_logits`) | untouched: capture runs via hooks, the causal-LM patch never sees `output_router_logits` | VERIFIED pattern (primitives E3) |
| gradient checkpointing (`_dp_trainer.py:1336-1341`, non-reentrant) | hooks fire again on recompute during backward; the loss is already formed from the first-forward captures (graph nodes), recompute writes are ignored (overwrite-by-layer + reset-at-forward) | PLAUSIBLE — test T4 |
| `microbatch_size` chunks (`_clipped_fun.py:273-274`) | each vmap chunk is one call → one reset/capture cycle; captured tensors are batched `(B_chunk,T,E)` inside vmap | VERIFIED pattern (E3); test T4 |
| DDP (`sum_gradients_`, `gradients.py:150-200`) | probe leaf is all-reduced like every leaf; shared noise key ⇒ identical `f̂_t` on all ranks; `MoeLoadState` needs no reduction, only an equality assert | VERIFIED (read) |
| `torch.compile` of the transform (`_dp_trainer.py:4200-4228`) | the Python-side capture dict graph-breaks; fullgraph fallback handles it; a later refactor can return `(loss, probs)` from the patched decoder forward instead of hooks | PLAUSIBLE — test T6 |
| AUTO-S / adaptive clipping | rejected with the load leaf (§4.2) | — |
| second-moment streams | not enabled with the load leaf in v1 (`paired_noise_stddevs` would charge the squared load, primitives §8(b)) | — |

### 9.4 Test plan

- **T1 (opaque-patches, CPU):** `per_example_load_and_prob` on a tiny Mellum equals HF's `load_balancing_loss_func`
  statistics for a batch when token-weighted (`Σ_x w_x h_x = f(B)`, `Σ_x w_x P_x = P(B)`), and under vmap equals the
  eager per-example values; `Σ_e h_x = k`, `‖h_x‖₂ ≤ √k`, `clip_rate(probe) == 0`.
- **T2 (opaque-transformers, CPU):** surrogate gradient identity — with `f̃ = f(B)` (test-only injection) the mean
  of per-example gradients equals the HF batched `CE + α·aux` gradient to fp32 round-off (E4 replica); with
  `α = 0` the run is bit-identical to today's DP path (regression).
- **T3:** accountant unchanged — `DPTrainer._build_mechanism` returns the same process object with and without the
  load leaf; `per_group_noise_stddev` satisfies the Mahalanobis identity for the two-group `PerGroup`.
- **T4:** capture exactness under `gradient_checkpointing_enable()` and `microbatch_size=2` with B=4 (G10).
- **T5 (dpftrl):** `mf_gaussian_noise` latch accepts the two-group `PerGroup` across steps; the released
  `f̂_t` window mean has the predicted noise factor (Monte Carlo over the noise key, tolerance 10 %).
- **T6:** `torch_compile=True` smoke on the transform with the capture (fallback path).
- **T7:** checkpoint round-trip of `MoeLoadState`; DDP two-rank Gloo test (`distributed` marker) asserting
  `f̃` equality.
- **T8:** decision rule unit test on synthetic `f̂` streams (false-alarm / detection at ρ = 0.02, W = 256).
- Marker discipline per AGENTS.md; no tests that pin docstring prose.

---

## 10. Validation plan on the real checkpoint (GPU) — G2 / G3

Script `examples/validate_mellum_dp.py` (one GPU, ≤ 30 min; bf16 weights as in the presets; `JetBrains/KStack`,
T = 1024, right-padded, `attention_mask` all ones for packed data).

**Oracle definition (G2).** Same process, same patched module (`apply_model_patches` with the run's `grouped_moe`,
`router_logits_fp32` and dtype), HF eager **batched** forward on one microbatch (B = 8) with
`output_router_logits=False`, `loss.backward()` per example in a Python loop (the "HF loop") and once batched
(the "HF batched"). Compare with `clipped_grad(..., clipping_norm=∞, return_aux=True)` per-example gradients and
their mean. Report separately: (i) `rel-L2` per parameter group (attention LoRA; router; experts if trainable);
(ii) **route-flip counter** per layer = number of `(x,t)` whose executed top-8 set differs between the two paths
(from hooks on `MellumTopKRouter`, fp32 softmax); (iii) fraction of tokens with an exact bf16 tie at the k/k+1
boundary; (iv) the same three numbers for HF-batched vs HF-loop (upstream's own spread) and for bf16 vs fp32 (only
if memory allows an fp32 forward; otherwise skip). This reproduces PR #980's figure with a defined reference.

**Acceptance (requirement (d)).** vmap-vs-eager flips = 0 at equal precision (if not, bisect: SDPA kernel choice via
`all_valid_attention`, MoE path, RMSNorm); `rel-L2(vmap, HF loop) ≤ 2 × rel-L2(HF batched, HF loop)`; residual
drift attributable to accumulation order only (flip-free examples show the same drift as flip examples).

**Statistics (G3), all on ≥ 256 examples of a public proxy split (KStack is public) or, if on private data, treated
as private and not reported outside the lab:**

1. per-example gradient-norm quantiles (p5/p50/p95/max) under the preset partition → set C (§4.2);
2. `‖f(B) − k/E‖_∞` and `‖·‖₂/(k/E)` for B = 256 batches, plus its batch-to-batch std → decides whether the
   monitor can ever trip and calibrates τ;
3. per-example expert usage at T = 1024 (fraction of experts with zero tokens per example, per layer) → H5;
4. `‖Σ_x ∇_θ(α·aux_x)‖ / ‖Σ_x ∇_θ CE_x‖` on attention LoRA and, with the router unfrozen, on router weights → H4;
   compare against the per-step DP noise norm `nm·C·√d/B̄` (d = 8.26 M LoRA params at r = 16 → 5.7 at nm = 0.5622,
   C = 0.9) and against the *projected* noise `nm·C/B̄` in the aux direction (1.98e-3);
5. the surrogate's tracking error: `‖f̃_t − f(B_t)‖ / ‖f(B_t) − k/E‖` along a 512-step dry run of the monitor
   (noise from the mechanism, but f from the run's own batches — a lab-only diagnostic);
6. dense vs grouped MoE: step time and peak memory for one microbatch; flip counter dense-vs-grouped.

**Drift metric definition.** `drift = rel-L2(g_vmap, g_ref)`, reported with the flip counter and the tie fraction
side by side; never a single number. Route flips are counted on executed sets, not on HF's bf16 aux recomputation.

---

## 11. Risks, and what would falsify the design

1. **Balance assumption fails silently in Regime A `off`.** Mitigation: `monitor` is the recommended preset mode;
   falsifier: `D_t` on the real checkpoint rises above τ during an attention-only LoRA run — then Regime A `off` is
   the wrong default and `monitor_then_surrogate` becomes the default. (Statistic 2/5 in §10.)
2. **The surrogate is inert under DP noise even when needed.** Falsifier: with the router trainable, the ratio in
   §10 item 4 is below the projected noise `nm·C/B̄` per step and the trajectory of `D_t` with α = 1e-4 does not
   differ from α = 0 — then the honest recommendation is Tholoniat's (drop the aux, freeze the router) and the
   surrogate is documentation-only. The design is still correct; its utility claim would be false.
3. **Flip-free vmap does not hold at 28 layers** (toy depth only). Falsifier: §10 flip counter > 0 vmap-vs-eager at
   equal precision. Then the drift is a kernel-selection or accumulation issue to bisect (SDPA path, MoE path), not
   a DP-design issue; pinning routes would mask rather than fix it.
4. **Grouped-MoE default changes numerics** beyond the documented floor (Triton accumulation dtype PLAUSIBLE).
   Falsifier: dense-vs-grouped flip counter or rel-L2 above the bf16 floor on the real model → keep dense default,
   fix the kernel.
5. **`‖f(B) − k/E‖` on real code is far below the monitor's resolution** (0.8–2 %). Then the monitor is
   uninformative but harmless (1 % noise). Falsifier: statistic 2 at ≈ 0 with tiny batch-to-batch variance —
   drop the monitor from presets (ρ = 0).
6. **Gradient-checkpointing recompute or `torch.compile` corrupts the capture** (T4/T6). Falsifier: T4 inequality.
   Fallback: return router probs from the patched decoder forward instead of hooks.
7. **Per-example expert usage is sparse and Regime B expert training is wanted** (H5 open). The design offers no
   expert-training path in v1; M5 must be executed before any claim.
8. **Telemetry leakage**: any future PR that adds `h_x`/`P_x` or the probe group's norms to `loss_aux`/`group_metrics`
   silently releases un-noised statistics. Guard: T1 asserts the probe group is absent from `group_metrics` and a
   review-checklist line in the DP protocol's "Composition and accounting".
9. **Adjacency/normalisation mismatch under b-min-sep**: the design assumes `normalize_by=expected_batch_size`
   equals the per-step expected batch of the sampler (critic M11). Falsifier: a unit test comparing
   `BMinSepSampler`'s per-step expectation with `ctx.expected_batch_size`; if they differ the renormalisation of §2.2
   still fixes the *scale* of `f̂` but the gradient's `normalize_by` is a pre-existing question for the trainer.
10. **AUTO-S users** lose the monitor (v1 restriction). Falsifier of the restriction's necessity: an engine option
    exempting a group from AUTO-S scaling (small change in `_pytree.py:314-347`) — deferred, not blocking.

What would falsify the *core* claim (that Regime A with α = 0 is a faithful, correct DP objective for the presets):
a measured, DP-released `D_t` that rises materially during attention-only LoRA fine-tuning of the real checkpoint
**and** a measured loss/benchmark gap between α = 0 and the surrogate at equal ε. Absent both, the simplest design
stands, and the conditional path is fully specified for the day either appears.
