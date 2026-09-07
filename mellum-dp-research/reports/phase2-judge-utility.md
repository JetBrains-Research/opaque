# Phase 2 — judge-utility: faithfulness and utility review of the four Mellum2 DP designs

Judge: `judge-utility`. Lens: **faithfulness and utility** (weighted 1.5×). Repo `/home/user/opaque` @ branch
`claude/mellum-dp-representation-r6slaz`; nothing tracked modified. Independent checks run this session (CPU, < 20 s total):
`scratchpad/research/judge-utility/recompute.py` (real `per_group_noise_stddev`, real `poisson(gaussian|gaussian)` accountant,
real `band_mf_strategy` coefficients at both momenta) and a re-run of `design-skeptic/a_fp32_router_flips.py`.
Tags: **VERIFIED** = I ran/read it; **PLAUSIBLE** = derived/read, not executed. Repo paths relative to `/home/user/opaque`.

Scoring rule used for the total: `(dp + 1.5·faith + 1.5·util + impl + compl) / 6`.

---

## 0. Verdict in one paragraph

All four designs converge on the same mechanism family (the released-load surrogate carried as a second `PerGroup` leaf of the
same clipped pytree; accountant literally unchanged under both stacks). They differ in *defaults* and *post-processing*, and
that is where faithfulness and utility are decided. **`optimal` wins** (8.42): it targets the same objective as `faithful`
(HF logical-batch Fact A, executed routes, attention mask, pooled E-vector) but its post-processing is strictly better for
the same privacy price — EMA β chosen by lag tolerance (0.0249 vs 0.0824 noise factor under the preset's band-MF, VERIFIED),
positive-part James–Stein shrinkage so that a noise-dominated release degrades to *exactly* what the real objective does at
balance instead of injecting a random regularisation direction, ρ = 0.02 (×1.010 gradient noise), and the only genuinely
cheaper Poisson lever (independent forward-only draw) correctly identified and correctly barred under b-min-sep.
`faithful` is a close second (8.17) and its α-default and DPO/per-layer handling must be grafted. `minimal` (7.67) has
the most implementable plan and one important honest caveat (nm_MF ≠ 0.5622) but its MF-filter analysis was computed
for the wrong strategy. `skeptic` (6.83) contributes the single most important *correction* of the phase — the fp32-logit
router does not remove route flips against an fp32 reference (VERIFIED by re-run) — and a sound monitor rule, but its
α = 0 default for the presets is a faithfulness retreat the brief explicitly declined.

---

## 1. Numbers I recomputed (and what they settle)

### 1.1 Cost table rows (preset regime `nm = 0.5622, B̄ = 256, k = 8, E = 64, C_g = 0.9, q = 5.12e-4, T = 15625, δ = 1e-6`)

Run with the *real* `per_group_noise_stddev` (`packages/opaque-engine/src/opaque/api/engine/noise_allocation.py:103-110`)
and the real `opaque.dpsgd.accounting` (`poisson(gaussian(nm) | gaussian(c·nm), q) * T`, `epsilon_at(1e-6)`):

| ρ | grad-noise inflation (real allocator) | Mahalanobis `Σ(C_i/σ_i)²·nm²` | r₁ centred (Δ_h = 2.6458) | r₁ uncentred (√k) | ε if nm held | matches |
|---|---|---|---|---|---|---|
| 0.02 | ×1.0100 | 1.000000 | 33.2 % | 35.5 % | 3.234 | optimal ✓, skeptic ✓ (uncentred), faithful ✓ |
| 0.05 | ×1.0247 | 1.000000 | 21.3 % | 22.8 % | 3.417 | faithful ✓, minimal ✓ (uncentred), optimal ✓ |
| 0.10 | ×1.0488 | 1.000000 | 15.4 % | 16.5 % | 3.703 | minimal ✓ (uncentred), faithful ✓, optimal ✓ |

**VERIFIED**: every ρ row in all four cost tables is arithmetically right (centred vs uncentred bound explains the
33.2/35.5, 21.3/22.8, 15.4/16.5 pairs). The DP-SGD filter factors √((1−β)/(1+β)) = 0.1601 / 0.0709 and 1/√W = 0.125 / 0.0625
are right in all four.

### 1.2 Band-MF filter factors — the discrepancy between the designs is a *strategy* mismatch

The preset runs `band_mf_strategy(bands=64, momentum=0.95, lr_schedule=…)`: `examples/train_dpftrl.py:495-498`
(`--momentum` default 0.95 "per BandMF paper"), `:1563-1581` (`_workload_momentum()` → `args.momentum` for SGD, `beta1`
for Adam), header comment `:53` ("default mechanism: b=64, momentum=0.95"). The *library* default is `momentum=1.0`
(`packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/_band_mf.py:111`, `:142-147`). **VERIFIED.**

My recomputation (`n = 1024`, exact `‖row_t(F·C⁻¹)‖`):

| momentum | `‖row_t(C⁻¹)‖` stationary | EMA .95 | EMA .99 | window 64 | window 256 | who used it |
|---|---|---|---|---|---|---|
| **0.95 (preset)** | **1.431** (from t ≈ 7) | **0.0824** | **0.0249** | 0.0480 | **0.0198** | faithful (1.432 / 0.0824), optimal (1.4313 / 0.0824 / 0.0249 / 0.0198) — **all reproduced exactly** |
| 1.0 (library default) | 2.26 at n=1024 (grows with n: 3.80 at n=15625 per minimal) | 0.1157 | 0.0253 | 0.0496 | 0.0157 | skeptic's own n=1024 run (2.26 / 0.0253 / 0.0157) — **reproduced exactly**; minimal's n=15625 run (3.80 / 0.185 / 0.038 / 0.0216) is the same strategy at the full horizon |

Consequences (**VERIFIED**):
- `minimal` §6 and `skeptic` §6 analysed the wrong matrix. Under the preset the single-step MF penalty is **1.43×**, not
  3.8×; EMA .95 is **2× better** than iid (0.0824 vs 0.160), not "parity"; EMA .99 is 2.8× better (0.0249 vs 0.0709).
- The *decision* "boxcar W = 256 under MF" survives by luck: at momentum 0.95 the window-256 factor (0.0198) is still the
  lowest, but only 20 % below EMA .99 (0.0249) at a comparable lag (128 vs 100 steps). `optimal`'s argument — pick the
  filter by lag tolerance, the noise of any linear filter is an exact property of `F·C⁻¹` — is the right framing;
  `faithful`'s "only the workload-matched β inherits the guarantee, so β_f := 0.95" is a true statement about the
  strategy's optimality bound that costs 3.3× load accuracy for nothing (1.76 % vs 0.53 % at ρ = 0.05).
- Under momentum 1.0 the row norm is *not* stationary (2.26 → 3.80 as n grows), so minimal's "plateau by t ≈ 512" is a
  horizon artefact of the prefix-sum workload; under 0.95 it is genuinely stationary from t ≈ 7.

### 1.3 A caveat every MF column shares

Under band-MF + b-min-sep the trainer calibrates its own `nm_MF` for ε = 3 (`_calibrate_noise`); it is **not** 0.5622
(the DP-SGD/Poisson value). `minimal` says so explicitly (its MC-PLD evaluation at n = 15625 exceeded 170 s and it quotes
MF errors as `× (nm_MF/0.5622)`); `faithful`, `optimal` and `skeptic` print MF load errors "under the DP-FTRL preset" at
nm = 0.5622 without the caveat. With 8 participations the un-amplified MF sensitivity is > 1, so `nm_MF` is plausibly
several × larger and the *absolute* MF load errors correspondingly larger; the *relative* claims (×√(1+ρ) gradient price,
2–3.5× filter advantage, accountant unchanged) are unaffected. **PLAUSIBLE, not computed** — the synthesis must carry the
caveat or run the calibration on a machine with time.

### 1.4 The fp32-router claim (skeptic's experiment, re-run this session — **VERIFIED**)

`a_fp32_router_flips.py` output reproduced bit-for-bit: stock-bf16 vs fp32-logit router flips per 1024 rows —
random L0/L1: 46/54 vs 51/56; structured: 46/47 vs 39/43; vmap-vs-eager on the same bf16 model with the fp32 router:
**0/1024 in both layers**. Reading: the bf16-vs-fp32 route flips come from bf16 *hidden states* entering the router
(fixed relative resolution, E2), not from rounding the logits; an fp32 GEMM on bf16 inputs cannot reproduce the fp32
forward's routes. Therefore:
- `faithful` §3/§4.2, `minimal` §3/§4.3, `optimal` §3/§4.1.3 **overclaim**: the E1b result (router/expert gradient error
  12.9 %/11.1 % → 1.7 %/1.5 % when *pinned to fp32-forward routes*) cannot be credited to the fp32-logit patch, and "removes
  the bf16 route-flip source of DP-vs-oracle drift" is false — the DP-vs-oracle pair at equal precision already has 0 flips
  (F8, and this re-run).
- What the fp32 router *does* buy: it is the pretraining router (TR appendix, literature F.2 VERIFIED by phase 1), it removes
  the bf16 output-rounding of the logits (so exact bf16 ties in `router_logits` no longer exist; ties are resolved at fp32
  resolution given the same bf16 input), and HF's bf16-softmax aux top-k and the executed top-k then coincide (critic R9).
  That is a **faithfulness** and **tie-robustness** argument, not a drift argument. The oracle must be precision-matched
  either way (skeptic §3, §10) — with the patch on both sides or on neither.

### 1.5 Optimal's shrinkage threshold is mislabelled by √E (**VERIFIED**)

`optimal` §2.6: "the shrinkage zeroes the term automatically when δ is below `s·√63/(k/E)` ≈ 0.07 (MF) / 0.19 (SGD)"
where δ was defined as the *per-coordinate RMS* imbalance in units of k/E. The JS rule engages when
`‖d̃‖² < (E−1)·s²`; with `‖d‖ = δ·(k/E)·√E` this is `δ < √(63/64)·s/(k/E)` = **0.0082 (MF) / 0.0233 (SGD)** — the quoted
0.066 / 0.187 are thresholds on `‖d‖/(k/E)`, not on δ. Risk §11.1 ("δ < 0.07·k/E") inherits the slip. Direction of the
error: shrinkage kicks in only when the per-coordinate imbalance is below the per-coordinate noise — the intended
behaviour — so the design is *more* conservative than its text says; the relative-error formula `0.0083/δ` in the same
paragraph is correct.

### 1.6 Other arithmetic spot-checks (all **VERIFIED**)

LoRA r = 16 q/k/v/o: 294 912 params/layer, d = 8 257 536, √d = 2873.6, per-step noise norm `nm·C·√d/B̄ = 5.68` (optimal §4.2,
skeptic §10 item 4). Expert MAC/token 6.19 M; routed ×8 = 49.5 M; dense ×64 = 396 M (8× expert FLOPs); router fp32 GEMM
64×2304 = 0.30 % of routed expert MACs (faithful/minimal ✓; **skeptic writes 0.15 % — a 2× slip**). Total-forward ratio
dense/routed ≈ 5× including attention + LM head (skeptic ✓; faithful's "≈6×" counts expert+attention only ✓).
RR per-step ε = 64·ln(0.55/0.45) = 12.8, 64·ln 3 = 70.3 (optimal ✓). Optimal's independent-draw rows (c=2, m=4: ε 3.003,
×1.0002; 4× batch: ε 5.30; amortised ρ=0.2, m=16: ε 3.194) match its `cost_table.json`. Per-layer ×5.29 columns are the
pooled columns × √28 in faithful and optimal ✓. Faithful's SNR reading (11 %/6 % at imbalance 0.3·k/E) ✓.
HF `OutputRecorder` mechanics (minimal §9.1): the collector `ContextVar` is set immediately before the backbone forward and
reset in a `finally` (`transformers/utils/output_capturing.py:266-272`), hooks return early when it is `None`
(`:104-108`) — so a gradient-checkpoint recompute during backward captures nothing. **VERIFIED by reading**; the chunked-CE
patch's `output_router_logits` fallback (`components/cross_entropy.py:212-234`) and the `router_logits` pass-through
return (`:359-370`) are as minimal describes.

---

## 2. Per-design assessment along the lens questions

### 2.1 Does the per-example objective reproduce the intended batch objective? Which one, and is the choice justified?

| | target | executed-route f | mask | CE weighting | α default | pooled/per-layer | verdict |
|---|---|---|---|---|---|---|---|
| faithful | (a) HF logical-batch Fact A | yes (fp32) | attention | equal examples (public N̄ optional) | `config.router_aux_loss_coef` = 1e-3; presets 1e-4 | pooled; per-layer opt-in priced ×5.29 | **most faithful**: the config value is what the HF artefact declares; presets pinned to Mellum2's own SFT value |
| minimal | (a) | yes (fp32) | attention | equal examples | None → config (1e-3); presets 1e-4; feature opt-in | pooled | faithful; centred *surrogate* but uncentred *release* (needs renormalisation by the noisy sum — fine) |
| optimal | (a) | yes (fp32) | attention | equal examples | **1e-4** (utility prior); 1e-3 by flag | pooled; per-layer opt-in | faithful target; α default is a utility choice — it is Mellum2's *SFT* value, defensible, but the config value should be the "faithful" default with presets overriding (graft from faithful) |
| skeptic | (a) with **α = 0** for presets (Regime A); (a) with 1e-4 in Regime B | yes (bf16 stock) | attention | equal examples | 0 / 1e-4 | pooled | Regime A is the HF *default-forward* objective, not the training objective the brief asked to be faithful to; the argument "regulariser on a frozen router is inert" is measured, not assumed — but it is still the retreat the user declined |

All four correctly reject (b) HF-Trainer-realised per-microbatch `G·α` (an accumulation artefact; the DP path has no
accumulation) and (d) per-sequence aux (a different regulariser, cos 0.26–0.39, anti-specialisation closed form). All adopt
the *lagged/running-average* character of (c) Megatron — which is exactly what pretraining used (TR §3.6) — and reject
per-layer as default (×5.29). The identity `∇L_aux(B) = Σ_x ∇S(x; f(B))` is exact (F3, 1e-17) so the only faithfulness
gap in every design is `f̃_t ≠ f(B_t)`: (i) noise, (ii) lag, (iii) smoothing replaces the batch's own sampling noise by
its expectation. (iii) is arguably *more* faithful to the regulariser's intent (Megatron's running average) than HF's
per-forward f; (ii) is 20–128 steps against a 15625-step horizon; (i) is what the cost tables price.

### 2.2 Bias / noise of f̃ at the preset regime

Unbiased in all four (structural bound, never clipped; faithful and optimal add the `(1+1e-6)` ULP guard). Per-entry noise
after smoothing, in units of k/E, at each design's default (DP-SGD / band-MF at the preset's momentum 0.95):

| design | ρ | grad noise | filter | DP-SGD | band-MF (nm = 0.5622 assumed) | lag |
|---|---|---|---|---|---|---|
| faithful | 0.05 | ×1.025 | EMA .95 (workload-matched) | 3.4 % | 1.76 % | 20 |
| minimal | 0.10 | ×1.049 | EMA .95 / window 256 | 2.6 % | 0.33 % (16.5 % × 0.0198, corrected from its 0.36 %) | 20 / 128 |
| optimal | 0.02 | ×1.010 | EMA .99 + JS shrink | 2.35 % | 0.83 % | 100 |
| skeptic | 0.02 | ×1.010 | EMA .99 / window 256 | 2.2–2.5 % | 0.70 % (35.5 % × 0.0198, corrected from 0.8 %) | 100–128 |

The error that matters is `‖noise‖/‖f(B) − k/E‖` (all four say so). Nobody has the real `‖f(B) − k/E‖` (G3). At a
per-coordinate imbalance of 0.1·k/E (a well-balanced checkpoint) the DP-SGD relative aux-gradient error is ≈ 24–35 % for
every design; under band-MF ≈ 8 % (optimal) to 18 % (faithful). Only `optimal` handles the below-noise-floor case
correctly: without shrinkage a noise-dominated `f̃` injects a *random* 64-dim regularisation direction with
`‖f̃ − k/E‖ ≈ s√64` every step — small in absolute terms at α = 1e-4 (`α·E·‖noise‖·‖∇P̄‖`) but a systematic deviation from
the objective's behaviour at balance (where its gradient is exactly zero). This is the decisive utility point in the
winner's favour; `faithful`'s "the design degrades to what the true objective does, not to a different regulariser" is
only true with the shrinkage it does not have.

### 2.3 Is the clipping-norm story credible?

Yes, in all four, and for the same reasons: the load release is its own group with a structural bound (never touches
`C_g`); the aux term at α ≤ 1e-3 moves per-example norms < 0.1 % (E3) and is zero at balance; the joint single-vector clip
(math 4(a)) is rejected because it would shrink the gradient's admissible norm and bias the histogram. Differences:
- `optimal` gives the only quantitative framing of *why* `C` is a learning-rate-like knob at this scale (noise norm 5.68 =
  6.3·C ≫ any clipped contribution), proposes the bias²+noise² curve, and defaults to AUTO-S with the load group exempt
  via a small engine change (`fixed_groups`). Switching the preset from fixed to AUTO-S is a utility gamble (Bu et al.,
  PLAUSIBLE) that also changes the preset's declared `C = 0.9` semantics — I would keep fixed as the default and ship
  `fixed_groups` for users who choose AUTO-S.
- `skeptic` alone states that the G3 `C`-selection pass on *private* data is itself a query (do it on a public proxy or
  account it via the quantile release adaptive clipping already provides) — correct and must be grafted (optimal notes it
  in §10.2 but only for the imbalance statistic).
- `minimal`/`skeptic` raise `ConfigurationError` for any non-fixed clipping mode with the load leaf; `faithful` says
  "simplest: reject". All consistent with the AUTO-S-rescales-every-load-vector caveat (primitives §1.4, critic R11).

### 2.4 Router precision / routing choice vs the checkpoint (fp32 pretraining router, bf16 HF)

- `faithful`, `minimal`, `optimal`: fp32-logit router default for the mellum family. Consistent with how the weights were
  learned (TR: router FP32, aux computations FP32); *not* consistent with the HF bf16 serving router, and the TR itself
  documents train/inference route disagreement on this checkpoint. Their stated *utility* motivation (removes flips,
  fixes the E1b tail) is **refuted** by §1.4; the surviving motivations are pretraining faithfulness and tie-robustness of
  the logits, plus removing the R9 aux-vs-executed discrepancy. Cost 0.3 % of routed expert MACs — negligible.
- `skeptic`: stock bf16 by default (HF-inference-faithful), fp32 opt-in "documented as not a drift fix", precision-matched
  oracle. This is the better-evidenced position on the *drift* question; on the *checkpoint-consistency* question it is a
  judgement call (fine-tuning under the routing that will serve vs under the routing that trained the weights). For a
  LoRA fine-tune that will be served through HF bf16, training under bf16 routes adapts the adapter to serving routes;
  for faithfulness to the objective the weights were shaped by, fp32 is right. I score this a draw with a documentation
  requirement: whichever default, the claim must be "pretraining router / serving router", never "drift fix".
- Route pinning: all four correctly reject frozen-base pinning (train/inference mismatch) and treat routes as per-example
  constants of `(x, θ_t)` — no DP consequence, continuity within a routing cell.

### 2.5 Imbalance when router / experts are trained

- `faithful`: in for the mechanism, per-group `{attention, router, experts, probe}` with MSE-optimal allocation, expects the
  router group to need the smallest bound (E3), ESFT/LoRA-on-experts as the practical form; presets out pending M5.
- `optimal`: same, plus a concrete recommendation (ρ = 0.1 or the independent draw with c = 1) because balance matters more
  there; LFB analysed as a free post-processing of the same release and correctly ruled inadmissible for training-only use
  (creates the TR §5.2 mismatch by construction), admissible as a shipped architecture extension.
- `skeptic`: Regime B (router trainable in; experts out until M5) with per-group clipping and the surrogate switched on
  mid-run without touching the accountant or the MF latch — a genuinely useful operational property (the monitor leaf *is*
  the release).
- `minimal`: out of scope (mechanism is partition-agnostic, but nothing specified).
None is tested (M5 open for all). The fp32 router matters most here (E1b) — but only if routes are actually fp32, which
per §1.4 requires more than the logit patch; the honest statement is that the router/expert gradient tail from near-ties
is a bf16-*activation* effect and is not removed by any design as written.

### 2.6 Cost-table audit summary

Every number I recomputed in `faithful` §2.6, `optimal` §2.6/§6, `minimal` §2.5 (DP-SGD columns) and `skeptic` §2.5
(DP-SGD columns) is right. The MF columns of `minimal` §6 and `skeptic` §6 are computed for `momentum = 1.0`, which is not
the preset; corrected values are in §1.2. All MF absolute errors carry the §1.3 caveat.

---

## 3. Scores (0–10; faithfulness and utility weighted 1.5×; total = weighted sum / 6)

| design | dp_correctness | faithfulness | utility | implementability | completeness | **total** |
|---|---|---|---|---|---|---|
| **optimal** | 9 | 8 | 9 | 7 | 9 | **8.42** |
| faithful | 9 | 9 | 7 | 8 | 8 | 8.17 |
| minimal | 8 | 8 | 6 | 9 | 8 | 7.67 |
| skeptic | 8 | 5 | 7 | 8 | 7 | 6.83 |

Notes per design:

- **optimal** — dp 9: in-stream mechanism sound; independent draw composed with fresh coins (ZDW Thm 10 + Feldman–Shenfeld
  Lemma 3.2); MF side-release correctly deferred pending concurrent composition; RR shown dominated; AUTO-S mixed mode keeps
  the per-record bound constant. −1 for MF numbers at nm = 0.5622 without caveat. faith 8: same target as faithful; α = 1e-4
  default is a utility prior rather than the config value; AUTO-S default changes the preset's clipping semantics; fp32-router
  overclaim; shrinkage threshold mislabelled by √E. util 9: best filter choice with exact factors, shrinkage, ρ = 0.02,
  independent-draw optimum, `C` selection by bias²+noise², grouped default, full lever ranking. impl 7: engine change
  (`fixed_groups`), independent-draw sampler + DDP sharding + `_account_independent_step` plumbing, more moving parts.
  compl 9: every G answered, DPO in, LFB, public-data option, falsifiers.
- **faithful** — dp 9: sound; ULP guard; AUTO-S exemption; hygiene; −1 for the nm_MF caveat. faith 9: config α as default,
  presets 1e-4, executed routes, DPO chosen+rejected with reference excluded, per-layer priced; −1 for the fp32-router
  overclaim. util 7: β_f := workload momentum leaves 3.3× load accuracy on the table; no shrinkage; ρ = 0.05 is fine.
  impl 8: concrete but touches the fixed `save_dp_runtime_state` signature and defers AUTO-S handling. compl 8.
- **minimal** — dp 8: sound; explicit nm_MF caveat (good); renormalisation by the noisy sum; but the MF filter analysis is
  for the wrong strategy. faith 8: Fact A, executed routes, opt-in flag keeps existing runs bit-identical; DPO out; no
  clamp on the loss path is a sensible faithfulness detail. util 6: ρ = 0.1 pays 2× the price of the others for a 25 %
  better single-step estimate; "EMA .95 ≈ parity, 3.8× per-step" wrong for the preset (boxcar decision survives by luck);
  performance arithmetic good. impl 9: engine untouched, callback on the existing seam, sidecar without signature churn,
  recorder mechanics verified, 19 placed tests. compl 8.
- **skeptic** — dp 8: correct carrier and rule; the private-query nature of the `C` calibration pass is the best hygiene
  catch of the phase; MF numbers wrong strategy. faith 5: α = 0 default for the presets is the HF *default-forward*
  objective, not the training objective; Regime B exists but is conditional; stock bf16 router is serving-faithful, not
  pretraining-faithful. util 7: monitor at ×1.010 with a well-analysed decision rule (false alarm < 1e-14 at W = 256,
  arithmetic VERIFIED); the fp32-flip correction is a real utility fact; no shrinkage; 0.15 % FLOP slip. impl 8:
  hook-capture with overwrite-by-layer, `RuntimeCheckpoint` field. compl 7: DPO Regime B out, experts out, no
  independent-draw analysis.

**Winner: `optimal`.**

---

## 4. Grafts the synthesis must keep (from non-winning designs)

1. **faithful — α default = `model.config.router_aux_loss_coef` (1e-3), presets explicitly pinned to 1e-4.** The config value
   is what the HF artefact declares; "1e-4 unless told otherwise" hides a utility prior inside a faithfulness default.
2. **faithful — DPO in scope with `P(x), d(x)` pooled over chosen + rejected policy forwards, reference forward excluded;
   TRL converter forwards the coefficient; per-layer opt-in priced at ×5.29; accounting-invariance test (T4) as the
   machine-checkable form of "accountant unchanged".**
3. **minimal — the memory-safe statistics path**: an explicit kwarg on the chunked-CE causal-LM forward that calls the
   backbone with `output_router_logits=True` (HF `OutputRecorder`, ContextVar collector active only inside the backbone
   forward, VERIFIED recompute-safe) while keeping chunked CE; `RouterLoadCallback` on the existing `on_pre_optimizer_step`
   seam; sidecar checkpoint without changing `save_dp_runtime_state`; probe group excluded from logged `group_norms`;
   `ConfigurationError` for non-fixed clipping until `fixed_groups` lands; the explicit **nm_MF ≠ 0.5622** caveat on every MF
   number; window-mean filter as a selectable alternative (best noise at momentum 0.95, 0.0198).
4. **skeptic — precision-matched oracle and the corrected fp32-router claim**: the fp32-logit router is pretraining-faithful
   and tie-robust, *not* a drift or flip fix (VERIFIED); the oracle applies the same router on both sides; the route-flip
   counter reported beside rel-L2, never a single number.
5. **skeptic — the DP-released imbalance monitor `D_t = max_e|f̃_e − k/E|/(k/E)` with a trip rule as a logged public
   curve** (free post-processing of the same release) and the observation that the surrogate can be switched on mid-run
   with no accountant or MF-latch change because the monitor leaf *is* the release.
6. **skeptic — the `C`-calibration pass on private data is itself a query**: use a public proxy or the accounted quantile
   release; under MF prefer the public proxy.
7. **skeptic / minimal — decouple `grouped_moe` from `use_performance_kernels`** with the full FLOP breakdown (dense ≈ 5×
   total forward FLOPs; 8× expert FLOPs) — all four agree, keep it.
8. **optimal itself (winner) — keep**: ρ = 0.02, EMA β = 0.99 chosen by lag tolerance with exact `F·C⁻¹` factors recomputed
   from the actual strategy at setup, sum-zero projection, positive-part James–Stein shrinkage (with the threshold restated
   per §1.5), independent forward-only draw as the DP-SGD opt-in, `fixed_groups` for mixed AUTO-S — **but keep fixed clipping
   as the preset default**, AUTO-S opt-in.

---

## 5. Errors found (design — error — correction; all VERIFIED unless marked)

1. **minimal §6** — band-MF filter table computed with `momentum = 1.0` (library default) at n = 15625 (`‖row_t(C⁻¹)‖ → 3.80`,
   EMA .95 = 0.185 "≈ parity", EMA .99 = 0.038, window-256 = 0.0216). The preset runs `momentum = 0.95`
   (`examples/train_dpftrl.py:495-498, 1563-1581`); correct values: 1.431 / 0.0824 / 0.0249 / 0.0198. The conclusion "an EMA is not
   MF-consistent, use the boxcar" is drawn from the wrong matrix (the boxcar remains marginally best, by 20 % over EMA .99).
   Its "r_MF,window256 ≈ 0.36 %" at ρ = 0.1 becomes 0.33 %.
2. **skeptic §2.5/§6** — same error (own n = 1024 run at momentum 1.0: 2.26 / 0.0157 / 0.0253, reproduced by me at momentum 1.0;
   n = 15625 numbers copied from minimal). "Single-step readings are 3.8× the DP-SGD value under band-MF" → 1.43× for the
   preset; "momentum 1.0 is the preset default" → 0.95 is. Window-256 MF error at ρ = 0.02: 0.70 % (not 0.8 %); EMA .99: 0.88 %
   (not 1.4 %).
3. **optimal §2.6, §11.1** — shrinkage-engagement threshold "δ < s√63/(k/E) ≈ 0.07 (MF) / 0.19 (SGD)" labels an L2-norm
   threshold (`‖d‖/(k/E)`) as a per-coordinate RMS threshold; in per-coordinate units it is 0.0082 / 0.0233 (off by √E = 8).
   The rule is *more* conservative than stated; the relative-error formula `0.0083/δ` next to it is correct.
4. **faithful §3/§4.2, minimal §3/§4.3, optimal §3/§4.1.3** — the fp32-logit router patch is credited with "removing bf16
   near-tie flips" and with E1b's 12.9 %/11.1 % → 1.7 %/1.5 % router/expert gradient-error reduction. E1b pinned routes to an
   fp32 *forward*; with bf16 hidden states the fp32 GEMM does not reproduce those routes (skeptic's experiment, re-run:
   flips vs fp32 reference 46→51, 54→56, 46→39, 47→43 per 1024 rows; vmap-vs-eager 0/1024). Correct claim: pretraining-faithful
   router, no exact-tie rounding in the logits, aux top-k = executed top-k; not a drift lever (drift at equal precision has
   0 flips, F8).
5. **faithful §2.6, optimal §2.6/§6, skeptic §2.5** — MF load-error columns stated "under the DP-FTRL preset" at nm = 0.5622;
   the preset calibrates its own `nm_MF` for band-MF + b-min-sep at ε = 3, which is not 0.5622 (minimal flags it; nobody
   computed it — MC-PLD > 170 s). Absolute MF errors scale with `nm_MF/0.5622`; ratio claims are unaffected. (PLAUSIBLE that
   `nm_MF` is several × larger.)
6. **skeptic §3 item 4, §5.2** — "64×2304 GEMM/token ≈ 0.15 % of the routed-expert FLOPs": it is 0.30 % (147 k MAC vs 49.5 M MAC
   per token per layer). Minor.
7. **faithful §6.1 (judgement, not arithmetic)** — "An EMA with a different β … does not inherit the guarantee" is used to fix
   `β_f := 0.95`; the noise of any linear filter of the MF stream is an exact property of `F·C⁻¹` and β = 0.99 is 3.3× more
   accurate at 5× the lag (100 steps = 0.64 % of the horizon). A utility misjudgement that costs 1.76 % → 0.53 % at ρ = 0.05.
8. **skeptic §2.3** — the rationale sentence for τ = 0.5 ("(k/E)·τ²·E/k ≈ 0.25·…") is incomplete/garbled; the false-alarm and
   miss probabilities themselves check out (22σ margin at ρ = 0.02, W = 256). Editorial.

Nothing I checked contradicts the shared DP claims (structural bound √(k(1−k/E)) = 2.6458 centred / √8 = 2.8284 uncentred;
Mahalanobis equality with the real allocator; adaptive composition for the lagged `f̃`; per-group MF via degree-1
homogeneity; latch acceptance of a constant `PerGroup`). Those rest on phase-1 VERIFIED facts (math §3–5, primitives E1–E3,
critic R6) and I did not re-derive them here beyond the numeric Mahalanobis check.

---

## 6. What the synthesis should look like (one paragraph, from this lens)

`optimal`'s mechanism and post-processing (ρ = 0.02, centred release, sum-zero projection, EMA β = 0.99 with factors
recomputed from the actual strategy, JS shrinkage with the corrected threshold, independent draw opt-in for DP-SGD, `fixed_groups`
for mixed AUTO-S) with `faithful`'s faithfulness defaults (α from config, presets 1e-4, DPO pooling, per-layer opt-in) and
`minimal`'s implementation plan (chunked-CE kwarg + HF recorder, callback seam, sidecar, probe-group telemetry suppression,
`ConfigurationError` on non-fixed clipping until `fixed_groups` lands, explicit `nm_MF` caveat), plus `skeptic`'s
precision-matched oracle, corrected fp32-router wording, public monitor curve `D_t`, and the private-query note on `C`
calibration. Fixed clipping stays the preset default. Every MF absolute number is re-derived once the preset's `nm_MF` is
calibrated; every magnitude on the trained checkpoint remains subject to the G2/G3 GPU pass all four designs specify.
