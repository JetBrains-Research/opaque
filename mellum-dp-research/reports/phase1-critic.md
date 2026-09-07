# Phase 1 — completeness critique (agent: `critic`)

Scope: I read the brief, the DP review protocol, and all five phase-1 reports in full
(`phase1-{divergence,primitives,literature,math,empirical}.md`), then re-verified the most
consequential VERIFIED claims against the code, the primary papers, and two new CPU experiments.
Scripts/outputs for this critique: `scratchpad/research/critic/` (`expA_hf_accum_aux.py`,
`expA_coef1.py` → `expA_coef1.json`; `expB_aux_topk_dtype.py` → `expB.json`; text extractions of
four papers `zdw2106.08567.txt`, `denisov2202.08312.txt`, `fs2602.17284.txt`, `drs1905.02383.txt`).

Status tags as in the other reports: **VERIFIED** (I read/ran it), **PLAUSIBLE**, **REFUTED**.
Repo paths relative to `/home/user/opaque`; HF paths under `.venv/lib/python3.11/site-packages/`.

---

## 0. Executive summary

The five reports agree on the core: H1 holds (only the load-balancing aux loss is inseparably
batch-coupled; CE normalisation is separably coupled), H2's algebra is exact (surrogate with
constant f̃ reproduces HF's aux gradient to fp round-off), and Opaque's vmap path is exact in fp32.
The phase is nevertheless **not ready for design** on five points:

1. **No number in phase 1 comes from a trained model.** Every magnitude (flip rates, gradient-norm
   spread, ||f(B) − k/E||, expert-usage sparsity, aux/CE ratio) is a random-init toy. H3 at depth,
   H4 under imbalance and H5 are therefore undecided, and the "usable clipping norm / noise level"
   requirement (c) in the brief has no empirical anchor yet.
2. **"Faithful to the real objective" has four candidate targets and the reports pick three
   different ones.** I verified by execution (Exp A) that HF `Trainer` with gradient accumulation
   realises `CE_tokenmean(logical batch) + coef · Σ_microbatches aux(mb)` exactly — i.e. f is a
   *microbatch* statistic and the coefficient is effectively multiplied by the accumulation count.
   Megatron (pretraining) used a per-layer, running-average f. HF's config pools layers into one
   product. The design cannot proceed without choosing.
3. **The "oracle drift 29% → 1.3%" figure of PR #980 is untraceable** (no tracked script mentions an
   oracle for Mellum; `grep -rn oracle` finds only kernel-level oracles), and the two experimental
   reports reach opposite H3 verdicts because they compare against different references.
4. **Two VERIFIED claims are wrong or loose**: divergence's "Σ_e f_e = K·L" (it is k; code accumulates
   `total_rows` per layer) and literature's count-vector sensitivity (loose by √k, turning a
   8.8 % relative noise into a stated 25 %). Empirical's "H5 REFUTED" rests on a 6-token-vocabulary
   construction that bounds distinct router inputs by construction.
5. **A new faithfulness nuance nobody caught (Exp B):** HF's `load_balancing_loss_func` recomputes
   top-k from a softmax in the *logits dtype* (bf16 for a bf16 model), while the forward router uses
   an fp32 softmax; 0.03–1.7 % of tokens (E=64, k=8, depending on logit scale) get a different top-k
   set in the aux than the one executed, and 4–7 % have an exact bf16 tie at the k/k+1 boundary.
   "f(B)" in HF is therefore not the executed routing.

All theorem citations the math/primitives reports relied on but had not re-fetched are now
VERIFIED from the primary PDFs (Zhu–Dong–Wang Def. 7/Thm 10/Thm 11; Feldman–Shenfeld Lemma 3.2 &
Thm 3.3; Denisov et al. Thm 2.1; Dong–Roth–Su Thm 2.7 & Cor. 3.3; Andrew et al. Thm 1).

---

## 1. Refutation attempts on consequential VERIFIED claims (what I re-checked myself)

### R1 — divergence §5.1: "Σ_e f_e = K·L … layer pooling is a sum of per-layer terms with a common denominator" — **REFUTED**

`modeling_mellum.py:575-603`: `total_rows` is *accumulated per layer* (`total_rows = total_rows +
flat_mask.sum()` inside the `for layer_gate in gate_logits` loop), so `total_rows = L·T_tot`,
`tokens_per_expert_sum` totals `L·T_tot·k`, and `Σ_e f_e = k` (math report §0 measured 2.0 for k=2;
literature §B.2 says k=8). Divergence's K·L is wrong. Consequence: any sensitivity or noise
calibration derived from "K·L" would be off by L = 28. The math report's bounds (‖h_x‖₂ ≤ √k pooled,
√(kL) per-layer) are the correct ones. (VERIFIED by reading the code.)

### R2 — math §1.3 (PLAUSIBLE): "HF Trainer with gradient accumulation scales the aux term by the accumulation count" — **VERIFIED by execution (Exp A)**

`trainer.py:1961-1963` divides the loss by `current_gradient_accumulation_steps` only if
`not model_accepts_loss_kwargs or num_items_in_batch is None`; `MellumForCausalLM.forward` has
`**kwargs` (`modeling_mellum.py:641`), so `model_accepts_loss_kwargs=True` (`trainer.py:498-503`;
printed `True` in Exp A). CE is `sum/num_items_in_batch` over the logical batch
(`loss_utils.py:32-46`, count over `labels[...,1:]`, `trainer.py:518-528`), while
`loss += coef·aux(mb)` is added per forward call (`modeling_mellum.py:693-700`).

Exp A (`critic/expA_coef1.py`, tiny Mellum, fp32, 4 ragged examples, `per_device_train_batch_size=2`,
`gradient_accumulation_steps=2`, sequential sampler, coef=1 so the term is visible; gradient captured
in `on_pre_optimizer_step`): relative L2 of the HF accumulated gradient versus

| reference | rel-L2 (all params) | rel-L2 (router `gate.weight`) |
|---|---|---|
| R1: `CE_tokenmean(all) + coef·[aux(mb1)+aux(mb2)]` | **0.0** | **0.0** |
| R2: `CE_tokenmean(all) + coef·mean(aux(mb1),aux(mb2))` | 0.365 | 1.003 |
| R3: `CE_tokenmean(all) + coef·aux(all 4 examples)` | 0.479 | 1.315 |

So what HF Trainer *actually* optimises for a Mellum run with accumulation is the per-microbatch aux
with an effective coefficient `G·coef` (G = accumulation steps). For the presets (batch 256,
microbatch 8) an HF reference run would use f on 8 sequences and coef 0.032. This is a **decision
point** (see G1), not a flaw in the surrogate.

### R3 — divergence §1.2/§7.8: "DPTrainer default `use_performance_kernels=False` ⇒ dense every-token-through-every-expert MoE on every host, incl. CUDA; grouped flag captured by the first class-level patch per process" — **VERIFIED**

`_training_arguments.py:436` (`use_performance_kernels: bool = False`); `_dp_trainer.py:841-847`
(`kernels=bool(self.args.use_performance_kernels)`); `_factory.py:316-321`
(`grouped_moe = kwargs.get("grouped_moe", kernels)`); `_router.py:59-92` (class-level
`__opaque_patched__` guard, first factory wins); `kernels/moe.py:602-633` (`grouped=False` ⇒
`Opaque_MoE.apply` = dense). Note the `use_performance_kernels` docstring
(`_training_arguments.py:429-435`) lists rope/rms_norm/activation/cross_entropy and does not mention
MoE — a documentation gap. Consequence at 12B: the DPTrainer default computes all 64 experts for every
token (8× the routed expert FLOPs), unless `performance_kernels_config={"grouped_moe": True}`.
No report quantified this cost.

### R4 — primitives §2.3/§4.1: "same noise key on every rank; `_create_grad_fn` hardcodes `argnums=0`, no `pre_clipping_transform`" — **VERIFIED**

`_dp_trainer.py:1478` (`quantile_noise_key = gradient_noise_key = key(a.seed)`), `:1611-1614`,
`:1628-1636` (`key=key(a.seed)` for MF); `:4256-4290` (all three branches pass `argnums=0`,
`normalize_by=expected_batch_size`, `return_aux=True`; no `pre_clipping_transform`). Noise is added
after `sum_gradients_` (`:2172-2183`), so a noised probe leaf is rank-identical.

### R5 — divergence §3.4: "`all_valid_attention` reads the whole physical microbatch mask" — **VERIFIED**

`runtime/masking.py:195-216` unwraps functorch batched tensors and tests `physical_mask.all()`.

### R6 — Theorem citations the reports could not re-fetch — **all VERIFIED from primary PDFs** (`critic/*.txt`)

| claim (report) | source text found | verdict |
|---|---|---|
| Zhu–Dong–Wang Def. 7 (dominating pair), Thm 10 (adaptive composition) — math §4(b),(d) | `zdw2106.08567.txt:272-280, 323-326`: "Theorem 10 (Adaptive composition of dominating pairs). If (P,Q) dominates M and (P′,Q′) dominates M′, then (P×P′, Q×Q′) dominates the composed mechanism (M, M′)." | VERIFIED |
| Feldman–Shenfeld arXiv:2602.17284 Thm 3.3 / Alg. 8–9 applies to an arbitrary dominating PLD — primitives §3.2 (quoted from Rust doc comment) | `fs2602.17284.txt:494-540`: Lemma 3.2 (= ZDW Thm 11: Poisson subsampling of a dominated mechanism is dominated by (Pλ,Q) remove / (Q,Pλ) add); "Theorem 3.3. Given λ∈(0,1] and a PLD realization L … φ_λ(l) = ln(1+(e^l−1)/λ)" and "Theorem 3.3 directly implies a practical approach for computing an upper bound on the PLD of any subsampled algorithm M dominated by a pair of distributions (P,Q) … (Alg. 8, 9)." | VERIFIED — the Rust citation is accurate; composing `gaussian(nm)|gaussian(σ_h)` inside `poisson` is theorem-backed (the composed pair dominates the joint un-subsampled release by Thm 10; Lemma 3.2 then subsamples it once). |
| Denisov et al. Thm 2.1 (adaptive rows) — math §5 | `denisov2202.08312.txt:225-230`: "Theorem 2.1 … ‖C(G−H)‖_F ≤ κ … M(G)=B(CG+Z) … satisfies the same DP guarantee … even when the rows of the input are chosen adaptively." | VERIFIED |
| Dong–Roth–Su Gaussian-mechanism GDP theorem number — primitives open item | `drs1905.02383.txt:586-587`: "Theorem 2.7 … ξ ∼ N(0, sens(θ)²/μ²). Then M is μ-GDP."; `:928`: "Corollary 3.3. The n-fold composition of μ_i-GDP mechanisms is √(Σμ_i²)-GDP". | VERIFIED — the whitening/Mahalanobis fact used for per-group allocation is Thm 2.7 applied to the whitened statistic. |
| Andrew et al. Thm 1 — literature B1 | `literature/txt/1905.03871.txt:399-425` (statement and proof as quoted). | VERIFIED |
| Mellum2 TR §3.6 Megatron running-average f; appendix "Router bias update rate 10⁻³"; "all auxiliary-loss computations use FP32" — literature F | `literature/txt/mellum2_tr.txt:929-945, 2579` | VERIFIED |

### R7 — literature §B.2 item 1: count-vector sensitivity "L2 ≤ L·T_x·k (worst case all on one expert) … Δ₂ ≤ 0.031 per step … ~25 % relative noise per coordinate at noise multiplier 1" — **REFUTED as a tight bound (valid but loose by √k)**

Top-k assigns each (token, layer) to k *distinct* experts, so one expert receives at most L·T_x
assignments and ‖c(x)‖₂ ≤ √(max·sum) = √(L T_x · L T_x k) = L T_x √k, not L T_x k. After the divisor
L·B̄·T the per-step Δ₂ is √k/B̄ = 2.83/256 = 0.011 (math §3, VERIFIED numerically there), giving
r = 8.8 % at σ_h = 1, not 25 %. Literature's own "clipping is needed on the count vector" is also
moot for the pooled-fraction release (structural bound). Privacy-safe (an over-bound) but a 2.8×
utility misestimate that would mis-size σ_h.

### R8 — empirical E5: "H5 REFUTED for topically clustered documents (≈40 % of experts unused per example at T=512, E=64/k=8)" — **downgraded to artifact / OPEN**

`empirical/common.py:50-57`: each "structured" example draws all T tokens from its own **6-token**
vocabulary subset; E5 uses a **1-layer random-init** model. The router sees hidden states that are
(embedding of one of 6 tokens) + one attention mix, so the set of distinct routing decisions is
tiny by construction and the unused-expert fraction is largely predetermined (6 token types × 8 slots
cannot cover 64 experts even before context effects). The observation "deviation does not shrink
with T" is the same artifact. H5 for real code documents at T=1024 with a trained router is **open**;
the experiment must be re-done on the checkpoint (G3).

### R9 — New (Exp B): HF's aux-loss top-k is computed from a bf16 softmax, the forward's from fp32 — **VERIFIED (toy)**

`modeling_mellum.py:584` (`softmax(layer_gate, dim=-1)` in the logits dtype) vs `:335`
(`softmax(router_logits, dtype=torch.float)`). `critic/expB.json` (200 k synthetic bf16 logit rows,
E=64, k=8):

| logit scale | median top-1 prob | tokens whose aux top-k set ≠ forward top-k set | bf16 ties at k/k+1 boundary | max |f_aux − f_fwd| / (k/E) |
|---|---|---|---|---|
| 0.5 | 0.044 | 1.68 % | 6.7 % | 1.0e-3 |
| 1.0 | 0.096 | 0.26 % | 4.5 % | 4.4e-4 |
| 2.0 | 0.261 | 0.03 % | 4.3 % | 1.6e-4 |
| 4.0 | 0.570 | 0.01 % | 4.3 % | 8.0e-5 |

Small for f(B) (bounded by 1e-3 k/E) but it means "HF's f(B)" is not "the executed routes"; the
design should define f from the executed (fp32-softmax) top-k and say so. It also means an aux
implemented under vmap from `OutputRecorder`/hook logits must choose a dtype for the recomputation.

### R10 — divergence open question 4 (padding side of alignment collators) — **closed: right padding**

`packages/opaque-alignment/src/opaque/api/alignment/dpo/collator/_preference.py:15-17, 40-46` and
`sft/collator/_language_modeling.py:19, 54, 206`: both collators right-pad. The fully-masked-row
convention issue (divergence §3.5) cannot arise with the repo's own collators.

### R11 — preset clipping mode — **VERIFIED "fixed"**

`examples/train_dpftrl.py:611-622`: `--clipping-mode` default `"fixed"`, `--clipping-norm` default
0.9; the `mellum2-kstack` preset (`:873-896`) does not override them. Primitives' AUTO-S caveat for a
load group (every load vector rescaled to norm λ) therefore does not bite the preset, but the design
must exempt the load group if `auto` is ever selected.

---

## 2. Contradictions between reports (specific, with quotes)

**C1 — H3 verdict (divergence vs empirical).**
Divergence: "H3 REFUTED at toy scale: bf16 Opaque-vs-HF drift (4.6e-3–5.0e-3) exists with 0/128
route flips, so residual drift is accumulation/staging, not routing discontinuity".
Empirical: "H3: VERIFIED for router/expert grads (12.9 %/11.1 % → 1.7 %/1.5 % when pinned); minor for
attention-only (1.34 % vs 1.13 %)".
They measured different pairs: divergence compares Opaque-bf16-vmap with HF-bf16 (the DP-vs-oracle
pair; both agree it has 0 flips at 2 layers), empirical compares bf16 with fp32 (precision, not vmap).
Neither is the PR #980 measurement, which is untraceable (no tracked oracle script for Mellum; the
figure lives only in the PR body). At 28 layers the two bf16 implementations can diverge enough to
flip routes *between themselves* — untested. **Resolution needed (G2): define the oracle.**

**C2 — Σ_e f_e.** Divergence "Σ_e f_e = K·L" vs math "Σ_e f_e = k (verified 2.0 for k=2)" and
literature "Σ_e f_e = k = 8". Math/literature are right (R1).

**C3 — Load-release noise level.** Literature "~25 % relative noise per coordinate per step at noise
multiplier 1" vs math "r = σ_h E/(B̄√k) = 8.8 % at σ_h = 1, B̄ = 256" vs empirical "0.177 σ (k/E units)
at B=128" (= 8.8 % at B=256, consistent with math). Literature is loose by √k (R7).

**C4 — Which f̃ is "faithful".**
Literature: "A lagged / running-average f is not an approximation of what Mellum2 saw — it is what
Mellum2 saw" (Megatron per-layer running average) and "Lagged f is the faithful choice".
Math: "'faithful to HF' should mean 'faithful to Fact A on the logical batch'" and warns HF Trainer
itself deviates under accumulation (now VERIFIED, R2).
Empirical: "A lagged f̃ … gives 65–166 % gradient error at balanced init … 23–55 % with a persistent
induced imbalance".
Divergence/primitives: silent on the choice. Four candidate objectives (logical-batch HF pooling;
HF-Trainer-realised per-microbatch with G× coefficient; Megatron per-layer running-average;
per-sequence) — **the design must choose one and state why (G1).**

**C5 — H4 "negligible aux".** Divergence: "aux gradient is 2.4e-4 of the CE gradient at coef 0.001 …
negligible". Literature/empirical/math: the surrogate gradient is ∝ (f̃ − k/E) and identically zero at
balance (empirical: "uniform f̃ yields an identically zero gradient (‖grad‖ ~3e-8)"); with induced
imbalance ‖∇aux‖ rose 5× (empirical E4b: 0.330 → 1.711). So 2.4e-4 is a balanced-random-init
artifact; H4's "negligible" holds exactly when the loss is unnecessary. Not a factual contradiction
but the interpretation "H4 supported" is unsafe for the design.

**C6 — Clipping the load vector.** Literature: "clipping is needed on the count vector too";
math: "needs no clipping — bound is structural (√k)"; primitives: "PerGroup clipping bounds it
separately (λ)". Resolvable: release the pooled *fraction* h_x (structural ‖h_x‖₂ ≤ √k), set the group
bound to the structural value scaled by λ so clipping never triggers and the release is unbiased
(math 4(b) remark (ii)); the PerGroup entry is then a *bound*, not a clip.

**C7 — Cost framing of the same-batch release.** Primitives: "The price is utility, not ε: σ_grad
rises by √(1+λ/C)"; math: "the right knob is the budget split η … λ … is how you choose η"; literature:
"the joint-mechanism cost on the gradient stream is the B1 formula and can be made <1 % by choosing
σ_b large". All three are the same Mahalanobis constraint in different parametrisations (primitives
E1(f) checks it numerically; math shows the naive per-group σC_g allocation is REFUTED). The design
should present **one** table (ρ or η → gradient-σ inflation, r after EMA, ε if nm is held) rather than
three.

**C8 — Route pinning consequences.** Divergence: "pinning routes would remove a source of *variance*
but cannot make the two paths agree better than the accumulation floor"; empirical: "pinning routes to
the fp32 choice collapses [router/expert error] to 1.66 % / 1.47 %"; math: "pinning … makes the
per-example loss C^∞ … no DP consequence; frozen-base pinning trades this for train/inference
routing mismatch". Consistent once the reference is fixed (C1), but the reports disagree on
*whether pinning is worth it for the presets* (attention-only LoRA): empirical says +0.2 pp only.

**C9 — OLMoE evidence.** Literature notes its own internal contradiction ("appendix line 3985 states
the opposite; Section 4.3 is the reasoned statement") — cite with that caveat only.

**C10 — Hook vs `output_router_logits`.** Divergence: "Route information is still obtainable under
vmap by calling the backbone `model.model(..., output_router_logits=True)`"; primitives: "A forward
hook on `MellumTopKRouter` … works under vmap(grad) with patches applied"; both note the chunked-CE
patch falls back to the full-logit forward when `output_router_logits` is set on the causal-LM.
Consistent, but the memory-safe route (hook + chunked CE, or extending the chunked-CE patch to return
router logits) is a design choice with unexecuted interactions (gradient checkpointing recompute
double-fires hooks; `torch.compile`).

---

## 3. What is missing

**Modalities not run**
- M1. Nothing on a trained model. Unknown at scale: ||f(B) − k/E|| on KStack at B=128–256 (decides
  whether a noised/lagged f̃ carries signal; empirical's noise-error ratio is linear in it); route-flip
  rate between Opaque-bf16-vmap and HF-bf16 at 28 layers; per-example gradient-norm distribution with
  LoRA r=16 on q/k/v/o under C=0.9; per-example expert usage on real code at T=1024.
- M2. No CUDA path executed: `Opaque_FusedMoE` (Triton), `Opaque_LinearCrossEntropyLoss`, fused LoRA,
  flash/cuDNN SDPA batching rules under vmap, bf16-autocast staging (HF grouped_mm casts input to the
  weight dtype, `integrations/moe.py:331-334`; Opaque casts to the autocast dtype on CUDA).
- M3. The dense-MoE default (R3) has no cost measurement at 12B; the reports do not say whether the
  presets set `grouped_moe`.
- M4. DP-FTRL specifics for the preset (bands=64, b-min-sep): per-step noise on a load leaf
  (base σ·‖row_t(C⁻¹)‖) was measured only at bands=4 (primitives E2); the EMA-vs-MF-workload question
  (math: "prefix sums accurate for free" — true only if the consumer is the linear workload the
  strategy was optimised for, i.e. momentum-weighted prefix sums, not an arbitrary EMA) is
  unquantified.
- M5. PEFT `target_parameters` LoRA on stacked expert weights under `functional_call`+vmap: untested
  (blocks any experts-trainable variant and H5's levers).

**Claims/decisions unverified or undecided**
- M6. The oracle of PR #980 and the 29 % → 1.3 % figure (C1).
- M7. Which objective is "faithful" (C4, R2). Includes: Megatron per-layer mean-of-products vs HF
  product-of-means (literature open Q2), attention-mask vs label-mask weighting (math: HF uses the
  attention mask, so prompt tokens count in aux under SFT prompt masking), token- vs example-weighting
  of CE (divergence §4: representable with a public N̄).
- M8. Router precision: Mellum2 pretraining used an FP32 router; HF computes logits in bf16 and the aux
  top-k from a bf16 softmax (R9). Nobody proposed the cheap `MellumTopKRouter` fp32-logits patch that
  H3's "pin in fp32" needs, nor decided whether it should be default (it would move Opaque *away* from
  the HF-bf16 oracle and *towards* the pretraining router).
- M9. z-loss (Mellum2 pretraining 1e-3, per-token separable, absent in HF/Opaque): in or out of the
  "faithful" objective? Also directly targets the bf16 router round-off behind H3 (ST-MoE §3.1).
- M10. DPO path (`examples/train_dpo.py` `mellum2-codesec`): no report analysed the reference-model
  forward, the aux term over chosen+rejected, or how TRL's dropping of the coefficient interacts.
- M11. Under Poisson / b-min-sep the released f̂ = (1/B̄)Σh_x has Σ_e f̂_e = k·|B_t|/B̄ ≠ k; math's
  post-release renormalisation fixes scale but the design must state it and confirm that
  `normalize_by=expected_batch_size` in `_create_grad_fn` equals the b-min-sep per-step expected batch.
- M12. Telemetry: `_dp_trainer.py:2256-2272` logs un-noised loss / group grad-norm means / clip rates
  (primitives flagged). If the design ever logs f(B) un-noised for monitoring it is an unaccounted
  release; only f̂ may be logged. (Pre-existing telemetry is outside this task but the DP review
  protocol's "all releases … included in the privacy statement" applies.)
- M13. Opaque's `Composed` inside Rust `poisson_pld`: valid for symmetric Gaussian pairs (Lemma 3.2 needs
  remove-direction domination); nobody checked the add/remove bookkeeping for asymmetric inner
  mechanisms — flag for whoever composes non-Gaussian side releases.

**Hypothesis scorecard after phase 1**

| H | status | what decides it |
|---|---|---|
| H1 | **decided** (yes; CE weighting is a separable coupling) | — |
| H2 | **algebra decided** (exact); utility of noised/lagged f̃ **open at scale** | M1 (||f(B) − k/E|| on real data) |
| H3 | **undecided**: depends on the oracle (C1) and depth | G2 |
| H4 | decided **only for balanced routing** (C5) | M1 |
| H5 | **open** (E5 is an artifact, R8) | G3 |

---

## 4. Ranked gaps the design phase must not proceed without

**G1 (blocking) — Fix the target objective.** Question: which of (a) logical-batch HF pooling
(Fact A), (b) HF-Trainer-realised (per-microbatch f, G× coefficient — Exp A), (c) Megatron per-layer
running-average f, (d) per-sequence aux, is "faithful", with which mask (attention vs label), which
CE weighting (example vs public-N̄ token), and which coefficient (1e-3 pretrain / 1e-4 Mellum2 SFT)?
How: a one-page decision note; recommended default (a) with executed-route f (R9), attention mask,
coefficient configurable, and an explicit statement that HF Trainer itself deviates (Exp A) so the
DP path is *more* faithful to Fact A than HF.

**G2 (blocking) — Define the oracle and the drift metric; reproduce PR #980's number.** Question:
what exactly is compared (HF bf16 batched vs Opaque bf16 vmap? fp32?), at what depth/batch, and how
much of the residual is route flips vs accumulation? How: obtain the script behind "29 % → 1.3 %" from
the author or write one (real checkpoint, one microbatch, GPU): report rel-L2 *and* a route-flip
counter (empirical E1b's method) separately; without this, requirement (d) "numerically stable vs the
non-DP HF path" cannot be claimed.

**G3 (blocking for requirement (c)) — Real-model statistics.** Question: with the `mellum2-kstack`
preset (LoRA r=16 q/k/v/o, C=0.9, B=256, T=1024) what are the per-example gradient-norm quantiles, the
per-batch ||f(B) − k/E||, and per-example expert usage on KStack? How: one `clipped_grad(..., return_aux)`
pass over ~256 examples plus a router-hook histogram, on one GPU (minutes). These numbers anchor λ/σ_h
(math §7's r formula), the clipping norm, and decide H4/H5.

**G4 — Write the exact per-step mechanism and its accountant for both stacks.** Question: DP-SGD:
PerGroup probe leaf (primitives Route B, accounting `gaussian(nm)` unchanged, gradient σ ×√(1+ρ)) or
explicit σ_eff (math 4(c))? DP-FTRL: second PerGroup group in the same MF stream (math 5(i), latch OK
per primitives E2) — with what consumer filter (G7)? How: one table (ρ/η → gradient-σ inflation, r
after smoothing, ε at fixed nm) for the preset regime (nm=0.5622: σ_h=4·nm ⇒ ε 3.0→3.51 or ×1.041
gradient noise — primitives E1), plus the `mf_gaussian` variant via `calibrate`. Include the
renormalisation/clamp post-processing (M11) and the adjacency statement (add/remove; ×2 replace-one).

**G5 — Router precision / pinning design.** Question: fp32 router logits (pretraining-faithful,
removes bf16 ties, cost ≈ 64×2304 GEMM/token) as a Mellum patch option? Pin routes per example from the
current model or not? Does f use executed routes? How: implement a `MellumTopKRouter` forward
replacement behind a flag; measure flips vs HF-bf16 and vs fp32 on the checkpoint (part of G2).

**G6 — Dense-MoE default.** Question: should Mellum default to `grouped_moe=True` on CUDA in DPTrainer
(and should `use_performance_kernels` docs mention MoE)? What is the 12B time/memory of the dense
default with frozen experts? How: measure both paths on one microbatch on GPU; decide the default.

**G7 — MF-consistent smoothing of the load stream.** Question: for bands=64, what is the per-step
σ·‖row_t(C⁻¹)‖ on the load leaf, and which linear filter (momentum-matched prefix sum vs EMA) inherits
the strategy's error guarantee? How: compute row norms from `band_mf_strategy(64, momentum, lr)`
(`_band_mf.py:142-171`) and the filter's noise variance in closed form; tabulate r after filtering.

**G8 — Scope decisions: DPO, z-loss, expert-trainable variants.** Question: is the DP-DPO preset in
scope (aux over chosen+rejected, reference forward)? Is z-loss part of "faithful"? Is any
experts/router-trainable configuration in scope (then M5 must be executed)? How: explicit scope list;
for M5 a 10-line `functional_call`+vmap smoke test with PEFT `target_parameters`.

**G9 — Privacy hygiene of the new state.** Question: is anything un-noised about f(B) logged,
checkpointed, or synced? How: enumerate every new tensor (probe leaf, f̂ EMA, pinned routes) and mark it
public-post-processing or private-internal; make the checkpoint sidecar carry only f̂/EMA state; add the
release to the privacy statement (DP review protocol "Composition and accounting").

**G10 — Hook mechanics under checkpointing / compile / microbatching.** Question: does the router-hook
capture stay per-example-exact with gradient checkpointing (recompute double-fire) and `microbatch_size`
chunks, and does `torch.compile` of the transform tolerate it? How: extend primitives' E3 to
`gradient_checkpointing_enable()` and a 2-chunk `microbatch_size`; assert equality with the unhooked f_x.
