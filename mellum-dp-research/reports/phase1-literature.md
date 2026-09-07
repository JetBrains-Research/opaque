# Phase 1 — Literature sweep (agent: `literature`)

Scope: seven angles (A–G) from the brief, each searched separately; primary sources
downloaded to `scratchpad/research/literature/pdf/` and text-extracted to
`scratchpad/research/literature/txt/` (pymupdf). Line numbers below of the form
`txt/<id>.txt:NNN` refer to those extractions; page numbers are the paper's own.
Repo/venv facts are cited as `path:line`. Status legend: **VERIFIED** = I read the
primary text or ran the command; **PLAUSIBLE** = abstract/secondary summary only;
**REFUTED** = contradicted by primary text.

Nothing in the tracked repo was modified. All files written under
`scratchpad/research/literature/` and this report.

---

## 0. Executive summary (what the literature settles, and what it does not)

1. **There is essentially one prior DP-MoE paper** (Tholoniat, Inan, Kulkarni, Sim,
   PPAI'24, arXiv:2402.07334). It (a) identifies exactly Opaque's problem — the batch
   balancing loss ℓ(B) "in which all the samples are entangled: there is no immediate
   way of defining ℓ(sᵢ) such that ℓ(B)=Σⱼℓ(sⱼ)" — and (b) *drops the loss* while
   keeping the router trainable, arguing pretrained gates are already balanced. It lists
   "decoupling it into per-sample load-balancing losses" and "finer-grained noise addition
   (e.g., exclusively to the experts in use, or scaled by load) compared to … isotropic
   noise to all experts" as open problems. No later paper closes those gaps
   (arXiv API sweep + web search, Sept 2026). **VERIFIED.**
2. **Switch eq. (4)–(6): only P is differentiable, f is not** (Fedus et al. 2022, p. 7–8:
   "the P-vector is differentiable, but the f-vector is not"). The HF Mellum implementation
   pools f and P over *all 28 layers and all batch tokens* into one product
   `E·Σ_e f_e P_e` (not a per-layer sum) and normalises f by rows, not by k. Consequently
   ∇aux = E·Σ_e f_e(B)·∇P_e(B) is *linear in per-token router probabilities* once f(B) is
   held fixed — the algebraic core of H2 is confirmed. **VERIFIED.**
3. **A lagged / running-average f is not an approximation of what Mellum2 saw — it is what
   Mellum2 saw.** The Mellum2 tech report states Megatron's global aux loss "maintains a
   running average of per-expert token counts across microbatches … The loss at each
   microbatch is computed against this running estimate rather than against a true global
   count" (arXiv:2605.31268 §3.6). Qwen's "Demons in the Detail" (ACL'25) Algorithm 1 is
   the same buffered scheme. **VERIFIED.**
4. **The "release the batch statistic privately, then use it as a constant" pattern has
   primary-source precedent with a clean joint accounting**: Andrew et al. NeurIPS'21
   Theorem 1 (adaptive clipping) shows two Gaussian sum queries on the *same* sample
   (clipped-gradient sum + clipped-count sum) are "equivalent to pre- and post-processing
   of a single query with sensitivity S" and gives the closed form
   z_Δ = (z⁻² − (2σ_b)⁻²)^(−1/2). Ponomareva et al. ("How to DP-fy ML") §5 option (b)
   describes exactly the same recipe for BatchNorm statistics. Opaque already implements
   this joint-allocation pattern (`opaque.api.engine.noise_allocation.paired_noise_stddevs`,
   `adaptive_clipped_grad(fraction_noise_std=…)`). **VERIFIED.**
5. **Mellum2 pretraining used**: Switch-style *global-batch* aux loss, coefficient 1e-3,
   router z-loss 1e-3, router in FP32, dropless top-8 routing, seq 8192; SFT reduced the
   aux coefficient to **1e-4** "since the router is already well-balanced after
   pre-training and a smaller coefficient avoids over-constraining expert utilization on
   the narrower SFT distribution". Their ablation found *per-sequence* aux "slightly
   better test loss than global-batch balancing on short runs" (A.4). The HF config's
   `router_aux_loss_coef=0.001` therefore matches *pretraining*, not their own SFT recipe.
   **VERIFIED.**
6. **Evidence that dropping the aux loss during fine-tuning of a well-balanced checkpoint is
   benign**: OLMoE (§4.3) measured the balancing loss on SFT data and found it *decreased*
   during SFT without the loss (12.16 vs 12.22) and that expert activation patterns "remain
   around the same"; adaptation without the loss scored higher (54.0 vs 52.8 SFT). Tholoniat
   made the same call. Counter-evidence: harmful-fine-tuning work measures nontrivial
   routing drift (KL) under benign SFT; Mellum2 itself found the router equilibrates over
   ~100B tokens of long-context training. So: benign for LoRA-scale updates, not a theorem.
   **VERIFIED (OLMoE numbers); PLAUSIBLE (generalisation to Mellum2).**
7. **Top-k discontinuity is documented** (ReMoE §3.2 example (0.51,0)→(0,0.51); ST-MoE §3.1
   bf16 roundoff → router z-loss). Mellum2's own RL section reports that "for the same
   hidden state, the inference-time router may dispatch a token to a different expert than
   the trainer-side router … even though the weights are identical" — route flips from
   numerics are a known property of *this* checkpoint. DenseMixer (NeurIPS'25) shows the
   *conventional* router gradient is imprecise and that an STE through top-k (all-expert
   forward) improves post-training. **VERIFIED** (H3 mechanism is real); the claim that
   pinning routes tightens per-example gradient norms is untested in the literature.
8. **Per-example gradient-norm distributions for MoE under DP-SGD: no primary source
   found.** Nearest: "Preserving Long-Tailed Expert Information" §3.1 proves routed expert
   i receives Θ(T·pᵢ) effective updates and observes lower gradient magnitude for cold
   experts (non-DP). H5 must be settled empirically by the other agents.
9. **Heterogeneous per-step composition is standard** (Koskela & Honkela 2021 Thm 4 and
   §6.2 heterogeneous Poisson-subsampled Gaussian; Gopi–Lee–Wutschitz PRV; Google
   `dp_accounting` PLD compose; Opaque `a | b`). The subtle point for H2 is that two releases
   computed on the **same** Poisson sample must be accounted as **one** subsampled Gaussian
   mechanism with joint sensitivity (Andrew Thm 1 route), not as two independently
   amplified mechanisms. **VERIFIED (theorem) / derivation for the Poisson case is mine.**

---

## A. Differentially private training / fine-tuning of MoE models

### A.1 Sources

| # | Source | What it establishes | Status |
|---|--------|---------------------|--------|
| A1 | Tholoniat, Inan, Kulkarni, Sim. *Differentially Private Training of Mixture of Experts Models*. PPAI@AAAI 2024. https://arxiv.org/abs/2402.07334 | First (and, per my sweep, still only) DP-SGD-on-MoE paper. Switch-base-8 (top-1), SST-2/MNLI, ε=8, δ=1/N, clip 1.0, batch 1024, 20 epochs, PRV accountant; 92.0/78.7 vs 94.5/85.4 non-private. | VERIFIED (full text read: `txt/2402.07334.txt`) |
| A2 | Asadian et al.? — *NoEsis: Differentially Private Knowledge Transfer in Modular LLM Adaptation*, ICLR'25 workshop. https://arxiv.org/abs/2504.18147 | "MoE" = per-domain LoRA experts with *deterministic domain routing* (App. D.1); DP-SGD only on shared prompt tokens; no learned router, no balancing. Not relevant to routed MoE. | VERIFIED (HTML) |
| A3 | Makni et al. *SPARTA: An Optimization Framework for Differentially Private Sparse Fine-Tuning*. https://arxiv.org/abs/2503.12822 | DP sparse (masked) fine-tuning improves DP-SGD dynamics; relevant to H5's "which params to train" lever, not MoE-specific. | PLAUSIBLE (abstract) |
| A4 | arXiv API queries `"differential privacy" AND "mixture of experts"`, `"differentially private" AND MoE`, `… AND "sparse experts"`, federated variants (run 2026-09-07) | Only A1, A2 and unrelated hits (federated insider-threat, face de-ID, risk bounds). No 2025–2026 DP-MoE paper exists on arXiv under those terms. | VERIFIED (command output in transcript) |
| A5 | PC-MoE (arXiv:2506.02965), CryptoMoE, PM-MoE (2502.00354) | Collaborative / cryptographic / personalised-FL MoE; no DP guarantee on training. | PLAUSIBLE (search snippets only) |

### A.2 What A1 says, verbatim-critical parts (`txt/2402.07334.txt`)

- Balancing loss (their eq. 1) = Switch eq. 4 with f from argmax.
- "The first problem is that per-sample balancing loss ℓ(sᵢ) is ill-defined … We only have a
  per-batch balancing loss ℓ(B) … in which all the samples are entangled: there is no
  immediate way of defining ℓ(sᵢ) such that ℓ(B)=Σℓ(sⱼ)."
- "The simplest solution … is to simply remove the load-balancing loss, which is the solution
  we adopt. This is particularly relevant for fine-tuning use-cases, where pretrained
  networks start with well-balanced gating layers. To avoid expert collapse, it is possible
  to additionally freeze the gating layers. One can also modify the load-balancing loss,
  replacing Eq. 1 by an expression that can be decomposed into per-sample load-balancing
  losses. The design … is left for future work."
- Experiments: "we remove the load-balancing loss, without freezing the gating layers,
  relying on the observation that our pretrained networks have already good gating layers
  that are relatively insensitive to the load-balancing loss."
- Per-sample gradients for experts: they re-introduce a batch dimension per expert
  (`[B, max_b C_b^i, H]` zero-padded) — same idea as Opaque's dense `opaque_moe`; note the
  cost "we route B × max_b C_b^i tokens, many of which can be zero".
- Noise on unused experts: "the current approach of adding isotropic noise to all experts";
  open problem: "finer-grained noise addition (e.g., exclusively to the experts in use, or
  scaled by load)".
- Accounting: PRV (Gopi et al. 2021).

### A.3 Bearing on hypotheses

- **H1** – supported: A1 identifies the balancing loss as *the* non-separable term; nothing
  else in the MoE forward is flagged.
- **H2** – A1 explicitly names the per-sample decomposition as open; A1 does not attempt a
  DP-released f. So H2 is novel relative to the DP-MoE literature (and consistent with it).
- **H4** – A1's practice (drop aux, train router, clip 1.0, ε=8) worked on Switch-base-8
  fine-tuning; it is the closest empirical precedent for "aux negligible when the checkpoint
  is balanced". Caveat: 8 experts, top-1, classification.
- **H5** – A1 notes isotropic noise on all experts as a limitation; no measurement.

---

## B. Batch-level (non-separable) losses under DP-SGD; releasing a batch statistic and using it as a constant

### B.1 Sources

| # | Source | What it establishes | Status |
|---|--------|---------------------|--------|
| B1 | Andrew, Thakkar, McMahan, Ramaswamy. *Differentially Private Learning with Adaptive Clipping*. NeurIPS 2021. https://arxiv.org/abs/1905.03871 | Algorithm 1 releases a clipped-count statistic b̃ₜ with Gaussian noise σ_b each round and uses it (as a constant) to update the next round's threshold. **Theorem 1**: one step with σ_b on Σbᵢ and z_Δ on Σ∆ᵢ "is equivalent (so far as privacy accounting is concerned) to one step of non-adaptive DP-FedAvg with noise multiplier z if we set z_Δ = (z⁻² − (2σ_b)⁻²)^(−1/2)". Proof: send (∆ᵢ/σ_Δ, (bᵢ−½)/σ_b), norm ≤ S = ((C/σ_Δ)² + (1/2σ_b)²)^½; "the two Gaussian sum queries … are equivalent to pre- and post-processing of a single query with sensitivity S". Cost: σ_b=m/20 ⇒ z_Δ≈1.005 for z=1, m=100 ("0.5% more noise"). | VERIFIED (`txt/1905.03871.txt:270-290, 399-455`) |
| B2 | Ponomareva et al. *How to DP-fy ML*. JAIR 2023. https://arxiv.org/abs/2303.00654 §5 (p. 46–47) | Lists BatchNorm and "losses that can't be decomposed into per-example losses, such as pairwise losses" as breaking per-example reasoning. Option (b): "privatize BatchNorm per-batch mean and standard deviation … per example clipping (with a norm different from DP-SGD clipping norm) and Gaussian noise addition to the sum … Such privatized batch mean then would be employed during the forward pass … Privacy accounting for a sequential combination of Gaussian Mechanism for a BatchNorm and Gaussian Mechanism for DP-SGD can be handled for example via accounting for adaptive streams Denisov et al. (2022)." Option (a): public data for the statistic (Davody et al. 2020). | VERIFIED (`txt/2303.00654.txt:2896-2965`) |
| B3 | Davody, Adelani, Kleinbauer, Klakow. *On the effect of normalization layers on DP training*. 2020. https://arxiv.org/abs/2006.10919 | BN under DP by concatenating a *public* dataset to every lot and computing BN statistics on it (Alg. lines ~494-540). No private release of the statistic. | VERIFIED (`txt/2006.10919.txt:455-540`) |
| B4 | Kong, Muñoz Medina, Ribero, Syed. *Differentially Private Optimization for Non-Decomposable Objective Functions*. ICLR 2025. https://arxiv.org/abs/2310.03104 | Contrastive/InfoNCE: naïve per-example clipping gives sensitivity O(nB) (eq. 2); they instead clip *pairwise similarity gradients* ∇Zᵢⱼ (Lemma 4.1 decomposition eq. 3), and Theorem 4.2 bounds Δ₂(∇L) ≤ (G₁+G₂+(n−1)L)B; Lemma 4.5 shows L=O(1/n) for cosine InfoNCE so sensitivity is O(1) in n. Corollary 4.4: each step is the Gaussian mechanism at that sensitivity. **This is the "bound the coupling constants, not the per-example gradient" template.** | VERIFIED (`txt/2310.03104.txt:298-420`) |
| B5 | Li et al. *Differentially Private Contrastive Learning via Bounding Group-level Contribution*. 2026. https://arxiv.org/abs/2604.26467 | Sample-level clipping ⇒ Δ=(2B+1)C (eq. 6); batch-level clipping (DP-CLIP, Huang et al. 2023) ⇒ Δ=2C but loses accumulation; Theorem 2: restrict negatives to within-group ⇒ Δ=2C independent of group size. RDP accounting. Surveys prior DP-contrastive work (DP-CLIP batch-level; Kong et al. pairwise). | VERIFIED (HTML summary; theorem statements as reported by fetch) |
| B6 | Luo, Wu, Adeli, Fei-Fei. *Scalable Differential Privacy with Sparse Network Finetuning*. CVPR 2021. | Search summaries claimed it implements "private BN"; the paper actually says "batch normalization is incompatible with the computation of per-sample gradients in DP-SGD, and so we tune group normalization as a close analog." | REFUTED claim of private BN (`txt/luo2021.txt:456-460, 629-632`) |
| B7 | Asadian et al. *Self-Supervised Pretraining for DP Learning*. 2022. https://arxiv.org/abs/2206.07125 | Table 9 footnote: "Batch normalization trained by the public dataset" — i.e., Davody-style public statistics, not noisy release. | VERIFIED (`txt/2206.07125.txt:138-139`) |
| B8 | Nguyen et al. *Batch Clipping and Adaptive Layerwise Clipping for DP-SGD*. 2023. https://arxiv.org/abs/2307.11939 | Batch clipping (clip the *batch* gradient) with f-DP analysis "allows us to implement Batch Normalization Layers" — a different route: give up per-example structure, treat the batch as the unit. | VERIFIED (`txt/2307.11939.txt:93-130, 711-716`) |
| B9 | Huang, Xie. *Revisiting Privacy Amplification by Subsampling in Selective Release DPSGD*. 2026. https://arxiv.org/abs/2606.04384 | Cautionary: DPSUR's accounting "overlooks the variation in sampling probability introduced by the selective release mechanism". Any side release that changes what is sampled/kept must be re-analysed. | PLAUSIBLE (abstract) |

### B.2 Synthesis for H2 (the surrogate loss)

Let the HF loss be `aux(B) = E·Σ_e f_e(B)·P_e(B)` with, per the venv implementation
(`.venv/lib/python3.11/site-packages/transformers/models/mellum/modeling_mellum.py:540-612`),

- `f_e(B) = (Σ_l Σ_{t∈B} m_t·1[e ∈ topk_l(t)]) / (L·Σ_t m_t)` (mask-weighted; note Σ_e f_e = k = 8,
  not 1 — HF divides by rows, not by k·rows as in DeepSeek eq. 18),
- `P_e(B) = (Σ_l Σ_t m_t·p_e^l(t)) / (L·Σ_t m_t)`,
- loss = `E·Σ_e f_e·P_e`, **one** product on layer-pooled statistics (Switch defines it per layer and sums).

Because f is argmax-derived (Switch p. 8: "the P-vector is differentiable, but the
f-vector is not"), autograd gives exactly

    ∇_θ aux(B) = E · Σ_e f_e(B) · ∇_θ P_e(B) = (E / (L·N_B)) · Σ_x Σ_{l,t∈x} Σ_e f_e(B) · ∇_θ p_e^l(t)

which is a sum over examples of `∇_θ s(x; f)` with `s(x; f) := (E/L)·Σ_e f_e · P_e(x)` and
`P_e(x)` the example's own mean router prob (weights 1/N_B → per-example token-mean with
`normalize_by=B` handles the batch divisor exactly as Opaque already does for CE). So:

- **H2 algebra is confirmed by the primary definition.** The only batch coupling is the
  constant vector f(B) ∈ ℝ⁶⁴.
- Three legitimate ways to obtain f without breaking per-example sensitivity, each with a
  primary-source precedent:
  1. **Privately release f(B) each step and use it in the same step** — B1 Theorem 1 pattern
     (joint Gaussian on the concatenated (clipped grad, count-vector) query). Sensitivity of
     the count contribution of one example (add/remove): its assignment vector c(x) ∈ ℕ⁶⁴ has
     L1 = L·T_x·k and L2 ≤ L·T_x·k (worst case all on one expert); after the fixed divisor
     L·N_expected the per-coordinate scale is ≤ k/B. With B≈256, k=8: Δ₂ ≤ 0.031 per step
     versus f_e ≈ k/E = 0.125 at balance — i.e. ~25 % relative noise per coordinate per step at
     noise multiplier 1 if released alone; the joint-mechanism cost on the gradient stream is
     the B1 formula and can be made <1 % by choosing σ_b large (an EMA over steps then gives a
     low-variance f̃). *Note for the implementer*: clipping is needed on the count vector too
     (a single very long or fully padded example otherwise breaks the bound); Opaque's
     `normalize_by` semantics apply.
  2. **Use the previous step's released f̃ (lagged / EMA)** — this is *closer* to what
     Megatron computed during Mellum2 pretraining than a true batch f: "The implementation
     maintains a running average of per-expert token counts across microbatches within each
     optimizer step, resetting the accumulator only at gradient finalization. The loss at
     each microbatch is computed against this running estimate rather than against a true
     global count" (arXiv:2605.31268 §3.6, `txt/mellum2_tr.txt:929-945`). Demons Alg. 1
     (`txt/2501.11873.txt:207-232`) is the same buffered f. With lagged f̃ the per-step
     mechanism is *standard* DP-SGD on a per-example loss (f̃ is post-processing of earlier
     releases), so no new accounting is needed beyond the count release itself.
  3. **Use a public / frozen-base estimate of f** (Davody/B2 option (a)) — e.g. f computed from
     the frozen base model on public code, or simply the uniform vector k/E. With f ≡ k/E the
     surrogate reduces to `(E·k/E)·Σ_e P_e(x) = k` — a constant, zero gradient. So the uniform
     prior is *not* a usable surrogate; the loss only acts through deviations of f from
     uniform. This is a genuine subtlety: the balancing gradient is *entirely* driven by the
     (private) imbalance vector f(B) − k/E.
- **Per-example (per-sequence) aux is a different regulariser, and the literature says how**:
  DeepSeek-V2 eq. 23–25 and DeepSeek-V3 eq. 17–20 define f and P *per sequence* ("T denotes
  the number of tokens in a sequence"); Qwen's Demons paper §2.2 shows micro-batch LBL "is
  almost at the sequence level, and the router is pushed to distribute tokens evenly within
  each sequence, thereby inhibiting expert specialization"; Mellum2 A.4 found "per-sequence
  auxiliary loss produced slightly better test loss than global-batch balancing on short
  runs" but chose global for flexibility. So a per-example aux is (i) a well-known variant,
  (ii) strictly per-example separable with zero extra privacy cost, (iii) a stronger
  constraint (within-document uniformity), and (iv) the *only* form for which the primary
  model authors report a loss *advantage* (short runs). Quantifying the difference for
  Mellum2 fine-tuning is an experiment, not a literature question.

### B.3 Contrastive-loss lessons that transfer

B4/B5 show the productive move for non-separable losses is to *rewrite the gradient as a
sum over bounded per-example (or per-pair) terms with data-independent coupling constants*
and clip those terms — not to clip the batch gradient (B5: loses accumulation; Kong's
"Naive-DP … does not materially reduce the loss"). For the aux loss the coupling constant is
f(B) itself (bounded in [0,k]), which is why the surrogate has bounded per-example
sensitivity once f is fixed — and why f must be released or lagged, mirroring B1.

---

## C. MoE load balancing: definitions, gradients, sequence vs batch, loss-free, z-loss, coefficients, dropping during fine-tuning

### C.1 Sources

| # | Source | What it establishes | Status |
|---|--------|---------------------|--------|
| C1 | Fedus, Zoph, Shazeer. *Switch Transformers*. JMLR 23 (2022). https://arxiv.org/abs/2101.03961 §2.2 eq. (4)–(6) | loss = α·N·Σᵢ fᵢPᵢ; fᵢ = (1/T)Σ_{x∈B} 1{argmax p(x)=i}; Pᵢ = (1/T)Σ_{x∈B} pᵢ(x); "P-vector is differentiable, but the f-vector is not"; multiplied by N so the uniform value is 1; α=10⁻² chosen from a sweep 10⁻¹…10⁻⁵ ("10⁻² balanced load quickly without interfering with training loss"). Mesh-TF pseudocode Fig. 14 confirms per-core computation. | VERIFIED (`txt/2101.03961.txt:418-466, 2131-2150`) |
| C2 | Zoph et al. *ST-MoE*. 2022. https://arxiv.org/abs/2202.08906 §3.1 eq. (5) | Router z-loss L_z = (1/B)Σᵢ(log Σⱼ e^{x_j^{(i)}})²; L_tot = L_CE + c_B L_B + c_z L_z with c_z = 10⁻³, c_B = 10⁻²; motivation: bf16 roundoff in router softmax ("bfloat16 has up to 65,536x worse roundoff errors than float32"). Fine-tuning: aux losses during regular fine-tuning give "small performance gains" (as summarised by OLMoE §4.3 citing ST-MoE). | VERIFIED (`txt/2202.08906.txt:473-520, 562`) |
| C3 | DeepSeek-V2 tech report. https://arxiv.org/abs/2405.04434 §2.2.3 eq. (23)–(25) | Expert-level balance loss with fᵢ = (N_r/(K_r T))Σₜ 1(token t selects expert i), Pᵢ = (1/T)Σₜ sᵢ,ₜ, **T = tokens in a sequence**; α₁=0.003 (plus device-level α₂=0.05, comm α₃=0.02). | VERIFIED (`txt/2405.04434.txt`, eq. block + line 805) |
| C4 | DeepSeek-V3 tech report. https://arxiv.org/abs/2412.19437 §2.1.2 eq. (16)–(20) | Loss-free bias bᵢ added to affinity only for top-K selection (eq. 16), γ=0.001 for 14.3T tokens then 0; **complementary sequence-wise loss** eq. (17)–(20) with normalised sᵢ,ₜ′ and α=0.0001 "just to avoid extreme imbalance within any single sequence"; "DeepSeek-V3 does not drop any tokens". | VERIFIED (`txt/2412.19437.txt`, eq. block; lines 699, 782-784, 1661-1662) |
| C5 | Wang, Gao, Zhao, Sun, Dai. *Auxiliary-Loss-Free Load Balancing Strategy for MoE*. 2024. https://arxiv.org/abs/2408.15664 | Eq. (2) aux loss stated **per sequence of length T** with fᵢ=(N/KT)Σ 1(·); Alg. 1: count cᵢ per batch, eᵢ=cᵢ−c̄, bᵢ += u·sign(eᵢ); eq. (3) bias affects top-K only, not gating weights; biases updated from the *previous* batch "since utilizing the load information of the current sequence will break the causal constraint … leakage of the information of future tokens"; α sweep 1e-2/1e-3/1e-4/0 shows small α ⇒ routing collapse, large α ⇒ worse PPL (Fig. 2); §5.2: Expert Choice leaks > K·log₂((1−R)/R) bits/token. | VERIFIED (`txt/2408.15664.txt` pp. 2–5, 8) |
| C6 | Qiu et al. *Demons in the Detail*. ACL 2025. https://arxiv.org/abs/2501.11873 + Qwen blog https://qwenlm.github.io/blog/global-load-balance/ | LBL = N_E Σ fᵢPᵢ (eq. 2); micro-batch LBL eq. (3) averages per-group products; global-batch eq. (4)–(6) syncs only f̄ᵢ (an N_E-vector) and keeps per-group Pᵢ; Alg. 1 buffer for gradient accumulation; global-batch gives ≈0.1 PPL and ≈2 benchmark points and visible domain specialisation; micro-batch "is almost at the sequence level". | VERIFIED (`txt/2501.11873.txt:15-28, 149-232`) |
| C7 | Qwen3 Technical Report. https://arxiv.org/abs/2505.09388 §2 | "We adopt the global-batch load balancing loss (Qiu et al., 2025) to encourage expert specialization"; 128 experts, 8 active, no shared experts. Post-training use of the loss not stated. | VERIFIED (HTML) |
| C8 | Muennighoff et al. *OLMoE*. 2024. https://arxiv.org/abs/2409.02060 §4.1.6, §4.3, App. | LB loss weight 0.01, z-loss 0.001 in pretraining. **Adaptation**: "not using it leads to better performance (54.0 vs. 52.8 after SFT and 57.7 vs. 57.1 after DPO)"; "when measuring the load balancing loss … on our SFT data, we find that the loss actually decreases slightly during SFT (12.16 vs. 12.22)"; activation patterns after SFT/DPO "remain around the same" (App. G Fig. 33); "we do not use load balancing during adaptation". (App. text at line 3985 says the opposite — an internal inconsistency in the paper; §4.3 is the reasoned statement.) | VERIFIED (`txt/2409.02060.txt:1447, 1525, 1931-1950, 3985`) |
| C9 | Zhou et al. *Expert Choice Routing*. 2022. https://arxiv.org/abs/2202.09368 | Perfect balance by construction, no aux loss; Limitations: "takes in the past and future tokens to perform the top-k selection" — not applicable to autoregressive LMs as-is. | VERIFIED (`txt/2202.09368.txt:845-856`) |
| C10 | Gu et al. *Path-Constrained MoE*. 2026. https://arxiv.org/abs/2603.18297 | "removing LBL from Indep-MoE leads to more erratic training dynamics" (pretraining from scratch). | VERIFIED (`txt/2603.18297.txt:498-510`) |
| C11 | Zeng et al. *Rectify-Router*. 2024. https://arxiv.org/abs/2402.12399 App. C.3 | Removing the LB loss from a top-1 router degrades unless rectification is used; "still preferable to employ a load-balance loss". | VERIFIED (`txt/2402.12399.txt:1657-1721`) |
| C12 | Komatsuzaki et al. *Sparse Upcycling*. ICLR 2023. https://arxiv.org/abs/2212.05055 App. | Router randomly initialised N(0, 0.02); aux loss 0.01 for top-2 decoder routing. | VERIFIED (`txt/2212.05055.txt:872-880`) |
| C13 | HF transformers issue #44242 (Feb 2026) and TRL issue #1544 | HF: aux loss is *not* added unless `output_router_logits=True` regardless of `router_aux_loss_coef`; TRL DPO/KTO did not add the Mixtral aux loss (issue closed via PR #1765). | VERIFIED (fetched issue pages) |

### C.2 Exact gradient of the Switch/HF aux loss

With p = softmax(z) per token and P_e = mean_t p_e(t):

    ∂aux/∂p_e(t) = E · f_e / N_tot          (f treated as constant by autograd)
    ∂aux/∂z_j(t) = (E / N_tot) · p_j(t) · ( f_j − Σ_e f_e p_e(t) )

so each token's router-logit gradient is a *softmax-weighted deviation of the batch load
vector from its own expected load* — small (∝ 1/N_tot) and, for a balanced batch
(f ≈ k/E), proportional to `p_j(t)·(k/E − k/E) = 0`. This is why the loss is "quiet" once
balanced and why coefficients of 1e-3 (Mellum2 pretrain) / 1e-4 (Mellum2 SFT) suffice.
(Derivation mine; definitions VERIFIED from C1 and the venv file.)

### C.3 Coefficient practice (all VERIFIED unless noted)

| Model | Loss form | Coefficient | Source |
|---|---|---|---|
| Switch | per-layer, per-core batch | 1e-2 | C1 p. 8 |
| ST-MoE | + z-loss | c_B=1e-2, c_z=1e-3 | C2 |
| DeepSeek-V2 | per-sequence | 0.003 | C3 |
| DeepSeek-V3 | bias (γ=1e-3) + per-sequence | 1e-4 | C4 |
| OLMoE | batch | 1e-2 (+ z 1e-3) ; **0 during SFT/DPO** | C8 |
| Qwen3 | global-batch | not stated | C7 |
| **Mellum2 pretrain** | **global-batch (Megatron running average)** | **1e-3 (+ z 1e-3)** | F1 §3.4.5 |
| **Mellum2 SFT** | same | **1e-4** | F1 §5.1.2 |
| HF `MellumConfig` default | layer-pooled batch | 1e-3, off unless `output_router_logits` | venv `configuration_mellum.py:102-103` |

### C.4 Bearing on hypotheses

- **H1**: C1/C6 confirm the aux loss is the only cross-example term; z-loss (C2) is per-token
  and separable (note: Mellum2 used it in pretraining; HF does not implement it for Mellum).
- **H2**: per-sequence form is a first-class variant (C3, C4, C5 eq. 2 all *define* it per
  sequence); global vs sequence trade-off is a utility question with clear direction in the
  literature (sequence-level ⇒ less specialisation, C6).
- **H4**: C8 gives the strongest evidence that omitting the loss in fine-tuning a balanced
  checkpoint is harmless (even with router trained). C10/C11 caution is about *pretraining*
  from scratch / top-1 routers.
- Loss-free balancing (C4/C5) is a **non-loss** alternative that needs only a 64-dim
  sign(eᵢ) update per step from *previous-batch* counts — under DP this is a 1-bit-per-expert
  release of the sign of a noisy count deviation (randomised-response-able), and it does not
  touch gradients at all. Mellum2 authors "plan to switch to auxiliary-loss-free balancing
  in the next iteration" (F1 A.2). The checkpoint has no bias tensor (brief), but the
  appendix table lists "Router bias update rate 10⁻³" (`txt/mellum2_tr.txt:2579`) — see open
  questions.

---

## D. MoE fine-tuning practice: freezing routers, ESFT, MoE-LoRA, routing drift

### D.1 Sources

| # | Source | What it establishes | Status |
|---|--------|---------------------|--------|
| D1 | Wang et al. (DeepSeek). *Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning*. 2024. https://arxiv.org/abs/2407.01906 | ESFT trains only task-relevant experts (2–15 of 66 per layer, §6.1); "only the selected experts … can be updated; other experts and modules remain frozen" — router frozen. Table 3 ablation columns: non-shared experts / shared experts / non-expert params (gates, attention, embeddings): training relevant experts only (1.4B) gives 49.4/61.5; adding non-expert params (1.85B→2.7B) 49.8/60.7 → 50.8/60.3 — i.e., training gates+attention raises specialised ability slightly and lowers general ability slightly. ESFT-Token/Gate beat LoRA at every budget (Fig. 6). | VERIFIED (`txt/2407.01906.txt:484-496, 813-845, 876-935`) |
| D2 | Yao et al. *DenseMixer: Improving MoE Post-Training with Precise Router Gradient*. NeurIPS 2025. https://github.com/yaof20/DenseMixer | STE through top-k with all-expert forward gives a more precise router gradient; +46 % FLOPs, +9–29 % time; consistent gains on Qwen3-30B-A3B, Qwen1.5-MoE-14B, OLMoE-7B; "compatible with … LoRA". (OpenReview/Notion blocked; formula not read.) | PLAUSIBLE (README + search) |
| D3 | He et al. *Preserving Long-Tailed Expert Information in MoE Tuning*. 2026. https://arxiv.org/abs/2604.23036 | §3.1: routed expert i receives Θ(T·pᵢ) effective updates (eq. 4) — "gradient starvation is fundamentally a multiplicative bottleneck in both gradient magnitude and update frequency"; §6.2 cold experts have lower gradient norms; SFT baselines (ESFT, DenseMixer) "still suffer from the additional noise introduced by auxiliary balancing losses"; proposes aux-free bias sparsification + always-active condenser experts. | VERIFIED (`txt/2604.23036.txt:230-282, 357-361`) |
| D4 | *Defending MoE LLMs against Harmful Fine-Tuning via Safety Routing Alignment*. 2025. https://arxiv.org/abs/2509.22745 | Routing drift metric d = KL(σ(r(x∣w_align)) ‖ σ(r(x∣w_ft))); drift "even under benign fine-tuning" across OLMoE, Qwen1.5-MoE, DeepSeek-V2-Lite; router freezing not evaluated. | VERIFIED (HTML fetch) |
| D5 | PASs-MoE (2601.13020), MoE-Sieve (2603.24044), TT-LoRA MoE (2504.21190) | "Routing Gate Drift" measured in continual learning; routing-guided LoRA placement; frozen expert adapters + separately trained router. | PLAUSIBLE (snippets) |
| D6 | OLMoE §4.3 (C8) | Router *trained* during SFT/DPO without LB loss; routing distribution unchanged. | VERIFIED |
| D7 | Tholoniat (A1) | Router trained under DP without LB loss on switch-base-8. | VERIFIED |

### D.2 Bearing on hypotheses

- **H4** (attention-only LoRA, router+experts frozen): no paper evaluates exactly this on a
  64-expert code model; D1 Table 3 suggests router/attention training adds ~+1 specialised
  point at some general-ability cost (non-DP, DeepSeek-V2-Lite). Frozen router ⇒ routes are a
  fixed function of the (LoRA-modified) hidden states, so route flips can still occur via
  LoRA-induced hidden-state changes — H3's pinning idea remains relevant even with a frozen
  router.
- **H5** levers: D1 (train few experts), D2 (STE router gradient), D3 (starvation ⇒ cold
  experts see noise without signal under DP — a utility argument for grouping/skipping noise
  on cold experts, which is exactly A1's open problem). Under DP, *which* experts are hot is
  itself data-dependent; selecting them from private data must be accounted (ESFT-style
  selection from the fine-tuning set is a private query).

---

## E. Top-k discontinuity, gradient estimators, per-example gradient variance

### E.1 Sources

| # | Source | What it establishes | Status |
|---|--------|---------------------|--------|
| E1 | Wang et al. *ReMoE*. ICLR 2025. https://arxiv.org/abs/2412.14711 §3.2 | "a small weight update that alters the softmax result from (0.51,0.49) to (0.49,0.51) shifts the TopK output from (0.51,0) to (0,0.51), creating a discontinuity"; ReLU router eq. (5) continuous; adaptive L1 load regulariser eq. (6)–(7), (10)–(11). | VERIFIED (HTML) |
| E2 | ST-MoE §3.1 (C2) | bf16 router roundoff ⇒ instability; z-loss compresses logits. | VERIFIED |
| E3 | Mellum2 TR §5.2 (F1) | "for the same hidden state, the inference-time router may dispatch a token to a different expert than the trainer-side router, and the resulting logits and log-probabilities differ even though the weights are identical. BF16 numerical stability contributes additional noise." They mitigate with per-token IcePop truncation of the train/inference ratio. | VERIFIED (`txt/mellum2_tr.txt:1530-1545`) |
| E4 | Puigcerver et al. *Soft MoE*. ICLR 2024. https://arxiv.org/abs/2308.00951 | Fully differentiable slot mixing; explicitly not suited to autoregressive decoding (slots mix all tokens). | VERIFIED (`txt/2308.00951.txt:135, 245-270`) |
| E5 | Expert Choice (C9) | not causal. | VERIFIED |
| E6 | DenseMixer (D2) | STE top-k improves post-training. | PLAUSIBLE |
| E7 | Per-example gradient-norm distributions for MoE under DP | **No primary source found** (searches: "per-example gradient norm mixture of experts DP-SGD", "gradient variance top-k routing per-token gradient norm heavy tail"). Nearest: D3 §3.1/§6.2 (cold experts lower grad norm), "Normalization Layer Per-Example Gradients … Gradient Noise Scale" (2411.00999, dense). | GAP |

### E.2 Bearing on H3

- The discontinuity is real and specific to this checkpoint (E3). Pinning routes (compute
  indices once per example, ideally from FP32 logits as Mellum2's pretraining did:
  "Router precision FP32", `txt/mellum2_tr.txt:2581`) removes the *forward* discontinuity
  between vmap and oracle; the HF path computes `router_logits` in bf16 via `F.linear` and
  only the softmax in fp32 (`modeling_mellum.py:334-335`), so the top-k decision itself is
  made on bf16-rounded logits — a plausible flip source that fp32 logits would reduce.
- Whether pinning tightens the per-example gradient-norm distribution is untested in the
  literature; DenseMixer's finding that the router gradient through top-k is "imprecise"
  suggests routes are a large-variance component, but no norm distributions are reported.
- Any *learned* router modification (ReLU, soft, expert-choice) changes the model and is out
  of scope for faithfulness; STE (DenseMixer) only changes the backward and is usable inside
  `vmap(grad)` in principle, at +46 % FLOPs (already paid by Opaque's dense `opaque_moe`).

---

## F. Mellum2: what the primary documents say

### F.1 Sources

| # | Source | Status |
|---|--------|--------|
| F1 | Kojic et al. *Mellum 2 Technical Report* v1.0, May 2026. https://arxiv.org/abs/2605.31268 (PDF downloaded → `txt/mellum2_tr.txt`, 34 pp.) | VERIFIED |
| F2 | Model card README `JetBrains/Mellum2-12B-A2.5B-Base` (curl → `literature/mellum2_readme.md`) | VERIFIED |
| F3 | HF blog "Introducing Mellum2" and JetBrains blog (June 2026) | VERIFIED (no training details beyond F1) |

### F.2 Facts (all F1 unless noted; `txt/mellum2_tr.txt` line refs)

- Architecture "closely follows the Qwen3-MoE recipe" (:114); 64 routed experts, top-8, no
  shared expert, expert intermediate 896 (:395-396); MTP head α=0.1 removed at eval (:397-398);
  native context 8,192 → 131,072 by layer-selective YaRN on global-attention layers (:399, F2).
- **Balancing**: "global auxiliary load-balancing loss [18] with a coefficient of 10⁻³,
  combined with a router z-loss of 10⁻³ … The router operates in FP32 precision. We explored
  both per-sequence and global-batch balancing strategies and chose global-batch balancing for
  its flexibility, despite per-sequence balancing producing slightly better loss on short
  runs." (:806-809). Dropless routing, no capacity factor (:810). Appendix table: aux type
  "Global batch", coefficient 1e-3, z-loss 1e-3, "Router bias update rate 10⁻³", router FP32,
  token dropping disabled, grouped GEMM (:2573-2587).
- **Megatron semantics of "global"**: running average of per-expert token counts across
  microbatches, reset at gradient finalisation; per-microbatch loss computed against the
  running estimate; a cluster change (32→16 nodes) visibly lowered the reported loss purely
  through this accumulation artefact (:929-945). ⇒ f used in the loss was *lagged/partial*
  throughout pretraining.
- A.2: considered DeepSeek-V3 loss-free balancing ("matched or slightly improved expert
  utilisation"), kept aux loss for Qwen3-MoE ecosystem compatibility, "plan to switch …
  next iteration" (:2465-2471). A.4: 1e-2 better on short runs, chose 1e-3 "to avoid
  over-constraining expert utilization" (:2497-2498).
- **Pretraining shape**: seq 8,192, global batch 4,096 seqs (33.6M tokens/step), micro-batch 2,
  BF16+FP8 hybrid, Muon, expert parallelism 8 (:748-756, 799, 824).
- **FIM**: §3.2.2 — prefix/middle/suffix at two uniform cut points, sentinel tokens, 50/50
  PSM/SPM; FIM rate 50 % (phase 1, all data) → 10 % (phase 2) → 50 % code-only (phase 3);
  long-context stage injects repo-level FIM examples (:629-645, 1162-1167). Sentinel token
  strings are not given in F1/F2 (check tokenizer).
- **Long-context stage**: after ~30B tokens "the only quantity that continued to change
  meaningfully was the MoE router's load-balancing loss, which decreased substantially as the
  router adapted to the new sequence-length regime"; run extended to ~117B tokens "allowing the
  router to fully equilibrate before annealing" (:1171-1177). ⇒ the router *does* move under
  distribution shift, even from a balanced checkpoint.
- **SFT**: aux coefficient 1e-3 → **1e-4** "since the router is already well-balanced after
  pre-training and a smaller coefficient avoids over-constraining expert utilization on the
  narrower SFT distribution"; seq 131,072 packed, batch 64, gradient clipping 1.0, loss on all
  assistant turns (:1268-1272, Table 6).
- **RL**: router-induced train/inference route disagreement acknowledged (E3).

### F.3 Bearing on hypotheses

- **H1/H2**: the training-time objective really was CE (+MTP) + 1e-3·aux_global + 1e-3·z-loss;
  HF reproduces only CE (+ optional aux with a *different* pooling: HF pools f and P over
  layers into one product; Megatron computes per-layer losses — Switch semantics — and
  averages). Neither HF nor Opaque implements z-loss for Mellum. "Faithful to the real
  objective" therefore has a choice of targets; the SFT recipe (1e-4) is the one JetBrains
  used for the released Instruct/Thinking models.
- **H3**: FP32 router in pretraining vs bf16 logits in HF; route flips documented by the
  authors.
- **H4**: JetBrains' own fine-tuning kept the router trainable with a 10× smaller aux; they
  did not fine-tune with LoRA.

---

## G. DP accounting for heterogeneous per-step releases

### G.1 Sources

| # | Source | What it establishes | Status |
|---|--------|---------------------|--------|
| G1 | Koskela, Honkela. *Computing DP Guarantees for Heterogeneous Compositions Using FFT*. 2021. https://arxiv.org/abs/2102.12412 | Theorem 4: non-adaptive composition of k *different* mechanisms is tightly (ε,δ)-DP with δ(ε) from the convolution ω₁∗…∗ω_k of their PLDs; Algorithm 1 (FFT) with strict upper bound (Thm 7/8); §6.2 heterogeneous *Poisson-subsampled Gaussian* with ∼_R (remove/add) neighbouring: per-mechanism PLD ω(s) given in closed form, upper bound valid for heterogeneous σ, q. | VERIFIED (`txt/2102.12412.txt:136-160, 823-870`) |
| G2 | Gopi, Lee, Wutschitz. *Numerical Composition of DP*. NeurIPS 2021. https://arxiv.org/abs/2106.02848 | PRVs add under adaptive composition (Y = ΣYᵢ), so heterogeneous mechanisms compose by convolution; used by Tholoniat. | VERIFIED (`txt/2106.02848.txt:150-160`) |
| G3 | Doroshenko et al. *Connect the Dots*. PETS 2022. https://arxiv.org/abs/2207.04380 | Pessimistic PLD discretisation gives valid upper bounds after composition; underlies Google `dp_accounting` PLD `compose`. | VERIFIED (`txt/2207.04380.txt:16-26, 142-153`) |
| G4 | Andrew et al. Theorem 1 (B1) | Two Gaussian sum queries on the same sample ⇒ one Gaussian mechanism with joint sensitivity; closed-form noise trade. | VERIFIED |
| G5 | Ponomareva et al. (B2) | For BN-statistic + gradient releases: "accounting for adaptive streams Denisov et al. (2022)". | VERIFIED |
| G6 | Huang, Xie (B9) | Side releases that alter sampling break naive amplification accounting. | PLAUSIBLE |
| G7 | Opaque: `packages/opaque-accounting/src/opaque/api/accounting/core/composition/__init__.py:50-70` (`compose(left,right)` ≡ `left \| right`, "Multi-phase training with different noise"); `core/_base.py:123` ("`a \| b` (heterogeneous), `a * k` (homogeneous)"); `packages/opaque-engine/src/opaque/api/engine/noise_allocation.py:153-218` (`paired_noise_stddevs`: "MSE-optimal joint Gaussian allocation for the paired release"); `packages/opaque-dpsgd/src/opaque/api/dpsgd/clipping/_adaptive.py:176-200` (`fraction_noise_std` on the clipping-fraction release); `packages/opaque-dpsgd/src/opaque/api/accounting/dpsgd/mechanisms/_adaclip.py:131` (`adaclip` accounting factory). | Repo already has (i) heterogeneous composition, (ii) a paired-stream joint Gaussian allocation, (iii) an adaptive-clipping count release with its own accounting factory — the three pieces H2's side release needs. | VERIFIED (files read) |

### G.2 Correct accounting shape for H2 option 1 (same-step release)

Per step t with Poisson sample S_t, the mechanism releases
M_t(S_t) = ( Σ_{x∈S_t} clip_C(g(x; f̃_{t−1})) + N(0, σ_g²I),  Σ_{x∈S_t} clip_K(c(x)) + N(0, σ_f²I₆₄) ).
Because both sums are over the *same* S_t, follow G4: rescale to unit noise, concatenate,
and treat as **one** Poisson-subsampled Gaussian mechanism with sensitivity
S = (C²/σ_g² + K²/σ_f²)^½, i.e. effective noise multiplier z = 1/S — exactly what
`paired_noise_stddevs` computes for the first/second-moment pair. Then compose across steps
homogeneously (`* T`) or heterogeneously (`|`) if σ's change. Do **not** account it as
`poisson(gauss σ_g) | poisson(gauss σ_f)` per step: that models two *independent* samples
and is not what runs (this is my reasoning from G1/G4; the direction of the error is not
established here, so use the exact joint form). For option 2 (lagged f̃ only), the count
release of step t is post-processed into step t+1's loss, so per-step accounting is one joint
mechanism as above but with the gradient stream depending on f̃_{t−1}, which is fine under
adaptive composition (G2).

---

## H. Consolidated verdicts on H1–H5 from the literature (not experiments)

| Hyp. | Literature verdict | Key evidence |
|---|---|---|
| H1 | **Supported** — aux (and Megatron's z-loss, which HF/Opaque don't implement) are the only non-per-token terms; z-loss is per-token separable. | C1, C2, A1, F1 |
| H2 | **Algebra confirmed; mechanism has precedent; per-sequence variant is standard.** Zero-gradient caveat at f = k/E. Lagged f is the faithful choice. | C1 p. 8, venv impl., B1 Thm 1, B2 §5(b), F1 §3.6, C3–C6 |
| H3 | **Mechanism verified** (top-k discontinuity, bf16, this checkpoint's own route disagreement); **effect on per-example norm distribution: untested anywhere.** | E1, E2, E3, D2 |
| H4 | **Consistent with evidence**: OLMoE/Tholoniat drop aux at fine-tune with no collapse; Mellum2 SFT used 1e-4. Not a theorem — router does move under distribution shift (F1 long-context stage; D4 drift). | C8, A1, F1, D4 |
| H5 | **Open in the literature**: no MoE per-example gradient-norm study; Θ(T·pᵢ) starvation and A1's "isotropic noise on all experts" remark support the utility framing; ESFT/DenseMixer/sparse-DP-FT are the levers. Private expert selection must itself be accounted. | D3, A1, D1, A3 |

---

## I. Open questions / things I could not verify

1. Mellum2 appendix lists "Router bias update rate 10⁻³" next to the aux-loss rows
   (`txt/mellum2_tr.txt:2579`), yet A.2 says they *stayed with* the aux-loss formulation and
   the HF checkpoint has no expert bias. Either a Megatron default reported verbatim, or a
   bias was used and folded/dropped at export. Affects whether the released router was shaped
   by loss-free balancing.
2. Megatron computes the aux loss **per layer** (Switch semantics, averaged) with a running
   f; HF pools f and P over layers into one product. The two objectives differ (mean of
   products vs product of means). Which one "faithful" should target is a design decision.
3. DenseMixer's exact STE formula not read (OpenReview/Notion blocked); only README.
4. No source reports per-example gradient-norm distributions for MoE under DP (E7).
5. Whether amplify-then-compose over-/under-estimates privacy loss for two releases on the
   same Poisson sample: not established here; the exact joint form (G4) sidesteps it.
6. FIM sentinel token strings are not in the tech report or model card; must be read from the
   tokenizer.
7. TRL PR #1765's actual resolution (add aux to DPO or not) not read.

---

## J. File inventory

- `scratchpad/research/literature/mellum2_readme.md` — model card (curl).
- `scratchpad/research/literature/mellum2_tr.pdf`, `txt/mellum2_tr.txt` — tech report.
- `scratchpad/research/literature/pdf/*.pdf`, `txt/*.txt` — 32 primary PDFs and extractions
  (ids listed in §A–G tables) plus `luo2021.pdf`.
- `scratchpad/research/literature/extract.py` — pymupdf extraction script; `.pdfvenv/` — scratch venv.
