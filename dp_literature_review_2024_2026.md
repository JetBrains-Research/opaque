# DP Literature Review (2024–2026): Directions for Opaque

*Prepared April 2026 — Evgeny Grigorenko, Privacy Preserving Compute, JetBrains Research*

---

## Executive Summary

This document surveys the differential privacy literature from 2024–2026 across four axes — DP-SGD mechanics, privacy accounting, DP for LLMs, and systems/emerging directions — and maps findings to concrete integration opportunities for Opaque. Each section ends with an **Opaque Impact** assessment rated by effort and expected payoff.

---

## 1. DP-SGD Mechanics: Clipping, Noise, Optimizers

### 1.1 Clipping Innovations

**Automatic Clipping (AUTO-S / Auto DP-SGD)**
Auto DP-SGD (2024, OpenReview) eliminates the clipping threshold hyperparameter entirely by scaling gradients rather than clipping them, and introduces automatic noise multiplier decay. Reported: +6.7% accuracy and 94.9% privacy budget reduction on standard benchmarks.

**DP-SGD Without Clipping Bias** (ICLR 2024)
Error-feedback algorithm avoids constant clipping bias. Allows arbitrary clipping threshold selection independent of problem parameters — higher accuracy than standard DP-SGD at identical privacy.

**DP-SGD Without Clipping: The Lipschitz Neural Network Way** (ICLR 2024)
Lipschitz-constrained networks remove the need for clipping altogether, reducing both memory and compute overhead by bounding each layer's Lipschitz constant w.r.t. parameters.

**Adaptive Robust Clipping (ARC)** (ICLR 2025)
Per-sample adaptive clipping combined with layer-wise perturbation (DP-FedPUAC). Tighter privacy bounds from smaller average sensitivity.

> **Opaque Impact**: Auto-clipping is **high priority / low effort**. Opaque already has `clipping/` — adding an `auto` strategy that eliminates `max_grad_norm` tuning is a direct quality-of-life improvement for users. The error-feedback approach is a natural companion.

### 1.2 Correlated Noise / Matrix Factorization

**Banded Square Root Matrix Factorization (BSR)** (NeurIPS 2024)
Analytical expressions for SGD with momentum + weight decay with negligible overhead. Scales to 10⁶+ iterations and 10⁹+ parameters.

**DP-λCGD** (arXiv, Jan 2026)
Correlates noise only with the immediately preceding iteration via a pseudo-random generator. Zero extra memory over standard DP-SGD — no need to store past noise vectors. Empirically improves accuracy.

**Curvature-Informed Noise Correlation** (arXiv, Oct 2024)
First method to use model curvature (Hessian information) to improve cross-iteration noise correlation quality.

> **Opaque Impact**: Opaque already has `noise/` with MF support. **DP-λCGD is the most attractive addition** — zero memory overhead, drop-in replacement for independent Gaussian noise. BSR is relevant for long training runs (LLM fine-tuning). **Medium effort / high payoff.**

### 1.3 Optimizers Beyond DP-SGD

**DP-AdamW** (arXiv, Nov 2025)
Decoupled weight decay in the DP setting yields +15% on text classification, +5% on image classification vs. DP-SGD/DP-Adam. Note: bias correction (DP-AdamW-BC) actually *hurts* performance — the noisy gradient estimates interact badly with bias correction.

**DP-FTRL** (Google, 2024 — in production on Gboard)
Follow-The-Regularized-Leader avoids shuffling/subsampling requirements. Key insight: convergence depends on accuracy of *cumulative* gradient sums, not individual gradients. Uses negatively correlated noise for accurate sums.

**DP-MicroAdam** (arXiv, Nov 2025)
Lightweight adaptive optimizer for DP — frugal memory footprint.

> **Opaque Impact**: **DP-AdamW is high priority**. Most Opaque users are fine-tuning LLMs where AdamW is the default optimizer. Providing a DP-native AdamW (not just SGD + DP noise) would be a significant usability win. DP-FTRL is architecturally different but worth a research spike for federated scenarios.

### 1.4 Noise Multiplier Scheduling

**Auto DP-SGD** and **DP-SGD-global-adapt-V2-S** (2024–2025)
Step decay, exponential decay, and time decay schedules for the noise multiplier — analogous to learning rate scheduling. Privacy budget allocated more efficiently across epochs.

> **Opaque Impact**: **Low effort / medium payoff**. Add noise schedule support to the `noise/` module, mirroring PyTorch's `lr_scheduler` API.

---

## 2. Privacy Accounting

### 2.1 PLD Accounting Advances

**Laplace Transform Interpretation of DP** (FORC 2025)
Connects RDP and (ε,δ)-DP curves as Laplace/inverse-Laplace transform pairs. Unlocks frequency-domain reasoning about PLD properties — could enable faster, more numerically stable computation in Opaque's Rust PLD engine.

**Characteristic Function Accounting** (peer-reviewed)
Unifies RDP, privacy profiles, f-DP, and PLD via characteristic functions. Provides a single formalism for tight accounting with interpretable tradeoff functions.

> **Opaque Impact**: The Laplace transform approach could **improve numerical stability** in the Rust engine for long compositions. Worth a research spike. **Medium effort.**

### 2.2 Subsampling Amplification

**Privacy Amplification by Random Allocation** (NeurIPS 2025, Spotlight — Feldman & Shenfeld)
First theoretical guarantees and numerical algorithms for k-out-of-t random allocation. Bounds are at most (1+o(1))k/t times Poisson subsampling — without Monte Carlo overhead.

**Structured Subsampling for Time Series** (ICML 2025, Spotlight)
Privacy amplification via sequential, contiguous, and partitioned subsampling. Enables event- and user-level DP for forecasting models.

**Fixed-Size Without Replacement (FSwoR)** (arXiv, Aug 2024)
Improved RDP accountant for DP-SGD with fixed-size minibatches. Improves on best computable bound by a factor of 4×.

> **Opaque Impact**: Opaque's `sampling/` uses Poisson subsampling. The **FSwoR result is directly applicable** — most real DataLoaders use fixed-size batches, not true Poisson sampling. Integrating the tighter FSwoR bounds into the Rust accountant would give users more accurate (tighter) privacy guarantees for free. **High priority.**

### 2.3 Adaptive Composition & Privacy Filters

**α-Confidence DP Filters and Odometers** (2025)
Dynamic estimation of privacy loss per round using adaptive mechanisms and concurrent composition. Advances the pay-as-you-go framework.

> **Opaque Impact**: Useful for **production deployment** scenarios where training may be stopped early based on privacy budget consumption. Could integrate with Opaque's accounting as a `PrivacyFilter` wrapper. **Medium effort / medium payoff.**

### 2.4 f-DP Composition

**f-DP for Federated Learning** (OpenReview 2025, arXiv 2024)
f-DP accounting yields consistently tighter (ε,δ) bounds than RDP in federated settings. The CLT for privacy loss distributions means that for large compositions, f-DP gives near-optimal bounds.

> **Opaque Impact**: Adding f-DP as an alternative accounting mode (alongside PLD and RDP) in the Rust engine. **Medium effort, valuable for federated use cases.**

---

## 3. DP for Large Language Models

### 3.1 DP-LoRA & Parameter-Efficient Fine-Tuning

**LoRA Provides DP by Design via Random Sketching** (arXiv, 2024)
LoRA itself provides inherent privacy protections when adaptation matrices are frozen — noise variance decreases as a function of adaptation rank. This is a **theoretical validation of Opaque's core approach**.

**DP-QLoRA + Prefix Tuning** (2024)
Combines 4-bit quantized LoRA with prefix tuning under DP. Up to 75% memory reduction, 95%+ original performance at ε=3.

**ANADP: Adaptive Noise Allocation** (EMNLP 2024 Findings)
Allocates noise adaptively based on parameter importance. Narrows the gap between regular and DP fine-tuning significantly.

**Sparse DP Fine-Tuning** (arXiv, Mar 2025)
Uses private gradient information for parameter selection — better accuracy than full-model or existing sparse DP methods.

> **Opaque Impact**: **DP-QLoRA is the highest-impact LLM direction.** Opaque already has fused LoRA kernels — extending to QLoRA (4-bit) would enable DP fine-tuning of 70B+ models on a single H200. ANADP's per-parameter noise allocation could be integrated into the `noise/` module. **High effort / very high payoff.**

### 3.2 DP Synthetic Data Generation

**Google's DP Synthetic Training Data** (2025)
Inference-only approach: prompt LLMs with sensitive examples, aggregate predictions with DP. No fine-tuning required — generates thousands of synthetic datapoints with formal guarantees.

**Aug-PE** (ICML 2024, Spotlight)
API-based, training-free DP text generation using black-box LLM access. Competitive utility with DP fine-tuning baselines.

> **Opaque Impact**: This is a **new product direction** rather than a library feature. Could be offered as a higher-level utility built on Opaque's accounting primitives. **Exploratory.**

### 3.3 DP for Alignment (RLHF / DPO)

**Privately Aligning LMs with RL** (ICLR 2025)
End-to-end DP integration into RLHF: DP-SGD applied to SFT, reward modeling, and policy optimization stages.

**DP Steering for LLM Alignment** (ICLR 2025)
Private Steering for Alignment (PSA): activation editing with formal DP guarantees. Reduces membership inference risk without fine-tuning.

**AUP-RLHF** (2025)
User-level DP in preference alignment with adaptive sampling for reward and policy networks.

> **Opaque Impact**: As LLM alignment becomes standard, **DP-DPO support** would be a differentiator. The pipeline is: SFT with DP → reward model with DP → policy optimization with DP. Opaque's functional primitives (`clipped_grad`, noise injection) already compose well with custom training loops. **Medium effort — mostly documentation and examples.**

### 3.4 Zeroth-Order DP Methods

**DP Zeroth-Order Optimization** (USENIX Security 2025)
Significant memory reduction vs. first-order DP-SGD by estimating gradients from function evaluations. Enables DP fine-tuning on resource-constrained devices without per-example gradient computation.

> **Opaque Impact**: Interesting **alternative to vmap(grad())** for very large models where even LoRA per-example gradients are too expensive. Worth tracking but not an immediate integration candidate — it trades compute for memory. **Research watch.**

### 3.5 Industry Deployments

| Company | Deployment | Details |
|---------|-----------|---------|
| **Google** | VaultGemma (2025) | 1B-param Gemma 2 trained from scratch with DP. Largest DP LLM to date. |
| **Google** | Gboard (2024) | All production language models now use FL + DP. BLT-DP-FTRL for datacenter fine-tuning. |
| **Apple** | iOS 18+ (2024) | DP synthetic data for summarization. Siri LLM rebuild with on-device DP. |

---

## 4. Privacy Auditing

### 4.1 LLM-Specific Auditing

**Privacy Auditing of Large Language Models** (ICLR 2025 — Panda et al.)
Optimized canary design for MIA on LLMs. Achieves 49.6% TPR@1% FPR (vs 4.2% prior art) on Qwen2.5-0.5B. First nontrivial privacy audit of LLM with realistic guarantees (ε≈1 empirical, ε≈4 theoretical).

**The Canary's Echo** (ICML 2025)
In-distribution canary design (rare suffixes + typical prefixes) for auditing DP synthetic text generation.

**Sequential MIA** (arXiv, Feb 2025)
Uses sequences of model snapshots rather than single snapshots — identifies leaky targets and optimal insertion timing for tighter audits.

> **Opaque Impact**: Opaque already has `auditing/` with one-run estimators and loss-based attacks. **Adding optimized canary insertion** (Panda et al.'s approach) specifically for LLM auditing would make Opaque's auditing module state-of-the-art. **Medium effort / high payoff.**

### 4.2 One-Run Auditing Advances

**Enhancing One-Run Auditing with Quantile** (TPDP 2025)
Quantile-based improvements to Steinke et al.'s one-run methodology.

**Auditing f-DP in One Run** (NeurIPS 2024)
Extends one-run framework to f-DP — enables empirical bounds under functional DP definitions.

**Sequential Auditing** (arXiv, Sep 2025)
Practical sequential test for auditing DP guarantees of black-box mechanisms. Detects violations with orders-of-magnitude smaller sample sizes. Can identify DP-SGD violations in a single training run.

> **Opaque Impact**: Opaque's `auditing/one_run/` is directly based on Steinke et al. The **quantile enhancement and f-DP extension** are natural additions. Sequential auditing could be offered as a runtime monitor. **Low–medium effort.**

---

## 5. Systems & Distributed

### 5.1 FSDP + DP

Amazon's **fastDP** library demonstrates FSDP and DeepSpeed integration with DP. Google's distributed DP uses SecAgg protocols.

> **Opaque Impact**: Opaque currently supports DDP via `distributed/`. **FSDP support is the key missing piece** for training models that don't fit in single-GPU memory even with LoRA. This is architecturally significant — FSDP shards parameters, so per-example gradient computation needs careful handling. **High effort / high payoff for 13B+ models.**

### 5.2 Communication-Efficient DP

**CLFLDP** (2024) — layer-wise clipping with local DP reduces communication overhead.
**FedSA-LoRA-DP** (2025) — selective low-rank adaptation + DP for federated settings.

> **Opaque Impact**: Relevant if Opaque extends to federated scenarios. **Watch list.**

### 5.3 Regulatory

**NIST SP 800-226** (finalized March 2025)
Guidelines for evaluating DP guarantees. Emphasizes comprehensive parameter disclosure (ε, δ, composition method) and provides flowcharts and sample code.

> **Opaque Impact**: Opaque should ensure its accounting output is **NIST-compliant** — clearly reporting ε, δ, the accounting method used, subsampling rate, and composition assumptions. Could add a `PrivacyReport` dataclass. **Low effort / important for adoption.**

---

## 6. Prioritized Roadmap Suggestions

### Tier 1 — High Impact, Achievable Now

| Direction | Module | Effort | Key Paper |
|-----------|--------|--------|-----------|
| Auto-clipping (AUTO-S) | `clipping/` | Low | Auto DP-SGD (2024) |
| DP-AdamW optimizer | new `optimizers/` | Medium | DP-AdamW (2025) |
| FSwoR accounting bounds | Rust engine | Medium | arXiv:2408.10456 |
| Noise multiplier scheduling | `noise/` | Low | Auto DP-SGD / V2-S |
| NIST-compliant privacy reports | `accounting/` | Low | NIST SP 800-226 |

### Tier 2 — High Impact, Requires Research Spike

| Direction | Module | Effort | Key Paper |
|-----------|--------|--------|-----------|
| DP-λCGD correlated noise | `noise/` | Medium | arXiv:2601.22334 |
| DP-QLoRA (4-bit LoRA + DP) | `kernels/lora.py` | High | QLoRA+DP (2024) |
| LLM canary auditing | `auditing/` | Medium | Panda et al. (ICLR 2025) |
| One-run quantile enhancement | `auditing/one_run/` | Low | TPDP 2025 |
| FSDP support | `distributed/` | High | Amazon fastDP |

### Tier 3 — Strategic / Exploratory

| Direction | Module | Effort | Key Paper |
|-----------|--------|--------|-----------|
| DP-DPO/RLHF examples | `examples/` | Medium | ICLR 2025 |
| f-DP accounting mode | Rust engine | High | arXiv:2408.15621 |
| Laplace transform PLD | Rust engine | High | FORC 2025 |
| DP synthetic data generation | new module | High | Google (2025) |
| Zeroth-order DP | research | High | USENIX Security 2025 |
| Privacy filters/odometers | `accounting/` | Medium | α-confidence (2025) |
| BSR matrix factorization | `noise/` | Medium | NeurIPS 2024 |

---

## 7. Key Conferences & Venues to Track

- **TPDP 2025** (June 2–3, Google Mountain View) — Theory and Practice of DP
- **ICML 2025** — Multiple DP-for-LLM papers accepted
- **NeurIPS 2025** — Privacy amplification, auditing, benchmarks
- **ICLR 2025** — DP alignment, adaptive clipping, auditing
- **USENIX Security 2025** — Zeroth-order DP, practical attacks
- **PoPETs / PETS** — Privacy-focused venue, shuffling/composition results
- **ACL / EMNLP** — NLP-specific DP fine-tuning and synthetic data

---

## References (Selected)

1. Auto DP-SGD (2024) — OpenReview: QlFlo5533z
2. DP-SGD Without Clipping Bias — ICLR 2024
3. DP-SGD Without Clipping: Lipschitz Networks — ICLR 2024
4. Adaptive Robust Clipping (ARC) — ICLR 2025
5. Banded Square Root MF — NeurIPS 2024
6. DP-λCGD — arXiv:2601.22334 (Jan 2026)
7. Curvature-Informed Noise Correlation — arXiv:2510.05416
8. DP-AdamW — arXiv:2511.07843 (Nov 2025)
9. DP-FTRL — Google Research (2024)
10. Laplace Transform DP — FORC 2025 (arXiv:2411.09142)
11. Privacy Amplification by Random Allocation — NeurIPS 2025 (arXiv:2502.08202)
12. Structured Subsampling — ICML 2025 (arXiv:2502.02410)
13. FSwoR RDP Bounds — arXiv:2408.10456
14. LoRA Provides DP by Design — arXiv:2409.17538
15. ANADP — EMNLP 2024 Findings
16. Sparse DP Fine-Tuning — arXiv:2503.12822
17. Privacy Auditing of LLMs — ICLR 2025 (arXiv:2503.06808)
18. The Canary's Echo — ICML 2025 (arXiv:2502.14921)
19. Sequential MIA — arXiv:2602.16596
20. DP RLHF — ICLR 2025 (arXiv:2501.18532)
21. DP Steering for Alignment — ICLR 2025
22. VaultGemma — Google Research Blog (2025)
23. Aug-PE — ICML 2024 Spotlight
24. NIST SP 800-226 — March 2025
25. Almost Sure Convergence of DP-SGD — arXiv:2511.16587
26. DP Zeroth-Order — USENIX Security 2025
27. Enhancing One-Run Auditing with Quantile — TPDP 2025
28. Auditing f-DP in One Run — NeurIPS 2024 (arXiv:2410.22235)
29. Sequentially Auditing DP — arXiv:2509.07055
30. DP-FedAdamW — arXiv:2602.19945 (Feb 2026)
