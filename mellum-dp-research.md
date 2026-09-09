# Mellum2 under Opaque DP: representing the MoE training objective per example

Research snapshot: 2026-09-07 UTC. Branch `claude/mellum-dp-representation-r6slaz`, base commit
[`ef1abc5`](https://github.com/JetBrains-Research/opaque/commit/ef1abc58eb180250d4a2f930abdcd156d363f4e9)
(PR #980). Environment: torch 2.14.0 CPU-only, transformers 5.16.1, no GPU. No tracked file was
modified by this research.

Evidence tags: **VERIFIED** means the cited lines were read or the claim was established by a check that
was run during this research; **PLAUSIBLE** means derived or read but not executed end to end.
The experiment scripts, their outputs and the full design specification were recorded in this branch's
history under a companion `mellum-dp-research` directory and were removed once the feature landed. The
numbers quoted here come from those runs. Repo paths are relative to
`/home/user/opaque`; Hugging Face (HF) paths are under `.venv/lib/python3.11/site-packages/transformers/`.
Every magnitude that is not an accounting number, an allocator identity, a matrix-factorization (MF)
filter factor or a hockey-stick computation comes from a random-init tiny model on CPU. No trained-checkpoint
or GPU number exists yet; Section 6.2 says how to obtain them.

## Executive conclusion

The premise that Mellum2 cannot be fine-tuned faithfully under differential privacy (DP) because of batch
statistics or an unusable clipping norm is wrong. Mellum2's HF training objective has exactly one
term that is not per-example separable: the Switch-style load-balancing loss
`L_aux(B) = E * sum_e f_e(B) P_e(B)`, pooled over the whole batch and all 28 layers with one common
denominator (VERIFIED, `modeling_mellum.py:575-606`). Everything else in the forward (embedding,
RMSNorm, sliding and full attention, router, top-k, renormalisation, experts, LM head, per-token cross-entropy (CE))
is per-token or per-example up to floating point (including the microbatch-dependent attention-kernel
selection of Section 7.7), and Opaque's `vmap(grad)` reproduces the HF per-example gradient to
3.5e-7 relative L2 in fp32 across dense and grouped mixture-of-experts (MoE), both attention layer types,
padding, gradient checkpointing and chunked cross-entropy (VERIFIED by a vmap-vs-HF-loop comparison at
toy scale).

The load vector `f(B)` is argmax-derived, so its gradient is zero almost everywhere, and the gradient
of the batch aux loss is exactly the sum of per-example gradients of a surrogate in which `f(B)` is a
constant (Section 2.2; rel-L2 0.0 unpatched, 2.2e-7 on the patched model with ragged lengths,
VERIFIED). The only object a per-example pipeline lacks is that constant. The design obtains it as a
DP release of the per-example, token-weighted, centred load vector, carried as a second `PerGroup`
group of the same clipped pytree through a zero "probe" parameter and noised by the same Gaussian or
matrix mechanism as the gradient. Opaque's per-group allocator makes the joint release one
sensitivity-`1/nm` Gaussian (Section 2.7), so the accountant call is literally unchanged under both
stacks, and the whole price is a `sqrt(1+rho)` inflation of the gradient noise: at the preset-regime budget
share `rho = 0.02`, x1.010 gradient noise with epsilon unchanged (the allocator identity VERIFIED with the
real allocator; the epsilon and noise rows VERIFIED with the real accountant).

The validation phase ran the mechanism end to end on a tiny random-init Mellum with the real Opaque
primitives (`clipped_grad` with a direct-construction two-group `PerGroup`, `gaussian_noise`,
`mf_gaussian_noise`, `PoissonSampler`, `BMinSepSampler`, the real accountant) under both stacks: every
identity held, the DP load estimate recovered 74 % of the non-private oracle's balancing at toy scale,
and the accountant call was identical with and without the release (Section 6.1). The DDP-resume bug
of Section 7.1 was independently confirmed at the sampler level for both samplers, and its impact is
stronger than the design stated (Section 6.1.1, finding 6).

| requirement | how it is met | status |
|---|---|---|
| (a) provably DP, accounting matches what runs | one joint Gaussian / MF release per step with per-group bounds `(C_g, C_h)`; structural, never-active bound on the load group; Mahalanobis allocation `sum_i (C_i/sigma_i)^2 = 1/nm^2`; Poisson or b-min-sep subsampling of the one joint mechanism (shared coin); adaptive use of `f~_t` covered by adaptive composition; accountant unchanged | VERIFIED (mechanism). "As run" holds for single-process runs and for distributed data parallel (DDP) runs never resumed from a checkpoint, until the pre-existing sampler-key resume bug (Section 7.1) is fixed |
| (b) faithful to the real objective, batch term included | target is the HF logical-batch formula: token-weighted pooling over all layers, attention mask, executed routes, `alpha = model.config.router_aux_loss_coef` by default, presets pinned to Mellum2's own supervised fine-tuning (SFT) value 1e-4; the batch mean of per-example gradients equals `(T_tot/(B_bar T_bar)) * grad[alpha L_aux^HF]` exactly in direction for any length pattern (cosine 1.000000000000, VERIFIED); CE keeps Opaque's example-mean convention | VERIFIED (identity, direction); scale factor and CE convention stated, not hidden |
| (c) usable clipping norm and noise | the load leaf is its own group with a structural bound, so `C_g` is chosen as for any dense low-rank adaptation (LoRA) fine-tune; `rho = 0.02` (the preset-regime share) costs x1.010; at `alpha <= 1e-3` the aux term moves per-example norms < 0.1 % and is zero at balance | VERIFIED (toy), PLAUSIBLE (trained checkpoint, Section 6.2) |
| (d) numerically stable vs the non-DP HF path | precision-matched oracle; vmap vs eager vs module backward rel-L2 0.0 at equal precision on the patched model; residual bf16 drift is accumulation order inside HF's own batched-vs-loop spread; attention-kernel selection becomes a public property so a per-example gradient no longer depends on microbatch-mates | VERIFIED (toy); acceptance criteria in Section 6.2 |

Caveats. Every utility magnitude is toy-scale; Section 6.2 defines the GPU validation that settles
(c) and (d) on the real checkpoint. A pre-existing DDP + checkpoint-resume bug restores rank 0's
sampler key on every rank and voids the subsampling amplification of every release of the trainer
after the resume point (Section 7.1, P0, not caused by this design). `DPTrainer` runs the dense
every-token-through-every-expert MoE kernel by default on every host including CUDA, about five times
the forward FLOPs of the routed model (Section 7.2). The band-MF noise multiplier of the preset could
not be calibrated in this container; the design carries the deterministic bracket `[0.5622, 1.544]`
and uses the upper end as a conservative, valid multiplier where a number is needed (valid because the
un-amplified `mf_gaussian(nm, strategy)` accountant is itself a legitimate, looser guarantee at `nm = 1.544`;
the amplified `b_min_sep(...)` accountant, if it finishes, reports something at or below `epsilon = 3` there,
and it was not evaluated in this container). The design of Sections 4 and 5 has since been implemented
on this branch (`DPTrainer` exposes `router_load_release` and its companion `router_load_*` arguments, and
the mechanism page `docs/mechanisms/dp-sgd/moe-load-balancing.md` documents it), so those sections describe
shipped code rather than a proposal.

## 1. What "kinda works" meant: the exact divergences between the DP path and the HF path

BATCH-COUPLED: semantics differ between batched HF and per-example vmap. NUMERICAL: same mathematics,
floating-point differences. STRUCTURAL: different code path, same mathematics.

| # | item | class | magnitude / status | evidence |
|---|---|---|---|---|
| 1 | Load-balancing aux: `f`, `P` pooled over batch and all layers, denominator `L * T_tot`, `sum_e f_e = k = 8` | BATCH-COUPLED, inseparable as written | the only inseparable term; `f` has zero gradient | `modeling_mellum.py:575-606, 692-700` (VERIFIED) |
| 2 | CE reduction: HF token-weighted (`num_items_in_batch`); Opaque per-example token mean, equal example weights | BATCH-COUPLED, separable | 0.40 rel-L2 on the aggregate gradient of a 15/13/10/7-token toy batch; coincide iff equal lengths; representable with a public constant | measured on the toy batch (VERIFIED) |
| 3 | HF Trainer with accumulation `G` optimises `CE_tokenmean(logical batch) + coef * sum_mb aux(mb)`: per-microbatch `f`, effective coefficient `G * coef` | BATCH-COUPLED (HF artefact) | rel-L2 0.0 against that formula, 0.365 / 0.479 against the microbatch-mean / logical-batch aux | toy HF Trainer accumulation run (VERIFIED); `trainer.py:1961-1963` |
| 4 | HF's aux recomputes top-k from a bf16 softmax; the forward uses fp32 | NUMERICAL (HF-internal) | 0.03 to 1.7 % of tokens get a different aux top-k set; 4 to 7 % exact bf16 ties at k/k+1; `f` deviates at most 1e-3 in units of `k/E` | bf16-vs-fp32 aux top-k count on the toy model (VERIFIED); `modeling_mellum.py:584` vs `:335` |
| 5 | Experts kernel: HF `grouped_mm` (bf16 accumulate, not vmappable) vs `opaque_moe` (fp32 accumulate, custom vmap rules) | STRUCTURAL + NUMERICAL | fp32 3.5e-7; bf16 0.45 to 0.5 % with **zero** route flips, inside HF batched-vs-loop 2.7e-3 and below HF bf16-vs-fp32 1.1e-2 | vmap-vs-HF-loop comparison in fp32 and bf16 on the toy model (VERIFIED) |
| 6 | bf16 routing: tokens change their top-k set between a bf16 and an fp32 forward, invariant to router sharpness | NUMERICAL, precision not vmap | about 1 %/layer (top-2 toy); 46 to 54 changed top-8 sets per 1024 rows per layer for the stock bf16 router (39 to 56 with the fp32-logit router) (top-8 toy); flips give about 8x excess error on router/expert gradients, +0.2 pp on attention-only | route-flip counts between bf16 and fp32 forwards, stock and fp32-logit routers (VERIFIED toy) |
| 7 | Scaled dot-product attention (SDPA) `is_causal` fast path chosen from `physical_mask.all()` over the whole physical microbatch (PR #980) | NUMERICAL, microbatch-coupled | one padded example switches every mate to the materialised-mask kernel; exact in fp32; systematic bf16 dependence on batch composition | `runtime/masking.py:195-216` (VERIFIED) |
| 8 | `DPTrainer` default `use_performance_kernels=False` selects dense `Opaque_MoE` everywhere incl. CUDA; grouped flag captured by the first class-level patch per process | STRUCTURAL (performance) | 8x routed expert FLOPs, about 5x forward FLOPs | `_training_arguments.py:436`, `_factory.py:319`, `_router.py:59-92` (VERIFIED) |
| 9 | HF `output_router_logits=True` fails under vmap with a mask (`scatter_add_`) and bypasses chunked CE | STRUCTURAL | 403 MB of logits per example on the fallback | `modeling_mellum.py:596-598`, `cross_entropy.py:212-233` (VERIFIED) |
| 10 | Fully masked query rows under left padding: Boolean SDPA gives zeros, additive masks give `mean(V)`; HF disagrees across its own backends | NUMERICAL, convention | never arises: both repo collators right-pad | masked-row behaviour checked per SDPA backend; both repo collators read (VERIFIED) |
| 11 | CUDA autocast staging: HF `grouped_mm` upcasts to fp32 masters, Opaque casts to the autocast dtype | NUMERICAL | 5.5e-3 on CPU autocast, equal to HF's own backend gap; not applicable to the pure-bf16 presets | CPU autocast comparison against HF (VERIFIED CPU, PLAUSIBLE CUDA) |

PR #978 installed the chunked linear cross-entropy path (no per-example 98304-vocabulary logits).
PR #980 kept upstream RMSNorm (the Triton bf16 reduction order flipped routes), kept HF's router-loss
semantics, aligned chunked-CE precision staging with eager bf16, and took the SDPA causal fast path for
fully valid vmapped microbatches (item 7). Its body reports "oracle gradient relative L2 from 29.26 %
to approximately 1.3 %". The figure is untraceable: no tracked script mentions an oracle for Mellum
(`grep -rn oracle` finds only kernel-level oracles), and the bf16-vs-fp32 pair and the DP-vs-HF pair at
equal precision have different flip rates, so the oracle must be precision-matched before a rel-L2 means
anything. The mechanism (kernel parity on all-valid
microbatches) is VERIFIED from the diff; the number is not. Section 6.2 therefore fixes the oracle
(same process, same patched module, precision-matched, HF eager per-example loop) and reports the
route-flip counter separately from the rel-L2.

Items 1 to 4 show the batch coupling is one term whose gradient is separable; items 5 to 7 show the
residual drift is floating point, not routing. That is the whole content of "kinda works".

## 2. The mathematics

### 2.1 The batch aux loss as HF computes it

With `output_router_logits=True`, HF collects every layer's router logits and computes
(`modeling_mellum.py:575-606`, VERIFIED):

```
f_e(B) = (1/(L T_tot)) sum_l sum_{x,t} m_{x,t} 1{e in S^l_{x,t}}      P_e(B) = (1/(L T_tot)) sum_l sum_{x,t} m_{x,t} p^l_{x,t,e}
L_aux(B) = E * sum_e f_e(B) P_e(B),      loss += router_aux_loss_coef * L_aux(B)
```

`m` is the attention mask the caller passed (float-cast, `:581`), `T_tot = sum_x T_x`; `total_rows`
accumulates inside the layer loop (`:600`), so the denominator is `L T_tot` and `sum_e f_e = k = 8`. `f`
and `P` are pooled over layers before the product. The coefficient
0.001 is the pretraining value; Mellum2's own SFT used 1e-4 (technical report section 5.1.2, VERIFIED). Only `P` is differentiable (Switch Transformer eqs. (4) to (6)).

### 2.2 The separability identity

**Theorem.** Wherever the top-k sets are locally constant in `theta` (almost everywhere),

```
grad_theta L_aux(B) = sum_{x in B} grad_theta S(x; f~) |_{f~ = f(B)},     S(x; f~) = E (T_x/T_tot) sum_e f~_e P_e(x),
P_e(x) = (1/(L T_x)) sum_l sum_t m_{x,t} p^l_{x,t,e}.
```

**Proof.** `f` is piecewise constant in `theta`, so `grad L_aux = E sum_e f_e(B) grad P_e(B)`, and
`P_e(B) = sum_x (T_x/T_tot) P_e(x)` is linear in the per-example means. Substitute.

VERIFIED: rel-L2 0.0 on an unpatched toy; 2.4e-7 to 3.3e-7 on three batches including ragged lengths
(float64); 2.2e-7 on the patched model with ragged lengths (feasibility check F).

### 2.3 The surrogate gradient is proportional to imbalance

`sum_e P_e(x) = 1`, so `grad S(x; f~) = E w_x sum_e (f~_e - k/E) grad P_e(x)`. The signal is the
imbalance; a uniform `f~` gives an identically zero gradient (measured `||grad|| = 2.8e-8`). This
is exactly what the real objective does at balance, and it is why "the aux is negligible" (H4) held
in the toy: 2.4e-4 of the CE gradient at balanced random init, 5x larger under induced imbalance
(VERIFIED at toy scale). "Negligible" is not a design assumption anywhere below.

### 2.4 Why the per-example aux is a different regulariser

`E sum_e f_e(x) P_e(x)` with the example's own load penalises within-document imbalance. A single code
file naturally routes to few experts; the batch loss allows that as long as other files compensate,
the per-example loss pushes every document toward uniform expert use (an anti-specialisation
pressure). Cosine to the batch aux gradient: 0.26 to 0.39 at balanced init, 0.76 to 0.79 under induced
imbalance (VERIFIED toy). DeepSeek-V2/V3 and Wang et al. define the per-sequence form as a
distinct, deliberately weak complementary loss. It is an opt-in, never labelled faithful.

### 2.5 Token weighting for ragged rows

The `mellum2-kstack` preset tokenises with `truncation=True, max_length=args.max_seq_len` and no packing
(`examples/train_dpftrl.py:1090-1094`; the preset pins `max_seq_len = 1024` at `:889`) and `DataCollatorForLanguageModeling` pads (`:1130-1135`):
rows are ragged (VERIFIED). The weight `T_x/T_tot` in 2.2 is a batch quantity; replace it by the
public constant `w_x = T_x/T_bar` (`T_bar` public, default `T_max`, better the mean length of a
held-out split). Then, by linearity in `w_x`,

```
(1/B_bar) sum_{x in B_t} grad l_x = (1/B_bar) sum_x grad CE_x + alpha (T_tot/(B_bar T_bar)) grad L_aux^HF(B_t)|_{f = f~_t}
```

exactly: cosine 1.000000000000, scale `T_tot/(B_bar T_bar)` to six digits (VERIFIED by a numerical identity check).
The direction is HF's for every batch. The scale factor has mean `T_bar_true/T_bar` and relative sd
about 6 %, the same kind of factor Poisson's `|B_t|/B_bar` already puts on the whole gradient; it is 1
under packing or with `T_bar = T_bar_true`. An equal-weight surrogate is 15.6 % rel-L2 off on a
16/12/10/16 length spread (feasibility check F) and is rejected. CE keeps Opaque's equal-example-weight
convention (item 2 of Section 1), pre-existing and stated on the mechanism page.

### 2.6 The load vector, centring, structural bounds

Per example and layer, `h^l_e(x) = (1/T_x) sum_t m_{x,t} 1{e in S^l_{x,t}}` has `0 <= h^l_e <= 1`,
`sum_e h^l_e = k`, hence `||h^l||^2 <= k`. Centred, `d^l = h^l - (k/E) 1`, `sum_e d^l_e = 0` and
`||d^l||^2 <= k(1 - k/E) = 7`. Over `L` layers with `w_x <= T_max/T_bar`:

```
per layer:  ||w_x d^{(L,E)}(x)||_2 <= Delta_L = (T_max/T_bar) sqrt(k L (1 - k/E)) = 14.0   (L = 28, T_bar = T_max)
pooled:     ||w_x d(x)||_2        <= Delta_h = (T_max/T_bar) sqrt(k (1 - k/E))   = 2.6458
```

The bounds are structural (given the binarised mask, the capture-count assert and the guard `g` of
Section 4; a duplicate capture would make the clip fire at 2.035 `C_h` and bias the release 2x): no clipping
is needed, the `PerGroup` entry is a bound and never an active clip, the release is unbiased. Adversarial
routing of every token to the same `k` experts through the real clipper attains 0.99999896 of the bound and
never exceeds it, for any non-negative mask
(bound attacks A1 to A5 through the real clipper, VERIFIED). Adjacency is add/remove (repo default); the divisor is the public expected batch size
`B_bar`, never the realised batch. Replace-one: gradient bound `2 C_g`; tight load bound
`(T_max/T_bar) sqrt(2 k L)` (pooled `sqrt(2k) = 4.000`, attained, VERIFIED, attack A4). The general
replace-one form of the per-layer centred-load bound is `(T_max/T_bar) sqrt(2 min(k, E - k) L)`, equal to
`sqrt(2 k L)` for `E >= 2k` (attained by disjoint expert sets); `sqrt(2 k L (1 - k/E))` is not a bound (5.29 against
5.66 attained at `E = 64, k = 8, L = 2`; Section 6.1.1, finding 8, check K3). Centring buys the 6.5 %
smaller bound and `sum_e = 0` structurally, so no renormalisation of a noisy sum is needed. A fully
masked row (`T_x = 0`) contributes `d = 0`, `P = 0` explicitly; a bare `clamp(min=1)` denominator would
give `d = -(k/E) 1`.

### 2.7 The joint Gaussian release and the Mahalanobis constraint

Each step releases one Gaussian on `[sum_x clip(g_x) ; lambda sum_x w_x d^{(L,E)}(x)]` with diagonal
covariance `(sigma_g^2 I, sigma_h^2 I)`. Whitening is a bijection, so the whitened statistic has the
same privacy; its add/remove L2 sensitivity is `sqrt(C_g^2/sigma_g^2 + C_h^2/sigma_h^2)`, and the
mechanism is a sensitivity-1 Gaussian at multiplier `nm` iff

```
(C_g/sigma_g)^2 + (C_h/sigma_h)^2 = 1/nm^2.
```

(Dong, Roth, Su Thm 2.7 on the whitened statistic; Zhu, Dong, Wang Def. 7 for the dominating pair;
precedent Andrew et al. 2021 Thm 1: a clipped-count release and the gradient release on the same
sample are pre- and post-processing of one query with joint sensitivity.) The naive "`nm C_g` on the
gradient, `nm C_h` on the load" gives `2/nm^2`: it is `gaussian(nm/sqrt(2))`, an under-noising
(by the constraint above). Opaque's allocator implements `sigma_i = nm sqrt(C_i sum_j C_j)`
(`noise_allocation.py:103-110`, VERIFIED), which satisfies the constraint with equality. With
`rho = C_h/C_g`:

```
sigma_g = nm C_g sqrt(1 + rho) / B_bar        (every gradient leaf)
sigma_h = nm C_h sqrt(1 + 1/rho) / B_bar      (each of the L*E probe entries)
```

The identity prints 1.000000 at six `rho` with the real allocator (VERIFIED). The
gradient pays `sqrt(1+rho)`; the accountant sees `gaussian(nm)`.

### 2.8 Subsampling and adaptivity

Both halves are computed on the same sampled batch, so the subsampling applies to the one joint
mechanism. Poisson subsampling of the joint mechanism is Feldman and Shenfeld Lemma 3.2 / Thm 3.3 as
implemented in `src/amplification/poisson.rs:15-42` (the code comment cites Thm 3.3 and Algorithms 8 and 9;
Lemma 3.2 is this research's own reading of the paper). Under b-min-sep the joint
whitened stream goes through the unchanged `b_min_sep(mf_gaussian(nm, strategy))` accountant (Section 2.9),
whose amplification theorem (Dong and Ganesh 2026, Section 9) was not re-fetched from the primary text. The
Poisson lemma needs each record's inclusion coin independent of every other record's; Section 7.1 shows
where the trainer violates it today. `f~_t` is a function of previous outputs, the same dependence `l_x` already has on `theta_t`;
Zhu, Dong, Wang Thm 10 charges nothing for it, so lagged and filtered use is free. The opt-in
independent forward-only draw (DP-SGD only) is two mechanisms with fresh coins and noise composed
adaptively, `poisson(gaussian(nm), q) * T | poisson(gaussian(c nm), q2) * (T/m)`, `epsilon = 3.003` at
`c = 2, m = 4` (VERIFIED with the real accountant). Its sampler key is rank-domain-separated,
its noise key is shared across ranks and applied after the all-reduce, and both keys are serialised in the
sidecar and re-folded by rank on resume (the five conditions stated in the privacy statement of Section 4).
An empty Poisson batch releases `0 + noise` on the probe and keeps the MF column index contiguous.

### 2.9 DP-FTRL

The probe is a second `PerGroup` group of the same clipped stream. The constant-max_norm latch accepts
the constant two-group `PerGroup` (`_engine.py:473-517`; VERIFIED with the real primitives);
`base_stddev = per_group_noise_stddev(max_norm, nm)`; the correlated noise `C^{-1} Z` is applied
leaf-wise, probe included; realised per-step sigma is `base * ||row_t(C^{-1})||`, 1.431 for
`band_mf_strategy(bands=64, momentum=0.95)`, stationary from `t ~ 7` (VERIFIED from the instantiated strategy's inverse row norms).
Correctness is Denisov et al. Thm 2.1 on the per-group-whitened stream: the participation-pattern
sensitivity is homogeneous of degree 1 in the row bound, so for a shared pattern `pi`,
`sum_g ||C(G_g - H_g)||_F^2/sigma_g^2 <= s(pi)^2/nm^2`, and the sup over `pi` is `sens(C)^2/nm^2`,
the same scalar privacy loss distribution (PLD). The pattern is shared across groups (same example, same step), so the gradient's
`min_sep` and `max_participations` apply to the probe. The accountant
`b_min_sep(mf_gaussian(nm, band_mf_strategy(64, 0.95, lr_schedule)), n_steps, p0)` is unchanged. The
noise variance of any fixed linear filter `F` of the stream is the deterministic `sigma_h ||row_t(F C^{-1})||`,
computed at setup from the instantiated strategy's inverse coefficients, never a dense solve.

### 2.10 Post-processing

All public and free. From the noised probe leaf `y_hat_t in R^{L x E}` (already `/B_bar`,
rank-identical under DDP):

```
1. d_hat^{(L,E)} = y_hat_t / lambda                 # signal + N(0, (sigma_h/lambda)^2) per entry
2. d_hat = mean_l d_hat^{(L,E)}                     # pooled; per-entry noise (sigma_h/lambda)/sqrt(L) = a direct pooled release
3. d_hat <- d_hat - mean_e(d_hat)                   # sum-zero projection; the signal already sums to 0
4. m_{t+1} = beta m_t + (1-beta) d_hat  (exponential moving average, EMA);  d~_{t+1} = m_{t+1}/(1-beta^{t+1});  s_{t+1} = (sigma_h/(lambda sqrt L)) phi_{t+1}/(1-beta^{t+1})
   phi: DP-SGD  phi_{t+1}^2 = beta^2 phi_t^2 + (1-beta)^2 (E-1)/E, stationary sqrt((1-beta)/(1+beta)) sqrt((E-1)/E);  MF  phi_t = ||row_t(F C^{-1})|| (Section 2.9)
5. d~+ = 0 if ||d~||^2 < c E s^2 (dead zone, c = 2), else d~ (1 - E s^2/||d~||^2)   (positive-part James-Stein)
6. f~_{t+1} = clamp(k/E + d~+, 0, 1)
7. D_{t+1} = max_e |d~_e|/(k/E); D^l likewise from the per-layer EMA     (public monitors)
```

The per-layer carrier costs nothing for the pooled estimate (0.04149 either way, VERIFIED by closed form and Monte Carlo) and gives
per-layer entries at `x sqrt(L) = x5.29` per-entry noise for diagnostics. The projection removes the
pure-noise mean component. Bias correction: the uncorrected EMA from `m_0 = 0` has gain 0.634 /
0.866 / 0.951 / 0.990 at `t = 100/200/300/460` (VERIFIED numerically),
a surrogate 37 % too weak early on; the corrected estimate is honestly noisier at the start (corrected std
0.1032 at `t = 100` against the stationary 0.0710), which the dead zone accounts for. The dead
zone exists because plain James-Stein zeroes only when `||d~||^2 < (E-1) s^2`, which pure noise exceeds
about 48 % of the time; with `c = 2` a pure-noise step passes with probability 4.2e-6 (exact chi-square
tail, 63 dof), 0.07 expected false passes over 15 625 steps (VERIFIED numerically). That
false-pass rate `P(chi^2_{E-1} > 2E)` is `E`-dependent: 4.2e-6 at `E = 64` but about 5e-2 at `E = 8`
(Section 6.1.1, finding 2), and with bias-corrected `s_t` the first few steps pass more easily. In signal units the
dead zone engages below a per-coordinate RMS imbalance `delta < sqrt(c) s/(k/E)`: 0.033 (DP-SGD) and
0.012 to 0.032 (band-MF) at `rho = 0.02`. The clamp is inactive unless an expert is dead or hot
(`|d~_e| > k/E`, at least 40 sigma). The monitor trips when `D_t > 0.5` on two consecutive
evaluations, 21 sigma (DP-SGD) or 22 to 60 sigma (band-MF) at `rho = 0.02`; the per-layer monitor `D^l_t`,
at x5.29 per-entry noise, is still 4 sigma at the 0.5 threshold, enough to locate a collapsing layer but not
for fine decisions. Switching `alpha` from 0
to its configured value on a trip changes an adaptive row only (Denisov Thm 2.1, ZDW Thm 10): no
clipping, noise, accountant or MF-latch change.

## 3. What was tried and refuted

- **fp32 router as a flip fix.** REFUTED at toy scale. Flips originate in the bf16 hidden states
  entering the router (fixed relative resolution), not in the logits' rounding: the fp32-logit router
  changes 51/56 and 39/43 top-8 sets per 1024 rows per layer where the stock bf16 router changes
  46/54 and 46/47 (flip counts reproduced bit-for-bit by an independent re-run, VERIFIED). The pair that
  matters for (d), Opaque bf16 `vmap(grad)` vs HF eager at equal precision, has 0 flips and 0.0 rel-L2
  on the patched model (feasibility check B). The fp32 router ships only as an opt-in, documented as
  pretraining-faithful (Mellum2 pretrained with an FP32 router) and tie-robust, cost 0.30 % of routed
  expert MACs, default off because adapters served through stock HF run bf16 routes.
- **Per-example aux as a faithful stand-in.** REFUTED (Section 2.4).
- **Naive per-group noise `sigma = nm C_g` per group.** REFUTED: `gaussian(nm/sqrt(2))` (Section 2.7).
- **The HF-Trainer-realised objective as target.** REJECTED: per-microbatch `f` with effective
  coefficient `G alpha` is an artefact of `trainer.py:1961-1963`; the DP path has no accumulation
  (`gradient_accumulation_steps` is hard-wired to 1, `_training_arguments.py:1314-1316`, VERIFIED),
  so it is more faithful to the logical-batch formula than HF Trainer itself. The Megatron per-layer
  running-average `f` is rejected as target on faithfulness grounds only (mean of per-layer products
  is not reproducible from the HF artefact); its lagged character is adopted and a per-layer surrogate
  is an opt-in.
- **"A per-layer release costs x5.29 relative noise at equal privacy."** REFUTED for the release: at
  the same `rho`, `lambda_L = rho C_g/sqrt(7L)` is `sqrt(L)` smaller and the layer mean divides the
  noise by exactly `sqrt(L)`; pooled per-entry noise 0.04150 vs 0.04152 Monte Carlo, closed forms equal
  (VERIFIED). The x5.29 applies to per-layer entries used on their own.
- **Randomised response on `sign(d)`**: per-step `epsilon = 64 ln((1-p)/p) = 12.8` at `p = 0.45`,
  dominated. **Independent draw with a 4x batch** (`q2 = 4q, c = 1`): `epsilon 5.30`. **Amortised
  same-batch release every `m` steps**: the same budget in bursts. **Public-data `f~`** (Davody et al.;
  Ponomareva et al. section 5, option (a)): zero cost but the public distribution's imbalance; opt-in. **Loss-free
  balancing** (Wang et al. Alg. 1; DeepSeek-V3 eq. (16)): free post-processing of the same release, but
  the checkpoint's router has no bias tensor, so it is an architecture extension with a serving
  patch; out.
- **Count-vector sensitivity `L T_x k`**: loose by `sqrt(k)` (top-k assigns distinct experts), so the
  stated 25 % relative noise was 8.8 % (corrected during review); superseded by the structural fraction bound.
- **"KStack is public, so the calibration pass is free."** REJECTED: a non-DP pass over protected
  rows is a query on the protected set; `C_g` and `T_bar` are calibrated on a held-out shard disjoint
  from the training rows, or accounted as one quantile release.
- **The "packed presets" premise.** REFUTED: the kstack preset is ragged (Section 2.5). Packing is an
  option, off by default, because it changes the protected unit to a packed 1024-token row that may
  contain pieces of several files.
- **H3** (route flips explain residual DP-vs-oracle drift): undecided as posed, because the bf16-vs-fp32
  pair and the DP-vs-HF pair have different flip rates and only a precision-matched oracle can decide it;
  decided pieces: bf16-vs-fp32 flips dominate the excess error on router and expert gradients
  (12.9 % / 11.1 % to 1.7 % / 1.5 % when pinned, VERIFIED toy); the DP-vs-HF pair at equal precision has
  no flips; 28-layer behaviour is acceptance test A1. **H5** (per-example expert sparsity at `T = 1024`):
  open; the only measurement used a 6-token vocabulary, which bounds the number of distinct router inputs
  and so cannot decide the question; with frozen experts no expert gradient
  exists, so H5 matters only for the experts-trainable variant.

## 4. The design (condensed)

**Objective.** With `sg` = `detach`, probe parameter `z in R^{L x E}` kept at zero, public
`alpha, lambda, T_bar, f~_t`:

```
l_x(theta; f~_t) = CE_x(theta) + alpha (S_x - sg[S_x]) [+ zeta (Z_x - sg[Z_x])] + < z, lambda w_x d^{(L,E)}(x) >_sg
S_x = E w_x sum_e (f~_{t,e} - k/E) P_e(x; theta),      w_x = T_x/T_bar
```

The surrogate and the optional router z-loss `Z_x` (ST-MoE section 3.1, eq. (5), `zeta = 0`) enter
value-neutrally: the loss value is exactly `CE_x`, so the trainer's un-noised logged loss carries no
router-derived term; the gradients are `grad CE + alpha grad S` on the model leaves and
`lambda w_x d^{(L,E)}(x)` on the probe. Routes are computed per example inside vmap from the current
model (no pinning); `f` uses the executed routes (same fp32 softmax, same `topk`), which differs from
HF's bf16 recomputation by at most 1e-3 in units of `k/E`. The mask is the collator's attention mask,
binarised (`m = attention_mask != 0`); `None` counts every position. `h` is normalised by
`len(router_logits) * T_x` and `len(router_logits) == num_hidden_layers` is asserted, otherwise a duplicate
capture would double `sum_e h`; with the `(L, E)` carrier a duplicate capture also fails the reshape loudly.

**Per-step mechanism, both stacks.** (1) sample `B_t` (Poisson at `q = B_bar/N`, or `BMinSepSampler`
with the paper's warm start so `E|B_t| = B_bar` from step 0); (2) once per step outside vmap, copy
`f~_t` into the loss closure, assert the probe is zero; (3) per example inside `vmap(grad_and_value)`,
forward with `opaque_router_logits=True` through the chunked-CE path, compute `P(x), h^{(L,E)}(x), w_x`,
the loss above; (4) per-group clip: gradient groups by `min(1, C_g/||g_x||)`, probe never rescaled; sum,
divide by `B_bar` (`normalize_by=B_bar`, so the stored per-group bounds are `C_g/B_bar` and `C_h/B_bar`);
(5) noise by `gaussian_noise` / `mf_gaussian_noise` with the sigmas of Section 2.7;
(6) release the noised pytree; (7) post-process the probe leaf (Section 2.10) and zero it in place so
the optimizer's update for `z` is exactly 0 (VERIFIED for AdamW-BC, AdamW, SGD-momentum with the real optimizers);
(8) accountant unchanged.

**THE cost table.** Preset regime `B_bar = 256, k = 8, E = 64, C_g = 0.9, L = 28, q = 256/5e5,
T = 15625, delta = 1e-6, T_bar = T_max`; every `rho` row reproduced with the real
`per_group_noise_stddev` and the real accountant; baseline
`poisson(gaussian(0.5622), q)*T` gives `epsilon = 3.0004` (VERIFIED). `r1 = nm Delta_h sqrt(1+1/rho)/(B_bar k/E)`
is the single-release per-entry noise on the pooled `d_hat` in units of `k/E = 0.125` (multiply by
`T_max/T_bar` if `T_bar < T_max`). Filter
factors: DP-SGD EMA .99 **0.0709**, window-256 0.0625; band-MF(64, 0.95) single step **1.431**, EMA .99
**0.0249**, window-256 0.0198. Band-MF columns are given at two multipliers: `nm = 0.5622` (the
DP-SGD/Poisson `epsilon = 3` calibration; PLAUSIBLE as a lower bound for `nm_MF`: b-min-sep amplification of a
64-banded strategy composes about `T/64` rounds at participation rate about `64 q`, which costs more epsilon at
equal `nm` than `T` Poisson rounds at rate `q`) and `nm = 1.544`, the
deterministic un-amplified band-MF bound at `epsilon = 3` (`mf_gaussian(nm, band_mf(64, 0.95),
n_steps=15625, min_sep=64)`, `strategy.sensitivity = 1.0000`, VERIFIED with the real accountant). Amplification
cannot require a larger multiplier than the un-amplified bound (PLAUSIBLE only insofar as the Monte
Carlo accountant's upper bound could in principle be looser than the deterministic one), so
`0.5622 <= nm_MF <= 1.544`.

| rho = C_h/C_g | c = sqrt(1+1/rho) | gradient-noise inflation sqrt(1+rho) (epsilon held, accountant unchanged) | epsilon if nm were held instead ("pay in epsilon") | r1 single release | DP-SGD after EMA .99 / window 256 | band-MF after EMA .99 at nm 0.5622 / **1.544** | per-layer *entries* (x5.29) after EMA .99, SGD / MF@1.544 | dead zone (c = 2) engages below delta, SGD / MF@0.5622 / MF@1.544 |
|---|---|---|---|---|---|---|---|---|
| 0.50 | 1.732 | x1.225 | 5.443 | 8.1 % | 0.57 % / 0.50 % | 0.20 % / **0.55 %** | 3.0 % / 2.9 % | 0.008 / 0.003 / 0.008 |
| 0.20 | 2.449 | x1.095 | 4.219 | 11.4 % | 0.81 % / 0.71 % | 0.28 % / **0.77 %** | 4.3 % / 4.1 % | 0.011 / 0.004 / 0.011 |
| **0.10** (router/experts trainable) | 3.317 | **x1.049** | 3.703 | 15.4 % | 1.09 % / 0.96 % | 0.38 % / **1.04 %** | 5.8 % / 5.5 % | 0.015 / 0.005 / 0.015 |
| 0.05 | 4.583 | x1.025 | 3.417 | 21.3 % | 1.51 % / 1.33 % | 0.53 % / **1.46 %** | 8.0 % / 7.7 % | 0.021 / 0.007 / 0.020 |
| **0.02 (preset-regime default)** | 7.141 | **x1.010** | 3.234 | 33.2 % | **2.35 %** / 2.07 % | 0.83 % / **2.28 %** | 12.5 % / 12.1 % | **0.033 / 0.012 / 0.032** |
| 0.01 | 10.05 | x1.005 | 3.172 | 46.7 % | 3.31 % / 2.92 % | 1.16 % / **3.19 %** | 17.5 % / 16.9 % | 0.047 / 0.016 / 0.045 |
| *opt-in, DP-SGD only: independent forward-only draw, c = 2, every m = 4 steps* | n/a | x1.0002 (epsilon-matched) | **3.003** | 9.3 % | 2.1 % (EMA .9 over releases, lag 40) | n/a | 11 % | n/a |
| *independent draw, c = 1, m = 4* | n/a | x1.013 | 3.128 | 4.6 % | 1.1 % | n/a | 5.6 % | n/a |
| *independent draw, c = 2, m = 16* | n/a | x1.0001 | 3.001 | 9.3 % | 2.1 % (lag 160) | n/a | 11 % | n/a |

Two notes on the table. The "pay in epsilon" column is the accountant's conservative bound for a
composed-of-Gaussians inner mechanism, which it routes through its generic discretised Poisson path
(`poisson.rs:31-41`) rather than the exact-Gaussian path it takes for a single Gaussian; the tight values
for the mathematically identical same-coin joint Gaussian `poisson(gaussian(nm_eff), q) * T` with
`nm_eff = nm sqrt((1 + rho)/(1 + 2 rho))` are 5.305, 4.100, 3.590, 3.306, 3.126, 3.064 for
`rho = 0.5, 0.2, 0.1, 0.05, 0.02, 0.01` (Section 6.1.1, finding 7, check K2c); the shipped route (epsilon
held, gradient noise x `sqrt(1 + rho)`) is unaffected. And `rho = 0.02` is the preset-regime value, not a
constant: the rule is the smallest `rho` whose smoothed load noise `s_inf` (the stationary `s_t` of
Section 2.10) is below about 30 % of the expected imbalance. At the prototype's regime (`B_bar = 32`,
`nm = 1.08`, `beta = 0.9`) the same noise formula puts `s_inf` at 0.254 of `k/E`, above the imbalance
(0.238), and the rule gives `rho* = 0.34` (Section 6.1.1, finding 1); at the preset regime it gives the
2.35 % of this table. Setup should print `s_inf/(k/E)` and the dead-zone threshold, and the preset should
carry the rule or this table rather than a bare 0.02.

Reading: at the worst admissible `nm_MF` the band-MF smoothed error (2.28 %) equals the DP-SGD one
(2.35 %); the 2.8x filter advantage of anti-correlated noise pays for the un-amplified multiplier.
With per-coordinate RMS imbalance `delta k/E`, the relative aux-gradient error at the default row is
about `0.0083/delta` to `0.0228/delta` (band-MF) or `0.0235/delta` (DP-SGD): 8 to 24 % at
`delta = 0.1`, 3 to 8 % at `delta = 0.3` (VERIFIED toy: 0.231 / 0.078). Below the dead zone the term is
exactly zero, where the true objective's gradient is negligible too.

**Router precision.** Stock HF precision by default; fp32-logit router opt-in as a removable
instance-level `types.MethodType` swap (the class-level patch is guarded by `__opaque_patched__`,
`_router.py:59-92`, and cannot be toggled off in-process).

**Clipping-norm procedure.** The load release never competes with the gradient for the budget. `C_g`
is governed by CE and chosen as for any dense LoRA fine-tune: one `clipped_grad(..., clipping_norm=1e9,
return_aux=True)` pass over about 256 held-out examples under the preset partition, p10/p50/p90/p99/max
of `aux.grad_norms`, the bias-plus-noise curve, a 40 to 60 % clip rate in the first 100 steps; the
preset pins the measured value (0.9 today, `train_dpftrl.py:611-612`). LoRA r = 16 on q/k/v/o is
`d = 8 257 536` parameters, per-step noise norm `nm C_g sqrt(d)/B_bar = 5.68 = 6.3 C_g`, so `C_g` acts
mostly as a learning-rate scale. The aux term moves the per-example norm < 0.1 % for `alpha <= 1e-3` (VERIFIED toy); at `alpha = 1`, the prototype's
lab value chosen to make its arms distinguishable at toy scale, the median per-example norm rose x1.24 with the
clip rate up to 60 % in the first steps under MF, so `C_g` is `alpha`-independent only for `alpha << 1`
(Section 6.1.1, finding 4). Mode
is fixed in both presets; `auto` and `adaptive` raise `ConfigurationError` with the feature on (AUTO-S
would rescale every `d(x)` to norm about `C_h`, a biased mean unit direction); a deferred engine
option `fixed_groups=("router_load_probe",)` lifts this. The guard `C_h = lambda Delta_L (1 + g)`,
`g = 1e-3`, keeps the clipper's unit-in-the-last-place (ULP) guard from firing on round-off (shrink 1.8e-7
on CPU/CUDA, 1.3e-4 on Apple Metal Performance Shaders (MPS), `_pytree.py:111-148, 214-224`, VERIFIED); T8
asserts scale exactly 1 on every CI device, and setup additionally asserts `g > 2 (u_store + norm_roundoff)`
with the engine's own round-off bound where it is exposed.

**Performance default.** Proposed default, replacing `_factory.py:319`:
`grouped_moe = kwargs.get("grouped_moe", kernels or _grouped_route_available())`, true wherever `torch._grouped_mm` exists with at least 16 experts. Dense is 8x the routed expert
FLOPs, about 5x forward FLOPs (24.4 vs 4.9 GFLOP/token), roughly 11 to 16 days vs about 2 days for the
15 625-step preset on one H100 (arithmetic VERIFIED, timing PLAUSIBLE). The CPU build also has
`torch._grouped_mm`, so the CPU MoE test surface flips to the grouped route for `E >= 16`
(`kernels/moe.py:613-632`); CUDA fp32 falls to the dense kernel (`:606-607, 633`). The mechanism itself costs < 0.15 GB per
microbatch of 8 (29.4 MB recorder-retained bf16 router logits, 58.7 MB fp32 softmax for the surrogate
backward) against about 24 GB of weights.

**Smoothing under MF.** The preset runs `band_mf_strategy(bands=64, momentum=0.95, ...)`
(`train_dpftrl.py:1567-1581`; `--momentum` default 0.95 at `:497`, `_set("bands", 64)` at `:893`); the library default `momentum=1.0` gives different factors (2.260 single
step at `n = 1024`). The filter is chosen by lag tolerance: EMA `beta = 0.99` (lag 100 steps, 0.64 % of
the horizon) gives 0.0249 under band-MF vs 0.0709 under DP-SGD; `beta = 0.95` "to inherit the
strategy's guarantee" costs 3.3x accuracy for no privacy gain. Factors are computed from the
instantiated strategy at setup, never hand constants: take the strategy's `coefficients(n_steps)`, obtain the
lower-triangular Toeplitz inverse coefficients for lags `0..N_phi = 2048` by the triangular recursion (the
object `inverse_as_streaming_matrix` builds, `_band_mf.py:135-136`, `_toeplitz.py:177`), form
`phi_t = ||row_t(F C^{-1})||` for `t < N_phi`, assert stabilisation, and store `phi_t` and `phi_inf` in
`RouterLoadState`; never a dense `n x n` solve. The EMA consumer uses this filtered factor for `s_t` and for the
logged `router_load/noise_std`, never the per-step `NoisedPytree.noise_stddev`.

**Scope.**

| item | decision |
|---|---|
| causal-LM SFT via `DPTrainer` / `DPSFTTrainer` | in (`surrogate`, `alpha` from config, `rho = 0.02`) |
| `mellum2-kstack` preset (a manual functional loop with its own `per_example_loss_fn`, `train_dpftrl.py:1369-1378`) | in via a trainer-independent helper at four seams; `surrogate`, `alpha = 1e-4`, `T_bar` from the held-out split |
| DP-FTRL strategies (band-MF, buffered linear Toeplitz (BLT), banded square root (BSR), banded inverse square root (BiSR), lambda-CGD) with b-min-sep / Poisson / balls-in-bins | in; all go through `mf_gaussian_noise`'s `PerGroup` path |
| DP-DPO, direct preference optimisation (DPO) under DP (`DPDPOTrainer`; `mellum2-codesec`, also a manual loop) | in: protected unit is the pair, `P`, `h` pooled over chosen + rejected policy forwards with denominator `L (T_c + T_r)` and token weight `w_x = (T_c + T_r)/T_bar_pair` (`T_bar_pair` public, default `2 T_max`); the reference forward never receives the kwarg; preset ships `monitor` |
| router z-loss | opt-in, `zeta = 0`; separable, zero DP cost |
| router / experts trainable | mechanism in (`rho = 0.1`; group patterns must be non-overlapping substrings of the dotted paths, `mlp.router`, `experts.`, `self_attn`, since `experts.gate_up_proj` matches both `gate` and `experts`; at most 4 groups; bounds at each group's per-example median from the held-out pass; the probe group added by direct construction, never by pattern), presets out (parameter-efficient fine-tuning (PEFT) `target_parameters` on stacked experts under `functional_call` + vmap untested) |
| per-layer surrogate, per-sequence aux | opt-in, documented as different objectives |
| independent forward-only Poisson draw | opt-in, DP-SGD only; sampler key rank-domain-separated, noise key rank-shared and applied after the all-reduce |
| packing to fixed rows | option, off; changes the protected unit |
| loss-free balancing, HF-Trainer-realised objective, token-weighted CE with public `N_bar`, STE through top-k | out |
| `output_router_logits=True` under DP | rejected with a clear error |

**Privacy hygiene (condensed).** Everything computed from private examples inside the grad transform
is private-internal until it has passed clip then noise. Router logits, probabilities, executed
routes, `h(x)`, `P(x)`, `d(x)`, `S_x` and the per-example probe gradient never leave `clipped_grad`
and are not added to `loss_aux`; `group_norms["router_load_probe"]` is excluded from telemetry; the
logged `grad_norm` and `clipped_grad_norm` are per-example totals over all leaves
(`_dp_trainer.py:2251, 2257`), so with the feature on they are recomputed as
`sqrt(sum_{g != probe} group_norms_g^2)`. The noised probe leaf is the only added release;
`d_hat, d~, s_t, f~, D, D^l` and the flags are public post-processing, logged under `router_load/*`
and checkpointed in a sidecar `router_load_state.pt`; the probe parameter is a public constant 0. The
pre-existing un-noised logging of `loss`, `grad_norm`, `clip_rate` and `batch_size` is unchanged and
outside this accounting (Section 7.5).

**Privacy statement** (mechanism page, DP-FTRL page, trainer docstring). "When `router_load_release`
is `monitor`, `surrogate` or `monitor_then_surrogate`, each step releases one Gaussian (or matrix)
mechanism on the concatenation of the clipped per-example gradients and the per-example
token-weighted centred router-load vectors `lambda (T_x/T_bar) (h^{(L,E)}(x) - k/E)`, with per-record
bounds `C_g` and `C_h = lambda (T_max/T_bar) sqrt(k L (1 - k/E)) (1 + g)`; the two are one mechanism
under Opaque's per-group allocation and the accountant is unchanged (`gaussian(nm)` per step under the
stated sampler, `mf_gaussian(nm, strategy)` for the horizon). The gradient noise is inflated by
`sqrt(1 + rho)`. The load estimate `f~_t` consumed by the loss and the monitors `D_t`, `D^l_t` are
post-processing of previous releases. No other quantity derived from private routing is released: the
per-example probe-group norms are excluded from telemetry and the logged gradient norms are computed
over the non-probe groups; the per-example loss value carries no router-derived term. Guarantee as
run: single-process, or DDP without checkpoint resume (a DDP resume today restores one sampler key on
every rank, which invalidates the subsampling amplification for every release of the trainer).
Residual non-per-example effects are floating-point only: attention-kernel selection is derived from
the public packing flag, not from the batch, and per-example gradients differ from real arithmetic by
accumulation order. The pre-existing un-noised logging of the batch-mean loss, gradient norms, clip
rate and realised batch size is unchanged by this feature and is outside its accounting. With
`router_load_source="independent"` (DP-SGD only) a second Poisson-subsampled Gaussian
`poisson(gaussian(c nm), q2) * (T/m)` is composed adaptively with the training mechanism; its sampler key is
rank-domain-separated, its noise key is shared across ranks and applied after the all-reduce, both are
checkpointed and re-folded by rank on resume, each rank samples its shard at `q2` against the global `N`, and
any non-Gaussian mechanism raises." If packing is enabled: "the protected unit is one packed 1024-token row."

**Modes and defaults.** `router_load_release in {"off", "monitor", "surrogate",
"monitor_then_surrogate"} = "off"` (bit-identical to today); `router_load_ratio = 0.02` (the preset-regime value;
the rule is the smallest `rho` whose smoothed load noise is below about 30 % of the expected imbalance, which gave
`rho* = 0.34` at the prototype's `B_bar = 32`, and setup prints `s_inf/(k/E)` and the dead-zone threshold; see the
cost-table note);
`router_aux_loss_coef = None` (config value in surrogate modes, 0 in `monitor`);
`router_load_mean_tokens = None` (`T_max`); `router_load_filter = {"kind": "ema", "beta": 0.99}` or
`{"kind": "window", "steps": 256}`; `router_load_shrink = True`; `router_load_dead_zone = 2.0`;
`router_load_trip = 0.5`; `router_aux in {"pooled", "per_layer", "per_sequence"} = "pooled"`;
`router_z_loss_coef = 0.0`; `router_fp32 = False`; `router_load_source in {"same_batch", "independent",
"public"} = "same_batch"` (`independent` is DP-SGD only; `public` is the forward-only public-batch estimate with
no release). Any non-`off` mode requires `clipping_mode ==
"fixed"`, `second_moment == False`, a family whose backbone records `router_logits`, and the chunked
causal-LM forward installed. The TRL (Transformer Reinforcement Learning library) converter maps `router_aux_loss_coef > 0` to `surrogate` for SFT
on a supported family and to `monitor` for DPO, replacing the drop-with-warning at
`trl/_convert.py:69-76`. `monitor` differs from `off` by the `sqrt(1+rho)` inflation and the noise
stream (both engines draw leaves sequentially from one per-step generator); only `off` is
bit-identical.

## 5. Implementation plan and tests (condensed)

**P0 prerequisite.** `_dp_trainer.py:1798-1800`: after `from_state_dict(ctx.current_sampler,
saved_sampler_state)` under `world_size > 1`, re-fold the restored stream key by rank exactly as
`:3802-3803` does on fresh construction. The fold must be applied after `from_state_dict`, because the
restore overwrites the template's folded key (`_poisson.py:261`, `_b_min_sep.py:259`); the alternatives are
to restore only the cursor onto the rank-folded template, or to write one sampler snapshot per rank. Scope
of the bug: `DPTrainer` and its SFT and DPO subclasses, every configured sampler, only under
`world_size > 1` with a checkpoint resume and `ignore_data_skip = False`; not single-process resume, DDP
without resume, `ignore_data_skip = True`, or the example loops. Impact: per-step epsilon at the
accountant's delta 10.5 instead of 3, and a rigorous whole-run lower bound `epsilon(1e-6) >= 6.0` after the
resume (Section 7.1). Correct the comment block at
`:3740-3755` (its `:3743` claim that "the privacy accounting is unaffected") and the `DPTrainer.__init__`
docstring at `:963-968` ("Privacy budget is unchanged ... DP-valid either way"); T25, which no existing
`distributed` test covers. Engine, `opaque-dpsgd`, `opaque-dpftrl`, `opaque-accounting`, Rust: no other change
in the first version. Reused as-is: `PerGroup` (direct construction), `clipped_grad`,
`per_group_noise_stddev`, `gaussian_noise`, `mf_gaussian_noise`, the serialization registry,
`sum_gradients_`, the streaming Toeplitz inverse.

**`opaque-patches`.** `transformers/components/moe_stats.py` (new): `router_load_and_probs(router_logits,
attention_mask, *, top_k, num_layers) -> (h_layers (L,E), P (E,), T_x)`, out-of-place,
`softmax(z.float())`, `topk`, broadcast-compare one-hot (`F.one_hot`, `scatter_add_` and `bincount` are
not vmap-safe here), explicit zero at `T_x = 0`; `centred_load`, `load_balancing_surrogate`,
`router_z_loss`. `components/router.py` (new): `install_fp32_router(model) -> undo`, instance-level.
`components/cross_entropy.py`: named kwarg `opaque_router_logits: bool = False` on the chunked
causal-LM forward, calling the backbone with `output_router_logits=True` without the HF-aux fallback
(`:212-233`, kept for HF's contract); the existing `hasattr(outputs, "router_logits")` return path
(`:359-371`) builds `MoeCausalLMOutputWithPast` carrying `router_logits`, with `logits=None` and
`aux_loss=None` on this path; the forward is installed only when `fused_linear_cross_entropy=True` reaches
`apply_model_patches` (`_factory.py:378-392`), and detection requires `__opaque_patched__` on
`type(model).forward` plus the named parameter, never `VAR_KEYWORD` (HF's forward has `**kwargs`).
`runtime/masking.py`: `all_valid_attention` from a public `packed_sequences` flag under DP; the
batch-content probe (`:195-206`) only when the flag is unset and the caller is not a DP grad transform.
`_factory.py`: grouped default; `router_fp32`. `kernels/moe.py`: `_grouped_route_available()`.
`models/mellum.py`: docstring. Docs: `docs/mechanisms/dp-sgd/moe-load-balancing.md` (+ DP-FTRL section),
`docs/user-guide/huggingface.md`.

How the statistics reach the loss: `MellumModel.forward` is `@capture_outputs` with
`OutputRecorder(MellumTopKRouter, index=0)` (`modeling_mellum.py:431-433, 475`); hooks append only
while a `ContextVar` collector is active. Under non-reentrant checkpointing the hook fires `2L` times,
the recorder returns exactly `L` tensors, and every gradient leaf including the probe equals the
non-checkpointed run to rel-L2 0.0 (VERIFIED at toy scale, feasibility checks A to E).
Microbatch chunks are separate vmapped calls with separate collectors (chunk invariance 4.7e-8, VERIFIED,
feasibility check C). `torch.compile`
remains PLAUSIBLE (T18).

**`opaque-transformers`.** `api/transformers/moe_load.py` (new, trainer-independent): `attach_probe`
(before `make_functional` / `partition_trainable`); `probe_bounds(clipping_norm, trainable, *, ratio,
...) -> (PerGroup, lam)` building the user's `PerGroup` over `trainable` minus the probe and adding the
probe by direct construction (`per_group` matches by substring and raises on two matches,
`_per_group.py:143-158`; `"router"` is a substring of `router_load_probe`); frozen dataclass
`RouterLoadState`; `filter_factors(strategy | None, ...)`; `update(state, noised_leaf, lam)`;
`summary`; `telemetry_without_probe`. `trainer/_router_load.py` (new): `RouterLoadCallback.on_pre_optimizer_step`
reads `grads.pytree["router_load_probe"]`, calls `update`, zeros the leaf in place, sets `alpha` for
the next step in `monitor_then_surrogate`, on the existing `call_event("on_pre_optimizer_step", ...,
grads=noisy_grads)` seam (`_dp_trainer.py:2198-2206`). `_training_arguments.py`: fields and validation.
`_dp_trainer.py`: `_apply_model_patches` passes `fused_linear_cross_entropy=True` when on and requires both
`__opaque_patched__` on `type(model).forward` and the named `opaque_router_logits` parameter; `_setup_training` attaches the probe, replaces the clip-norm block by `probe_bounds`,
builds `phi` and the state, registers the callback, sets the public packing flag; `_augment_inputs`
copies `f~_t` into the closure tensor (a closure tensor rather than a batch column, because
`_remove_unused_columns` prunes seeded columns under plain `DPTrainer`, `_dp_trainer.py:3446-3470`) and
asserts the probe is zero; `compute_per_example_loss` (and
the SFT and DPO variants) applies the router-load terms through a shared helper; metrics use
`telemetry_without_probe`; checkpoint sidecar, and on resume `rho`, `beta` or window, `dead_zone`, `E`, `k`,
`L`, `C_g` and `T_bar` must match or `CheckpointError` is raised.
`trl/_convert.py`: the converter mapping. Second-moment streams: `ConfigurationError` in the first
version; a later engine option `zero_groups=("router_load_probe",)` makes the probe's squared stream
structurally zero inside the vmapped clip. `examples/train_dpftrl.py` and `examples/train_dpo.py`: call
the helper at four seams (attach before `make_functional`; `probe_bounds` where the clip is built,
`train_dpftrl.py:1427-1454`; between `noise_fn` and `optimizer.update` in the step loop, `:2061-2076`;
checkpoint save/restore); thread the collator's `attention_mask` into `per_example_loss_fn` (none is passed
today, `:1369-1378`); pass `opaque_router_logits=True`; new CLI args. Under DDP the manual loop must (a)
all-reduce the probe with `sum_gradients_`, (b) use a shared noise key and (c) not reuse the trainer's
resume-key bug; without (a) and (b) `f~` diverges across ranks.

**Tests** (placement per ARC-006; behaviour only).

- T1 fp32 router: dtypes, indices equal the unpatched fp32 router, works under `vmap(grad)`, removable in-process.
- T2 `router_load_and_probs`: vmap-safe; `sum_e h^l = k`, `0 <= h^l <= 1`, `||d^{(L,E)}|| <= sqrt(7L)`, `sum_e P = 1`; equals an eager loop; padding excluded; `None`, additive and mixed-sign masks; `T_x = 0`; a duplicated tuple raises.
- T3 surrogate identity in float64 on ragged lengths: cosine `1 - 1e-12`, scale `T_tot/(B_bar T_bar)`; value-neutral form has value `CE_x` exactly.
- T4 chunked forward with `opaque_router_logits=True`: `L` logits, `logits is None`, loss unchanged; `output_router_logits=True` still takes the HF fallback; without the flag the named parameter is absent and the trainer raises.
- T5 gradient checkpointing: `(h, P, grads)` equal to the non-checkpointed run; `len(router_logits) == L`.
- T6 grouped vs dense with statistics on: identical `h`; default resolution on CPU and CUDA. T6b masking: `packed_sequences=True` gives the `None` path; otherwise materialised regardless of content; an all-valid example's gradient is invariant to a padded microbatch-mate.
- T7 setup: probe in `trainable_params`; `PerGroup` with `C_h = rho C_g (1 + 1e-3)`; a user `{"router": ...}` pattern coexists; sigmas match the allocator; Mahalanobis identity.
- T8 pre-noise probe leaf equals `(lambda/B_bar) sum_x w_x (h_x - k/E)` exactly; adversarial routing of every token to the same `k` experts at `T_max` never clips, on every CI device including MPS.
- T9 post-processing with injected known noise matches closed forms; `f~_0 = k/E`; dead zone and James-Stein (JS+) factor.
- T10 probe hygiene: probe exactly 0; loss value `CE_x` bit-for-bit; optimizer state zero; `off` bit-identical to today.
- T11 checkpoint round-trip; mismatched `rho`/`T_bar` raises. T12 (`distributed`, 2 Gloo ranks): identical `f~` on both ranks; probe all-reduced. T13 `microbatch_size=2` equals one chunk.
- T14 DP-FTRL: latch accepts the two-group `PerGroup`; realised probe `noise_stddev` equals `base * row_l2(t)`; `phi` matches a dense small-n reference; Monte Carlo reproduces the filtered factor within 10 %.
- T15 accounting invariance: `epsilon_at(delta)` identical with and without the feature under both stacks. T16 converter mapping.
- T17 hygiene: logged `grad_norm` invariant to routing; probe absent from `group_metrics`; `loss` equals mean `CE_x`.
- T18 (`slow`) `torch_compile=True` equals eager. T19 `ConfigurationError` cases. T20 second-moment exclusion on the clipped squared stream. T21 DPO pair pooling; reference forward contributes nothing.
- T22 decision rule: no false alarm under exact balance; detection of deviation 1.0 within one window; the switch changes neither `max_norm` nor `epsilon_at`.
- T23 (engine, deferred) `auto_clipped_grad(fixed_groups=...)`. T24 independent draw: accounting, per-rank counts, keys, resume re-fold, MF + independent raises.
- T25 (P0, `distributed`): after save then resume the two ranks' inclusion masks over 20 steps differ; a single-process resume is bit-identical to pre-fix behaviour.
- T26 (examples, `slow`): the manual-loop helper round-trips through a 3-step tiny-Mellum DP-FTRL loop shaped like `train_dpftrl.py`.

## 6. Validation

### 6.1 Verified here at toy scale

Random-init tiny Mellum models on CPU (1 to 2 layers, 8 to 64 experts), torch 2.14 CPU. The consolidated
results table from the prototype, verifier and auditor follows:

All rows: tiny random-init Mellum on CPU (E = 8, k = 2, L = 2, hidden 64, vocab 128, `T_max = 32`, ragged lengths in [16, 32], `B_bar = 32`, `N = 1024`), real Opaque primitives (`clipped_grad` with a direct-construction `PerGroup`, `gaussian_noise`, `mf_gaussian_noise`, `PoissonSampler`, `BMinSepSampler`, the real accountant), `nm = 1.082` calibrated to `epsilon = 3` at `delta = 1e-5` for 300 steps. The auditor re-ran the unmodified prototype (exit 0, 3 min 23 s; deterministic to every printed digit, only timing leaves differ), compared it line by line against the design, and re-ran every verifier script.

| check | what was tested | result | key numbers |
|---|---|---|---|
| V1 | separability identity on the patched model with ragged lengths: batch mean of per-example surrogate gradients at `f~ = f(B)` vs HF's own `load_balancing_loss_func` gradient | PASS | cosine 1.000010 (fp32 rounding), norm ratio 0.748046 vs predicted `T_tot/(B_bar T_bar)` = 0.748047; out-of-place aux vs HF's function rel-L2 0.0 |
| V2 | pre-noise probe leaf equals `(lambda/B_bar) sum_x w_x d^{(L,E)}(x)`; structural bound never clipped, including the adversarial all-same-route example | PASS | max abs error 3.7e-9; adversarial norm 1.732051 = `Delta_L`; probe group norm max 1.944646 = `lambda Delta_L` vs `C_h` = 1.946591; examples clipped on the probe group: 0 |
| V3 | Mahalanobis identity through the real `per_group_noise_stddev` and `gaussian_noise` | PASS | `(C_g/B)^2/sigma_g^2 + (C_h/B)^2/sigma_h^2 = 1/nm^2`, ratio 1.000000000000; gradient-noise inflation 1.157731 = `sqrt(1 + 0.34)` |
| V4 | accountant invariance under both stacks | PASS | `poisson(gaussian(1.082), 1/32) * 300`: `epsilon = 2.998599` with and without the probe; `mf_gaussian(1.082, band_mf(4, 0.95), n_steps=50)`: 3.996299 both; neither factory takes a pytree or `PerGroup` |
| V5 | 300-step DP-SGD, router and attention trainable, induced imbalance `delta_0 = 0.317`, arms OFF / ORACLE (exact non-private `f(B_t)`) / DP at `rho* = 0.34` / DP at `rho = 0.02`, plus noise-free ablations | PASS (6 of 6 criteria, one borderline) | held-out imbalance: OFF 0.317 to 0.420, ORACLE to 0.075, DP `rho*` to 0.165 (74 % of the ORACLE's reduction), DP `rho = 0.02` to 0.314 with the dead zone engaged on 86.7 % of steps (the noise floor above the imbalance, as the design's own formula predicts at `B_bar = 32`); noise-free DP ablation reaches 0.052 vs noise-free ORACLE 0.054 (lag and EMA cost nothing); cosine of the DP surrogate gradient to the population aux direction while active 0.825 (0.774 with a 64-example reference); realised pooled probe noise / predicted 0.999; probe never clipped; `z` stays 0; held-out CE identical across arms (4.804) |
| V6 | 50 steps under `mf_gaussian_noise(band_mf_strategy(bands=4, momentum=0.95))` with `BMinSepSampler` | PASS (two undeclared deviations found by the auditor, both privacy-neutral) | constant-max_norm latch held for all 50 steps (`_validate_constant_max_norm` runs on every call and raises on change; the prototype hard-coded the flag, the auditor's patched copy asserts it explicitly); realised `noise_stddev` per group vs `base * row_l2(t)`: max rel error 3.4e-16; empirical probe-noise RMS / predicted 0.993; EMA-filtered factor 0.1488 vs 0.2146 under DP-SGD (x0.69 at bands 4); imbalance 0.317 to 0.161. The prototype passed `sampling_prob = q` instead of the paper's per-iteration `p = q/(1 - q(b - 1))` that the trainer derives, so its expected batch was 29.3 (observed 29.9) against `normalize_by = 32`; the auditor's one-line patch gives mean batch 31.98 with every identity unchanged (imbalance 0.317 to 0.170) |
| V7 | gradient checkpointing on vs off (opaque HF checkpoint glue) | PASS | probe leaf rel-L2 0.0; max over all clipped leaves 0.0 |
| K1a | DDP + checkpoint resume: does a restored sampler on rank `r != 0` re-derive a rank-specific key? (code reading plus a sampler-level reproduction for `PoissonSampler` and `BMinSepSampler`) | CONFIRMED (bug) | restored rank-1 stream key equals rank 0's; 20 of 20 post-resume inclusion masks identical across ranks (0 of 20 before the checkpoint); co-inclusion rate of the same local index 0.0506 vs `q^2 = 0.0025` if independent; single-process resume exact (auditor: applies to `DPTrainer`, the SFT and DPO trainers and every configured sampler, only under `world_size > 1` with a checkpoint resume and `ignore_data_skip=False`; not to single-process resume, DDP without resume, `ignore_data_skip=True`, or the example loops) |
| K1b | privacy effect of the shared coin, own hockey-stick integration at `sigma = 0.5622, q = 5.12e-4` | CONFIRMED, under-stated by the design | per-step `delta(epsilon = 3)`: 3.1597e-11 independent vs 2.7587e-5 shared coin with an aligned partner (ratio 8.7e5); per-step `epsilon` at the accountant's `delta` 10.5 instead of 3; rigorous whole-run lower bound `epsilon(1e-6) >= 6.0`; with `W` ranks the partner mass can be `u = W - 1`, and the auditor's sweep shows the divergence saturates at the cap `q delta_G(3) = 5.75e-5` from `u >= 3` with `epsilon_1(1e-6)` rising only to 6.13, so the `u = 1` figures are within 2x (delta) and 0.12 (epsilon) of the worst case for any world size |
| K2a | accountant invariance in `_build_mechanism` | CONFIRMED | `num_groups` consumed only inside the adaptive-clipping closure (`_dp_trainer.py:4329, 4332-4340`); baseline `epsilon = 3.0004`; naive per-group `nm C_i` would be 11.84 |
| K2b | allocator identity at six `rho` | CONFIRMED | `sigma_g/(nm C_g) = sqrt(1 + rho)` and the Mahalanobis sum `= 1/nm^2` to six digits |
| K2c | "pay in epsilon" column | REPRODUCED, shown conservative | composed-inner call reproduces 3.234 / 3.703 (`rho` = 0.02 / 0.1) exactly; the mathematically identical joint Gaussian `poisson(gaussian(nm_eff), q) * T`, `nm_eff = nm sqrt((1 + rho)/(1 + 2 rho))`, gives 3.126 / 3.590: the accountant's generic-inner Poisson path is about 0.11 looser than its exact-Gaussian path; the column is informational only |
| K3 | structural bound `||d^{(L,E)}(x)|| <= sqrt(k L (1 - k/E))` (random, adversarial, exhaustive at E = 8) and the replace-one bound | CONFIRMED | equality attained in every cell; `T_x = 0` gives 0; replace-one per-layer bound is `sqrt(2 min(k, E - k) L)` = `sqrt(2 k L)` for `E >= 2k` (attained by disjoint expert sets); `sqrt(2 k L (1 - k/E))` is violated (5.29 vs 5.66 attained at E = 64, k = 8, L = 2) |
| K4 | value-neutral surrogate `CE + alpha (S - sg[S])` | CONFIRMED | value bit-identical to `CE` (`torch.equal`); gradient bit-identical to `grad(CE + alpha S)` under `torch.func.grad` and under `vmap`, fp32 and fp64 |

Established before that table (VERIFIED at toy scale): fp32 vmap-vs-HF-loop exactness 3.5e-7 across
dense and grouped MoE, both layer types, right and left padding, `attention_mask=None`, gradient
checkpointing and chunked CE; bf16 drift 0.45 to 0.5 % with zero route flips; the separability
identity at 0.0 / 2.2e-7 (unpatched / patched ragged); the ragged token-weight identity (cosine
1.000000000000); the structural load bound attained and never exceeded through the real clipper;
the Mahalanobis identity 1.000000 at six `rho` with the real allocator; accountant invariance
(`epsilon = 3.0004` with and without the probe group); the MF latch accepting the two-group
`PerGroup`; realised probe sigma `base * ||row_t(C^{-1})||`; the band-MF(64, 0.95) filter factors; the
probe at zero through three optimizers; the recorder under non-reentrant checkpointing; chunk
invariance 4.7e-8; the `scatter_add_` failure of HF's aux under vmap; the exact chi-square dead-zone
tails; the EMA bias-correction gains; the deterministic un-amplified band-MF multiplier 1.544 at
`epsilon = 3`; the shared-coin hockey-stick computation of Section 7.1.

### 6.1.1 What the validation adds to the design

Findings from the prototype and the verifier that refine the design (all VERIFIED at toy scale unless marked):

1. `rho` is a regime number, not a constant. At `B_bar = 32`, `nm = 1.08`, `beta = 0.9` the design's own noise formula puts the smoothed load noise at 0.254 of `k/E`, above the released imbalance (0.238), so at `rho = 0.02` the dead zone kept the surrogate off on 86.7 % of steps; the rule "smallest `rho` whose smoothed noise is below 30 % of the expected imbalance" gave `rho* = 0.34` there and recovers 74 % of the non-private oracle's balancing. At the preset regime (`B_bar = 256`, `k/E = 0.125`, `nm = 0.56`, `beta = 0.99`) the same formula gives the 2.35 % of the cost table. The trainer should print `s_inf/(k/E)` and the dead-zone threshold at setup, and the preset should carry the rule or the table rather than a bare 0.02.
2. The dead zone's false-pass rate is `E`-dependent: `P(chi^2_{E-1} > 2E)` is 4.2e-6 at `E = 64` but about 5e-2 at `E = 8` (PLAUSIBLE arithmetic, consistent with the observed non-dead steps at `rho = 0.02`); with bias-corrected `s_t` the first few steps pass more easily. State it as `E`-dependent.
3. At small `B_bar` the batch load `f(B_t)` is itself a noisy estimator of the population load (per-entry batch-sampling std 0.10 to 0.13 of `k/E` here), so the exact batch aux gradient has cosine only 0.48 to the population direction once the imbalance is small, while the EMA-smoothed DP estimate has 0.80. The metric "cosine to the exact batch aux gradient" (0.64 at `rho*`) is bounded by that batch-sampling noise, not by the mechanism; a noise-free DP ablation (same lag and EMA, `nm = 0`) matches the oracle's imbalance trajectory (0.052 vs 0.054). The design's smoothing is a feature even without privacy noise. Preset-regime magnitude is a GPU question.
4. The clipping-norm claim "the aux term moves per-example norms by under 0.1 %" holds for `alpha <= 1e-3` (phase-1 E3); the prototype ran `alpha = 1` to make the arms distinguishable at toy scale and saw the median per-example norm rise x1.24 with the clip rate up to 60 % in the first steps under MF. `C_g` is `alpha`-independent only for `alpha << 1`.
5. Under band-MF (bands 4, momentum 0.95) everything holds unchanged: the latch, the per-group realised sigma (`base * row_l2(t)` to 3e-16), the probe receiving the correlated noise, and the strategy-derived filter factor (x0.69 of DP-SGD at bands 4; the design's x0.35 is for bands 64 and `beta = 0.99`). The b-min-sep Monte Carlo accountant was not run; the un-amplified `mf_gaussian` bound (`epsilon = 4.00` at `nm = 1.082` for 50 steps) is the valid looser guarantee.
6. The DDP-resume finding is under-stated in the design: beyond the six-orders-of-magnitude per-step `delta`, the per-step `epsilon` at the accountant's `delta` is 10.5 instead of 3 and a rigorous whole-run lower bound is `epsilon(1e-6) >= 6.0`, so the accountant's certificate is void for the worst-case pair, not merely loose. The P0 fix must apply the rank fold after `from_state_dict` (the restore overwrites the template's folded key at `_poisson.py:261` and `_b_min_sep.py:259`), or restore only the cursor onto the rank-folded template, or write one snapshot per rank. The `DPTrainer.__init__` docstring at `_dp_trainer.py:963-968` makes the same wrong claim as the comment at `:3741-3744`. No existing `distributed` test covers resume. Scope: `DPTrainer` and its SFT and DPO subclasses with every configured sampler, only under `world_size > 1` with a checkpoint resume and `ignore_data_skip=False`; the example loops are not affected. With `W` ranks the shared-coin partner mass can be `u = W - 1`; the divergence saturates from `u >= 3`, so the quoted figures are within 2x in `delta` and 0.12 in `epsilon` of the worst case for any world size.
7. The "pay in epsilon" column of the cost table is a valid but conservative bound: the accountant routes a composed-of-Gaussians inner mechanism through its generic discretised Poisson path (`poisson.rs:31-41`) instead of the exact-Gaussian path it takes for a single Gaussian; the tight values for the same-coin joint mechanism are 3.126 / 3.590 at `rho = 0.02 / 0.1` (5.305, 4.100, 3.306, 3.064 at `rho = 0.5, 0.2, 0.05, 0.01`). The shipped route ("epsilon held, gradient noise x sqrt(1 + rho)") is unaffected. Routing a composed-Gaussian inner through the exact path in the accountant would remove the looseness.
8. The general replace-one bound on the per-layer centred load is `sqrt(2 min(k, E - k) L)`, which is `sqrt(2 k L)` for `E >= 2k` (the design's stated form, tight); `sqrt(2 k L (1 - k/E))` is not a bound.
9. Two prototype details the auditor corrected, both privacy-neutral: the band-MF arm fed the base rate `q` to `BMinSepSampler` where the trainer feeds the paper's per-iteration `p = q/(1 - q(b - 1))` (expected batch 29.3 instead of 32 against `normalize_by = 32`; patched: 31.98), and the "latch held" flag was hard-coded rather than asserted (the latch does run and raise on every call; the patched copy asserts it). The dense `Opaque_MoE` path ran in the prototype because the grouped route requires at least 16 experts, although the test helper requests it; the opaque non-reentrant checkpoint glue was active in V7.

### 6.2 What needs the real checkpoint on a GPU

Script `examples/validate_mellum_dp.py`, one 80 GB GPU, `JetBrains/Mellum2-12B-A2.5B-Base` bf16
weights, a KStack held-out shard disjoint from the preset's training rows, `T = 1024` right-padded
ragged rows, PEFT LoRA r = 16 / alpha = 32 on q/k/v/o. Budget: oracle plus statistics about 30 min;
the fp32 floor runs the dense kernel on CUDA (fp32 is not routed to Triton), 20 to 40 min; the
preset's band-MF calibration runs once offline on a many-core host with its wall-clock recorded, and
every validation run receives `--noise-multiplier` (or the bracket value 1.544); the 200-step DP runs
about 30 min each with grouped MoE. Comparisons with the fp32 router on one side run in separate
processes. Every number is a design-time measurement on held-out data; nothing here is logged by a DP
run.

Pitfall for the script: under b-min-sep pass `BMinSepSampler` the per-iteration probability `p = q/(1 - q(b - 1))`,
which is what the trainer derives, not the base rate `q`, so that the expected batch equals `B_bar` against
`normalize_by = B_bar` (Section 6.1.1, finding 9).

**Oracle definition (G2).** O0, the precision-matched "non-DP HF path": same process, same patched
module (`apply_model_patches` with the run's `grouped_moe`, `router_fp32` and dtype), HF eager forward
on one microbatch (B = 8) with `output_router_logits=False`, `loss.backward()` per example in a Python
loop ("HF loop") and once batched ("HF batched"). O1, the floor: the same loop in fp32 weights
(48 GB). DP side: `clipped_grad(..., clipping_norm=1e9, return_aux=True)` internals, bf16 `vmap(grad)`
per-example vectors with the full patch set (grouped and dense; chunked CE; checkpointing on and off;
one ragged microbatch with the public-flag mask path).

**Drift metric.** (m1) per-example rel-L2 of the whole LoRA gradient, vmap vs HF loop and vs O1,
median and max; (m2) route-flip counter per `(token, layer)`: symmetric difference of executed top-8
sets from the same fp32 softmax, plus the fraction of tokens with margin `p_(k) - p_(k+1) < 1e-6` and
the exact-bf16-tie fraction; (m3) per-parameter-group rel-L2; (m4) HF batched vs HF loop; (m5) loss
absolute difference; (m6) the same with `router_fp32=True` on both sides, separate process, and the
bf16-vs-fp32-forward flip rate per layer.

**Statistics to collect (G3)**, 256 held-out examples under the preset partition: per-example
gradient-norm quantiles (`C_g`, clip rate); the length distribution `T_x` (the preset's `T_bar`, the
`T_max/T_bar` noise factor, whether packing is worth its unit change); per-coordinate RMS imbalance
`delta = ||f(B) - k/E||_2/(k/E)/sqrt(E)` and its infinity norm over 32 batches of 256, batch-to-batch
drift over 100 LoRA steps, per-layer `delta^l` (signal: need `r_smoothed <~ 0.3 delta`); per-example
`||d(x)||`, `||d^{(L,E)}(x)||` and unused-expert fractions (bound tightness, expect about 1.0 / 5.3
balanced; H5); bf16-vs-fp32 router flip rate per layer, logit scale, top-8/9 margin; dense vs grouped
step time and peak memory at microbatch 8 with a flip counter (acceptance: grouped at least 3x faster,
dense-vs-grouped flips 0); aux/CE gradient-norm ratio at
`alpha = 1e-4, 1e-3` on attention LoRA and on router weights (H4 at scale); surrogate tracking error
`||f~_t - f(B_t)||/||f(B_t) - k/E||` over a 512-step dry run; the preset's calibrated `nm_MF`.

**Acceptance.** Requirement (d): (A1) vmap-vs-HF-loop flips = 0 at equal precision, or every flip's
margin `< 1e-6`; otherwise bisect SDPA path, MoE path, RMSNorm before touching the DP design;
(A2) rel-L2(vmap, HF loop) at most 2x rel-L2(HF batched, HF loop); (A3) fp32 rel-L2 at most 1e-5 and
0 flips; (A4) PR #980's "about 1.3 %" reproduced to plus or minus 0.5 pp under this definition;
(A5) the surrogate identity on the checkpoint in fp32 on a ragged microbatch, cosine at least
`1 - 1e-6`, scale `T_tot/(B_bar T_bar)` to 1e-4, equality on a packed microbatch. Mechanism, 200-step
`surrogate` run at `rho = 0.02` under both stacks (DP-SGD calibrated at `epsilon = 3` for `n = 200`;
band-MF at the fixed conservative `nm = 1.544` unless calibrated offline): (a) `f~_t` tracks the
lab-only token-weighted `f(B_t)` with per-entry error at most 10 % of `k/E`, and
`||d~+_t - d_true||/||d_true|| <= 0.25` whenever `||d_true||^2 >= c E s^2`; (b) cosine between the DP
surrogate aux gradient and the exact batch aux gradient at least 0.8 whenever `delta >= 0.1`; (c) eval
loss within run-to-run noise of the feature-off run, expert-usage entropy not below the base model's,
no expert below `0.25 k/E`; (d) reported epsilon identical with and without the feature; (e) probe
group norm never exceeds `C_h`; (f) throughput within 20 % of the feature-off run; (g) no trip under
exact balance and the dead zone zeroes the surrogate in at least 99.99 % of balanced steps; the
switch changes neither `max_norm` nor `epsilon_at`; (h) per-example rel-L2 vs the oracle every 100
steps at most 1.5 % with zero flips; (i) logged `grad_norm` equals the non-probe group norm.

## 7. Findings outside the design that need their own action

1. **DDP + checkpoint resume restores rank 0's sampler key on every rank (P0).** The trainer's own
   comment says so: "the sampler snapshot ... is written once on rank 0, so resuming a DDP run
   currently restores rank 0's per-rank key on every rank, re-introducing the cross-rank correlation
   after the resume point" (`_dp_trainer.py:3750-3755`, VERIFIED). The restore installs the saved
   sampler with no rank re-fold (`:1798-1800`, VERIFIED); `fold_in(sampler_key, rank)` exists only on
   fresh construction (`:3802-3803`, VERIFIED). Records sharing a local shard index on different ranks
   then have identical inclusion coins for the rest of the run, Poisson and b-min-sep alike. The same
   comment block (`:3740-3755`) asserts at `:3743` that "the privacy accounting is unaffected", and the
   `DPTrainer.__init__` docstring at `:963-968` states "Privacy budget is unchanged ... DP-valid either
   way" for sampler resume; both are wrong: the tight
   subsampled-Gaussian PLD needs the added record's coin independent of the others' (Feldman and
   Shenfeld Lemma 3.2 / ZDW Thm 11); with a shared coin the add/remove pair is
   `((1-q) A + q B_0, (1-q) A + q B_1)` with `A != B_0`, which keeps delta-amplification but loses
   epsilon-amplification. One-dimensional hockey-stick computation at `sigma = 0.5622, q = 5.12e-4,
   epsilon = 3`: per-step delta 3.16e-11 with independent coins vs 2.76e-5 with a shared coin and an
   aligned neighbour; `u = 0` reproduces the standard value exactly; the analytic estimate
   `q delta_Gauss ~ 5.7e-5` agrees in order
   (VERIFIED by direct computation). Six
   orders of magnitude, on every step after the resume, for every release of the trainer. The validation
   phase confirmed the bug at the sampler level (20 of 20 post-resume inclusion masks identical across
   two ranks for `PoissonSampler` and `BMinSepSampler`, check K1a) and strengthened the impact (check K1b):
   the per-step epsilon at the accountant's delta is 10.5 instead of 3, and a rigorous whole-run lower
   bound after the resume is `epsilon(1e-6) >= 6.0`, so the accountant's certificate is void for the
   worst-case pair, not merely loose; with `W` ranks the shared-coin partner mass can be `u = W - 1`, but
   the divergence saturates from `u >= 3`, so the `u = 1` figures are within 2x in delta and 0.12 in
   epsilon of the worst case for any world size. Scope: `DPTrainer` and its SFT and DPO subclasses,
   every configured sampler, only under `world_size > 1` with a checkpoint resume and
   `ignore_data_skip = False`; not single-process resume (bit-exact), DDP without resume,
   `ignore_data_skip = True`, or the example loops. Fix at the resume site (Section 5): the rank fold
   must be applied after `from_state_dict`, which overwrites the template's folded key at
   `_poisson.py:261` and `_b_min_sep.py:259`; alternatives are to restore only the cursor onto the
   rank-folded template, or one snapshot per rank. Correct both comments, add T25 (no existing
   `distributed` test covers resume); until then every resumed DDP run's privacy statement reads
   accordingly.
2. **`DPTrainer` runs the dense MoE kernel by default on CUDA.** `kernels=bool(args.use_performance_kernels)`
   (`_dp_trainer.py:841-847`), default `False` (`_training_arguments.py:436`), reaches
   `grouped_moe = kwargs.get("grouped_moe", kernels)` (`_factory.py:319`): every token through all 64
   experts. The docstring (`:429-435`) does not mention MoE; the flag is captured by the first
   class-level patch per process (`_router.py:59-92`). Decouple (Section 4) and document.
3. **HF `output_router_logits=True` under vmap with a mask fails** (`scatter_add_` into an unbatched
   `torch.zeros(E)`, `modeling_mellum.py:596-598`, reproduced VERIFIED) and on the chunked-CE forward
   falls back to the full-vocabulary forward (`cross_entropy.py:212-233`), 403 MB of logits per example.
   Reject the kwarg under DP with a clear error; the named `opaque_router_logits` path replaces it.
4. **HF Trainer's gradient-accumulation aux artefact.** With accumulation `G`, HF Trainer optimises
   the per-microbatch aux with effective coefficient `G * coef` (`trainer.py:1961-1963` divides by the
   accumulation count only when the model does not accept loss kwargs; `MellumForCausalLM.forward`
   has `**kwargs`; VERIFIED on a toy HF Trainer run). At the presets' shape an HF reference run would use `f` on
   8 sequences and coefficient 0.032. Anyone comparing against an HF Trainer baseline must know this;
   worth an upstream report.
5. **Un-noised telemetry outside the accountant.** `metrics["loss"]`, `["grad_norm"]`, `["clip_rate"]`,
   `["batch_size"]` (`_dp_trainer.py:2249-2257`, VERIFIED) are un-noised functions of the private batch
   (the realised batch size is `Binomial(N, q)` vs `Binomial(N+1, q)` under add/remove). Pre-existing;
   under the review protocol's "all releases ... included in the privacy statement": document or noise.
6. **`mc_resolution` is clamped to `delta/2`.** `epsilon_at` uses `min(configured_resolution, delta/2)`
   (`opaque-accounting/.../core/_base.py:214-218`, VERIFIED) regardless of the example's
   `--mc-resolution` default 1e-5 (`train_dpftrl.py:704-707`); at `delta = 1e-6` the accountant reports
   64 997 003 transcripts per adjacency direction and no `b_min_sep(mf_gaussian(...))` evaluation
   finished on 4 CPU cores, not even at `n = 256`. Make the CLI honour or warn about the clamp; budget
   preset calibration on a many-core host.
7. **Data-dependent SDPA kernel selection under vmap.** `masking.py:195-206` selects the kernel from
   `physical_mask.all()` over the whole physical microbatch (VERIFIED). Under add/remove one padded
   example changes the kernel for every mate, a systematic floating-point dependence of `g_y` on the
   batch composition (0.45 to 0.5 % rel-L2, or up to `2 C_g` after clipping on a flip). Each
   contribution is still clipped and privacy proofs are stated for real arithmetic, but the switch is
   avoidable: derive it from the public `packed_sequences` flag under DP (Section 4). Any future
   batch-content-dependent branch is the same class of issue (T6b).

## 8. Risks and falsifiers

1. **No usable signal on real data.** If `delta <~ 0.03` on KStack, the smoothed noise at `rho = 0.02`
   sits at the dead zone and the release buys nothing beyond the monitor; not a faithfulness failure
   (the true aux gradient is equally silent). `rho = 0.02` is the preset-regime value, not a constant: the
   rule is the smallest `rho` whose smoothed load noise is below about 30 % of the expected imbalance
   (the prototype needed `rho* = 0.34` at `B_bar = 32`, where 0.02 left the dead zone engaged on 86.7 % of
   steps, Section 6.1.1, finding 1), and setup prints `s_inf/(k/E)` and the dead-zone threshold so the
   regime is visible before training. Mitigation: a larger `rho` by the rule (`rho = 0.1` as a first step),
   a longer filter, the independent
   draw (DP-SGD), or `monitor` only; OLMoE section 4.3 and Tholoniat et al. report dropping the aux in
   fine-tuning is benign (transfer PLAUSIBLE).
2. **Lag bias.** If `f(B)` drifts faster than 100 steps, `f~` chases a stale target. Falsifier: drift
   over 100 steps larger than `r_smoothed`. Mitigation: shorter filter at higher `rho`.
3. **The surrogate is inert even when needed** (router trainable, aux/CE ratio below the noise): the
   honest recommendation becomes Tholoniat's (drop the aux, freeze the router); the design stays
   correct, its utility claim would be false.
4. **Recorder under `torch.compile`**: PLAUSIBLE, T18.
5. **Flip-free vmap does not hold at 28 layers.** Falsifier: A1 fails; bisect kernels, not the design.
6. **The grouped-MoE default changes numerics** beyond the documented floor, including the CPU test
   surface. Falsifier: a parity tolerance moves or a dense-vs-grouped flip count above 0 on the real
   model; keep the dense default and fix the kernel.
7. **`nm_MF` bracket looseness.** If the offline calibration lands above 1.544, use the calibrated
   value; privacy holds either way, only the MF load-error column moves.
8. **Probe leaf and the optimizer.** A future optimizer treating a zero gradient differently could
   drift the probe; T10 and the `_augment_inputs` assertion make silent drift a hard error.
9. **AUTO-S users lose the feature** until `fixed_groups` lands.
10. **The fp32 router opt-in moves the fine-tune away from the HF-bf16 serving router.** Falsifier of
    the default: bf16-routed eval after an fp32-router fine-tune degrades measurably; document, do not
    flip the default.
11. **Telemetry leakage.** Any change adding `h(x)`, `P(x)`, `S_x` or the probe norms to `loss_aux` /
    `group_metrics`, or reverting `grad_norm` to the all-leaf total, silently releases un-noised
    statistics. Guard: T17 in its routing-invariance form.
12. **DDP + resume.** Until P0 lands, any resumed DDP run lacks the accounted guarantee. Falsifier of
    the fix: T25.
13. **Data-dependent kernel selection elsewhere.** Guard: T6b extended to any new kernel switch.
14. **Faithfulness is to the logical-batch formula, not to HF Trainer** under accumulation; stated.
15. **Concurrent releases under DP-FTRL** would need concurrent composition (theorem numbers not
    verified); avoided, the independent draw is DP-SGD only.
16. **DPO pair pooling** not exercised; T21 and the `monitor` preset contain it.
17. **Experts-trainable variant**: PEFT `target_parameters` under `functional_call` + vmap may not
    work; the mechanism is unaffected.
18. **Ragged-length scale factor.** With `T_bar = T_max` the aux is down-weighted by
    `(T_bar_true/T_max)^2`; with a measured `T_bar` the release noise grows by `T_max/T_bar`; neither
    affects privacy. Falsifier of the default: `T_bar_true/T_max < 0.5` on KStack; then set `T_bar` in
    the preset (`rho = 0.05` keeps the smoothed error at most 2.4 %).
19. **Unverified at scale.** Every magnitude except the accounting table, the allocator identity, the
    MF filter factors and the hockey-stick computation comes from random-init toys; (c) is settled only
    by Section 6.2's statistics and (d) only by its oracle comparison.

## 9. Sources

Theorem and equation numbers marked VERIFIED were extracted from the primary PDFs during this research;
PLAUSIBLE marks numbers taken from a citing document
and not re-fetched. Nothing beyond what those reads established is cited.

- Fedus, Zoph, Shazeer. *Switch Transformers*. JMLR 23 (2022). https://arxiv.org/abs/2101.03961, section 2.2, eqs. (4) to (6). VERIFIED.
- Zoph et al. *ST-MoE*. 2022. https://arxiv.org/abs/2202.08906, section 3.1, eq. (5) (router z-loss; bf16 round-off motivation). VERIFIED.
- Kojic et al. *Mellum 2 Technical Report*. 2026. https://arxiv.org/abs/2605.31268, section 3.6 (Megatron running-average load), section 5.1.2 (SFT coefficient 1e-4), section 5.2 (trainer/inference route disagreement), appendix (FP32 router). VERIFIED.
- Andrew, Thakkar, McMahan, Ramaswamy. *Differentially Private Learning with Adaptive Clipping*. NeurIPS 2021. https://arxiv.org/abs/1905.03871, Theorem 1. VERIFIED.
- Dong, Roth, Su. *Gaussian Differential Privacy*. 2019. https://arxiv.org/abs/1905.02383, Theorem 2.7, Corollary 3.3. VERIFIED.
- Zhu, Dong, Wang. *Optimal Accounting of Differential Privacy via Characteristic Function*. 2021. https://arxiv.org/abs/2106.08567, Definition 7, Theorem 10, Theorem 11. VERIFIED.
- Feldman, Shenfeld. 2026. https://arxiv.org/abs/2602.17284, Lemma 3.2, Theorem 3.3, Algorithms 8 and 9; as cited by `src/amplification/poisson.rs`. VERIFIED.
- Denisov, McMahan, Rush, Smith, Thakurta. *Improved Differential Privacy for SGD via Optimal Private Linear Operators on Adaptive Streams*. 2022. https://arxiv.org/abs/2202.08312, Theorem 2.1. VERIFIED.
- Dong, Ganesh. 2026. https://arxiv.org/abs/2602.09338, Algorithm 2 (b-min-separation sampling), as cited by `opaque-dpftrl/.../sampling/_b_min_sep.py`. PLAUSIBLE for the algorithm number.
- Ponomareva et al. *How to DP-fy ML*. JAIR 2023. https://arxiv.org/abs/2303.00654, section 5, options (a) and (b). VERIFIED.
- Davody, Adelani, Kleinbauer, Klakow. *On the effect of normalization layers on DP training*. 2020. https://arxiv.org/abs/2006.10919. VERIFIED.
- Kong, Muñoz Medina, Ribero, Syed. *Differentially Private Optimization for Non-Decomposable Objective Functions*. ICLR 2025. https://arxiv.org/abs/2310.03104, Lemma 4.1, Theorem 4.2, Corollary 4.4. VERIFIED.
- Tholoniat, Inan, Kulkarni, Sim. *Differentially Private Training of Mixture of Experts Models*. PPAI@AAAI 2024. https://arxiv.org/abs/2402.07334. VERIFIED.
- Muennighoff et al. *OLMoE*. 2024. https://arxiv.org/abs/2409.02060, section 4.3 (an appendix line states the opposite; section 4.3 is the reasoned statement). VERIFIED with that caveat.
- Wang, Gao, Zhao, Sun, Dai. *Auxiliary-Loss-Free Load Balancing Strategy for MoE*. 2024. https://arxiv.org/abs/2408.15664, eq. (2), Algorithm 1. VERIFIED.
- DeepSeek-V3 Technical Report. https://arxiv.org/abs/2412.19437, section 2.1.2, eqs. (16) to (20). VERIFIED. DeepSeek-V2. https://arxiv.org/abs/2405.04434, section 2.2.3, eqs. (23) to (25). VERIFIED.
- Qiu et al. *Demons in the Detail*. ACL 2025. https://arxiv.org/abs/2501.11873, Algorithm 1, eqs. (4) to (6). VERIFIED.
- Wang et al. *ReMoE*. ICLR 2025. https://arxiv.org/abs/2412.14711, section 3.2 (top-k discontinuity). VERIFIED.
- Bu, Wang, Zha, Karypis. *Automatic Clipping*. 2023. https://arxiv.org/abs/2206.07136. PLAUSIBLE, theorem numbers not re-fetched.
- Koskela, Honkela. 2021. https://arxiv.org/abs/2102.12412, Theorem 4, section 6.2. VERIFIED. Gopi, Lee, Wutschitz. NeurIPS 2021. https://arxiv.org/abs/2106.02848. VERIFIED. Doroshenko et al. *Connect the Dots*. PETS 2022. https://arxiv.org/abs/2207.04380. VERIFIED.
- The James-Stein and chi-square dead-zone arithmetic is elementary (tails from `scipy.stats.chi2`); no external source is claimed.

Repository sources: `.junie/differential-privacy-review.md` (adjacency, evidence standard);
`docs/user-guide/{accounting,clipping,sampling,dp-ftrl}.md`; HF `models/mellum/modeling_mellum.py` and
`configuration_mellum.py` (5.16.1); the model card https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Base.

## Appendix A. Provenance

The experiment scripts, their outputs, the per-phase agent reports and the full design specification were
recorded in this branch's history under a companion `mellum-dp-research` directory and were removed once the
feature landed. The numbers quoted in this report come from those runs. The mechanism is documented for
users at `docs/mechanisms/dp-sgd/moe-load-balancing.md` and implemented in `opaque.transformers.moe_load`,
`opaque.api.transformers.trainer._router_load`, and `opaque.api.patches.transformers.components.moe_stats` /
`router`.
