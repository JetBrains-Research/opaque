# Phase 1 digest — established facts, decisions still open, and gaps (read with BRIEF.md)

Full reports (read them; this digest is the index): phase1-divergence.md, phase1-primitives.md,
phase1-literature.md, phase1-math.md, phase1-empirical.md, phase1-critic.md (all in this dir).
All magnitudes below come from RANDOM-INIT TINY MODELS on CPU unless stated; the algebra and code
facts are exact. No trained-checkpoint or GPU number exists yet (critic gaps G2, G3, G6).

## Settled (VERIFIED)
F1. Only inseparable batch-coupled term in Mellum2's HF objective: the Switch-style aux loss
    L_aux(B) = E * sum_e f_e(B) P_e(B), f and P pooled over ALL layers and the whole batch with one
    common denominator L*T_tot (sum_e f_e = k = 8; HF product-of-pooled-means, NOT per-layer). Mask =
    attention mask (prompt tokens count). Coef 0.001 = Mellum2 PRETRAIN coefficient; Mellum2's own SFT
    used 1e-4; pretraining also used a per-token router z-loss 1e-3 (separable; HF/Opaque don't have it)
    and an FP32 router; Megatron's "global" f was a running average across microbatches (i.e. lagged).
F2. HF Trainer with gradient accumulation actually optimises CE_tokenmean(logical batch) +
    coef * sum_over_microbatches aux(microbatch) — f is a MICROBATCH statistic and the coefficient is
    effectively G*coef (critic Exp A, rel-L2 0.0). So "faithful to HF Trainer" != "faithful to the
    logical-batch formula". HF also recomputes the aux top-k from a bf16 softmax while the executed
    forward uses an fp32 softmax (0.03-1.7% of tokens differ; 4-7% exact bf16 ties at the k/k+1 boundary).
F3. Exact identity (H2): grad L_aux(B) = sum_x grad S(x; f~) at f~ = f(B), with
    S(x; f~) = E * w_x * sum_e f~_e P_e(x), w_x = T_x/T_tot, P_e(x) = per-example token-mean softmax prob.
    Because sum_e P_e = 1, grad S = E * sum_e (f~_e - k/E) grad P_e: the signal is the IMBALANCE; uniform
    f~ gives zero gradient. Per-example aux (own f(x)) is a different regulariser (anti-specialisation,
    cos 0.26-0.39 to the batch gradient at toy scale; pushes within-document uniformity).
F4. Opaque's per-example convention = equal example weights (per-example token-mean CE, opaque-alignment
    _nll.py). HF weights tokens (num_items_in_batch). Coincide iff equal lengths (true for the packed
    1024-token presets). Representable exactly with a PUBLIC constant N-bar if wanted.
F5. Load-vector release: pooled fraction h_x = f(x) in [0,1]^E, sum = k, ||h_x||_2 <= sqrt(k) structurally
    (no clipping needed; centred: sqrt(k(1-k/E))). Per-layer variant ||.||_2 <= sqrt(kL). Add/remove
    sensitivity = sup ||h_x||; divisor must be the public expected batch size. Single-step relative noise
    r = sigma_h * E / (B-bar * sqrt(k)): 8.8% at sigma_h=1, B=256 (per-layer: x sqrt(L) = 46.8%).
    EMA beta: noise variance x (1-beta)/(1+beta) -> 2.0%/1.4%/0.6% at beta .9/.95/.99 (sigma_h=1).
F6. Accounting: two Gaussian releases on the SAME sampled batch = ONE Gaussian on the concatenation with
    Mahalanobis budget sum_g C_g^2/sigma_g^2 = 1/nm^2 (whitening; Andrew et al. 2021 Thm 1 form). Opaque's
    per_group_noise_stddev implements sigma_g = nm*sqrt(C_g * sum C) which satisfies it exactly; the naive
    "sigma*C_g per group" is WRONG (= gaussian(nm/sqrt(K))). Numbers at the preset regime (eps=3, q=256/5e5,
    T=15625, delta=1e-6 => nm=0.5622): extra same-batch release at sigma_h = 2nm -> eps 4.81 (or x1.165 grad
    noise if eps-matched); 4nm -> eps 3.51 / x1.041; 8nm -> eps 3.21 / x1.016. Joint PerGroup route with
    lambda/C = .1/.25/.5 -> grad sigma x1.049/1.118/1.225 (optimal alloc) — accounting literally unchanged
    (gaussian(nm)). An INDEPENDENTLY subsampled second release (own Poisson draw, forward-only pass) is much
    cheaper in eps (same-batch sigma_h=1: eps 1.00 vs independent 0.489 vs baseline 0.342 at nm=1) but is
    unavailable under b-min-sep (no second draw). Lagged/EMA use of previous releases is free (adaptive
    composition; same dependence as theta_t). Under DP-FTRL: put the load leaf in the same clipped pytree as
    a second PerGroup group; the constant-max_norm latch accepts it; accountant mf_gaussian(nm, strategy)
    unchanged; per-step realised sigma on the leaf = base sigma * ||row_t(C^-1)||; MF's prefix-sum accuracy
    makes a running average of the stream accurate "for free" (workload-matched filter is the momentum
    prefix sum, not an arbitrary EMA — unquantified, critic G7).
F7. Implementation seams (VERIFIED end to end at toy scale): a zero "probe" nn.Parameter z in R^E with
    loss += <z, f_x.detach()> makes d/dz = f_x a leaf of the clipped pytree; per_group(trainable,
    router_load_probe=lambda, fallback=C) clips it as its own group; gaussian_noise / mf_gaussian_noise
    allocate per-group sigma; on_pre_optimizer_step receives the NOISED pytree keyed by name (identical on
    all DDP ranks because the noise key is shared); _augment_inputs runs once per step OUTSIDE vmap and can
    inject the public f~_t as a broadcast tensor; probe must be zeroed each step / excluded from optimizer;
    EMA state serialises via the registry (frozen dataclass); checkpoint sidecar needed (fixed
    save_dp_runtime_state signature). Router logits are obtainable under vmap via a forward hook on
    MellumTopKRouter (F.one_hot is NOT vmap-safe; broadcast-compare is) or via model.model(...,
    output_router_logits=True); HF's own output_router_logits=True FAILS under vmap with an attention mask
    (in-place scatter_add_) and bypasses the chunked-CE memory path (full 98304-vocab logits).
F8. Numerics: Opaque vmap(grad) == HF per-example loop to 3.5e-7 rel-L2 in fp32 (dense and grouped MoE,
    sliding+full layers, padding, checkpointing, chunked CE). In bf16, Opaque-vs-HF drift 0.45-0.5% with ZERO
    route flips (accumulation order: HF grouped_mm bf16-accumulates, Opaque fp32-accumulates) — inside HF's
    own batched-vs-loop spread. bf16-vs-fp32 flips ~1% of tokens/layer independent of router sharpness
    (bf16 has fixed RELATIVE resolution; the router linear runs in bf16, only the softmax is fp32); flips
    cause ~8x excess gradient error on ROUTER/EXPERT params (12.9%/11.1% -> 1.7%/1.5% when routes pinned to
    fp32) and only +0.2pp on attention-only params. Mellum2's own tech report documents trainer/inference
    route disagreement for this checkpoint and used an FP32 router in pretraining. PR #980's "29% -> 1.3%
    oracle drift" is untraceable (no script in the repo; the mechanism — all_valid_attention SDPA kernel
    parity — is verified from the diff).
F9. DPTrainer default use_performance_kernels=False => DENSE every-token-through-every-expert Opaque_MoE on
    EVERY host including CUDA (8x the routed expert FLOPs for 64/top-8); grouped_moe=True opt-in via
    performance_kernels_config; the grouped flag is captured by the FIRST class-level patch per process.
    Presets: LoRA r=16 on q/k/v/o only (experts+router frozen; MoE backward skips expert weight grads),
    fixed clipping C=0.9 (train_dpftrl default), bf16 model (no autocast/master weights), microbatch 8,
    B=256, T=1024 (= sliding window, so windows never activate), band-MF bands=64, b-min-sep, eps=3.
    HF's aux gradient at coef 1e-3 was 2.4e-4 of the CE gradient at BALANCED random init (i.e. zero signal
    by F3) and 5x larger with induced imbalance — "negligible" is not a safe design assumption.
F10. Literature: exactly one prior DP-MoE paper (Tholoniat et al. 2024) — drops the aux, trains the router,
    lists per-sample decomposition and non-isotropic expert noise as OPEN problems; OLMoE dropped the aux
    during SFT/DPO with no collapse (balance loss even decreased); DeepSeek-V2/V3 and Wang et al. define
    per-sequence aux as a first-class variant; loss-free balancing (expert bias, sign of previous-batch
    count deviation) is what Mellum2 authors plan next; Andrew et al. Thm 1 / Ponomareva "DP-fy" §5(b)
    are the precedents for "release the batch statistic privately and use it as a constant"; Kong et al.
    ICLR'25 for "bound the coupling constants, not the batch gradient". Route discontinuity: ReMoE,
    ST-MoE (z-loss motivated by bf16 router round-off), DenseMixer (STE through top-k improves router
    gradient, +46% FLOPs — already paid by Opaque's dense path).
F11. Privacy hygiene already flagged: trainer logs un-noised per-example loss means (pre-existing); any
    un-noised f(B) logging/checkpointing would be an unaccounted release; only f-hat may be logged.
    Alignment collators right-pad (fully-masked-row issue cannot arise). Clipping mode in presets = fixed.

## Open decisions the design MUST make (critic G1-G10)
G1 Target objective: (a) logical-batch HF pooling; (b) HF-Trainer-realised per-microbatch f with G*coef;
   (c) Megatron per-layer running-average f (what pretraining saw); (d) per-sequence. Which mask, which CE
   weighting, which coefficient (1e-3 pretrain / 1e-4 Mellum2 SFT / 0), executed-route f (fp32) or HF's.
G2 Define the oracle + drift metric (rel-L2 AND route-flip counter separately) and give the script to
   reproduce PR #980's figure on the real checkpoint on GPU.
G3 Real-model statistics needed to anchor lambda/sigma_h/C: per-example grad-norm quantiles under the
   preset, ||f(B) - k/E|| on KStack at B=256, per-example expert usage at T=1024 — give the script.
G4 Exact per-step mechanism + accountant for BOTH stacks (DP-SGD Poisson; DP-FTRL band-MF b-min-sep),
   one cost table, renormalisation/clamp post-processing, adjacency statement (add/remove; x2 replace-one),
   Poisson realised-batch scaling (sum_e f-hat_e = k*|B_t|/B-bar).
G5 Router precision / pinning: fp32 router logits patch (pretraining-faithful) as Mellum default? routes
   from the current model inside vmap (already per-example constants) vs frozen base; f from executed routes.
G6 Dense-MoE default: should Mellum default grouped_moe=True on CUDA in DPTrainer; docs gap.
G7 MF-consistent smoothing of the load stream (bands=64): which filter inherits the strategy's guarantee.
G8 Scope: DP-DPO preset (aux over chosen+rejected, reference forward), z-loss, experts/router-trainable
   variants (PEFT target_parameters on stacked experts under functional_call+vmap is UNTESTED).
G9 Enumerate every new tensor (probe leaf, f-hat EMA, pinned routes) as public-post-processing vs
   private-internal; checkpoint only public state; add the release to the privacy statement.
G10 Hook mechanics under gradient checkpointing (recompute double-fire), microbatch chunks, torch.compile.
