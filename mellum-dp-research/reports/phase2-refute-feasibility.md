# Phase 2 — `refute-feasibility`: faithfulness / numerics / feasibility attack on the final design

Agent: `refute-feasibility`. Target: `phase2-final-design.md` (read in full). Repo `/home/user/opaque` @ `ef1abc5`, nothing
tracked modified. Torch `2.14.0+cu130` (CPU only), transformers `5.16.1`, 4 cores. Evidence tags: **VERIFIED** = I read the
cited lines or ran the cited script in this session; **PLAUSIBLE** = derived, not executed. My scripts and raw outputs are in
`scratchpad/research/refute-feasibility/`: `feas_check.py` → `feas_check.out` (the 2-layer CPU check, 27 s),
`mc_timing.py` → `mc_timing_15625.out` (b-min-sep accountant timing), `nm_bracket.py` (deterministic `nm_MF` bracket, 23 s).
Paths are relative to `/home/user/opaque`; HF = `.venv/lib/python3.11/site-packages/transformers/`.

Verdict in one line: **the mechanism is feasible and its mechanics survive every attack I could run (vmap, eager, module
backward, microbatching, `PerGroup` clipping/noise, MF latch, gradient checkpointing with the recorder — all exact); but the
design's faithfulness claim rests on a false premise about the preset (packed equal lengths), its implementation plan does not
reach the two presets it names (they are manual loops, not `DPTrainer`), the new forward kwarg would be silently swallowed on
the plain-`DPTrainer` default path, and the validation plan's `nm_MF` row assumes a calibration that did not finish in any
phase-2 run.** Verdict: `sound-with-fixes`.

---

## 0. What I ran (so the numbers below are reproducible)

`feas_check.py` builds the repo's tiny Mellum (`build_moe_model('mellum','cpu', num_hidden_layers=2, num_experts=8,
num_experts_per_tok=2, ...)`, `packages/opaque-patches/tests/transformers/models/_test_utils.py:109-170`, VERIFIED), freezes
everything except q/k/v/o (attention-only proxy for the LoRA presets), registers the zero probe `router_load_probe ∈ R^8`, and
**emulates the design's §9.1 change** exactly: a causal-LM `forward` that, on `opaque_router_logits=True`, calls the backbone
with `output_router_logits=True` (HF recorder `_can_record_outputs["router_logits"] = OutputRecorder(MellumTopKRouter, index=0)`,
`modeling_mellum.py:431-433`, VERIFIED) and computes the chunked CE with `linear_nll_sum_chunked`
(`kernels/_linear_ce_chunked.py:239-264`, VERIFIED) — no HF aux. The vmap-safe statistics helper is the design's
`router_load_and_probs` (layer loop, `softmax(z.float())`, `topk`, broadcast-compare one-hot, attention-mask weighting, common
denominator `L·T_x`). Loss `ℓ_x = CE_x + α·E·Σ_e (f̃_e − k/E)·P_e(x) + ⟨z, λ·d(x).detach()⟩` with `λ = ρ·C_g/Δ_h`,
`ρ = 0.02`, `C_g = 0.9`, a fixed non-uniform `f̃` (so the surrogate carries gradient). Batch `B = 4`, `T = 16`, **ragged** lengths
`(16, 12, 10, 16)`, right-padded, labels `−100` at pads (both repo collators right-pad, F11). All fp32.

Results (`feas_check.out`, VERIFIED):

| check | result |
|---|---|
| A. `vmap(grad)` through the recorder | runs; **2 router-logit tensors per example** (= L); `∂ℓ/∂z = λ·d(x)` exactly (`atol 1e-7`); `Σ_e h = k`, `Σ_e P = 1`; `max_x ‖d(x)‖ = 0.561 < Δ_h = 1.225` |
| B. vmap vs eager `torch.func.grad` loop; vmap vs plain `loss.backward()` on the module | `h`, `P` max-abs diff **0.0**; whole-gradient rel-L2 **0.0** for every example, both comparisons |
| C. `clipped_grad(per_group(trainable, router_load_probe=C_h, fallback=C_g), normalize_by=B̄=5, microbatch_size ∈ {None, 2, 3})` | probe leaf = `(λ/B̄)Σ_x d(x)` to **1.2e-10**; `group_norms["router_load_probe"]` max 0.0082 < `C_h = 0.0180`, clip rate **0**; microbatch 2 vs none rel-L2 **4.7e-8**, 3 vs none **0.0**; `max_norm.values = {probe: C_h/B̄, fallback: C_g/B̄}`; `Σ_e` probe leaf = 3.5e-10 |
| D. `gaussian_noise(nm=0.5622)` on that pytree | `σ_probe = 0.014454`, `σ_fallback = 0.102203` = closed form `nm·√(C_i·S)/B̄` to 1e-15; Mahalanobis `Σ(C_i/B̄)²/σ_i² · nm² = 1.0000`; `mf_gaussian_noise(band_mf_strategy(bands=4, momentum=0.95), n_steps=6)` accepts the two-group `PerGroup` across 3 steps (latch OK); realised σ ratios 1.139 → 1.245 → 1.248 (bands = 4) |
| E. gradient checkpointing (`gradient_checkpointing_enable()`, forced non-reentrant by `patches/torch/checkpoint/huggingface.py:31-56`) | vmap runs; **still 2 logits per example (no double append)**; `h` identical; whole gradient rel-L2 vs no-checkpoint **0.0**; the **surrogate gradient (α=1 minus α=0) through the captured logits inside the checkpoint region is carried exactly** (norm 1.2242 both ways, rel 0.0) |
| F. surrogate identity vs HF `load_balancing_loss_func` (`modeling_mellum.py:540-606`) on the **patched** model, ragged lengths | with `w_x = T_x/T_tot`: value 2.0256021 vs 2.0256023, gradient rel-L2 **2.2e-7**; with **equal example weights (the design's §1.4 choice)**: gradient rel-L2 **15.6 %** |
| G. HF's own `output_router_logits=True` under vmap with a mask | `RuntimeError: vmap: scatter_add_ …` — the design's rejection (§7 last row) is right |

`mc_timing.py`: `b_min_sep(mf_gaussian(1.0, band_mf_strategy(64, 0.95)), n_steps=n, p0=256/5e5).epsilon_at(1e-6,
mc_resolution=1e-5)` — the accountant clamps the resolution to `min(1e-5, δ/2) = 5e-7` (`opaque-accounting/.../core/_base.py:216-221`,
VERIFIED) and warns it needs **64,997,003 transcripts per adjacency direction**; killed at the 280 s bound at n = 15625 and at the
150 s bound at **n = 256** (exit 124 both; synthesizer's own run: killed at 570 s). `nm_bracket.py`: the deterministic un-amplified
bound `mf_gaussian(nm, band_mf_strategy(64, 0.95), n_steps=15625, min_sep=64, max_participations ∈ {8, 245})` gives
`strategy.sensitivity = 1.0000` and **nm = 1.5439 at ε = 3, δ = 1e-6** (23 s wall).

---

## 1. Refuted claims (each with why, severity, fix)

### R1 — "equal lengths (`T_x ≡ T`, true for the packed T = 1024 presets)" (§1.2, §1.4 CE-weighting row, §0.2 (b)) — **major**

**Why.** The `mellum2-kstack` preset does not pack. `examples/train_dpftrl.py:1090-1094` tokenises with
`truncation=True, max_length=args.max_seq_len` (1024, `:889`) and no grouping/packing; the collator pads (`:1133-1135`) and the
loss masks pads to `−100` (`:1376`) — i.e. **ragged, right-padded** rows (VERIFIED read). KStack is a corpus of Kotlin files;
any file under 1024 tokens is shorter than the row. Under ragged lengths the design's objective — equal example weights for the
aux, "the released statistic is therefore the *example-mean* load `(1/B̄)Σ_x d(x)`" (§1.4) — is **not** HF's objective: HF pools
`f` and `P` over tokens with one denominator `L·T_tot` (`modeling_mellum.py:575-603`). My check F quantifies it on the patched
model at a 16/12/10/16 length spread: the `T_x/T_tot`-weighted surrogate reproduces HF's aux gradient to 2.2e-7, the design's
equal-weight surrogate is **15.6 % rel-L2 off** (VERIFIED). The design's own phase-1 evidence (E4 "ragged lengths … 3.3e-7")
was computed *with* the `T_x/T_tot` reweighting, which the design then dropped for the preset on the false premise that the
preset is packed. Requirement (b) is therefore "VERIFIED (identity)" only for the equal-length case; at the preset it is
PLAUSIBLE-at-best with a quantified toy deviation of order the length spread. (The same premise also hides the pre-existing
CE weighting mismatch F4 — example-mean vs token-mean CE — which the design correctly attributes to Opaque's convention but
wrongly declares inactive for the preset.)

**Fix (pick one, all concrete).** (i) **Pack the preset** (concatenate tokenised files with EOS/FIM separators into fixed
1024-token rows — how Mellum2 itself is trained; TR §3, phase-1 literature F.2): `T_x ≡ 1024`, both identities exact, no
mechanism change, and the CE convention question disappears. (ii) Keep ragged rows and implement the **public-constant token
weighting** the design lists as "out" (§7): `w_x = T_x/N̄` with `N̄ = B̄·T_max` (public; `T_max = 1024`) on *both* the surrogate
and the probe term `⟨z, λ·w_x·d(x)⟩`. The load bound stays structural (`‖w_x d(x)‖ ≤ Δ_h·T_x/T_max ≤ Δ_h`), the released
statistic is `(1/B̄)Σ_x (T_x/T_max) d(x)`, an unbiased estimate of `(T̄/T_max)·d_token-weighted`; either divide by a public
proxy `T̄/T_max` or (iii) add a **third `PerGroup` group** carrying the scalar `T_x/T_max` (bound 1, ρ′ ≈ 0.005) and form the
token-weighted `f̂` by post-processing division — this is exactly HF's pooling. Re-label §0.2 (b) accordingly and make §10.1
(A5) run on a ragged microbatch.

### R2 — §9.2's preset rows (`examples/train_dpftrl.py mellum2-kstack → router_load_release="surrogate"`, `examples/train_dpo.py mellum2-codesec → "monitor"`) — **major**

**Why.** Neither example uses the trainers the design instruments. `train_dpftrl.py` is a manual functional loop: imports
`clipped_grad`, `per_group`, `auto_clipped_grad` (`:136`), `mf_gaussian_noise` (`:144`), `BMinSepSampler` (`:148`); builds its
own `per_example_loss_fn(trainable, input_ids)` (`:1369-1378`), its own clip fn (`:1427-1454`), its own noise fn (`:1735-1760`),
its own calibration (`:1678-1685`) and logging; `train_dpo.py` likewise (`clipped_grad`/`gaussian_noise`/`PoissonSampler`/`per_group`
at `:121-148`, `per_example_loss` at `:546-560`) — VERIFIED read; neither file references `DPTrainer`/`DPDPOTrainer`. So
`_setup_training`, `_augment_inputs`, `RouterLoadCallback.on_pre_optimizer_step`, the TRL converter and the checkpoint sidecar
(§9.2) never execute for the two presets that motivate the work, and `router_load_release=` is not an argument these scripts
have. Additionally the kstack loop passes **no attention mask** to the model (`:1377`), so "mask = attention mask" (§1.4) is
vacuous there (see R6).

**Fix.** Factor the mechanism into a trainer-independent helper with four seams — e.g. `opaque.api.transformers.moe_load`
(or in `opaque-patches` if it must stay torch-only): `attach_probe(model, E) -> name`, `probe_group(clip_norm, ratio) ->
PerGroup`, `RouterLoadState` + `update(state, noised_leaf, lam)`, `state_dict` — and call it from both manual loops at
(1) parameter registration before `make_functional`, (2) `per_group(...)` construction, (3) between `noise_fn` and
`opt.update` (zero the leaf in place, update `f̃`), (4) checkpoint save/restore; thread `attention_mask` into
`per_example_loss_fn`. Or migrate the presets onto `DPSFTTrainer`/`DPDPOTrainer` — then §9.2 applies unchanged, but that is a
larger, separate change. Either way §9.2's last row must say which.

### R3 — "`DPTrainer.compute_per_example_loss` reads the loss … through the chunked LM-head path (`chunked_linear_cross_entropy=2048`, `models/mellum.py:44`)" (§1.1) and "marker detected by signature exactly like `_fused_forward_uses_marker`" (§9.2) — **major**

**Why.** The chunked/fused causal-LM forward is installed **only** when the *caller* passes `fused_linear_cross_entropy=True`:
`_factory.py:378-392` — `if (fused_linear_cross_entropy or chunked_linear_ce) and kwargs.get("fused_linear_cross_entropy",
False) and causal_lm_obj is not None: _patch_forward(...)` (VERIFIED). The Mellum recipe's `chunked_linear_cross_entropy=2048`
(`models/mellum.py:44`) only selects `force_chunked` *inside* that branch. `DPTrainer` passes `kwargs =
self.args.performance_kernels_config or {}` (`_dp_trainer.py:841-847`, VERIFIED) — nothing unless the user sets it;
`DPSFTTrainer` sets it only for `chunked_nll` (`_sft_trainer.py:245`); the example passes it explicitly (`train_dpftrl.py:1068`).
My tiny build via `build_moe_model` confirms: `causal_lm_forward_is_opaque_patched: false` (`feas_check.out`). So on the
plain-`DPTrainer` default path for Mellum the forward is HF's original (full 98304-vocab logits, the PR #978 memory path:
`98304×1024×4 B = 403 MB` fp32 per example, 3.2 GB per microbatch of 8 before the logits' gradient), and the design's new
`opaque_router_logits=True` kwarg does not exist there. Worse, the proposed detection copies `_fused_forward_uses_marker`
(`_sft_trainer.py:287-291`: true if the parameter is named **or** the signature has `**kwargs`) — HF's forward has
`**kwargs: Unpack[TransformersKwargs]` (`modeling_mellum.py:641`), so the marker would be **passed to and silently swallowed by
the unpatched HF forward**, which returns `router_logits=None`; the helper then fails on `None` (or, if it is written to tolerate
`None`, the feature silently degrades to α = 0 with the probe releasing zeros). Not a privacy bug (zeros are still a valid
release) but a feasibility/faithfulness bug.

**Fix.** (a) Detect the *named* parameter only (`parameter.name == "opaque_router_logits"`; never `VAR_KEYWORD`), and at setup
require `hasattr(type(model).forward, "__opaque_patched__")` (the `_router.py:59-92` marker) — else `ConfigurationError`
naming `performance_kernels_config={"fused_linear_cross_entropy": True}`. (b) When `router_load_release != "off"`, have
`DPTrainer._apply_model_patches` add `fused_linear_cross_entropy=True` itself for families with `chunked_linear_cross_entropy`
set (or fix the pre-existing gotcha at the source: let a recipe-level `chunked_linear_cross_entropy` install the chunked forward
without the caller flag). (c) T4 must include a case where the flag is absent and assert the error.

### R4 — "the preset's calibrated `nm_MF` … (the trainer computes it at start) … the number is free to record there" (§6.4, §10.2 last row) and the `≈ 30 min` validation budget (§10) — **major for the validation plan, not for the mechanism**

**Why.** The preset's calibration is `cal.calibrate` over `b_min_sep(mf_gaussian(·, band_mf(64, 0.95)), n_steps=15625, p0)`
(`train_dpftrl.py:1613-1685`). Its MC PLD needs 64,997,003 transcripts per adjacency direction because `epsilon_at` clamps
`mc_resolution` to `δ/2 = 5e-7` (`_base.py:216-221`) **regardless of the example's `--mc-resolution` default 1e-5**
(`train_dpftrl.py:704-707`; VERIFIED). One evaluation did not finish in 280 s at n = 15625 nor in **150 s at n = 256** on 4 cores
(my runs), nor in 570 s (synthesizer). The transcript corpus is built once per `(coef, n, p, num_samples, seed)` and reused per σ
(`src/amplification/b_min_sep/registry.rs:38-60, 75-110`, VERIFIED read), so calibration ≈ one corpus build + one PLD per
bisection step — which part dominates I could not measure (PLAUSIBLE: the corpus). Nothing in phase 2 bounds this wall-clock;
"free to record" is unsupported, and a 200-step run "at ε = 3 calibrated" under band-MF at n = 200 is a *different* `nm` than the
preset's. The design also leaves `nm_MF` unbounded ("several × larger").

**Fix.** (a) Bracket it now, deterministically: the **un-amplified** band-MF(64, 0.95) bound at n = 15625, `min_sep = 64`
(sensitivity exactly 1.0000 for both `max_participations = 8` and 245) calibrates to **nm = 1.5439 at ε = 3** (`nm_bracket.py`,
23 s, VERIFIED). Amplification can only reduce the required `nm` (a looser MC upper bound could in principle exceed this,
PLAUSIBLE caveat), so `nm_MF/0.5622 ≤ 2.746`: every band-MF column of §2.6 scales by at most 2.75× — default row smoothed
error **≤ 2.28 % of k/E** (vs DP-SGD 2.35 %), shrinkage threshold δ ≤ 0.022. Put this bracket in §2.6 in place of "several ×".
(b) In §10, budget the calibration explicitly (record wall-clock; run it once offline on a many-core host; pass
`--noise-multiplier` to the validation runs), and for the 200-step run either use `n_steps = 200` calibration and say so, or
use the bracket `nm = 1.544` as a conservative fixed multiplier. (c) Consider exposing `--target-delta`-aware
`mc_resolution` (the clamp makes the CLI flag inert at δ ≤ 2e-5).

### R5 — "router logits tuple (recorder): 0 extra — already live in the autograd graph (softmax → top-k weights)" (§5.2) — **minor (wording; conclusion unchanged)**

**Why.** `softmax` saves its *output* for backward and `F.linear` saves its *inputs*; the router-logit tensor itself is not
retained by autograd, so the recorder's references are the thing keeping it alive: `28 × 1024 × 64 × 2 B = 3.67 MB` per example
(bf16), **29.4 MB per microbatch of 8**; the statistics' fp32 softmax saved for the surrogate's backward adds `7.34 MB`/example
(**58.7 MB** per microbatch); with the opt-in fp32 router the logits are `7.34 MB`/example (58.7 MB). The `(T, k, E)` one-hot
compare is a 0.5 MB transient per layer. Total < 0.15 GB per microbatch against ~24 GB of bf16 weights — the asked-for
"(L, T, E) fp32 stash at T = 1024, L = 28, E = 64, microbatch 8" is **58.7 MB** and is not a constraint. (VERIFIED arithmetic;
the retention semantics VERIFIED by check E: the captured logits survive the checkpoint region and carry gradient.)

**Fix.** Replace "0 extra" with the numbers above.

### R6 — "mask for `h`, `P`: attention mask … HF passes `attention_mask` to the aux" (§1.4) — **minor**

**Why.** HF passes whatever the caller passed; the kstack loop passes none (`train_dpftrl.py:1377`: `fmodel(params, input_ids,
labels=labels)`), so HF-faithful behaviour *there* is `attention_mask=None`: pad rows are routed (pad-token embeddings) and
counted in both `f` and `P` (`modeling_mellum.py:583-592`). The helper's `inputs.get("attention_mask")` → `None` reproduces that,
but it is not what the design describes, and it silently differs from the `DPTrainer`/SFT collator path (mask present).

**Fix.** State both cases; thread the collator's mask into the manual loop (also correct for SDPA under right padding), and make
T2 cover `attention_mask=None`.

### R7 — §11.10 falsifier "eval with bf16 routing after an fp32-router fine-tune" and §10.1 (m6) — **minor (executability)**

**Why.** The fp32 router is proposed as a `classes["router"]` role installed by `_patch_forward` — a **class-level** replacement
guarded by `__opaque_patched__` (`_router.py:59-92`, VERIFIED): once applied in a process it cannot be toggled off, so an
in-process bf16-routing eval of the fp32-trained adapter (and the m6 pair) is not possible; it needs a second process or an
instance-level swap. Faithfulness reading (asked by the task): the fp32 router changes the *executed routing function* on
≈1 %/layer of tokens (phase-1 E2) but not the saved weights — adapters exported and served through stock HF run bf16 routes;
the design's "opt-in, default off, train/serve mismatch documented" is the right call (upheld, U9).

**Fix.** Install the fp32 router as an instance-level `types.MethodType` swap that is removable (or accept a process boundary
in §10/§11 and say so).

### R8 — §10 budget "≤ 30 min for the oracle + statistics" including the fp32 floor O1 — **minor**

**Why.** `opaque_moe` on CUDA + Triton routes only bf16/fp16 to the fused kernel; the `else` at `kernels/moe.py:614-632` belongs to
`if _TRITON_AVAILABLE and x.is_cuda`, so **CUDA fp32 falls to the dense `Opaque_MoE`** (8× routed expert FLOPs, §5.1) — the O1
loop on the 48 GB fp32 model runs dense (VERIFIED read). Fine for a floor, but budget it (or run O1 with `grouped_moe=True`
forced to the `_grouped_mm` route, which the dispatcher never selects on CUDA+Triton).

---

## 2. Upheld (what I tried and could not refute)

- **U1 Recorder under `vmap(grad)`.** HF's `capture_outputs`/`OutputRecorder` ContextVar mechanism (`utils/output_capturing.py:104-115,
  259-272`, VERIFIED read) works inside `torch.func.functional_call` + `vmap(grad)`: L tensors of shape `(T, E)` per example, probe
  gradient exactly `λ·d(x)` (check A).
- **U2 Numerics vs the non-DP path at equal precision.** vmap vs eager functional loop vs plain `loss.backward()` on the same
  patched module: **0.0** rel-L2 (check B). The design's "precision-matched oracle" definition (§10.1) is the right one.
- **U3 `clipped_grad` + `PerGroup` + microbatching.** Probe leaf exact (1e-10), never clipped (structural bound 0.0082 ≪ 0.0180),
  microbatch-invariant (≤ 4.7e-8), `Σ_e = 0` (check C). Microbatch chunks are separate vmapped calls (`_clipped_fun.py:273-296`,
  VERIFIED), so separate collectors — G10's chunk question is closed.
- **U4 Allocator and MF latch.** `gaussian_noise` per-group σ = closed form to 1e-15; Mahalanobis identity = 1.0000;
  `mf_gaussian_noise` accepts the constant two-group `PerGroup` across steps and reports the per-leaf realised σ (check D).
- **U5 Gradient checkpointing (design risk 4 / T5, marked PLAUSIBLE).** With the repo's non-reentrant glue: **no double append**
  (still L logits), identical `h`, identical gradients, and the surrogate gradient through logits captured *inside* the
  checkpoint region is carried exactly (check E). Upgrade T5's status to VERIFIED at toy scale on torch 2.14; the fallback
  hook-with-overwrite route is not needed.
- **U6 Surrogate identity on the patched model with the chunked-CE emulation**, ragged lengths, `T_x/T_tot` weights: 2.2e-7
  (check F) — F3 holds end to end through the design's forward path, not only on the unpatched HF model.
- **U7 Rejecting HF's `output_router_logits=True` under vmap** — reproduced the `scatter_add_` failure (check G).
- **U8 Memory of the statistics path** — 58.7 MB fp32 stash per microbatch of 8 (R5 numbers); not a blocker. The design's
  "peak memory stays that of PR #978" holds *iff* the chunked forward is installed (R3).
- **U9 fp32 router faithfulness framing** — changes executed routing, not weights; opt-in/default-off is right (R7 for the
  toggling caveat). Cost arithmetic 0.30 % (design §3) not re-derived.
- **U10 DPO row feasibility** — `_fused_logp` calls the backbone functionally through `_last_hidden_state`
  (`_dpo_trainer.py:998-1063`, VERIFIED read), so `output_router_logits=True` there is a one-line addition and the reference
  forward (`:807-841`) is untouched; the pooling rule is implementable as written.
- **U11 grouped-MoE default (G6) risks are contained but not free.** `Opaque_GroupedMoE` has a vmap rule (`_grouped_moe.py:593`)
  and CPU/MPS tests cover `vmap(grad)` parity incl. frozen experts (`tests/kernels/test_grouped_moe.py:59-72, 113-116, 145-148`,
  VERIFIED read; bf16 CPU tolerance is Frobenius 1e-2 at `:116`). The proposed `_grouped_route_available()` is true on this CPU
  build too (`torch._grouped_mm` exists, VERIFIED), so the *entire CPU MoE test/CI surface* flips to the grouped route for
  E ≥ 16 subject to the workspace gate (`_moe_memory.py:120-134`) — the design's risk 6 should name CPU CI explicitly and the
  fused Triton kernel's "perf comparison pending — #417" (`fused_moe.py:19-20`) is performance-only. First-patch capture
  (`_router.py:59-92`) is real and documented. No refutation.
- **U12 Executed-route `f`.** `h` from `topk(softmax(z.float()))` is the same op sequence as `MellumTopKRouter.forward`
  (`modeling_mellum.py:333-336`: `softmax(router_logits, dtype=torch.float)` then `topk`), so identical input ⇒ identical indices;
  the deviation from HF's bf16-softmax aux top-k (critic R9) is the design's documented choice.
- **U13 Layer pooling and coefficient.** HF's one-denominator pooling (`modeling_mellum.py:575-603`) is what the helper does
  (common `L·T_x` per example; `T_x/T_tot` across examples per R1 fix); `Σ_e f = k` (check F: `fB.sum() = 2.0`);
  α default = config 1e-3 / presets 1e-4 is explicit and reasonable.
- **U14 `torch.compile` path** — not tested (CPU inductor would exceed the budget); the design correctly labels it PLAUSIBLE with
  the fullgraph fallback. Not refuted, not upheld.

---

## 3. Feasibility answers to the task's specific questions

| question | answer |
|---|---|
| Does the per-example loss reproduce the target objective? | Yes for equal lengths and for ragged lengths **with `T_x/T_tot` weights** (2.2e-7); **no** at the 15 % level with the design's equal-example weights under ragged lengths, and the preset is ragged (R1). Layer pooling, mask semantics (given a mask), executed routes and coefficient all match HF. |
| fp32 router: model change at inference or training numerics only? | Executed-function change (route selection at bf16-resolution ties, ≈1 %/layer), no weight change; serving through stock HF uses bf16 routes ⇒ train/serve routing mismatch by construction; opt-in default off is correct (U9/R7). |
| Does the hook/recorder path run under `vmap(grad)` + chunked CE + checkpointing + microbatching? | **Yes, all four, exactly** (checks A–E), *provided the chunked forward is actually installed* (R3). |
| Memory of the `(L, T, E)` fp32 stash, T = 1024, L = 28, E = 64, microbatch 8? | 58.7 MB (fp32) / 29.4 MB (bf16 logits), + 58.7 MB fp32 softmax for the surrogate backward; negligible (R5). |
| `grouped_moe` default risks? | Contained by existing vmap(grad) parity tests; flips CPU CI too; CUDA fp32 still dense (U11, R8). |
| Is the validation plan executable? | Mostly, with edits: the two presets are not `DPTrainer` runs (R2); the `nm_MF` row needs a budget or the 1.544 bracket (R4); O1 runs dense (R8); the bf16-routing eval of an fp32-router run needs a process boundary (R7); A5 must include a ragged microbatch (R1). |

---

## 4. Sources relied on

Repo lines as cited (all VERIFIED by `sed -n`/`grep` this session). Primary literature is not re-fetched here; I make no
theorem-dependent claim beyond those in the design (Switch Transformer eqs. (4)–(6), https://arxiv.org/abs/2101.03961 §2.2, for
the definition of `L_aux`, as implemented in `modeling_mellum.py:540-606`). The "amplification can only lower `nm`" step in R4 is
the standard monotonicity of subsampling amplification under add/remove adjacency — PLAUSIBLE as applied to the accountant's
MC upper bound, stated with its caveat.
