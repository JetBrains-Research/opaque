# Phase 2 — judge-impl: implementability and completeness review of the four Mellum2 DP designs

Agent: `judge-impl`. Lens: IMPLEMENTABILITY AND COMPLETENESS (weighted 1.5x). Repo `/home/user/opaque` @ `ef1abc5`
(branch `claude/mellum-dp-representation-r6slaz`), nothing tracked modified. Inputs read in full: BRIEF.md,
PHASE1-DIGEST.md, `.junie/differential-privacy-review.md`, phase1-{critic,primitives,math}.md, the four designs.
The other two judge reports were skimmed for their error lists after my own pass, so overlaps are noted as such.

Tags: **VERIFIED** = I read the cited lines with `sed -n` or ran the cited script in this session; **PLAUSIBLE** =
consistent with what I read, not executed; **VERIFIED (phase-1 <agent>)** = established by a named phase-1 script.
Repo paths relative to `/home/user/opaque`; HF = `.venv/lib/python3.11/site-packages/transformers/`.
My one script: `scratchpad/research/judge-impl/mf_rownorm.py` (CPU, 6 s).

---

## 0. Verdict in one paragraph

All four designs share the same correct core (probe leaf as a second `PerGroup` group of the same clipped pytree,
accountant call literally unchanged under both stacks, lagged public `f̃`), and all four address every gap G1–G10 with
a dedicated section. They differ in *surface area* and in *how much of the plan touches seams I could verify*. Under
this lens **minimal wins narrowly**: engine/accounting/dpftrl untouched, every trainer seam it uses exists exactly as
cited (`call_event("on_pre_optimizer_step", …, grads=noisy_grads, …)` at `_dp_trainer.py:2198-2206`, `_augment_inputs`
`:2302-2312`, clip-dict → `PerGroup` `:1455-1470`, `make_functional` `:1357-1361`, sidecar around the fixed
`save_dp_runtime_state` signature `_checkpoint.py:311-333`), its explanation of why HF's `OutputRecorder` does not
double-fire under checkpoint recompute is the only one that is *right about the mechanism* (`output_capturing.py:259-272`,
VERIFIED), its closure-tensor route for `f̃` sidesteps a column-pruning wrinkle the other three inherit, and it ships
19 placed tests. Its one real defect — the band-MF filter table computed for the factory-default strategy
(`momentum=1.0`) instead of the preset's (`0.95`) — is shared with skeptic and is fixed by grafting optimal's numbers
(which I reproduced exactly). Optimal is the most complete analytically but the widest to implement (engine change,
independent-draw sampler + accounting plumbing, LFB serving patch, DPO). Faithful is close to minimal but adds a
checkpoint-signature change, DPO, and a matched-filter choice 2–3x noisier than necessary. Skeptic is the leanest
and contributes the one genuinely new experimental fact (fp32 router logits do not remove flips) plus the monitor,
but sets α=0 by default, which under-delivers on the brief's "represent the batch-level parts".

---

## 1. Citation spot-checks (>= 5 per design, all with `sed -n` this session)

Legend: OK = line range contains what the design says; ~ = right content, range off by a few lines; X = wrong.

### 1.1 faithful
| citation | what it claims | result |
|---|---|---|
| `_dp_trainer.py:1456-1468` | dict `clipping_norm` → `per_group(...)` | OK (`:1455-1470`) VERIFIED |
| `_dp_trainer.py:1357-1361` | `make_functional(partition_trainable=True)` | OK VERIFIED |
| `_dp_trainer.py:2302-2312` | `_augment_inputs` runs once per step outside vmap | OK VERIFIED |
| `_dp_trainer.py:2210-2218` | `on_pre_optimizer_step` receives noised pytree | ~ (actually `:2198-2206`) VERIFIED |
| `_checkpoint.py:311-333` | `save_dp_runtime_state` fixed keyword signature | OK (`:311-335`) VERIFIED |
| `cross_entropy.py:212-234` | `output_router_logits` fallback to original forward | OK (`:212-233`) VERIFIED |
| `modeling_mellum.py:432,475` | `OutputRecorder(MellumTopKRouter, index=0)`, `@capture_outputs` | OK VERIFIED |
| `_band_mf.py:35-60,86-91` | momentum workload coefficients, `optimize_toeplitz` call | OK VERIFIED |
| `_clipped_fun.py:255-260` "chunk path" | microbatch loop | ~ (loop at `:273-276`) VERIFIED |
| `examples/train_dpftrl.py:495-498,1563-1581` | workload momentum 0.95 passed to `band_mf_strategy` | OK VERIFIED — this is what makes its MF numbers the right ones |

### 1.2 minimal
| citation | what it claims | result |
|---|---|---|
| `_dp_trainer.py:1394` | `normalize_by = expected_batch_size = a.train_batch_size` | OK VERIFIED |
| `_dp_trainer.py:2210-2218` | `call_event("on_pre_optimizer_step", grads=noisy_grads, trainable_params=…)` | ~ (`:2198-2206`) VERIFIED |
| `output_capturing.py:266-272` | collector `ContextVar` set before backbone forward, reset in `finally` | OK (`:259-272`) VERIFIED |
| `output_capturing.py:97` | `CompileableContextVar` | OK VERIFIED |
| `functional/__init__.py:189-214` | `_squeeze_output` squeezes only top-level tensors | OK VERIFIED |
| `cross_entropy.py:359-370` | `hasattr(outputs,"router_logits")` return path builds `MoeCausalLMOutputWithPast` | OK (`:359-371`) VERIFIED |
| `_sft_trainer.py:287-291, 576-590` | `_fused_forward_uses_marker`, chunked-NLL path | OK VERIFIED |
| `_factory.py:316-324`, `_router.py:59-92` | `grouped_moe = kwargs.get("grouped_moe", kernels)`; first-patch capture | OK VERIFIED |
| `_b_min_sep.py:6-11` | `p = p0/(1 - p0(b-1))` | OK VERIFIED |
| `trl/_convert.py:69-76` | `_drop_router_aux_loss` warning | OK VERIFIED |

### 1.3 optimal
| citation | what it claims | result |
|---|---|---|
| `_dp_trainer.py:4267/4277/4288` | `normalize_by=expected_batch_size` in all three clip branches | OK (`:4266,4277,4288`) VERIFIED |
| `_dp_trainer.py:2197-2199` | `on_pre_optimizer_step` seam | ~ (`:2198-2206`) VERIFIED |
| `_dp_trainer.py:296` | `_account_independent_step` composes `ctx.step_process` | OK (`:295-299`) VERIFIED |
| `_mf_gaussian_noise.py:163-186` | `per_group_noise_stddev` base σ then `C⁻¹`; realised σ = base·row_l2 | OK (`:163-192`) VERIFIED |
| `_engine.py:473-493` | constant-max_norm latch | OK (`:473-517`) VERIFIED |
| `_auto.py:117` | `auto_clipped_grad` | X-ish: `:117` is `R: float|PerGroup` inside another signature; `auto_clipped_grad` is at `:203` |
| `_pytree.py:350` | `auto_scale_pytree` | OK VERIFIED; `_auto_scale_per_group` at `:313-347` with `clamp_to_one=False` VERIFIED |
| `_training_arguments.py:429-435,436` | kernel docstring lists rope/rms_norm/activation/cross_entropy, no MoE; default False | OK VERIFIED |
| `_dpo_trainer.py:807,1065` | TR-DPO `_augment_inputs`; `compute_per_example_loss_and_metrics` | OK VERIFIED |
| `_toeplitz.py:448-520` | b-min-sep sensitivities from Toeplitz coefs | ~ (`sensitivity_squared` at `:448`) VERIFIED |

### 1.4 skeptic
| citation | what it claims | result |
|---|---|---|
| `modeling_mellum.py:323-341, 350-355, 692-700` | router forward; block discards logits; aux only under `output_router_logits` | OK VERIFIED |
| `_dp_trainer.py:4252` | `target_clipping_rate` default 0.5 | OK (`:4251`) VERIFIED |
| `_dp_trainer.py:2258-2272` | un-noised `group_metrics` per group | OK (`:2259-2275`) VERIFIED |
| `_checkpoint.py:201-208` | "adding a new field is a single edit" docstring | OK VERIFIED (but see error E11) |
| `_band_mf.py:60, 142-171` "momentum 1.0 is the preset default" | | X — factory default is 1.0 (`:145`), but the preset passes `_workload_momentum()` = `args.momentum` = 0.95 (`train_dpftrl.py:459` default optimizer `sgd`, `:495-498`, `:1563-1581`) VERIFIED |
| `examples/train_dpftrl.py:611-612`; `grep grouped_moe examples/` no hits | fixed C=0.9; presets never set `grouped_moe` | OK VERIFIED |
| `moe.py:536-539, 596-598, 610-612, 614-632` | `needs_input_grad`; chunked temporaries; dispatch | OK VERIFIED |
| `_state.py:537` | `register_sync_type` | OK VERIFIED |

---

## 2. Composition checks (the lens's core question)

For each interaction I record what the code actually does and which designs get it right.

**Chunked CE.** `_make_fused_ce_causal_lm_forward` takes the HF fallback whenever `output_router_logits` is truthy
(`cross_entropy.py:212-233`) and otherwise calls the backbone with explicit args **plus `**kwargs`**
(`:252-264`, VERIFIED). So (i) HF's own `output_router_logits=True` on the causal-LM still hits the full-vocab
path (F7), and (ii) any new kwarg must be a *named* parameter of the patched forward (so it is not in `**kwargs`)
and must set `output_router_logits=True` only on the backbone call. faithful (`opaque_router_logits_only`), minimal
(`opaque_router_logits`) and optimal (`opaque_router_stats`) all specify exactly this; minimal is the only one that
also names the existing `hasattr(outputs,"router_logits")` return path (`:359-371`) that already carries the tuple out
with `aux_loss=None`. skeptic bypasses the forward entirely (hooks) so chunked CE is untouched by construction.
Backbone-only `output_router_logits=True` under vmap is VERIFIED (phase-1 divergence exp1/exp4, `phase1-divergence.md:351`).

**Gradient checkpointing.** HF installs the recorder hooks once (`maybe_install_capturing_hooks`,
`output_capturing.py:255-258`) and they stay registered; a hook *appends only while the `ContextVar` collector is
set*, which `capture_outputs` does immediately before `func(self, …)` and resets in `finally` (`:259-272`, VERIFIED).
Consequently a non-reentrant recompute during backward runs the router hook with `collected_outputs is None` and
appends nothing. minimal states exactly this (§9.1) — **correct**. faithful says the hooks are "registered for the
duration of one forward call" — wrong mechanism, right conclusion (E6). optimal says "under recompute it fires again
… nothing reads it" — wrong mechanism (it fires but is inactive), harmless (E6). skeptic's own hooks *do* fire on
recompute and it handles that with overwrite-by-layer-index + reset-per-call — correct for its route. Whether the
captured tensors carry gradient through the non-reentrant region is PLAUSIBLE for all four (HF trains MoE aux under
checkpointing this way in eager; under vmap + Opaque's `_force_non_reentrant`, `checkpoint/huggingface.py:31-35`, it is
untested — every design has a test for it).

**Microbatching.** `clipped_fun` slices the batch and calls the vmapped function per chunk (`_clipped_fun.py:273-276`,
VERIFIED); the probe leaf accumulates like any leaf; recorder/hook state is per call. All four correct.

**DDP.** `sum_gradients_` all-reduces every leaf pre-noise and asserts identical `max_norm`
(`gradients.py:150-162` VERIFIED read start; primitives §2.3); the shared noise key (`_dp_trainer.py:1478` VERIFIED)
makes the noised probe leaf rank-identical. All four correct; minimal and optimal additionally note no reduction of the
filter state is needed. optimal's independent draw needs a rank-sharded second sampler (own key, own serialization) —
acknowledged as an omission by judge-dp; I agree it is extra surface.

**torch.compile.** `_grad_compiler` compiles the *transform* and falls back to `fullgraph=False`
(`_dp_trainer.py:4191-4228`, `_compile_with_fullgraph_fallback` at `:198`, VERIFIED). A Python-side list/dict capture
graph-breaks; HF's `CompileableContextVar` (`output_capturing.py:97`) is friendlier. All four say this; minimal and
optimal prefer the recorder for that reason; skeptic accepts the break. Nobody executed it (each has a test).

**MF latch.** `_validate_constant_max_norm` compares `max_norm` by equality including `PerGroup`
(`_engine.py:473-517`, VERIFIED); a constant two-group `PerGroup` passes (primitives E2). All four correct. skeptic's
mid-run α switch changes only the loss, not `max_norm` — correct.

**Fixed `save_dp_runtime_state` signature.** VERIFIED (`_checkpoint.py:311-335`, 23 keyword params, no extension
slot; call site `_dp_trainer.py:5091-5122`; restore `:5335-5358` restores fields *by hand*). Routes: minimal and
optimal write a sidecar next to `DP_STATE_NAME` (no signature churn); faithful adds an `extra_state` slot (signature +
dataclass + restore edits); skeptic adds a typed field to `RuntimeCheckpoint` — the docstring invites that, but it is
three edits, not one (E11). All implementable; sidecar is least invasive.

**Delivery of `f̃_t` into the vmapped loss.** faithful/optimal/skeptic use a seeded batch column (`load_target`,
TR-DPO pattern `_dpo_trainer.py:719-750`, VERIFIED). One wrinkle none of them mentions: `DPTrainer._remove_unused_columns`
(`_dp_trainer.py:3446-3470`, VERIFIED) prunes dataset columns not in the model forward signature when
`remove_unused_columns=True` (HF default). The SFT/DPO configs set it `False` (`_sft_config.py:113`,
`_dpo_config.py:139`, VERIFIED), so the pattern works there, but plain `DPTrainer` users would need the column named in
the forward or pruning disabled. minimal's primary route (a device tensor on the trainer read as a closure constant
inside the loss, updated in `_augment_inputs`) avoids the collator entirely — the simplest implementable route; its
compile behaviour (closure tensor lifted as a graph input) is PLAUSIBLE.

**Telemetry side channel.** `_dp_trainer.py:2259-2275` logs, for every `PerGroup` group, the un-noised batch mean
of the per-example group norms and a clip rate (VERIFIED). With a probe group this becomes an unaccounted release of
`mean_x ‖λ d(x)‖`. minimal (§8, T17) and skeptic (§8, §9.2) suppress the probe group; faithful and optimal mark
`group norms` "as today" (E5). Implementation-wise the fix is one `continue` in that loop.

**Second-moment streams.** `paired_noise_stddevs` sums Δ over all groups (primitives §2.1/§8(b)); faithful (T6)
excludes the probe; skeptic disallows second moments with the leaf; minimal/optimal are silent (judge-dp E8).

---

## 3. G1–G10 coverage matrix

| gap | faithful | minimal | optimal | skeptic |
|---|---|---|---|---|
| G1 objective | (a), executed fp32 routes, attn mask, equal weights, α=config/preset 1e-4, pooled | same; feature opt-in; uncentred h | same; α=1e-4 default; centred d | (a) with **α=0** default (regime A); surrogate conditional |
| G2 oracle | O1 loop / O2 batched / O3 fp32; rel-L2 + flip counter; script named | ladder fp32/bf16/vmap with fp32 router on both sides; flip + near-tie margin | same + unpatched-bf16 column | **precision-matched** oracle (same module, same dtype); flips expected 0 |
| G3 statistics | 6 items, ~20 batches of 256 | 6 items + 200-step DP acceptance run | 7 items + 3-seed both-stack runs (heaviest) | 6 items incl. tracking error over 512-step dry run |
| G4 mechanism + accountant | both stacks, one table, adjacency, centred release (no renorm) | both stacks, table (VERIFIED accountant), noisy-sum renorm, guard | both stacks + independent draw + lever table | both stacks, monitor rule, table |
| G5 router | fp32 default, no pinning | fp32 default | fp32 default | **stock bf16 default**, fp32 opt-in, new evidence |
| G6 dense MoE | grouped default + docstring | grouped default + `_grouped_moe_available()` | grouped default | decouple from `use_performance_kernels` |
| G7 MF filter | β_f = workload momentum (0.0824) — preset-consistent | window W=256 — **wrong strategy** | EMA .99 by lag tolerance (0.0249) — preset-consistent | window W=256 — **wrong strategy** |
| G8 scope | DPO in, z-loss opt-in, experts "in for mechanism" | DPO out, experts out, z-loss flag | DPO in, LFB opt-in, experts in for mechanism | DPO regime A, experts regime B, LFB out |
| G9 hygiene | table; **group_norms gap** | table; probe group suppressed | table; **group_norms gap** | table; probe group suppressed |
| G10 hooks/compile/chunks | table; GC mechanism misstated | table; GC mechanism **correct** | table; GC mechanism misstated | table; own-hook handling correct |

Every design answers every gap; the quality differences are in G7 (strategy mismatch) and G9 (telemetry).

---

## 4. Validation-plan executability on one GPU with the real checkpoint

Common facts: 12B bf16 weights = 24 GB; LoRA r=16 q/k/v/o; microbatch 8 at T=1024 with chunked CE is the memory path
PR #978 established. An fp32 copy is 48 GB — feasible on an 80 GB card only one model at a time.

- **faithful** §10: O1/O2/O3 on one microbatch — O3 (fp32 loop) and O2 (unpatched bf16 batched) fit sequentially;
  G3 on ≈20 batches of 256 (5k examples) is minutes with grouped MoE, hours with the dense default; acceptance
  criteria are concrete (6 items). Executable.
- **minimal** §10: three-rung ladder + m1–m5 metrics + six statistics + a 200-step DP run with six acceptance
  criteria. The 200-step run at 256/step ≈ 30 min grouped. It is the only plan that gives a *mechanism* acceptance
  criterion tied to a DP run ((a)–(f)). Executable; notes the b-min-sep MC-PLD cost (>170 s CPU) honestly.
- **optimal** §10: oracle in a separate process + fp32 floor; 7 statistics; end-to-end "short DP runs, ε=3, both
  stacks, 3 seeds" — the heaviest (≥6 runs) and the least bounded in wall-clock. Executable but should be trimmed.
- **skeptic** §10: single script ≤30 min, precision-matched oracle, fp32 forward "only if memory allows", tracking
  error over a 512-step dry run (~1.5 h grouped). The most realistic budget; the flip counter is defined on executed
  sets, and it is the only plan that anticipates a non-zero vmap-vs-eager flip count as a *bisection* trigger.

---

## 5. Scores (0–10; implementability and completeness weighted 1.5x; total = weighted sum / 7)

| design | dp_correctness | faithfulness | utility | implementability (x1.5) | completeness (x1.5) | **total** |
|---|---|---|---|---|---|---|
| faithful | 8.0 | 8.5 | 7.5 | 8.0 | 8.5 | **6.96** |
| **minimal** | 8.5 | 7.5 | 7.0 | 9.0 | 8.5 | **7.04** |
| optimal | 7.5 | 8.0 | 9.0 | 6.5 | 9.0 | **6.82** |
| skeptic | 8.0 | 7.5 | 7.0 | 8.0 | 8.0 | **6.64** |

Rationale (implementability / completeness only; the other columns follow the shared core plus the errors in §7):

- **minimal 9.0 / 8.5.** Zero engine change; every seam it names exists as cited; reuses the callback seam verbatim;
  closure-tensor delivery; sidecar checkpoint; correct recorder-under-checkpointing analysis; `_squeeze_output` analysis
  correct; 19 tests with package placement and markers; DPO out keeps v1 small. Deductions: probe zeroing relies on the
  optimizer producing a null update from a zeroed noised leaf (belt-and-braces assert is there); G7 table for the wrong
  strategy; second-moment exclusion absent.
- **faithful 8.0 / 8.5.** Same seams, all real; adds `save_dp_runtime_state` signature change, DPO override, and an
  under-specified AUTO-S story ("simplest: reject" — fine, but it is a paragraph of options rather than a decision);
  batch-column delivery inherits the pruning wrinkle; matched-filter β_f=0.95 is a preset-consistent but suboptimal
  choice. Completeness: full hygiene table, T1–T8, acceptance criteria, risk list with falsifiers.
- **optimal 6.5 / 9.0.** The plan is the widest: engine `fixed_groups` in `auto_clipped_grad`/`_auto_scale_per_group`
  (`clamp_to_one` is currently a single flag for all groups, `_pytree.py:337`, VERIFIED — a per-group variant is a
  modest but real change); an independent Poisson side-release needing a second sampler, rank sharding, key
  serialization, and composition inside `_build_mechanism`/`_calibrate_noise` (only `_account_independent_step` is
  cited); extra output fields on the patched forward; LFB bias + serving patch; DPO. "One n×n solve" for filter row
  norms at n=15625 is not implementable as written (E8). Completeness is the best: lever table with dominated
  variants, LFB admissibility analysis, T1–T10, hygiene table with the side release.
- **skeptic 8.0 / 8.0.** Engine untouched; hooks route VERIFIED at toy scale (primitives E3); internal step instead of
  callback; typed checkpoint field; decision rule with tail arithmetic; mode state machine (`monitor_then_surrogate`)
  adds a little logic. Completeness: all gaps answered, but the surrogate path is "fully specified" only conditionally,
  DPO regime B and expert training are deferred, and G7 uses the wrong strategy.

**Winner: minimal**, on the condition that the grafts in §6 are applied (in particular the corrected MF filter numbers).

---

## 6. Grafts the synthesis must keep (from non-winning designs)

1. **optimal §6 / faithful §6.2 — the preset-consistent band-MF filter factors** (momentum 0.95, bands 64):
   `‖row_t(C⁻¹)‖` 1.431, EMA .95 0.0824, EMA .99 0.0249, window-256 0.0198 (all reproduced by my run at n=1024:
   1.4309 / 0.0824 / 0.0249 / 0.0198). Replace minimal's/skeptic's momentum-1.0 table (2.26–3.80 / 0.116–0.185 /
   0.0253–0.038 / 0.0157–0.0216). Choose the filter by lag tolerance (optimal): EMA .99 or window 256 are equivalent
   at the preset; do not claim "EMA .95 only reaches parity".
2. **optimal §2.1/§2.2 — centred release `d(x) = h(x) − k/E`** (bound 2.646 vs 2.828, sum structurally zero, no
   Σ=k renormalisation needed, sum-zero projection strips 1/64 of the noise variance) and **ρ = 0.02 default with
   EMA .99** (x1.010 gradient noise; 2.35 % SGD / 0.83 %·(nm_MF/0.5622) MF smoothed error).
3. **optimal §6(iv) — compute the filter's noise factor from the actual strategy at setup**, but via the streaming
   inverse / Toeplitz inverse coefficients (`_band_mf.py:135-136`), not a dense n×n solve.
4. **optimal §4.1.5 / §9.2 — `fixed_groups` for AUTO-S** as the v2 engine change; v1 keeps minimal's/skeptic's
   `ConfigurationError` for any non-fixed clipping mode with the leaf.
5. **optimal §2.3 — the independent forward-only draw** as a DP-SGD-only, explicitly opt-in appendix, with the
   second sampler's key/position serialized and rank-sharded (judge-dp E7) and rejected under any MF mechanism.
6. **faithful §9.1.2 — return contract of the new kwarg**: `MoeCausalLMOutputWithPast(loss, logits=None,
   router_logits=outputs.router_logits, aux_loss=None)` on the chunked path; keep the existing HF-aux fallback and its
   test.
7. **faithful §9.3 T6 — exclude the probe group from `second_moment` paired streams**; **faithful §7 — DPO pooling
   spec** (chosen + rejected policy forwards pooled, reference excluded) for the day DPO is in scope.
8. **skeptic §3 — the fp32-router refutation** (`a_fp32_router_flips.py`: fp32 logits 51/56 vs stock 46/54 flips per
   1024 rows; vmap-vs-eager 0/1024): ship the fp32 router as an *opt-in* justified only as pretraining-faithful, define
   the **precision-matched oracle** (same patched module, same dtype), and expect 0 vmap-vs-eager flips as the
   acceptance criterion; do not cite E1b's fp32-*forward* pinning as evidence for the logit patch.
9. **skeptic §2.3 — the DP-released imbalance monitor** `D_t = max_e |f̄_e − k/E|/(k/E)` and the
   `monitor_then_surrogate` mode: free post-processing of the same leaf; lets the preset keep α small (or 0) while
   measuring the balance assumption; the α switch is loss-only and MF-latch-safe.
10. **skeptic §8 / minimal §8 — suppress the probe group in `group_metrics`** (`_dp_trainer.py:2259-2275`) and never
    add `h`, `P`, per-example aux to `loss_aux`.
11. **skeptic §5.1 — decouple `grouped_moe` from `use_performance_kernels`** with a `_grouped_route_available()`
    helper next to `kernels/moe.py:575-635`, log the chosen path once, document the first-patch capture.
12. **minimal §9.2 (kept from the winner but worth naming)** — the built-in `RouterLoadCallback` on the existing
    `on_pre_optimizer_step` seam, closure-tensor `f̃` delivery (no collator column, no pruning issue), sidecar
    `router_load_state.pt`, and T1–T19 with placement.
13. **minimal §2.5 — the `nm_MF ≠ 0.5622` caveat** on every MF error column; the calibrated band-MF/b-min-sep
    multiplier at ε=3 was not computed by any design.

---

## 7. Errors found (design — error — correction)

- **E1 (minimal §6, skeptic §6; VERIFIED by `judge-impl/mf_rownorm.py`)** — band-MF filter numbers computed with
  `band_mf_strategy(bands=64)` at the factory default `momentum=1.0` (`_band_mf.py:145`), whereas the preset passes
  `momentum=_workload_momentum()=args.momentum=0.95` (`examples/train_dpftrl.py:459` default `sgd`, `:495-498`,
  `:1563-1581`). Skeptic's sentence "momentum 1.0 is the preset default" is wrong. At momentum 0.95 (n=1024):
  row norm 1.431, EMA .95 0.0824, EMA .99 0.0249, W256 0.0198 (matches faithful/optimal). Minimal's "EMA .95 only
  reaches parity (0.185 vs 0.160)" is therefore false for the preset (0.0824 vs 0.160 = 2x better), and the "window,
  not EMA" conclusion is a wash at the preset. (Also found by judge-dp E1/E2 and judge-utility.)
- **E2 (skeptic §2.2 step 2; arithmetic VERIFIED)** — "denominator's noise std `√E·σ_h/s` = 0.044 against k = 8,
  i.e. 0.55 %": 0.0444 is the *per-entry* std at ρ=0.02 (σ_h/s); the sum over 64 entries has std 8 x 0.0444 = 0.355 =
  4.4 % of k per single step. The renormalisation should therefore run *after* smoothing, or use the centred release
  (graft 2) which needs none.
- **E3 (optimal §2.6; arithmetic VERIFIED)** — "shrinkage zeroes the term when δ is below `s·√63/(k/E)` ≈ 0.07 (MF) /
  0.19 (SGD)": that is the threshold on `‖d̃‖`, not on the per-coordinate RMS δ. The positive-part rule
  `1 − (E−1)s²/‖d̃‖²` hits zero when `√E·δ·(k/E) < √63·s`, i.e. δ < ≈ `s/(k/E)` = 0.0083 (MF) / 0.0235 (SGD). The rule
  is more conservative than the text; the `0.0083/δ` relative-error line beside it is correct.
- **E4 (faithful §3/§4.2, minimal §3/§4.3, optimal §3/§4.1.3)** — "fp32 router logits remove the bf16 tie flips
  (≈1 %/layer) and the 8x router/expert gradient-error (E1b)": refuted at toy scale by skeptic's
  `design-skeptic/a_fp32_router_flips.py` (fp32-logit router 51/56 and 39/43 flips per 1024 rows vs stock 46/54 and
  46/47; flips come from bf16 hidden states). E1b pinned routes to an fp32 *forward*, not an fp32 router linear. I did
  not re-run the script (judge-utility reports re-running it); marked VERIFIED-by-others. Correction: fp32 router =
  pretraining-faithful opt-in, not a drift fix; oracle must be precision-matched.
- **E5 (faithful §8, optimal §8; VERIFIED by reading `_dp_trainer.py:2259-2275`)** — hygiene tables classify group
  norms as "pre-existing, as today", but a probe group makes `group_metrics["router_load_probe"]["grad_norm"]` an
  un-noised batch mean of `‖λ d(x)‖` — an unaccounted release. Correction: skip the probe group in that loop
  (minimal T17 / skeptic §8).
- **E6 (faithful §9.3, optimal §9.4; VERIFIED by reading `output_capturing.py:104-120, 255-272`)** — faithful:
  "capture_outputs registers its recorder hooks for the duration of one forward call" — hooks are installed once and
  persist; only the `ContextVar` collector is scoped. optimal: "under checkpoint recompute it fires again … nothing
  reads it" — the hook runs but appends nothing because the collector is reset. Both conclusions (no double capture)
  are right; minimal's explanation is the correct one.
- **E7 (faithful §2.6, optimal §2.6/§6, skeptic §2.5)** — MF load-error columns labelled "under the DP-FTRL preset"
  are evaluated at nm = 0.5622, the DP-SGD/Poisson calibration; the band-MF/b-min-sep multiplier at ε=3 was not
  computed (minimal §2.5 says so explicitly and scales by `nm_MF/0.5622`). Relative claims (x√(1+ρ), MF/SGD ratios)
  are unaffected. Not verified in magnitude by anyone.
- **E8 (optimal §6(iv); implementability)** — "recompute the row norms from the actual strategy at setup (one n×n
  solve …)": at n = 15625 a dense inverse is 15625² x 8 B ≈ 1.95 GB and O(n³); the streaming inverse
  (`inverse_as_streaming_matrix`, `_band_mf.py:135-136`) or the Toeplitz inverse coefficients must be used.
- **E9 (line drift, minor)** — faithful/primitives "`on_pre_optimizer_step` `:2210-2218`" → `:2198-2206`; optimal
  "`:2197-2199`" → same; faithful "`_clipped_fun.py:255-260` chunk path" → loop at `:273-276`; optimal "`_auto.py:117`
  `auto_clipped_grad`" → `:203`; skeptic "`_band_mf.py:60`" for the momentum default → `:145`; faithful
  "`huggingface.py` `_force_non_reentrant` `:28-33`" → call at `:31`, def at `:35`.
- **E10 (faithful §9.2, optimal §9.3, skeptic §9.2; omission)** — the seeded `load_target` batch column is pruned by
  `_remove_unused_columns` (`_dp_trainer.py:3446-3470`) under plain `DPTrainer` with HF's default
  `remove_unused_columns=True` unless the column is a named forward parameter; it works under `DPSFTTrainer`/`DPDPOTrainer`
  only because their configs set it `False` (`_sft_config.py:113`, `_dpo_config.py:139`). Minimal's closure tensor avoids it.
- **E11 (skeptic §9.2; minor)** — "adding a new field to `RuntimeCheckpoint` is a single edit … picked up
  automatically": `save_dp_runtime_state` enumerates keyword parameters (`_checkpoint.py:311-335`) and
  `_apply_runtime_state` restores fields by hand (`_dp_trainer.py:5342-5358`), so it is three edits (dataclass,
  saver signature + call site `:5091-5122`, restore). Still implementable.
- **E12 (all four; not an error, a shared unverified claim)** — gradients flowing through recorder-captured router
  logits inside a non-reentrant checkpoint region under vmap: PLAUSIBLE (HF eager practice), every design has a test
  (faithful T3, minimal T5, optimal T4, skeptic T4); none executed it.

---

## 8. What the synthesis should look like from this lens (one paragraph)

Take minimal's implementation skeleton verbatim (engine/accounting/dpftrl untouched; fp32-router and stats helpers
in `opaque-patches`; named `opaque_router_logits` kwarg on the chunked-CE forward that sets `output_router_logits=True`
on the backbone only and returns the existing `MoeCausalLMOutputWithPast(logits=None, aux_loss=None, router_logits=…)`;
probe parameter registered before `make_functional`; clip-dict extension at `_dp_trainer.py:1455-1470`; built-in callback
on `on_pre_optimizer_step`; closure-tensor `f̃`; sidecar checkpoint; probe group suppressed from `group_metrics`;
`ConfigurationError` for non-fixed clipping; 19 tests). Replace its statistic with optimal's centred `d(x)`, ρ = 0.02,
EMA .99 (or window 256) and the momentum-0.95 filter factors, and compute those factors from the actual strategy via the
streaming inverse. Make the fp32 router opt-in with skeptic's corrected rationale and precision-matched oracle, add
skeptic's `D_t` monitor as a free mode on the same leaf, and keep faithful's second-moment exclusion and DPO pooling spec
in an appendix for the DPO preset. Keep optimal's independent-draw and `fixed_groups` as explicitly deferred v2 items with
their serialization/engine requirements written down.
