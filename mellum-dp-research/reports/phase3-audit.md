# Phase 3 — audit of the prototype and the verifier (consolidated results for the research report)

Agent: `auditor`. Repo `/home/user/opaque`, branch `claude/mellum-dp-representation-r6slaz`. **HEAD is now `5c65897`**
("docs(research): add Mellum2 DP representation research artefacts", 143 files, 18 685 insertions) — one commit past the
`ef1abc5` the prototype and verifier cite; `git diff --stat ef1abc5 HEAD -- packages examples` is **empty** (VERIFIED), so every
`packages/…` / `examples/…` line number below is the same at both commits. `git status` clean before and after this audit
(0 modified tracked files, VERIFIED). Everything I wrote lives under `scratchpad/research/auditor/`:

| file | what |
|---|---|
| `rerun_output.txt`, `rerun_stderr.txt`, `rerun_results.json` | my re-run of the unmodified prototype script (stdout, `time`, results) |
| `prototype_output_original.txt`, `prototype_results_original.json` | byte copies of the prototype's own final-run outputs, taken **before** my re-run (the script overwrites `prototype/prototype_results.json`) |
| `verifier_rerun.txt` | my re-run of all seven verifier scripts |
| `check_glue.py` | were the opaque HF checkpoint glue and the dense MoE path active in the prototype's build? |
| `prototype_load_release_patched.py`, `patched.diff`, `run_mf_arm_patched.py`, `mf_arm_patched.out` | a patched copy fixing two prototype deviations (D-4, D-5 below) and the V6 arm re-run with it — **not** needed to make the script run |
| `k1_u_sweep.py`, `k1_u_sweep.out` | shared-coin hockey stick for partner mass `u ∈ {0,1,3,7}` (W ranks) |

Evidence tags: **VERIFIED** = I ran the script or read the cited lines in this session; **PLAUSIBLE** = inferred, not executed.
CPU only, 4 cores, `torch.set_num_threads(2)` in every script.

---

## 1. Prototype re-run (task item 1)

Command: `uv run python …/prototype/prototype_load_release.py` from `/home/user/opaque` (unmodified script). **Exit 0, wall-clock
3 m 23 s** (`real 3m23.409s`, script-reported `total elapsed 199s`; the prototype reported 199 s too). No patch was needed to run it.

**Every PASS/FAIL matches**: `V1..V7 = PASS` (7/7), identical to the prototype report. Stronger than that: the run is
**deterministic to every printed digit** — `diff` of the two stdouts differs only in the accountant's wall-clock string
("1.9s" → "1.8s"); a leaf-by-leaf comparison of the two `prototype_results.json` files (≈ 800 kB each, all per-step histories
included) finds **9 differing leaves, all of them `elapsed` timings** (`arms[0..7].elapsed`, `elapsed_total_s`). VERIFIED. The
2-process pool (deviation D10 of the prototype) therefore did not introduce any nondeterminism.

Headline numbers as I observed them (all VERIFIED, identical to the prototype report): `nm = 1.0820` (ε = 3, δ = 1e-5, q = 1/32,
T = 300); `C_g = 5.7195`; `δ₀ = 0.317`, `D₀ = 0.620`; `ρ* = 0.34`; V1 cosine 1.000010, norm ratio 0.748046 vs 0.748047; V2 leaf
error 3.7e-9, adversarial `‖d‖ = 1.732051 = Δ_L`, probe norms max 1.944646 < C_h 1.946591, 0 over; V3 Mahalanobis ratio
1.000000000000, inflation 1.157731 = √1.34; V4 ε 2.998599 = 2.998599 (DP-SGD), 3.996299 = 3.996299 (band-MF un-amplified);
V5 δ_T OFF 0.420 / ORACLE 0.075 / DP ρ* 0.165 / DP ρ=0.02 0.314, recovered fraction 0.740, cosPop-active 0.825, dead zone
24.0 % / 86.7 %, pooled noise 0.999 of prediction, all six criteria True; V6 latch (no raise) 50/50, σ rel-err 2.91e-16 /
3.38e-16, probe noise 0.993 of prediction, mean batch 29.94; V7 rel-L2 0.0 with flags `[True, True]`.

Two things the prototype left PLAUSIBLE that I settled (`check_glue.py`, VERIFIED):

- **The opaque HF checkpoint glue was active in V7.** After `build_moe_model` (which calls `apply_model_patches(model,
  eager_attention=True)`, `_test_utils.py:169`), `transformers.PreTrainedModel.gradient_checkpointing_enable` is opaque's
  `_force_non_reentrant.<locals>.gradient_checkpointing_enable` (installed by `patches/torch/checkpoint/__init__.py:58 →
  huggingface.apply()`, `huggingface.py:31-56`), and the model's `_gradient_checkpointing_func` is
  `partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)`. So V7 tested the non-reentrant path the design §9.1 relies on.
- **The dense `Opaque_MoE` path ran, not the grouped route.** `apply_model_patches` resolves `kernels=None → kernels =
  performance = True` (`_factory.py:269-279`) and `grouped_moe = kwargs.get("grouped_moe", kernels)` (`_factory.py:319`), i.e.
  the *test helper* asks for the grouped route (unlike `DPTrainer`, which passes `kernels=False`); but the runtime gate
  `use_grouped_route(..., min_experts=_SPARSE_MOE_MIN_EXPERTS)` with `_SPARSE_MOE_MIN_EXPERTS = 16` (`kernels/moe.py:575,
  624-632`; `_moe_memory.py:120-134`, `experts >= min_experts and grouped_fits`) is False at E = 8 whenever the dense workspace
  fits, so `Opaque_MoE.apply` (`moe.py:633`) executed. VERIFIED by code reading, not by a runtime assertion.

---

## 2. Prototype implementation vs design §1.1 / §2.2 / §2.4, line by line (task item 2)

Script: `prototype/prototype_load_release.py`. "OK" = matches the design text exactly.

### 2.1 §1.1 — per-example loss

| design element | script | verdict |
|---|---|---|
| mask `m = 1{attention_mask ≠ 0}` (binarised, 2-D) | `router_stats:141` `m = (mask_row != 0).float()` | OK |
| `T_x = Σ m`; `h^l = (m·onehot)/T_x`, `P = (1/(L T_x)) Σ_l Σ_t m p` | `:142-150` `Tx = m.sum()`, `denom = Tx.clamp(min=1)`, `hL.append((onehot*m).sum(0)/denom)`, `P = P/(len(router_logits)*denom)` | OK |
| explicit `where(T_x > 0, ·, 0)` on the centred quantities (not a bare clamp) | `:152-154` `where(valid, hL, 0)`, `where(valid, P, 0)`; again `:165` `dL = where(Tx > 0, hL − k/E, 0)` | OK — the spurious `d = −k/E·1` the design warns about cannot occur |
| fp32 softmax + `topk` = the router's own ops; broadcast-compare one-hot (no `F.one_hot`/`scatter_add_`) | `:146-148` `softmax(z.float())`, `torch.topk(p, K)`, `(idx[..., None] == arange(E)).sum(-2)` | OK |
| `w_x = T_x/T̄`, `T̄` public, default `T_max` | `:164` `w = Tx / CTX["t_bar"]`, `t_bar = T_MAX` everywhere | OK (`T̄ = T_max`; the preset would set `T̄` = measured mean; the prototype states the resulting 0.750 release scale) |
| `S_x = E·w_x·Σ_e (f̃_e − k/E)·P_e(x)` | `:166` | OK (centred form; `Σ_e P_e = 1` makes it gradient-identical to HF's uncentred one — V1) |
| `ℓ_x = CE_x + α(S − sg S) + ⟨z, sg[λ w_x d^{(L,E)}]⟩` | `:167-169` | OK; value-neutral confirmed (`aux.loss_values == CE` True) |
| `CE_x` = HF per-example token-mean CE through the chunked LM head | emulated forward `:99-105`: `linear_nll_sum_chunked(out[0], lm_head.weight, labels, -100, 0, 0.0, False, chunk_vocab=None) / n_valid` | **D-8** — an emulation (the named kwarg `opaque_router_logits` does not exist in the repo: `grep -rn opaque_router_logits packages/` is empty, VERIFIED); equality of this CE to HF's `.loss` was not asserted (PLAUSIBLE — same shift and same valid-count; irrelevant to V1–V7, which never compare against HF's CE) |
| z-loss opt-in, `ζ = 0` | absent | OK (default off) |
| capture-count assert `len(router_logits) == num_hidden_layers` (§1.4 last row) | absent; `h` normalised by `len(router_logits)` | **D-6** (minor): a double capture would give a `(2L, E)` `hL` and fail loudly at the `(L, E)` probe dot product, so the failure mode is loud, but the design's explicit assert is not there |
| `attention_mask=None` → all positions count (§1.4, T2) | not supported: `router_stats` and the emulated forward require a mask | **D-7** (scope): the `None` path is untested by the prototype |
| probe `z ∈ R^{L×E}` zero parameter registered **before** `make_functional(partition_trainable=True)` | `build_model:87-88`, `:107-109` (asserted in the trainable pytree) | OK |

### 2.2 §2.2 — per-step mechanism (DP-SGD / Poisson)

| step | design | script | verdict |
|---|---|---|---|
| 1 sample | `PoissonSampler(q = B̄/N)` | `run_arm:311` `PoissonSampler(ids, sample_rate=q, n_steps, key)` | OK |
| 2 augment | once per step outside vmap; assert probe is zero | `:343-350` `CTX["f_tilde"] = f_used` before the grad call; `assert count_nonzero(params[PROBE]) == 0` | OK (module-level `CTX` dict instead of a closure tensor — equivalent) |
| 3 per-example inside vmap | `vmap(grad_and_value)` with router logits | `clipped_grad(loss_fn, has_aux=True, batch_argnums=(1,2,3), return_aux=True)` `:304-305` | OK |
| 4 clip | `PerGroup` by direct construction; `normalize_by = B̄`; stored bounds `C_g/B̄`, `C_h/B̄` | `make_pergroup:174-177` (`{(leaf,): "fallback"}` for every non-probe leaf, `{(PROBE,): "router_load_probe"}`); `normalize_by=float(B_BAR)`; V2 prints `max_norm stored {fallback: 0.17874, router_load_probe: 0.06083}` = `C_g/32`, `C_h/32` | OK |
| `C_h = λΔ_L(1+g)`, `g = 1e-3`, `Δ_L = √(kL(1−k/E))`, `λ = ρC_g/Δ_L` | `:298-299`, `DELTA_L`, `GUARD = 1e-3` | OK |
| 5 noise | `gaussian_noise(nm)` → `per_group_noise_stddev` | `:310`; V3 confirms `σ_g, σ_h` closed forms and the Mahalanobis identity | OK |
| 6 release = noised pytree | `noised, nstate = noise_fn(clipped, nstate)` `:354` | OK |
| 7 post-process, then zero the probe entry in place before the optimizer | `:361-362` `postprocess(...)`; `noised.pytree[PROBE].zero_()`; SGD skips the probe `:394-397`; final `count_nonzero(params[PROBE]) == 0` recorded | OK |
| 8 accountant unchanged | `poisson(gaussian(nm), q) * T` built from `(nm, q, T)` | V4 `:620-621` | OK — but note the "with/without probe" comparison is two identical calls; it demonstrates the API takes no pytree argument, it does **not** exercise the trainer's `_build_mechanism` (the verifier's K2a does, by reading `:4329/:4339/:4404-4405`) |
| §2.3 MF variant: b-min-sep with the paper's `p = p₀/(1 − p₀(b−1))` so `E|B_t| = B̄` from step 0 | `:317` `BMinSepSampler(ids, bands=4, sampling_prob=q, …)` — **`p₀` passed as the paper's `p`** | **D-4 (undeclared)** — see below |
| §2.3 latch accepts the two-group `PerGroup` | `:455` `res["mf_latch_ok"] = True` **hard-coded** | **D-5** — see below |

**D-4 (undeclared deviation, MF arm).** `BMinSepSampler.sampling_prob` is the paper's per-iteration `p`, and the docstring says
"for the same expected batch size as Poisson with per-example rate `p₀` … set `sampling_prob = p₀/(1 − p₀(bands−1))`"
(`_b_min_sep.py:7-10, 40-49`, VERIFIED); the trainer does exactly that through the amplifier
(`_dpftrl.py:246-252` → `amp.sampling_prob` → `participation_p_from_per_example_rate(self.p0, bands)`,
`accounting/dpftrl/amplification/_b_min_sep/__init__.py:95-100`, VERIFIED). The prototype passes `q` directly, so the stationary
expected batch is `N·p/(1+(b−1)p) = 29.26` instead of `B̄ = 32` (observed mean 29.94 over 50 steps, VERIFIED) while
`normalize_by = 32`: the released signal in the MF arm is scaled by ≈ 0.914 relative to the design's §2.3 statement
`E|B_t| = B̄`. Privacy is unaffected (the divisor is public and the sampler's `p` is what an accountant would be given), and none
of V6's three assertions (latch, realised-σ identity, probe receives noise) depends on it — but the prototype report's
"mean batch 29.9" is presented without explanation, and the arm is not the preset's sampling model. With the one-line fix
(`patched.diff`) the arm gives mean batch **31.98**, δ 0.317 → 0.148 → **0.170** (vs 0.161), cosB 0.733 (vs 0.592), cosPop 0.744,
dead zone 4 %, σ identities unchanged to 3e-16 (`mf_arm_patched.out`, VERIFIED) — V6's conclusion stands.

**D-5 (evidence quality, V6).** "latch ok 50/50" is not an assertion in the script: `mf_latch_ok` is set to `True`
unconditionally after the loop. The *inference* is valid — `_validate_constant_max_norm` runs on every `mf_gaussian_noise` call
(`_mf_gaussian_noise.py:163-165`) and raises `ConfigurationError` on any change or negative group value (`_engine.py:473-517`,
VERIFIED), so 50 calls without an exception do mean the two-group `PerGroup` was accepted — but the report words it as a check.
The patched copy asserts it explicitly (`nstate._first_max_norm == clipped.max_norm`, a `PerGroup` with exactly the two groups):
**True** (VERIFIED).

### 2.3 §2.4 — post-processing

| step | design | script (`postprocess:248-268`) | verdict |
|---|---|---|---|
| 1 `d̂^{(L,E)} = ŷ/λ` | | `d_hat_L = y / lam` | OK |
| 2 layer mean | | `d_hat = d_hat_L.mean(0)` | OK |
| 3 sum-zero projection | | `d_hat − d_hat.mean()` | OK |
| 4 bias-corrected EMA; `s_t = (σ_h/(λ√L))·φ_t/(1−β^t)`, `φ` from the exact recursion (DP-SGD) or `‖row_t(F·C⁻¹)‖` (MF), both with the `(E−1)/E` factor | `m = βm + (1−β)d̂`; `d_tilde = m/(1−β^t)`; `s = σ_h/(λ√L)·phi[t−1]/(1−β^t)`; `phi_table_sgd` = `√(Σ_{i<t} ((1−β)β^i)²·(E−1)/E)` (equivalent to the recursion); `phi_table_mf` = `‖row_t(F·C⁻¹)‖·√((E−1)/E)` from `strategy.coefficients(n_steps)` | OK in substance; **β = 0.9 not 0.99** (declared D2); MF `C⁻¹` by a dense `np.linalg.inv` of the 50×50 Toeplitz (design §6.2: "never a dense n×n solve" — an implementation-scale rule, harmless at n = 50) — **D-18** (method only) |
| 5 dead zone `‖d̃‖² < c·E·s²` → 0, else JS+ `1 − E s²/‖d̃‖²` | `n2 < c*E*s*s` → zeros; else `shrink = 1 − E*s*s/n2` | OK (`c = 2`) |
| 6 `f̃ = clamp(k/E + d̃⁺, 0, 1)` | `torch.clamp(K/E + d_plus, 0, 1)` | OK |
| 7 monitors `D`, `D^l` from the bias-corrected (un-shrunk) EMAs | `D = max|d_tilde|/(k/E)`; `D_layer` from the per-layer projected EMA `/corr` | OK |
| `f̃_0 = k/E`; `f̃_{t+1}` consumed at step `t+1` | `f_tilde = full(k/E)` initially; `f_used = f_tilde` read at the top of the step, replaced after the release | OK |

### 2.4 Declared deviations I checked (prototype §4) — all confirmed as stated

D1 sizes; D2 β = 0.9; **D3 α = 1.0** (consequence VERIFIED in my re-run: per-example median norm 5.720 → 7.073, ×1.237 — the
design's §4.2 "aux does not move the per-example norm" is an α ≤ 1e-3 statement and is *contradicted* at α = 1, exactly as the
prototype says); D4 ρ* = 0.34 chosen from `δ₀` measured on the disjoint held-out split (design §4.4-compliant, but a lab
quantity); D5 emulated forward (see D-8); D6 ORACLE extra forward; D7 V6 accountant `mf_gaussian` un-amplified (ε = 4.00 at
nm = 1.082, n = 50) rather than `b_min_sep(...)`; D8 plain SGD; D9–D12 as stated. Nothing else undeclared beyond D-4/D-5/D-6/D-7
above.

### 2.5 Two prototype findings I re-derived

- **Dead-zone false-pass rate is E-dependent (prototype F2, was PLAUSIBLE).** The projected pure-noise statistic is
  `‖d̃‖²/s² = (E/(E−1))·χ²_{E−1}`, so the dead zone `‖d̃‖² < c·E·s²` is a `χ²_{E−1} < c·(E−1)` test:
  `P(χ²₇ > 14) = 5.12e-2` at E = 8 and `P(χ²₆₃ > 126) = 4.25e-6` at E = 64 (design: 4.2e-6; `c = 1.5`: 6.25e-3, design 6.3e-3).
  VERIFIED (scipy). The prototype's "≈ 5e-2" and the design's figures are both right; the effect is a toy-scale artefact.
- **ρ = 0.02 is a preset-regime constant.** The prototype's r-formula at `B̄ = 32, nm = 1.082, β = 0.9` gives a smoothed noise of
  0.254 k/E vs a released imbalance of 0.238 (ratio 1.07) — reproduced in my run. The design should ship the rule (or print
  `s_∞/(k/E)` at setup), not the constant; at the preset regime the same formula gives the 2.35 % of design §2.6.

---

## 3. K1 — the DDP + resume sampler-key bug, checked by reading the trainer myself (task item 3)

**What the bug is, when it applies, and what it costs (one paragraph for the final report).** On `resume_from_checkpoint` with
`ignore_data_skip=False` (the default, `_training_arguments.py:370`), `DPTrainer._inner_training_loop` builds a template sampler
through `get_train_dataloader()` — fresh construction, which folds the rank into the key (`_dp_trainer.py:3802-3803`,
`fold_in(sampler_key, self._ddp.rank)`) — and then **replaces it** with `from_state_dict(ctx.current_sampler,
saved_sampler_state)` (`:1785-1800`). The sampler serialisers store the *already rank-folded, domain-separated* stream key
(`_poisson.py:217-218` `key_seed = s._stream_key.seed`; `_b_min_sep.py:219-220`; `_balls_in_bins.py:221`) and their
deserialisers install it verbatim with no rank re-fold (`_poisson.py:261`, `_b_min_sep.py:259`, `_balls_in_bins.py:260`; the
dpftrl `CyclicPoissonSampler` at `dpftrl/sampling/_poisson.py:304` behaves the same but is not config-reachable). The snapshot
`dp_state.pt` is written by the single rank passing `should_save` (world-rank 0 by default; each node's local zero under
`save_on_each_node`; `_dp_trainer.py:4962, 5004`, `_save_dp_runtime:5070-5079`, `_distributed.py:209-220`) and read by **every**
rank (`:5317-5318` → `:1244-1246`). Because each rank runs its sampler over its own contiguous, equal-length shard
(`_effective_train_dataset_size:4534-4564` trims to `(n // W)·W`; `_shard.py:20-33`), after the resume all `W` ranks continue
rank 0's stream and the `W` records that share a local shard index receive **identical inclusion coins on every remaining
step** — for `PoissonSampler`, `BMinSepSampler` and `BallsInBinsSampler` alike, and for `DPTrainer` and its subclasses
`SFTTrainer` (`trl/_sft_trainer.py:118`) and `DPOTrainer` (`trl/_dpo_trainer.py:274`), neither of which overrides the loop.
It does **not** apply to single-process resume (bit-exact, K1b "restored rank 0 == continuous rank 0: True"), to DDP runs never
resumed, to `ignore_data_skip=True` (a fresh sampler with the restart fold *and* the rank fold, `:3789-3803`), or to the manual
example loops (fresh construction with `fold_in(key(seed), rank)`, `examples/train_dpftrl.py:1206`; no `from_state_dict` anywhere
under `examples/`, VERIFIED). The trainer's own comments say the opposite of the truth: "the per-record marginal stays
Bernoulli(q) either way, so the privacy accounting is unaffected" (`:3743`), the resume caveat (`:3750-3755`), and the docstring
"Privacy budget is unchanged … DP-valid either way" (`:963-968`) — all VERIFIED. The accountant the trainer runs —
`poisson(gaussian(nm), q)` (`_build_mechanism:4404-4405`) or `b_min_sep(mf_gaussian(nm, strategy), n_steps, p0)`
(`trainer/_dpftrl.py:150-158`) — prices the pair `(P, (1−q)P + qQ)`, which needs the added record's coin to be independent of
every other record's (Feldman–Shenfeld Lemma 3.2 as implemented in `src/amplification/poisson.rs:15-42` — the design's
citation; I did not re-read the Rust file, PLAUSIBLE). With a shared coin the worst-case add/remove pair is
`((1−q)A + qB_u, (1−q)A + qB_{u+1})`: **per step at the preset regime** (σ = 0.5622, q = 5.12e-4, ε = 3) **δ = 3.16e-11 with
independent coins vs 2.76e-5 with a shared coin and an aligned partner (u = 1), ratio 8.7e5**; the ε at which the shared-coin
pair reaches the accountant's per-step δ is **10.5**; and because a T-fold composition post-processes to its first step,
**the whole-run guarantee after the resume is rigorously ε(δ = 1e-6) ≥ 6.0, not 3** (the true composed value is far larger; a
crude RDP bound gives ≤ 44, PLAUSIBLE). My own sweep adds that with `W` ranks the co-indexed partner mass can be `u = W−1`:
δ saturates at the cap `q·δ_G(3) = 5.75e-5` from `u = 3` on and `ε₁(1e-6)` rises only to 6.13, so the `u = 1` figures are within
2× (δ) and 0.12 (ε) of the worst case for any world size (`k1_u_sweep.out`, VERIFIED). **Consequence: after a DDP resume the
accountant's certificate is void for every release of the trainer — the gradient and the probe alike — not merely loose**;
the design's P0 fix (re-fold by rank *after* `from_state_dict`, because the restore overwrites the template's key at
`_poisson.py:261` / `_b_min_sep.py:259`, or per-rank snapshots; T25) and its narrowed privacy statement ("single-process, or DDP
without checkpoint resume") are the right disposition.

**Side by side** (per-step hockey-stick at σ = 0.5622, q = 256/5e5, sensitivity 1; the refute-composition column is read from
`refute-composition/corr_coins.out` (not re-run by me); the verifier column is VERIFIED by my re-run of both scripts in this
session, `verifier_rerun.txt`; the auditor column is my own script):

| quantity | refute-composition (`corr_coins.py`, `np.trapz`, 400 001 pts on [−12, 14]) | verifier (`k1_hockey_stick.py`, rectangle rule, 4 000 001 pts) | auditor (`k1_u_sweep.py`, 6 000 001 pts) |
|---|---|---|---|
| δ(ε=3), independent coins | 3.160e-11 | 3.1597e-11 | 3.1597e-11 |
| δ(ε=3), shared coin, u = 0.5 | 1.016e-06 | 1.0164e-06 | — |
| δ(ε=3), shared coin, u = 1 | 2.759e-05 | 2.7587e-05 | 2.7587e-05 |
| δ(ε=3), shared coin, u = 3 / 7 | — | — | 5.7481e-05 / 5.7481e-05 (= cap `q·δ_G`) |
| ε at the accountant's per-step δ | — | 10.52 | — |
| single-step ε₁(δ=1e-6), indep / shared u=1 / u=7 | — | 0.210 / 6.006 | 0.210 / 6.006 / 6.127 |
| whole-run statement | "six orders of magnitude per step; accounting comment is wrong" | "right per step, **under-stated** for the run: ε(1e-6) ≥ 6.0 rigorous" | agree with both; add the `u ≤ W−1` cap |
| sampler-level reproduction | (not run) | 20/20 identical post-resume masks, Poisson and b-min-sep; co-inclusion 0.0506 vs q² 0.0025 | re-run: identical output |

**Disagreements: none on any number.** Two citation nits, confirmed: the design §8.1 attributes both comments to `:3750-3757`;
the "accounting is unaffected" sentence is at `:3743` (block `:3735-3744`) and the resume caveat spans `:3750-3755` (the
verifier says `:3750-3756`, off by one at the end; `:3756` is blank and `:3757` is the `if`). The docstring at `:963-968` is a
third place that needs correcting. No `distributed`-marked test under `packages/opaque-transformers/tests` mentions `resume` or
`from_state_dict` (VERIFIED by grep), so T25 is genuinely new.

---

## 4. Consolidated results table (task item 4)

Result codes: PASS = the check's stated assertion holds as I observed it; PARTIAL = holds with a material qualification.
"Script" names the producing script; every row was **re-executed by me** (V-rows via `auditor/rerun_output.txt`, K-rows via
`auditor/verifier_rerun.txt`) and reproduced to every printed digit.

| id | check | result | key numbers | script |
|---|---|---|---|---|
| **V1** | surrogate identity on the patched model, ragged rows `[25,21,…,30]`: batch-mean of per-example surrogate grads at `f̃ = f(B)` vs `∇L_aux^HF` (HF's own `load_balancing_loss_func` outside vmap and an out-of-place copy) | PASS | cosine 1.000010 (fp32 > 1); norm ratio 0.748046 vs `T_tot/(B·T̄)` 0.748047 (rel 9.6e-7); out-of-place vs HF-own rel-L2 0.0; aux value 2.049564 | `prototype/prototype_load_release.py` |
| **V2** | probe leaf `== (λ/B̄) Σ_x w_x d^{(L,E)}(x)`; structural bound attained by an adversarial all-same-token row; clip never fires | PASS | max abs err 3.7e-9; `‖d‖ = 1.732051 = Δ_L`; max group norm 1.944646 (= λΔ_L) vs `C_h` 1.946591; 0 examples over | same |
| **V3** | Mahalanobis identity with the real `per_group_noise_stddev` via `gaussian_noise` | PASS | σ_g 0.223903, σ_h 0.130622 (closed forms equal); `Σ(C_i/B̄)²/σ_i² · nm² = 1.000000000000`; inflation 1.157731 = √1.34 | same |
| **V4** | accountant built from `(nm, q, T)` / `(nm, strategy, n_steps)` only; no pytree/PerGroup/max_norm parameter | PASS | DP-SGD ε 2.998599 both; band-MF(4,.95) n=50 `mf_gaussian` ε 3.996299 both (un-amplified, sensitivity 1.0000). Qualification: identical calls by construction — the trainer's `_build_mechanism` is covered by K2a, not here | same |
| **V5** | 300-step DP-SGD, B̄=32, N=1024, Poisson, router+attention trainable, induced δ₀=0.317: OFF / ORACLE / DP ρ*=0.34 / DP ρ=0.02 + nm=0 ablations | PASS (one criterion borderline) | δ_T OFF 0.420 / ORACLE 0.075 / DP ρ* **0.165** (74 % of ORACLE's reduction) / DP ρ=0.02 0.314; cosPop-active **0.825** (criterion > 0.8; 0.774 in the prototype's run 2 with a 64-example reference); cosB 0.636; dead zone 24 % / 86.7 %; pooled noise 0.999 of prediction; nm=0: OFF 0.335, ORACLE 0.054, DP 0.052 (lag/EMA free); CE_T 4.804 all noisy arms; α=1 lifts median norm ×1.237 (D3) | same |
| **V6** | 50 steps `mf_gaussian_noise(band_mf(4, 0.95))` + `BMinSepSampler`: two-group latch, probe receives MF noise, realised σ per group `= base·‖row_t(C⁻¹)‖` | PASS (with deviation D-4/D-5) | σ rel-err probe 2.9e-16, fallback 3.4e-16; probe noise RMS 0.993 of prediction; row norms `[1.2489, 1.4192, 1.4262, 1.4273, 1.4595]`; φ_MF 0.1488 vs SGD 0.2146; δ 0.317→0.161; **mean batch 29.94 because `sampling_prob = q` (paper-p not converted)**; `mf_latch_ok` hard-coded (inference valid) | same |
| **V6b** (auditor) | V6 arm with the paper-p conversion and an explicit latch assertion | PASS | mean batch **31.98**; latch True (explicit); σ rel-err 2.9e-16 / 3.4e-16; probe noise 0.993; δ 0.317→0.148→0.170; cosB 0.733 | `auditor/run_mf_arm_patched.py` (+ `prototype_load_release_patched.py`, `patched.diff`) |
| **V7** | gradient checkpointing on/off equality of the probe leaf and all clipped grads | PASS | flags `[True, True]`; probe rel-L2 0.0; max over leaves 0.0; opaque non-reentrant glue confirmed active (`use_reentrant=False`, `check_glue.py`) | `prototype_load_release.py`; `auditor/check_glue.py` |
| **K1a** | code reading: restored sampler on rank ≠ 0 does not re-derive a rank key; snapshot written by one rank | PASS (bug CONFIRMED) | rank fold only at `:3802-3803`; restore `:1799-1800`; keys verbatim `_poisson.py:261`, `_b_min_sep.py:259`, `_balls_in_bins.py:260`; save `:4962/:5004`, `_distributed.py:209-220`; wrong comments `:3743`, `:3750-3755`, `:963-968`; applies to `DPTrainer`/`SFTTrainer`/`DPOTrainer`, all config samplers, only `world_size > 1` + resume + `ignore_data_skip=False`; examples unaffected (`train_dpftrl.py:1206`) | reading (verifier + auditor) |
| **K1b** | sampler-level reproduction, 2 "ranks", rank-1 template restored from rank-0 snapshot | PASS (bug reproduced) | Poisson 20/20 identical masks post-resume (0/7 pre); BMinSep 20/20; co-inclusion 0.0506 / 0.0503 vs q² 0.0025; restored rank 0 == continuous rank 0 True; continuous rank 1 vs restored rank 1: 0/20 | `verifier/k1_sampler_resume.py` |
| **K1c** | privacy effect: hockey stick, independent vs shared coin | PASS (confirmed; under-stated for the run) | δ(3): 3.1597e-11 vs 2.7587e-05 (×8.7e5); u=0 reproduces the standard value; ε at the accountant's δ 10.52; ε₁(1e-6) 0.210 vs 6.006 ⇒ whole-run ε(1e-6) ≥ 6.0 rigorous; cap `q·δ_G = 5.75e-5` at u ≥ 3, ε₁ ≤ 6.13 for any W; refute-composition's 3.160e-11 / 2.759e-05 / 1.016e-06 identical | `refute-composition/corr_coins.py`; `verifier/k1_hockey_stick.py`, `k1_run_bounds.py`; `auditor/k1_u_sweep.py` |
| **K2a** | accountant path in `_build_mechanism` unchanged by the probe group; baseline ε | PASS | `num_groups` consumed only in the `clipping_mode == "adaptive"` closure (`:4329`, `:4339`); fixed mode → `poisson(_unamplified(nm), q)` (`:4404-4405`); `poisson(gaussian(0.5622), 256/5e5)*15625` @1e-6 → **ε 3.0004**; naive per-group `σ = nm·C_i` would be 11.84 | `verifier/k2_accounting.py` |
| **K2b** | real allocator: `σ_g/(nm C_g) = √(1+ρ)`, `σ_h/(nm C_h) = √(1+1/ρ)`, Mahalanobis `= 1/nm²` | PASS | six ρ ∈ {0.5,…,0.01}: 1.224745…1.004988 and 1.732051…10.049876 match to 6 digits; `Mahalanobis·nm² = 1.000000` at all six | same |
| **K2c** | design §2.6 "pay in ε" column 3.234 (ρ=0.02) / 3.703 (ρ=0.1) | PARTIAL | reproduces **exactly** with `poisson(g(nm)\|g(nm_h), q)*T`: 3.2345 / 3.7035 — but the mathematically identical joint Gaussian `poisson(g(nm_eff), q)*T`, `nm_eff = nm·√((1+ρ)/(1+2ρ))`, gives **3.1256 / 3.5897** (inner PLDs equal to 7 digits; single-step Poisson δ(3) 6.32e-11 vs 4.52e-11): the column is a valid but conservative bound, high by ≈ 0.11 ε in every row. The shipped ε-held route is unaffected | `verifier/k2_composed_inner_v2.py` |
| **K3** | `‖d^{(L,E)}(x)‖ ≤ √(kL(1−k/E))`, equality attained; replace-one bound form | PASS | E=64,k=8,L=2: bound 3.741657 = max random = adversarial = T=1; E=8 exhaustive equal; T_x=0 → 0; replace-one attained `√(2kL) = 5.656854` (disjoint sets) > `√(2kL(1−k/E)) = 5.2915` (that form is **violated**); general tight form `√(2·min(k,E−k)·L)`; design's `√(2kL)` correct | `verifier/k3_structural_bound.py` |
| **K4** | `CE + α(S − sg S)`: value bit-identical to CE, gradient bit-identical to `∇(CE + αS)`, under `torch.func.grad` and `vmap`, fp32/fp64 | PASS | `torch.equal(value)` True ×4; grad max\|diff\| 0.0 ×4; `α∇S` present (5.6e-4…7.4e-4); vmap vs loop 0.0 | `verifier/k4_value_neutral.py` |

Auditor-only checks folded into the rows above: re-run determinism (9 timing leaves differ, nothing else); checkpoint glue
active; dense MoE path at E = 8; dead-zone rates `5.12e-2` (E=8) / `4.25e-6` (E=64); the `u`-sweep cap.

---

## 5. What the final report should say (auditor's summary of findings)

1. **The prototype's seven checks are real and reproducible** — deterministic to every digit on a second run — and its
   implementation follows design §1.1 / §2.2 / §2.4 step for step, with the declared toy-scale deviations (β 0.9, α 1.0,
   ρ* 0.34, emulated `opaque_router_logits` forward, un-amplified MF accountant, plain SGD). Two deviations were **not** declared
   and should be: the MF arm's b-min-sep `sampling_prob` is `p₀` rather than the paper's `p` (expected batch 29.3 vs 32 —
   privacy-neutral, faithfulness-relevant; fixed value 31.98 in `V6b`), and "latch ok 50/50" is an inference from "no
   `ConfigurationError`", not an assertion (explicit assertion added: True). Neither changes any conclusion.
2. **K1 is a confirmed, pre-existing trainer bug with a real privacy consequence**, agreed by three independent computations to
   every printed digit; the report should state it as in §3 above, cite `:1785-1800`, `:3802-3803`, `:3743`, `:3750-3755`,
   `:963-968`, and name the fix ordering (re-fold **after** `from_state_dict`).
3. **K2c**: the design's "pay in ε" column is conservative by ≈ 0.11 ε; replace or annotate. **K3**: keep `√(2kL)`; never
   `√(2kL(1−k/E))`. **K4**: the value-neutral trick is exact, including under vmap.
4. **ρ = 0.02 and c = 2 are preset-regime constants** (B̄ = 256, E = 64): at the prototype's `B̄ = 32, E = 8` the noise floor sits
   above the imbalance and the dead zone's false-pass rate is 5 % per step; ship the r-rule and the χ²_{E−1} statement, and print
   `s_∞/(k/E)` at setup.
5. `α = 1` (lab) lifts per-example norms ×1.24; the design's "aux does not move `C_g`" holds only for α ≤ 1e-3 — the real-checkpoint
   G3 pass at the preset α is still required.
