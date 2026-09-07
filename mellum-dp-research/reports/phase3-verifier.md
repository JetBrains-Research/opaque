# Phase 3 — verifier: independent re-check of K1–K4 of the v2 design

Agent: `verifier`. Repo `/home/user/opaque` @ `ef1abc5` (branch `claude/mellum-dp-representation-r6slaz`); `git status` clean
before and after (0 tracked files modified). Target: `phase2-final-design-v2.md` §8.1/§9.0 (K1), §2.2 step 8 + §2.6 (K2),
§1.1/§2.1 (K3), §1.1 (K4). All scripts are my own, written before reading the earlier agents' scripts, and live in
`scratchpad/research/verifier/` with their raw outputs (`*.out`). CPU only, `torch.set_num_threads(2)`; every script ran in
< 60 s. Evidence tags: **VERIFIED** = I read the cited lines or ran the cited script in this session; **PLAUSIBLE** = derived,
not executed end to end. Repo paths relative to `/home/user/opaque`.

Summary of verdicts:

| claim | verdict |
|---|---|
| K1a — DDP + resume restores rank 0's sampler key on every rank, no rank re-fold (design §8.1) | **CONFIRMED** by code reading and by a sampler-level reproduction: 20/20 post-resume inclusion masks identical across "ranks" for both `PoissonSampler` and `BMinSepSampler` |
| K1b — privacy effect: per-step δ(ε=3) 3.16e-11 (independent coins) vs 2.76e-5 (shared coin) | **CONFIRMED** with my own integration (3.1597e-11 / 2.7587e-05, ratio 8.7e5); the characterisation "keeps δ-amplification, loses ε-amplification, six orders of magnitude" is **right per step and under-stated for the run**: a rigorous lower bound puts the whole-run ε(δ=1e-6) at ≥ 6.0 instead of 3 |
| K2a — accountant invariance (probe group changes nothing in `_build_mechanism`) | **CONFIRMED**: `num_groups` is consumed only inside the `clipping_mode == "adaptive"` closure; baseline ε = 3.0004 |
| K2b — allocator: `σ_g/(nm·C_g) = √(1+ρ)`, Mahalanobis `= 1/nm²` | **CONFIRMED** at six ρ to 6 digits with the real `per_group_noise_stddev` |
| K2c — "pay in ε" column 3.234 (ρ=0.02) / 3.703 (ρ=0.1) | **REPRODUCED EXACTLY** (3.2345 / 3.7035) with the accountant call the earlier agents used, `poisson(gaussian(nm) \| gaussian(nm_h), q)*T`; **but that call is a conservative upper bound**: the mathematically identical joint Gaussian `poisson(gaussian(nm_eff), q)*T`, `nm_eff = nm·√((1+ρ)/(1+2ρ))`, gives **3.1256 / 3.5897**. The column overstates the ε price by ≈ 0.11 at every row (accountant generic-inner Poisson path is looser than its exact-Gaussian path). Minor: the column is informational, not the shipped route |
| K3 — `‖d^{(L,E)}(x)‖₂ ≤ √(kL(1−k/E))` with equality attained | **CONFIRMED** (random + adversarial + exhaustive at E=8; max = bound to 1e-6; T_x=0 → 0). Replace-one: the tight per-layer bound is **`√(2·min(k, E−k)·L)` = `√(2kL)` for E ≥ 2k** (attained by disjoint expert sets); `√(2kL(1−k/E))` is **violated** (5.29 < 5.66 attained at E=64,k=8,L=2). The design's §2.1 `√(2kL)` is correct and tight |
| K4 — value-neutral surrogate | **CONFIRMED**: value bit-identical to CE (`torch.equal`), gradient bit-identical to `∇(CE + α·S)` (max\|diff\| = 0.0) under `torch.func.grad` and under `vmap`, fp32 and fp64; vmap vs per-example loop 0.0 |

---

## K1 — DDP + checkpoint-resume sampler-key bug

### K1.1 Code reading (VERIFIED)

Fresh construction, `packages/opaque-transformers/src/opaque/api/transformers/trainer/_dp_trainer.py`:

- `:3785-3803` — `if ctx.current_sampler is None:` … `sampler_key = key(a.data_seed or a.seed)` (`:3788`), optional
  ignore-data-skip fold (`:3789-3797`), then **`if self._ddp.world_size > 1: sampler_key = fold_in(sampler_key, self._ddp.rank)`
  (`:3802-3803`)**, then `_dpftrl.build_sampler(..., key=sampler_key, ...)` (`:3804-3816`). The dataset passed is the rank's
  contiguous shard after the equal-length trim (`:3758-3769`; `_effective_train_dataset_size` `:4534-4564` trims to
  `(n // world_size) * world_size`; `opaque-engine/.../distributed/_shard.py:20-33`).
- `build_sampler` (`trainer/_dpftrl.py:180-211`) constructs `PoissonSampler(dataset, sample_rate, n_steps, key=key)` for
  `sampling_mode="poisson"` and `BMinSepSampler` for `"b_min_sep"` (`:236+`).
- `PoissonSampler.__init__` domain-separates once more: `stream_key = fold_in(key, POISSON_STREAM_FOLD)`
  (`opaque-dpsgd/.../sampling/_poisson.py:97`); `BMinSepSampler` likewise (`opaque-dpftrl/.../sampling/_b_min_sep.py:66`).
  So the live stream key on rank r is `fold_in(fold_in(key(seed), r), FOLD)` — rank-specific.

Checkpoint save:

- `_save_checkpoint` `:4910+` — the DP runtime is written only when `_distributed.should_save(a, self._ddp)` (`:4962`, the
  call `self._save_dp_runtime(staging_dir, ctx)` at `:5004` is inside that block). `should_save`
  (`trainer/_distributed.py:209-220`): world-rank 0 only unless `save_on_each_node` (then local zero). **One snapshot, from
  rank 0.**
- `_save_dp_runtime` `:5070-5079`: `sampler_state = opaque_state_dict(ctx.current_sampler)`.
- `_state_dict_poisson` (`_poisson.py:210-227`) serialises **`key_seed = s._stream_key.seed`** — i.e. rank 0's already
  rank-folded, domain-separated stream key — plus `consumed`, `num_samples`, `sample_rate`. `_state_dict_b_min_sep`
  (`_b_min_sep.py:209-227`) does the same.

Resume:

- `_inner_training_loop` `:1785-1800`: when `resume_path` and `saved_sampler_state` are set and `not a.ignore_data_skip`,
  a template sampler is built through `get_train_dataloader()` (fresh construction, **with** the rank fold) and then
  **overwritten**: `ctx.current_sampler = from_state_dict(ctx.current_sampler, saved_sampler_state)` (`:1799-1800`).
- `_from_state_dict_poisson` (`_poisson.py:229-266`): takes `data_source` and `n_steps` from the template but
  **`stream_key = RngKey(seed=int(sd["key_seed"]), impl=...)` from the snapshot (`:261`)**, then replays `consumed` steps.
  Nothing re-folds by rank. `_from_state_dict_b_min_sep` (`_b_min_sep.py:229-266`, key restore at `:259`) is identical in
  this respect. `saved_sampler_state` comes from `runtime_payload.sampler_state` read from `dp_state.pt` (`:1244-1246`,
  `:5317-5318`) — the same file on every rank.

**Conclusion (VERIFIED by reading): a restored sampler on rank r ≠ 0 does NOT re-derive a rank-specific key; every rank
continues on rank 0's stream, so records sharing a local shard index on different ranks have identical inclusion coins for the
rest of the run.** The `num_samples` template check passes on every rank precisely because the trimmed shards are equal-length
(`:4555`) — had they been unequal the restore would have raised `ConfigurationError` instead of silently correlating.

What the trainer comments say (VERIFIED, quoted):

- `:3736-3744` (fresh-construction block): "The sampler key is folded by rank (below) so each rank draws an **independent**
  Bernoulli(q) mask: with a shared key every rank would select the *same* local offsets, perfectly co-including the records
  that happen to share a local index across shards — not the i.i.d. global Poisson draw the design intends (the per-record
  marginal stays Bernoulli(q) either way, **so the privacy accounting is unaffected**; this is a sampling *diversity* fix)."
- `:3750-3756`: "Resume caveat (multi-GPU only): the sampler snapshot is self-contained (carries its own key) and is written
  once on rank 0, so resuming a DDP run currently restores rank 0's per-rank key on every rank, re-introducing the cross-rank
  correlation after the resume point. Fully fixing that needs per-rank sampler snapshots; tracked for the multi-GPU work and
  validated there."

Citation nit for the design: §8.1 attributes both quotes to `:3750-3757`; the "accounting is unaffected" sentence is at
`:3741-3744` (it is the general shared-key comment, which the resume caveat then says is re-introduced). Substance unchanged.
The docstring at `:953-968` ("Privacy budget is unchanged … DP-valid either way") is also wrong for the multi-rank case.

No existing test covers this: the 7 `pytest.mark.distributed` tests under `packages/opaque-transformers/tests` contain no
`resume`/`from_state_dict` (VERIFIED by grep), so T25 of the design is genuinely new.

### K1.2 Sampler-level reproduction without DDP (VERIFIED — `verifier/k1_sampler_resume.py` → `.out`)

Construction mirrors the trainer exactly: dataset of 2000, world size 2, `local_shard` after the equal-length trim,
`fold_in(key(1234), rank)`, `PoissonSampler(shard, q=0.05, n_steps=60, key=…)` / `BMinSepSampler(shard, bands=4,
p = q/(1−q(b−1)), …)`; 7 steps consumed on both ranks; `state_dict(rank0)`; resume on each rank = fresh template
`from_state_dict(template_r, sd_rank0)`; compare the next 20 masks.

| | PoissonSampler | BMinSepSampler |
|---|---|---|
| pre-checkpoint, rank 0 vs rank 1 identical steps | 0 / 7 | 0 / 7 |
| rank 1 live stream-key seed vs rank 0 | 12442763432782722717 vs 7692528677094032581 (differ) | 4350963696868884131 vs 3458620349621444162 (differ) |
| restored rank 1 stream-key seed == rank 0's | **True** | **True** |
| post-resume identical masks (restored rank 1 vs restored rank 0) | **20 / 20** (100.0000 % of entries) | **20 / 20** |
| restored rank 0 == continuous rank 0 (resume exactness, single-process) | True | True |
| restored rank 1 vs continuous rank 1 (what a correct resume would give) | 0 / 20 identical | 0 / 20 |
| co-inclusion rate of the same local index across ranks after resume | **0.0506** (q = 0.05; independent would be q² = 0.0025) | 0.0503 |

Both samplers reproduce the bug; single-process resume is exact (as the trainer docstring claims for that case).

### K1.3 Privacy effect — own 1-D hockey-stick integration (VERIFIED — `verifier/k1_hockey_stick.py` → `.out`)

Grid `[-14σ−1, 14σ+3]`, 4 000 001 points, float64, `σ = 0.5622`, `q = 256/5e5 = 5.12e-4`, sensitivity 1; both directions
computed, the max reported (the remove direction is 0 at these ε). Grid accuracy checked against the analytic Gaussian
`δ_G(3) = Φ(1/(2σ) − 3σ) − e³Φ(−1/(2σ) − 3σ) = 1.1227e-1`: relative error 3e-11.

| pair | δ(ε = 3) |
|---|---|
| (a) independent coins: `P = N(0)`, `Q = (1−q)N(0) + qN(1)` | **3.1597e-11** |
| (b) shared coin, `u = 0` (partner contributes nothing) | 3.1597e-11 (= (a), sanity) |
| (b) shared coin, `u = 0.5` | 1.0164e-06 |
| (b) shared coin, `u = 1` (partner aligned, e.g. a duplicate record): `((1−q)N(0)+qN(1), (1−q)N(0)+qN(2))` | **2.7587e-05** — ratio to (a) **8.73e5** |
| (b') shared coin, `u = 1`, x anti-aligned | 3.1597e-11 |
| unsubsampled `q·δ_G(3)` (analytic) | 5.75e-5 (same order as (b), i.e. δ-amplification by ≈ q is retained, ε-amplification is not) |

Same numbers as `refute-composition/corr_coins.out` to all printed digits (3.160e-11 / 2.759e-05 / 1.016e-06) from
independent code — the design's figures are right. The pair in (b) is a legitimate add/remove neighbour: `D = {y, …}` vs
`D ∪ {x}` with `y` the record sharing `x`'s local index on the other rank; the 1-D projection onto `g_y`'s direction is a
post-processing, so the divergence of the full-dimensional release is at least this (and equal when `g_x ∥ g_y`). DP is a
worst case over datasets and `y = x` (a duplicate) realises `u = 1`.

Across ε (independent / shared u=1 / ratio): 0.5: 1.50e-7 / 1.43e-4 / 9.5e2; 1: 1.85e-8 / 1.01e-4 / 5.5e3; 2: 7.09e-10 /
5.46e-5 / 7.7e4; **3: 3.16e-11 / 2.76e-5 / 8.7e5**; 4: 1.21e-12 / 1.19e-5 / 9.8e6; 6: 8.09e-16 / 1.01e-6 / 1.3e9. The ε at
which the shared-coin pair reaches the accountant's per-step δ = 3.16e-11 is **10.52** instead of 3.

Whole-run consequence (VERIFIED — `verifier/k1_run_bounds.py` → `.out`): because the T-fold composition post-processes to its
first step, `δ_T(ε) ≥ δ_1(ε)` and `ε_T(δ) ≥ ε_1(δ)` rigorously. Single-step `ε_1(δ = 1e-6)` = **0.210** for independent coins
vs **6.006** for the shared coin. Hence after a DDP resume the run is **not** (3, 1e-6)-DP for the worst-case pair: **ε ≥ 6.0 at
δ = 1e-6** is a rigorous lower bound (the true composed value is much larger; a crude RDP upper bound over 15 625 steps gives
≤ 44, PLAUSIBLE — α-grid with overflow warnings at large α, treat as order-of-magnitude only; the independent-coin RDP bound on
the same grid is 4.9 vs the accountant's tight 3.0).

**Assessment of the design's characterisation.** "Loses ε-amplification, keeps δ-amplification, six orders of magnitude in
per-step δ at ε = 3" is **right** (8.7e5 ≈ 5.9 orders; the shared-coin δ sits at ≈ q·δ_G/2, i.e. the δ-amplification factor q
survives and nothing else does). It is **under-stated** in two respects that the design could state explicitly: (i) the
per-step ε at the accountant's δ is 10.5, not 3; (ii) the whole-run guarantee at δ = 1e-6 is provably ≥ 6.0 (and realistically
tens), not 3 — i.e. the accountant's certificate is void, not merely loose. It is **not over-stated**: my only softening is that
the effect needs an aligned partner (`u = 1`), which the worst-case-over-datasets definition grants (duplicates). The design's
disposition (P0 fix: re-fold by rank at `:1799-1800`, or per-rank snapshots; narrow the privacy statement to "single-process,
or DDP without checkpoint resume"; T25) is the right one. Note also the not-yet-stated fix subtlety: with `world_size = W`
each local index is shared by W records, so the re-fold must be applied *after* `from_state_dict` (the restore overwrites the
template's folded key, `_poisson.py:261`), e.g. by restoring only the cursor onto the rank-folded template, or by writing
`W` snapshots.

---

## K2 — accountant invariance and the cost-table default row

### K2.1 Accountant call path (VERIFIED by reading `_dp_trainer.py:4301-4407`)

`_build_mechanism`: non-Gaussian mechanisms return the MF `mf_amplifier_factory` untouched (`:4319-4327`); for Gaussian,
`num_groups = clip_norm.num_groups if isinstance(clip_norm, PerGroup) else 1` (`:4329`) is referenced **only** inside the
`if a.clipping_mode == "adaptive":` closure that wraps the base in `dpsgd_acc.adaclip(..., num_groups=num_groups)`
(`:4332-4340`). With `clipping_mode == "fixed"` the chain is `poisson(_unamplified(nm), sample_rate)` (`:4404-4405`) or the
truncated variant (`:4390-4402`), neither of which sees the `PerGroup`. `grep -n num_groups` over the file returns only
`:4329` and `:4339`. **Adding the probe group changes nothing in the accountant call** (the design's step 8 is correct; the
design requires `clipping_mode == "fixed"` with the feature, §2.5, so the adaclip branch is excluded).

Baseline (VERIFIED — `verifier/k2_accounting.py` → `.out`): `poisson(gaussian(0.5622), 256/5e5) * 15625` at `δ = 1e-6` →
**ε = 3.0004** (0.3 s), matching the design's 3.0004. Naive per-group `σ_i = nm·C_i` for two groups (`= gaussian(nm/√2)`) would
give ε = 11.84 — the design correctly never uses it.

### K2.2 Real allocator (VERIFIED — same script)

`per_group_noise_stddev` (`opaque-engine/.../noise_allocation.py:106-109`: `σ_k = nm·√(c_k·Σc)`) with
`PerGroup(values={"fallback": 0.9, "router_load_probe": ρ·0.9})`, `nm = 0.5622`:

| ρ | σ_g/(nm·C_g) | √(1+ρ) | σ_h/(nm·C_h) | √(1+1/ρ) | `(C_g/σ_g)² + (C_h/σ_h)²` · nm² |
|---|---|---|---|---|---|
| 0.5 | 1.224745 | 1.224745 | 1.732051 | 1.732051 | 1.000000 |
| 0.2 | 1.095445 | 1.095445 | 2.449490 | 2.449490 | 1.000000 |
| 0.1 | 1.048809 | 1.048809 | 3.316625 | 3.316625 | 1.000000 |
| 0.05 | 1.024695 | 1.024695 | 4.582576 | 4.582576 | 1.000000 |
| 0.02 | 1.009950 | 1.009950 | 7.141428 | 7.141428 | 1.000000 |
| 0.01 | 1.004988 | 1.004988 | 10.049876 | 10.049876 | 1.000000 |

`σ_g/(nm·C_g) = √(1+ρ)` and the Mahalanobis identity hold to 6 digits — the "×1.010 at ρ = 0.02, ε unchanged" row is confirmed.

### K2.3 "Pay in ε" column (VERIFIED — `k2_accounting.py`, `k2_composed_inner_v2.py` → `.out`)

Derivation. Hold the gradient noise at `nm·C_g` and release the load at `σ_h = nm·C_h·√(1+1/ρ)` on the **same** batch. The
joint release is one Gaussian on the concatenation with whitened sensitivity² `1/nm² + 1/nm_h²`, `nm_h = nm·√(1+1/ρ)`, i.e.
`gaussian(nm_eff)` with `1/nm_eff² = 1/nm² + 1/nm_h²` ⇔ **`nm_eff = nm·√((1+ρ)/(1+2ρ))`** (asserted in the script to 1e-12). Two
Gaussian PLRVs add, so `gaussian(nm) | gaussian(nm_h)` and `gaussian(nm_eff)` are the *same* single-step PLD — confirmed by the
accountant itself: unsubsampled `delta_at(ε)` agree to 7 significant digits at ε ∈ {0.5, 1, 2, 3} and match the analytic
Gaussian formula (e.g. ρ = 0.02, ε = 3: 1.173564e-1 all three).

| ρ | nm_h | nm_eff | ε of `poisson(g(nm) \| g(nm_h), q)*T` (earlier agents' call) | ε of `poisson(g(nm_eff), q)*T` (joint form) | design |
|---|---|---|---|---|---|
| 0.5 | 0.9738 | 0.48688 | 5.4431 | 5.3047 | 5.443 |
| 0.2 | 1.3771 | 0.52050 | 4.2189 | 4.1001 | 4.219 |
| **0.1** | 1.8646 | 0.53827 | **3.7035** | **3.5897** | 3.703 |
| 0.05 | 2.5763 | 0.54927 | 3.4171 | 3.3062 | 3.417 |
| **0.02** | 4.0149 | 0.55677 | **3.2345** | **3.1256** | 3.234 |
| 0.01 | 5.6500 | 0.55944 | 3.1717 | 3.0635 | 3.172 |

The design's 3.234 / 3.703 **reproduce exactly** with the composed-inner call (I later confirmed that is the call
`refute-sensitivity/costtable.py:22` and `synthesizer/final_table.py:22` used). **But the two columns should be equal and are
not**: the gap is entirely in the Poisson wrapper. `poisson.rs:27-28` takes the exact-Gaussian path
(`poisson_from_gaussian`) only when `base.gaussian_source()` is `Some`; a composed inner has no Gaussian source and goes
through the generic discretised realisation path (`poisson.rs:31-41`, Feldman–Shenfeld Thm 3.3 pointwise transform on the
already-discretised PLD). Single-step Poisson `delta_at(3)` at ρ = 0.02: composed 6.32e-11 vs joint 4.52e-11 (ρ = 0.1:
1.67e-10 vs 1.49e-10). Domination is preserved (the doc comment says so), so **3.234 / 3.703 are valid but conservative upper
bounds; the tight "pay in ε" values for the described same-coin joint mechanism are 3.126 / 3.590** — the column overstates the
alternative's cost by ≈ 0.11 ε at every row. Since the shipped route is "ε held, accountant unchanged", this affects only the
informational column and the narrative "pay in σ or pay in ε are the same family re-parametrised" (they are; the tight ε
number is just smaller than printed). Recommendation: express the column as `poisson(gaussian(nm_eff), q)*T` (or note the
looseness), and — outside this design — the accountant could route a composed-of-Gaussians inner through the exact path.

For completeness, the *separate-coin* form `(poisson(g(nm), q) | poisson(g(nm_h), q))*T` gives ε = 3.0010 (ρ = 0.02) /
3.0033 (ρ = 0.1) — a different mechanism (independent second draw), consistent with phase-1 F6's "independently subsampled
second release is much cheaper", and not applicable to the same-batch design.

---

## K3 — structural bound on the per-layer centred load vector (VERIFIED — `verifier/k3_structural_bound.py` → `.out`)

Definition used (§1.1): `m = (attention_mask != 0)`, `T_x = Σ m`, `p^l = softmax_fp32(z^l)`, `S^l_t = topk_k(p^l_t)` via
`torch.topk`, one-hot by broadcast compare, `h^l = (m @ onehot^l)/T_x`, `d^l = h^l − k/E`, explicit `where(T_x > 0, ·, 0)`;
`‖d^{(L,E)}‖² = Σ_l ‖d^l‖²`. L = 2, E ∈ {8, 64}, k ∈ {2, 8}, T ∈ [1, 64]; 400 random routings per cell (logit scales
0.1/1/10, mask densities 0.3/0.7/1.0), adversarial "all tokens → the same k experts", T = 1, and for E = 8 an exhaustive
enumeration over all `C(E,k)^L` vertex combinations.

| E | k | `√(kL(1−k/E))` | max random | adversarial same-k | T = 1 | exhaustive (E=8) |
|---|---|---|---|---|---|---|
| 8 | 2 | 1.732051 | 1.732051 | 1.732051 | 1.7321 | 1.732051 |
| 8 | 8 | 0.000000 | 0.000000 | 0.000000 | 0.0000 | 0.000000 |
| 64 | 2 | 1.968502 | 1.968502 | 1.968502 | 1.9685 | — |
| 64 | 8 | 3.741657 | 3.741657 | 3.741657 | 3.7417 | — |

No violation (assert at bound + 1e-6 never fired over 1 600 random draws); **equality is attained** by any example whose every
layer routes every token to one fixed k-set (and by every T = 1 example, since a single token is a vertex). Fully-masked rows
return 0. The bound and its tightness hold; with `w_x = T_x/T̄ ≤ T_max/T̄` the group bound `Δ_L = (T_max/T̄)·√(kL(1−k/E))`
follows by homogeneity (PLAUSIBLE, trivial).

Replace-one (`‖w_x d(x) − w_{x'} d(x')‖`, `T̄ = T_max`):

| E | k | max random pairs | adversarial disjoint k-sets (w = w' = 1) | adversarial vs empty row | `√(2kL)` | `√(2kL(1−k/E))` | `√(2·min(k,E−k)·L)` |
|---|---|---|---|---|---|---|---|
| 8 | 2 | 0.407 | **2.828427** | 1.732 | 2.8284 | 2.4495 | 2.8284 |
| 8 | 8 | 0.000 | 0.000000 | 0.000 | 5.6569 | 0.0000 | 0.0000 |
| 64 | 2 | 0.375 | **2.828427** | 1.969 | 2.8284 | 2.7839 | 2.8284 |
| 64 | 8 | 0.717 | **5.656854** | 3.742 | 5.6569 | 5.2915 | 5.6569 |

By construction the per-layer replace-one bound is **`√(2·min(k, E−k)·L)`**, which equals **`√(2kL)`** whenever `E ≥ 2k`
(the preset: E = 64, k = 8 → `√(16L)`; pooled L = 1 → `√16 = 4.000`, agreeing with the design's A4 figure): the centring
cancels in a difference, so it is the maximal distance between two vertices of `{0 ≤ h ≤ 1, Σh = k}` per layer, `|S Δ S'| =
2·min(k, E−k)`, summed over L independent layers. **`√(2kL(1−k/E))` is not a valid bound** — the disjoint-set pair attains
5.657 > 5.292 at E = 64, k = 8, L = 2. Which is tight: `√(2kL)` (design §2.1 says exactly this — correct), and it is strictly
tighter than the generic doubling `2√(kL(1−k/E))` (7.48 for that cell). Mixed weights do not help the adversary: with
`w ≤ 1`, `‖w(h−c) − w'(h'−c)‖² = 7w² + 7w'² + 2ww'` (E = 64, k = 8, per layer, disjoint sets), maximised at `w = w' = 1`.

---

## K4 — value-neutral surrogate (VERIFIED — `verifier/k4_value_neutral.py` → `.out`)

Tiny model (embedding → linear router → top-2 of 8 SiLU experts with renormalised weights → LM head, V = 40, T = 12, B = 6),
`S = E·w_x·Σ_e (f̃_e − k/E)·P_e(x)` with an imbalanced random `f̃`, `α = 1e-3`, `f = functional_call`:

| setting | `value(CE + α(S − sg S)) == value(CE)` (`torch.equal`) | `value(CE + αS) − value(CE)` | `grad` neutral vs full: bit-identical / max\|diff\| | `α∇S` present (max) |
|---|---|---|---|---|
| fp32, `torch.func.grad` | True | 1.93e-4 | True / 0.0 | 5.6e-4 |
| fp32, `vmap(grad_and_value)` | True | 2.42e-4 | True / 0.0 | 7.4e-4 |
| fp64, `torch.func.grad` | True | 1.93e-4 | True / 0.0 | 5.6e-4 |
| fp64, `vmap(grad_and_value)` | True | 2.42e-4 | True / 0.0 | 7.4e-4 |

Also `vmap` vs per-example loop gradient of the neutral loss: max|diff| = 0.0. The value is exactly `CE` (IEEE `a − a = +0`
for finite `a`, and `CE + α·0 = CE`), and the gradient is exactly `∇CE + α∇S` (the subtraction's backward routes 1 to `S`
and nothing to the detached branch; the autograd graph is otherwise identical, so the sums happen in the same order and the
result is bit-identical, not merely close). The design's §1.1 claim holds, including under vmap.

---

## Cross-agent comparison (read after my runs)

- `refute-composition/corr_coins.py` uses `np.trapz` on a 400 001-point grid over `[-12, 14]`; my rectangle rule on
  4 000 001 points over `[-8.9, 10.9]` agrees to every printed digit for all (σ, q, ε, u) it reports.
- `refute-sensitivity/costtable.py:22` and `synthesizer/final_table.py:22` compute the "pay in ε" column as
  `poisson(gaussian(nm) | gaussian(c·nm), q)*T` — the composed-inner form whose looseness K2.3 quantifies.
- `refute-sensitivity/adversarial.py:65-67` (A4) checks the pooled replace-one `√(2k) = 4.000`; my per-layer `√(2kL)` reduces
  to it at L = 1.

## Findings for the design (what to change)

1. §2.6 "pay in ε" column: replace 3.234 / 3.703 (and the other rows) by the tight joint-Gaussian values 3.126 / 3.590
   (5.305, 4.100, 3.306, 3.064 for ρ = 0.5, 0.2, 0.05, 0.01), or annotate that the printed values are the accountant's
   conservative composed-inner bound. No effect on the shipped route (ε held, ×√(1+ρ) on σ_g), which is confirmed exactly.
2. §8.1: cite `:3741-3744` for "accounting is unaffected" and `:3750-3756` for the resume caveat; add `:953-968` (docstring
   "Privacy budget is unchanged … DP-valid either way") to the comments to correct. Optionally strengthen the impact
   statement with the rigorous run-level bound (ε(1e-6) ≥ 6.0 for the worst-case pair; per-step ε at the accountant's δ is
   10.5). The P0 fix must apply the rank fold *after* `from_state_dict` (the restore overwrites the template's key,
   `_poisson.py:261`, `_b_min_sep.py:259`).
3. §2.1 replace-one: keep `√(2kL)`; if a general form is wanted it is `√(2·min(k, E−k)·L)`. Do not use `√(2kL(1−k/E))`.

## Scripts and outputs

`scratchpad/research/verifier/`: `k1_sampler_resume.py/.out`, `k1_hockey_stick.py/.out`, `k1_run_bounds.py/.out`,
`k2_accounting.py/.out`, `k2_composed_inner.py/.out` (first attempt; its unsubsampled T-fold cross-check hit the accountant's
FFT-buffer limit — `RuntimeError: Exact self-composition requires an FFT buffer of 8589934592 elements` — and was replaced by
the single-step comparison), `k2_composed_inner_v2.py/.out`, `k3_structural_bound.py/.out`, `k4_value_neutral.py/.out`.
