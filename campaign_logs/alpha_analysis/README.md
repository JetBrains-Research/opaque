# α-verdict analysis scripts

Every number in `docs/renyi-alpha-theory-final.md` comes from these four scripts, run against a
local cache of W&B. No GPU time involved.

```bash
cd campaign_logs/alpha_analysis
export WANDB_BASE_URL=https://jetbrains.wandb.io

uv run python fetch_cache.py   # ~5 min -> /tmp/xse/{runs,hist}.json  (297 runs, 82 with the Renyi grid)
uv run python an2.py           # depth trajectories, N_alpha grid per run, alpha dose at matched margin
uv run python an3.py           # mediator curve L(d), replicate floors, the mediation bound table
uv run python an4.py           # dose-response rho, noise response of N_alpha, heterogeneity audit
uv run python an5.py           # drift vs alpha, r-dependence, Theorem 2 certificates per run
uv run python an6.py           # sweep of all ~35 rotation/ and xs/ metrics
uv run python an7.py           # corrected readings + the two retractions (sec 9 of the doc)
uv run python an8.py           # per-alpha collapse audit, span vs operating point, margin dominance
uv run python an9.py           # the margin objection: compensation test, ranking stability, dose-vs-m
uv run python an11.py          # validates the iid/MP momentum model; bulk edge in absolute units
```

`an6.py` / `an7.py` read `/tmp/xse/hist2.json`, a second cache with the wider metric key set. Build it
by re-running `fetch_cache.py` with `KEYS` extended to the `rotation/*` and `xs/*` fields listed at the
top of `an6.py`.

`an*.py` read only the cache, so they are instant and offline. Re-run `fetch_cache.py` when new runs
land (e.g. the `fixed-re13` seed triple in §8 of the doc).

| script | produces which section of the doc |
|---|---|
| `an2.py` | §3.2 realised depth, §4.3 dose columns, the "N_0.5 < 2 always" collapse check |
| `an3.py` | §4.1 mediator curve, §4.2 replicate floors A–E, §4.3 bound table, §5 spike-count column |
| `an4.py` | §4.4 Spearman/Pearson dose-response, §5 cross-regime noise response, §3.2 heterogeneity |
| `an5.py` | §6 drift + sensitivity table, §5 r-dependence, §3.1 certificate table |
| `an6.py` | §9 raw sweep: cut-point gaps, promotion counts, alpha span, cond(R) |
| `an7.py` | §9.1–9.6 as published, incl. the `grad_snr` and spectral-gap retractions |
| `an8.py` | §6.1 which α are collapsed where + span-vs-depth, §6.2 the margin dominance table |
| `an9.py` | §6.3 the margin objection: 4-margin grid, amplification, ranking, span-vs-spread |
| `an10.py` | first pass at the bulk edge (superseded by an11; kept for the normalisation hunt) |
| `an11.py` | §8.3 validates the iid/MP momentum model vs the accountant; the absolute-units count |
| `an12.py` | §11 rotation schedule: promotion-vs-time, drift direction, schedule residuals, cool-down |

The analysis leans on one lucky property of the instrumentation: `rotation/r_eff_{a0,a0p5,a1,a2,ainf}`
is logged **every rotation regardless of the α the run was configured with** (`xse.py:108-123`), so a
single run yields the full α curve. That is why the α question could be settled from runs already on
disk.
