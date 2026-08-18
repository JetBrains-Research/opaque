"""Cache every W&B run + per-step history into /tmp/xse so an2..an5 can run offline.

    WANDB_BASE_URL=https://jetbrains.wandb.io uv run python fetch_cache.py

Writes /tmp/xse/runs.json (297 runs: config + summary) and /tmp/xse/hist.json
(per-rotation history for the 82 runs that carry the Renyi diagnostic grid).
Takes ~5 min. Re-run only when new runs land.
"""
import json, os, pathlib

os.environ.setdefault("WANDB_BASE_URL", "https://jetbrains.wandb.io")
import wandb

PROJECT = "federated-compute/opaque-lora-xs"
OUT = pathlib.Path("/tmp/xse")
OUT.mkdir(parents=True, exist_ok=True)

# Per-rotation keys. rotation/r_eff_a* is the *whole* alpha curve, logged every
# rotation regardless of the alpha the run was configured with (xse.py:108-123) —
# that is what makes the counterfactual alpha analysis possible without new runs.
KEYS = [
    "_step",
    "rotation/r_eff_a0", "rotation/r_eff_a0p5", "rotation/r_eff_a1",
    "rotation/r_eff_a2", "rotation/r_eff_ainf", "rotation/r_eff_renyi",
    "rotation/r_e_dyn", "rotation/spectral_gap", "rotation/energy_ratio",
    "eval/loss", "train/noise_std", "xs/m_info", "xs/r_effective_rank",
    "xs_spread/rec_rank_std", "xs_spread/rec_rank_min",
    "xs_spread/rec_rank_max", "xs_spread/rec_rank_median",
]


def scalar(v):
    return v if isinstance(v, (int, float, str, bool, type(None))) else str(v)[:200]


def main() -> None:
    api = wandb.Api(timeout=120)
    runs = list(api.runs(PROJECT))
    print(f"{len(runs)} runs")

    meta = []
    for i, r in enumerate(runs):
        try:
            meta.append(dict(
                name=r.name, id=r.id, state=r.state, created=str(r.created_at),
                config={k: v for k, v in r.config.items() if not k.startswith("_")},
                summary={k: scalar(v) for k, v in dict(r.summary).items()},
            ))
        except Exception as e:  # a handful of very old runs have unreadable summaries
            meta.append(dict(name=r.name, id=r.id, state=f"ERR:{e!s:.80}"))
        if i % 50 == 0:
            print(" meta", i, flush=True)
    (OUT / "runs.json").write_text(json.dumps(meta))

    targets = [m for m in meta if m.get("summary", {}).get("rotation/r_eff_a1") is not None]
    print(f"{len(targets)} runs carry the Renyi grid")

    hist = {}
    for i, m in enumerate(targets):
        try:
            run = api.run(f'{PROJECT}/{m["id"]}')
            hist[f'{m["name"]}|{m["id"]}'] = [
                {k: v for k, v in row.items() if v is not None}
                for row in run.scan_history(keys=KEYS, page_size=2000)
            ]
        except Exception as e:
            print(" ERR", m["name"], str(e)[:150])
        if i % 15 == 0:
            print(" hist", i, flush=True)
    (OUT / "hist.json").write_text(json.dumps(hist))
    print(f"wrote {OUT}/runs.json ({len(meta)}) and {OUT}/hist.json ({len(hist)})")


if __name__ == "__main__":
    main()
