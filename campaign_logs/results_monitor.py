"""Report each matching run's outcome as it reaches a terminal state."""
import os, sys, time
os.environ.setdefault("WANDB_BASE_URL", "https://jetbrains.wandb.io")
import wandb

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "med-nodp-"
seen = set()
for it in range(120):  # ~6h @ 180s
    try:
        api = wandb.Api(timeout=60)
        runs = list(api.runs("federated-compute/opaque-lora-xs",
                             filters={"display_name": {"$regex": f"^{PREFIX}"}}))
    except Exception as e:
        print(f"[warn] {type(e).__name__}", flush=True)
        time.sleep(180); continue
    for r in runs:
        if r.id in seen or r.state not in ("finished", "failed", "crashed"):
            continue
        seen.add(r.id)
        s = dict(r.summary)
        def g(k, p=5):
            v = s.get(k)
            return f"{v:.{p}f}" if isinstance(v, (int, float)) else "-"
        print(f"RESULT {r.name} state={r.state} loss_min={g('eval/loss_min')} "
              f"loss_fin={g('eval/loss')} r_e_dyn={g('rotation/r_e_dyn',2)} "
              f"r_eff={g('rotation/r_eff_renyi',2)}", flush=True)
    time.sleep(180)
print("MONITOR END", flush=True)
