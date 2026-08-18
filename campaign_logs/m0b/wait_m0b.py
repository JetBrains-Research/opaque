"""Poll W&B until every m0b-* run reaches a terminal state, then print a summary.

    WANDB_BASE_URL=https://jetbrains.wandb.io uv run python campaign_logs/m0b/wait_m0b.py

Exits 0 when all 5 are terminal (finished/crashed/failed/killed). Prints a table each
poll so the log doubles as a progress record.
"""
import os, sys, time

os.environ.setdefault("WANDB_BASE_URL", "https://jetbrains.wandb.io")
import wandb

EXPECT = 5
TERMINAL = {"finished", "crashed", "failed", "killed"}
POLL = 300          # 5 min
MAX_WAIT = 6 * 3600  # give up after 6h rather than hang forever


def poll(api):
    rows = []
    for r in api.runs("federated-compute/opaque-lora-xs",
                      filters={"display_name": {"$regex": "^m0b-nodp-"}}):
        rows.append((r.name, r.state, r.summary.get("_step"),
                     r.summary.get("rotation/r_e_dyn"),
                     r.summary.get("eval/loss"), r.summary.get("eval/loss_min")))
    return sorted(rows)


def main() -> int:
    api = wandb.Api(timeout=60)
    t0 = time.time()
    while True:
        try:
            rows = poll(api)
        except Exception as e:                      # transient API/VPN blips
            print(f"[poll error] {e!s:.120}", flush=True)
            time.sleep(POLL)
            continue

        el = int(time.time() - t0)
        print(f"\n--- t+{el // 60}m  ({len(rows)}/{EXPECT} registered) ---", flush=True)
        for nm, st, step, depth, loss, lmin in rows:
            d = "-" if depth is None else f"{depth:.3f}"
            l = "-" if loss is None else f"{loss:.5f}"
            m = "-" if lmin is None else f"{lmin:.5f}"
            print(f"  {nm:26s} {st:9s} step={str(step):>5s} depth={d:>7s} "
                  f"loss={l:>8s} min={m:>8s}", flush=True)

        done = [r for r in rows if r[1] in TERMINAL]
        if len(rows) >= EXPECT and len(done) == len(rows):
            print(f"\nALL {len(done)} TERMINAL after {el // 60}m", flush=True)
            bad = [r[0] for r in done if r[1] != "finished"]
            if bad:
                print(f"NOT finished cleanly: {bad}", flush=True)
            short = [(r[0], r[2]) for r in done if r[1] == "finished" and (r[2] or 0) < 260]
            if short:
                print(f"FINISHED BUT SHORT (<260 steps, do not read): {short}", flush=True)
            return 0

        if time.time() - t0 > MAX_WAIT:
            print("\nGAVE UP waiting (6h). Still not terminal:", flush=True)
            print("  " + ", ".join(f"{r[0]}={r[1]}" for r in rows if r[1] not in TERMINAL),
                  flush=True)
            return 1
        time.sleep(POLL)


if __name__ == "__main__":
    sys.exit(main())
