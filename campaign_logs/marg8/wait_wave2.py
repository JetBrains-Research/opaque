"""Poll W&B until the batch-B wave-2 runs are terminal, then print the analysis inputs.

    WANDB_BASE_URL=https://jetbrains.wandb.io uv run python campaign_logs/marg8/wait_wave2.py

NOTE: a fresh wandb.Api() is constructed on EVERY poll. Reusing one Api instance caches
run objects, which made an earlier version of this script report step=3 for five hours
while the runs had actually finished. Do not "optimise" that away.
"""
import os, sys, time

os.environ.setdefault("WANDB_BASE_URL", "https://jetbrains.wandb.io")
import wandb

PROJECT = "federated-compute/opaque-lora-xs"
REGEX = "^(marg8-ctrl-fixed-re6-s42|marg8-nodp-ainf-m8-s43)$"
EXPECT = 2
TERMINAL = {"finished", "crashed", "failed", "killed"}
POLL = 300
MAX_WAIT = 5 * 3600

# pre-registered expectations (docs/alpha-margin-experiment-prep.md 2.3)
EXPECTED = {
    "marg8-ctrl-fixed-re6-s42":
        "uniform depth 6.00; matches alpha=0.5@m=8 (0.34423) to within the floor "
        "=> heterogeneity adds nothing. A >3e-4 win for the ADAPTIVE arm would reopen 6.1.",
    "marg8-nodp-ainf-m8-s43":
        "depth 6.98 (same as s42); paired with s42's 0.34404 gives the first floor near depth 7.",
}


def rows():
    api = wandb.Api(timeout=60)          # FRESH each poll — see module docstring
    out = []
    for r in api.runs(PROJECT, filters={"display_name": {"$regex": REGEX}}):
        s = dict(r.summary)
        out.append((r.name, r.state, s.get("_step"), s.get("rotation/r_e_dyn"),
                    s.get("eval/loss"), s.get("eval/loss_min")))
    return sorted(out)


def main() -> int:
    t0 = time.time()
    while True:
        try:
            rs = rows()
        except Exception as e:
            print(f"[poll error] {e!s:.120}", flush=True)
            time.sleep(POLL)
            continue

        el = int(time.time() - t0)
        print(f"\n--- t+{el // 60}m  ({len(rs)}/{EXPECT} registered) ---", flush=True)
        for nm, st, step, d, l, lm in rs:
            f = lambda v, p=5: "-" if v is None else f"{v:.{p}f}"
            print(f"  {nm:28s} {st:9s} step={str(step):>5s} depth={f(d, 3):>8s} "
                  f"loss={f(l):>9s} min={f(lm):>9s}", flush=True)

        done = [r for r in rs if r[1] in TERMINAL]
        if len(rs) >= EXPECT and len(done) == len(rs):
            print(f"\nALL TERMINAL after {el // 60}m\n", flush=True)
            for nm, st, step, d, l, lm in done:
                ok = st == "finished" and (step or 0) >= 260
                print(f"  {nm}: state={st} step={step} "
                      f"{'READABLE' if ok else 'DISCARD (see gotcha: state AND step)'}")
                print(f"    depth={d} loss_min={lm}")
                print(f"    pre-registered: {EXPECTED.get(nm, '')}\n")
            return 0

        if time.time() - t0 > MAX_WAIT:
            print("\nGAVE UP after 5h. Non-terminal: "
                  + ", ".join(f"{r[0]}={r[1]}" for r in rs if r[1] not in TERMINAL), flush=True)
            return 1
        time.sleep(POLL)


if __name__ == "__main__":
    sys.exit(main())
