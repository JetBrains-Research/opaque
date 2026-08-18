"""Wait until named runs are terminal, then print them with the readability check applied.

    WANDB_BASE_URL=... uv run python campaign_logs/queue/wait_runs.py NAME:STEPS [NAME:STEPS ...]

e.g.  wait_runs.py tau1-ep2-fixed-re5-s42:520 tau2-ep2-fixed-re5-s42:520

A run counts as READABLE only if state == "finished" AND it reached its full step count.
Reading a partial run as if it were complete has caused a published retraction in this
project, so the check is enforced here rather than left to the reader.

NOTE: a FRESH wandb.Api() is built on every poll. Reusing one instance caches run objects
and reports stale state -- that cost five hours of false "step=3" earlier in this campaign.
"""
import os, sys, time

os.environ.setdefault("WANDB_BASE_URL", "https://jetbrains.wandb.io")
import wandb

PROJECT = "federated-compute/opaque-lora-xs"
TERMINAL = {"finished", "crashed", "failed", "killed"}
POLL, MAXWAIT = 600, 10 * 3600


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    want = {}
    for a in argv:
        nm, _, st = a.partition(":")
        want[nm] = int(st) if st else 0

    t0 = time.time()
    while True:
        try:
            api = wandb.Api(timeout=60)          # fresh per poll - see module docstring
            seen = {
                r.name: (r.state, dict(r.summary).get("_step"),
                         dict(r.summary).get("rotation/r_e_dyn"),
                         dict(r.summary).get("eval/loss_min"))
                for r in api.runs(PROJECT, filters={"display_name": {"$in": list(want)}})
            }
        except Exception as e:
            print(f"[poll error] {e!s:.110}", flush=True)
            time.sleep(POLL)
            continue

        el = int(time.time() - t0)
        print(f"\n--- t+{el // 60}m  ({len(seen)}/{len(want)} registered) ---", flush=True)
        for nm, need in want.items():
            if nm not in seen:
                print(f"  {nm:30s} pending", flush=True)
                continue
            st, step, d, lm = seen[nm]
            f = lambda v, p=6: "-" if v is None else f"{v:.{p}f}"
            print(f"  {nm:30s} {st:9s} {str(step):>4s}/{need} "
                  f"depth={f(d, 2):>6s} min={f(lm)}", flush=True)

        if len(seen) == len(want) and all(v[0] in TERMINAL for v in seen.values()):
            print("\nALL TERMINAL — readability check:", flush=True)
            for nm, need in want.items():
                st, step, d, lm = seen[nm]
                ok = st == "finished" and (step or 0) >= need
                print(f"  {nm:30s} {'READABLE' if ok else 'DISCARD'}  "
                      f"state={st} step={step}/{need} loss_min={lm}", flush=True)
            return 0

        if time.time() - t0 > MAXWAIT:
            print("\nGAVE UP. Non-terminal: "
                  + ", ".join(f"{n}={v[0]}" for n, v in seen.items() if v[0] not in TERMINAL),
                  flush=True)
            return 1
        time.sleep(POLL)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
