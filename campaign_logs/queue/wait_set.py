"""Wait until a named set of runs is terminal, then print them with the checks applied.

NOTE: a fresh wandb.Api() per poll. Reusing one instance caches run objects and reports
stale state (this cost five hours of false "step=3" earlier in this campaign).
"""
import os, sys, time
os.environ.setdefault("WANDB_BASE_URL", "https://jetbrains.wandb.io")
import wandb

PROJECT = "federated-compute/opaque-lora-xs"
WANT = {"ep2-fixed-re9-s42": 520, "ep2-fixed-re13-s42": 520,
        "disent-int10-ep2-re1-s42": 520, "disent-int10-ep2-re5-s42": 520,
        "phase-eval7-nodp-s42": 260}
TERMINAL = {"finished", "crashed", "failed", "killed"}
POLL, MAXWAIT = 600, 8 * 3600

def snap():
    api = wandb.Api(timeout=60)          # fresh per poll - do not hoist out of the loop
    out = {}
    for r in api.runs(PROJECT, filters={"display_name": {"$in": list(WANT)}}):
        s = dict(r.summary)
        out[r.name] = (r.state, s.get("_step"), s.get("rotation/r_e_dyn"),
                       s.get("eval/loss_min"), s.get("eval/loss"))
    return out

t0 = time.time()
while True:
    try:
        s = snap()
    except Exception as e:
        print(f"[poll error] {e!s:.100}", flush=True); time.sleep(POLL); continue
    el = int(time.time() - t0)
    print(f"\n--- t+{el//60}m ({len(s)}/{len(WANT)} registered) ---", flush=True)
    for nm, need in WANT.items():
        if nm not in s: print(f"  {nm:28s} pending", flush=True); continue
        st, sp, d, lm, lf = s[nm]
        f = lambda v: "-" if v is None else f"{v:.6f}"
        print(f"  {nm:28s} {st:9s} {str(sp):>4s}/{need} depth={('-' if d is None else f'{d:.2f}'):>6s} "
              f"min={f(lm)} final={f(lf)}", flush=True)
    done = [n for n in WANT if n in s and s[n][0] in TERMINAL]
    if len(s) == len(WANT) and len(done) == len(WANT):
        print("\nALL TERMINAL — readability check (state==finished AND full step count):", flush=True)
        for nm, need in WANT.items():
            st, sp, d, lm, lf = s[nm]
            ok = st == "finished" and (sp or 0) >= need
            print(f"  {nm:28s} {'READABLE' if ok else 'DISCARD'}  "
                  f"state={st} step={sp}/{need} loss_min={lm}", flush=True)
        sys.exit(0)
    if time.time() - t0 > MAXWAIT:
        print("\nGAVE UP after 8h.", flush=True); sys.exit(1)
    time.sleep(POLL)
