"""Paired per-example BPB test for the batch-1 reference frame.

Why paired: the three arms are scored on the SAME 512 eval examples, so the
between-example variance (which dominates the ~0.0073 unpaired SEM and is larger
than the effect) cancels. Pairing tightens the standard error 17-32x and is the
only reason a 3.2e-3 BPB difference is resolvable at all.

Scope: this rejects eval-sampling noise at n=1 RUN per arm. It does NOT address
seed variance. Reproduces the table in docs/batch1-reference-frame-RESULTS.md.

    WANDB_BASE_URL=https://jetbrains.wandb.io uv run --with numpy python \
        campaign_logs/batch1/paired_bpb.py
"""
import json
import os

os.environ.setdefault("WANDB_BASE_URL", "https://jetbrains.wandb.io")
import numpy as np
import wandb

PROJECT = "federated-compute/opaque-lora-xs"
ARMS = {
    "lora": ("ref-lora-r16-mb8-s42", "full LoRA r=16 (40.4M)"),
    "frozen": ("ref-xs-norot-s42", "frozen LoRA-XS (p_e=0)"),
    "xse": ("ref-xse-d5t1-s42", "LoRA-XSe d5 tau=1 (200,704)"),
}
EXPECTED_STEPS = 520


def load() -> dict[str, np.ndarray]:
    # A fresh Api(); reusing one caches run objects and has reported stale state
    # for hours in this campaign.
    api = wandb.Api(timeout=90)
    out = {}
    for key, (name, _) in ARMS.items():
        runs = list(api.runs(PROJECT, filters={"display_name": name}))
        if len(runs) != 1:
            raise SystemExit(f"{name}: expected 1 run, found {len(runs)}")
        r = runs[0]
        step = r.summary.get("_step")
        if r.state != "finished" or not isinstance(step, (int, float)) or step < EXPECTED_STEPS:
            raise SystemExit(
                f"{name}: NOT READABLE (state={r.state} step={step}). Reading a "
                f"partial run as complete caused a published retraction here once."
            )
        v = r.summary["eval/bpb_per_example_json"]
        out[key] = np.asarray(json.loads(v) if isinstance(v, str) else v, dtype=float)
    n = {len(v) for v in out.values()}
    if len(n) != 1:
        raise SystemExit(f"per-example arrays differ in length: {n}; not paired")
    return out


def paired(b, a, bkey, seed=0, boots=20000):
    d = b[a[0]] - b[a[1]]           # negative => first arm better
    n = len(d)
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    rng = np.random.default_rng(seed)
    boot = rng.choice(d, (boots, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot >= 0).mean(), (boot <= 0).mean())
    print(f"{ARMS[a[0]][1]} vs {ARMS[a[1]][1]}")
    print(f"  mean paired diff : {m:+.6f}  (negative = first better)")
    print(f"  95% CI           : [{lo:+.6f}, {hi:+.6f}]")
    print(f"  paired t         : {m/se:+.2f}   bootstrap p = {p:.2e}")
    print(f"  examples won     : {int((d < 0).sum())}/{n}")
    print(f"  paired SE {se:.6f} vs ~0.0073 unpaired ({0.0073/se:.0f}x tighter)\n")


if __name__ == "__main__":
    b = load()
    print(f"paired per-example BPB, n={len(b['lora'])} identical eval examples\n")
    for pair in (("xse", "lora"), ("xse", "frozen"), ("lora", "frozen")):
        paired(b, pair, None)
