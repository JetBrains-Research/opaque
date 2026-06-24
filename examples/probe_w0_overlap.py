"""W0-cage overlap probe (wandb-instrumented, robust).

Forms the clean full-layer gradient dL/dW and measures the fraction of its
energy captured by W0's rank-r left/right singular subspaces, swept over
r in {8,16,32,64,128}, vs a random-subspace baseline.

captured(r) = || U_r^T G V_r ||_F^2 / || G ||_F^2

Every prior metric (R/M_R entropy, recoverable rank) is a projection INTO the
LoRA-XS cage and is blind to out-of-cage energy by construction; this looks
outside it. No DP: this is loss-landscape geometry, not a private release.

Results + any error traceback go to wandb (cadence stdout is buried under
rclone sync noise, so we don't rely on it).
"""

from __future__ import annotations

import os
import re
import traceback

import torch

MODEL = "Qwen/Qwen2.5-Coder-7B"
R_SWEEP = [8, 16, 32, 64, 128]
R_MAX = max(R_SWEEP)
N_MICROBATCH = 64
SEQ = 1024
LAYER_IDS = [0, 7, 14, 21, 27]
MODULES = ["q_proj", "v_proj", "down_proj"]
TEXT_FIELDS = ["content", "text", "code", "data"]

import wandb

run = wandb.init(
    project=os.environ.get("WANDB_PROJECT", "opaque-lora-xs"),
    entity=os.environ.get("WANDB_ENTITY", "federated-compute"),
    name=os.environ.get("RUN_NAME", "probe-w0-overlap"),
)


def log_stage(msg: str):
    print(f"PROBE: {msg}", flush=True)
    try:
        run.log({"probe/stage": msg})
    except Exception:
        pass


def top_subspace(M: torch.Tensor, q: int):
    q = min(q + 8, M.shape[0], M.shape[1])
    U, S, V = torch.svd_lowrank(M.float(), q=q)
    return U, V


try:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    log_stage(f"loading {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    model.eval()
    model.config.use_cache = False
    log_stage("model loaded")

    for p in model.parameters():
        p.requires_grad_(False)
    targets: dict[str, torch.nn.Module] = {}
    _pat = re.compile(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)$")
    for name, mod in model.named_modules():
        m = _pat.search(name)
        if m and int(m.group(1)) in LAYER_IDS and m.group(2) in MODULES:
            mod.weight.requires_grad_(True)
            targets[name] = mod
    log_stage(f"probing {len(targets)} layers")

    ds = load_dataset("JetBrains/KStack", split="train", streaming=True)
    _it = iter(ds)
    # detect the text field once
    _first = next(_it)
    field = next((f for f in TEXT_FIELDS if isinstance(_first.get(f), str) and _first[f].strip()), None)
    if field is None:
        field = next((k for k, v in _first.items() if isinstance(v, str) and len(v) > 20), None)
    log_stage(f"text field = {field}; example keys = {list(_first.keys())[:8]}")

    def texts_iter():
        yield _first[field]
        for ex in _it:
            t = ex.get(field) or ""
            if t.strip():
                yield t

    gen = texts_iter()

    def get_ids():
        t = next(gen)
        enc = tok(t, return_tensors="pt", truncation=True, max_length=SEQ)
        return enc["input_ids"].to(device)

    log_stage("accumulating gradient")
    model.zero_grad(set_to_none=True)
    done = 0
    for i in range(N_MICROBATCH):
        ids = get_ids()
        if ids.shape[1] < 8:
            continue
        out = model(input_ids=ids, labels=ids)
        (out.loss / N_MICROBATCH).backward()
        done += 1
        if (i + 1) % 16 == 0:
            log_stage(f"step {i+1}/{N_MICROBATCH} loss={out.loss.item():.4f}")
    log_stage(f"accumulated over {done} sequences; analyzing")

    rows = []
    agg = {r: [] for r in R_SWEEP}
    agg_rand = []
    for name, mod in targets.items():
        G = mod.weight.grad
        if G is None:
            print(f"PROBE_WARN {name}: no grad", flush=True)
            continue
        G = G.float()
        gnorm2 = (G.norm() ** 2).clamp(min=1e-12)
        Uw, Vw = top_subspace(mod.weight.data, R_MAX)
        caps = {}
        for r in R_SWEEP:
            proj = Uw[:, :r].T @ G @ Vw[:, :r]
            caps[r] = float((proj.norm() ** 2 / gnorm2).item())
            agg[r].append(caps[r])
        Ur, _ = torch.linalg.qr(torch.randn(G.shape[0], 32, device=device))
        Vr, _ = torch.linalg.qr(torch.randn(G.shape[1], 32, device=device))
        rand = float(((Ur.T @ G @ Vr).norm() ** 2 / gnorm2).item())
        agg_rand.append(rand)
        rows.append([name] + [caps[r] for r in R_SWEEP] + [rand])
        print(f"PROBE_RESULT {name} | " + " ".join(f"{caps[r]:.3f}" for r in R_SWEEP) + f" | rand={rand:.5f}", flush=True)

    tbl = wandb.Table(columns=["layer"] + [f"capt@{r}" for r in R_SWEEP] + ["rand@32"], data=rows)
    summary = {f"probe/captured_mean_r{r}": (sum(agg[r]) / len(agg[r])) for r in R_SWEEP if agg[r]}
    summary["probe/captured_min_r32"] = min(agg[32]) if agg[32] else None
    summary["probe/captured_max_r32"] = max(agg[32]) if agg[32] else None
    summary["probe/random_mean_r32"] = (sum(agg_rand) / len(agg_rand)) if agg_rand else None
    summary["probe/n_layers"] = len(rows)
    run.log({"probe/table": tbl, **summary})
    for k, v in summary.items():
        run.summary[k] = v
        print(f"PROBE_SUMMARY {k} = {v}", flush=True)
    print("PROBE: done", flush=True)
    run.finish()

except Exception as e:
    tb = traceback.format_exc()
    print("PROBE_ERROR\n" + tb, flush=True)
    try:
        run.summary["probe/error"] = str(e)
        run.summary["probe/traceback"] = tb[-4000:]
        run.finish(exit_code=1)
    except Exception:
        pass
    raise
