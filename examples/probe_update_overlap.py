"""Corrected cage probe: where does a FREE-basis LoRA's learned update land
relative to W0's top-r subspace (the LoRA-XS cage)?

Trains a short rank-32 LoRA (free basis, clean / non-DP) on KStack, then for
each target layer forms the learned update dW* = scaling * B @ A and measures

    captured(r) = || U_r^T dW* V_r ||_F^2 / || dW* ||_F^2

where U_r, V_r are W0's top-r singular subspaces, swept r in {8,16,32,64,128},
vs a random-subspace baseline.

This is the RIGHT object (the trained update, not the instantaneous gradient).
captured@32 ~1.0 => free-LoRA's solution lives inside the r=32 cage, so
escaping W0 buys nothing (the HumanEval gap is something else). captured@32
low and flat across r => free-LoRA uses directions outside W0 that LoRA-XS
structurally cannot reach -> the "undefeatable directions" are real.

Clean (non-DP) on purpose: this shows the task's preferred update geometry —
the most generous case for finding out-of-cage signal. If it's in-cage here,
it's in-cage under DP too. Results + traceback go to wandb.
"""

from __future__ import annotations

import os
import re
import traceback

import torch

MODEL = "Qwen/Qwen2.5-Coder-7B"
R_LORA = 32
R_SWEEP = [8, 16, 32, 64, 128]
R_MAX = max(R_SWEEP)
N_STEPS = 200
MB = 4
SEQ = 1024
LR = 2e-4
LAYER_IDS = [0, 7, 14, 21, 27]
MODULES = ["q_proj", "v_proj", "down_proj"]
TEXT_FIELDS = ["content", "text", "code", "data"]

import wandb

run = wandb.init(
    project=os.environ.get("WANDB_PROJECT", "opaque-lora-xs"),
    entity=os.environ.get("WANDB_ENTITY", "federated-compute"),
    name=os.environ.get("RUN_NAME", "probe-update-overlap"),
)


def log_stage(msg: str):
    print(f"PROBE: {msg}", flush=True)
    try:
        run.log({"probe/stage_msg": msg})
    except Exception:
        pass


def top_subspace(M: torch.Tensor, q: int):
    q = min(q + 8, M.shape[0], M.shape[1])
    U, S, V = torch.svd_lowrank(M.float(), q=q)
    return U, V


try:
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    log_stage(f"loading {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    model.config.use_cache = False

    lcfg = LoraConfig(r=R_LORA, lora_alpha=R_LORA, lora_dropout=0.0,
                      target_modules=MODULES, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.enable_input_require_grads()  # makes checkpointing work w/ frozen base
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    log_stage("LoRA attached; training")

    ds = load_dataset("JetBrains/KStack", split="train", streaming=True)
    _it = iter(ds)
    _first = next(_it)
    field = next((f for f in TEXT_FIELDS if isinstance(_first.get(f), str) and _first[f].strip()), None)
    if field is None:
        field = next((k for k, v in _first.items() if isinstance(v, str) and len(v) > 20), None)
    log_stage(f"text field = {field}")

    def texts():
        yield _first[field]
        for ex in _it:
            t = ex.get(field) or ""
            if t.strip():
                yield t

    gen = texts()

    def get_batch():
        ts = []
        while len(ts) < MB:
            ts.append(next(gen))
        enc = tok(ts, return_tensors="pt", truncation=True, max_length=SEQ, padding="max_length")
        return enc["input_ids"].to(device)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    for step in range(N_STEPS):
        ids = get_batch()
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        run.log({"train/loss": float(loss.item())})
        if (step + 1) % 25 == 0:
            log_stage(f"step {step+1}/{N_STEPS} loss={loss.item():.4f}")

    log_stage("analyzing dW* vs W0 cage")
    model.eval()
    rows = []
    agg = {r: [] for r in R_SWEEP}
    agg_rand = []
    _pat = re.compile(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)$")
    for name, mod in model.named_modules():
        if not hasattr(mod, "lora_A"):
            continue
        m = _pat.search(name)
        if not (m and int(m.group(1)) in LAYER_IDS and m.group(2) in MODULES):
            continue
        A = mod.lora_A["default"].weight.data.float()   # (r, in)
        B = mod.lora_B["default"].weight.data.float()   # (out, r)
        scaling = float(mod.scaling["default"])
        dW = scaling * (B @ A)                           # (out, in)
        W0 = mod.base_layer.weight.data.float()          # (out, in)
        dn2 = (dW.norm() ** 2).clamp(min=1e-12)
        Uw, Vw = top_subspace(W0, R_MAX)
        caps = {}
        for r in R_SWEEP:
            proj = Uw[:, :r].T @ dW @ Vw[:, :r]
            caps[r] = float((proj.norm() ** 2 / dn2).item())
            agg[r].append(caps[r])
        Ur, _ = torch.linalg.qr(torch.randn(dW.shape[0], 32, device=device))
        Vr, _ = torch.linalg.qr(torch.randn(dW.shape[1], 32, device=device))
        rand = float(((Ur.T @ dW @ Vr).norm() ** 2 / dn2).item())
        agg_rand.append(rand)
        rows.append([name] + [caps[r] for r in R_SWEEP] + [rand])
        print(f"PROBE_RESULT {name} | " + " ".join(f"{caps[r]:.3f}" for r in R_SWEEP) + f" | rand={rand:.4f}", flush=True)

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
