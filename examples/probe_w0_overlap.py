"""W0-cage overlap probe.

Decisive measurement for the "escape the cage" question: does the fine-tuning
gradient point INSIDE W0's top-r singular subspace (the LoRA-XS cage), or is
there substantial signal OUTSIDE it that a frozen-W0-basis method structurally
cannot reach?

Every metric we've logged so far (R, M_R entropy, recoverable rank) is computed
on the r x r core — a projection INTO the cage — so it is blind to out-of-cage
energy by construction. This probe instead forms the full-layer gradient
dL/dW (m x n) and measures the fraction of its energy captured by W0's rank-r
left/right singular subspaces:

    captured(r) = || U_r^T G V_r ||_F^2 / || G ||_F^2

across a sweep of cage-ranks r. captured(32) near 1.0 => the cage holds the
signal, escaping it buys ~nothing. captured(32) low (and not recovered by
larger r) => real signal lives outside W0, and a method that can reach it
could beat LoRA-XS where the cage currently costs us (HumanEval).

No DP here on purpose: this is a question about the loss-landscape geometry
(where the task gradient points), independent of privacy. The clean gradient
is the honest signal; DP noise would only blur it.
"""

from __future__ import annotations

import os
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-Coder-7B"
R_SWEEP = [8, 16, 32, 64, 128]
R_MAX = max(R_SWEEP)
N_MICROBATCH = 32          # gradient-accumulation steps (denoise the direction)
MB = 2                     # sequences per microbatch
SEQ = 1024
LAYER_IDS = [0, 7, 14, 21, 27]
MODULES = ["q_proj", "v_proj", "down_proj"]

device = "cuda"
print(f"PROBE: loading {MODEL} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
model.eval()
model.config.use_cache = False
model.gradient_checkpointing_enable()  # cut activation memory -> fits 80GB H100

# Freeze everything; enable grad only on the probed weights.
for p in model.parameters():
    p.requires_grad_(False)
targets: dict[str, torch.nn.Module] = {}
_pat = re.compile(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)$")
for name, mod in model.named_modules():
    m = _pat.search(name)
    if m and int(m.group(1)) in LAYER_IDS and m.group(2) in MODULES:
        mod.weight.requires_grad_(True)
        targets[name] = mod
print(f"PROBE: probing {len(targets)} layers", flush=True)

# Streaming KStack batches.
ds = load_dataset("JetBrains/KStack", split="train", streaming=True)
_it = iter(ds)


def get_batch() -> torch.Tensor:
    texts: list[str] = []
    while len(texts) < MB:
        ex = next(_it)
        t = ex.get("content") or ""
        if t.strip():
            texts.append(t)
    enc = tok(texts, return_tensors="pt", truncation=True, max_length=SEQ,
              padding="max_length")
    return enc["input_ids"].to(device)


# Accumulate the clean full-layer gradient over N_MICROBATCH batches.
print("PROBE: accumulating gradient ...", flush=True)
model.zero_grad(set_to_none=True)
for i in range(N_MICROBATCH):
    ids = get_batch()
    out = model(input_ids=ids, labels=ids)
    (out.loss / N_MICROBATCH).backward()
    if (i + 1) % 8 == 0:
        print(f"PROBE: step {i+1}/{N_MICROBATCH} loss={out.loss.item():.4f}", flush=True)


def top_subspace(M: torch.Tensor, q: int):
    U, S, V = torch.svd_lowrank(M.float(), q=min(q + 8, *M.shape))
    return U[:, :q], V[:, :q]


# Per-layer captured-energy curve + random-subspace baseline.
agg = {r: [] for r in R_SWEEP}
agg_rand = []
print("\nPROBE_HEADER layer | " + " ".join(f"capt@{r}" for r in R_SWEEP) + " | rand@32", flush=True)
for name, mod in targets.items():
    W0 = mod.weight.data.float()
    G = mod.weight.grad
    if G is None:
        print(f"PROBE_WARN {name}: no grad", flush=True)
        continue
    G = G.float()
    gnorm2 = (G.norm() ** 2).clamp(min=1e-12)
    Uw, Vw = top_subspace(W0, R_MAX)
    caps = {}
    for r in R_SWEEP:
        proj = Uw[:, :r].T @ G @ Vw[:, :r]
        caps[r] = float((proj.norm() ** 2 / gnorm2).item())
        agg[r].append(caps[r])
    # random rank-32 subspace baseline (what an uninformative cage captures)
    Ur, _ = torch.linalg.qr(torch.randn(G.shape[0], 32, device=device, dtype=torch.float32))
    Vr, _ = torch.linalg.qr(torch.randn(G.shape[1], 32, device=device, dtype=torch.float32))
    rand = float(((Ur.T @ G @ Vr).norm() ** 2 / gnorm2).item())
    agg_rand.append(rand)
    print(f"PROBE_RESULT {name} | " + " ".join(f"{caps[r]:.3f}" for r in R_SWEEP)
          + f" | {rand:.5f}", flush=True)

print("\nPROBE_SUMMARY (mean over layers):", flush=True)
for r in R_SWEEP:
    vals = agg[r]
    if vals:
        print(f"PROBE_SUMMARY captured@r={r}: mean={sum(vals)/len(vals):.3f} "
              f"min={min(vals):.3f} max={max(vals):.3f}", flush=True)
if agg_rand:
    print(f"PROBE_SUMMARY random@r=32: mean={sum(agg_rand)/len(agg_rand):.5f}", flush=True)
print("PROBE: done", flush=True)
