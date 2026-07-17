"""Compute a per-layer LoRA-XS rank allocation (Rényi / stable-rank) → JSON.

Turns per-layer singular-value spectra into a ``rank_pattern`` JSON
({module: r_l}) consumable by
``train_causal_lm.py --lora-xs-rank-pattern-json``. The total budget
``sum_l r_l^2`` is held fixed to a uniform-``r`` run, so the resulting model has
the SAME parameter count (hence the SAME DP noise) as uniform LoRA-XS — the only
difference is *where* the rank goes.

Sources of the spectra:
  --from-spectra probe.json   # {module: [sigmas]} produced by a short probe run
                              #   (train_causal_lm.py --max-steps N --dump-core-spectra probe.json)
                              #   THIS is the DP-faithful source: it scores the
                              #   already-noised trained cores, so alpha matters.
  --from-model  <hf_name>     # SVD each target W0 to --probe-r sigmas (data-free,
                              #   build-time). alpha barely matters here (W0 is
                              #   clean) — a variable-rank method, NOT the DP thesis.

Scoring:
  --score-mode renyi --alpha inf   # stable rank (OURS; robust in the DP regime)
  --score-mode renyi --alpha 1     # Shannon (noise-naive baseline)
  --score-mode frob                # total-energy importance (AdaLoRA-style proxy)
  --score-mode uniform             # control: reproduces uniform-r allocation

Example:
  # 1) probe (short DP warm-up, uniform r=16), dump spectra:
  uv run python examples/train_causal_lm.py --preset qwen-coder-kstack-lora \
      --lora-method lora-xs --lora-xse-p-e 0.333 --target-epsilon 3 \
      --max-steps 50 --dump-core-spectra probe_eps3.json
  # 2) allocate by stable rank at the same budget (uniform r=16):
  uv run python examples/compute_rank_allocation.py --from-spectra probe_eps3.json \
      --score-mode renyi --alpha inf --uniform-r 16 --out alloc_ainf.json
  # 3) train with the allocation:
  uv run python examples/train_causal_lm.py --preset qwen-coder-kstack-lora \
      --lora-method lora-xs --lora-xse-p-e 0.333 --target-epsilon 3 \
      --lora-xs-rank-pattern-json alloc_ainf.json
"""

from __future__ import annotations

import argparse
import json
import math

from lora_privacy.peft_lora_xs.allocation import allocate_ranks, score_layer


def _alpha(x: str) -> float:
    return float("inf") if x.lower() in ("inf", "infinity") else float(x)


def spectra_from_model(model_name: str, target_suffixes: list[str], probe_r: int) -> dict:
    """Data-free source: SVD each target W0 to its top singular values."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    spectra: dict[str, list[float]] = {}
    for name, mod in model.named_modules():
        if not hasattr(mod, "weight") or mod.weight is None or mod.weight.ndim != 2:
            continue
        if target_suffixes and not any(name.endswith(s) for s in target_suffixes):
            continue
        q = min(probe_r + 8, *mod.weight.shape)
        sv = torch.linalg.svdvals(mod.weight.detach().float())[:q]
        spectra[name] = sv.tolist()
    return spectra


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-spectra", type=str, help="JSON {module: [sigmas]} (probe dump)")
    src.add_argument("--from-model", type=str, help="HF model name (SVD W0, data-free)")
    ap.add_argument("--target-suffixes", nargs="*",
                    default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    help="(--from-model) module-name suffixes to target")
    ap.add_argument("--probe-r", type=int, default=16, help="(--from-model) top singular values to take")
    ap.add_argument("--score-mode", choices=["renyi", "frob", "uniform"], default="renyi")
    ap.add_argument("--alpha", type=_alpha, default=float("inf"), help="Rényi order (inf = stable rank)")
    ap.add_argument("--uniform-r", type=int, default=16,
                    help="budget = n_layers * uniform_r^2 (matches a uniform-r run)")
    ap.add_argument("--budget", type=int, default=None, help="override total sum r_l^2")
    ap.add_argument("--r-min", type=int, default=2)
    ap.add_argument("--r-max", type=int, default=None)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    if args.from_spectra:
        with open(args.from_spectra) as f:
            spectra = json.load(f)
    else:
        spectra = spectra_from_model(args.from_model, args.target_suffixes, args.probe_r)
    if not spectra:
        raise SystemExit("no spectra found (check target suffixes / probe dump)")

    budget = args.budget if args.budget is not None else len(spectra) * args.uniform_r ** 2
    scores = {k: score_layer(v, mode=args.score_mode, alpha=args.alpha) for k, v in spectra.items()}
    alloc = allocate_ranks(scores, budget, r_min=args.r_min, r_max=args.r_max)

    achieved = sum(r * r for r in alloc.values())
    ranks = sorted(alloc.values())
    print(f"layers={len(alloc)}  score-mode={args.score_mode} alpha={args.alpha}")
    print(f"budget (sum r^2) target={budget} achieved={achieved} "
          f"({100*achieved/budget:.1f}%)  vs uniform r={args.uniform_r}")
    print(f"rank spread: min={ranks[0]} median={ranks[len(ranks)//2]} max={ranks[-1]}")
    with open(args.out, "w") as f:
        json.dump(alloc, f, indent=0)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
