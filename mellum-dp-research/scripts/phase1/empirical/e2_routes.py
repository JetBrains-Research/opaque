"""E2 (unpatched process): bf16 vs fp32 forward route sensitivity + margin distribution."""
import sys, time, copy; sys.path.insert(0, "/tmp/claude-0/-home-user-opaque/a97cca5e-900f-5617-b2df-d73b55081cac/scratchpad/research/empirical")
from common import *
import torch
assert "opaque" not in sys.modules
B, T, k = 32, 64, 2
res = {}
t0 = time.time()
base = build_unpatched(seed=0); base.eval()
for router_scale in [1.0, 5.0, 20.0]:
    m32 = copy.deepcopy(base)
    with torch.no_grad():
        for n, p in m32.named_parameters():
            if ".mlp.gate." in n: p.mul_(router_scale)
    mbf = copy.deepcopy(m32).to(torch.bfloat16)
    for kind, ids in [("random", random_batch(B, T, seed=2)), ("structured", structured_batch(B, T, seed=2))]:
        mask = torch.ones_like(ids)
        caps = {}
        for tag, m in [("fp32", m32), ("bf16", mbf)]:
            cap = RouterCapture(m)
            with torch.no_grad(): m(input_ids=ids, attention_mask=mask)
            caps[tag] = [r[0].float() for r in cap.records]; cap.remove()
        per_layer = []
        for l in range(len(caps["fp32"])):
            a, b = caps["bf16"][l], caps["fp32"][l]
            frac, rows = route_flip_fraction(a, b, k)
            mg = margins(b, k)
            # margin measured in *logit* space too (bf16 logits have ~3 significant digits)
            lg_sorted = torch.sort(b, -1, descending=True).values; lm = lg_sorted[:, k-1] - lg_sorted[:, k]
            per_layer.append({"flip_frac": frac, "n_rows": int(a.shape[0]),
                "prob_margin": {"median": mg.median().item(), "p10": mg.quantile(0.1).item(), "frac_lt_1e-3": (mg < 1e-3).float().mean().item(), "frac_lt_1e-2": (mg < 1e-2).float().mean().item()},
                "logit_margin": {"median": lm.median().item(), "frac_lt_1e-2": (lm < 1e-2).float().mean().item(), "frac_lt_1e-1": (lm < 1e-1).float().mean().item()},
                "flipped_rows_margin_max": (mg[rows].max().item() if rows.any() else None),
                "flipped_rows_margin_median": (mg[rows].median().item() if rows.any() else None),
                "top1_prob_median": torch.softmax(b, -1).max(-1).values.median().item(),
                "logit_abs_median": b.abs().median().item()})
        res[f"scale{router_scale}_{kind}"] = per_layer
        print(f"router x{router_scale:>4} {kind:>10}: " + " | ".join(f"L{l} flip {d['flip_frac']:.3f} pmargin med {d['prob_margin']['median']:.2e} <1e-3:{d['prob_margin']['frac_lt_1e-3']:.3f} <1e-2:{d['prob_margin']['frac_lt_1e-2']:.3f} top1 {d['top1_prob_median']:.3f}" for l, d in enumerate(per_layer)))
        if router_scale == 1.0:
            torch.save({"ids": ids, "logits_fp32": caps["fp32"], "logits_bf16": caps["bf16"]}, OUT / f"e2_logits_{kind}.pt")
dump("e2_results.json", res)
print("done %.1fs" % (time.time() - t0))
