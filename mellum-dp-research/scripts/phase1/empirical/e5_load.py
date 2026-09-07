"""E5 (unpatched process, 1-layer model): per-example load fractions f(x) vs batch f(B)."""
import sys, time, copy; sys.path.insert(0, "/tmp/claude-0/-home-user-opaque/a97cca5e-900f-5617-b2df-d73b55081cac/scratchpad/research/empirical")
from common import *
import torch
assert "opaque" not in sys.modules
B = 32
res = {}
t0 = time.time()
for (E, k) in [(8, 2), (64, 8)]:
    base = build_unpatched(seed=0, num_hidden_layers=1, num_experts=E, num_experts_per_tok=k, max_position_embeddings=1024); base.eval()
    for router_scale in [1.0, 5.0]:
        m = copy.deepcopy(base)
        with torch.no_grad():
            for n, p in m.named_parameters():
                if ".mlp.gate." in n: p.mul_(router_scale)
        for T in [64, 512]:
            for kind in ["random", "structured"]:
                ids = random_batch(B, T, seed=5) if kind == "random" else structured_batch(B, T, seed=5)
                cap = RouterCapture(m)
                with torch.no_grad(): m(input_ids=ids, attention_mask=torch.ones_like(ids))
                lg = cap.records[0][0].float().reshape(B, T, E); cap.remove()
                sel = topk_sets(lg.reshape(-1, E), k).reshape(B, T, k)
                oh = torch.nn.functional.one_hot(sel, E).float().sum(2)          # (B, T, E)
                fx = oh.mean(1)                                                     # per-example load fractions (B, E)
                fB = oh.reshape(-1, E).mean(0)
                dev = (fx - fB).abs().mean().item() / (k / E)
                # iid-null: permute tokens across examples -> sampling-noise floor for the same batch f(B)
                g = torch.Generator().manual_seed(7)
                devs_null = []
                for _ in range(5):
                    perm = torch.randperm(B * T, generator=g)
                    fx_null = oh.reshape(-1, E)[perm].reshape(B, T, E).mean(1)
                    devs_null.append((fx_null - fB).abs().mean().item() / (k / E))
                dev_null = sum(devs_null) / len(devs_null)
                # batch-to-batch variation of f(B) itself (what a lagged estimate would see)
                ids2 = random_batch(B, T, seed=6) if kind == "random" else structured_batch(B, T, seed=6)
                cap = RouterCapture(m)
                with torch.no_grad(): m(input_ids=ids2, attention_mask=torch.ones_like(ids2))
                lg2 = cap.records[0][0].float().reshape(B, T, E); cap.remove()
                fB2 = torch.nn.functional.one_hot(topk_sets(lg2.reshape(-1, E), k), E).float().sum(1).mean(0)
                dev_batch = (fB2 - fB).abs().mean().item() / (k / E)
                # max per-example deviation and fraction of experts unused by an example
                frac_unused = (fx == 0).float().mean().item()
                key = f"E{E}k{k}_scale{router_scale}_T{T}_{kind}"
                res[key] = {"mean_abs_dev_over_kE": dev, "iid_null_dev_over_kE": dev_null, "ratio_dev_to_null": dev / dev_null,
                            "max_abs_dev_over_kE": (fx - fB).abs().max().item() / (k / E), "batch_to_batch_dev_over_kE": dev_batch,
                            "frac_expert_unused_per_example": frac_unused, "fB_max_over_kE": (fB.max() / (k / E)).item(), "fB_min_over_kE": (fB.min() / (k / E)).item()}
                print(f"{key:>36}: dev {dev:.3f} k/E (iid-null {dev_null:.3f}, ratio {dev/dev_null:.2f}) max {res[key]['max_abs_dev_over_kE']:.2f} | batch-to-batch {dev_batch:.3f} | unused/ex {frac_unused:.3f} | fB range [{res[key]['fB_min_over_kE']:.2f},{res[key]['fB_max_over_kE']:.2f}]")
dump("e5_results.json", res)
print("done %.1fs" % (time.time() - t0))
