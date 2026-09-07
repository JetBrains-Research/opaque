"""E6 (unpatched, 1-layer): empirical per-example load-vector norms vs the analytic bound
||f(x)||_2 <= sqrt(k) (entries in [0,1], sum = k) -> Gaussian-release noise scale for f(B)."""
import sys, copy, math; sys.path.insert(0, "/tmp/claude-0/-home-user-opaque/a97cca5e-900f-5617-b2df-d73b55081cac/scratchpad/research/empirical")
from common import *
import torch
B = 32; res = {}
for (E, k) in [(8, 2), (64, 8)]:
    m = build_unpatched(seed=0, num_hidden_layers=1, num_experts=E, num_experts_per_tok=k, max_position_embeddings=1024); m.eval()
    for T in [64, 512]:
        for kind in ["random", "structured"]:
            ids = random_batch(B, T, seed=5) if kind == "random" else structured_batch(B, T, seed=5)
            cap = RouterCapture(m)
            with torch.no_grad(): m(input_ids=ids, attention_mask=torch.ones_like(ids))
            lg = cap.records[0][0].float().reshape(B, T, E); cap.remove()
            sel = topk_sets(lg.reshape(-1, E), k).reshape(B, T, k)
            fx = (sel[..., None] == torch.arange(E)).float().sum(2).mean(1)      # (B, E), sum_e = k, entries in [0,1]
            fB = fx.mean(0)
            r = {"max_l2_fx": fx.norm(dim=1).max().item(), "bound_sqrt_k": math.sqrt(k), "max_l2_fx_minus_fB": (fx - fB).norm(dim=1).max().item(),
                 "l2_fB_minus_uniform_over_kE": ((fB - k / E).norm() / (k / E)).item(),
                 # Gaussian release of the mean load vector f(B) = (1/B) sum_x f(x) with sensitivity sqrt(k)/B (add/remove), noise multiplier 1:
                 "noise_std_per_coord_over_kE_at_B128_sigma1": (math.sqrt(k) / 128) / (k / E), "noise_std_per_coord_over_kE_at_B1024_sigma1": (math.sqrt(k) / 1024) / (k / E)}
            res[f"E{E}k{k}_T{T}_{kind}"] = r
            print(f"E{E}k{k} T{T:>3} {kind:>10}: max||f(x)||2 {r['max_l2_fx']:.3f} (bound {r['bound_sqrt_k']:.3f}) max||f(x)-f(B)||2 {r['max_l2_fx_minus_fB']:.3f} | ||f(B)-k/E||2/(k/E) {r['l2_fB_minus_uniform_over_kE']:.3f} | release noise/coord in k/E units: B128 {r['noise_std_per_coord_over_kE_at_B128_sigma1']:.3f}, B1024 {r['noise_std_per_coord_over_kE_at_B1024_sigma1']:.3f}")
dump("e6_results.json", res)
