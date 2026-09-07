"""E4 (unpatched process): aux-loss gradient representations on one batch."""
import sys, time, copy; sys.path.insert(0, "/tmp/claude-0/-home-user-opaque/a97cca5e-900f-5617-b2df-d73b55081cac/scratchpad/research/empirical")
from common import *
import torch
import transformers.models.mellum.modeling_mellum as mod
assert "opaque" not in sys.modules
B, T = 16, 32
E, k = TINY["num_experts"], TINY["num_experts_per_tok"]
res = {}
t0 = time.time()
model = build_unpatched(seed=0); model.train()
params = [p for p in model.parameters()]
names = [n for n, _ in model.named_parameters()]

def flatgrad(loss):
    gs = torch.autograd.grad(loss, params, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, params)])

def router_logits(ids, mask):
    cap = RouterCapture(model, detach=False)
    model(input_ids=ids, attention_mask=mask)
    lg = [r[0] for r in cap.records]; cap.remove()
    return lg  # list over layers of (rows, E) with grad graph

for kind in ["random", "structured", "random_padded"]:
    ids = random_batch(B, T, seed=3) if kind.startswith("random") else structured_batch(B, T, seed=3)
    mask = torch.ones_like(ids)
    if kind == "random_padded":
        # ragged lengths: example b keeps T - 2*b tokens (right padding)
        for b in range(B): mask[b, T - 2 * b:] = 0
    # (i) HF batch aux loss gradient (coef 1), exactly HF's function, batched
    out = model(input_ids=ids, attention_mask=mask, output_router_logits=True)
    aux_hf = mod.load_balancing_loss_func(out.router_logits, E, k, mask)
    aux_mine = aux_loss_vmap_safe(out.router_logits, E, k, mask)
    assert abs(aux_hf.item() - aux_mine.item()) < 1e-5, (aux_hf, aux_mine)
    g_i = flatgrad(aux_hf)
    fB, PB, Ttot = load_stats_from_logits([l.detach() for l in out.router_logits], E, k, mask)
    # (ii) mean of per-example aux gradients (own f(x), own P(x))
    g_ii = torch.zeros_like(g_i); per_ex_aux = []; per_ex_f = []
    Tx = mask.sum(1).float()
    for b in range(B):
        lg = router_logits(ids[b:b+1], mask[b:b+1])
        a_x = aux_loss_vmap_safe(tuple(lg), E, k, mask[b:b+1])
        per_ex_aux.append(a_x.item()); per_ex_f.append(load_stats_from_logits([l.detach() for l in lg], E, k, mask[b:b+1])[0])
        g_ii += flatgrad(a_x) / B
    # (iii) surrogate with f~ = f(B) constant:  l_x = E * sum_e f~_e * Pbar_e(x) * T_x / T_tot
    def surrogate_grad(f_tilde):
        g = torch.zeros_like(g_i)
        for b in range(B):
            lg = router_logits(ids[b:b+1], mask[b:b+1])
            _, Px, _ = load_stats_from_logits(lg, E, k, mask[b:b+1])   # differentiable P(x)
            l_x = E * torch.sum(f_tilde * Px) * (Tx[b] / Tx.sum())
            g += flatgrad(l_x)
        return g
    g_iii = surrogate_grad(fB)
    r = {"aux_hf": aux_hf.item(), "mean_per_example_aux": float(sum(per_ex_aux) / B), "fB": fB.tolist(),
         "per_example_f_mean_abs_dev_over_kE": (torch.stack(per_ex_f) - fB).abs().mean().item() / (k / E),
         "norm_i": g_i.norm().item(), "err_ii_vs_i": rel_l2(g_ii, g_i), "err_iii_vs_i": rel_l2(g_iii, g_i),
         "cos_ii_vs_i": torch.nn.functional.cosine_similarity(g_ii, g_i, dim=0).item()}
    # (iv) f~ perturbed by Gaussian noise, std = rel * (k/E)  (relative to the mean load fraction)
    for rel in [0.05, 0.1, 0.3, 1.0]:
        errs = []
        for s in range(5):
            gn = torch.Generator().manual_seed(100 + s)
            f_t = fB + rel * (k / E) * torch.randn(E, generator=gn)
            errs.append(rel_l2(surrogate_grad(f_t), g_i))
        r[f"err_noisy_rel{rel}"] = {"mean": float(sum(errs) / len(errs)), "max": max(errs)}
    # lagged f~: load vector of a different batch (same distribution)
    ids2 = random_batch(B, T, seed=4) if kind.startswith("random") else structured_batch(B, T, seed=4)
    with torch.no_grad():
        lg2 = router_logits(ids2, torch.ones_like(ids2)); f_lag, _, _ = load_stats_from_logits(lg2, E, k, None)
    r["err_lagged_f"] = rel_l2(surrogate_grad(f_lag), g_i); r["lagged_f_abs_dev_over_kE"] = (f_lag - fB).abs().mean().item() / (k / E)
    # uniform f~ = k/E (no batch information at all): surrogate becomes E*(k/E)*sum_e P_e = k -> constant -> zero grad
    r["err_uniform_f"] = rel_l2(surrogate_grad(torch.full((E,), k / E)), g_i)
    r["norm_uniform_f_grad"] = surrogate_grad(torch.full((E,), k / E)).norm().item()
    res[kind] = r
    print(f"[{kind}] aux_hf {r['aux_hf']:.4f} mean per-ex aux {r['mean_per_example_aux']:.4f} | ||i|| {r['norm_i']:.3e} | err(ii) {r['err_ii_vs_i']:.3e} cos {r['cos_ii_vs_i']:.3f} | err(iii) {r['err_iii_vs_i']:.3e} | noisy: " + ", ".join(f"rel{x}:{r[f'err_noisy_rel{x}']['mean']:.3f}" for x in [0.05,0.1,0.3,1.0]) + f" | lagged {r['err_lagged_f']:.3f} (f dev {r['lagged_f_abs_dev_over_kE']:.3f} k/E) | uniform {r['err_uniform_f']:.3f} (norm {r['norm_uniform_f_grad']:.1e})")
dump("e4_results.json", res)
print("done %.1fs" % (time.time() - t0))
