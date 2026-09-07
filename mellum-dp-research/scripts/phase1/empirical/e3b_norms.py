"""E3 (patched process): per-example gradient norm distributions via clipped_grad(return_aux)."""
import sys, time, copy; sys.path.insert(0, "/tmp/claude-0/-home-user-opaque/a97cca5e-900f-5617-b2df-d73b55081cac/scratchpad/research/empirical")
from common import *
import torch
from opaque.functional import make_functional
from opaque.dpsgd.clipping import clipped_grad
import transformers.models.mellum.modeling_mellum as mod
mod.load_balancing_loss_func = aux_loss_vmap_safe   # per-example (own f(x), P(x)) aux, vmap-safe, mask-aware
B, T = 32, 64
E, k = TINY["num_experts"], TINY["num_experts_per_tok"]
model, _ = build_patched(seed=0, initializer_range=0.125)
res = {}
t0 = time.time()
batches = {"random": random_batch(B, T, seed=8), "structured": structured_batch(B, T, seed=8)}
for partition in ["attn", "attn+router", "attn+experts"]:
    set_trainable(model, partition)
    model.train()
    fmodel, trainable, frozen = make_functional(model, disable_autograd_tracking=True, partition_trainable=True)
    n_tr = sum(p.numel() for p in trainable.values())
    for loss_kind in ["ce", "ce+0.001aux", "aux_only"]:
        coef = {"ce": 0.0, "ce+0.001aux": 0.001, "aux_only": 1.0}[loss_kind]
        def per_ex(tr, fr, i, mk, lb, coef=coef, loss_kind=loss_kind):
            o = fmodel({**fr, **tr}, i, attention_mask=mk, labels=lb, output_router_logits=(coef > 0))
            if loss_kind == "aux_only":
                return o.aux_loss   # per-example Switch aux on the example's own tokens (pooled over layers)
            return o.loss           # HF: loss = CE + router_aux_loss_coef * aux (per example under vmap)
        model.router_aux_loss_coef = coef if coef < 1 else 1.0
        gfn, st = clipped_grad(per_ex, argnums=0, batch_argnums=(2, 3, 4), clipping_norm=1e9, return_aux=True)
        for kind, ids in batches.items():
            mask = torch.ones_like(ids)
            (g, aux), _ = gfn(trainable, frozen, ids, mask, ids, state=st)
            n = aux.grad_norms.float()
            key = f"{partition}|{loss_kind}|{kind}"
            res[key] = {"n_trainable": n_tr, "median": n.median().item(), "p95": n.quantile(0.95).item(), "max": n.max().item(), "min": n.min().item(),
                        "ratio_max_median": (n.max() / n.median()).item(), "ratio_p95_median": (n.quantile(0.95) / n.median()).item(), "loss_mean": aux.loss_values.float().mean().item(), "norms": n.tolist()}
            print(f"{key:>40}: n_tr {n_tr:>7d} median {n.median():.4f} p95 {n.quantile(0.95):.4f} max {n.max():.4f} min {n.min():.4f} max/med {n.max()/n.median():.2f} loss {aux.loss_values.float().mean():.4f}  [{time.time()-t0:.0f}s]")
dump("e3b_init0.125_results.json", res)
print("done %.1fs" % (time.time() - t0))
