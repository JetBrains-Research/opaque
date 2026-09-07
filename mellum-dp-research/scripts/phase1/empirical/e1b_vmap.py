"""E1b patched process: bf16 vmap (opaque) vs bf16 loop (HF) on B=32,T=64; flips between the two bf16 impls."""
import sys, time, copy; sys.path.insert(0, "/tmp/claude-0/-home-user-opaque/a97cca5e-900f-5617-b2df-d73b55081cac/scratchpad/research/empirical")
from common import *
import torch
from opaque.functional import make_functional
from opaque.dpsgd.clipping import clipped_grad
import transformers.models.mellum.modeling_mellum as mod
ref = torch.load(OUT / "e1b_ref.pt"); ids, mask = ref["ids"], ref["mask"]; B, T = ids.shape; k = 2
model, _ = build_patched(seed=0); model.load_state_dict(ref["state_dict"])
mod.load_balancing_loss_func = aux_loss_vmap_safe; model.router_aux_loss_coef = 0.0; set_trainable(model, "all")
mbf = model.to(torch.bfloat16); mbf.train()
fmodel, trainable, frozen = make_functional(mbf, disable_autograd_tracking=True, partition_trainable=True)
def per_ex(tr, fr, i, mk, lb):
    o = fmodel({**fr, **tr}, i, attention_mask=mk, labels=lb, output_router_logits=True); return o.loss, tuple(o.router_logits)
t0 = time.time()
grads, lg = torch.func.vmap(torch.func.grad(per_ex, argnums=0, has_aux=True), in_dims=(None, None, 0, 0, 0))(trainable, frozen, ids, mask, ids)
print("vmap(grad) bf16 B=32,T=64: %.1fs" % (time.time() - t0))
def flat(d, b): return torch.cat([d[n][b].reshape(-1).float() for n in sorted(d)])
flip_v = torch.zeros(B, dtype=torch.bool); ntok = 0; flip_v32 = torch.zeros(B, dtype=torch.bool); ntok32 = 0
for l in range(len(lg)):
    _, rows = route_flip_fraction(lg[l].float().reshape(-1, 8), ref["lgbf"][l].reshape(-1, 8), k); flip_v |= rows.reshape(B, T).any(1); ntok += int(rows.sum())
    _, rows = route_flip_fraction(lg[l].float().reshape(-1, 8), ref["lg32"][l].reshape(-1, 8), k); flip_v32 |= rows.reshape(B, T).any(1); ntok32 += int(rows.sum())
err_vb = torch.tensor([rel_l2(flat(grads, b), flat(ref["gbf"], b)) for b in range(B)])
err_v32 = torch.tensor([rel_l2(flat(grads, b), flat(ref["g32"], b)) for b in range(B)])
res = {"tokens_flipped_vmapbf16_vs_loopbf16": ntok, "examples_flipped": int(flip_v.sum()), "tokens_flipped_vmapbf16_vs_fp32": ntok32,
       "err_vmapbf16_vs_loopbf16": {"flip": err_vb[flip_v].tolist(), "noflip": err_vb[~flip_v].tolist()},
       "err_vmapbf16_vs_fp32": {"flip": err_v32[flip_v32].tolist(), "noflip": err_v32[~flip_v32].tolist()}}
print(f"vmap-bf16 vs loop-bf16: tokens flipped {ntok}/{B*T}, examples {int(flip_v.sum())}/{B}; err flip-ex mean {err_vb[flip_v].mean() if flip_v.any() else float('nan'):.4f} max {err_vb[flip_v].max() if flip_v.any() else float('nan'):.4f} | no-flip mean {err_vb[~flip_v].mean():.4f} max {err_vb[~flip_v].max():.4f}")
print(f"vmap-bf16 vs fp32-loop: tokens flipped {ntok32}/{B*T}, examples {int(flip_v32.sum())}/{B}; err flip-ex mean {err_v32[flip_v32].mean():.4f} | no-flip mean {err_v32[~flip_v32].mean():.4f}")
dump("e1b_vmap_results.json", res)
