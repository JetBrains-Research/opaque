import sys, time; sys.path.insert(0, "/tmp/claude-0/-home-user-opaque/a97cca5e-900f-5617-b2df-d73b55081cac/scratchpad/research/empirical")
from common import *
import torch
from opaque.functional import make_functional
from opaque.dpsgd.clipping import clipped_grad

t0=time.time()
model, mod = build_patched()
print("built patched in", time.time()-t0, "experts patched:", hasattr(mod.MellumExperts.forward, "__opaque_patched__"))
print("attn impl", model.config._attn_implementation, "layer_types", model.config.layer_types, "mlp", model.config.mlp_layer_types)
print("params:", sum(p.numel() for p in model.parameters()))
for n,p in model.named_parameters(): print(n, tuple(p.shape))
ids = random_batch(4, 16); mask = torch.ones_like(ids)
model.train()
out = model(input_ids=ids, attention_mask=mask, labels=ids, output_router_logits=True)
print("loss", out.loss.item(), "aux", out.aux_loss, "router_logits", type(out.router_logits), len(out.router_logits), out.router_logits[0].shape)
# under vmap with output_router_logits=True?
fmodel, trainable, frozen = make_functional(model, disable_autograd_tracking=True, partition_trainable=True)
def per_ex(tr, fr, ids, mask, lbl):
    o = fmodel({**fr, **tr}, ids, attention_mask=mask, labels=lbl, output_router_logits=True)
    return o.loss, (o.aux_loss,)
t0=time.time()
gfn, st = clipped_grad(per_ex, argnums=0, batch_argnums=(2,3,4), clipping_norm=1e9, has_aux=True, return_aux=True)
(g, aux), _ = gfn(trainable, frozen, ids, mask, ids, state=st)
print("vmap w/ router logits ok in", time.time()-t0, "loss_values", aux.loss_values, "aux", aux.loss_aux)
