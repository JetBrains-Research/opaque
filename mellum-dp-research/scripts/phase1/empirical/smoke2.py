import sys, time; sys.path.insert(0, "/tmp/claude-0/-home-user-opaque/a97cca5e-900f-5617-b2df-d73b55081cac/scratchpad/research/empirical")
from common import *
import torch
from opaque.functional import make_functional
from opaque.dpsgd.clipping import clipped_grad
model, mod = build_patched()
ids = random_batch(4, 16); mask = torch.ones_like(ids)
model.train()
fmodel, trainable, frozen = make_functional(model, disable_autograd_tracking=True, partition_trainable=True)
# no attention mask -> bincount path
def per_ex(tr, fr, ids, lbl):
    o = fmodel({**fr, **tr}, ids, labels=lbl, output_router_logits=True)
    return o.loss, (o.aux_loss,)
gfn, st = clipped_grad(per_ex, argnums=0, batch_argnums=(2,3), clipping_norm=1e9, has_aux=True, return_aux=True)
try:
    (g, aux), _ = gfn(trainable, frozen, ids, ids, state=st)
    print("no-mask vmap OK; per-example aux:", aux.loss_aux)
except Exception as e:
    print("no-mask vmap FAILS:", type(e).__name__, str(e)[:300])
# can we get router_logits out under vmap (has_aux) to build our own per-example aux?
def per_ex2(tr, fr, ids, mask, lbl):
    o = fmodel({**fr, **tr}, ids, attention_mask=mask, labels=lbl, output_router_logits=False, output_hidden_states=False)
    return o.loss, ()
gfn, st = clipped_grad(per_ex2, argnums=0, batch_argnums=(2,3,4), clipping_norm=1e9, has_aux=True, return_aux=True)
(g, aux), _ = gfn(trainable, frozen, ids, mask, ids, state=st); print("baseline vmap OK", aux.loss_values)
