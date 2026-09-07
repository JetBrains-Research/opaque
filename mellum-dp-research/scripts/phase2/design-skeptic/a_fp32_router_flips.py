"""A: where do bf16 route flips come from, and does an fp32-logit router run under vmap(grad)?

Tiny Mellum (E=64, top-8, 2 layers, hidden 64), random init, CPU. Compare top-k sets against the
fp32 model for (i) bf16 model + stock router (bf16 F.linear, fp32 softmax) and (ii) bf16 model +
fp32-logit router (hidden.float() @ W.float()). Then run opaque clipped_grad (vmap(grad)) on the
patched model with the fp32 router installed and check finiteness + zero flips vs the eager fp32-router
forward of the same bf16 model.
"""
import copy, sys, time, json
import torch, torch.nn.functional as F
torch.set_num_threads(2)
sys.path.insert(0, "/tmp/claude-0/-home-user-opaque/a97cca5e-900f-5617-b2df-d73b55081cac/scratchpad/research/empirical")
from common import build_unpatched, build_patched, random_batch, structured_batch, copy_weights

from transformers.models.mellum.modeling_mellum import MellumTopKRouter

K, E = 8, 64
OV = dict(num_experts=E, num_experts_per_tok=K, moe_intermediate_size=32)


def fp32_router_forward(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    logits = F.linear(hidden_states.float(), self.weight.float())  # fp32 logits
    probs = torch.softmax(logits, dim=-1, dtype=torch.float)
    top_v, top_i = torch.topk(probs, self.top_k, dim=-1)
    if self.norm_topk_prob:
        top_v = top_v / top_v.sum(dim=-1, keepdim=True)
    top_v = top_v.to(hidden_states.dtype)
    return logits.to(hidden_states.dtype), top_v, top_i


class Cap:
    def __init__(self, model):
        self.recs = []
        self.h = [m.register_forward_hook(lambda mod, i, o: self.recs.append(o[2].detach().clone()))
                  for m in model.modules() if isinstance(m, MellumTopKRouter)]
    def remove(self):
        for h in self.h: h.remove()


def routes(model, ids):
    cap = Cap(model)
    with torch.no_grad():
        model(input_ids=ids, attention_mask=torch.ones_like(ids))
    cap.remove()
    return [r.reshape(-1, K) for r in cap.recs]


def flips(a, b):
    sa = torch.zeros(a.shape[0], E, dtype=torch.bool).scatter_(1, a, True)
    sb = torch.zeros(b.shape[0], E, dtype=torch.bool).scatter_(1, b, True)
    return int((sa != sb).any(1).sum()), a.shape[0]


t0 = time.time()
B, T = 16, 64
out = {}
m32 = build_unpatched(seed=0, **OV).eval()
for kind, ids in [("random", random_batch(B, T, seed=3)), ("structured", structured_batch(B, T, seed=3))]:
    ref = routes(m32, ids)
    mbf = copy.deepcopy(m32).to(torch.bfloat16).eval()
    stock = routes(mbf, ids)
    orig = MellumTopKRouter.forward
    MellumTopKRouter.forward = fp32_router_forward
    fp32r = routes(mbf, ids)
    MellumTopKRouter.forward = orig
    res = {}
    for l in range(len(ref)):
        fs, n = flips(stock[l], ref[l]); ff, _ = flips(fp32r[l], ref[l])
        res[f"L{l}"] = {"stock_bf16_flips": fs, "fp32_logit_flips": ff, "rows": n}
    out[kind] = res
    print(kind, res)

# vmap(grad) with the fp32 router on the patched (opaque) model
from opaque.dpsgd.clipping import clipped_grad
from opaque.functional import make_functional
mp, mod = build_patched(seed=0, **OV)
copy_weights(m32, mp)
mp = mp.to(torch.bfloat16)
for n, p in mp.named_parameters():
    p.requires_grad_(("q_proj" in n) or ("v_proj" in n))
mod.MellumTopKRouter.forward = fp32_router_forward
fmodel, trainable, frozen = make_functional(mp, disable_autograd_tracking=True, partition_trainable=True)
ids = random_batch(B, T, seed=3)
def loss_fn(params, input_ids, attention_mask, labels):
    return fmodel({**frozen, **params}, input_ids=input_ids, attention_mask=attention_mask, labels=labels)["loss"]
grad_fn, st = clipped_grad(loss_fn, batch_argnums=(1, 2, 3), clipping_norm=1.0, normalize_by=B, return_aux=True)
(g, aux), st = grad_fn(trainable, ids, torch.ones_like(ids), ids.clone(), state=st)
finite = all(torch.isfinite(v).all().item() for v in g.pytree.values())
print("vmap(grad) with fp32 router: keys", sorted(g.pytree)[:2], "... finite", finite,
      "grad_norms median", aux.grad_norms.median().item())
# eager per-example routes (fp32 router, bf16 model) vs routes captured under the vmap forward
from torch.func import vmap
cap = Cap(mp)
def fwd_routes(i, m):
    cap.recs.clear()
    fmodel({**frozen, **trainable}, input_ids=i, attention_mask=m)
    return torch.stack([r.reshape(-1, K) for r in cap.recs])  # (L, T, K) per example
with torch.no_grad():
    vm = vmap(fwd_routes)(ids, torch.ones_like(ids))  # (B, L, T, K)
cap.remove()
eager = routes(mp, ids)
vm = [vm[:, l].reshape(-1, K) for l in range(vm.shape[1])]
print("vmap-vs-eager flips per layer (fp32 router, bf16 model):", [flips(vm[l], eager[l]) for l in range(len(eager))])
out["vmap_finite"] = finite
json.dump(out, open("/tmp/claude-0/-home-user-opaque/a97cca5e-900f-5617-b2df-d73b55081cac/scratchpad/research/design-skeptic/a_out.json", "w"), indent=1)
print(f"done {time.time()-t0:.1f}s")
