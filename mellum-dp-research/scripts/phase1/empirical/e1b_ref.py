"""E1b (unpatched process): bf16 vs fp32 loop grads on B=32,T=64 with per-example route-flip
labels, plus a PINNED-ROUTING control (bf16 forward forced to the fp32 top-k choice)."""
import sys, time, copy; sys.path.insert(0, "/tmp/claude-0/-home-user-opaque/a97cca5e-900f-5617-b2df-d73b55081cac/scratchpad/research/empirical")
from common import *
import torch, torch.nn.functional as F
import transformers.models.mellum.modeling_mellum as mod
assert "opaque" not in sys.modules
B, T = 32, 64
k = TINY["num_experts_per_tok"]
t0 = time.time()
model = build_unpatched(seed=0)
ids = random_batch(B, T, seed=9); mask = torch.ones_like(ids)
names = [n for n, _ in model.named_parameters()]

PIN = {"active": False, "routes": None, "calls": 0}
_orig_router_forward = mod.MellumTopKRouter.forward
def pinned_router_forward(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states, self.weight)
    router_probs = torch.softmax(router_logits, dtype=torch.float, dim=-1)
    if PIN["active"]:
        idx = PIN["routes"][PIN["calls"]]; PIN["calls"] += 1
        top = torch.gather(router_probs, -1, idx)
    else:
        top, idx = torch.topk(router_probs, self.top_k, dim=-1)
    if self.norm_topk_prob:
        top = top / top.sum(dim=-1, keepdim=True)
    top = top.to(router_logits.dtype)
    return router_logits, top, idx
mod.MellumTopKRouter.forward = pinned_router_forward

def loop_grads(m, pin_routes=None):
    m.train(); cap = RouterCapture(m)
    grads = {n: [] for n in names}; logits = []
    for b in range(B):
        m.zero_grad(set_to_none=True); cap.clear()
        if pin_routes is not None:
            PIN["active"] = True; PIN["routes"] = pin_routes[b]; PIN["calls"] = 0
        out = m(input_ids=ids[b:b+1], attention_mask=mask[b:b+1], labels=ids[b:b+1])
        PIN["active"] = False
        out.loss.backward()
        logits.append([r[0].float() for r in cap.records])
        for n, p in m.named_parameters():
            grads[n].append(p.grad.detach().float().clone())
    cap.remove()
    return {n: torch.stack(g) for n, g in grads.items()}, [torch.stack([logits[b][l] for b in range(B)]) for l in range(len(logits[0]))]

g32, lg32 = loop_grads(model)
routes32 = [[topk_sets(lg32[l][b], k) for l in range(len(lg32))] for b in range(B)]
mbf = copy.deepcopy(model).to(torch.bfloat16)
gbf, lgbf = loop_grads(mbf)
gpin, lgpin = loop_grads(mbf, pin_routes=routes32)          # bf16 numerics, fp32 routes
def flat(d, b): return torch.cat([d[n][b].reshape(-1) for n in sorted(d)])
flip_ex = torch.zeros(B, dtype=torch.bool); flip_tok = 0
for l in range(len(lg32)):
    _, rows = route_flip_fraction(lgbf[l].reshape(-1, 8), lg32[l].reshape(-1, 8), k)
    flip_ex |= rows.reshape(B, T).any(1); flip_tok += int(rows.sum())
err_bf = torch.tensor([rel_l2(flat(gbf, b), flat(g32, b)) for b in range(B)])
err_pin = torch.tensor([rel_l2(flat(gpin, b), flat(g32, b)) for b in range(B)])
# param-group breakdown of the bf16-vs-fp32 error, flipped examples only
groups = {"attn(qkvo)": lambda n: ".self_attn." in n and "_norm" not in n, "router": lambda n: ".mlp.gate." in n, "experts": lambda n: ".mlp.experts." in n, "embed/lm_head/norms": lambda n: True}
def group_err(gd, b, pred):
    a = torch.cat([gd[n][b].reshape(-1) for n in sorted(gd) if pred(n)]); r = torch.cat([g32[n][b].reshape(-1) for n in sorted(g32) if pred(n)]); return rel_l2(a, r)
res = {"n_tokens_flipped": flip_tok, "n_examples_with_flip": int(flip_ex.sum()), "B": B, "T": T,
       "err_bf16_vs_fp32": {"flip": err_bf[flip_ex].tolist(), "noflip": err_bf[~flip_ex].tolist()},
       "err_bf16_pinned_vs_fp32": {"flip": err_pin[flip_ex].tolist(), "noflip": err_pin[~flip_ex].tolist()},
       "group_err_bf16_flip_examples": {gname: [group_err(gbf, b, pred) for b in range(B) if flip_ex[b]] for gname, pred in groups.items()},
       "group_err_bf16_noflip_examples": {gname: [group_err(gbf, b, pred) for b in range(B) if not flip_ex[b]] for gname, pred in groups.items()},
       "group_err_pinned_flip_examples": {gname: [group_err(gpin, b, pred) for b in range(B) if flip_ex[b]] for gname, pred in groups.items()}}
print(f"tokens flipped (bf16 vs fp32, any layer): {flip_tok}/{B*T}; examples with >=1 flip: {int(flip_ex.sum())}/{B}")
print(f"bf16 vs fp32 per-example rel L2: flip-examples mean {err_bf[flip_ex].mean():.4f} max {err_bf[flip_ex].max():.4f} | no-flip mean {err_bf[~flip_ex].mean():.4f} max {err_bf[~flip_ex].max():.4f}")
print(f"bf16 PINNED-to-fp32-routes vs fp32: flip-examples mean {err_pin[flip_ex].mean():.4f} max {err_pin[flip_ex].max():.4f} | no-flip mean {err_pin[~flip_ex].mean():.4f}")
for gname in groups:
    fe = res["group_err_bf16_flip_examples"][gname]; ne = res["group_err_bf16_noflip_examples"][gname]; pe = res["group_err_pinned_flip_examples"][gname]
    print(f"  {gname:>20}: flip-ex bf16 err mean {sum(fe)/max(len(fe),1):.4f} max {max(fe) if fe else 0:.4f} | no-flip mean {sum(ne)/max(len(ne),1):.4f} | pinned flip-ex mean {sum(pe)/max(len(pe),1):.4f}")
dump("e1b_results.json", res)
torch.save({"state_dict": model.state_dict(), "ids": ids, "mask": mask, "g32": g32, "gbf": gbf, "lg32": lg32, "lgbf": lgbf, "flip_ex": flip_ex}, OUT / "e1b_ref.pt")
print("done %.1fs" % (time.time() - t0))
