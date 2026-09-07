"""E1 reference (UNPATCHED HF process): per-example autograd loop grads in fp32 and bf16,
router indices per token, plus HF batched-vs-loop check (H1 on the HF side)."""
import sys, time, copy; sys.path.insert(0, "/tmp/claude-0/-home-user-opaque/a97cca5e-900f-5617-b2df-d73b55081cac/scratchpad/research/empirical")
from common import *
import torch
assert "opaque" not in sys.modules
B, T = 8, 32
t0 = time.time()
model = build_unpatched(seed=0)
print("experts impl:", getattr(model.config, "_experts_implementation", None))
sd = {k: v.clone() for k, v in model.state_dict().items()}
ids = random_batch(B, T, seed=1); mask = torch.ones_like(ids)
res = {"state_dict": sd, "ids": ids, "mask": mask}

def loop_grads(m):
    m.train()
    cap = RouterCapture(m)
    names = [n for n, p in m.named_parameters()]
    grads = {n: [] for n in names}; losses = []; logits = []  # logits: per example, per layer
    for b in range(B):
        m.zero_grad(set_to_none=True); cap.clear()
        out = m(input_ids=ids[b:b+1], attention_mask=mask[b:b+1], labels=ids[b:b+1], output_router_logits=False)
        out.loss.backward()
        losses.append(out.loss.detach().float())
        logits.append([r[0].float() for r in cap.records])
        for n, p in m.named_parameters():
            grads[n].append(p.grad.detach().float().clone() if p.grad is not None else torch.zeros_like(p, dtype=torch.float32))
    cap.remove()
    grads = {n: torch.stack(g) for n, g in grads.items()}
    L = len(logits[0])
    logits = [torch.stack([logits[b][l] for b in range(B)]) for l in range(L)]  # (B, T, E) per layer
    return grads, torch.stack(losses), logits

def batched_grads(m):
    m.train(); m.zero_grad(set_to_none=True)
    out = m(input_ids=ids, attention_mask=mask, labels=ids, output_router_logits=False)
    out.loss.backward()
    return {n: (p.grad.detach().float().clone() if p.grad is not None else torch.zeros_like(p, dtype=torch.float32)) for n, p in m.named_parameters()}, out.loss.detach().float()

res["fp32"] = {}
g, l, lg = loop_grads(model); res["fp32"]["loop_grads"], res["fp32"]["loop_losses"], res["fp32"]["logits"] = g, l, lg
gb, lb = batched_grads(model); res["fp32"]["batched_grads"], res["fp32"]["batched_loss"] = gb, lb
# H1 check on HF side: grad(batched token-mean loss) * B == sum_x grad(per-example mean loss)
errs = {n: rel_l2(gb[n] * B, g[n].sum(0)) for n in g}
print("HF batched*B vs loop-sum: max rel L2 err over params = %.3e (loss batched %.6f, mean loop %.6f)" % (max(errs.values()), lb.item(), l.mean().item()))
res["fp32"]["h1_batched_vs_loop_relerr"] = errs

mbf = copy.deepcopy(model).to(torch.bfloat16)
res["bf16"] = {}
g, l, lg = loop_grads(mbf); res["bf16"]["loop_grads"], res["bf16"]["loop_losses"], res["bf16"]["logits"] = g, l, lg
gb, lb = batched_grads(mbf); res["bf16"]["batched_grads"], res["bf16"]["batched_loss"] = gb, lb
errs = {n: rel_l2(gb[n] * B, g[n].sum(0)) for n in g}
print("bf16 HF batched*B vs loop-sum: max rel L2 err = %.3e" % max(errs.values()))
res["bf16"]["h1_batched_vs_loop_relerr"] = errs
torch.save(res, OUT / "e1_ref.pt")
print("done in %.1fs" % (time.time() - t0))
