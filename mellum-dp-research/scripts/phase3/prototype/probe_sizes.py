"""Sizing probe: step time, grad norms, induced imbalance, nm calibration."""
import sys, time, math, torch
torch.set_num_threads(2)
sys.path.insert(0, "/home/user/opaque/packages/opaque-patches/tests/transformers/models")
sys.path.insert(0, "/tmp/claude-0/-home-user-opaque/a97cca5e-900f-5617-b2df-d73b55081cac/scratchpad/research/empirical")
from _test_utils import build_moe_model
from common import structured_batch
from opaque.api.patches.kernels._linear_ce_chunked import linear_nll_sum_chunked
from opaque.functional import make_functional
from opaque.dpsgd.clipping import clipped_grad
from opaque.types import PerGroup
import opaque.accounting as acc
import opaque.dpsgd.accounting as dpsgd_acc

E, K, L = 8, 2, 2
TINY = dict(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=L,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
            pad_token_id=0, bos_token_id=1, eos_token_id=2, rope_theta=10000.0,
            num_experts=E, num_experts_per_tok=K, moe_intermediate_size=32)
torch.manual_seed(0)
model, mod = build_moe_model("mellum", "cpu", **TINY)
model.train()
for n, p in model.named_parameters():
    p.requires_grad_(any(s in n for s in ("q_proj", "k_proj", "v_proj", "o_proj")) or ".mlp.gate." in n)
model.register_parameter("router_load_probe", torch.nn.Parameter(torch.zeros(L, E)))
CausalLM = type(model); _orig = CausalLM.forward
def forward(self, input_ids=None, attention_mask=None, labels=None, opaque_router_logits=False, **kw):
    if not opaque_router_logits:
        return _orig(self, input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kw)
    out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_router_logits=True)
    nll = linear_nll_sum_chunked(out[0], self.lm_head.weight, labels, -100, 0, 0.0, False, chunk_vocab=None)
    n_valid = (labels[..., 1:] != -100).sum().float().clamp(min=1)
    return {"loss": nll / n_valid, "router_logits": out.router_logits}
CausalLM.forward = forward
fmodel, trainable, frozen = make_functional(model, disable_autograd_tracking=True, partition_trainable=True)
print("trainable:", {k: tuple(v.shape) for k, v in trainable.items()})
print("n_trainable:", sum(v.numel() for v in trainable.values()))

def stats(rl, m):
    h = []; P = torch.zeros(E)
    Tx = m.sum()
    for z in rl:
        z = z.reshape(-1, E); p = torch.softmax(z.float(), -1)
        idx = torch.topk(p, K, -1).indices
        oh = (idx[..., None] == torch.arange(E)).sum(-2).float()
        h.append((oh * m[:, None]).sum(0) / Tx.clamp(min=1))
        P = P + (p * m[:, None]).sum(0)
    return torch.stack(h), P / (len(rl) * Tx.clamp(min=1)), Tx

def loss_fn(tr, ids, mask, labels):
    params = {**frozen, **tr}
    out = fmodel(params, input_ids=ids[None], attention_mask=mask[None], labels=labels[None], opaque_router_logits=True)
    m = (mask != 0).float()
    hL, P, Tx = stats(out["router_logits"], m)
    return out["loss"], (hL, P)

for T in (32, 48):
    N = 1024
    ids = structured_batch(N, T, seed=1)
    g = torch.Generator().manual_seed(7)
    lengths = torch.randint(T // 2, T + 1, (N,), generator=g)
    mask = (torch.arange(T)[None] < lengths[:, None]).long()
    ids = torch.where(mask.bool(), ids, torch.zeros_like(ids))
    labels = torch.where(mask.bool(), ids, torch.full_like(ids, -100))
    for B in (16, 32):
        pg = PerGroup(groups={**{(k,): "fallback" for k in trainable if k != "router_load_probe"}, ("router_load_probe",): "router_load_probe"},
                      values={"fallback": 1e9, "router_load_probe": 1e9})
        gf, st = clipped_grad(loss_fn, has_aux=True, clipping_norm=pg, normalize_by=B, batch_argnums=(1, 2, 3), return_aux=True)
        (out, aux), st = gf(trainable, ids[:B], mask[:B], labels[:B], state=st)
        t0 = time.time()
        (out, aux), st = gf(trainable, ids[B:2*B], mask[B:2*B], labels[B:2*B], state=st)
        dt = time.time() - t0
        gn = aux.group_norms["fallback"]
        print(f"T={T} B={B}: step {dt:.2f}s; grad norms p10/p50/p90/max = {gn.quantile(0.1):.3f}/{gn.median():.3f}/{gn.quantile(0.9):.3f}/{gn.max():.3f}; CE mean {aux.loss_values.mean():.3f}")

# induced imbalance: scale two router rows
T = 32; N = 256
ids = structured_batch(N, T, seed=1)
g = torch.Generator().manual_seed(7)
lengths = torch.randint(T // 2, T + 1, (N,), generator=g)
mask = (torch.arange(T)[None] < lengths[:, None]).long()
ids = torch.where(mask.bool(), ids, torch.zeros_like(ids))
def batch_f(tr, ids, mask):
    with torch.no_grad():
        out = fmodel({**frozen, **tr}, input_ids=ids, attention_mask=mask, output_router_logits=True) if False else model.model(input_ids=ids, attention_mask=mask, output_router_logits=True)
    m = mask.reshape(-1).float()
    cnt = torch.zeros(E); tot = 0.0
    for z in out.router_logits:
        p = torch.softmax(z.float(), -1); idx = torch.topk(p, K, -1).indices
        oh = (idx[..., None] == torch.arange(E)).sum(-2).float()
        cnt += (oh * m[:, None]).sum(0); tot += m.sum()
    return cnt / tot * len(out.router_logits)
router_keys = [k for k in trainable if ".mlp.gate." in k]
print("router keys:", router_keys)
base = {k: v.clone() for k, v in trainable.items()}
for scale in (1.0, 2.0, 3.0, 5.0):
    for k in router_keys:
        w = model.get_parameter(k) if False else dict(model.named_parameters())[k]
        w.data.copy_(base[k]); w.data[:2] *= scale
    f = batch_f(trainable, ids, mask)
    d = f - K / E
    delta = (d.pow(2).mean().sqrt() / (K / E)).item()
    print(f"router row scale {scale}: f(B)={[round(x,3) for x in f.tolist()]} sum={f.sum():.3f} delta={delta:.3f} D={d.abs().max()/(K/E):.3f}")
for k in router_keys:
    dict(model.named_parameters())[k].data.copy_(base[k])

# nm calibration at the tiny regime
for Tsteps, Bbar in ((300, 32), (200, 32)):
    q = Bbar / 1024
    for eps in (3.0, 8.0):
        t0 = time.time()
        r = acc.calibrate(acc.epsilon_budget(eps, 1e-5), lambda x: dpsgd_acc.poisson(dpsgd_acc.gaussian(x), sample_rate=q) * Tsteps, 0.3, 8.0, tolerance=1e-3)
        print(f"T={Tsteps} B={Bbar} eps={eps}: nm={r.param:.4f} ({time.time()-t0:.1f}s)")
