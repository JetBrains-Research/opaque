"""Probe 2: correct imbalance per router-row scale, aux-vs-CE router gradient norms, adversarial example."""
import sys, time, math, torch
torch.set_num_threads(2)
sys.path.insert(0, "/home/user/opaque/packages/opaque-patches/tests/transformers/models")
sys.path.insert(0, "/tmp/claude-0/-home-user-opaque/a97cca5e-900f-5617-b2df-d73b55081cac/scratchpad/research/empirical")
from _test_utils import build_moe_model
from common import structured_batch
from opaque.api.patches.kernels._linear_ce_chunked import linear_nll_sum_chunked
from opaque.functional import make_functional
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
T = 32; N = 256
ids = structured_batch(N, T, seed=1)
g = torch.Generator().manual_seed(7)
lengths = torch.randint(T // 2, T + 1, (N,), generator=g)
mask = (torch.arange(T)[None] < lengths[:, None]).long()
ids = torch.where(mask.bool(), ids, torch.zeros_like(ids))
labels = torch.where(mask.bool(), ids, torch.full_like(ids, -100))

def batched_stats(params, ids, mask):
    out = fmodel({**frozen, **params}, input_ids=ids, attention_mask=mask, labels=labels[:ids.shape[0]], opaque_router_logits=True)
    B = ids.shape[0]; m = (mask != 0).float()
    Tx = m.sum(1)  # (B,)
    hL = []; P = torch.zeros(B, E)
    for z in out["router_logits"]:
        z = z.reshape(B, T, E); p = torch.softmax(z.float(), -1)
        idx = torch.topk(p, K, -1).indices
        oh = (idx[..., None] == torch.arange(E)).sum(-2).float()  # (B,T,E)
        hL.append((oh * m[..., None]).sum(1) / Tx.clamp(min=1)[:, None])
        P = P + (p * m[..., None]).sum(1)
    hL = torch.stack(hL, 1)  # (B,L,E)
    P = P / (L * Tx.clamp(min=1)[:, None])
    return out["loss"], hL, P, Tx

router_keys = [k for k in trainable if ".mlp.gate." in k]
base = {k: v.clone() for k, v in trainable.items()}
for scale in (1.0, 2.0, 3.0, 5.0):
    params = {k: v.clone() for k, v in base.items()}
    for k in router_keys:
        params[k][:2] *= scale
    with torch.no_grad():
        _, hL, P, Tx = batched_stats(params, ids, mask)
    f = (hL.mean(1) * Tx[:, None]).sum(0) / Tx.sum()
    d = f - K / E
    delta = (d.pow(2).mean().sqrt() / (K / E)).item()
    print(f"scale {scale}: f_tw={[round(x,3) for x in f.tolist()]} sum={f.sum():.3f} delta={delta:.3f} D={d.abs().max()/(K/E):.3f}")
    # aux vs CE gradient on router leaves (batch mean, token-weighted HF aux) at this scale
    def ce_fn(p):
        loss, *_ = batched_stats(p, ids[:32], mask[:32]); return loss
    def aux_fn(p):
        _, hL, P, Tx = batched_stats(p, ids[:32], mask[:32])
        fB = (hL.mean(1) * Tx[:, None]).sum(0) / Tx.sum()
        PB = (P * Tx[:, None]).sum(0) / Tx.sum()
        return E * (fB.detach() * PB).sum()
    gce = torch.func.grad(ce_fn)(params); gaux = torch.func.grad(aux_fn)(params)
    nce_r = math.sqrt(sum(gce[k].pow(2).sum().item() for k in router_keys)); naux_r = math.sqrt(sum(gaux[k].pow(2).sum().item() for k in router_keys))
    nce_a = math.sqrt(sum(gce[k].pow(2).sum().item() for k in gce if k not in router_keys and k != "router_load_probe"))
    naux_a = math.sqrt(sum(gaux[k].pow(2).sum().item() for k in gaux if k not in router_keys and k != "router_load_probe"))
    print(f"   batch-mean grad norms: CE router {nce_r:.4f} attn {nce_a:.4f} | aux(alpha=1) router {naux_r:.4f} attn {naux_a:.4f}")

# adversarial: all-same-token sequence
adv = torch.full((1, T), 5, dtype=torch.long); advm = torch.ones(1, T, dtype=torch.long)
with torch.no_grad():
    _, hL, P, Tx = batched_stats(base, adv, advm)
print("adversarial all-same-token h per layer:", hL[0].tolist(), " ||d||=", (hL[0] - K / E).norm().item(), " Delta_L=", math.sqrt(K * L * (1 - K / E)))
