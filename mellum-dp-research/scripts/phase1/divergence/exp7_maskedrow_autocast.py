import copy, math, sys, torch
torch.set_num_threads(2); torch.manual_seed(0)
from transformers.models.mellum.configuration_mellum import MellumConfig
from transformers.models.mellum.modeling_mellum import MellumForCausalLM
def log(*a): print(*a, flush=True)
def rel_l2(a, b):
    a = a.float().flatten(); b = b.float().flatten(); den = b.norm().item(); return (a - b).norm().item() / den if den > 0 else (a-b).norm().item()
# (a) raw SDPA fully-masked row convention
q = torch.randn(1, 2, 4, 8); k = torch.randn(1, 2, 4, 8); v = torch.randn(1, 2, 4, 8)
bm = torch.ones(1, 1, 4, 4, dtype=torch.bool).tril(); bm[..., 0, :] = False   # row 0 fully masked
am = torch.zeros(1, 1, 4, 4); am.masked_fill_(~bm, torch.finfo(torch.float32).min)
o_bool = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bm)
o_add = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=am)
log("SDPA fully-masked row: boolean-mask row0 =", o_bool[0, 0, 0, :4].tolist(), " additive-min row0 =", o_add[0, 0, 0, :4].tolist(), " mean(v) row0 =", v[0, 0].mean(0)[:4].tolist())
kw = dict(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
          max_position_embeddings=128, pad_token_id=0, bos_token_id=1, eos_token_id=2, num_experts=8, num_experts_per_tok=2,
          moe_intermediate_size=64, norm_topk_prob=True, layer_types=["sliding_attention", "full_attention"], sliding_window=8)
cfg = MellumConfig(**kw); cfg._attn_implementation = "sdpa"
B, T = 4, 16
ref = MellumForCausalLM(cfg).train()
ids = torch.randint(3, cfg.vocab_size, (B, T)); mask = torch.ones(B, T, dtype=torch.long); mask[1, -3:] = 0; mask[2, :4] = 0
labels = ids.clone(); labels[mask == 0] = -100
def hf_loss(model, i, attn, labels_=labels):
    model = copy.deepcopy(model); model.config._attn_implementation = attn
    with torch.no_grad():
        return model(input_ids=ids[i:i+1], attention_mask=mask[i:i+1], labels=labels_[i:i+1]).loss.item()
log(f"HF ex2 (left pad 4) loss: sdpa {hf_loss(ref, 2, 'sdpa'):.5f} vs eager {hf_loss(ref, 2, 'eager'):.5f}; ex1 (right pad): sdpa {hf_loss(ref, 1, 'sdpa'):.5f} eager {hf_loss(ref, 1, 'eager'):.5f}")
labels_b = labels.clone(); labels_b[2, 4] = -100   # drop the boundary target predicted from the pad row
log(f"HF ex2 with boundary target masked: sdpa {hf_loss(ref, 2, 'sdpa', labels_b):.5f} vs eager {hf_loss(ref, 2, 'eager', labels_b):.5f}")
# (b) autocast precision staging: fp32 master weights + bf16 autocast on CPU
from transformers.integrations.moe import _grouped_mm
def loop_grads(model, impl, autocast):
    model = copy.deepcopy(model); model.config._experts_implementation_internal = impl
    gs = []
    for i in range(B):
        model.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            out = model(input_ids=ids[i:i+1], attention_mask=mask[i:i+1], labels=labels_b[i:i+1])
        out.loss.backward()
        gs.append({n: p.grad.detach().clone() for n, p in model.named_parameters()})
    return gs
r32 = loop_grads(ref, "eager", False)
r_ac_grouped = loop_grads(ref, "grouped_mm", True)
r_ac_eager = loop_grads(ref, "eager", True)
r_ac_batched = loop_grads(ref, "batched_mm", True)
def cmp(name, a, b):
    num = sum((a[i][n].float() - b[i][n].float()).pow(2).sum().item() for i in range(B) for n in a[0]); den = sum(b[i][n].float().pow(2).sum().item() for i in range(B) for n in b[0])
    exp_keys = [n for n in a[0] if "experts" in n]
    num_e = sum((a[i][n].float() - b[i][n].float()).pow(2).sum().item() for i in range(B) for n in exp_keys); den_e = sum(b[i][n].float().pow(2).sum().item() for i in range(B) for n in exp_keys)
    log(f"[{name}] overall rel-L2 {math.sqrt(num/den):.3e}; experts-only {math.sqrt(num_e/den_e):.3e}")
cmp("HF autocast grouped_mm vs HF fp32", r_ac_grouped, r32)
cmp("HF autocast eager-experts vs HF fp32", r_ac_eager, r32)
cmp("HF autocast batched_mm vs HF fp32", r_ac_batched, r32)
cmp("HF autocast grouped_mm vs HF autocast eager-experts", r_ac_grouped, r_ac_eager)
# check what dtype the HF grouped_mm actually runs in under autocast with fp32 weights
seen = {}
import transformers.integrations.moe as moe
_orig = moe._grouped_mm
def spy(inp, w, offs):
    seen["input"] = inp.dtype; seen["weight"] = w.dtype; return _orig(inp, w, offs)
moe._grouped_mm = spy
loop_grads(ref, "grouped_mm", True); moe._grouped_mm = _orig
log("HF grouped_mm under CPU bf16 autocast with fp32 weights: input dtype", seen.get("input"), "weight dtype", seen.get("weight"))
from opaque.patches import apply_runtime_patches, apply_model_patches
from opaque.functional import make_functional
apply_runtime_patches(compat=True)
m = copy.deepcopy(ref); apply_model_patches(m, compat=True, performance=True, kernels=False)
from opaque.api.patches.kernels import moe as omoe
seen2 = {}
_of = omoe._moe_forward
def spy2(x, W1, W2, idx, tw):
    seen2["x"] = x.dtype; seen2["W1"] = W1.dtype; return _of(x, W1, W2, idx, tw)
omoe._moe_forward = spy2
fmodel, tr, fr = make_functional(m, disable_autograd_tracking=True, partition_trainable=True)
def f(tr, fr, i, mk, l):
    return fmodel({**fr, **tr}, i, attention_mask=mk, labels=l).loss
with torch.autocast("cpu", dtype=torch.bfloat16):
    g = torch.func.vmap(torch.func.grad(f), in_dims=(None, None, 0, 0, 0))(tr, fr, ids, mask, labels_b)
log("Opaque dense MoE under CPU bf16 autocast: x dtype", seen2.get("x"), "W1 dtype", seen2.get("W1"))
gl = [{n: g[n][i] for n in g} for i in range(B)]
cmp("Opaque vmap autocast vs HF fp32", gl, r32)
cmp("Opaque vmap autocast vs HF autocast grouped_mm loop", gl, r_ac_grouped)
cmp("Opaque vmap autocast vs HF autocast eager-experts loop", gl, r_ac_eager)
