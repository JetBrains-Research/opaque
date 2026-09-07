import copy, math, sys, torch
torch.set_num_threads(2); torch.manual_seed(0)
from transformers.models.mellum.configuration_mellum import MellumConfig
from transformers.models.mellum.modeling_mellum import MellumForCausalLM
def log(*a): print(*a, flush=True)
def rel_l2(a, b):
    a = a.float().flatten(); b = b.float().flatten(); den = b.norm().item(); return (a - b).norm().item() / den if den > 0 else (a-b).norm().item()
kw = dict(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
          max_position_embeddings=128, pad_token_id=0, bos_token_id=1, eos_token_id=2, num_experts=8, num_experts_per_tok=2,
          moe_intermediate_size=64, norm_topk_prob=True, layer_types=["sliding_attention", "full_attention"], sliding_window=8)
cfg = MellumConfig(**kw); cfg._attn_implementation = "sdpa"
B, T = 4, 16
ref = MellumForCausalLM(cfg).train()
ids = torch.randint(3, cfg.vocab_size, (B, T)); mask = torch.ones(B, T, dtype=torch.long)
mask[1, -3:] = 0; mask[2, :4] = 0
labels = ids.clone(); labels[mask == 0] = -100
def hf_one(model, i):
    with torch.no_grad():
        out = model(input_ids=ids[i:i+1], attention_mask=mask[i:i+1], labels=labels[i:i+1], output_hidden_states=True)
    return out.loss.item(), out.hidden_states[-1][0], out.logits[0]
loop = [hf_one(copy.deepcopy(ref), i) for i in range(B)]
from opaque.patches import apply_runtime_patches, apply_model_patches
apply_runtime_patches(compat=True)
m = copy.deepcopy(ref); apply_model_patches(m, compat=True, performance=True, kernels=False)
def fl(i, mk, l):
    out = m(input_ids=i, attention_mask=mk, labels=l, output_hidden_states=True)
    return out.loss, out.hidden_states[-1], out.logits
for name, sel in [("B=1 [ex2]", [2]), ("B=2 [ex2,ex2]", [2, 2]), ("B=2 [ex0,ex2]", [0, 2]), ("B=2 [ex1,ex2]", [1, 2]), ("B=4 all", [0, 1, 2, 3]), ("B=1 [ex1]", [1]), ("B=2 [ex0,ex1]", [0, 1])]:
    idx = torch.tensor(sel)
    with torch.no_grad():
        lv, hv, lgv = torch.func.vmap(fl, in_dims=(0, 0, 0))(ids[idx], mask[idx], labels[idx])
    msgs = []
    for j, i in enumerate(sel):
        valid = mask[i] == 1
        h = hv[j][0] if hv[j].ndim == 3 else hv[j]
        lg = lgv[j][0] if lgv[j].ndim == 3 else lgv[j]
        msgs.append(f"ex{i}: loss vmap {lv[j].item():.5f} loop {loop[i][0]:.5f}; final-hidden valid rel {rel_l2(h[valid], loop[i][1][valid]):.1e}; logits valid rel {rel_l2(lg[valid], loop[i][2][valid]):.1e}")
    log(f"{name}: " + " | ".join(msgs))
# non-vmap batched patched model with mixed masks, vs loop
with torch.no_grad():
    outb = m(input_ids=ids, attention_mask=mask, labels=labels, output_hidden_states=True)
log("patched NON-vmap B=4:", " | ".join(f"ex{i}: logits valid rel {rel_l2(outb.logits[i][mask[i]==1], loop[i][2][mask[i]==1]):.1e}" for i in range(B)))
with torch.no_grad():
    outr = copy.deepcopy(ref)(input_ids=ids, attention_mask=mask, labels=labels, output_hidden_states=True)
log("HF NON-vmap B=4:", " | ".join(f"ex{i}: logits valid rel {rel_l2(outr.logits[i][mask[i]==1], loop[i][2][mask[i]==1]):.1e}" for i in range(B)))
