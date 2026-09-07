import copy, math, sys, torch
torch.set_num_threads(2); torch.manual_seed(0)
from transformers.models.mellum.configuration_mellum import MellumConfig
from transformers.models.mellum.modeling_mellum import MellumForCausalLM
grouped = bool(int(sys.argv[1])); E = int(sys.argv[2]); K = int(sys.argv[3]); pad = sys.argv[4]
def log(*a): print(*a, flush=True)
def rel_l2(a, b):
    a = a.float().flatten(); b = b.float().flatten(); den = b.norm().item(); return (a - b).norm().item() / den if den > 0 else (a-b).norm().item()
kw = dict(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
          max_position_embeddings=128, pad_token_id=0, bos_token_id=1, eos_token_id=2, num_experts=E, num_experts_per_tok=K,
          moe_intermediate_size=64, norm_topk_prob=True, layer_types=["sliding_attention", "full_attention"], sliding_window=8)
cfg = MellumConfig(**kw); cfg._attn_implementation = "sdpa"
B, T = 4, 16
ref = MellumForCausalLM(cfg).train()
ids = torch.randint(3, cfg.vocab_size, (B, T)); mask = torch.ones(B, T, dtype=torch.long)
if pad == "both": mask[1, -3:] = 0; mask[2, :4] = 0
elif pad == "left4": mask[2, :4] = 0
elif pad == "right3": mask[1, -3:] = 0
labels = ids.clone(); labels[mask == 0] = -100
def hf_loop(model):
    ls, hs = [], []
    for i in range(B):
        with torch.no_grad():
            out = model(input_ids=ids[i:i+1], attention_mask=mask[i:i+1], labels=labels[i:i+1], output_hidden_states=True, output_router_logits=True)
        ls.append(out.loss.item()); hs.append(([h[0] for h in out.hidden_states], [r for r in out.router_logits]))
    return ls, hs
l_loop, hs_loop = hf_loop(copy.deepcopy(ref))
from opaque.patches import apply_runtime_patches, apply_model_patches
apply_runtime_patches(compat=True)
m = copy.deepcopy(ref); apply_model_patches(m, compat=True, performance=True, kernels=grouped, grouped_moe=grouped)
from opaque.api.patches.kernels import moe as moe_mod, _grouped_moe as gmoe_mod
cnt = {"dense": 0, "grouped": 0}
_of = moe_mod._moe_forward; _og = gmoe_mod._fused_moe_forward
moe_mod._moe_forward = lambda *a, **k: (cnt.__setitem__("dense", cnt["dense"]+1), _of(*a, **k))[1]
gmoe_mod._fused_moe_forward = lambda *a, **k: (cnt.__setitem__("grouped", cnt["grouped"]+1), _og(*a, **k))[1]
backbone = m.model; params = {**dict(backbone.named_parameters()), **dict(backbone.named_buffers())}
def f(i, mk):
    out = torch.func.functional_call(backbone, params, (), {"input_ids": i[None], "attention_mask": mk[None], "output_hidden_states": True, "output_router_logits": True})
    return tuple(h[0] for h in out.hidden_states), tuple(out.router_logits)
with torch.no_grad():
    hs_v, rl_v = torch.func.vmap(f, in_dims=(0, 0))(ids, mask)
fm = copy.deepcopy(m)
def fl(i, mk, l):
    return fm(input_ids=i, attention_mask=mk, labels=l).loss
with torch.no_grad():
    l_v = torch.func.vmap(fl, in_dims=(0, 0, 0))(ids, mask, labels)
log(f"grouped={grouped} E={E} K={K} pad={pad}: forward dispatch {cnt}")
log("  losses loop:", ["%.5f"%x for x in l_loop], " vmap:", ["%.5f"%x for x in l_v.tolist()])
for i in range(B):
    errs = ["%.1e"%rel_l2(hs_v[j][i], hs_loop[i][0][j]) for j in range(len(hs_v))]
    rerr = ["%.1e"%rel_l2(rl_v[j][i], hs_loop[i][1][j]) for j in range(len(rl_v))]
    flips = [int(sum(set(a.tolist()) != set(b.tolist()) for a, b in zip(torch.topk(rl_v[j][i].float(), K, -1).indices, torch.topk(hs_loop[i][1][j].float(), K, -1).indices))) for j in range(len(rl_v))]
    log(f"  ex{i} hidden-state rel-L2 per layer (embed, L0, L1(final-normed)): {errs}; router-logit err per layer {rerr}; route flips per layer {flips}")
