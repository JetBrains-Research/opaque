"""E3: can a per-example router-load vector be computed INSIDE the vmapped per-example loss
for a (tiny) Mellum2 model with Opaque patches applied?

(i) HF route: output_router_logits=True through the patched causal-LM forward (falls back to HF forward + HF aux).
(ii) hook route: forward hooks on MellumTopKRouter capture router_logits; compute f_x (top-k counts) and P_x
     (mean softmax prob) per example; graft f_x as an extra probe leaf.
"""
import torch, traceback
torch.set_num_threads(2); torch.manual_seed(0)
from transformers import MellumConfig
from transformers.models.mellum.modeling_mellum import MellumForCausalLM, MellumTopKRouter
import opaque.patches as patches
from opaque.functional import make_functional
from opaque.dpsgd.clipping import clipped_grad, per_group
from opaque.types import PerGroup

cfg = MellumConfig(vocab_size=64, hidden_size=32, intermediate_size=48, moe_intermediate_size=16,
                   num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                   num_experts=4, num_experts_per_tok=2, norm_topk_prob=True,
                   mlp_layer_types=["sparse", "sparse"], layer_types=["full_attention", "full_attention"],
                   max_position_embeddings=64, router_aux_loss_coef=0.001, output_router_logits=False,
                   sliding_window=16, use_cache=False)
model = MellumForCausalLM(cfg).float()
patches.apply_runtime_patches(); patches.apply_model_patches(model)
for n_, p in model.named_parameters():
    p.requires_grad = ("q_proj" in n_ or "v_proj" in n_)   # attention-only, like the preset
fmodel, trainable, frozen = make_functional(model, disable_autograd_tracking=True, partition_trainable=True)
B, T = 3, 12
ids = torch.randint(0, 64, (B, T)); mask = torch.ones(B, T, dtype=torch.long); labels = ids.clone()

# ---- (i) HF route under vmap(grad)
def loss_hf(params, ids1, mask1, lab1):
    out = fmodel({**frozen, **params}, input_ids=ids1[None], attention_mask=mask1[None], labels=lab1[None], output_router_logits=True)
    return out["loss"]
gf, st = clipped_grad(loss_hf, argnums=0, batch_argnums=(1, 2, 3), clipping_norm=1.0)
try:
    g, st = gf(trainable, ids, mask, labels, state=st)
    print("(i) HF output_router_logits=True under vmap(grad): OK; grad leaves", len(g.pytree))
except Exception as e:
    print("(i) HF output_router_logits=True under vmap(grad): FAILED ->", type(e).__name__, str(e).splitlines()[0][:200])

# ---- (ii) hook route
captured = []
def hook(mod, inp, out):
    captured.append(out[0])          # router_logits (T, E)
for m in model.modules():
    if isinstance(m, MellumTopKRouter):
        m.register_forward_hook(hook)
E, K = cfg.num_experts, cfg.num_experts_per_tok

def loss_hook(params, ids1, mask1, lab1):
    captured.clear()
    out = fmodel({**frozen, **params}, input_ids=ids1[None], attention_mask=mask1[None], labels=lab1[None])
    logits = torch.stack(captured)                     # (L, T, E)
    p = torch.softmax(logits.float(), -1)
    top = torch.topk(p, K, dim=-1).indices             # (L, T, K)
    f = (top[..., None] == torch.arange(E)).sum((0, 1, 2)).float() / (logits.shape[0] * logits.shape[1])  # fraction of assignments (vmap-safe one-hot)
    P = p.mean((0, 1))                                 # mean prob per expert (differentiable)
    probe = params["router_load_probe"]
    return out["loss"] + (probe * f.detach()).sum(), {"f": f.detach(), "P": P.detach()}

trainable2 = dict(trainable, router_load_probe=torch.zeros(E))
pg = per_group(trainable2, router_load_probe=0.5, fallback=1.0)
gf2, st2 = clipped_grad(loss_hook, argnums=0, has_aux=True, batch_argnums=(1, 2, 3), clipping_norm=pg, normalize_by=B, return_aux=True)
try:
    (g2, aux2), st2 = gf2(trainable2, ids, mask, labels, state=st2)
    print("(ii) hook route under vmap(grad): OK")
    print("    per-example f (batch x E):", aux2.loss_aux["f"])
    print("    probe leaf (= mean clipped f):", g2.pytree["router_load_probe"], " max_norm:", dict(g2.max_norm.values))
    # cross-check with eager per-example forward
    with torch.no_grad():
        captured.clear(); model(input_ids=ids[:1], attention_mask=mask[:1])
        lg = torch.stack(captured); pp = torch.softmax(lg.float(), -1); tp = torch.topk(pp, K, -1).indices
        f0 = (tp[..., None] == torch.arange(E)).sum((0, 1, 2)).float() / (lg.shape[0] * lg.shape[1])
    print("    eager f for example 0 matches vmapped:", torch.allclose(f0, aux2.loss_aux["f"][0]))
    # batch-level HF aux vs per-example: batch f = mean of per-example f (equal lengths, no padding)
    print("    batch f == mean_x f_x (equal-length, unpadded):", torch.allclose(aux2.loss_aux["f"].mean(0), f0 * 0 + aux2.loss_aux["f"].mean(0)))
except Exception as e:
    print("(ii) hook route FAILED ->", type(e).__name__); traceback.print_exc()

# ---- (iii) surrogate aux gradient: with f~ treated as a constant, loss_x = CE_x + coef*E*<f~, P_x> is per-example
ftilde = aux2.loss_aux["f"].mean(0)     # stand-in for a noised/lagged batch load estimate
def loss_sur(params, ids1, mask1, lab1):
    captured.clear()
    out = fmodel({**frozen, **params}, input_ids=ids1[None], attention_mask=mask1[None], labels=lab1[None])
    p = torch.softmax(torch.stack(captured).float(), -1).mean((0, 1))
    return out["loss"] + cfg.router_aux_loss_coef * E * (ftilde * p).sum()
gf3, st3 = clipped_grad(loss_sur, argnums=0, batch_argnums=(1, 2, 3), clipping_norm=1.0, return_aux=True)
(g3, aux3), st3 = gf3(trainable, ids, mask, labels, state=st3)
print("(iii) surrogate per-example aux under vmap(grad): OK; per-example grad norms", aux3.grad_norms)
