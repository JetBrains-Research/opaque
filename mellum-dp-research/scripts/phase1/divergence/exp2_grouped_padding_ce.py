"""Exp2: grouped-GEMM CPU path, kernel-level bf16 attribution, padding masks, no-mask band path,
gradient checkpointing, chunked linear-CE, HF bf16 batched-vs-loop coupling."""
import copy, json, math, sys, time
import torch
torch.set_num_threads(2)
torch.manual_seed(0)
from transformers.models.mellum.configuration_mellum import MellumConfig
from transformers.models.mellum.modeling_mellum import MellumForCausalLM
from transformers.integrations.moe import grouped_mm_experts_forward, batched_mm_experts_forward
OUT = {}
def log(*a): print(*a, flush=True)
def rel_l2(a, b):
    a = a.float().flatten(); b = b.float().flatten(); den = b.norm().item()
    return (a - b).norm().item() / den if den > 0 else (a - b).norm().item()

def make_cfg(**over):
    kw = dict(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
              num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
              pad_token_id=0, bos_token_id=1, eos_token_id=2, num_experts=16, num_experts_per_tok=4,
              moe_intermediate_size=64, norm_topk_prob=True,
              layer_types=["sliding_attention", "full_attention"], sliding_window=8)
    kw.update(over); cfg = MellumConfig(**kw); cfg._attn_implementation = "sdpa"; return cfg

B, T, K = 4, 16, 4
cfg = make_cfg(); ref = MellumForCausalLM(cfg).train()
ids = torch.randint(3, cfg.vocab_size, (B, T)); labels = ids.clone()
mask = torch.ones(B, T, dtype=torch.long)
# right padding for ex 1 (3 pads), left padding for ex 2 (4 pads)
mask_pad = mask.clone(); mask_pad[1, -3:] = 0; mask_pad[2, :4] = 0
labels_pad = labels.clone(); labels_pad[mask_pad == 0] = -100

def loop_grads(model, ids, mask, labels, impl="eager", **fw):
    model.config._experts_implementation_internal = impl
    grads, losses = [], []
    for i in range(ids.shape[0]):
        model.zero_grad(set_to_none=True)
        kw = dict(input_ids=ids[i:i+1], labels=labels[i:i+1], **fw)
        if mask is not None: kw["attention_mask"] = mask[i:i+1]
        out = model(**kw); out.loss.backward()
        grads.append({n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for n, p in model.named_parameters() if p.requires_grad})
        losses.append(out.loss.detach().clone())
    return grads, torch.stack(losses)

def batched_grads(model, ids, mask, labels, impl="eager"):
    model.config._experts_implementation_internal = impl
    model.zero_grad(set_to_none=True)
    out = model(input_ids=ids, attention_mask=mask, labels=labels)
    out.loss.backward()
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}

def compare(name, g, refs):
    num = den = 0.0; worst = (0, "")
    for n in refs[0]:
        for i in range(len(refs)):
            d = (g[n][i].float() - refs[i][n].float()).pow(2).sum().item(); num += d; den += refs[i][n].float().pow(2).sum().item()
            e = rel_l2(g[n][i], refs[i][n]); worst = max(worst, (e, f"{n}[{i}]"))
    ov = math.sqrt(num / den); log(f"[{name}] overall rel-L2 {ov:.3e}; worst {worst[1]}={worst[0]:.2e}"); return ov

# --- refs (unpatched) fp32
r_full, l_full = loop_grads(copy.deepcopy(ref), ids, mask, labels)
r_pad, l_pad = loop_grads(copy.deepcopy(ref), ids, mask_pad, labels_pad)
r_nomask, l_nomask = loop_grads(copy.deepcopy(ref), ids, None, labels)
log("ref grad norm (sanity):", math.sqrt(sum(r_full[0][n].pow(2).sum().item() for n in r_full[0])))
# --- refs bf16
ref16 = copy.deepcopy(ref).to(torch.bfloat16)
r16_loop, _ = loop_grads(copy.deepcopy(ref16), ids, mask, labels, impl="grouped_mm")
g16_batched = batched_grads(copy.deepcopy(ref16), ids, mask, labels, impl="grouped_mm")
# HF bf16 batched vs HF bf16 loop: upstream's own batch-composition numerics (loss is token-mean over batch, so compare direction only)
r16_loop_batched_like = batched_grads(copy.deepcopy(ref16), ids, mask, labels, impl="grouped_mm")  # same call, determinism check
log("HF bf16 batched determinism (same call twice) max rel-L2:", max(rel_l2(g16_batched[n], r16_loop_batched_like[n]) for n in g16_batched))
# compare batched grad with per-example combination: valid tokens all equal (15 each) so batch loss = mean of per-example losses
comb16 = {n: sum(r16_loop[i][n].float() for i in range(B)) / B for n in r16_loop[0]}
log("HF bf16 batched(B=4) vs mean of HF bf16 loop(B=1): max rel-L2 %.2e" % max(rel_l2(g16_batched[n], comb16[n]) for n in comb16))
r32_batched = batched_grads(copy.deepcopy(ref), ids, mask, labels, impl="grouped_mm")
comb32 = {n: sum(r_full[i][n] for i in range(B)) / B for n in r_full[0]}
log("HF fp32 batched(B=4) vs mean of HF fp32 loop(B=1): max rel-L2 %.2e" % max(rel_l2(r32_batched[n], comb32[n]) for n in comb32))
OUT["hf_bf16_batched_vs_loop"] = max(rel_l2(g16_batched[n], comb16[n]) for n in comb16)

# --- kernel-level bf16 attribution: HF experts forward vs opaque_moe on identical inputs
from opaque.api.patches.kernels.moe import opaque_moe
E, H, I = cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size
exp_mod = ref16.model.layers[0].mlp.experts
for dt in (torch.float32, torch.bfloat16):
    torch.manual_seed(3)
    x = torch.randn(B * T, H, dtype=dt, requires_grad=True)
    W1 = torch.randn(E, 2 * I, H, dtype=dt) * 0.05; W2 = torch.randn(E, H, I, dtype=dt) * 0.05
    W1.requires_grad_(True); W2.requires_grad_(True)
    idx = torch.stack([torch.randperm(E)[:K] for _ in range(B * T)])
    tw = torch.softmax(torch.randn(B * T, K), -1).to(dt).requires_grad_(True)
    class Exp: pass
    ex = Exp(); ex.gate_up_proj = W1; ex.down_proj = W2; ex.num_experts = E; ex.has_gate = True; ex.has_bias = False; ex.is_transposed = False
    ex.act_fn = torch.nn.functional.silu; ex._apply_gate = lambda t: torch.nn.functional.silu(t[:, :I]) * t[:, I:]
    y_hf = grouped_mm_experts_forward(ex, x, idx, tw); go = torch.randn_like(y_hf)
    gx_hf, gW1_hf, gW2_hf, gtw_hf = torch.autograd.grad(y_hf, (x, W1, W2, tw), go)
    y_bm = batched_mm_experts_forward(ex, x, idx, tw)
    y_op = opaque_moe(x, W1, W2, idx, tw, grouped=False)
    gx_op, gW1_op, gW2_op, gtw_op = torch.autograd.grad(y_op, (x, W1, W2, tw), go)
    # fp64 oracle
    x64, W164, W264, tw64 = (t.detach().double().requires_grad_(True) for t in (x, W1, W2, tw))
    def eager_experts(x_, W1_, W2_, idx_, tw_):
        out = torch.zeros_like(x_)
        for e in range(E):
            sel = (idx_ == e)
            tok, pos = torch.where(sel)
            if tok.numel() == 0: continue
            gu = torch.nn.functional.linear(x_[tok], W1_[e]); h = torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]
            out = out.index_add(0, tok, torch.nn.functional.linear(h, W2_[e]) * tw_[tok, pos, None])
        return out
    y64 = eager_experts(x64, W164, W264, idx, tw64)
    gx64, gW164, gW264, gtw64 = torch.autograd.grad(y64, (x64, W164, W264, tw64), go.double())
    log(f"MoE kernel {dt}: fwd hf-vs-opaque {rel_l2(y_op, y_hf):.2e} | hf-vs-fp64 {rel_l2(y_hf, y64):.2e} opaque-vs-fp64 {rel_l2(y_op, y64):.2e} batched_mm-vs-fp64 {rel_l2(y_bm, y64):.2e}")
    log(f"   grads dx: hf-vs-op {rel_l2(gx_op, gx_hf):.2e} (hf/op vs fp64 {rel_l2(gx_hf, gx64):.2e}/{rel_l2(gx_op, gx64):.2e}); "
        f"dW1: {rel_l2(gW1_op, gW1_hf):.2e} ({rel_l2(gW1_hf, gW164):.2e}/{rel_l2(gW1_op, gW164):.2e}); "
        f"dW2: {rel_l2(gW2_op, gW2_hf):.2e} ({rel_l2(gW2_hf, gW264):.2e}/{rel_l2(gW2_op, gW264):.2e}); "
        f"dtw: {rel_l2(gtw_op, gtw_hf):.2e} ({rel_l2(gtw_hf, gtw64):.2e}/{rel_l2(gtw_op, gtw64):.2e})")
    OUT[f"kernel_{str(dt).split('.')[-1]}"] = dict(fwd=rel_l2(y_op, y_hf), hf_vs_64=rel_l2(y_hf, y64), op_vs_64=rel_l2(y_op, y64),
        dx=rel_l2(gx_op, gx_hf), dW1=rel_l2(gW1_op, gW1_hf), dW2=rel_l2(gW2_op, gW2_hf), dtw=rel_l2(gtw_op, gtw_hf),
        dW1_hf64=rel_l2(gW1_hf, gW164), dW1_op64=rel_l2(gW1_op, gW164))

# ================================================================ Opaque (grouped path forced ON at first patch)
from opaque.patches import apply_runtime_patches, apply_model_patches
from opaque.functional import make_functional
from opaque.api.patches.kernels import moe as moe_mod, _grouped_moe as gmoe_mod
calls = {"dense": 0, "grouped": 0}
_od = moe_mod._moe_backward; _og = gmoe_mod._fused_moe_backward
moe_mod._moe_backward = lambda *a, **k: (calls.__setitem__("dense", calls["dense"] + 1), _od(*a, **k))[1]
gmoe_mod._fused_moe_backward = lambda *a, **k: (calls.__setitem__("grouped", calls["grouped"] + 1), _og(*a, **k))[1]
log("torch._grouped_mm available:", hasattr(torch, "_grouped_mm"), " F.grouped_mm:", hasattr(torch.nn.functional, "grouped_mm"))
apply_runtime_patches(compat=True)

def vmap_grads(model, ids, mask, labels, extra=None):
    fmodel, trainable, frozen = make_functional(model, disable_autograd_tracking=True, partition_trainable=True)
    def f(tr, fr, i, m, l):
        kw = {"labels": l, **(extra or {})}
        if m is not None: kw["attention_mask"] = m
        out = fmodel({**fr, **tr}, i, **kw); return out.loss, out.loss.detach()
    if mask is None:
        g, loss = torch.func.vmap(torch.func.grad(lambda tr, fr, i, l: f(tr, fr, i, None, l), has_aux=True), in_dims=(None, None, 0, 0))(trainable, frozen, ids, labels)
    else:
        g, loss = torch.func.vmap(torch.func.grad(f, has_aux=True), in_dims=(None, None, 0, 0, 0))(trainable, frozen, ids, mask, labels)
    return g, loss

m = copy.deepcopy(ref); apply_model_patches(m, compat=True, performance=True, kernels=True, grouped_moe=True)
g, loss = vmap_grads(m, ids, mask, labels)
log(f"dispatch counts after grouped run: {calls}")
OUT["grouped_fp32"] = compare("opaque GROUPED (torch._grouped_mm) fp32 vmap vs HF eager loop", g, r_full)
OUT["grouped_dispatch"] = dict(calls)
g, loss = vmap_grads(m, ids, mask_pad, labels_pad)
log("padded losses vmap:", loss.tolist(), " loop:", l_pad.tolist())
OUT["padding_fp32"] = compare("opaque fp32 vmap vs HF loop with right+left padding (sliding+full layers)", g, r_pad)
g, loss = vmap_grads(m, ids, None, labels)
OUT["nomask_fp32"] = compare("opaque fp32 vmap vs HF loop with attention_mask=None (Boolean SDPA band path)", g, r_nomask)
# gradient checkpointing
m_gc = copy.deepcopy(ref); apply_model_patches(m_gc, compat=True, performance=True, kernels=True, grouped_moe=True)
m_gc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
try:
    g, loss = vmap_grads(m_gc, ids, mask, labels)
    OUT["gc_fp32"] = compare("opaque fp32 vmap + HF gradient checkpointing vs HF loop", g, r_full)
except Exception as e:  # noqa: BLE001
    log("gradient checkpointing under vmap FAILED:", type(e).__name__, str(e)[:200]); OUT["gc_error"] = str(e)[:200]
# chunked linear CE (Mellum: chunked_linear_cross_entropy=2048, opt-in via fused_linear_cross_entropy=True)
m_ce = copy.deepcopy(ref); apply_model_patches(m_ce, compat=True, performance=True, kernels=True, grouped_moe=True, fused_linear_cross_entropy=True)
g, loss = vmap_grads(m_ce, ids, mask, labels, extra={"opaque_fused_loss_only": True})
log("chunked-CE losses vmap:", loss.tolist(), " loop:", l_full.tolist())
OUT["chunked_ce_fp32"] = compare("opaque fp32 vmap + chunked linear CE vs HF loop", g, r_full)
m_ce16 = copy.deepcopy(ref16); apply_model_patches(m_ce16, compat=True, performance=True, kernels=True, grouped_moe=True, fused_linear_cross_entropy=True)
g16c, _ = vmap_grads(m_ce16, ids, mask, labels, extra={"opaque_fused_loss_only": True})
m16 = copy.deepcopy(ref16); apply_model_patches(m16, compat=True, performance=True, kernels=True, grouped_moe=True)
g16, _ = vmap_grads(m16, ids, mask, labels)
OUT["bf16_grouped_vs_hf_loop"] = compare("opaque GROUPED bf16 vmap vs HF grouped_mm bf16 loop", g16, r16_loop)
OUT["bf16_chunkedce_vs_hf_loop"] = compare("opaque GROUPED bf16 vmap + chunked CE vs HF grouped_mm bf16 loop", g16c, r16_loop)
OUT["bf16_grouped_vs_fp32"] = compare("opaque GROUPED bf16 vmap vs HF fp32 loop", g16, r_full)
OUT["hf_bf16_loop_vs_fp32"] = compare("HF bf16 loop vs HF fp32 loop (floor)", {n: torch.stack([r16_loop[i][n] for i in range(B)]) for n in r16_loop[0]}, r_full)
log("dispatch counts total:", calls)
json.dump(OUT, open(sys.argv[1], "w"), indent=1, default=str); log("DONE")
