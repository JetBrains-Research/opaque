"""Divergence experiment: Opaque vmap(grad) per-example path vs upstream HF Mellum.

Tiny Mellum2-shaped model (sliding+full layers, MoE every layer, norm_topk_prob=True).
All references are computed BEFORE any Opaque patch touches the modeling module.
"""
import copy, json, math, sys, time
import torch
torch.set_num_threads(2)
torch.manual_seed(0)

from transformers.models.mellum.configuration_mellum import MellumConfig
from transformers.models.mellum.modeling_mellum import (
    MellumForCausalLM, MellumTopKRouter, load_balancing_loss_func)

OUT = {}
def log(*a):
    print(*a, flush=True)

def make_cfg(**over):
    kw = dict(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
              num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
              pad_token_id=0, bos_token_id=1, eos_token_id=2, rope_theta=10000.0,
              num_experts=8, num_experts_per_tok=2, moe_intermediate_size=64,
              norm_topk_prob=True, layer_types=["sliding_attention", "full_attention"],
              sliding_window=8, router_aux_loss_coef=0.001, output_router_logits=False)
    kw.update(over)
    cfg = MellumConfig(**kw)
    cfg._attn_implementation = "sdpa"
    return cfg

B, T = 4, 16
cfg = make_cfg()
ref = MellumForCausalLM(cfg).train()
log("experts_implementation (default):", ref.config._experts_implementation)
log("rope_parameters:", ref.config.rope_parameters)
ids = torch.randint(3, cfg.vocab_size, (B, T))
mask = torch.ones(B, T, dtype=torch.long)
labels = ids.clone()
# different valid-token counts per example (prefix masked)
for i, k in enumerate([0, 3, 6, 9]):
    labels[i, :k] = -100
n_valid_shift = [(labels[i, 1:] != -100).sum().item() for i in range(B)]
log("valid shifted tokens per example:", n_valid_shift)

def names_trainable(m):
    return [n for n, p in m.named_parameters() if p.requires_grad]

def loop_grads(model, ids, mask, labels, experts_impl=None, output_router_logits=False):
    """Per-example HF reference: Python loop, batch of 1, model.loss (token-mean)."""
    if experts_impl is not None:
        model.config._experts_implementation_internal = experts_impl
    grads, losses, auxes = [], [], []
    for i in range(ids.shape[0]):
        model.zero_grad(set_to_none=True)
        out = model(input_ids=ids[i:i+1], attention_mask=mask[i:i+1], labels=labels[i:i+1],
                    output_router_logits=output_router_logits)
        out.loss.backward()
        grads.append({n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
                      for n, p in model.named_parameters() if p.requires_grad})
        losses.append(out.loss.detach().clone())
        auxes.append(None if out.aux_loss is None else out.aux_loss.detach().clone())
    return grads, torch.stack(losses), auxes

def batched_grads(model, ids, mask, labels, num_items_in_batch=None, output_router_logits=False):
    model.zero_grad(set_to_none=True)
    kw = {}
    if num_items_in_batch is not None:
        kw["num_items_in_batch"] = num_items_in_batch
    out = model(input_ids=ids, attention_mask=mask, labels=labels,
                output_router_logits=output_router_logits, **kw)
    out.loss.backward()
    g = {n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
         for n, p in model.named_parameters() if p.requires_grad}
    return g, out.loss.detach().clone(), out.aux_loss

def rel_l2(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    den = b.norm().item()
    return (a - b).norm().item() / den if den > 0 else (a - b).norm().item()

def compare(name, per_ex_test, per_ex_ref, top=6):
    """per_ex_test: dict name -> (B, ...) ; per_ex_ref: list of dict."""
    worst = []
    tot_num = 0.0; tot_den = 0.0
    for n in per_ex_ref[0]:
        for i in range(len(per_ex_ref)):
            a = per_ex_test[n][i]; b = per_ex_ref[i][n]
            tot_num += (a.float() - b.float()).pow(2).sum().item()
            tot_den += b.float().pow(2).sum().item()
            worst.append((rel_l2(a, b), n, i))
    worst.sort(reverse=True)
    overall = math.sqrt(tot_num / tot_den) if tot_den > 0 else float("nan")
    log(f"[{name}] overall rel-L2 = {overall:.3e}; worst per-param/example: "
        + "; ".join(f"{n}[{i}]={e:.2e}" for e, n, i in worst[:top]))
    return overall, worst[0][0]

# ---------------------------------------------------------------- fp32 references (unpatched)
t0 = time.time()
ref_eager, loss_eager, _ = loop_grads(copy.deepcopy(ref), ids, mask, labels, experts_impl="eager")
ref_grouped, loss_grouped, _ = loop_grads(copy.deepcopy(ref), ids, mask, labels, experts_impl="grouped_mm")
ref_batched, loss_batched, _ = loop_grads(copy.deepcopy(ref), ids, mask, labels, experts_impl="batched_mm")
log("fp32 refs done in %.1fs" % (time.time() - t0))
log("loop losses eager:", loss_eager.tolist())
e_vs_g = max(rel_l2(ref_eager[i][n], ref_grouped[i][n]) for i in range(B) for n in ref_eager[0])
e_vs_b = max(rel_l2(ref_eager[i][n], ref_batched[i][n]) for i in range(B) for n in ref_eager[0])
log(f"HF eager vs grouped_mm per-example grad max rel-L2: {e_vs_g:.2e}; eager vs batched_mm: {e_vs_b:.2e}")
OUT["hf_eager_vs_grouped_mm_fp32"] = e_vs_g
OUT["hf_eager_vs_batched_mm_fp32"] = e_vs_b

# HF Trainer-style batched loss: num_items_in_batch = count of valid shifted labels over the batch
N_tot = sum(n_valid_shift)
g_batch, loss_batch, _ = batched_grads(copy.deepcopy(ref), ids, mask, labels, num_items_in_batch=torch.tensor(N_tot))
# token-weighted combination of per-example token-mean grads
comb = {n: sum(ref_eager[i][n] * (n_valid_shift[i] / N_tot) for i in range(B)) for n in ref_eager[0]}
plain_mean = {n: sum(ref_eager[i][n] / B for i in range(B)) for n in ref_eager[0]}
err_comb = max(rel_l2(comb[n], g_batch[n]) for n in comb)
err_mean = max(rel_l2(plain_mean[n], g_batch[n]) for n in comb)
log(f"HF batched(num_items_in_batch) loss={loss_batch.item():.6f}; token-weighted per-example mean loss="
    f"{sum(loss_eager[i]*n_valid_shift[i] for i in range(B)).item()/N_tot:.6f}; plain mean of per-example losses={loss_eager.mean().item():.6f}")
log(f"grad(batched HF loss) vs token-weighted sum of per-example grads: max rel-L2 {err_comb:.2e}")
log(f"grad(batched HF loss) vs plain mean of per-example grads:        max rel-L2 {err_mean:.2e}")
OUT["batched_vs_token_weighted"] = err_comb; OUT["batched_vs_plain_mean"] = err_mean

# ---------------------------------------------------------------- aux loss (unpatched, fp32)
m_aux = copy.deepcopy(ref)
g_aux_batch, loss_aux_batch, aux_batch = batched_grads(m_aux, ids, mask, labels, output_router_logits=True)
_, _, aux_per_ex = loop_grads(copy.deepcopy(ref), ids, mask, labels, experts_impl="eager", output_router_logits=True)
aux_per_ex = torch.stack(aux_per_ex)
log(f"aux_batch = {aux_batch.item():.6f}; per-example aux = {aux_per_ex.tolist()}; mean = {aux_per_ex.mean().item():.6f}")
OUT["aux_batch"] = aux_batch.item(); OUT["aux_per_example"] = aux_per_ex.tolist()

# H2 identity: grad(aux_batch) == grad of surrogate with f_e(B) detached
m2 = copy.deepcopy(ref)
out = m2(input_ids=ids, attention_mask=mask, labels=labels, output_router_logits=True)
E, K = cfg.num_experts, cfg.num_experts_per_tok
gate_logits = out.router_logits
flat_mask = mask.reshape(-1).float()
tokens_per_expert = torch.zeros(E); prob_sum = torch.zeros(E); total_rows = 0.0
for lg in gate_logits:
    rw = torch.softmax(lg.float(), -1)
    _, sel = torch.topk(rw, K, -1)
    tokens_per_expert = tokens_per_expert + torch.zeros(E).scatter_add_(0, sel.reshape(-1), flat_mask.repeat_interleave(K))
    prob_sum = prob_sum + (rw * flat_mask[:, None]).sum(0)
    total_rows += flat_mask.sum()
f = (tokens_per_expert / total_rows).detach()          # constant load vector f(B)
surrogate = E * torch.sum(f * (prob_sum / total_rows))  # only P_e differentiable
aux_direct = load_balancing_loss_func(gate_logits, E, K, mask)
log(f"aux via load_balancing_loss_func={aux_direct.item():.6f}; surrogate (f detached)={surrogate.item():.6f}")
m2.zero_grad(set_to_none=True)
(cfg.router_aux_loss_coef * surrogate + out.loss.detach() * 0).backward()
g_sur = {n: p.grad.detach().clone() for n, p in m2.named_parameters() if p.grad is not None}
# grad of aux only from the batched run: g_aux_batch includes CE; recompute aux-only grad
m3 = copy.deepcopy(ref); out3 = m3(input_ids=ids, attention_mask=mask, labels=labels, output_router_logits=True)
m3.zero_grad(set_to_none=True); (cfg.router_aux_loss_coef * out3.aux_loss).backward()
g_aux_only = {n: p.grad.detach().clone() for n, p in m3.named_parameters() if p.grad is not None}
err_h2 = max(rel_l2(g_sur[n], g_aux_only[n]) for n in g_aux_only)
log(f"H2: grad(coef*aux_batch) vs grad(coef*surrogate with f detached): max rel-L2 {err_h2:.2e} over {len(g_aux_only)} params")
OUT["h2_surrogate_grad_err"] = err_h2
# which params does the aux gradient reach, and how big vs CE gradient?
ce_norm = math.sqrt(sum(g_batch[n].pow(2).sum().item() for n in g_batch))
aux_norm = math.sqrt(sum(g_aux_only[n].pow(2).sum().item() for n in g_aux_only))
attn_aux = math.sqrt(sum(g_aux_only[n].pow(2).sum().item() for n in g_aux_only if any(s in n for s in ("q_proj","k_proj","v_proj","o_proj"))))
attn_ce = math.sqrt(sum(g_batch[n].pow(2).sum().item() for n in g_batch if any(s in n for s in ("q_proj","k_proj","v_proj","o_proj"))))
log(f"||grad CE||={ce_norm:.4e}  ||grad coef*aux||={aux_norm:.4e}  ratio={aux_norm/ce_norm:.3e};  attention-proj only: CE {attn_ce:.4e} aux {attn_aux:.4e} ratio {attn_aux/attn_ce:.3e}")
OUT["aux_over_ce_grad_ratio"] = aux_norm / ce_norm; OUT["aux_over_ce_attn_ratio"] = attn_aux / attn_ce
reach = sorted({n.split(".")[-2] + "." + n.split(".")[-1] for n in g_aux_only if g_aux_only[n].abs().sum() > 0})
log("aux gradient reaches (module.param suffixes):", reach)

# ---------------------------------------------------------------- bf16 references (unpatched)
ref16 = copy.deepcopy(ref).to(torch.bfloat16)
ref16_loop, loss16_loop, _ = loop_grads(copy.deepcopy(ref16), ids, mask, labels, experts_impl="grouped_mm")
ref16_eager, loss16_eager, _ = loop_grads(copy.deepcopy(ref16), ids, mask, labels, experts_impl="eager")
e16 = max(rel_l2(ref16_loop[i][n], ref16_eager[i][n]) for i in range(B) for n in ref16_loop[0])
log(f"bf16 HF grouped_mm vs eager per-example grad max rel-L2: {e16:.2e}")
# bf16 vs fp32 HF (same loop) : intrinsic bf16 noise floor of the upstream path itself
e16_32 = compare("HF bf16 loop vs HF fp32 loop (upstream bf16 noise floor)",
                 {n: torch.stack([ref16_loop[i][n] for i in range(B)]) for n in ref16_loop[0]}, ref_eager)
OUT["hf_bf16_vs_fp32_loop"] = e16_32[0]
# route tables of the unpatched bf16 batched forward
def routes_of(model, ids, mask):
    out = model(input_ids=ids, attention_mask=mask, output_router_logits=True)
    return [torch.topk(lg.float(), K, -1).indices for lg in out.router_logits]
with torch.no_grad():
    routes16_batched = routes_of(copy.deepcopy(ref16), ids, mask)
    routes32_batched = routes_of(copy.deepcopy(ref), ids, mask)
def route_flips(ra, rb):
    flips = 0; tot = 0
    for a, b in zip(ra, rb):
        sa = [set(r.tolist()) for r in a]; sb = [set(r.tolist()) for r in b]
        flips += sum(x != y for x, y in zip(sa, sb)); tot += len(sa)
    return flips, tot
log("route flips bf16-batched vs fp32-batched (unpatched):", route_flips(routes16_batched, routes32_batched))

# ================================================================ Opaque patched path
from opaque.patches import apply_runtime_patches, apply_model_patches
from opaque.functional import make_functional
apply_runtime_patches(compat=True)

def vmap_grads(model, ids, mask, labels, extra=None):
    fmodel, trainable, frozen = make_functional(model, disable_autograd_tracking=True, partition_trainable=True)
    def per_example_loss(tr, fr, i, m, l):
        out = fmodel({**fr, **tr}, i, attention_mask=m, labels=l, **(extra or {}))
        aux = {"loss": out.loss.detach()}
        if getattr(out, "aux_loss", None) is not None:
            aux["aux_loss"] = out.aux_loss.detach()
        if getattr(out, "router_logits", None) is not None:
            aux["router_logits"] = tuple(r.detach() for r in out.router_logits)
        return out.loss, aux
    g, aux = torch.func.vmap(torch.func.grad(per_example_loss, has_aux=True),
                             in_dims=(None, None, 0, 0, 0))(trainable, frozen, ids, mask, labels)
    class _O: pass
    out = _O(); out.aux_loss = aux.get("aux_loss"); out.router_logits = aux.get("router_logits")
    return g, aux["loss"], out

results = {}
for label, model_src, extra_patch in [
    ("opaque dense fp32", ref, {"grouped_moe": False}),
    ("opaque grouped(auto) fp32", ref, {}),
]:
    m = copy.deepcopy(model_src)
    apply_model_patches(m, compat=True, performance=True, kernels=False, **extra_patch)
    g, loss, _ = vmap_grads(m, ids, mask, labels)
    log(f"{label}: vmap per-example losses {loss.tolist()}  (loop eager {loss_eager.tolist()})")
    ov, wo = compare(f"{label} vs HF eager loop", g, ref_eager)
    ov2, _ = compare(f"{label} vs HF grouped_mm loop", g, ref_grouped)
    results[label] = {"vs_eager": ov, "worst": wo, "vs_grouped": ov2, "loss_err": rel_l2(loss, loss_eager)}
OUT["fp32_opaque"] = results

# force Opaque_GroupedMoE (CPU torch._grouped_mm) by using >=16 experts on a separate model
cfg16 = make_cfg(num_experts=16, num_experts_per_tok=4)
torch.manual_seed(1)
ref_e16 = MellumForCausalLM(cfg16).train()
r_e16, l_e16, _ = loop_grads(copy.deepcopy(ref_e16), ids, mask, labels, experts_impl="eager")
from opaque.api.patches.kernels import moe as moe_mod, _grouped_moe as gmoe_mod
calls = {"dense": 0, "grouped": 0, "flags": []}
_orig_dense = moe_mod._moe_backward; _orig_grouped = gmoe_mod._fused_moe_backward
def _spy_dense(*a, **k):
    calls["dense"] += 1; calls["flags"].append(("dense", k.get("compute_gate_wgrad"), k.get("compute_down_wgrad"))); return _orig_dense(*a, **k)
def _spy_grouped(*a, **k):
    calls["grouped"] += 1; calls["flags"].append(("grouped", k.get("compute_gate_wgrad"), k.get("compute_down_wgrad"))); return _orig_grouped(*a, **k)
moe_mod._moe_backward = _spy_dense; gmoe_mod._fused_moe_backward = _spy_grouped
m = copy.deepcopy(ref_e16); apply_model_patches(m, compat=True, performance=True, kernels=False)
g, loss, _ = vmap_grads(m, ids, mask, labels)
log(f"16-expert model dispatch counts: {calls['dense']} dense / {calls['grouped']} grouped backward calls; flags(sample)={calls['flags'][:2]}")
ov, wo = compare("opaque 16-expert (grouped path) fp32 vs HF eager loop", g, r_e16)
OUT["fp32_grouped16"] = {"overall": ov, "worst": wo, "dispatch": {k: calls[k] for k in ("dense","grouped")}}

# ---------------------------------------------------------------- frozen experts/router: which grads + flags
calls["dense"] = calls["grouped"] = 0; calls["flags"] = []
m = copy.deepcopy(ref); apply_model_patches(m, compat=True, performance=True, kernels=False)
for n, p in m.named_parameters():
    p.requires_grad_(any(s in n for s in ("q_proj", "k_proj", "v_proj", "o_proj")))
g, loss, _ = vmap_grads(m, ids, mask, labels)
log("attention-only trainable: per-example grad keys:", sorted(g.keys()))
log("MoE backward flags with frozen experts:", calls["flags"])
sub = {n: g[n] for n in g}
ov, wo = compare("opaque attention-only fp32 vs HF eager loop (same params)", sub, [{n: r[n] for n in sub} for r in ref_eager])
OUT["frozen_experts_flags"] = calls["flags"]; OUT["attention_only_err"] = ov

# ---------------------------------------------------------------- aux loss under vmap (patched)
m = copy.deepcopy(ref); apply_model_patches(m, compat=True, performance=True, kernels=False)
try:
    g, loss, out = vmap_grads(m, ids, mask, labels, extra={"output_router_logits": True})
    log("vmap with output_router_logits=True: per-example aux =", out.aux_loss.tolist(), " loop per-example aux =", aux_per_ex.tolist(),
        " router_logits shapes:", [tuple(r.shape) for r in out.router_logits])
    OUT["vmap_aux_per_example"] = out.aux_loss.tolist()
    OUT["vmap_aux_matches_loop"] = rel_l2(out.aux_loss, aux_per_ex)
except Exception as e:  # noqa: BLE001
    log("vmap with output_router_logits=True FAILED:", type(e).__name__, str(e)[:300])
    OUT["vmap_aux_error"] = f"{type(e).__name__}: {str(e)[:300]}"

# ---------------------------------------------------------------- bf16 Opaque vs bf16 HF loop, route flips
# also test the no-attention-mask branch of the aux loss under vmap (bincount path)
m = copy.deepcopy(ref); apply_model_patches(m, compat=True, performance=True, kernels=False)
try:
    fmodel, trainable, frozen = make_functional(m, disable_autograd_tracking=True, partition_trainable=True)
    f = lambda tr, fr, i, l: fmodel({**fr, **tr}, i, labels=l, output_router_logits=True).loss
    torch.func.vmap(torch.func.grad(f), in_dims=(None, None, 0, 0))(trainable, frozen, ids, labels)
    log("vmap with output_router_logits=True and NO attention_mask: OK")
    OUT["vmap_aux_nomask"] = "ok"
except Exception as e:  # noqa: BLE001
    log("vmap with output_router_logits=True and NO attention_mask FAILED:", type(e).__name__, str(e)[:200])
    OUT["vmap_aux_nomask_error"] = f"{type(e).__name__}: {str(e)[:200]}"

def vmap_routes(model, ids, mask):
    backbone = model.model
    params = dict(backbone.named_parameters()); bufs = dict(backbone.named_buffers())
    def f(i, m):
        out = torch.func.functional_call(backbone, {**params, **bufs}, (), {"input_ids": i[None], "attention_mask": m[None], "output_router_logits": True})
        return tuple(r for r in out.router_logits)
    with torch.no_grad():
        rl = torch.func.vmap(f, in_dims=(0, 0))(ids, mask)
    return [torch.topk(r.float(), K, -1).indices.reshape(-1, K) for r in rl]

m16 = copy.deepcopy(ref16); apply_model_patches(m16, compat=True, performance=True, kernels=False, grouped_moe=False)
g16, loss16, _ = vmap_grads(m16, ids, mask, labels)
routes_vmap16 = vmap_routes(m16, ids, mask)
ov, wo = compare("opaque dense bf16 vmap vs HF grouped_mm bf16 loop", g16, ref16_loop)
ov2, _ = compare("opaque dense bf16 vmap vs HF eager bf16 loop", g16, ref16_eager)
ov3, _ = compare("opaque dense bf16 vmap vs HF fp32 eager loop", g16, ref_eager)
OUT["bf16_opaque_vs_hf_bf16_loop"] = ov; OUT["bf16_opaque_vs_hf_bf16_eager_loop"] = ov2; OUT["bf16_opaque_vs_hf_fp32"] = ov3
log("route flips: opaque-bf16-vmap vs HF-bf16-batched:", route_flips(routes_vmap16, routes16_batched),
    "; opaque-bf16-vmap vs fp32:", route_flips(routes_vmap16, routes32_batched))
OUT["route_flips_bf16_vmap_vs_hf_bf16"] = route_flips(routes_vmap16, routes16_batched)
# per-example: does grad error correlate with route flips?
for i in range(B):
    fl = sum(int(set(a[t].tolist()) != set(b[t].tolist())) for a, b in zip(routes_vmap16, routes16_batched) for t in range(i*T, (i+1)*T))
    err = math.sqrt(sum((g16[n][i].float()-ref16_loop[i][n].float()).pow(2).sum().item() for n in g16) /
                    sum(ref16_loop[i][n].float().pow(2).sum().item() for n in g16))
    log(f"  example {i}: route flips={fl}, rel-L2 grad err vs HF bf16 loop={err:.3e}")

# ---------------------------------------------------------------- SDPA numerics: vmap vs batched, bf16 and fp32
q = torch.randn(B, 4, T, 16); k = torch.randn(B, 2, T, 16); v = torch.randn(B, 2, T, 16)
for dt in (torch.float32, torch.bfloat16):
    qq, kk, vv = q.to(dt), k.to(dt), v.to(dt)
    kk_r = kk.repeat_interleave(2, dim=1); vv_r = vv.repeat_interleave(2, dim=1)
    batched = torch.nn.functional.scaled_dot_product_attention(qq, kk_r, vv_r, is_causal=True)
    vm = torch.func.vmap(lambda a, b, c: torch.nn.functional.scaled_dot_product_attention(a, b, c, is_causal=True))(qq, kk_r, vv_r)
    ref64 = torch.nn.functional.scaled_dot_product_attention(q.double(), kk_r.double(), vv_r.double(), is_causal=True)
    log(f"SDPA {dt}: vmap vs batched rel-L2 {rel_l2(vm, batched):.2e}; batched vs fp64 {rel_l2(batched, ref64):.2e}; vmap vs fp64 {rel_l2(vm, ref64):.2e}")
    OUT[f"sdpa_vmap_vs_batched_{str(dt).split('.')[-1]}"] = rel_l2(vm, batched)

json.dump(OUT, open(sys.argv[1] if len(sys.argv) > 1 else "/dev/null", "w"), indent=1, default=str)
log("DONE")
