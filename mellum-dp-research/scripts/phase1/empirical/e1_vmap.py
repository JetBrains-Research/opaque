"""E1 patched process: opaque clipped_grad (vmap) per-example grads vs unpatched loop reference."""
import sys, time, copy; sys.path.insert(0, "/tmp/claude-0/-home-user-opaque/a97cca5e-900f-5617-b2df-d73b55081cac/scratchpad/research/empirical")
from common import *
import torch
from opaque.functional import make_functional
from opaque.dpsgd.clipping import clipped_grad
import transformers.models.mellum.modeling_mellum as mod

ref = torch.load(OUT / "e1_ref.pt")
ids, mask = ref["ids"], ref["mask"]; B, T = ids.shape
model, _ = build_patched(seed=0)
model.load_state_dict(ref["state_dict"])
mod.load_balancing_loss_func = aux_loss_vmap_safe   # process-local: make mask path vmap-safe
model.router_aux_loss_coef = 0.0                    # loss == CE exactly; router logits still captured
set_trainable(model, "all")
out = {}

def run(m, tag):
    m.train()
    fmodel, trainable, frozen = make_functional(m, disable_autograd_tracking=True, partition_trainable=True)
    def per_ex(tr, fr, i, mk, lb):
        o = fmodel({**fr, **tr}, i, attention_mask=mk, labels=lb, output_router_logits=True)
        return o.loss, tuple(o.router_logits)
    t0 = time.time()
    # (a) the actual DP path: clipped_grad, no effective clipping, sum of per-example grads
    gfn, st = clipped_grad(per_ex, argnums=0, batch_argnums=(2, 3, 4), clipping_norm=1e9, has_aux=True, return_aux=True)
    (gsum, aux), _ = gfn(trainable, frozen, ids, mask, ids, state=st)
    t_cg = time.time() - t0
    # (b) per-example grads via torch.func.vmap(grad) (what clipped_grad does internally)
    t0 = time.time()
    pe = torch.func.vmap(torch.func.grad(per_ex, argnums=0, has_aux=True), in_dims=(None, None, 0, 0, 0))(trainable, frozen, ids, mask, ids)
    grads_pe, logits_pe = pe
    t_pe = time.time() - t0
    R = ref[tag]
    lg_ref = R["logits"]; lg_v = [l.float() for l in logits_pe]
    # route flips vmap-vs-loop, per layer (top-k set per token)
    flips = []; flip_rows = torch.zeros(B, T, dtype=torch.bool)
    for l in range(len(lg_ref)):
        fr, rows = route_flip_fraction(lg_v[l].reshape(-1, lg_v[l].shape[-1]), lg_ref[l].reshape(-1, lg_ref[l].shape[-1]), m.config.num_experts_per_tok)
        flips.append(fr); flip_rows |= rows.reshape(B, T)
    ex_with_flip = flip_rows.any(1)
    # per-param max relative L2 error (over examples) per-example
    per_param = {}
    for n, g in grads_pe.items():
        gr = R["loop_grads"][n]
        errs = torch.stack([rel_l2(g[b], gr[b]) * torch.ones(()) for b in range(B)])
        per_param[n] = {"max_over_examples": errs.max().item(), "mean_over_examples": errs.mean().item()}
    # whole-vector per-example error
    def flat(d, b): return torch.cat([d[n][b].reshape(-1).float() for n in sorted(d)])
    ex_err = torch.tensor([rel_l2(flat(grads_pe, b), flat(R["loop_grads"], b)) for b in range(B)])
    sum_err = {n: rel_l2(gsum.pytree[n], R["loop_grads"][n].sum(0)) for n in grads_pe}
    loss_err = (aux.loss_values.float() - R["loop_losses"]).abs().max().item()
    norms_v = aux.grad_norms.float()
    norms_ref = torch.tensor([flat(R["loop_grads"], b).norm().item() for b in range(B)])
    r = {"t_clipped_grad_s": t_cg, "t_vmap_grad_s": t_pe,
         "route_flip_frac_per_layer": flips, "n_tokens_flipped": int(flip_rows.sum()), "examples_with_flip": ex_with_flip.tolist(),
         "per_example_rel_l2": ex_err.tolist(), "max_per_param_rel_l2": max(v["max_over_examples"] for v in per_param.values()),
         "worst_param": max(per_param, key=lambda n: per_param[n]["max_over_examples"]), "per_param": per_param,
         "sum_rel_l2_max": max(sum_err.values()), "loss_max_abs_err": loss_err,
         "grad_norm_vmap": norms_v.tolist(), "grad_norm_ref": norms_ref.tolist(),
         "per_example_err_flip": ex_err[ex_with_flip].tolist(), "per_example_err_noflip": ex_err[~ex_with_flip].tolist()}
    print(f"[{tag}] clipped_grad {t_cg:.1f}s vmap(grad) {t_pe:.1f}s | route flip frac/layer {flips} tokens flipped {int(flip_rows.sum())}/{B*T} examples w/ flip {int(ex_with_flip.sum())}/{B}")
    print(f"[{tag}] per-example rel L2 err: {ex_err.tolist()}")
    print(f"[{tag}] max per-param rel L2 (worst {r['worst_param']}): {r['max_per_param_rel_l2']:.3e}; sum-vs-sum max {r['sum_rel_l2_max']:.3e}; loss max abs err {loss_err:.3e}")
    if ex_with_flip.any(): print(f"[{tag}] err w/ flip mean {ex_err[ex_with_flip].mean():.3e} vs w/o flip mean {ex_err[~ex_with_flip].mean():.3e}")
    return r, grads_pe

out["fp32"], g32 = run(model, "fp32")
mbf = copy.deepcopy(model).to(torch.bfloat16)
out["bf16"], gbf = run(mbf, "bf16")
# cross-precision: bf16 vmap vs fp32 loop; bf16 loop vs fp32 loop (how much is bf16 itself?)
def flat(d, b): return torch.cat([d[n][b].reshape(-1).float() for n in sorted(d)])
R32, Rbf = ref["fp32"]["loop_grads"], ref["bf16"]["loop_grads"]
out["cross"] = {"bf16_vmap_vs_fp32_loop": [rel_l2(flat(gbf, b), flat(R32, b)) for b in range(B)],
                "bf16_loop_vs_fp32_loop": [rel_l2(flat(Rbf, b), flat(R32, b)) for b in range(B)],
                "fp32_vmap_vs_fp32_loop": [rel_l2(flat(g32, b), flat(R32, b)) for b in range(B)]}
fl = []
for l in range(len(ref["fp32"]["logits"])):
    a = ref["bf16"]["logits"][l].reshape(-1, 8); b_ = ref["fp32"]["logits"][l].reshape(-1, 8)
    fl.append(route_flip_fraction(a, b_, 2)[0])
out["cross"]["route_flip_bf16loop_vs_fp32loop_per_layer"] = fl
print("cross:", {k: (v if isinstance(v, list) and len(v) < 4 else [round(x, 4) for x in v]) for k, v in out["cross"].items()})
dump("e1_results.json", out)
