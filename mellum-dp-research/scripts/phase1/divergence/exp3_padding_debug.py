import copy, math, json, sys, torch
torch.set_num_threads(2); torch.manual_seed(0)
from transformers.models.mellum.configuration_mellum import MellumConfig
from transformers.models.mellum.modeling_mellum import MellumForCausalLM
def log(*a): print(*a, flush=True)
def rel_l2(a, b):
    a = a.float().flatten(); b = b.float().flatten(); den = b.norm().item()
    return (a - b).norm().item() / den if den > 0 else (a - b).norm().item()
def make_cfg(**over):
    kw = dict(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
              max_position_embeddings=128, pad_token_id=0, bos_token_id=1, eos_token_id=2, num_experts=8, num_experts_per_tok=2,
              moe_intermediate_size=64, norm_topk_prob=True, layer_types=["sliding_attention", "full_attention"], sliding_window=8)
    kw.update(over); cfg = MellumConfig(**kw); cfg._attn_implementation = over.get("attn", "sdpa"); return cfg
B, T = 4, 16
OUT = {}
for attn in ("sdpa", "eager"):
  for pad_side in ("right", "left"):
    cfg = make_cfg(attn=attn); ref = MellumForCausalLM(cfg).train()
    ids = torch.randint(3, cfg.vocab_size, (B, T)); mask = torch.ones(B, T, dtype=torch.long)
    if pad_side == "right": mask[1, -3:] = 0; mask[2, -6:] = 0
    else: mask[1, :3] = 0; mask[2, :6] = 0
    labels = ids.clone(); labels[mask == 0] = -100
    # HF loop (B=1) and HF batched (B=4) references, fp32, unpatched
    def hf_loop(model):
        gs, ls, lg = [], [], []
        for i in range(B):
            model.zero_grad(set_to_none=True)
            out = model(input_ids=ids[i:i+1], attention_mask=mask[i:i+1], labels=labels[i:i+1]); out.loss.backward()
            gs.append({n: p.grad.detach().clone() for n, p in model.named_parameters()}); ls.append(out.loss.item()); lg.append(out.logits.detach()[0])
        return gs, ls, lg
    g_loop, l_loop, logits_loop = hf_loop(copy.deepcopy(ref))
    mb = copy.deepcopy(ref); outb = mb(input_ids=ids, attention_mask=mask, labels=labels)
    logits_b = outb.logits.detach()
    # per-example loss from batched logits
    l_b = []
    for i in range(B):
        sl = logits_b[i, :-1]; tl = labels[i, 1:]
        l_b.append(torch.nn.functional.cross_entropy(sl, tl, ignore_index=-100).item())
    log(f"--- attn={attn} pad={pad_side}: HF loop losses {['%.5f'%x for x in l_loop]} | HF batched per-example {['%.5f'%x for x in l_b]}")
    valid_logit_err = [rel_l2(logits_b[i][mask[i]==1], logits_loop[i][mask[i]==1]) for i in range(B)]
    log("   HF batched vs HF loop logits at VALID positions rel-L2:", ["%.1e"%e for e in valid_logit_err])
    # patched
    from opaque.patches import apply_runtime_patches, apply_model_patches
    from opaque.functional import make_functional
    apply_runtime_patches(compat=True)
    m = copy.deepcopy(ref); apply_model_patches(m, compat=True, performance=True, kernels=False)
    fmodel, tr, fr = make_functional(m, disable_autograd_tracking=True, partition_trainable=True)
    def f(tr, fr, i, mk, l):
        out = fmodel({**fr, **tr}, i, attention_mask=mk, labels=l); return out.loss, (out.loss.detach(), out.logits.detach())
    g, (lv, logits_v) = torch.func.vmap(torch.func.grad(f, has_aux=True), in_dims=(None, None, 0, 0, 0))(tr, fr, ids, mask, labels)
    log("   opaque vmap losses:", ["%.5f"%x for x in lv.tolist()], " finite:", bool(torch.isfinite(logits_v).all()))
    log("   opaque vmap vs HF loop logits at VALID positions rel-L2:", ["%.1e"%rel_l2(logits_v[i][mask[i]==1], logits_loop[i][mask[i]==1]) for i in range(B)])
    log("   opaque vmap vs HF loop logits at PAD positions rel-L2:", ["%.1e"%rel_l2(logits_v[i][mask[i]==0], logits_loop[i][mask[i]==0]) if (mask[i]==0).any() else "-" for i in range(B)])
    log("   HF loop logits at PAD positions finite:", [bool(torch.isfinite(logits_loop[i][mask[i]==0]).all()) if (mask[i]==0).any() else "-" for i in range(B)],
        " max|logit| pad:", ["%.1e"%logits_loop[i][mask[i]==0].abs().max().item() if (mask[i]==0).any() else "-" for i in range(B)])
    per_ex_err = []
    for i in range(B):
        num = sum((g[n][i] - g_loop[i][n]).pow(2).sum().item() for n in g); den = sum(g_loop[i][n].pow(2).sum().item() for n in g)
        per_ex_err.append(math.sqrt(num / den))
    log("   opaque vmap vs HF loop per-example grad rel-L2:", ["%.1e"%e for e in per_ex_err])
    # also: opaque model run NON-vmapped batched, vs HF batched (mask path w/o vmap)
    mp = copy.deepcopy(m); outp = mp(input_ids=ids, attention_mask=mask, labels=labels)
    log("   opaque patched batched(B=4) vs HF batched logits at valid positions:", ["%.1e"%rel_l2(outp.logits.detach()[i][mask[i]==1], logits_b[i][mask[i]==1]) for i in range(B)])
    OUT[f"{attn}/{pad_side}"] = dict(loop=l_loop, batched=l_b, vmap=lv.tolist(), grad_err=per_ex_err)
    # reset family patches for next config (module globals): reload not possible; eager patch persists but is semantics-preserving
json.dump(OUT, open(sys.argv[1], "w"), indent=1); log("DONE")
