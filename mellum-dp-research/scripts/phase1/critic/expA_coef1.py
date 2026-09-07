"""Exp A: does HF Trainer scale the MoE aux loss by gradient_accumulation_steps for Mellum?
Tiny random-init Mellum, CPU, fp32. Compare the accumulated .grad after one optimizer step
(gradient_accumulation_steps=2, per_device_train_batch_size=2, 4 examples) against
manual references: (R1) CE_tokenmean(all 4) + coef*(aux(mb1)+aux(mb2))   [sum of per-microbatch aux]
                   (R2) CE_tokenmean(all 4) + coef*mean(aux(mb1),aux(mb2)) [logical-batch-consistent scaling]
                   (R3) CE_tokenmean(all 4) + coef*aux(all 4)              [true logical-batch aux]
"""
import torch, math, json, sys
torch.set_num_threads(2)
from transformers import MellumConfig, MellumForCausalLM, Trainer, TrainingArguments, TrainerCallback
from torch.utils.data import SequentialSampler
class SeqTrainer(Trainer):
    def _get_train_sampler(self, *a, **k):
        return SequentialSampler(self.train_dataset)

torch.manual_seed(0)
cfg = MellumConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                   num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_experts=8,
                   num_experts_per_tok=2, moe_intermediate_size=32, max_position_embeddings=256,
                   output_router_logits=True, router_aux_loss_coef=1.0, attn_implementation="eager")
for k in ("mlp_layer_types",):
    pass
model = MellumForCausalLM(cfg).float()
model.config.use_cache = False
B, T = 4, 16
ids = torch.randint(3, 128, (B, T))
# ragged lengths via labels -100 tail and attention mask
attn = torch.ones(B, T, dtype=torch.long)
labels = ids.clone()
for b, n in enumerate([16, 14, 12, 10]):
    attn[b, n:] = 0; labels[b, n:] = -100; ids[b, n:] = 0
ds = [{"input_ids": ids[b], "attention_mask": attn[b], "labels": labels[b]} for b in range(B)]

class Cap(TrainerCallback):
    def __init__(self): self.grads=None
    def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
        self.grads = {n: p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
        return control
cap = Cap()
args = TrainingArguments(output_dir="/tmp/claude-0/-home-user-opaque/a97cca5e-900f-5617-b2df-d73b55081cac/scratchpad/research/critic/hf_out",
                         per_device_train_batch_size=2, gradient_accumulation_steps=2, max_steps=1,
                         learning_rate=0.0, report_to=[], logging_steps=1, dataloader_num_workers=0,
                         remove_unused_columns=False, seed=0, use_cpu=True, max_grad_norm=0, save_strategy="no")
def collate(feats):
    return {k: torch.stack([f[k] for f in feats]) for k in feats[0]}
tr = SeqTrainer(model=model, args=args, train_dataset=ds, data_collator=collate, callbacks=[cap])
print("model_accepts_loss_kwargs:", tr.model_accepts_loss_kwargs)
tr.train()
g_hf = cap.grads
order = [ds[i] for i in range(B)]  # Trainer with seed 0 shuffles; recover order from the sampler
# Recover the actual microbatch order the Trainer used
dl = tr.get_train_dataloader()
batches = [collate([ds[i] for i in idx]) if False else b for b in dl]
mbs = list(dl)
assert len(mbs) == 2, len(mbs)
allb = {k: torch.cat([mb[k] for mb in mbs]) for k in mbs[0]}
def refgrad(fn):
    model.zero_grad(set_to_none=True)
    fn().backward()
    return {n: p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
def ce_sum(mb):
    out = model(**{k:v for k,v in mb.items() if k!="labels"}, output_router_logits=True)
    lab = torch.nn.functional.pad(mb["labels"], (0,1), value=-100)[..., 1:]
    ce = torch.nn.functional.cross_entropy(out.logits.float().view(-1,128), lab.reshape(-1), ignore_index=-100, reduction="sum")
    return ce, out.aux_loss
n_items = sum((torch.nn.functional.pad(mb["labels"], (0,1), value=-100)[...,1:] != -100).sum() for mb in mbs)
coef = 1.0
def R1():
    tot = 0
    for mb in mbs:
        ce, aux = ce_sum(mb); tot = tot + ce/n_items + coef*aux
    return tot
def R2():
    tot = 0
    for mb in mbs:
        ce, aux = ce_sum(mb); tot = tot + ce/n_items + coef*aux/len(mbs)
    return tot
def R3():
    ce, aux = ce_sum(allb); return ce/n_items + coef*aux
def rel(a, b):
    num = math.sqrt(sum(((a[k]-b[k])**2).sum().item() for k in a)); den = math.sqrt(sum((b[k]**2).sum().item() for k in a)); return num/den
def rel_router(a,b):
    ks=[k for k in a if "gate.weight" in k]
    num = math.sqrt(sum(((a[k]-b[k])**2).sum().item() for k in ks)); den = math.sqrt(sum((b[k]**2).sum().item() for k in ks)); return num/den
res = {}
for name, fn in (("R1_sum_aux", R1), ("R2_mean_aux", R2), ("R3_logical_aux", R3)):
    g = refgrad(fn)
    res[name] = dict(rel_all=rel(g_hf, g), rel_router=rel_router(g_hf, g))
    print(name, res[name])
# also: aux magnitudes
with torch.no_grad():
    auxs = [ce_sum(mb)[1].item() for mb in mbs]; aux_all = ce_sum(allb)[1].item()
print("per-microbatch aux:", auxs, "logical-batch aux:", aux_all)
json.dump(dict(res=res, auxs=auxs, aux_all=aux_all, n_items=int(n_items)), open("/tmp/claude-0/-home-user-opaque/a97cca5e-900f-5617-b2df-d73b55081cac/scratchpad/research/critic/expA_coef1.json","w"), indent=1)
