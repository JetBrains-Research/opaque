"""Auditor: was the opaque HF checkpoint glue active in the prototype's import path, and which MoE path ran?"""
import sys, torch, warnings
warnings.filterwarnings("ignore")
torch.set_num_threads(1)
sys.path.insert(0, "/home/user/opaque/packages/opaque-patches/tests/transformers/models")
from _test_utils import build_moe_model
import transformers
print("gradient_checkpointing_enable qualname:", transformers.PreTrainedModel.gradient_checkpointing_enable.__qualname__,
      "module:", transformers.PreTrainedModel.gradient_checkpointing_enable.__module__)
TINY = dict(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=128, pad_token_id=0, bos_token_id=1, eos_token_id=2,
            rope_theta=10000.0, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32)
model, mod = build_moe_model("mellum", "cpu", **TINY)
print("after build_moe_model:", transformers.PreTrainedModel.gradient_checkpointing_enable.__qualname__,
      transformers.PreTrainedModel.gradient_checkpointing_enable.__module__)
model.gradient_checkpointing_enable()
layer = model.model.layers[0]
print("layer.gradient_checkpointing:", getattr(layer, "gradient_checkpointing", None),
      "| _gradient_checkpointing_func:", getattr(model.model, "_gradient_checkpointing_func", None))
import inspect
f = getattr(model.model, "_gradient_checkpointing_func", None)
if f is not None:
    try:
        print("  keywords:", getattr(f, "keywords", None))
    except Exception as e:
        print("  ", e)
exp = model.model.layers[0].mlp.experts
print("experts forward:", type(exp).forward.__module__, getattr(type(exp).forward, "__opaque_patched__", None))
from opaque.api.patches.kernels import moe as moe_mod
print("grouped flag / helpers in moe.py:", [n for n in dir(moe_mod) if "grouped" in n.lower()][:10])
