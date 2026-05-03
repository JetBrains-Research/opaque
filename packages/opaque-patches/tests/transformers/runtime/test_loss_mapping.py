import pytest
from opaque.patches import apply_runtime_patches

def test_loss_mapping():
    # Patch the global registry
    apply_runtime_patches(use_fused_loss=True)
    
    from transformers.loss.loss_utils import LOSS_MAPPING
    from opaque.patches.transformers.components.cross_entropy import _opaque_causal_lm_loss
    
    assert "ForCausalLM" in LOSS_MAPPING
    # Transformers creates instances in LOSS_MAPPING, or stores functions.
    # Let's check if the CausalLM loss maps to our function.
    
    loss_fn = LOSS_MAPPING["ForCausalLM"]
    assert loss_fn.__name__ == "_opaque_causal_lm_loss"
    
    # We can also check if we can call it (smoke test)
    import torch
    logits = torch.randn(2, 5, 10)
    labels = torch.randint(0, 10, (2, 5))
    vocab_size = 10
    
    loss = loss_fn(logits, labels, vocab_size)
    assert loss.dim() == 0
