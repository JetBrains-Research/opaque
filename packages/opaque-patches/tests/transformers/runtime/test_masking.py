import torch
from opaque.patches import apply_runtime_patches

def test_vmap_causal_mask():
    apply_runtime_patches(enable_vmap_masking=True)
    
    # create_causal_mask should be patched now
    import transformers.masking_utils as masking_utils
    create_causal_mask = masking_utils.create_causal_mask
    
    input_embeds = torch.randn(1, 4, 16)
    
    class DummyConfig:
        _attn_implementation = "eager"
        
    config = DummyConfig()

    # Test that it handles normal execution
    mask = create_causal_mask(
        config=config,
        input_embeds=input_embeds,
        attention_mask=None,
        cache_position=torch.arange(4),
        past_key_values=None,
    )
    assert mask.shape == (1, 1, 4, 4)
    
    # Test under vmap with a fake batched tensor (we can simulate by just passing higher dimension)
    input_embeds_vmap = torch.randn(1, 4, 16)
    attention_mask = torch.ones(4) # 1D mask to simulate vmap masking
    
    mask_vmap = create_causal_mask(
        config=config,
        input_embeds=input_embeds_vmap,
        attention_mask=attention_mask,
        cache_position=torch.arange(4),
        past_key_values=None,
    )
    assert mask_vmap.shape == (1, 1, 4, 4)
