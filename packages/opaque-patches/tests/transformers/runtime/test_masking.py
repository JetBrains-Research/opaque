import pytest

pytest.importorskip("transformers")
import torch

from opaque.patches import apply_runtime_patches


def test_vmap_causal_mask():
    apply_runtime_patches(vmap_masking=True)

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
    attention_mask = torch.ones(4)  # 1D mask to simulate vmap masking

    mask_vmap = create_causal_mask(
        config=config,
        input_embeds=input_embeds_vmap,
        attention_mask=attention_mask,
        cache_position=torch.arange(4),
        past_key_values=None,
    )
    assert mask_vmap.shape == (1, 1, 4, 4)


def test_vmap_causal_mask_follows_autocast(monkeypatch):
    """Mask dtype mirrors autocast so SDPA's attn_mask matches a bf16 query."""
    from opaque.api.patches.transformers.runtime import masking

    apply_runtime_patches(vmap_masking=True)
    import transformers.masking_utils as masking_utils

    create_causal_mask = masking_utils.create_causal_mask
    input_embeds = torch.randn(1, 4, 16, dtype=torch.float32)

    class DummyConfig:
        _attn_implementation = "eager"

    mask = create_causal_mask(
        config=DummyConfig(),
        input_embeds=input_embeds,
        attention_mask=None,
        cache_position=torch.arange(4),
        past_key_values=None,
    )
    assert mask.dtype == torch.float32

    # Simulate CUDA autocast through the helper used by vmap_create_causal_mask;
    # ``torch.is_autocast_enabled("cuda")`` only flips inside a real CUDA ctx.
    monkeypatch.setattr(
        masking,
        "_active_mask_dtype",
        lambda _ie: torch.bfloat16,
    )
    mask_bf16 = create_causal_mask(
        config=DummyConfig(),
        input_embeds=input_embeds,
        attention_mask=None,
        cache_position=torch.arange(4),
        past_key_values=None,
    )
    assert mask_bf16.dtype == torch.bfloat16


def test_masking_runtime_patch_idempotent_for_ignore_causal_mask_sdpa():
    apply_runtime_patches(vmap_masking=True)

    import transformers.masking_utils as masking_utils

    patched_fn = masking_utils._ignore_causal_mask_sdpa
    original_fn = getattr(patched_fn, "_original", None)
    assert original_fn is not None
    assert original_fn is not patched_fn

    # Re-applying runtime patches must preserve the same original binding.
    apply_runtime_patches(vmap_masking=True)
    patched_fn_2 = masking_utils._ignore_causal_mask_sdpa
    original_fn_2 = getattr(patched_fn_2, "_original", None)
    assert patched_fn_2 is patched_fn
    assert original_fn_2 is original_fn
