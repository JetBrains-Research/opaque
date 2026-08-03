import pytest

pytest.importorskip("transformers")
import torch

from opaque.api.patches.transformers.runtime.masking import (
    vmap_create_causal_mask,
    vmap_create_sliding_window_causal_mask,
)
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


class TestSlidingWindowCausalMask:
    """vmap_create_sliding_window_causal_mask enforces the look-back limit."""

    class _EagerConfig:
        _attn_implementation = "eager"
        sliding_window = 2

    class _EagerConfigNoWindow:
        _attn_implementation = "eager"

    def _make_mask(self, seq_len, sliding_window=2, past_seen_tokens=0):
        config = type(
            "Cfg",
            (),
            {"_attn_implementation": "eager", "sliding_window": sliding_window},
        )()
        input_embeds = torch.randn(1, seq_len, 8)
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + seq_len)
        return vmap_create_sliding_window_causal_mask(
            config,
            inputs_embeds=input_embeds,
            attention_mask=None,
            past_key_values=None,
            cache_position=cache_position,
        )

    def test_shape(self):
        mask = self._make_mask(seq_len=4, sliding_window=2)
        assert mask.shape == (1, 1, 4, 4)

    def test_within_window_causal_positions_are_zero(self):
        # seq_len=4, sliding_window=2; positions 0..3
        mask = self._make_mask(seq_len=4, sliding_window=2)
        # q=0, k=0: causal (k==q) and in-window (0 >= 0-2+1=-1) → 0.0
        assert mask[0, 0, 0, 0] == 0.0
        # q=1, k=1: in-window → 0.0
        assert mask[0, 0, 1, 1] == 0.0
        # q=2, k=1: causal (k<q), in-window (1 >= 2-2+1=1) → 0.0
        assert mask[0, 0, 2, 1] == 0.0
        # q=3, k=2: causal, in-window (2 >= 3-2+1=2) → 0.0
        assert mask[0, 0, 3, 2] == 0.0

    def test_outside_window_positions_are_neg_inf(self):
        # seq_len=4, sliding_window=2
        mask = self._make_mask(seq_len=4, sliding_window=2)
        neg_inf = torch.finfo(mask.dtype).min
        # q=2, k=0: causal, but out-of-window (0 < 2-2+1=1) → -inf
        assert mask[0, 0, 2, 0] == neg_inf
        # q=3, k=0: out-of-window (0 < 3-2+1=2) → -inf
        assert mask[0, 0, 3, 0] == neg_inf
        # q=3, k=1: out-of-window (1 < 3-2+1=2) → -inf
        assert mask[0, 0, 3, 1] == neg_inf

    def test_anti_causal_positions_remain_neg_inf(self):
        # Future tokens must always be -inf regardless of window.
        mask = self._make_mask(seq_len=4, sliding_window=100)
        neg_inf = torch.finfo(mask.dtype).min
        # q=0, k=1 (future)
        assert mask[0, 0, 0, 1] == neg_inf
        assert mask[0, 0, 0, 3] == neg_inf
        assert mask[0, 0, 1, 2] == neg_inf

    def test_window_larger_than_seq_reduces_to_causal(self):
        # When sliding_window >= seq_len, the result equals plain causal mask.
        seq_len = 4
        causal_cfg = type("Cfg", (), {"_attn_implementation": "eager"})()
        input_embeds = torch.randn(1, seq_len, 8)
        cache_position = torch.arange(seq_len)
        causal_mask = vmap_create_causal_mask(
            causal_cfg, inputs_embeds=input_embeds, cache_position=cache_position
        )
        sliding_mask = self._make_mask(seq_len=seq_len, sliding_window=seq_len + 10)
        assert torch.equal(causal_mask, sliding_mask)

    def test_none_passthrough_when_no_sliding_window_attr(self):
        # Without config.sliding_window the function must return the causal mask
        # unchanged (not None just because the attribute is absent).
        config = type("Cfg", (), {"_attn_implementation": "eager"})()
        input_embeds = torch.randn(1, 4, 8)
        cache_position = torch.arange(4)
        mask = vmap_create_sliding_window_causal_mask(
            config, inputs_embeds=input_embeds, cache_position=cache_position
        )
        # Should be a valid mask (not None) equal to the plain causal mask.
        causal_mask = vmap_create_causal_mask(
            config, inputs_embeds=input_embeds, cache_position=cache_position
        )
        assert mask is not None
        assert torch.equal(mask, causal_mask)

    def test_cached_kv_window_uses_correct_absolute_positions(self):
        # With past_seen_tokens=3, seq_len=2, sliding_window=2:
        # vmap_create_causal_mask lays out the key dim as:
        #   cols 0-1: current tokens at abs positions 3, 4  (cache_position)
        #   cols 2-4: past cached tokens at abs positions 0, 1, 2
        # Query at abs pos 4 (q_idx=1), window=2 → lower bound = 3.
        # → col 0 (abs 3) ✓, col 1 (abs 4) ✓ (future — blocked by causal),
        #   col 2 (abs 0) ✗, col 3 (abs 1) ✗, col 4 (abs 2) ✗.
        seq_len = 2
        past_seen_tokens = 3
        sliding_window = 2
        config = type(
            "Cfg",
            (),
            {"_attn_implementation": "eager", "sliding_window": sliding_window},
        )()
        input_embeds = torch.randn(1, seq_len, 8)
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + seq_len)

        # Provide a minimal DynamicCache-like object so past_seen_tokens is read.
        class _FakeCache:
            def get_seq_length(self):
                return past_seen_tokens

        mask = vmap_create_sliding_window_causal_mask(
            config,
            inputs_embeds=input_embeds,
            past_key_values=_FakeCache(),
            cache_position=cache_position,
        )
        neg_inf = torch.finfo(mask.dtype).min
        target_length = past_seen_tokens + seq_len  # 5

        assert mask.shape == (1, 1, seq_len, target_length)

        # q=1 (abs pos 4): can see col 0 (abs pos 3, in window), not cols 2-4 (abs 0-2)
        assert mask[0, 0, 1, 0] == 0.0, "col 0 (abs 3) should be in window"
        assert mask[0, 0, 1, 2] == neg_inf, "col 2 (abs 0) should be out of window"
        assert mask[0, 0, 1, 3] == neg_inf, "col 3 (abs 1) should be out of window"
        assert mask[0, 0, 1, 4] == neg_inf, "col 4 (abs 2) should be out of window"
