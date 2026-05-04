import pytest

pytest.importorskip("transformers")


def test_loss_mapping():
    """``apply_loss_mapping_patch`` registers opaque's fused-CE in the
    global ``LOSS_MAPPING``.

    Triggered automatically from :func:`opaque.patches.apply_model_patches`
    when any model is patched with ``cross_entropy=True``; this test
    invokes the underlying runtime patch directly.
    """
    from opaque.patches.transformers.runtime.loss_mapping import (
        apply_loss_mapping_patch,
    )

    apply_loss_mapping_patch(cross_entropy=True)

    from transformers.loss.loss_utils import LOSS_MAPPING

    assert "ForCausalLM" in LOSS_MAPPING
    loss_fn = LOSS_MAPPING["ForCausalLM"]
    assert loss_fn.__name__ == "_opaque_causal_lm_loss"

    # Smoke-test that the registered loss is callable with the standard signature.
    import torch

    logits = torch.randn(2, 5, 10)
    labels = torch.randint(0, 10, (2, 5))
    vocab_size = 10

    loss = loss_fn(logits, labels, vocab_size)
    assert loss.dim() == 0
