import pytest
from opaque.patches.transformers._router import apply_transformers_model_patches

def test_graceful_ignore_kwargs():
    """Verify that unsupported kwargs (like rope) are gracefully ignored by specialized functions."""
    
    # Mistral does not have a custom RoPE patch in opaque (rope_apply_fn is False in _families.py)
    # However, we can pass rope=True to apply_kernels without raising an error.
    try:
        apply_transformers_model_patches("mistral", rope=True)
    except TypeError:
        pytest.fail("apply_kernels raised TypeError for unsupported kwarg on mistral")

    # Likewise, passing unknown kwargs should be gracefully ignored by **kwargs catch-all.
    try:
        apply_transformers_model_patches("llama", some_random_flag=True, another_flag=False)
    except TypeError:
        pytest.fail("apply_kernels raised TypeError for unknown kwarg on llama")
