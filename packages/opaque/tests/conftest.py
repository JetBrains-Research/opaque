"""Pytest configuration and shared fixtures for Opaque tests."""

import pytest
import torch


def get_default_device():
    """Get the default device for testing (CUDA > MPS > CPU in priority order).

    Returns:
        torch.device: Best available device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@pytest.fixture
def device():
    """Provide device for tests (CUDA > MPS > CPU in priority order)."""
    return get_default_device()


@pytest.fixture(params=["cpu", "cuda", "mps"])
def all_devices(request):
    """Parametrize tests over all available devices."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if request.param == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    return torch.device(request.param)


@pytest.fixture
def set_random_seed():
    """Fixture to set random seed for reproducibility."""

    def _set_seed(seed: int = 42):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

    return _set_seed


@pytest.fixture
def simple_pytree():
    """Provide a simple PyTree for testing."""
    return {
        "weight": torch.tensor([3.0, 4.0]),
        "bias": torch.tensor([0.0, 12.0]),
    }


@pytest.fixture
def nested_pytree():
    """Provide a nested PyTree for testing."""
    return {
        "layer1": {
            "weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "bias": torch.tensor([0.5, -0.5]),
        },
        "layer2": {"weight": torch.tensor([1.0]), "bias": torch.tensor([0.0])},
    }


# =============================================================================
# Model Testing Utilities (Shared across all model tests)
# =============================================================================

# Standard LoRA configuration used across all tests
STANDARD_LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
}

# Model configurations for testing (memory-aware for A100 40GB)
# Shared between single-GPU, multi-GPU, and validation tests
MODEL_CONFIGS = {
    "qwen2-0.5b": {
        "model_id": "Qwen/Qwen2-0.5B",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "min_mem_gb": 12,
        "max_length": 2048,
        "trust_remote_code": False,
        "batch_size": 8,
        "accum_steps": 32,
    },
    "qwen2-1.5b": {
        "model_id": "Qwen/Qwen2-1.5B",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "min_mem_gb": 12,
        "max_length": 1024,
        "trust_remote_code": False,
        "batch_size": 8,
        "accum_steps": 16,
    },
    "tinyllama-1.1b": {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "min_mem_gb": 12,
        "max_length": 1024,
        "trust_remote_code": False,
        "batch_size": 8,
        "accum_steps": 16,
    },
    "phi3-mini": {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "target_modules": ["qkv_proj"],  # Phi-3 uses fused QKV projection
        "min_mem_gb": 16,
        "max_length": 2048,
        "trust_remote_code": True,
        "batch_size": 8,
        "accum_steps": 16,
    },
}


def build_text_batch(batch_size):
    """Create deterministic text batch for testing.

    Args:
        batch_size: Number of texts to generate

    Returns:
        list[str]: List of text strings
    """
    base_texts = [
        "Hello world test",
        "Another example",
        "Third sample",
        "Final one",
        "Short prompt",
        "Yet another example",
        "Testing a longer input",
        "Final sample in batch",
    ]
    if batch_size <= len(base_texts):
        return base_texts[:batch_size]
    repeats = (batch_size + len(base_texts) - 1) // len(base_texts)
    return (base_texts * repeats)[:batch_size]


def has_min_gpu_memory(min_gb):
    """Check if GPU has minimum required memory.

    Args:
        min_gb: Minimum required GB of GPU memory

    Returns:
        bool: True if GPU has sufficient memory
    """
    if not torch.cuda.is_available():
        return False
    try:
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        return total_bytes >= min_gb * (1024**3)
    except Exception:
        return False


def load_model_with_lora(
    model_config, device="cuda", dtype=torch.float16, lora_config=None
):
    """Load HuggingFace model with LoRA adapters.

    Shared utility for loading models consistently across all tests.

    Args:
        model_config: Model configuration dict from MODEL_CONFIGS
        device: Device to load model on
        dtype: Model dtype
        lora_config: Optional LoRA config dict, uses STANDARD_LORA_CONFIG if None

    Returns:
        tuple: (model, tokenizer)
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_id = model_config["model_id"]
    trust_remote_code = model_config["trust_remote_code"]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model config and model
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    config._attn_implementation = "eager"  # For vmap compatibility

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    ).to(device)

    # Apply LoRA
    if lora_config is None:
        lora_config = STANDARD_LORA_CONFIG.copy()

    lora_config["target_modules"] = model_config["target_modules"]
    model = get_peft_model(model, LoraConfig(**lora_config))

    return model, tokenizer


def run_dp_training_step(
    model,
    tokenizer,
    batch_size,
    max_length,
    accum_steps,
    training_steps=3,
    learning_rate=1e-3,
    l2_clip_norm=1.0,
):
    """Run DP-SGD training with clipped gradients and gradient accumulation.

    Shared training logic used across single-GPU and multi-GPU tests.

    Args:
        model: Model with LoRA (or full parameters)
        tokenizer: Tokenizer
        batch_size: Batch size
        max_length: Max sequence length
        accum_steps: Gradient accumulation steps
        training_steps: Number of training steps
        learning_rate: Learning rate
        l2_clip_norm: L2 clipping norm

    Returns:
        tuple: (final_accumulated_grads, final_clip_state)
    """
    from opaque import clipped_grad, make_functional
    from opaque.utils.pytree import tree_map

    device = next(model.parameters()).device

    # Prepare batch
    texts = build_text_batch(batch_size)
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        max_length=max_length,
        truncation=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    labels = input_ids.clone()

    # Make functional
    fmodel, trainable, frozen = make_functional(
        model,
        disable_autograd_tracking=True,
        partition_trainable=True,
    )

    # Define per-example loss
    def per_example_loss(
        trainable_params, frozen_params, ids_single, mask_single, labels_single
    ):
        all_params = {**frozen_params, **trainable_params}
        outputs = fmodel(
            all_params, ids_single, attention_mask=mask_single, labels=labels_single
        )
        return outputs.loss

    # Create clipped gradient function
    grad_fn, clip_state = clipped_grad(
        per_example_loss,
        argnums=0,
        batch_argnums=(2, 3, 4),
        l2_clip_norm=l2_clip_norm,
    )

    state = clip_state
    trainable_params = trainable
    last_accumulated = None

    # Training loop
    for _step in range(training_steps):
        accumulated = None

        # Gradient accumulation
        for _ in range(accum_steps):
            grads, state = grad_fn(
                trainable_params,
                frozen,
                input_ids,
                attention_mask,
                labels,
                state=state,
            )

            if accumulated is None:
                accumulated = tree_map(lambda x: x.detach().clone(), grads)
            else:
                accumulated = tree_map(lambda x, y: x + y, accumulated, grads)

        # Apply accumulated gradients
        scale = 1.0 / float(accum_steps)
        accumulated = tree_map(lambda x, s=scale: x * s, accumulated)

        trainable_params = tree_map(
            lambda p, g: p - learning_rate * g,
            trainable_params,
            accumulated,
        )

        last_accumulated = accumulated

    return last_accumulated, state


@pytest.fixture(scope="session")
def model_configs():
    """Provide model configurations to tests."""
    return MODEL_CONFIGS


@pytest.fixture(scope="session")
def standard_lora_config():
    """Provide standard LoRA configuration."""
    return STANDARD_LORA_CONFIG
