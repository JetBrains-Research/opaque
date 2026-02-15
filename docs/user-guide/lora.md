# LoRA Fine-tuning with Differential Privacy

[LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation) is a **parameter-efficient fine-tuning** method that is
ideally suited for differential privacy. This guide shows you how to combine LoRA with DP-SGD for private LLM
fine-tuning.

## Why LoRA + DP is Powerful

Training large language models (LLMs) with differential privacy is challenging:

| Challenge        | Full Fine-tuning              | LoRA Fine-tuning             |
|------------------|-------------------------------|------------------------------|
| **Parameters**   | All (~7B for Llama-2-7B)      | ~0.1% (adapters only)        |
| **Memory**       | Very high (per-example grads) | Low (small adapters)         |
| **DP Overhead**  | ~10x slower                   | **~2x slower**               |
| **Accuracy**     | Good with enough data         | **Often matches full FT**    |
| **Privacy Cost** | High (more parameters)        | **Lower (fewer parameters)** |

**Key insight**: LoRA trains only a tiny fraction of parameters, making per-example gradient computation much cheaper!

### What is LoRA?

LoRA adds **low-rank adapter matrices** to transformer layers:

```python
# Standard transformer layer
output = W @ input  # W is large (e.g., 4096 × 4096)

# LoRA adapter
output = (W + B @ A) @ input
# B: 4096 × r  (rank r = 8, 16, 32)
# A: r × 4096
# W: frozen (not trained)
```

**During training**:

- Freeze pre-trained weights **W**
- Train only adapters **A** and **B** (much smaller!)
- Per-example gradients only for A, B

**During inference**:

- Merge adapters: W' = W + B @ A
- No additional latency!

## Quick Start: LoRA + DP with HuggingFace

Here's a minimal example using Opaque with HuggingFace PEFT:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
import opaque.accounting as acc
from opaque import clipped_grad, gaussian_noise

# 1. Load base model
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Add LoRA adapters
lora_config = LoraConfig(
    r=8,  # Rank (higher = more parameters, better quality)
    lora_alpha=16,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)

# 3. Convert to functional (only LoRA parameters!)
from opaque import make_functional
fmodel, lora_params = make_functional(model)

# 4. Define per-example loss
def loss_fn(params, example):
    input_ids, labels = example
    outputs = fmodel(params, input_ids=input_ids.unsqueeze(0), labels=labels.unsqueeze(0))
    return outputs.loss

# 5. Calibrate noise for target privacy
batch_size = 8
dataset_size = 10000
sample_rate = batch_size / dataset_size
num_epochs = 3
num_steps = num_epochs * (dataset_size // batch_size)

noise_multiplier = acc.find_noise_multiplier_for_epsilon_delta(
    epsilon=3.0,
    delta=1e-5,
    sample_rate=sample_rate,
    num_steps=num_steps,
)

# 6. Create DP gradient function
clip_norm = 1.0
dp_grad_fn = clipped_grad(
    loss_fn,
    l2_clip_norm=clip_norm,
    argnums=0,
    batch_argnums=1,
)

# 7. Training loop
noise_fn, noise_state = gaussian_noise(stddev=noise_mult * clip_norm)
privacy_state = acc.create()
learning_rate = 0.0001

for epoch in range(num_epochs):
    for batch in dataloader:
        # Compute clipped gradients (only for LoRA params!)
        grads = dp_grad_fn(lora_params, batch)

        # Add noise
        noisy_grads, noise_state = noise_fn(grads, noise_state)

        # Update LoRA parameters
        lora_params = tuple(p - learning_rate * g for p, g in zip(lora_params, noisy_grads))

        # Track privacy
        privacy_state = acc.compose_poisson_gaussian(
            privacy_state,
            noise_multiplier=noise_mult,
            sample_rate=sample_rate,
            count=1,
        )

# 8. Get final privacy
final_epsilon = acc.get_epsilon(privacy_state, delta=1e-5)
print(f"Training complete! Privacy: (ε={final_epsilon:.2f}, δ=1e-5)")
```

## LoRA Hyperparameters

### Rank (r)

The **rank** controls adapter capacity:

```python
lora_config = LoraConfig(r=8)  # Low rank (fewer parameters)
lora_config = LoraConfig(r=16)  # Medium rank (balanced)
lora_config = LoraConfig(r=32)  # High rank (more parameters)
```

**Guidelines**:

- Start with **r=8** for most tasks
- Increase to **r=16** or **r=32** if accuracy is insufficient
- Higher r → more parameters → slower DP training

### Alpha (lora_alpha)

The **alpha** parameter controls scaling:

```python
lora_config = LoraConfig(r=8, lora_alpha=16)  # Common: alpha = 2 * rank
```

**Rule of thumb**: Set `lora_alpha = 2 * r` or `lora_alpha = r`

### Target Modules

Choose which transformer layers to adapt:

```python
# Minimal (fastest, lowest memory)
lora_config = LoraConfig(target_modules=["q_proj"])

# Standard (good balance)
lora_config = LoraConfig(target_modules=["q_proj", "v_proj"])

# Comprehensive (best quality, slower)
lora_config = LoraConfig(target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
```

**For DP training**: Start with `["q_proj", "v_proj"]` for best speed/quality tradeoff

## Memory Optimization

LoRA is already memory-efficient, but DP adds overhead. Use these techniques:

### 1. Microbatching

Process batches in smaller chunks:

```python
microbatch_size = 2  # Process 2 examples at a time

def compute_gradients_microbatched(params, batch):
    total_grads = None
    for i in range(0, len(batch), microbatch_size):
        microbatch = batch[i:i+microbatch_size]
        grads = dp_grad_fn(params, microbatch)

        if total_grads is None:
            total_grads = grads
        else:
            total_grads = {k: total_grads[k] + grads[k] for k in grads}

    return total_grads
```

### 2. Gradient Checkpointing

Reduce memory at the cost of compute:

```python
model.gradient_checkpointing_enable()
```

### 3. Mixed Precision Training

Use FP16 or BF16:

```python
from torch.cuda.amp import autocast

with autocast():
    grads = dp_grad_fn(params, batch)
```

### 4. Smaller Rank

Reduce LoRA rank:

```python
lora_config = LoraConfig(r=4)  # Minimal adapters
```

## Privacy Budget Planning

LoRA enables more training with same privacy budget:

```python
# Full fine-tuning: 7B parameters
# Gradient computation: ~10x slower
# Practical: ε=10 (weak privacy)

# LoRA fine-tuning: ~7M parameters (0.1%)
# Gradient computation: ~2x slower
# Practical: ε=3 (strong privacy!) ✓
```

**Example budget allocation**:

```python
# Target: (ε=3, δ=1e-5)
# Dataset: 10,000 examples
# Batch size: 8
# Epochs: 3

num_steps = 3 * (10000 // 8)  # 3750 steps

noise_multiplier = acc.find_noise_multiplier_for_epsilon_delta(
    epsilon=3.0,
    delta=1e-5,
    sample_rate=8/10000,
    num_steps=num_steps,
)
# Result: noise_multiplier ≈ 1.2 (achievable!)
```

## Complete Example: Sentiment Classification

Here's a complete example fine-tuning a model for sentiment analysis:

```python
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig
import torch
import opaque.accounting as acc
from opaque import clipped_grad, gaussian_noise

# 1. Load dataset
dataset = load_dataset("imdb")
train_dataset = dataset["train"].select(range(10000))  # Use 10k examples

# 2. Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 3. Add LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.05,
)
model = get_peft_model(model, lora_config)

print(f"Trainable parameters: {model.print_trainable_parameters()}")
# Output: trainable params: ~300K / 110M = 0.27%

# 4. Prepare data
def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = train_dataset.map(preprocess, batched=True)

# 5. Setup DP training
clip_norm = 1.0
batch_size = 8
sample_rate = batch_size / len(train_dataset)
num_epochs = 3
num_steps = num_epochs * (len(train_dataset) // batch_size)

noise_multiplier = acc.find_noise_multiplier_for_epsilon_delta(
    epsilon=3.0, delta=1e-5, sample_rate=sample_rate, num_steps=num_steps
)

print(f"Using noise multiplier: {noise_multiplier:.3f}")

# 6. Training (simplified)
# See Tutorial 06 for complete implementation
```

## Integration with HuggingFace Ecosystem

Opaque works seamlessly with HuggingFace libraries:

### PEFT (Parameter-Efficient Fine-Tuning)

```python
from peft import get_peft_model, LoraConfig

lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, lora_config)
```

### Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
```

### Datasets

```python
from datasets import load_dataset

dataset = load_dataset("imdb")
```

## Best Practices

### 1. Start Small, Scale Up

```python
# 1. Debug with small model + few examples
model = "gpt2"  # 124M params
dataset_size = 1000

# 2. Validate on medium model
model = "gpt2-medium"  # 355M params
dataset_size = 10000

# 3. Scale to full model
model = "meta-llama/Llama-2-7b-hf"  # 7B params
dataset_size = 100000
```

### 2. Use Appropriate Rank

```python
# Small models (< 500M params)
lora_config = LoraConfig(r=4)

# Medium models (500M - 3B params)
lora_config = LoraConfig(r=8)

# Large models (> 3B params)
lora_config = LoraConfig(r=16)
```

### 3. Monitor Privacy During Training

```python
if step % 100 == 0:
    eps = acc.get_epsilon(privacy_state, delta=1e-5)
    if eps > target_epsilon:
        print(f"Warning: Privacy budget exceeded at step {step}!")
        break
```

### 4. Use Gradient Accumulation

For larger effective batch sizes without memory issues:

```python
accumulation_steps = 4
for step in range(num_steps):
    accumulated_grads = None

    for micro_step in range(accumulation_steps):
        batch = next(dataloader)
        grads = dp_grad_fn(params, batch)

        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            accumulated_grads = {k: accumulated_grads[k] + grads[k] for k in grads}

    # Average accumulated gradients
    accumulated_grads = {k: v / accumulation_steps for k, v in accumulated_grads.items()}

    # Add noise and update
    noisy_grads, noise_state = noise_fn(accumulated_grads, noise_state)
    params = update(params, noisy_grads)
```

## Troubleshooting

### Out of Memory

**Solutions**:

1. Reduce LoRA rank: `r=8 → r=4`
2. Enable gradient checkpointing: `model.gradient_checkpointing_enable()`
3. Use microbatching
4. Reduce batch size
5. Use mixed precision (FP16/BF16)

### Poor Accuracy

**Solutions**:

1. Increase LoRA rank: `r=8 → r=16`
2. Add more target modules: `["q_proj", "v_proj", "k_proj", "o_proj"]`
3. Reduce privacy (increase ε temporarily): `ε=3 → ε=10`
4. Increase batch size (better privacy amplification)
5. Train longer (more epochs)

### Training Too Slow

**Solutions**:

1. Reduce target modules: `["q_proj", "v_proj"] → ["q_proj"]`
2. Use smaller batch size (fewer per-example gradients)
3. Reduce LoRA rank: `r=16 → r=8`
4. Use gradient checkpointing (trades compute for memory)

## See Also

- **[Tutorial 06](../tutorials/06_lora_huggingface_dp_training.ipynb)**: Complete LoRA + DP tutorial
- **[Gradient Clipping](clipping.md)**: Understanding per-example gradients
- **[Sampling](sampling.md)**: Privacy amplification with Poisson sampling
- **[PEFT Documentation](https://huggingface.co/docs/peft)**: Learn more about LoRA

---

**Ready to start?** Check out [Tutorial 06](../tutorials/06_lora_huggingface_dp_training.ipynb) for a complete
walkthrough!
