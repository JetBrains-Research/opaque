# User Guide

Welcome to the Opaque user guide! This guide will help you understand the core concepts of differential privacy and how
to use Opaque effectively.

## Overview

Opaque provides a functional API for training PyTorch models with differential privacy (DP). The library is organized
around five core concepts:

1. **[Per-Sample Gradient Clipping](clipping.md)** - Bound sensitivity by clipping gradients
2. **[Noise Addition](noise.md)** - Add calibrated Gaussian noise for privacy
3. **[Privacy Accounting](accounting.md)** - Track and query privacy budgets
4. **[Optimizers & Adaptive Clipping](optimizers.md)** - Adaptive clipping with TorchOpt integration
5. **[Poisson Sampling & Microbatching](sampling.md)** - Privacy amplification through sampling
6. **[LoRA Fine-tuning](lora.md)** - Parameter-efficient DP training for LLMs

## Learning Path

### Beginners

If you're new to differential privacy, start here:

1. **[Differential Privacy Basics](dp-basics.md)** - Understand what DP is and why it matters
2. **[Per-Sample Gradient Clipping](clipping.md)** - Learn how gradients are clipped
3. **[Quick Start Guide](../getting-started/quickstart.md)** - Train your first DP model
4. **[Tutorial 01](../tutorials/01_gradient_clipping_from_basics.ipynb)** - Interactive gradient clipping tutorial

### Intermediate

Once you understand the basics:

1. **[Noise Addition](noise.md)** - Understand noise calibration
2. **[Privacy Accounting](accounting.md)** - Learn to track privacy budgets
3. **[Tutorial 02](../tutorials/02_differential_privacy_noise_and_accounting.ipynb)** - Complete DP-SGD walkthrough
4. **[Tutorial 03](../tutorials/03_complete_dp_sgd_training.ipynb)** - End-to-end training example

### Advanced

For production use and optimization:

1. **[Optimizers & Adaptive Clipping](optimizers.md)** - Use adaptive clipping for better utility
2. **[Poisson Sampling & Microbatching](sampling.md)** - Optimize privacy-utility tradeoffs
3. **[LoRA Fine-tuning](lora.md)** - Train large language models efficiently
4. **[Tutorial 04](../tutorials/04_dp_optimizers.ipynb)** - TorchOpt integration
5. **[Tutorial 05](../tutorials/05_sampling_and_microbatching.ipynb)** - Advanced sampling techniques
6. **[Tutorial 06](../tutorials/06_lora_huggingface_dp_training.ipynb)** - Real-world LLM fine-tuning

## Key Concepts Summary

### Differential Privacy (DP)

DP provides mathematically rigorous privacy guarantees. A mechanism is (ε, δ)-differentially private if:

> Adding or removing any single training example changes the output distribution by at most e^ε with probability 1-δ

**Smaller ε = stronger privacy** (but potentially lower model utility)

### DP-SGD Algorithm

The DP-SGD algorithm ([Abadi et al. 2016](https://arxiv.org/abs/1607.00133)) has three key steps:

1. **Clip** per-example gradients to bounded L2 norm
2. **Add** calibrated Gaussian noise to summed gradients
3. **Update** model parameters

### Privacy Budget

Your **privacy budget** is the total privacy loss across all training steps:

- **ε (epsilon)**: Privacy loss parameter (lower = more private)
- **δ (delta)**: Failure probability (typically 1/n or 1/n²)

!!! warning "Budget Exhaustion"
Once you've spent your privacy budget, you cannot train more without weakening guarantees!

### Privacy-Utility Tradeoff

There's a fundamental tradeoff between privacy and model utility:

- **Stronger privacy** (lower ε) → More noise → Lower accuracy
- **Weaker privacy** (higher ε) → Less noise → Higher accuracy

Opaque helps you navigate this tradeoff through:

- **Calibration**: Find minimum noise for target privacy
- **Adaptive clipping**: Reduce gradient clipping impact
- **LoRA**: Train only a small subset of parameters

## Common Workflows

### Basic DP-SGD Training

```python
import opaque.accounting as acc
from opaque import clipped_grad, add_gaussian_noise

# 1. Calibrate noise
noise_multiplier = acc.find_noise_multiplier_for_epsilon_delta(
    epsilon=3.0, delta=1e-5, sample_rate=0.01, num_steps=1000
)

# 2. Create clipped gradient function
clipped_grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0, ...)

# 3. Training loop
privacy_state = acc.create()
for step in range(1000):
    grads = clipped_grad_fn(params, batch)
    noisy_grads = add_gaussian_noise(grads, stddev=noise_multiplier)
    params = update(params, noisy_grads)
    privacy_state = acc.compose_poisson_gaussian(privacy_state, ...)

# 4. Check final privacy
epsilon = acc.get_epsilon(privacy_state, delta=1e-5)
```

**See**: [Quick Start](../getting-started/quickstart.md) for complete example

### LoRA Fine-tuning with DP

```python
from peft import get_peft_model, LoraConfig
from opaque.optimizers import adaptive_clipping
import torchopt

# 1. Add LoRA adapters
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, lora_config)

# 2. Use adaptive clipping optimizer
base_opt = torchopt.sgd(lr=0.01)
optimizer = adaptive_clipping(
    base_opt,
    initial_clip_norm=1.0,
    target_quantile=0.5,
)

# 3. Train only LoRA parameters (much faster!)
```

**See**: [LoRA Guide](lora.md) and [Tutorial 06](../tutorials/06_lora_huggingface_dp_training.ipynb)

## Privacy Metrics

Opaque supports three privacy metrics:

### 1. (ε, δ)-Differential Privacy

**Standard metric** from [Dwork et al. 2006](https://link.springer.com/chapter/10.1007/11681878_14)

```python
epsilon = acc.get_epsilon(privacy_state, delta=1e-5)
```

### 2. f-DP Advantage

**Tighter bound** from [Dong et al. 2019](https://arxiv.org/abs/1905.02383)

```python
advantage = acc.get_advantage(privacy_state)
```

### 3. (α, β) Error Rates

**Hypothesis testing interpretation** from [Wasserman & Zhou 2010](https://www.stat.cmu.edu/~arinaldo/Fang_Zhou.pdf)

```python
beta = acc.get_beta(privacy_state, alpha=0.01)
```

## Best Practices

### 1. Start with High Privacy Budget

!!! tip "Iterate on privacy later"
Start with ε=10, get your model working, then tighten to ε=3 or ε=1

### 2. Use Calibration

!!! success "Let Opaque find the right noise"
Use `find_noise_multiplier_for_epsilon_delta()` instead of guessing

### 3. Monitor Privacy During Training

```python
if step % 100 == 0:
    current_eps = acc.get_epsilon(privacy_state, delta=delta)
    print(f"Step {step}: ε={current_eps:.2f}")
```

### 4. Use LoRA for LLMs

!!! info "LoRA makes DP practical"
Full fine-tuning of 7B model: ~10x slower with DP
LoRA fine-tuning of 7B model: ~2x slower with DP

## Troubleshooting

### Model doesn't train (accuracy stuck at chance)

- **Too much noise**: Increase ε or decrease training steps
- **Clipping too aggressive**: Increase `clip_norm` from 1.0 to 5.0
- **Learning rate too low**: Try 2-5x higher than non-DP training

### Privacy budget exceeded

- **Reduce training steps**: Train for fewer epochs
- **Increase ε**: Accept weaker privacy guarantees
- **Use truncated Poisson sampling**: Get tighter privacy bounds

### Out of memory

- **Use microbatching**: Process mini-batches sequentially
- **Use LoRA**: Train only adapter weights
- **Reduce batch size**: Lower `batch_size` (but increases privacy cost!)

## Next Steps

- **[API Reference](../api/index.md)**: Detailed function documentation
- **[Tutorials](../tutorials/README.md)**: Interactive Jupyter notebooks
- **[Development](../development/contributing.md)**: Contribute to Opaque

---

**Questions?** Open an issue on [GitHub](https://github.com/evgri243/opaque/issues)
