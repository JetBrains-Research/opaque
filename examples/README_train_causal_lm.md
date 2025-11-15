# DP-SGD LoRA Training for Causal Language Models

Universal training script for fine-tuning any HuggingFace causal language model with differential privacy using LoRA and DP-SGD.

## Features

- **Universal model support**: Works with any HuggingFace causal LM (Qwen, LLaMA, GPT-2, Mistral, etc.)
- **Flexible dataset support**: Compatible with any HuggingFace text dataset
- **LoRA fine-tuning**: Memory-efficient adaptation of large models
- **Differential Privacy**: DP-SGD with configurable privacy guarantees
- **Adaptive clipping**: Enabled by default, dynamically adjusts gradient clipping norm
- **Detailed telemetry**: Real-time monitoring of loss, gradients, clipping, and noise statistics
- **Fully configurable**: All hyperparameters controllable via command-line


### Basic Usage

```bash
uv run --extra examples examples/train_causal_lm.py \
    --model_name Qwen/Qwen2.5-7B \
    --dataset ag_news \
    --num_train_samples 2000 \
    --batch_size 16 \
    --num_epochs 80
```

## Configuration Examples

Configurations scale from small (quick testing) to large (production), based on production DP-SGD training parameters.

### 1. Small Model (Qwen 0.5B) - Quick Test

Fast configuration for development and testing:

```bash
uv run --extra examples examples/train_causal_lm.py \
    --model_name Qwen/Qwen2.5-0.5B \
    --dataset ag_news \
    --num_train_samples 500 \
    --batch_size 32 \
    --num_epochs 10
```

**Expected**: ~10 minutes, ~4 GB memory

**What's different**: Smaller model, higher batch size, fewer epochs for quick iteration.

### 2. Medium Model (Qwen 1.5B) - Balanced

Balanced configuration matching production parameters:

```bash
uv run --extra examples examples/train_causal_lm.py \
    --model_name Qwen/Qwen2.5-1.5B \
    --dataset ag_news \
    --num_train_samples 2000 \
    --batch_size 16 \
    --num_epochs 40
```

**Expected**: ~2-3 hours, ~8 GB memory

**What's different**: Default parameters, good balance of speed and quality.

### 3. Large Model (Qwen 7B) - Production

Full production configuration with long training:

```bash
uv run --extra examples examples/train_causal_lm.py \
    --model_name Qwen/Qwen2.5-7B \
    --dataset ag_news \
    --num_train_samples 2000 \
    --batch_size 8 \
    --num_epochs 80
```

**Expected**: ~8-12 hours on H200, ~16 GB memory

**What's different**: Largest model, full 80 epochs, reduced batch size for memory.

### Default Parameters (Applied to All Configs)

All configurations use these production-tested defaults:
- **LoRA**: rank=8, alpha=8, targets=all linear layers
- **DP-SGD**: clip_norm=0.15, noise_multiplier=0.025
- **Adaptive clipping**: enabled, target_clip_rate=0.50, clip_norm_max=10.0
- **Optimizer**: SGD with learning_rate=0.0001
- **Sequence length**: 512 tokens
- **Microbatch size**: 1 (for per-example gradients)

### Customize Parameters

Override any default:
```bash
# Higher privacy (more noise)
uv run --extra examples examples/train_causal_lm.py --noise_multiplier 0.1

# Fixed clipping instead of adaptive
uv run --extra examples examples/train_causal_lm.py --no_adaptive_clipping

# Different dataset
uv run --extra examples examples/train_causal_lm.py \
    --dataset JetBrains/KStack-clean \
    --dataset_text_field content

# Larger LoRA rank
uv run --extra examples examples/train_causal_lm.py --lora_r 16 --lora_alpha 16
```

## Training Telemetry

The script outputs detailed telemetry on every training step:

```
Step   10 [E1 B 10/125] | Loss: 2.3456 (min=1.2345, max=3.4567, std=0.5678) | Clip: norm=1.234, rate=25.0% (2/8) | Grad: μ=1.456, med=1.234, σ=[0.456, 2.345] | Noise: σ=0.6170
```

**Telemetry breakdown:**
- **Step info**: `Step 10 [E1 B 10/125]` - Step 10, Epoch 1, Batch 10 of 125
- **Loss**: Mean loss with min/max/std across batch
- **Clip**: Current clipping norm (adapts if using adaptive clipping), clipping rate percentage, and exact count of clipped gradients
- **Grad**: Gradient norm statistics (mean μ, median med, range σ)
- **Noise**: Current noise standard deviation applied for DP

This telemetry helps you:
- Monitor training stability (loss should decrease)
- Verify adaptive clipping is working (clip norm should adjust, clip rate should stay near target)
- Detect gradient explosions (large grad norms or hitting clip_norm_max)
- Understand privacy-utility tradeoff (higher noise = more privacy, potentially slower convergence)

## Command-Line Arguments

### Model Arguments

- `--model_name`: HuggingFace model name or path (default: `Qwen/Qwen2.5-7B`)
- `--use_eager_attention`: Use eager attention implementation (auto-enabled for MPS)

### Dataset Arguments

- `--dataset`: HuggingFace dataset name (default: `ag_news`)
- `--dataset_split`: Dataset split to use (default: `train`)
- `--dataset_text_field`: Field containing text data (default: `text`)
- `--num_train_samples`: Number of training samples (default: `2000`)
- `--max_seq_len`: Maximum sequence length (default: `512`)

### Training Arguments

- `--batch_size`: Batch size for training (default: `16`)
- `--num_epochs`: Number of training epochs (default: `80`)
- `--learning_rate`: Learning rate (default: `0.0001`)
- `--optimizer`: Optimizer to use: `sgd` or `adam` (default: `sgd`)
- `--seed`: Random seed (default: `42`)

### LoRA Arguments

- `--lora_r`: LoRA rank (default: `8`)
- `--lora_alpha`: LoRA alpha scaling factor (default: `8`)
- `--lora_target_modules`: Target modules for LoRA (default: all linear layers - `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`)

### DP-SGD Arguments

- `--clip_norm`: Gradient clipping norm, or initial value for adaptive clipping (default: `0.15`)
- `--noise_multiplier`: Noise multiplier for DP (default: `0.025`)
- `--microbatch_size`: Microbatch size for gradient computation (default: `1`)
- `--use_adaptive_clipping`: Enable adaptive clipping (default: `True`)
- `--no_adaptive_clipping`: Disable adaptive clipping and use fixed clip norm
- `--target_clip_rate`: Target clipping rate for adaptive clipping (default: `0.50`, targets 50% unclipped)
- `--clip_norm_max`: Maximum clip norm for adaptive clipping (default: `10.0`)

## Memory Optimization Tips

1. **Reduce batch size**: Lower `--batch_size` to 1 or 2 for large models
2. **Reduce microbatch size**: Set `--microbatch_size` to 1 for memory-constrained setups
3. **Shorter sequences**: Use `--max_seq_len 128` or `--max_seq_len 256`
4. **Smaller LoRA rank**: Use `--lora_r 8` instead of higher values
5. **Fewer target modules**: Only adapt attention layers with `--lora_target_modules q_proj v_proj`
6. **Smaller model**: Start with 0.5B or 1.5B models before scaling to 7B

## Performance Optimization

1. **Fixed clipping**: Default mode is fastest (avoids recompilation)
2. **Larger microbatches**: Increase `--microbatch_size` if memory allows
3. **Batch size**: Larger batches are more efficient but use more memory
4. **GPU selection**: Use CUDA when available (much faster than CPU/MPS)

## Supported Models

The script works with any HuggingFace causal language model, including:

- **Qwen**: Qwen2.5-0.5B, 1.5B, 3B, 7B, 14B, 32B
- **LLaMA**: LLaMA-2-7B, LLaMA-3-8B, LLaMA-3.2-1B/3B
- **Mistral**: Mistral-7B-v0.1/v0.3
- **GPT-2**: gpt2, gpt2-medium, gpt2-large
- **Gemma**: gemma-2b, gemma-7b
- **Phi**: microsoft/phi-2, microsoft/phi-3-mini

## Supported Datasets

Any HuggingFace dataset with text fields, including:

- **ag_news**: News article classification
- **imdb**: Movie review sentiment
- **wikitext**: Wikipedia articles
- **bookcorpus**: Books corpus
- **openwebtext**: Web text
- Custom datasets uploaded to HuggingFace

## Privacy Considerations

The noise multiplier and clipping norm control the privacy-utility tradeoff:

- **Stronger privacy**: Higher `--noise_multiplier` (e.g., 1.0-2.0), lower `--clip_norm` (e.g., 0.1-0.5)
- **Better utility**: Lower `--noise_multiplier` (e.g., 0.1-0.5), higher `--clip_norm` (e.g., 1.0-5.0)

Use privacy accounting tools (e.g., Opacus, TensorFlow Privacy) to compute formal (ε, δ)-DP guarantees.

## Troubleshooting

### Out of Memory Errors

1. Reduce `--batch_size` to 1
2. Reduce `--microbatch_size` to 1
3. Reduce `--max_seq_len` to 128 or 64
4. Use a smaller model
5. Reduce `--lora_r` to 8

### Slow Training

1. Ensure CUDA is available (check with `torch.cuda.is_available()`)
2. Disable `--use_adaptive_clipping` (default is disabled)
3. Increase `--microbatch_size` if memory allows
4. Use fewer training samples initially for testing

### Model Loading Errors

1. Check model name is correct on HuggingFace Hub
2. For gated models (LLaMA), authenticate with `huggingface-cli login`
3. Install required model-specific dependencies

## Examples Directory

The original `train_qwen.py` remains available for reference. The new `train_causal_lm.py` is a universal replacement with full configurability.
