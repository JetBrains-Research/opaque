# Opaque

**Differential Privacy for LoRA Fine-tuning**

Opaque is a minimal, focused library for training PyTorch models with Differential Privacy, specifically designed for LoRA (Low-Rank Adaptation) fine-tuning of Large Language Models.

## Why Opaque?

While [Opacus](https://github.com/pytorch/opacus) is a comprehensive DP library supporting many layer types, Opaque takes a different approach:

- **LoRA-first design**: Optimized for the most common LLM fine-tuning scenario
- **Minimal complexity**: ~1,200 lines vs 5,000+ in Opacus
- **Linear layers only**: Perfect for LoRA adapters (A and B matrices)
- **Production-ready**: Hooks-based implementation, no loss wrapping issues
- **Modern defaults**: PRV accounting, compatible with HuggingFace PEFT

## Quick Start

```python
from opaque import PrivacyEngine
from peft import get_peft_model, LoraConfig
import torch

# Setup LoRA model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, lora_config)

# Make it private
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
privacy_engine = PrivacyEngine(accountant="prv")

model, optimizer = privacy_engine.make_private(
    model=model,
    optimizer=optimizer,
    target_epsilon=8.0,
    target_delta=1e-5,
    epochs=3,
    max_grad_norm=1.0,
    batch_size=32,
    sample_size=len(train_dataset),
)

# Train normally
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()

# Check privacy spent
epsilon = privacy_engine.get_epsilon(delta=1e-5)
print(f"Privacy cost: ε = {epsilon:.2f}")
```

## Installation

```bash
pip install opaque-dp
```

Or from source:
```bash
git clone https://github.com/yourusername/opaque.git
cd opaque
pip install -e .
```

## Features

- ✅ **Hooks-based grad sampling**: Stable, production-ready
- ✅ **Functorch mode**: Optional faster alternative
- ✅ **RDP and PRV accounting**: Tight privacy bounds
- ✅ **Compatible with PEFT**: Works with any LoRA implementation
- ✅ **No DataLoader wrapping**: Use standard HuggingFace trainers
- ✅ **Auditable codebase**: ~1,200 lines, easy to verify

## What Opaque Doesn't Do

Opaque is intentionally limited to make it simple and maintainable:

- ❌ Conv, RNN, Embedding, Normalization layers (use Opacus)
- ❌ Full fine-tuning (LoRA only)
- ❌ Distributed training (coming in v2)
- ❌ Ghost clipping (hooks are more stable)

If you need these features, use [Opacus](https://github.com/pytorch/opacus) instead!

## Documentation

- [API Reference](docs/api.md)
- [Privacy Accounting Guide](docs/accounting.md)
- [Examples](examples/)
- [Migration from Opacus](docs/migration.md)

## Development

Opaque is extracted and simplified from [Opacus](../opacus). The reference implementation is kept nearby for easy comparison.

```bash
# Project structure
external/
├── opaque/     # This project
└── opacus/     # Reference implementation
```

## License

Apache 2.0 (same as Opacus)

## Citation

If you use Opaque, please cite both this work and the original Opacus paper:

```bibtex
@article{opacus,
  title={Opacus: User-Friendly Differential Privacy Library in PyTorch},
  author={Yousefpour, Ashkan and Shilov, Igor and Sablayrolles, Alexandre and others},
  journal={arXiv preprint arXiv:2109.12298},
  year={2021}
}
```

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md)

## Acknowledgments

Opaque builds on the excellent work of the [Opacus team](https://github.com/pytorch/opacus) at Meta AI.
