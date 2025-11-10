# LoRA Fine-tuning

!!! warning "Under Construction"
    This page will be updated as LoRA integration is implemented.

## Why LoRA for DP?

[LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation) is particularly well-suited for differential privacy:

- **Efficiency**: DP overhead ~2x (vs. ~10x for full fine-tuning)
- **Memory**: Per-example gradients only for small adapter weights
- **Quality**: Often matches full fine-tuning performance

## Coming Soon

- LoRA + DP-SGD tutorial
- Integration with Hugging Face PEFT
- LLM fine-tuning examples
