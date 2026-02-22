# Tutorials

These Jupyter notebooks provide hands-on practice with Opaque's DP-SGD
components. Each tutorial is self-contained and task-oriented: you will
build something concrete and see it work.

For conceptual explanations, see the [User Guide](../user-guide/index.md).
For complete function signatures, see the
[API Reference](../api/index.md).

## Notebooks

| Tutorial | Task | Components exercised |
|----------|------|---------------------|
| [DP-SGD Training](dp_sgd_training.ipynb) | Train a model with differential privacy from scratch | `clipped_grad`, `gaussian_noise`, `make_functional`, accounting, calibration |
| [Privacy Accounting & Calibration](accounting_and_calibration.ipynb) | Explore the accounting API, compare mechanisms, calibrate noise | `DpProcess`, composition, `calibrate`, privacy metrics |
| [Fine-tuning an LLM](llm_finetuning.ipynb) | Fine-tune a HuggingFace model with DP and LoRA | `make_functional`, `partition_trainable`, PEFT, adaptive clipping |
| [Sampling & Microbatching](sampling_and_microbatching.ipynb) | Compare sampling strategies and tune microbatch size | `PoissonSampler`, `TruncatedPoissonSampler`, `microbatch_size` |
| [Privacy Auditing](privacy_auditing.ipynb) | Validate privacy guarantees empirically | `auditing.setup`, `auditing.evaluate`, `AuditResult`, bootstrap |
| [Distributed Training](distributed_training.ipynb) | Run DP-SGD across multiple GPUs | `sum_gradients`, `local_shard`, `PoissonSampler`, shared RNG key |

Most tutorials run on CPU in reasonable time using small datasets.
The distributed training tutorial requires multiple GPUs and `torchrun`.

## Running the tutorials

Install Opaque and its tutorial dependencies:

```bash
git clone https://github.com/JetBrains-Research/opaque.git
cd opaque
uv sync --group docs
jupyter lab docs/tutorials/
```

Or install in an existing environment:

```bash
pip install opaque-dp[tutorials]
```
