# opaque.mlx

`opaque-mlx` is the Apple Silicon provider for Opaque's backend-neutral engine.
Install `opaque[mlx]` or `opaque-mlx` with the mechanism wheels your program
uses, then choose it with `set_backend("mlx")` or pass an MLX array to a
backend-bearing Opaque call.

The provider implements the portable `opaque.ops`, `opaque.pytree`,
`opaque.random`, `opaque.autodiff`, `opaque.execution`, serialization, and
runtime surfaces. MLX has no native `float64`; `opaque.ops.float64()` is
therefore unsupported rather than host-emulated.

## Functional modules

`opaque.mlx.functional.make_functional(module)` returns a callable accepting
an explicit MLX parameter pytree and the initial parameters. The caller-owned
module is restored after every invocation.

For frozen-base LoRA training, pass `partition_trainable=True`. The returned
trainable and frozen trees remain explicit; bind their merged final value back
to the module before ordinary model evaluation or export. The full
script-local MLX-LM integration is demonstrated by
[`examples/train_dpsgd_mlx.py`](https://github.com/JetBrains-Research/opaque/blob/main/examples/train_dpsgd_mlx.py).
It is intentionally not a public `opaque-mlx` model/data adapter API.

## MLX-LM model inputs

The causal-LM example accepts a Hugging Face repository ID or local MLX model
directory and delegates construction to `mlx_lm.load`. Support is limited to
architectures implemented by the installed `mlx-lm` version or repositories
that provide explicitly trusted custom MLX model code. Tokenizer and custom
model code require `--trust-remote-code`.

The example supports `float32`, `float16`, and `bfloat16`. MLX has no native
`float64`. Quantized repositories are governed by MLX-LM's model and adapter
support rather than by the Opaque provider.

## Distributed lifecycle

`opaque.mlx.distributed.initialize()` starts and registers an MLX group.
Applications that already own MLX initialization use `register_group(group)`;
`clear_group()` stops Opaque from issuing further collectives without destroying
the MLX global group. See [Distributed Training](../user-guide/distributed.md).

The MLX-LM DP-SGD example currently requires record sharding and Poisson
sampling in distributed runs. It clips local per-record gradients, synchronizes
adaptive state and auxiliary values, sums clipped gradients, then applies
Opaque's coordinated-noise protocol. It does not use MLX-LM's ordinary gradient
averaging. Random allocation, K-out-of-T, and unsharded parallel Poisson are
rejected because their distributed privacy-equivalence paths are not
implemented in the example. Rank zero owns logging, checkpoints, evaluation,
and adapter writes. A multi-process launcher and compatible MLX transport are
host/runtime responsibilities.

## Device capabilities

`opaque.mlx.device.device_capabilities()` reports the MLX unified-memory
capabilities exposed by the installed runtime. Values unavailable from MLX are
reported as unsupported rather than fabricated.