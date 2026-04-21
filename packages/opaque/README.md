# opaque

Metadata-only meta-package for the Opaque functional DP training library for
PyTorch. Installing `opaque` pulls in the default bundle of sub-packages; the
umbrella distribution itself ships no code.

The `opaque` namespace is a pure [PEP 420] implicit namespace: each sub-package
(`opaque-core`, `opaque-dpsgd`, `opaque-dpftrl`, …) installs under
`opaque.<name>/` with no `__init__.py`. There is **no** `opaque.patch_all()` and
no `OPAQUE_SKIP_COMPAT_PATCHES` env var — each sub-package that ships patches
applies them on import (`import opaque.performance`, `import opaque.huggingface`)
unless its own `OPAQUE_SKIP_*` env vars disable them.

## Default bundle

- `opaque-core` — RNG, pytree, clipping, `empty_collate`, base noise types
- `opaque-dpsgd` — Gaussian / truncated Gaussian noise, per-group, AdamW-BC, Poisson samplers
- `opaque-auditing` — empirical privacy auditing (one-run, coin-flip, loss attacks)
- `opaque-accounting` — PLD-based privacy accounting (Rust-backed; PyO3 extension mounted at `opaque.accounting._native`)

## Extras

```bash
pip install "opaque[dpftrl]"        # + opaque-dpftrl (correlated-noise mechanisms, AdamW-JME)
pip install "opaque[performance]"   # + opaque-performance (fused Triton kernels, checkpoint patches, HF kernel patches)
pip install "opaque[huggingface]"   # + opaque-huggingface (transformers compat patches) + performance
pip install "opaque[auditing]"      # + opaque-auditing (already in default bundle; explicit alias)
pip install "opaque[all]"           # everything above, including [optimizers] extras
```

Sub-packages are pinned with `==<ver>` to avoid mix-and-match skew.

## Patching

`opaque.performance` and `opaque.huggingface` apply their patches automatically
at import time. Disable with sub-package-specific env vars:

```bash
OPAQUE_SKIP_PYTORCH_PATCHES=all python train.py          # disable performance patches
OPAQUE_SKIP_TRANSFORMERS_PATCHES=all python train.py     # disable huggingface compat patches
```

See the docs of each sub-package for the full list of tokens.

## Import layout

```
opaque.core.{clipping,sampling,noise,random,pytree}        <- opaque-core
opaque.distributed.{collectives,gradients,state,shard}     <- opaque-core
opaque.functional                                           <- opaque-core
opaque.dpsgd.{noise,clipping,sampling,optimizers}          <- opaque-dpsgd
opaque.dpftrl.{noise,sampling,optimizers}                  <- opaque-dpftrl
opaque.auditing                                            <- opaque-auditing
opaque.performance.{kernels,torch,profiling,huggingface}   <- opaque-performance
opaque.huggingface.{patches,trainer,callbacks,...}         <- opaque-huggingface
opaque.accounting (._native)                               <- opaque-accounting
```

## Development

From the monorepo root:

```bash
uv sync --group dev --all-packages --extra all
uv run pytest -m "not cuda and not mps and not slow"
```

Each sub-package can be tested in isolation:

```bash
uv run pytest packages/opaque-core/tests/
uv run pytest packages/opaque-dpsgd/tests/
uv run pytest packages/opaque-dpftrl/tests/
```

[PEP 420]: https://peps.python.org/pep-0420/
