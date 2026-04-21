# opaque

Meta-package for the Opaque functional DP-SGD library for PyTorch. Provides a
small curated facade (`opaque.__version__`, `opaque.patch_all`) over the
independent `opaque-*` distributions.

Installing `opaque` pulls in the default bundle:

- `opaque-core` — RNG, pytree, clipping, sampling, distributed, profiling
- `opaque-dpsgd` — Gaussian / truncated Gaussian noise, per-group, AdamW-BC
- `opaque-auditing` — empirical privacy auditing (one-run, coin-flip, loss attacks)
- `opaque-accounting` — PLD-based privacy accounting (Rust-backed; PyO3 extension mounted at `opaque.accounting._native`)

## Extras

```bash
pip install "opaque[mf]"            # + opaque-mf (correlated-noise mechanisms, AdamW-JME)
pip install "opaque[performance]"   # + opaque-performance (fused Triton kernels, checkpoint patches)
pip install "opaque[huggingface]"   # + opaque-huggingface (transformers patches) + performance
pip install "opaque[auditing]"      # + opaque-auditing (already in default bundle; explicit alias)
pip install "opaque[optimizers]"    # + torchopt wiring for AdamW-BC / AdamW-JME
pip install "opaque[all]"           # everything above
```

Sub-packages are pinned with `==<ver>` to avoid mix-and-match skew.

## Curated facade

```python
import opaque

print(opaque.__version__)
opaque.patch_all()                     # apply all available perf + HF patches
opaque.patch_all(skip={"huggingface"}) # apply only perf patches
```

Set `OPAQUE_SKIP_COMPAT_PATCHES` to globally disable subsystems:

```bash
OPAQUE_SKIP_COMPAT_PATCHES=all python train.py           # full opt-out
OPAQUE_SKIP_COMPAT_PATCHES=huggingface python train.py   # skip HF patches only
OPAQUE_SKIP_COMPAT_PATCHES=performance,huggingface ...   # skip both
```

## Import layout

Each distribution owns its own namespace root with a real `__init__.py`:

```
opaque.core.{clipping,sampling,noise,random,distributed,profiling,utils}  <- opaque-core
opaque.dpsgd.{noise,clipping,sampling,optimizers}                         <- opaque-dpsgd
opaque.mf.{noise,sampling,optimizers}                                     <- opaque-mf
opaque.auditing                                                           <- opaque-auditing
opaque.performance.{kernels,torch}                                        <- opaque-performance
opaque.huggingface.{patches,trainer,callbacks,integrations,data,models}   <- opaque-huggingface
opaque.accounting (._native)                                              <- opaque-accounting
```

`opaque` itself is a PEP 420 namespace composed with `pkgutil.extend_path`
in the umbrella `__init__.py`, so sub-packages install and import cleanly on
their own (`pip install opaque-dpsgd` → `import opaque.dpsgd` works).

## Development

From monorepo root:

```bash
uv sync --group dev --all-packages --extra all
uv run pytest -m "not cuda and not mps and not slow"
```

Each subpackage can be tested in isolation:

```bash
uv run pytest packages/opaque-core/tests/
uv run pytest packages/opaque-dpsgd/tests/
uv run pytest packages/opaque-mf/tests/
```
