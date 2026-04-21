# AGENTS.md

Agent briefing for the **Opaque** monorepo — a functional DP-SGD / DP-FTRL
library for PyTorch. See `README.md` and `CONTRIBUTING.md` for user docs.

## Project snapshot

- **Language**: Python 3.11+ (< 3.13) + Rust stable (≥ 1.70)
- **Package manager**: `uv`
- **Hardware**: H200 80GB GPU for training runs; CPU/MPS for most tests
- **Testing**: `pytest` (Python, ~900+ tests) + `cargo test` (Rust, ~255 tests)

Opaque provides composable primitives for differentially private model
training in PyTorch. Built on `torch.func` (vmap, grad), every component
uses explicit state — no hooks, no subclassing, no hidden mutation.

## Packages (Option B layout)

Each sub-package owns its own namespace root under `opaque.*` with a real
`__init__.py`. `opaque/` itself is a PEP 420 namespace composed via
`pkgutil.extend_path` in the umbrella.

| Distribution | Import root | Purpose | Build |
|---|---|---|---|
| `opaque` (umbrella) | `opaque` | curated facade (`patch_all`, `__version__`) | setuptools |
| `opaque-core` | `opaque.core` | RNG, pytree, clipping, sampling, distributed, profiling, utils | setuptools |
| `opaque-dpsgd` | `opaque.dpsgd` | Gaussian / truncated-Gaussian / per-group noise, AdamW-BC, truncated Poisson, adaptive + auto clipping | setuptools |
| `opaque-mf` | `opaque.mf` | MF mechanisms (BLT, BSR, BiSR, band-MF, JME, λ-CGD), AdamW-JME, MF samplers | setuptools |
| `opaque-auditing` | `opaque.auditing` | empirical privacy auditing (one-run, coin-flip, loss attacks) | setuptools |
| `opaque-performance` | `opaque.performance` | fused Triton kernels (`.kernels`) + PyTorch patches (`.torch.checkpoint`) | setuptools |
| `opaque-huggingface` | `opaque.huggingface` | HF Transformers patches (`.patches`), future `trainer/callbacks/integrations` | setuptools |
| `opaque-accounting` | `opaque.accounting` | PLD privacy accounting (PyO3 extension mounted at `opaque.accounting._native`) | maturin |

Sub-packages are independently installable; `pip install opaque-dpsgd`
gives a working `import opaque.dpsgd` without pulling the umbrella.

## Key commands

```bash
uv sync --group dev --group compat      # install full workspace + HF compat extras
uv run pytest -m "not gpu"              # non-GPU Python tests (~900+ tests, ~13 min on CPU)
uv run ruff check packages/             # lint
uv run ruff format --check packages/    # format check
cargo test --workspace                  # Rust tests (~255 tests, ~3 min)
python scripts/check_namespaces.py      # CI: no stray opaque/__init__.py, no legacy tokens
python scripts/check_negative_imports.py  # CI: opaque_accounting / opaque.compat must be gone
```

Per-package tests:

```bash
uv run pytest packages/opaque-core/tests/
uv run pytest packages/opaque-dpsgd/tests/
uv run pytest packages/opaque-mf/tests/
uv run pytest packages/opaque-auditing/tests/
uv run pytest packages/opaque-performance/tests/
uv run pytest packages/opaque-huggingface/tests/
uv run pytest packages/opaque-accounting/tests/
uv run pytest packages/opaque/tests/       # umbrella facade tests
```

## Installation matrix

```bash
pip install opaque-core                  # primitives only
pip install opaque-dpsgd                 # + DP-SGD mechanisms
pip install opaque-mf                    # + MF (DP-FTRL) mechanisms
pip install opaque-accounting            # + Rust PLD accounting
pip install opaque-performance[kernels]  # + Triton fused kernels
pip install opaque-huggingface[peft]     # + HF patches + PEFT extras
pip install "opaque[all]"                # everything via umbrella
```

## Patching model (opt-in)

Patches are no longer applied on import. Call `opaque.patch_all()` once at
startup (or sub-system equivalents):

```python
import opaque
opaque.patch_all()                       # performance + HF patches
opaque.patch_all(skip={"huggingface"})   # performance only
```

Control via `OPAQUE_SKIP_COMPAT_PATCHES` (`all`, `huggingface`,
`performance`, comma-combo). Finer-grained vars still apply:
`OPAQUE_SKIP_PYTORCH_PATCHES`, `OPAQUE_SKIP_TRANSFORMERS_PATCHES`,
`OPAQUE_SKIP_TRANSFORMERS_VMAP_PATCHES`,
`OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES`.

## Key architectural notes

### Kernel pattern (`opaque.performance.kernels`)

Triton kernels use a two-level `autograd.Function` for `vmap(grad())`
support: `Opaque_Foo` main entry + `_FooBackward` with their own `vmap()`
methods. New-style API (`setup_context()`); **not** compatible with
`@torch.amp.custom_fwd`/`@custom_bwd` (PyTorch #132388). Forward runs
under caller's autocast, backward has autocast OFF.

### Accounting native module

The Rust/PyO3 extension is mounted at `opaque.accounting._native` via
maturin's `module-name = "opaque.accounting._native"`. The Python facade
at `opaque.accounting/__init__.py` re-exports an explicit `__all__`. No
top-level `opaque_accounting` module exists anywhere.

### Partition policy

`opaque.core` holds algorithm-agnostic primitives. Anything that only one
algorithm would construct (DP-SGD adaptive/auto clipping, truncated
Poisson; MF b-min-sep / cyclic / balls-in-bins / sequential sampling,
BLT/BSR/BiSR/band-MF/JME/λ-CGD noise) lives with that algorithm.

### Supported HF model families

LLaMA / Mistral / Qwen2 / Qwen3 / Phi-3 / Gemma / Gemma2 / Granite /
Cohere / Cohere2 / DeepSeek (inherits LLaMA). See
`docs/user-guide/huggingface.md`.

## Non-obvious notes

- `uv sync` triggers a full Rust build of `opaque-accounting` via maturin
  (first run ~30s; cached afterwards).
- Pure library — no application server or database; testing is entirely
  `pytest` + `cargo test`.
- GPU tests marked `@pytest.mark.gpu` and auto-skip without a GPU; use
  `-m "not gpu"` to exclude.
- HuggingFace compat tests skip via `pytest.importorskip()` when
  `transformers` / `peft` aren't installed.
- `test_deep_heterogeneous_tree_no_recursion_error` in accounting is slow
  (~2 min on CPU).
- CI guardrails: `scripts/check_namespaces.py` (no stray inits, no legacy
  tokens) + `scripts/check_negative_imports.py` (`opaque_accounting` and
  `opaque.compat` must be gone).

## Experiment tracking (W&B)

- Entity: `federated-compute`, Project: `opaque`
- Instance: `https://jetbrains.wandb.io`
- Always `PYTHONUNBUFFERED=1` for real-time output
- Offline by default; goes online when `WANDB_API_KEY` is set
- Env: `WANDB_API_KEY`, `WANDB_BASE_URL`, `WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_NAME`

## Training entry points

```bash
uv run python examples/train_causal_lm.py --preset mellum-kstack --max-steps 100
uv run python examples/train_dp_ftrl.py   # MF-based DP-FTRL training
```

Baseline without kernel patches:

```bash
OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES=all \
  uv run python examples/train_causal_lm.py --preset mellum-kstack --max-steps 100
```

## Documentation

- User-facing: `docs/` (MkDocs, Material theme)
- Development: `docs/development/`
- This file (`AGENTS.md`) is agent-oriented; users should read
  `docs/index.md` or `README.md`.
