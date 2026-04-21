# AGENTS.md

Agent briefing for the **Opaque** monorepo — a functional DP-SGD / DP-FTRL
library for PyTorch. See `README.md` and `CONTRIBUTING.md` for user docs.

## Project snapshot

- **Language**: Python 3.11+ (< 3.13) + Rust stable (≥ 1.70)
- **Package manager**: `uv`
- **Hardware**: H200 80GB GPU for training runs; CPU/MPS for most tests
- **Testing**: `pytest` (Python, ~1200 tests) + `cargo test` (Rust)

Opaque provides composable primitives for differentially private model
training in PyTorch. Built on `torch.func` (vmap, grad), every component
uses explicit state — no hooks, no subclassing, no hidden mutation.

## Packages (Option B layout)

Each sub-package owns its own namespace root under `opaque.*` with a real
`__init__.py`. `opaque/` itself is a PEP 420 namespace composed via
`pkgutil.extend_path` in the umbrella.

| Distribution | Import root | Purpose | Build |
| --- | --- | --- | --- |
| `opaque` (umbrella) | `opaque` | curated facade (`patch_all`, `__version__`) | setuptools |
| `opaque-core` | `opaque.core` | RNG, pytree, clipping, sampling, distributed, profiling, utils | setuptools |
| `opaque-dpsgd` | `opaque.dpsgd` | Gaussian / truncated-Gaussian / per-group noise, AdamW-BC, truncated Poisson, adaptive + auto clipping | setuptools |
| `opaque-mf` | `opaque.mf` | MF mechanisms (BLT, BSR, BiSR, band-MF, JME, λ-CGD), AdamW-JME, MF samplers | setuptools |
| `opaque-auditing` | `opaque.auditing` | empirical privacy auditing (one-run, coin-flip, loss attacks) | setuptools |
| `opaque-performance` | `opaque.performance` | fused Triton kernels (`.kernels`) + PyTorch patches (`.torch.checkpoint`) | setuptools |
| `opaque-huggingface` | `opaque.huggingface` | HF Transformers patches (`.patches`), future `trainer/callbacks/integrations` | setuptools |
| `opaque-accounting` | `opaque.accounting` | PLD privacy accounting (PyO3 extension at `opaque.accounting.opaque_accounting`, aliased as `_native`) | maturin |

Sub-packages are independently installable; `pip install opaque-dpsgd`
gives a working `import opaque.dpsgd` without pulling the umbrella.

## Umbrella contract

Three rules the umbrella upholds (rule 1 enforced in CI; rules 2 and 3 are design invariants):

1. **Only `packages/opaque` ships `src/opaque/__init__.py`.** Every other
   distribution leaves `opaque/` as a PEP 420 namespace. The umbrella uses
   `pkgutil.extend_path(__path__, __name__)` so it composes with sub-packages
   installed elsewhere on `sys.path` — `import opaque` works whether or not
   the umbrella is installed, and all sub-namespaces remain reachable.
2. **The umbrella is a facade, not a re-export.** It exposes exactly two
   names: `opaque.__version__` and `opaque.patch_all()`. It does not
   re-export `opaque.clip`, `opaque.noise`, `opaque.sampling`, etc. — each
   algorithm owns its own dotted path (`opaque.dpsgd.noise.gaussian`,
   `opaque.mf.sampling.b_min_sep`, …) and that's intentional. This matches
   the convention of `zope.*`, `google.cloud.*`, `azure.*`, `sphinxcontrib.*`.
3. **`opaque.accounting` stays a standalone package.** It is not split into
   `opaque.dpsgd.accounting` / `opaque.mf.accounting`: the PLD library is a
   general-purpose primitive consumed by (but not exclusive to) MF. The
   Rust/PyO3 extension is mounted at `opaque.accounting.opaque_accounting`
   (the `.so` filename matches the Rust crate) and aliased as `_native` in
   the package's `__init__.py` so internal code can keep using the short name.

## Key commands

```bash
uv sync --group dev --all-packages --extra all     # test suite: pytest, ruff, scipy + all package extras
uv sync --group examples --all-packages --extra all  # examples/: torchopt, datasets, wandb + all package extras
uv run pytest -m "not cuda and not mps and not slow"   # PR-equivalent suite
uv run pytest -m "slow"                           # slow tests (run on push to main)
uv run ruff check packages/                      # lint
uv run ruff format --check packages/             # format check
cargo test --workspace                           # Rust tests
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

### Dependency groups

The root `pyproject.toml` keeps only two dev-facing dependency groups:

- `dev` — pytest, pytest-cov, ruff, scipy (statistical tests).
- `docs` — mkdocs stack.

Everything else lives in the relevant package's
`[project.optional-dependencies]`:

| Extra | Pulls in |
| --- | --- |
| `opaque-huggingface[peft]` | `peft`, `transformers`, `datasets` |
| `opaque-huggingface[kernels]` | HF + `opaque-performance[kernels]` |
| `opaque-performance[kernels]` | `triton` |
| `opaque-dpsgd[optimizers]` | `torchopt` |
| `opaque-mf[optimizers]` | `torchopt` |
| `opaque-accounting[cross-validation]` | `dp-accounting`, `riskcal` |
| `opaque[all]` | everything |

## Patching model (opt-in)

Patches are not applied on import. Call `opaque.patch_all()` once at
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

- Rust crate name: `opaque_accounting` (Cargo `[lib].name`, valid Rust
  identifier; used by doctests via `use opaque_accounting::...`).
- PyO3 `#[pymodule]` function: `opaque_accounting` → compiled artifact is
  `opaque/accounting/opaque_accounting.abi3.so`.
- maturin `module-name = "opaque.accounting.opaque_accounting"`,
  `python-packages = ["opaque.accounting"]`.
- The Python facade at `opaque.accounting/__init__.py` does
  `from . import opaque_accounting as _native`; all submodules continue to
  use `_native` as the private-impl alias. No top-level `opaque_accounting`
  Python module exists anywhere.

### Partition policy

`opaque.core` holds algorithm-agnostic primitives. Anything that only one
algorithm would construct (DP-SGD adaptive/auto clipping, truncated
Poisson; MF b-min-sep / cyclic / balls-in-bins / sequential sampling,
BLT/BSR/BiSR/band-MF/JME/λ-CGD noise) lives with that algorithm.

### Test markers

Three orthogonal markers, declared in the root `pyproject.toml`:

- `cuda` — test needs CUDA; auto-skipped on non-CUDA hosts.
- `mps` — test needs Apple Metal; auto-skipped on non-MPS hosts.
- `slow` — test takes >5 s on CPU; excluded from PR CI (`and not slow`)
  and run on pushes to `main` (the CI job strips the `and not slow`
  clause conditionally).

Gated HuggingFace models use `@requires_hf_auth` imported from
`packages/opaque-huggingface/tests/huggingface/_helpers.py`. It is a
`skipif(not has_hf_token())` mark, not a pytest marker. Set `HF_TOKEN`
(or `HUGGINGFACEHUB_API_TOKEN` / `HUGGINGFACE_TOKEN`) to run them.

CI lane marker expressions:

- CPU (Ubuntu): `-m "not cuda and not mps and not slow"`.
- MPS (macOS): `-m "not cuda and not slow"`.
- CUDA (self-hosted): `-m "cuda"`.
- On push to `main` the CPU/MPS jobs drop `and not slow` so `slow` tests
  run there.

### Supported HF model families

LLaMA / Mistral / Qwen2 / Qwen3 / Phi-3 / Gemma / Gemma2 / Granite /
Cohere / Cohere2 / DeepSeek (inherits LLaMA). See
`docs/user-guide/huggingface.md`.

## Non-obvious notes

- `uv sync` triggers a full Rust build of `opaque-accounting` via maturin
  (first run ~30s; cached afterwards).
- Pure library — no application server or database; testing is entirely
  `pytest` + `cargo test`.
- CUDA/MPS tests auto-skip when the accelerator is unavailable (marker-
  driven). HuggingFace compat tests also skip via `pytest.importorskip()`
  when `transformers` / `peft` aren't installed.
- CI guardrail: a single shell step in `.github/workflows/ci.yml`
  enforces that no sub-package ships `src/opaque/__init__.py` (the
  PEP 420 invariant). Legacy-token and negative-import checks were
  refactor-diary guards and have been removed now that the migration
  is complete.

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
