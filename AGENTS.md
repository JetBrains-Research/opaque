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

Every sub-package lives under `opaque.*` as a PEP 420 implicit namespace
(no `__init__.py` at the `opaque/` level in any distribution). User-facing
primitives are hoisted to the namespace root; internal primitives stay
nested under `opaque.core.*`.

| Distribution | Import roots | Purpose | Build |
| --- | --- | --- | --- |
| `opaque` | — | pins the curated sub-package bundle; extras add the rest | setuptools |
| `opaque-core` | `opaque.core`, `opaque.functional`, `opaque.distributed`, `opaque.clipping`, `opaque.scheduling` | RNG, pytree, clipping, step-indexed schedules + warmup composition, `PerGroup`, `empty_collate`, `make_functional`, DDP plumbing | setuptools |
| `opaque-dpsgd` | `opaque.dpsgd` | Gaussian / truncated-Gaussian / per-group noise, AdamW-BC, Poisson + truncated-Poisson samplers, adaptive + auto clipping | setuptools |
| `opaque-dpftrl` | `opaque.dpftrl` | DP-FTRL mechanisms (BLT, BSR, BiSR, band-MF, λ-CGD), private second moments, cyclic Poisson + b-min-sep + balls-in-bins + sequential samplers | setuptools |
| `opaque-auditing` | `opaque.auditing` | empirical privacy auditing (one-run, coin-flip, loss attacks) | setuptools |
| `opaque-performance` | `opaque.performance`, `opaque.performance.huggingface`, `opaque.performance.profiling` | fused Triton kernels, PyTorch checkpoint patches, HF model kernel patches, memory/step profiler | setuptools |
| `opaque-transformers` | `opaque.transformers` | HF Transformers compat patches (vmap-safe attention, KV cache, Poisson collator) | setuptools |
| `opaque-accounting` | `opaque.accounting` | PLD privacy accounting (PyO3 extension at `opaque.accounting.opaque_accounting`, aliased as `_native`) | maturin |

Sub-packages are independently installable; `pip install opaque-dpsgd`
gives a working `import opaque.dpsgd` without pulling any other package.

## Namespace contract

Three rules (rule 1 enforced in CI; rules 2 and 3 are design invariants):

1. **No package ships `src/opaque/__init__.py`.** `opaque` is a pure PEP 420
   namespace. Each sub-package installs under `opaque/<name>/` and composes
   automatically. CI enforces that no `src/opaque/__init__.py` slips in.
2. **The root `opaque` distribution ships no code.** Installing `opaque[all]`
   pulls in the sub-packages, but the root distribution itself exposes no
   names. Each algorithm owns its own dotted path
   (`opaque.dpsgd.noise.gaussian`, `opaque.dpftrl.sampling.b_min_sep`, …)
   and that's intentional. This matches the convention of `zope.*`,
   `google.cloud.*`, `azure.*`, `sphinxcontrib.*`.
3. **`opaque.accounting` stays a standalone package.** It is not split into
   `opaque.dpsgd.accounting` / `opaque.dpftrl.accounting`: the PLD library is a
   general-purpose primitive consumed by (but not exclusive to) MF. The
   Rust/PyO3 extension is mounted at `opaque.accounting.opaque_accounting`
   (the `.so` filename matches the Rust crate) and aliased as `_native` in
   the package's `__init__.py` so internal code can keep using the short name.

## Pull requests

The repo squash-merges. The PR title becomes the commit subject; the
PR body becomes the commit body (repo-level squash setting =
`PR_TITLE` + `PR_BODY`). Both feed `git-cliff` when it builds the draft
Release body on the next main merge.

**Title** — Conventional Commits form `<type>(scope): <imperative subject>`:

- Types `git-cliff` categorizes (see [cliff.toml](cliff.toml)):
  `feat` / `add` → Added, `fix` → Fixed,
  `refactor` / `change` / `perf` → Changed, `docs` → Documentation,
  `test` → Tests, `ci` / `build` → CI/CD, `delete` → Removed,
  `chore` / `style` → skipped.
- Scope is optional but encouraged — e.g., `fix(accounting): …`.
- Breaking change: append `!` (`feat(dpsgd)!: …`) or include a
  `BREAKING CHANGE:` footer in the body.
- Subject starts with a lowercase letter and reads as an imperative
  (`add`, `fix`, `remove`) — not past-tense.
- The PR-gate workflow runs `amannn/action-semantic-pull-request@v6`
  and fails the check if the title doesn't parse.

**Body** — short prose:

- 2–4 sentences of "why" + what the change does. This text lands in
  `git log` on main and feeds the AI release-note summary in
  `ci.yml`'s `upsert-draft` job.
- Keep it readable for a future spelunker; avoid checklist-only bodies.

**Gate** — on every push the PR workflow runs: tests (CPU + MPS),
rust-tests, smoke-imports, docs build, title check, autoformat. All 8
are required for merge. Preview wheels
(`0.X.Y.devN+pr.<num>.g<sha>`) build alongside and appear as
downloadable workflow artifacts on the run page (14-day retention).

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
uv run pytest packages/opaque-dpftrl/tests/
uv run pytest packages/opaque-auditing/tests/
uv run pytest packages/opaque-performance/tests/
uv run pytest packages/opaque-transformers/tests/
uv run pytest packages/opaque-accounting/tests/  # smoke; PLD factory tests live under dpsgd/dpftrl
```

## Installation matrix

```bash
pip install opaque-core                  # primitives only
pip install opaque-dpsgd                 # + DP-SGD mechanisms
pip install opaque-dpftrl                    # + MF (DP-FTRL) mechanisms
pip install opaque-accounting            # + Rust PLD accounting
pip install opaque-performance[kernels]  # + Triton fused kernels
pip install opaque-transformers[peft]     # + HF patches + PEFT extras
pip install "opaque[all]"                # everything
```

### Dependency groups

The root `pyproject.toml` keeps only two dev-facing dependency groups:

- `dev` — pytest, pytest-cov, ruff, scipy (statistical tests).
- `docs` — mkdocs stack.

Everything else lives in the relevant package's
`[project.optional-dependencies]`:

| Extra | Pulls in |
| --- | --- |
| `opaque-transformers[peft]` | `peft`, `transformers`, `datasets` |
| `opaque-transformers[kernels]` | HF + `opaque-performance[kernels]` |
| `opaque-performance[kernels]` | `triton` |
| `opaque-dpsgd[optimizers]` | `torchopt` |
| `opaque-dpftrl[optimizers]` | `torchopt` |
| `opaque-accounting[cross-validation]` | `dp-accounting`, `riskcal` |
| `opaque[all]` | everything |

## Patching model (on-import)

Importing `opaque.performance` or `opaque.transformers` automatically applies
their respective patches. There is no top-level `opaque.patch_all()` —
each sub-package owns its own patching. Disable selectively with
sub-package-specific env vars set **before** the import:

```bash
OPAQUE_SKIP_PYTORCH_PATCHES=all            # skip all opaque.performance patches
OPAQUE_SKIP_TRANSFORMERS_PATCHES=all       # skip all opaque.transformers compat patches
OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES=all # skip the HF kernel patches (performance side)
```

Fine-grained variables also apply:
`OPAQUE_SKIP_PYTORCH_CHECKPOINT_PATCHES`,
`OPAQUE_SKIP_TRANSFORMERS_VMAP_PATCHES`,
`OPAQUE_SKIP_TRANSFORMERS_DATA_PATCHES`.

Patch layers:

- `opaque.performance` — gradient-checkpointing for `torch.utils.checkpoint`
  + HF Triton kernel patches (SwiGLU, GeGLU, RoPE, fused CE, LoRA) via
  `opaque.performance.huggingface`.
- `opaque.transformers` — compatibility-only (vmap-safe attention, KV cache,
  Poisson-collator compat). Performance patches live in
  `opaque.performance.huggingface`.

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
BLT/BSR/BiSR/band-MF/λ-CGD noise, private second-moment streams) lives with that algorithm.

### Test markers

Three orthogonal markers, declared in the root `pyproject.toml`:

- `cuda` — test needs CUDA; auto-skipped on non-CUDA hosts.
- `mps` — test needs Apple Metal; auto-skipped on non-MPS hosts.
- `slow` — test takes >5 s on CPU; excluded from PR CI (`and not slow`)
  and run on pushes to `main` (the CI job strips the `and not slow`
  clause conditionally).

Gated HuggingFace models use `@requires_hf_auth` imported from
`packages/opaque-transformers/tests/huggingface/_helpers.py`. It is a
`skipif(not has_hf_token())` mark, not a pytest marker. Set `HF_TOKEN`
(or `HUGGINGFACEHUB_API_TOKEN` / `HUGGINGFACE_TOKEN`) to run them.

CI lane marker expressions:

- CPU (Ubuntu): `-m "not cuda and not mps and not slow"`.
- MPS (macOS): `-m "not cuda and not slow"`.
- CUDA (self-hosted): `-m "cuda"`.
- On push to `main` the CPU/MPS jobs drop `and not slow` so `slow` tests
  run there.

### Supported HF model families

LLaMA / Mistral / Ministral / Qwen2 / Qwen3 / SmolLM3 / OLMo2 / OLMo3 /
GLM4 / Phi-3 / Gemma / Gemma2 / Gemma3 (text) / Granite / Cohere / Cohere2 /
Exaone4 / DeepSeek (inherits LLaMA). Text-first; see
`docs/user-guide/huggingface.md`. Nemotron is deferred (no
`eager_attention_forward` and a non-gated `NemotronMLP` in 4.57.1).

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
- Keep user-facing docs and code comments diary-free: describe the
  current API and behavior, not the development history (file moves,
  package regroups, removed dependencies, planned-but-unimplemented
  features). Migration narrative belongs in PR bodies, the changelog,
  or `docs/development/`. Forward-references to features that don't
  yet exist in the codebase don't belong anywhere.
