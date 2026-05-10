# tests/integration/

End-to-end integration tests that exercise multiple opaque wheels
together. The full workspace install (`uv sync --all-packages --extra
all`) brings in every wheel, so the imports always resolve when
running locally or in CI.

These tests are **not** shipped inside any wheel's tarball.

## Layout

```
tests/integration/
├── README.md
├── test_dpsgd_pipeline.py   — full DP-SGD step (clip → gaussian_noise → update)
│                              on a patched LoRA model. Synthetic + Qwen2
│                              variants in one file.
├── test_dpftrl_pipeline.py  — full DP-FTRL step (clip → mf_noise → update)
│                              on a patched LoRA model. Synthetic + Qwen2.
└── dpsgd_dpftrl/            — DP-SGD ↔ DP-FTRL cross-stack tests
    ├── accounting/          — composing per-stack accountants and round-tripping
    │                          mixed processes through opaque.serialization
    ├── distributed/         — DDP + DP step. CUDA-marked.
    └── noise/               — comparing band-MF / mf_noise vs the DP-SGD
                                Gaussian baseline on the same inputs
```

## What lives here vs. in a wheel's `tests/`

A test belongs in `packages/<wheel>/tests/` when every package it
imports is in `<wheel>`'s transitive dep cone — see
`tests/contracts/test_test_placement.py`. That covers the vast majority
of tests.

A test belongs here when it imports across two or more wheels with
**mutual non-dependency**, or when it exercises the end-to-end DP
pipeline (clipping + noise + optimizer + patches all in one).

Patches are part of the framework's normal usage, so applying them in
an integration test isn't reason on its own to call it a "patches
test" — it's still a DP-pipeline test.

## Markers

- **No marker**: synthetic-config integration tests run in the PR gate
  (CPU + MPS).
- **`slow`**: real-HF integration tests download weights from HF Hub on
  first run; excluded from the PR gate, run on push to main.
- **`cuda`**: distributed / multi-GPU tests; auto-skip on hosts without
  CUDA. The `dpsgd_dpftrl/distributed/` tests use this marker.

## Discovery

Pytest picks these up automatically: the root `pyproject.toml`
`testpaths` is `["packages", "tests"]`. Run them along with the rest of
the suite or in isolation:

```bash
uv run pytest tests/integration/                         # everything not slow / cuda
uv run pytest tests/integration/ -m slow                 # slow tests (HF downloads)
uv run pytest tests/integration/ -m cuda                 # CUDA tests (multi-GPU DDP)
uv run pytest tests/integration/dpsgd_dpftrl/            # cross-stack only
```
