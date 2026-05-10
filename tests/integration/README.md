# tests/integration/

Cross-wheel integration tests that don't fit any single wheel's
dependency cone. The full workspace install (`uv sync --all-packages
--extra all`) brings in every wheel, so the imports always resolve when
running locally or in CI.

These tests are **not** shipped inside any wheel's tarball.

## Layout

```
tests/integration/
├── dpsgd_dpftrl/           — DP-SGD ↔ DP-FTRL cross-stack tests
│   ├── accounting/
│   ├── distributed/
│   └── noise/
└── patches/                — patches × DP pipeline integration smoke tests
    ├── test_dpsgd_pipeline.py
    └── test_dpftrl_pipeline.py
```

## What lives here vs. in a wheel's `tests/`

A test belongs in `packages/<wheel>/tests/` when every package it
imports is in `<wheel>`'s transitive dep cone — see
`tests/contracts/test_test_placement.py`.

Examples:

- A patches test that does only `clipped_grad` (engine) — lives in
  `packages/opaque-patches/tests/` (patches depends on engine).
- A DP-FTRL test that exercises the MF noise mechanism alone — lives in
  `packages/opaque-dpftrl/tests/`.

A test belongs in `tests/integration/` when it imports across two or
more wheels with **mutual non-dependency**:

- `dpsgd_dpftrl/` — neither dpsgd nor dpftrl depends on the other; the
  test exercises both.
- `patches/test_dpsgd_pipeline.py` — patches and dpsgd are sibling wheels;
  patches doesn't depend on dpsgd. The test exercises a full DP-SGD step
  (clipping + ``gaussian_noise`` + manual update) on a patched HF LoRA
  model.
- `patches/test_dpftrl_pipeline.py` — symmetric for DP-FTRL
  (``mf_noise`` with identity strategy).

## Discovery

Pytest picks these up automatically: the root `pyproject.toml`
`testpaths` includes `tests`. Run them along with the rest of the
suite or in isolation:

```bash
uv run pytest tests/integration/
uv run pytest tests/integration/patches/
uv run pytest tests/integration/dpsgd_dpftrl/
```
