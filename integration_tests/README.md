# integration_tests/

Cross-wheel integration tests that don't fit any single wheel's
dependency cone.

Tests in `packages/<wheel>/tests/` must only import from packages that
`<wheel>` depends on (directly or transitively) — see
`contracts/test_test_placement.py`. When two wheels with mutual
non-dependency need to be exercised together (DP-SGD ↔ DP-FTRL,
DP-SGD ↔ patches, DP-FTRL ↔ patches), the integration test has no
natural home in any wheel. It lives here.

## Layout

```
integration_tests/
└── <pair>/<concern>/test_*.py
```

Examples:

- `dpsgd_dpftrl/accounting/test_state_dict.py` — round-trips an
  accountant containing both a DP-SGD `gaussian` process and a DP-FTRL
  `band_mf` process through `opaque.serialization.state_dict`.
- `dpsgd_dpftrl/noise/test_band_mf_noise.py` — compares DP-FTRL
  `band_mf` noise against the DP-SGD `gaussian` baseline on the same
  inputs.
- `dpsgd_patches/...` — exercises HF model patches through a DP-SGD
  clipping → noise → optimizer step.

## Discovery

Pytest picks these up automatically: the root `pyproject.toml`
`testpaths` includes `integration_tests`. The full workspace install
(`uv sync --all-packages --extra all`) brings in every wheel, so the
imports always resolve when running locally or in CI.

These tests are **not** shipped inside any wheel's tarball.
