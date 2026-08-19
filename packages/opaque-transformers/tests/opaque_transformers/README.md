# `opaque_transformers` integration tests

Trainer and HF-compat tests for `opaque-transformers` live here. The directory
is not named `transformers` to avoid shadowing the installed package under
pytest’s importlib mode.

Implementation code under test is in `opaque.api.transformers.trainer`;
public imports should use `opaque.transformers` or
`opaque.transformers.trainer`.
