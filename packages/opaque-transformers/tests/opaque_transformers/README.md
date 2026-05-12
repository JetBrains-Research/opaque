# `opaque_transformers` integration tests

Trainer and HF-compat tests for `opaque-transformers` live here (not under a
`transformers/` directory name, which would shadow the installed
`transformers` package under pytest’s importlib mode).

Implementation code under test is in `opaque.api.transformers.trainer`;
public imports should use `opaque.transformers` or
`opaque.transformers.trainer`.
