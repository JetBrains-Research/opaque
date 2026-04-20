# opaque-mf

Matrix-factorization correlated-noise mechanisms (DP-FTRL family):

- `opaque.noise.mf` — strategies (band, BLT, BSR, BISR, identity, JME, lambda-CGD) + dispatcher
- `opaque.optimizers.adamw_jme` — AdamW with JME dual-stream noise (requires `torchopt`)

Depends on `opaque-core` and `opaque-accounting`. Install with `pip install opaque-mf[optimizers]` for AdamW-JME.
