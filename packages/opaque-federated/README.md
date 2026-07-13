# opaque-federated

Federated twins of the central DP training loop's data primitives, executed on
[IFED](https://github.com/JetBrains-Research/ifed):

- `opaque.federated.clipped_grad` — per-**client** clipped gradients over a
  cohort; one factory = one interactive federated task, one call = one
  federated iteration. Returns the same `(grad_fn, FixedClipState)` /
  `ClippedPytree` shapes as `opaque.dpsgd.clipping.clipped_grad`, so the
  noise → optimizer → accountant chain is byte-identical to central.
- `opaque.federated.MinSepSampler` — the federated `BMinSepSampler`: yields
  `ifed.Cohort` specs and compiles minimum separation onto IFED's
  assign-separation policy. No sampling randomness is claimed (selection is
  platform-greedy), so it pairs with **non-amplified** BandMF accounting.
- `opaque.federated.DataLoader` — iterates a population for `rounds` cohorts.

```python
population = ifed.Population("/hive", datasets=[Iris])
sampler    = fed.MinSepSampler(population, batch_size=8, bands=4)
loader     = fed.DataLoader(population, batch_sampler=sampler, rounds=60)

grad_fn, clip_state = fed.clipped_grad(loss_fn, client, clipping_norm=1.0, params=params)
for cohort in loader:
    grads, clip_state = grad_fn(params, cohort, state=clip_state)
    ...  # gaussian_noise / optimizer / accountant — unchanged central opaque
```

See `docs/reference/federated.md` in this repo and `docs/dev/opaque-federated.md`
in the ifed repo for the design.
