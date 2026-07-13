# Federated

Federated-learning twins of the DP training loop's data primitives live in
`opaque.federated`, executed on [IFED](https://github.com/JetBrains-Research/ifed).
The privacy unit is the **client** (a federated agent holding a private
dataset) instead of the example, and one training round replaces one batch —
everything downstream of the clipped gradient (`gaussian_noise`, optimizers,
the `Accountant`) is the unchanged central chain.

## Overview

| central | federated | notes |
|-----------|-----------|-------|
| `Dataset` | `ifed.Population(uri, datasets=[...])` | a dynamic pool of clients holding named datasets; schema-free and non-indexable |
| `Sampler` | `opaque.federated.MinSepSampler` | yields `ifed.Cohort` *specs*; the platform enforces the participation constraint |
| `batch` | `ifed.Cohort` | symbolic — per-client gradients depend on the params, so a cohort is resolved by executing its round |
| `DataLoader` | `opaque.federated.DataLoader` | bounds the stream to `rounds` cohorts |
| `clipped_grad` | `opaque.federated.clipped_grad` | per-**client** clipping over a cohort; one call = one federated iteration |

```python
import ifed
import opaque.federated as fed
from ifed_client import Client

population = ifed.Population("/hive", datasets=[Iris])
sampler    = fed.MinSepSampler(population, batch_size=8, bands=4)
loader     = fed.DataLoader(population, batch_sampler=sampler, rounds=60)

with Client(server="http://host:15004") as client:
    grad_fn, clip_state = fed.clipped_grad(loss_fn, client, clipping_norm=1.0,
                                           params=params, data=Iris)
    for cohort in loader:
        grads, clip_state = grad_fn(params, cohort, state=clip_state)
        # gaussian_noise → optimizer.update → apply_updates → accountant: unchanged
```

`clipped_grad` compiles the functional loss once (eagerly, at the factory) and
executes each round on ONE long-lived interactive IFED task, so the
minimum-separation policy spans the whole run. The per-client clip runs in
IFED's trusted aggregator; the returned `ClippedPytree` carries
`max_norm = clipping_norm / cohort_size`, exactly like central.

## Sampling honesty

`MinSepSampler` deliberately drops two central `BMinSepSampler` parameters,
because federation cannot honestly claim them today:

- **No `sampling_prob` / `key`** — IFED assigns rounds greedily among live
  eligible agents; there is no randomized-selection guarantee, hence **no
  subsampling amplification**. Account with the *non-amplified* BandMF
  mechanism (`opaque.dpftrl.accounting.mf_gaussian` with a
  `band_mf_strategy`): `min_sep = sampler.bands`,
  `max_participations = ceil(loader.rounds / sampler.bands)`.
- **No `n_steps`** — the round horizon lives on the loader (`rounds=`).

Cohorts are **exact**: IFED collects contributions from exactly `batch_size`
clients per round (blocking until it has them), so the batch size is a
constant, the gradient normalizer is fixed, and the sensitivity
`clipping_norm / batch_size` is data-independent.

Accounting remains user-owned, as everywhere in opaque: the sampler and loader
only expose the participation structure (`bands`, `batch_size`, `rounds`).

## API

::: opaque.federated.clipped_grad

::: opaque.federated.MinSepSampler

::: opaque.federated.DataLoader

::: opaque.federated.make_clipping_aggregate

**See also**: `docs/dev/opaque-federated.md` in the ifed repository for the
platform-side design (interactive runs, assign-separation enforcement,
failure semantics).
