"""K1 sampler-level reproduction of the DDP + checkpoint-resume key bug.

Mirrors exactly what the trainer does (no DDP process group needed):
  fresh construction : sampler_key = fold_in(key(seed), rank); PoissonSampler(shard, ..., key=sampler_key)
                       (_dp_trainer.py:3788, 3803-3816)
  save               : opaque.serialization.state_dict(ctx.current_sampler) on world-rank 0 only
                       (_dp_trainer.py:5076-5079, 4962/5004 gated by _distributed.should_save)
  resume             : ctx.current_sampler = from_state_dict(template_sampler, saved_sampler_state)
                       (_dp_trainer.py:1799-1800) -- no rank re-fold.
"""
import numpy as np
import torch
from torch.utils.data import Subset

from opaque.random import key, fold_in
from opaque.distributed import local_shard
from opaque.serialization import state_dict, from_state_dict
from opaque.dpsgd.sampling import PoissonSampler
from opaque.dpftrl.sampling import BMinSepSampler

N = 2000
WORLD = 2
SEED = 1234
Q = 0.05
N_STEPS = 60
STEPS_BEFORE_CKPT = 7
STEPS_AFTER = 20


def make_shard(rank):
    ds = list(range(N))
    trimmed = (N // WORLD) * WORLD
    if trimmed < N:
        ds = Subset(ds, range(trimmed))
    return local_shard(ds, rank=rank, world_size=WORLD)


def make_poisson(rank):
    k = fold_in(key(SEED), rank)  # trainer: fold_in(sampler_key, self._ddp.rank)
    return PoissonSampler(make_shard(rank), sample_rate=Q, n_steps=N_STEPS, key=k)


def make_bmin(rank):
    k = fold_in(key(SEED), rank)
    bands = 4
    p = Q / (1 - Q * (bands - 1))
    return BMinSepSampler(make_shard(rank), bands=bands, sampling_prob=p, n_steps=N_STEPS, key=k)


def masks(sampler, n):
    it = iter(sampler)
    out = []
    for _ in range(n):
        idx = next(it)
        m = np.zeros(len(sampler.data_source), dtype=bool)
        m[np.asarray(idx, dtype=int)] = True
        out.append(m)
    return np.stack(out)


def run(name, factory):
    print(f"\n=== {name} ===")
    r0 = factory(0)
    r1 = factory(1)
    # sanity: before any checkpoint, ranks draw different masks (rank fold works)
    m0_pre = masks(r0, STEPS_BEFORE_CKPT)
    m1_pre = masks(r1, STEPS_BEFORE_CKPT)
    print("pre-ckpt: rank0 vs rank1 identical steps:",
          int(np.all(m0_pre == m1_pre, axis=1).sum()), "/", STEPS_BEFORE_CKPT,
          " mean|B|:", m0_pre.sum(1).mean(), m1_pre.sum(1).mean())
    print("saved snapshot keys/seed from rank0:", {k: v for k, v in state_dict(r0).items() if 'key' in k or k == 'consumed'})
    print("rank1 live stream key seed:", r1._stream_key.seed, " rank0:", r0._stream_key.seed)
    sd0 = state_dict(r0)  # written once, by rank 0
    # resume on every rank: template built by fresh construction (rank-folded), then overwritten by from_state_dict(sd0)
    rest0 = from_state_dict(factory(0), sd0)
    rest1 = from_state_dict(factory(1), sd0)
    print("restored rank1 stream key seed:", rest1._stream_key.seed, "== rank0's:", rest1._stream_key.seed == rest0._stream_key.seed)
    m0 = masks(rest0, STEPS_AFTER)
    m1 = masks(rest1, STEPS_AFTER)
    same_steps = int(np.all(m0 == m1, axis=1).sum())
    print(f"post-resume: identical inclusion masks on {same_steps}/{STEPS_AFTER} steps; "
          f"total identical entries {(m0 == m1).mean():.6f}")
    # continuous run of rank 0 (no ckpt) should equal restored rank 0 (resume is exact)
    cont0 = masks(factory(0), STEPS_BEFORE_CKPT + STEPS_AFTER)[STEPS_BEFORE_CKPT:]
    print("restored rank0 == continuous rank0 on all steps:", bool(np.array_equal(cont0, m0)))
    # what a CORRECT resume would do: restore cursor but keep the rank's own key -> compare to continuous rank 1
    cont1 = masks(factory(1), STEPS_BEFORE_CKPT + STEPS_AFTER)[STEPS_BEFORE_CKPT:]
    print("continuous rank1 vs restored rank1 identical steps:", int(np.all(cont1 == m1, axis=1).sum()), "/", STEPS_AFTER)
    # per-record co-inclusion: P(local index i in both ranks' batches) should be q^2 if independent, q if shared
    both = (m0 & m1).mean()
    print(f"co-inclusion rate of same local index across ranks after resume: {both:.4f}  (q={Q}, q^2={Q*Q:.4f})")
    return same_steps


s_p = run("PoissonSampler (sampling_mode='poisson')", make_poisson)
s_b = run("BMinSepSampler (sampling_mode='b_min_sep')", make_bmin)
print("\nRESULT poisson identical steps:", s_p, "/", STEPS_AFTER, " bmin identical steps:", s_b, "/", STEPS_AFTER)
