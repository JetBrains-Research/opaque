# Issue 685: cached-accountant continuation for horizon processes

Research date: 2026-08-24
Repository baseline: [`3a64ba3`](https://github.com/JetBrains-Research/opaque/commit/3a64ba36cad75c180daacfae1f1e4126018a4420) (`origin/main`, v0.15.3)

## Executive conclusion

[Issue 685](https://github.com/JetBrains-Research/opaque/issues/685) identifies a
real accounting-algebra defect. A cached prefix of a correlated horizon process
cannot be continued by composing the prefix PLD with the marginal one-step PLD.
The continuation must return to the horizon process and request the joint
`pld_at(K + 1)` bound.

The issue was partly reframed by a change merged shortly after it was opened.
[PR 679](https://github.com/JetBrains-Research/opaque/pull/679) made `cached()`
warn and return any process tree containing a `DpHorizonProcess` unchanged. That
prevents new bad boundaries in the ordinary trainer path, but it is a broad
safety stopgap:

- it skips caching unrelated portions of a heterogeneous prefix;
- it emits `RuntimeWarning` from normal DP-FTRL logging, evaluation, and resume
  checks;
- it does not make the cache algebra itself safe for continuation;
- a previously serialized `CachedProcess` horizon prefix remains opaque after
  restore.

The narrow architectural fix is to make `CachedProcess.__or__` transparent only
when its inner process ends in the same `PerStep` horizon sequence as the process
being appended. A homogeneous cached horizon prefix becomes one longer
`Repeated` node. For a heterogeneous tree, only its live horizon suffix is
reopened; the preceding process remains behind a cache barrier. All unrelated
cached composition keeps its current behavior.

No horizon-process redesign, execution-plan abstraction, trainer-specific
accounting rule, or privacy-bound change is needed.

## Issue record and scope

The issue was opened on 2026-08-20 and remains open as of this research. It has
no comments, labels, assignee, milestone, linked pull request, or dependency
metadata. Its timeline contains two facts that matter:

1. It is a child of [issue 683](https://github.com/JetBrains-Research/opaque/issues/683),
   whose post-review decision explicitly preserves the existing
   `DpHorizonProcess`/`PerStep` design.
2. It was renamed from “Classify and validate per-step prefix tightness for
   every horizon process” to the narrower cache-continuation defect. The broader
   prefix review moved to [issue 688](https://github.com/JetBrains-Research/opaque/issues/688).

The issue's required behavior is:

- `cached(per_step(P)) * K` must continue to materialize `P.pld_at(K)`;
- after K accumulated steps, caching the accountant and adding a matching step
  must materialize `P.pld_at(K + 1)`;
- arbitrary log, evaluation, save, and resume boundaries must not change that
  result;
- cached heterogeneous DP-SGD composition must remain an opaque optimization
  boundary.

## The process algebra

`DpHorizonProcess` represents a mechanism whose releases are defined together
over a declared horizon. Its `pld_at(K)` method is the privacy bound for the
deployed N-step mechanism stopped after K releases. `PerStep(P)` adapts that
process to training-loop composition:

```text
Repeated(PerStep(P), K).pld()
    -> PerStep(P).repeated_pld(K)
    -> P.pld_at(K)
```

The generic `DpProcess.__or__` optimizer merges structurally equal adjacent
leaves. Consequently, repeatedly appending one `PerStep` object produces a
single `Repeated` node rather than a deep heterogeneous tree.

`CachedProcess` has two roles:

1. a larger PLD memoization boundary; and
2. an opaque merge barrier, so later composition reuses the materialized prefix
   instead of looking through it.

Those roles are valid for ordinary sequential composition. They conflict with
continuing one correlated horizon sequence.

## Correct and defective paths

### Cached step: already correct

[PR 290](https://github.com/JetBrains-Research/opaque/pull/290) added a
`CachedProcess.repeated_pld` relay. It preserves the strategy-aware override on
`PerStep`:

```text
Repeated(CachedProcess(PerStep(P)), K)
    -> CachedProcess.repeated_pld(K)
    -> PerStep.repeated_pld(K)
    -> P.pld_at(K)
```

This path is used by `DPTrainer`, which constructs one cached step process and
reuses it for every optimizer step. It must not be removed or replaced.

### Cached accountant: defective without special continuation

After K steps, wrapping the whole accountant creates:

```text
CachedProcess(
    Repeated(CachedProcess(PerStep(P)), K)
)
```

Ordinary composition cannot see the repeated leaf through the outer cache:

```text
Composed(
    CachedProcess(Repeated(CachedProcess(PerStep(P)), K)),
    CachedProcess(PerStep(P)),
)
```

Materialization then computes:

```text
P.pld_at(K).compose(P.pld_at(1))
```

That is not the contract of the next correlated release. `pld_at(1)` is a
marginal prefix bound, not a conditional guarantee that can be substituted for
release K + 1 after observing the first K releases.

### Concrete BandMF reproduction

On baseline `3a64ba3`, the defect can still be reproduced by constructing the
opaque boundary directly (the public `cached()` guard otherwise skips it):

- `BandMfStrategy(bands=2)`, Poisson sample rate 0.1, noise multiplier 1.0;
- deployed horizon N = 6;
- cache after K = 3 and append one step;
- PLD discretization 0.01 and delta `1e-5`.

Observed results:

```text
process.pld_at(4):                         epsilon = 1.9175427240376783
process.pld_at(3).compose(process.pld_at(1)): epsilon = 2.0871288243103870
difference:                                          0.1695861002727086
```

The numerical result happens to be more pessimistic in this example. That does
not validate the decomposition: the required conditional-release argument is
absent, and no general ordering follows merely from this one parameter choice.
The structural path, rather than approximate epsilon equality, is therefore the
primary regression oracle.

## Facts that reframe the solution

### 1. Main already has a conservative partial fix

The final commits bundled into [PR 679](https://github.com/JetBrains-Research/opaque/pull/679)
were authored while issue 685 was being triaged. They added a recursive
`_contains_horizon_process()` check. Public `cached(process)` now warns and
returns the original object whenever any horizon process appears below it,
except that a bare `PerStep` adapter may still be cached.

This means the common current-main reproduction is safe:

```text
cached(accountant) is accountant
accountant | step -> Repeated(step, K + 1)
```

The issue is therefore not accurately described as an unfenced current-main
privacy failure. The remaining work is to replace an overbroad fence with the
precise algebraic rule, add the missing acceptance coverage, and cover restored
cache boundaries.

### 2. Horizon processes are cross-cutting, not synonymous with DP-FTRL

[PR 459](https://github.com/JetBrains-Research/opaque/pull/459) deliberately
moved `DpHorizonProcess` and `PerStep` into the torch-free accounting core.
DP-SGD random allocation, and now k-out-of-t, also use the abstraction. A fix
gated on `ctx.mf`, a BandMF class, or a DP-FTRL trainer mode would miss valid
callers and put composition semantics in the wrong package.

### 3. The cache is a merge barrier, not just a memoization decorator

Every process already has resolved-input PLD caching. `CachedProcess` also
prevents the composition optimizer from merging through a materialized prefix.
Globally unwrapping it, changing `_leaf_and_count()` to expose its inner leaf,
or making all caches transparent would change ordinary DP-SGD tree shape and
performance. The exception must be scoped to horizon continuation under `|`.

### 4. `repeated_pld` and continuation are different operations

The existing relay solves:

```text
(cached(step) * K).pld()
```

It cannot solve:

```text
cached(step * K) | step
```

The first is materialization of one `Repeated` node. The second is construction
of a new composition tree, so the correction belongs at cache-boundary
composition rather than in `pld()` or `repeated_pld()`.

### 5. Checkpoint save does not itself call `cached()`

`DPTrainer` caches at resume epsilon checks, evaluation, and logging. Checkpoint
and standalone accountant save serialize the live tree as-is. They can therefore
persist a boundary created immediately beforehand, but the save operation is
not itself the source of the boundary.

[PR 720](https://github.com/JetBrains-Research/opaque/pull/720) strengthens the
importance of this distinction: `save_accountant()` can now persist the live
mid-training context from a callback independently of model saving.

### 6. Serialization preserves the problematic node

`Accountant` serialization stores the complete polymorphic process tree,
including `CachedProcess`, `Composed`, `Repeated`, and `PerStep`. Restoring a
checkpoint does not normalize composition. A solution that only avoids new
trainer cache calls cannot repair an already serialized single cached horizon
prefix; continuation-aware `CachedProcess.__or__` can.

### 7. Horizon extension is a separate, already guarded lifecycle concern

[Issue 687](https://github.com/JetBrains-Research/opaque/issues/687) was closed
after confirming that DP-FTRL checkpoint metadata rejects total-step/horizon
drift and same-horizon resume restores noise and sampler state. Issue 685 should
not introduce a new execution plan or make a horizon extensible. It only joins
the next release within the already-declared horizon.

### 8. Prefix tightness is separate work

Some horizon implementations are exact, some round to atoms or epochs, and
some use conservative full-horizon or Monte Carlo bounds. Those classifications
are addressed by issue 688. Cache continuation must call the process's existing
`pld_at(K + 1)` implementation without reinterpreting its guarantee.

### 9. The current PLD cache design makes reopening affordable

PR 679 also replaced process-owning global LRU keys with weak identity-owned and
process-free canonical caches. Dropping the outer whole-prefix cache when a
horizon continues does not discard the horizon process's own bounded prefix
cache. The K-prefix query remains reusable, while K + 1 is computed through the
correct horizon API.

## Alternatives reviewed adversarially

### Keep the current recursive skip

Correct for new ordinary calls, but overbroad. A tree such as
`ordinary_prefix | horizon_suffix` loses caching for the ordinary prefix, normal
trainer lifecycle emits warnings, and restored opaque boundaries remain opaque.

### Skip caching only inside `DPTrainer`

Narrower than the current guard, but composition correctness would depend on
one caller. Manual loops, callbacks, other trainers, DP-SGD allocation modes,
and restored accountants would still need duplicate rules.

### Make all `CachedProcess` nodes expose `_leaf_and_count()`

Too broad. `_leaf_and_count()` is also used by multiplication. Exposing a cached
horizon prefix there could reinterpret independent repetition of a K-step
prefix as one K×M-step correlated horizon. It would also remove ordinary merge
barriers.

### Look through all cache barriers in `DpProcess.__or__`

Breaks the explicit generic barrier contract and the DP-SGD optimization
acceptance criterion.

### Add a new execution-plan or horizon-accountant type

The parent review explicitly rejected this as speculative. Existing process
identity, `PerStep`, `Repeated`, checkpoint validation, and runtime state are
sufficient.

### Chosen: a cache-local continuation rule

Override `CachedProcess.__or__`:

1. inspect the appended process only if it is a `PerStep` horizon group (one
   step or `Repeated` steps, optionally using the existing cached step leaf);
2. inspect the cached inner process's active group;
3. if both groups refer to structurally equal `PerStep` processes, add their
   counts and return one `Repeated` node;
4. if the cached inner process is heterogeneous and its right group matches,
   keep the left process cached and replace only the right group;
5. walk older cached right-spine fragments iteratively and coalesce them while
   they remain the same contiguous horizon sequence;
6. otherwise delegate unchanged to `DpProcess.__or__`.

Then restore `cached()` to its ordinary, idempotent behavior. The special rule
is attached to the barrier that needs the exception, not to each caller.

## Intended tree rewrites

Homogeneous horizon continuation:

```text
CachedProcess(Repeated(H, K)) | H
    -> Repeated(H, K + 1)
```

Heterogeneous prefix with a live horizon suffix:

```text
CachedProcess(Composed(X, Repeated(H, K))) | H
    -> Composed(CachedProcess(X), Repeated(H, K + 1))
```

Here `H` is a `PerStep` leaf, possibly wrapped by the existing per-step
`CachedProcess`.

Unrelated composition remains unchanged:

```text
CachedProcess(X) | Y
    -> Composed(CachedProcess(X), Y)
```

The same is true when `X` and `Y` are equal ordinary DP-SGD leaves: the cache
continues to block merging.

## Invariants and edge cases

- Match horizon processes by structural equality, consistent with the existing
  composition optimizer and checkpoint reconstruction.
- Retain the existing suffix leaf when merging. On resume this keeps the
  restored process, whose strategy and horizon correspond to the saved run.
- Let `PerStep.repeated_pld()` enforce `count <= n_steps`; do not duplicate or
  move horizon validation into construction.
- Do not treat a direct `DpHorizonProcess` as a step group. Only its `PerStep`
  adapter denotes incremental releases.
- Do not look through a different intervening process. Only a matching active
  suffix is one continuing horizon sequence.
- Keep `cached(per_step(P)) * K` unchanged.
- Keep ordinary cached homogeneous and heterogeneous composition unchanged.
- Preserve `Accountant` budget object identity when rebuilding its cached
  process.
- Keep the accounting wheel torch-free and add no cross-package imports.

## Validation plan

### Accounting algebra

- Correlated BandMF for every `1 <= K < N`: cache after K, append one, and
  assert materialization is the same `P.pld_at(K + 1)` result.
- Multiple cache boundaries at different prefix lengths.
- A heterogeneous ordinary prefix remains cached while its trailing BandMF
  sequence continues.
- An unrelated cached DP-SGD prefix remains opaque and does not merge.
- Existing `cached(per_step(P)) * K` tests remain green.

### Persistence

- Serialize and restore an accountant while its horizon prefix is cached, append
  a structurally equal step, and recover one `Repeated(K + 1)` horizon prefix.
- Exercise the existing DP-FTRL checkpoint/resume smoke tests, which restore the
  accountant together with noise, sampler, and optimizer state.

### Repository checks

- Focused opaque-accounting, opaque-dpsgd, and opaque-dpftrl tests.
- Focused DP-FTRL trainer smoke/checkpoint tests.
- Ruff check and format check on changed Python files.
- Full non-accelerator, non-slow PR-equivalent suite if time and environment
  permit.

## Implemented result

The implementation follows the cache-local design above:

- [`_cached.py`](packages/opaque-accounting/src/opaque/api/accounting/core/composition/_cached.py)
  gives `CachedProcess` a narrow composition override. Its scan is iterative,
  so the number of historical cache boundaries is heap-bounded rather than
  Python-recursion-bounded.
- [`test_per_step.py`](packages/opaque-dpftrl/tests/accounting/test_per_step.py)
  covers every K below a small correlated BandMF horizon, repeated boundaries,
  serialization/restore, multiple historical fragments, heterogeneous stable
  prefixes, different horizons, intervening releases, and the unchanged cached
  step path.
- [`test_accountant.py`](packages/opaque-dpsgd/tests/accounting/test_accountant.py)
  covers cross-stack random-allocation continuation and proves an ordinary
  DP-SGD cache remains an opaque merge barrier.
- [`test_dpftrl_trainer_smoke.py`](packages/opaque-transformers/tests/opaque_transformers/test_dpftrl_trainer_smoke.py)
  exercises BandMF logging, evaluation, checkpoint save, and resume with
  deliberately overlapping boundary cadences.
- [`accounting.md`](docs/reference/accounting.md) documents the stable public
  exception without implementation history.

Validation completed on 2026-08-24:

- full CPU PR lane: **4,186 passed, 174 skipped, 481 deselected, 1 expected
  xfail**;
- accounting-focused non-slow suites: **581 passed, 1 skipped**;
- DP-FTRL trainer smoke suite: **27 passed, 3 slow tests deselected**;
- changed-path and repository-wide Ruff check/format: passed;
- strict MkDocs build: passed;
- `git diff --check`: passed.

The environment does not contain `cargo` or `rustc`, so the Rust workspace
tests could not run. No Rust source or binding surface changed; the locally
built native accounting extension was exercised throughout the Python suites.

## Source map

- [Issue 685](https://github.com/JetBrains-Research/opaque/issues/685): defect and
  acceptance criteria.
- [Parent issue 683](https://github.com/JetBrains-Research/opaque/issues/683):
  reviewed scope and non-goals.
- [Issue 687](https://github.com/JetBrains-Research/opaque/issues/687): horizon
  lifecycle/resume guard already confirmed.
- [Issue 688](https://github.com/JetBrains-Research/opaque/issues/688): separate
  prefix-tightness and early-stop work.
- [PR 290](https://github.com/JetBrains-Research/opaque/pull/290): correct
  `cached(per_step(P)) * K` relay.
- [PR 459](https://github.com/JetBrains-Research/opaque/pull/459): generic
  `DpHorizonProcess` and `PerStep` architecture.
- [PR 679](https://github.com/JetBrains-Research/opaque/pull/679): weak PLD cache
  ownership plus the broad horizon-cache skip merged after issue creation.
- [PR 720](https://github.com/JetBrains-Research/opaque/pull/720): standalone live
  accountant persistence.
- [`CachedProcess` on the researched baseline](https://github.com/JetBrains-Research/opaque/blob/3a64ba36cad75c180daacfae1f1e4126018a4420/packages/opaque-accounting/src/opaque/api/accounting/core/composition/_cached.py).
- [`DpProcess` composition optimizer on the researched baseline](https://github.com/JetBrains-Research/opaque/blob/3a64ba36cad75c180daacfae1f1e4126018a4420/packages/opaque-accounting/src/opaque/api/accounting/core/_base.py#L421-L454).
- [`PerStep` on the researched baseline](https://github.com/JetBrains-Research/opaque/blob/3a64ba36cad75c180daacfae1f1e4126018a4420/packages/opaque-accounting/src/opaque/api/accounting/core/composition/_per_step.py).
- [`DPTrainer` cache call sites on the researched baseline](https://github.com/JetBrains-Research/opaque/blob/3a64ba36cad75c180daacfae1f1e4126018a4420/packages/opaque-transformers/src/opaque/api/transformers/trainer/_dp_trainer.py).
