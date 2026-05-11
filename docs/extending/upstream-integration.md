# Upstream integration

A lot of what an extension wheel does is some flavour of "integrate
with an upstream library." That can mean re-using it directly,
wrapping/extending it, or replacing it with an Opaque-internal
implementation. The right call depends on what the upstream library
does, how it does it, and what Opaque needs that it doesn't already
have.

This page describes today's three in-tree precedents — one for each
of *reuse*, *extend*, and *rewrite* — and the rubric they suggest.
It's a framework for making the decision, not a contract; the
trade-offs that pushed each precedent are recorded so a future
contributor can see why a similar decision came out one way.

## Three in-tree precedents

### Reuse: torchopt

torchopt is a functional PyTorch optimiser library — `optim.adam`
returns a `(init_fn, update_fn)` pair operating on pytrees, the
same shape Opaque needs for DP-SGD. Opaque doesn't ship its own
adam / adagrad / sgd implementations; the tutorials demonstrate
how to drop torchopt's optimisers directly into the DP pipeline.

Why reuse worked:

- torchopt's API is functional and pytree-shaped, which matches
  what `NoisedPytree` flows through.
- It composes naturally with `torch.func.vmap` — the rest of
  Opaque's vmap-safety story doesn't have to absorb it.
- The optimiser logic itself isn't privacy-sensitive: from the
  privacy accountant's point of view, what flows in is already
  noised, and what flows out is post-processing of a privatised
  quantity.
- The library's licence (Apache-2.0) is compatible with Opaque's.

The cost of reuse is a soft dependency: tutorials need torchopt
installed, but the core wheels don't. Users opt into the
combination at install time.

### Extend: opaque-optimizers

Some optimisers Opaque wants to ship — adaptive-second-moment
optimisers driven by Opaque's paired-stream second-moment release,
schedule-free training with DP-specific tweaks — don't exist in
torchopt or anywhere upstream. `opaque-optimizers` defines them as
a new wheel that *follows the torchopt pattern* (`init_fn` /
`update_fn`, pytree-typed) but adds the DP-specific machinery.

Why extend rather than reuse:

- The pattern is right, but the specific optimiser isn't upstream
  to reuse.
- Following the upstream pattern means a user who's used torchopt
  before doesn't have to learn a new optimiser interface.

Why extend rather than rewrite:

- The pytree / functional-update shape is doing real work — it
  composes with `vmap` and with `NoisedPytree` the same way every
  other Opaque component does. Replacing it with a stateful
  optimiser would lose that.
- Future cross-pollination is cheaper: a new torchopt-style
  optimiser can be picked up directly.

### Rewrite: opaque-patches

`torch.nn.BatchNorm2d` and similar layers carry per-batch running
statistics; under `torch.func.vmap` the statistics get the wrong
batch dimension and the layer either errors or quietly produces a
non-DP result. `opaque-patches` ships replacement implementations
of the offending layers — vmap-safe BatchNorm, vmap-safe RNNs, etc.

Why rewrite rather than wrap:

- The upstream layers don't have a "vmap-safe mode" to enable.
  The fix isn't a flag or a hook; it's a re-implementation of the
  forward pass.
- Wrapping the upstream layer (e.g. with a custom autograd
  function) doesn't help — `vmap`'s problem is with the
  statistics, not the autograd, and you'd be papering over a
  silent privacy bug rather than fixing it.
- Once the replacement is the right shape, it slots into a stock
  `nn.Module` tree via standard surgery (`patch_model` in the
  same wheel). User code doesn't see the replacement.

Why a rewrite was the right call here, and rarely is in general:

- The set of broken layers is small and bounded.
- The replacements are mathematically equivalent under DP-SGD's
  per-sample regime — they're not new mechanisms, just
  vmap-correct re-implementations.
- The silent failure mode (wrong batch dim, non-DP output) is
  privacy-critical. A fragile wrapper would be worse than no
  integration.

## A rubric for the decision

Most upstream-integration decisions can be made by working through
a handful of questions in order. They're listed in the order they
typically discriminate.

1. **Does the upstream library expose per-example state under
   `torch.func.vmap`?** If yes, reuse is on the table. If no,
   either the library has to be wrapped/extended to expose that
   state, or its functionality has to be rewritten Opaque-side. A
   library that hides per-example state under `vmap` *can't* be
   used directly in DP-SGD without a fix on one side of the seam.
2. **Is the failure mode of "use it wrong" a silent privacy
   violation, or a loud error?** If silent (BatchNorm under vmap,
   shared RNG state across ranks), the threshold for trusting the
   upstream library is higher. Rewriting the surface so the
   privacy-critical path is in code Opaque owns and tests is
   often the right call.
3. **Is the upstream API stable, or actively churning?** A
   churning upstream API is a maintenance liability for an
   extension that depends on it. Reuse becomes more expensive
   over time; extending often degrades into rewriting as upstream
   diverges.
4. **What does the upstream licence allow?** Opaque is Apache-2.0;
   a wheel that wants to ship in the same repo can't take a
   strong-copyleft upstream. Reusing as a soft dependency
   sidesteps this, but vendoring or extending in-tree doesn't.
5. **What's the test surface like — both yours and theirs?** A
   well-tested upstream library is cheaper to reuse than one
   without. An extension wheel's own tests need to cover *the
   seam*: the place where Opaque's types meet the upstream API.
   If the seam is wide or the upstream tests don't cover the
   shapes you depend on, expect ongoing surprise.

## Worked illustration: a Lipschitz layer family

Suppose someone wanted to add Lipschitz-bounded layers (AOL, SLL,
sandwich layers, the family from the constrained-Lipschitz
literature) as an extension wheel, using them to derive a global
per-record sensitivity from architecture rather than from a
per-example norm computation. There's a mature upstream library
(`deel-torchlip`) that implements many of the layer families.
What's the right call?

Walking the rubric:

1. **vmap exposure.** `deel-torchlip` layers are plain
   `nn.Module`s; per-example state under `vmap` works the same
   way it does for `nn.Linear`. Reuse is on the table.
2. **Failure-mode silence.** The Lipschitz constant is the
   privacy bound; if the upstream layer's claimed Lipschitz
   constant is wrong, the privacy account is wrong silently.
   That suggests Opaque should *own* the sensitivity bookkeeping
   even if it doesn't own the layer implementations.
3. **API stability.** `deel-torchlip` is reasonably stable but
   not pinned-API; ongoing maintenance is required.
4. **Licence.** `deel-torchlip` is MIT — compatible.
5. **Test surface.** The upstream tests cover the Lipschitz
   property of each layer family. They don't cover the *seam*:
   what Opaque needs is "this `nn.Module` tree has a known global
   Lipschitz constant, computed at attachment time and asserted
   per step." That's an extension-side test.

The combination suggests: **reuse the layers, write the
sensitivity bookkeeping yourself.** A hypothetical wheel would
depend on `deel-torchlip` for the layer implementations and own
the seam — a `lipschitz_constant(model)` analyser that walks the
module tree, an attachment helper that asserts the constant
hasn't moved, and an entry point that emits
`ClippedPytree(max_norm=R)` for the rest of the pipeline (see
[Composition](composition.md)).

This is presented as an illustration of how the rubric is applied,
not as a commitment to ship such a wheel. The reasoning would need
to be redone if `deel-torchlip` changed shape, if a more
DP-specialised Lipschitz library appeared, or if the layer-Lipschitz
literature consolidated on a different parameterisation.

## See also

- [Adding a new mechanism family](new-mechanism.md) — the recipe
  for the wheel itself, once the upstream-integration question is
  settled.
- [Composition](composition.md) — what such an extension emits at
  the seam.
