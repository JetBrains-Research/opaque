# Ruff deferred-rule workstreams

The Format Python gate (`autoformat.yml`) uses an **explicit**
`[tool.ruff.lint].select` in the root `pyproject.toml`. Rules that are cheap
to adopt are already on. What remains falls into two buckets:

1. **Permanent policy** — enabling the rule fights intentional design
   (unicode math, thematic `__all__`, fixed PyTorch / Triton signatures,
   PEP 420 façades). Keep these ignored or scoped via
   `per-file-ignores`; do not open cleanup PRs to "satisfy" them.
2. **Independent workstreams** — the rule is desirable, but adopting it is a
   product / API / docs project, not a drive-by lint pass. Track each as its
   own PR series with an acceptance criterion and a gate step that flips
   `select` / `ignore` only when that package (or façade surface) is clean.

This page plans bucket (2). Counts below are approximate `packages/`
snapshots under Ruff 0.16; re-measure with
`uv run ruff check packages/ --select <CODE> --statistics` before starting.

## How to run a workstream

1. Open a dedicated branch / PR series named after the workstream
   (e.g. `typing(engine): annotate public façades`).
2. Prefer **package- or façade-scoped** enablement over a repo-wide flip:
   start with `per-file-ignores` shrinking, or enable the rule only after a
   package is clean.
3. Land behavioral / API changes **before** turning the rule on in CI so a
   half-migrated tree cannot merge.
4. Acceptance: `uv run ruff check packages/ --select <CODES>` is clean for
   the scoped paths, PR-lane tests pass, and the `pyproject.toml` comment
   for that rule is updated or removed.
5. Do not mix unrelated lint families in one PR (e.g. ANN + TRY003).

Permanent ignores that are **not** workstreams (document only):

| Rule | Why it stays off |
| --- | --- |
| `E501` | Formatter owns line length |
| `RUF001` / `RUF002` / `RUF003` | Intentional unicode in math strings / docs / comments |
| `RUF022` | Alphabetical `__all__` destroys thematic section comments |
| `ARG004` (+ kernel / optimizer / test ARG per-file) | Fixed `autograd.Function` / vmap / torchopt signatures |
| `INP001` on `**/tests/**`, `tests/**`, `packages/*/src/opaque/*.py` | Pytest trees and PEP 420 façades must not grow `__init__.py` |

---

## Workstream A — Public type annotations (`ANN`)

**Goal.** Every public façade and user-callable factory has complete
argument and return annotations so editors, `mkdocstrings`, and
contributors share one source of truth.

**Scale.** ~9.6k `ANN*` findings repo-wide. Public-façade return types alone
(`ANN201` under `packages/*/src/opaque/`) are already far smaller (~tens),
so scope by surface, not by "fix all ANN".

**Suggested phasing.**

| Phase | Scope | Notes |
| --- | --- | --- |
| A0 | Decide policy | Ignore `ANN401` (`Any`) initially or only on internal helpers; allow untyped `*args`/`**kwargs` shims that forward to typed factories |
| A1 | Façade `__init__.py` + `opaque.<concern>` re-exports | Annotate what users import; keep impl trees for later |
| A2 | Factory callables (`clipped_grad`, `gaussian_noise`, samplers, optimizers, schedules) | Match CONTRIBUTING: public API additions already require annotations — extend that to the existing surface |
| A3 | `opaque.api.*` internals package-by-package | Start with `opaque-base` / `opaque-accounting` (torch-free), then engine, then stacks |
| A4 | Enable `ANN` in `select` (or a subset: `ANN001`, `ANN201`, `ANN202`) with `per-file-ignores` for tests / kernels if needed | Flip the gate only when A1–A2 are clean |

**Risks.** Pytree-heavy APIs need `TypeVar` / `Protocol` / structural typing;
over-annotating with `Any` is worse than leaving the rule off. Prefer
accurate aliases in `opaque.types` over copy-pasted unions.

**Out of scope for A.** Rewriting runtime behavior; adding runtime
`typeguard` checks.

---

## Workstream B — Google-style docstrings (`D`)

**Goal.** Public modules, classes, and functions have Google-style
docstrings consistent with CONTRIBUTING and `mkdocstrings`
(`docstring_style: google` in `mkdocs.yml`).

**Scale.** ~3.0k `D*` findings; ~2.2k of those are missing public docs
(`D100`–`D103`). ~500 style nits are autofixable once content exists.

**Suggested phasing.**

| Phase | Scope | Notes |
| --- | --- | --- |
| B0 | Lock convention | Keep Google style; configure Ruff `lint.pydocstyle.convention = "google"`; resolve D203/D212 vs D211/D213 when selecting `D` |
| B1 | Autofixable style only (`D413`, `D209`, …) on already-documented symbols | Mechanical PR; no new prose |
| B2 | Public façades and factories | One package per PR; include Args / Returns / Raises where calibration or privacy semantics matter |
| B3 | Mechanism reference alignment | Docstrings must not invent APIs — cross-check `docs/reference/` and `docs/mechanisms/` |
| B4 | Enable `D` (or `D1`, `D2`, `D4` subsets) for `src/` with tests ignored | Prefer documenting public surfaces over private `_*.py` |

**Risks.** Docstring churn that drifts from real signatures re-opens the
docs-audit class of bugs. Treat reference docs and docstrings as one review
unit for privacy-facing APIs.

**Out of scope for B.** Tutorials and user-guide prose (separate docs PRs).

---

## Workstream C — Exception taxonomy (`TRY003`)

**Goal.** Replace long `raise ValueError("…")` / `TypeError("…")` literals
with a small, documented exception hierarchy (or message constants) so
callers can catch by type and tests can use stable `match=` / exception
classes.

**Scale.** ~580 `TRY003` hits; densest in `opaque-transformers`,
`opaque-dpftrl`, `opaque-engine`, `opaque-optimizers`.

**Suggested phasing.**

| Phase | Scope | Notes |
| --- | --- | --- |
| C0 | Design | Decide package-local vs `opaque.errors` shared types; keep torch-free accounting free of engine imports |
| C1 | Accounting / calibration / budgets | Highest user impact for wrong ε messaging; align with fail-closed calibration behavior |
| C2 | Engine clipping / noise / distributed | Shared by DP-SGD and DP-FTRL |
| C3 | Stack packages + transformers trainer | Large surface; migrate raise sites and update `pytest.raises` |
| C4 | Enable `TRY` / `TRY003` once public raises go through the taxonomy | Allow listed escape hatches for ImportError install hints if needed |

**Risks.** A giant shared hierarchy becomes indirection without benefit.
Prefer a few semantic bases (`ConfigurationError`, `PrivacyBudgetError`,
`CheckpointError`) over one class per string. Message text stays part of
the API for `match=` tests — changing it is user-visible.

**Out of scope for C.** Catching bare `Exception`; restructuring control
flow solely to appease TRY\*.

---

## Workstream D — Intentional lazy imports (`PLC0415`)

**Goal.** Inventory every non-top-level import; keep the ones that break
cycles, defer optional extras, or avoid torch in accounting; lift the rest;
document the keepers with `# noqa: PLC0415` (or a short comment + per-file
ignore for known lazy modules).

**Scale.** ~539 hits (transformers and patches dominate).

**Suggested phasing.**

| Phase | Scope | Notes |
| --- | --- | --- |
| D0 | Classify | Tag each site: cycle / optional extra / torch-boundary / accidental |
| D1 | Lift accidentals | Pure style; safe mechanical PRs |
| D2 | Codify keepers | Prefer module-level lazy helpers (`def _import_foo(): …`) over scattered inline imports |
| D3 | Enable `PLC0415` with narrow `per-file-ignores` or noqa on keepers | Do **not** force top-level imports that pull torch into accounting or that re-introduce import cycles |

**Risks.** Blind hoisting breaks `opaque-accounting`'s torch-free contract
and PEP 420 façade lazy-loading. Any change that imports torch from
accounting must fail `tests/contracts/test_accounting_torch_free.py`.

---

## Workstream E — API shape and complexity (`PLR0913` / `PLR0917` / `PLR0915`)

**Goal.** Wide factories either gain a config object (matching Opaque's
"factory returns callable" pattern) or get an intentional raised limit;
oversized functions get extracted helpers without changing privacy
semantics.

**Scale.** ~199 too-many-args, ~142 too-many-positional, ~26 too-many-statements.

**Suggested phasing.**

| Phase | Scope | Notes |
| --- | --- | --- |
| E0 | Policy | For public factories, prefer keyword-only params or a frozen config dataclass over silent `max-args` bumps; document any raised thresholds in `pyproject.toml` |
| E1 | New APIs only | Enforce limits on greenfield code via review + optional pre-commit |
| E2 | Refactor the worst callables | Trainer / MF noise / collators — one behavior-preserving PR each with tests |
| E3 | Enable `PLR0913` / `PLR0917` / `PLR0915` with configured maxima that match the settled public APIs | Avoid enabling at defaults that ban the documented trainer surface |

**Risks.** Config-object migrations are breaking unless façades keep thin
wrappers. Treat as `feat!` / deprecation when signatures change.

---

## Workstream F — Named constants for magic comparisons (`PLR2004`)

**Goal.** Replace unexplained numeric literals in non-test comparisons with
named constants where the number encodes domain meaning (privacy δ floors,
block sizes, schedule breakpoints).

**Scale.** ~649 raw; ~63 outside `**/tests/**` if tests are ignored. Most
remaining hits are real candidates; many test hits should stay numeric.

**Suggested phasing.**

| Phase | Scope | Notes |
| --- | --- | --- |
| F0 | Configure | Ignore tests via `per-file-ignores`; consider allowing obvious literals (`-1`, `0`, `1`) via Ruff pylint settings if available for the locked version |
| F1 | Domain constants | Centralize shared numerics next to the concern (`opaque.api.*.types` or module-level `Final`) |
| F2 | Enable `PLR2004` for `src/` | Keep tests ignored |

**Risks.** Naming every `2` in a shape check adds noise. Focus on privacy-
and kernel-relevant constants first.

---

## Workstream G — PLD method caches (`B019`) — small, accounting-owned

**Goal.** Keep `@lru_cache` / `@cache` on PLD mechanism methods without a
blanket ignore, by moving caches to free functions / static helpers keyed on
immutable inputs, or by documenting `# noqa: B019` on frozen mechanism
types.

**Scale.** 15 sites, all in accounting / amplification / mechanisms.

**Suggested approach.** Single focused accounting PR: verify instances are
immutable / hashable, prefer `@cache` on pure functions taking the
mechanism's value tuple, then remove `B019` from `ignore` or replace the
global ignore with file-scoped noqa.

**Risks.** Incorrect cache keys silently wrong ε — treat as privacy-critical
review; add cache-clear tests where process state can change.

---

## Recommended order

Independent streams can run in parallel once owners are assigned. If
sequencing is required:

1. **G (B019)** — smallest, privacy-adjacent, clears a global ignore.
2. **D (PLC0415)** — inventory prevents accidental torch / cycle breakage in later typing work.
3. **A1–A2 (ANN façades)** and **B2 (docstrings on the same façades)** — pair per package so each public symbol lands typed and documented together.
4. **C (exceptions)** — before enabling more TRY\* rules; update tests as raises migrate.
5. **E / F** — after public APIs stabilize; otherwise config objects and constants churn twice.

## Measuring progress

```bash
# Re-count a workstream
uv run ruff check packages/ --select ANN --statistics
uv run ruff check packages/ --select D100,D101,D102,D103 --statistics
uv run ruff check packages/ --select TRY003 --statistics
uv run ruff check packages/ --select PLC0415 --statistics
uv run ruff check packages/ --select PLR0913,PLR0915,PLR0917,PLR2004 --statistics
uv run ruff check packages/ --select B019 --config 'lint.ignore=[]' --statistics
```

When a workstream finishes, update `[tool.ruff.lint]` in `pyproject.toml`
(select / ignore / per-file-ignores) in the **same** PR that makes the
scoped tree clean, and delete the corresponding "omitted" comment line.
