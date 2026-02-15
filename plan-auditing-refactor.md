# Privacy Auditing Refactoring Plan

Based on review of `feat/privacy-auditing` branch. To be applied after merging the noise API PR.

## Issues Found

1. **Built against old API** — `__init__.py` imports `gaussian, gaussian_stateful` (old names)
2. **`AuditResult` is a `namedtuple`** — should be frozen dataclass (our convention)
3. **`Scores` type alias** — `Sequence[float] | np.ndarray` is not useful, remove
4. **`threshold.py` is dead code** — strategies defined but never wired into epsilon functions
5. **`auditor.py` is a god module** — 350+ lines mixing epsilon, metrics, audit, bootstrap
6. **`helpers.py` exports everything** — internal helpers should be `_`-prefixed
7. **`bootstrap()` uses `ThreadPoolExecutor`** — misleading for CPU-bound numpy work
8. **Top-level re-exports 11 auditing symbols** — excessive, trim to essentials
9. **No internal/public `__all__` split** in submodules

## Refactoring Steps

1. Rebase auditing onto our branch (correct noise API base)
2. Convert `AuditResult` from namedtuple to `@dataclass(frozen=True)`
3. Split `auditor.py` → `epsilon.py`, `metrics.py`, `audit.py` (keep `bootstrap.py`)
4. Delete `threshold.py` + `test_threshold.py` (dead code, YAGNI)
5. Clean up `helpers.py` — `_`-prefix internal functions (`_pareto_frontier`, `_log_sub`)
6. Remove `Scores` type alias — use `np.ndarray` directly
7. Fix `bootstrap()` — replace `ThreadPoolExecutor` with sequential loop
8. Trim `opaque/__init__.py` — only re-export `auditing`, `audit`, `AuditResult`, `BootstrapParams`
9. Update `auditing/__init__.py` for new file structure
10. Update tests for new imports, delete `test_threshold.py`
11. Update docs (`api/auditing.md`, `user-guide/auditing.md`, tutorial notebook)
12. Run tests, commit, push
