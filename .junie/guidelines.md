Opaque — internal guidelines (Junie-only)
Updated: 2025-11-10 08:54 local

Scope
- Project = PyTorch port of JAX-Privacy (functional API) for DP-SGD with LoRA focus.
- Phase = Stage 0 complete (infra/docs); Stage 1 ready (no implementation yet). Source stubs only.

Authoritative orientation
- CLAUDE.MD is the agent brief. Treat it as the single source of truth for immediate next steps.
- Reference upstream code at ../jax_privacy (expected to exist locally; used for discovery and optional cross-validation).

Repository facts
- Core code (stubs):
  - src/opaque/core/pytree_utils.py → tree_leaves, tree_map, global_norm (NotImplementedError)
  - src/opaque/core/clipping.py → clip_pytree, clipped_grad (NotImplementedError)
- Tests layout: tests/core/, tests/jax_validation/ (both currently empty of test_*.py). tests/conftest.py defines markers and fixtures and imports torch.
- Docs (Material for MkDocs): docs/development/{tdd-workflow.md, stage1-plan.md, roadmap.md, architecture.md, design-decisions.md, jax-privacy-comparison.md, contributing.md}.
- Tooling: uv (package/env mgr), ruff (format+lint), pytest (+cov), hypothesis. mkdocs configured via mkdocs.yml. PyTorch >=2.0 required.
- PyPI package metadata present; version 0.0.0; build via hatchling.

Pytest configuration (pyproject.toml)
- testpaths = tests; python_files = test_*.py; addopts includes coverage html+term and -v by default.
- markers: jax_validation, slow, gpu.
- norecursedirs excludes typical build/venv dirs. Coverage target package = opaque.

uv dependency groups (pyproject.toml)
- default: torch.
- dev: pytest, pytest-cov, ruff, hypothesis.
- jax-validation: jax, jaxlib, jax-privacy. uv source maps jax-privacy to ../jax_privacy (editable).
- docs: mkdocs + mkdocs-material + mkdocstrings + mkdocs-jupyter.

Commands (canonical)
- Run unit tests (no JAX): uv run pytest
- Run with coverage: uv run pytest --cov=opaque --cov-report=html
- Run JAX validation-only tests: uv run --group jax-validation pytest -m jax_validation
- Lint/format: uv run ruff format src/ tests/; uv run ruff check src/ tests/
- Serve docs: uv run --group docs mkdocs serve

TDD workflow (enforced by docs/CLAUDE.MD + CONTRIBUTING.md)
1) Discover: inspect ../jax_privacy (esp. src/experimental/clipping.py and dp_sgd/noise_injection.py).
2) Optional JAX reference test under tests/jax_validation/ with @pytest.mark.jax_validation (requires --group jax-validation).
3) Write failing Opaque test under tests/core/ defining target API/behavior.
4) Implement minimal code to pass tests.
5) Document with Google-style docstrings including runnable examples.
6) Validate numerically vs JAX-Privacy (tolerances: atol=1e-5, rtol=1e-5).

Stage 1 (active scope)
- Implement pytree utilities: global_norm, tree_leaves, tree_map (wrapping torch.utils._pytree via thin wrapper layer).
- Implement clipping primitives: clip_pytree (edge cases: clip_norm=0 → zeros; clip_norm=inf → passthrough; tree_norm=0 → passthrough; nan_safe; optional rescale_to_unit_norm), and clipped_grad (per-example grads via torch.func.grad + torch.func.vmap; support normalize_by, keep_batch_dim, microbatch_size; optional return per-example norms).
- Write unit/property tests; add optional JAX validation tests; create a simple linear regression example later.

Key technical mappings (JAX→PyTorch)
- jax.vmap → torch.func.vmap
- jax.grad → torch.func.grad
- jax.tree_util.tree_map → torch.utils._pytree.tree_map (private API → wrap)
- PyTree = nested dict[str, Tensor]

Design decisions (summary)
- PyTree impl: depend on torch.utils._pytree via wrapper (risk: private API churn; mitigation: wrapper indirection + tests for dict structures).
- Microbatching: explicit microbatch_size argument; no auto heuristics initially.
- Numerical tolerance for cross-framework checks: atol=1e-5/rtol=1e-5; complement with property-based tests.
- Fail-fast error handling; explicit parameter validation.

Risk/edge-case checklist for implementation
- Private API dependency (torch.utils._pytree): keep wrapper small; avoid leaking private types in public signatures.
- Device/dtype handling: ensure functions preserve dtype and device; test CPU/GPU via fixtures (device, all_devices); skip CUDA when unavailable.
- NaN/Inf handling in clip_pytree when nan_safe=True (use torch.nan_to_num).
- Zero/inf norms; avoid divide-by-zero; support rescale_to_unit_norm.
- vmap in_dims for loss_fn signatures; keep_batch_dim semantics; ensure argnums and batch_argnums are flexible (int or tuple[int,...]).
- Microbatch path vs full-batch path must be numerically close; test both; avoid excessive Python loops if possible.
- normalize_by semantics (e.g., divide by batch size for averaging); document sensitivity when rescale_to_unit_norm=True.

Known documentation drift (2025-11-10)
- README.md still references IMPLEMENTATION_PLAN.md and OPEN_QUESTIONS.md which were deleted; authoritative versions moved under docs/development/.
- README contains minimal code example using clipped_grad that will not run until Stage 1 implements stubs.
- CLAUDE.MD’s “Essential Context Files” are present under docs/development/ (validated) and should be treated as current.

Test status (checked before writing this file)
- Command: uv run pytest -q (2025-11-10) → 0 tests collected; exit 0. Coverage plugin warns “Module opaque was never imported / No data collected” (expected with no tests and stub-only code). This confirms the test command works; scaffolding is in place.

Minimal next steps for a new task (Stage 1)
- Add tests first:
  - tests/core/test_pytree_utils.py → tree_leaves/tree_map/global_norm (depth, empty tree, dtype/device cases).
  - tests/core/test_clipping.py → clip_pytree edge cases; clipped_grad simple scalar param + small batch; property tests (Hypothesis) for norm bounds.
  - Optional tests/jax_validation/test_jax_clipping.py comparing to jax_privacy.experimental.clipping (guard with importorskip and @pytest.mark.jax_validation).
- Implement wrappers in src/opaque/core/pytree_utils.py; implement clip_pytree; then a basic clipped_grad (no microbatch) → extend to microbatching.
- Keep docstrings with minimal, deterministic examples; add to docs via mkdocstrings later.

Conventions
- Line length 100; type hints on public APIs; Google-style docstrings.
- Avoid premature optimization; prefer clear, functional style.
- Keep public API surface small in Stage 1; no high-level API yet.

Paths quick map
- Code: src/opaque/core/{pytree_utils.py, clipping.py}
- Tests: tests/{core/, jax_validation/} + tests/conftest.py
- Docs: docs/development/{tdd-workflow.md, stage1-plan.md, roadmap.md, architecture.md, design-decisions.md, jax-privacy-comparison.md, contributing.md}
- Config: pyproject.toml, mkdocs.yml

Notes to self (Junie)
- Always run tests through uv to ensure the managed environment is used (torch is required even for conftest import).
- For JAX validation, ensure ../jax_privacy is cloned and enable the jax-validation group.
- If adding examples, gate them behind Stage 1 completion; do not ship runnable examples that import unimplemented APIs.
