# GitHub Actions Workflows

This directory contains Opaque's CI/CD configuration. Workflow files define
triggers, permissions, and pipeline policy; shared implementation lives in
private reusable components described below.

## Entry-point workflows

| Workflow | Trigger | Responsibility |
|---|---|---|
| `pr.yml` | Pull requests to `main`, manual dispatch | Required PR checks, preview-wheel artifacts, and fork-safe GPU testing. |
| `ci.yml` | Pushes to `main`, manual dispatch | Full CPU/MPS/CUDA test coverage, development-wheel publication, and draft-release updates. |
| `release.yml` | Published GitHub Release | Tag protection, release tests, artifact validation, package publication, and Release assets. |
| `docs.yml` | Pushes to `main` or `v*` tags, manual dispatch | Builds and deploys versioned documentation. |
| `autoformat.yml` | Pull requests to `main` | Checks and, for trusted PRs, applies Python and Rust formatting fixes. |
| `build-devcontainer.yaml` | Devcontainer changes and manual dispatch | Builds, smoke-tests, and publishes the development container. |

## Reusable components

### `.github/actions/setup-python`

This composite action installs the pinned Python version and optionally uv and
its dependency cache. Use it in repository workflows that need Python; set
`install-uv: "false"` for Python-only jobs and `enable-uv-cache: "true"` for
dependency-heavy jobs. Release-capable reusable workflows keep their pinned
external setup steps inline so checking out an older release tag cannot remove
the local action implementation.

### `.github/actions/setup-rust`

This composite action installs Rust stable, optional toolchain components, and
the existing Cargo dependency cache. Use `components: clippy, rustfmt` for
formatting jobs and `enable-cache: "true"` for Rust test jobs.

### `.github/workflows/reusable-build-distributions.yml`

This private `workflow_call` workflow discovers package matrices and builds all
Python wheels, native wheels, the umbrella wheel, and the accounting sdist.
`pr.yml`, `ci.yml`, and `release.yml` supply an artifact prefix, retention
period, and (when necessary) an explicit build version or release tag.

The reusable workflow intentionally does not own credentials, environments,
publication, validation, or pipeline gates. Its callers keep those
responsibilities so trusted release behavior and PR protections remain
explicit at the entry point.

### `.github/workflows/reusable-python-tests.yml`

This private `workflow_call` workflow runs the shared Python test matrix:
Rust/Python/uv setup, dependency synchronization, pytest with coverage, and
Codecov upload. Callers provide their shard and device matrices, timeout, and
optional CUDA assertion or duration reporting.

`pr.yml` keeps CPU/MPS and fork-guarded GPU calls separate so untrusted fork
code never receives the self-hosted runner. `ci.yml` calls the same workflow
with its combined CPU/MPS/GPU matrix and slow-test markers.

### `.github/workflows/reusable-rust-tests.yml`

This private `workflow_call` workflow runs the accounting crate's Cargo tests,
including doc-tests, with the shared Rust setup and dependency cache. PR,
main CI, and release workflows all invoke it; the release workflow runs this
lane alongside the shared Python test workflow instead of running both
sequentially in a dedicated job.

### `.github/workflows/reusable-validate-distributions.yml`

This private `workflow_call` workflow downloads a caller-selected artifact
family and validates both synchronized internal wheel pins and the accounting
wheel/sdist policy. Preview and release pipelines differ only in artifact
prefix and checkout ref; the validation implementation is shared.

## Artifact contracts

The reusable distribution workflow uploads artifacts named
`<prefix>-<distribution>`. Callers preserve the existing prefixes:

| Caller | Prefix | Retention |
|---|---|---|
| PR previews | `preview-wheels` | 14 days |
| Main development builds | `wheels` | 30 days |
| Releases | `wheels` | 90 days |

Downstream validation and merge jobs consume these prefixes, so update the
caller and consumer together if a new artifact family is introduced.

## Security and maintenance

Actions remain pinned to immutable commit SHAs. Entry-point workflows default
to read-only permissions and elevate permissions only for trusted deployment
or publishing jobs. Fork pull requests do not run untrusted code on the
self-hosted GPU runner and never receive repository, package, or cloud
credentials.

The active `main` ruleset requires `Cross-package import smoke test`, `Build
documentation`, `Format Python`, `Format Rust`, `Conventional Commits PR
title`, `Python tests`, and `Rust tests`.
