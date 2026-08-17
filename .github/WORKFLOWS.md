# GitHub Actions Workflows

This directory contains Opaque's CI/CD configuration. Workflow files define
triggers, permissions, and pipeline policy; shared implementation lives in
private reusable components described below.

## Entry-point workflows

| Workflow | Trigger | Responsibility |
|---|---|---|
| `pr.yml` | Pull requests to `main`, manual dispatch | Required Linux amd64, dependency-boundary, macOS arm64, Linux arm64, and CUDA checks plus preview-wheel artifacts. |
| `ci.yml` | Pushes to `main`, manual dispatch | Linux amd64, dependency-boundary, macOS arm64, Linux arm64, and CUDA validation; development-wheel publication; and draft-release updates. |
| `release.yml` | Published GitHub Release | Tag protection, release tests, artifact validation, package publication, and Release assets. |
| `docs.yml` | Pushes to `main` or `v*` tags, manual dispatch | Builds and deploys versioned documentation. |
| `autoformat.yml` | Pull requests to `main` | Checks and, for trusted PRs, applies Python and Rust formatting fixes. |
| `junie-review.yml` | Pull requests to `main` | Runs Junie as a repository reviewer using the branch's Junie guidance and architecture contracts. |
| `junie.yml` | Trusted `@junie-agent` or `/junie` commands in issues and pull requests | Runs interactive Junie tasks, including code changes and pull-request updates. |
| `build-devcontainer.yaml` | Devcontainer changes and manual dispatch | Builds, smoke-tests, and publishes the development container. |

## Reusable components

### `.github/actions/setup-python`

This composite action installs the pinned Python version and optionally uv. Use
`install-uv: "false"` for Python-only jobs. uv cache restore/save is explicitly
disabled repository-wide: the large all-extras environments made cache
archiving costly and vulnerable to runner disk exhaustion. Release-capable called
workflows keep pinned external setup steps inline so checking out an older
release tag cannot remove the local action implementation.

### `.github/actions/setup-rust`

This composite action installs Rust stable, optional toolchain components, and
the existing Cargo dependency cache. Use `components: clippy, rustfmt` for
formatting jobs and `enable-cache: "true"` for Rust test jobs.

### `.github/workflows/build-distributions.yml`

This private `workflow_call` workflow builds all Python wheels, native wheels,
the umbrella wheel, and the accounting sdist from caller-supplied package
matrices. `pr.yml`, `ci.yml`, and `release.yml` discover the matrices once and
supply them with an artifact prefix, retention period, and (when necessary) an
explicit build version or release tag. This renders checks as
`Build / <distribution>` without an internal discovery child.

Each build job validates its own wheel metadata. Native artifact jobs also
validate accounting policy, and the sdist job proves that the source artifact
can rebuild a wheel. The reusable workflow intentionally does not own
credentials, publication, cross-package validation, or pipeline gates.
Builds start after package discovery in every entry workflow; main/release
publication waits for the complete Python environment matrix and Rust tests.

### `.github/workflows/python-tests.yml`

This private `workflow_call` workflow runs the shared Python test matrix:
Rust/Python/uv setup, dependency synchronization, pytest with coverage, and
Codecov upload. Each caller provides one runner environment plus its Python
version, dependency selection, pytest marker filter, xdist arguments, and shard
matrix. Dependency selection is `locked`, `minimum` (uv `lowest-direct`),
or `latest` (uv `highest`); adding a future platform such as Windows requires only
another caller. Validation callers report every test phase taking at least five
seconds, so newly slow tests cannot disappear behind a fixed-size duration table.
Lower-bound pytest failures are currently advisory while dependency resolution
and workflow failures remain blocking.

Qodana merges coverage from Linux amd64, macOS arm64, Linux arm64, and CUDA.
Dependency-boundary lanes are excluded because they repeat the same source
under alternate dependency graphs rather than add platform behavior. Every
test lane uploads coverage unconditionally; Qodana selects `coverage-locked-*`.
Main publishes the scan's SARIF report to GitHub code scanning under the stable
`qodana/python` category and waits for processing; missing or rejected reports
fail the Qodana job. PR scans keep Qodana annotations, comments, Cloud results,
report artifacts, and manually dispatched branch runs without uploading a
non-main code-scanning analysis.
Repository Code Security must remain enabled.

`pr.yml` and `ci.yml` invoke the reusable workflow separately for Linux amd64
tests, minimum and latest dependency boundaries, and
macOS arm64, Linux arm64, and CUDA platform coverage. Main and release add slow tests to
platform lanes; release reruns the same environment set at the
published tag. Fork pull requests never receive the self-hosted CUDA runner.

### `.github/workflows/rust-tests.yml`

This private `workflow_call` workflow runs the accounting crate's unit and doc
tests with the shared Rust setup and dependency cache. Entry workflows call it
as `Rust`, so checks render as `Rust / opaque-accounting`. Tests above five
seconds use `#[ignore = "slow"]`: PRs run the default set, while main and release
add `cargo test --lib -- --ignored` after the default unit/doc-test run.

### `.github/workflows/validate-distributions.yml`

This private `workflow_call` workflow downloads a complete caller-selected
artifact family, installs `opaque[all]` using only built Opaque wheels, and runs
a representative DP-SGD + DP-FTRL cross-stack accounting scenario without
checking out the source tree. PR, main, and release differ only in artifact
prefix.

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

The active `main` ruleset requires `Build documentation`, `Format Python`,
`Format Rust`, `Conventional Commits PR title`, the selected individual Python
environment/package checks, `Rust / opaque-accounting`, and `Junie review`.
The review workflow uses the `JUNIE_API_KEY` Actions secret.
Fork and Dependabot pull requests cannot receive the secret-backed Junie review;
the job records that limitation and completes without invoking Junie.
Interactive Junie sessions follow `.junie/guidelines.md`; automated reviews use
`.junie/review-guidelines.md`. The review file is also the canonical policy for
Copilot code review through
`.github/instructions/code-review.instructions.md`. Reviewers must apply every
relevant active architecture contract. Privacy-sensitive reviews trace the
guarantee end to end and use read-only web search and URL fetching to verify
primary literature before reporting theorem-dependent findings.

The automated review and interactive workflows use the same SHA-pinned upstream
Junie action and `JUNIE_API_KEY`. Only non-bot authors with an `OWNER`, `MEMBER`, or
`COLLABORATOR` association reach the action, which then verifies that the actor
has repository write or admin access. The job grants `contents`, pull-request,
and issue write access explicitly; repository-wide workflow permissions remain
read-only. Repository Settings > Actions > General must also allow GitHub
Actions to create and approve pull requests.

Interactive Junie tasks use the default `GITHUB_TOKEN`, so comments and commits
are attributed to `github-actions[bot]`. GitHub does not start new `push` or
`pull_request` workflow runs for changes made with that token. After Junie
changes a branch, a maintainer must trigger CI for the new head, for example by
closing and reopening the pull request or by pushing a maintainer-authored
commit. The automated repository review uses the same identity.
