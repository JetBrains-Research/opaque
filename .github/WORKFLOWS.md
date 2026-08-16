# GitHub Actions Workflows

This directory contains Opaque's CI/CD configuration. Workflow files define
triggers, permissions, and pipeline policy; shared implementation lives in
private reusable components described below.

## Entry-point workflows

| Workflow | Trigger | Responsibility |
|---|---|---|
| `pr.yml` | Pull requests to `main`, manual dispatch | Required Linux/amd64, dependency-boundary, MPS, and CUDA checks plus preview-wheel artifacts. |
| `ci.yml` | Pushes to `main`, manual dispatch | Linux/amd64, dependency-boundary, MPS, and CUDA validation; development-wheel publication; and draft-release updates. |
| `release.yml` | Published GitHub Release | Tag protection, release tests, artifact validation, package publication, and Release assets. |
| `docs.yml` | Pushes to `main` or `v*` tags, manual dispatch | Builds and deploys versioned documentation. |
| `autoformat.yml` | Pull requests to `main` | Checks and, for trusted PRs, applies Python and Rust formatting fixes. |
| `junie-review.yml` | Pull requests to `main` | Runs Junie as a repository reviewer using the branch's Junie guidance and architecture contracts. |
| `junie.yml` | Trusted `@junie-agent` or `/junie` commands in issues and pull requests | Runs interactive Junie tasks, including code changes and pull-request updates. |
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
Codecov upload. Each caller provides one runner environment plus its Python
version, dependency selection, pytest marker filter, xdist arguments, and shard
matrix. Dependency selection is `locked`, `lower-bounds` (uv `lowest-direct`),
or `latest` (uv `highest`); adding a future platform such as Windows requires only
another caller. Validation callers report every test phase taking at least five
seconds, so newly slow tests cannot disappear behind a fixed-size duration table.

`pr.yml` and `ci.yml` invoke the reusable workflow separately for canonical
Linux/amd64 tests, lower-bound and latest dependency boundaries, and MPS,
Linux/aarch64, and CUDA platform coverage. Main and release add slow tests to
canonical and platform lanes; release reruns the same environment set at the
published tag. Fork pull requests never receive the self-hosted CUDA runner.
All three entry points keep a direct `Rust tests` job so their check names stay
stable and uncluttered.

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

The active `main` ruleset requires `Build documentation`, `Format Python`,
`Format Rust`, `Conventional Commits PR title`, `Python tests`, `Rust tests`,
and `Junie review`. The review workflow uses the `JUNIE_API_KEY` Actions secret.
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
