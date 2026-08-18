# TeamCity build-configuration prototype

This directory defines TeamCity 2026.1 Kotlin DSL build configurations for the
GitHub Actions CPU test lanes: locked Linux/amd64, minimum dependencies, latest
dependencies, and Linux/aarch64. Each lane uses TeamCity's native Matrix Build
feature to generate one virtual build per package plus the integration suite.

Set `teamcity.server.url` in `pom.xml` to the TeamCity server that will import
these settings, then enable versioned settings for the Opaque TeamCity project
with this repository as the settings VCS root and `.teamcity` as its settings
path. The same VCS root must track the branches and pull requests that should
run the configurations; each configuration uses `DslContext.settingsRoot` and
builds only `main` and pull requests targeting `main`. The GitHub Checks
webhook trigger publishes each run's result to GitHub without a separate commit
status publisher. The VCS root must use a refreshable GitHub App token with
webhooks enabled; the App needs `Checks: Read and write` plus the `Check run`
and `Check suite` webhook events. Do not add pull-request refs to the root's
branch specification because the Pull Requests build feature discovers them.

The pipelines use JetBrains-hosted Linux-Large agents for amd64 lanes and
Linux-Large-Arm64 agents for the arm64 lane. uv installs each lane's requested
Python version before creating its environment, so agents need only Rust stable
and uv preinstalled. Each job runs the same dependency resolution and pytest
selection as its GitHub Actions counterpart, and publishes JUnit and coverage
XML artifacts. Matrix artifacts are grouped by shard. TeamCity processes the
JUnit reports through its XML Report Processing feature; coverage.py XML remains
an artifact because it is not a TeamCity coverage-report format. Every matrix
leg fails on reported test failures. Linux/amd64 and Linux/aarch64 include slow
tests only on the default branch; dependency-boundary lanes always exclude them.

`test-shards.tsv` is the matrix source. Regenerate it whenever package
`pyproject.toml` files change:

```bash
python .github/scripts/discover_package_matrices.py \
  --teamcity-shards .teamcity/test-shards.tsv
```

To validate the generated TeamCity configuration locally:

```bash
cd .teamcity
mvn teamcity-configs:generate
```
