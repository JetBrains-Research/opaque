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
triggers only on `main` and pull requests targeting `main`. The VCS root must
use GitHub HTTP or GitHub App credentials so TeamCity can query pull requests;
do not add pull-request refs to its branch specification because the Pull
Requests build feature discovers them.

The pipelines use JetBrains-hosted Linux-Large agents for amd64 lanes and
Linux-Large-Arm64 agents for the arm64 lane. uv installs each lane's requested
Python version before creating its environment, so agents need only Rust stable
and uv preinstalled. Each job runs the same dependency resolution and pytest
selection as its GitHub Actions counterpart, and publishes JUnit and coverage
XML artifacts. Matrix artifacts are grouped by shard. TeamCity processes the
JUnit reports through its XML Report Processing feature; coverage.py XML remains
an artifact because it is not a TeamCity coverage-report format. Every matrix
leg fails on reported test failures.

To validate the generated TeamCity configuration locally:

```bash
cd .teamcity
mvn teamcity-configs:generate
```
