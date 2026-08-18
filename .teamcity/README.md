# TeamCity build-configuration prototype

This directory defines TeamCity 2026.1 Kotlin DSL build configurations for the
GitHub Actions CPU test lanes: locked Linux amd64, minimum dependencies, latest
dependencies, and Linux arm64. Each lane uses TeamCity's native Matrix Build
feature to generate one virtual build per package plus the integration suite.

Set `teamcity.server.url` in `pom.xml` to the TeamCity server that will import
these settings, then enable versioned settings for the Opaque TeamCity project
with this repository as the settings VCS root and `.teamcity` as its settings
path. The same VCS root must track the branches and pull requests that should
run the configurations; each configuration uses `DslContext.settingsRoot` and
triggers on every tracked branch.

The pipelines use JetBrains-hosted Linux-Large agents for amd64 lanes and
Linux-Large-Arm64 agents for the arm64 lane. Each job creates its lane's Python
environment, runs the same dependency resolution and pytest selection as its
GitHub Actions counterpart, and publishes JUnit and coverage XML artifacts.
Matrix artifacts are grouped by shard. TeamCity processes the JUnit reports
through its XML Report Processing feature; coverage.py XML remains an artifact
because it is not a TeamCity coverage-report format. The minimum dependency
configuration does not fail on reported test failures, matching the advisory
GitHub Actions lane.

To validate the generated TeamCity configuration locally:

```bash
cd .teamcity
mvn teamcity-configs:generate
```
