import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.buildFeatures.XmlReport
import jetbrains.buildServer.configs.kotlin.buildFeatures.xmlReport
import jetbrains.buildServer.configs.kotlin.triggers.vcs

version = "2026.1"

private data class TestShard(
    val id: String,
    val label: String,
    val path: String,
)

private data class TestLane(
    val id: String,
    val name: String,
    val environmentName: String,
    val dependencyInstall: String,
    val python: String,
    val architecture: String,
    val hostedRunnerName: String,
    val timeoutMinutes: Int,
)

private val testShards = listOf(
    TestShard("Accounting", "opaque-accounting", "packages/opaque-accounting"),
    TestShard("Alignment", "opaque-alignment", "packages/opaque-alignment"),
    TestShard("Auditing", "opaque-auditing", "packages/opaque-auditing"),
    TestShard("Base", "opaque-base", "packages/opaque-base"),
    TestShard("Dpftrl", "opaque-dpftrl", "packages/opaque-dpftrl"),
    TestShard("Dpsgd", "opaque-dpsgd", "packages/opaque-dpsgd"),
    TestShard("Engine", "opaque-engine", "packages/opaque-engine"),
    TestShard("Optimizers", "opaque-optimizers", "packages/opaque-optimizers"),
    TestShard("Patches", "opaque-patches", "packages/opaque-patches"),
    TestShard("Transformers", "opaque-transformers", "packages/opaque-transformers"),
    TestShard("Integration", "integration", "tests"),
)

private val testLanes = listOf(
    TestLane(
        id = "LinuxAmd64",
        name = "Linux amd64",
        environmentName = "linux-amd64",
        dependencyInstall = "uv sync --locked --group dev --all-packages --extra all",
        python = "python3.11",
        architecture = "amd64",
        hostedRunnerName = "Linux-Large",
        timeoutMinutes = 30,
    ),
    TestLane(
        id = "MinimumDependencies",
        name = "minimum dependencies",
        environmentName = "minimum-dependencies",
        dependencyInstall = """
            uv sync --upgrade --resolution lowest-direct \
                --group dev --all-packages --extra all
        """.trimIndent(),
        python = "python3.11",
        architecture = "amd64",
        hostedRunnerName = "Linux-Large",
        timeoutMinutes = 30,
    ),
    TestLane(
        id = "LatestDependencies",
        name = "latest dependencies",
        environmentName = "latest-dependencies",
        dependencyInstall = """
            uv sync --upgrade --resolution highest \
                --group dev --all-packages --extra all
        """.trimIndent(),
        python = "python3.12",
        architecture = "amd64",
        hostedRunnerName = "Linux-Large",
        timeoutMinutes = 30,
    ),
    TestLane(
        id = "LinuxAarch64",
        name = "Linux arm64",
        environmentName = "linux-aarch64",
        dependencyInstall = "uv sync --locked --group dev --all-packages --extra all",
        python = "python3.11",
        architecture = "aarch64",
        hostedRunnerName = "Linux-Large-Arm64",
        timeoutMinutes = 30,
    ),
)

private fun setupScript(lane: TestLane) = """
    set -euo pipefail

    "${'$'}OPAQUE_PYTHON" --version
    rustc --version
    uv --version

    uv venv --python "${'$'}OPAQUE_PYTHON"
    ${lane.dependencyInstall}
"""

private fun reportPath(kind: String, lane: TestLane, shard: String) =
    "$kind-${lane.environmentName}-$shard.xml"

private fun testScript(lane: TestLane) = """
    set -euo pipefail

    test_path="${'$'}OPAQUE_TEST_PATH"
    shard_name="${'$'}{test_path##*/}"
    if [[ "${'$'}shard_name" == "tests" ]]; then
        shard_name="integration"
    else
        shard_name="${'$'}{shard_name#opaque-}"
    fi

    coverage_report="${reportPath("coverage", lane, "${'$'}shard_name")}"
    junit_report="${reportPath("junit", lane, "${'$'}shard_name")}"

    set +e
    timeout --preserve-status ${lane.timeoutMinutes}m \
        uv run --no-sync pytest "${'$'}test_path" \
        -m "not cuda and not mps and not slow" \
        -n auto --dist loadscope \
        --cov=opaque \
        --cov-report=xml:"${'$'}coverage_report" \
        --junitxml="${'$'}junit_report" \
        --durations=0 --durations-min=5 \
        -q
    status=${'$'}?
    set -e

    exit "${'$'}status"
"""

project {
    testLanes.forEach { lane ->
        buildType {
            id("Opaque${lane.id}TestMatrix")
            name = "Opaque ${lane.name} test matrix"
            artifactRules = """
                ${reportPath("coverage", lane, "*")}
                ${reportPath("junit", lane, "*")}
            """.trimIndent()

            vcs {
                root(DslContext.settingsRoot)
            }

            triggers {
                vcs {
                    branchFilter = "+:*"
                }
            }

            params {
                param("env.OPAQUE_PYTHON", lane.python)
            }

            requirements {
                equals("teamcity.agent.jbHosted", "true")
                startsWith("system.agent.name", lane.hostedRunnerName)
                equals("teamcity.agent.jvm.os.arch", lane.architecture)
            }

            steps {
                script {
                    name = "Set up test environment"
                    scriptContent = setupScript(lane)
                }
                script {
                    name = "Run tests"
                    scriptContent = testScript(lane)
                }
            }

            features {
                matrix {
                    param(
                        "env.OPAQUE_TEST_PATH",
                        testShards.map { shard -> value(shard.path, label = shard.label) },
                    )
                    groupArtifactsByBuild = true
                }
                xmlReport {
                    reportType = XmlReport.XmlReportType.JUNIT
                    rules = "+:${reportPath("junit", lane, "*")}"
                }
            }

            failureConditions {
                executionTimeoutMin = lane.timeoutMinutes
                testFailure = true
            }
        }
    }
}
