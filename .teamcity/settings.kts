import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.pipelines.*
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
    val timeoutMinutes: Int,
    val allowTestFailure: Boolean = false,
)

private val linuxAmd64TestShards = listOf(
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
        timeoutMinutes = 60,
        allowTestFailure = true,
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
        timeoutMinutes = 60,
    ),
    TestLane(
        id = "LinuxAarch64",
        name = "Linux arm64",
        environmentName = "linux-aarch64",
        dependencyInstall = "uv sync --locked --group dev --all-packages --extra all",
        python = "python3.11",
        architecture = "aarch64",
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

private fun reportPath(kind: String, lane: TestLane, shard: TestShard) =
    "$kind-${lane.environmentName}-${shard.id.lowercase()}.xml"

private fun testResultScript(lane: TestLane) =
    if (lane.allowTestFailure) {
        """
            if [[ "${'$'}status" -ne 0 ]]; then
                echo "##teamcity[message text='pytest exited with status ${'$'}status in advisory ${lane.environmentName} lane' status='WARNING']"
            fi

            exit 0
        """.trimIndent()
    } else {
        "exit \"${'$'}status\""
    }

private fun testScript(lane: TestLane, shard: TestShard) = """
    set -euo pipefail

    coverage_report="${reportPath("coverage", lane, shard)}"
    junit_report="${reportPath("junit", lane, shard)}"

    set +e
    timeout --preserve-status ${lane.timeoutMinutes}m \
        uv run --no-sync pytest ${shard.path} \
        -m "not cuda and not mps and not slow" \
        -n auto --dist loadscope \
        --cov=opaque \
        --cov-report=xml:"${'$'}coverage_report" \
        --junitxml="${'$'}junit_report" \
        --durations=0 --durations-min=5 \
        -q
    status=${'$'}?
    set -e

    if [[ -f "${'$'}junit_report" ]]; then
        echo "##teamcity[importData type='junit' path='${'$'}junit_report']"
    fi

    ${testResultScript(lane)}
"""

project {
    testLanes.forEach { lane ->
        pipeline {
            id("Opaque${lane.id}Tests")
            name = "Opaque ${lane.name} tests"

            repositories {
                repository(DslContext.settingsRoot)
            }

            triggers {
                vcs {
                    branchFilter = "+:*"
                }
            }

            linuxAmd64TestShards.forEach { shard ->
                job {
                    id("${lane.id}${shard.id}")
                    name = shard.label

                    params {
                        param("env.OPAQUE_PYTHON", lane.python)
                    }

                    requirements {
                        equals("teamcity.agent.jvm.os.family", "Linux")
                        equals("teamcity.agent.jvm.os.arch", lane.architecture)
                    }

                    steps {
                        script {
                            name = "Set up test environment"
                            scriptContent = setupScript(lane)
                        }
                        script {
                            name = "Run tests"
                            scriptContent = testScript(lane, shard)
                        }
                    }

                    outputFiles {
                        pipelineArtifacts(reportPath("coverage", lane, shard))
                        pipelineArtifacts(reportPath("junit", lane, shard))
                    }
                }
            }
        }
    }
}
