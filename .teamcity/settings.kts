import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.buildFeatures.XmlReport
import jetbrains.buildServer.configs.kotlin.buildFeatures.xmlReport
import jetbrains.buildServer.configs.kotlin.pipelines.*
import jetbrains.buildServer.configs.kotlin.triggers.vcs

version = "2026.1"

private val pipelineId = "Opaque_LinuxAmd64Tests"

private data class TestShard(
    val id: String,
    val label: String,
    val path: String,
)

private fun jobId(shard: TestShard) = "LinuxAmd64_${shard.id}"

private val linuxAmd64TestShards = listOf(
    TestShard("Accounting", "opaque-accounting", "packages/opaque-accounting"),
    TestShard("Alignment", "opaque-alignment", "packages/opaque-alignment"),
    TestShard("Auditing", "opaque-auditing", "packages/opaque-auditing"),
    TestShard("Base", "opaque-base", "packages/opaque-base"),
    TestShard("Dpftrl", "opaque-dpftrl", "packages/opaque-dpftrl"),
    TestShard("Dpsgd", "opaque-dpsgd", "packages/opaque-dpsgd"),
    TestShard("Engine", "opaque-engine", "packages/opaque-engine"),
    TestShard("Kernels", "opaque-kernels", "packages/opaque-kernels"),
    TestShard("Optimizers", "opaque-optimizers", "packages/opaque-optimizers"),
    TestShard("Torch", "opaque-torch", "packages/opaque-torch"),
    TestShard("Transformers", "opaque-transformers", "packages/opaque-transformers"),
    TestShard("Integration", "integration", "tests"),
)

private val setupScript = """
    set -euo pipefail

    "${'$'}OPAQUE_PYTHON" --version
    rustc --version
    uv --version

    uv venv --python "${'$'}OPAQUE_PYTHON"
    uv sync --locked --group dev --all-packages --extra all
"""

private fun testScript(shard: TestShard) = """
    set -euo pipefail

    coverage_report="coverage-linux-amd64-${shard.id.lowercase()}.xml"
    junit_report="junit-linux-amd64-${shard.id.lowercase()}.xml"

    set +e
    timeout --preserve-status 30m \
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

    exit "${'$'}status"
"""

project {
    pipeline {
        id(pipelineId)
        name = "Opaque Linux amd64 tests"

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
                id(jobId(shard))
                name = shard.label

                params {
                    param("env.OPAQUE_PYTHON", "python3.11")
                }

                requirements {
                    equals("teamcity.agent.jbHosted", "true")
                    startsWith("system.agent.name", "Linux-Large")
                }

                features {
                    xmlReport {
                        id = "JUnitResults"
                        reportType = XmlReport.XmlReportType.JUNIT
                        rules = "+:junit-linux-amd64-${shard.id.lowercase()}.xml"
                    }
                }

                steps {
                    script {
                        id = "SetupTestEnvironment"
                        name = "Set up test environment"
                        scriptContent = setupScript
                    }
                    script {
                        id = "RunTests"
                        name = "Run tests"
                        scriptContent = testScript(shard)
                    }
                }

                outputFiles {
                    pipelineArtifacts("coverage-linux-amd64-${shard.id.lowercase()}.xml")
                    pipelineArtifacts("junit-linux-amd64-${shard.id.lowercase()}.xml")
                }
            }
        }
    }
}
