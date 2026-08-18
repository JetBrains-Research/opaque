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

    if [[ -f "${'$'}junit_report" ]]; then
        echo "##teamcity[importData type='junit' path='${'$'}junit_report']"
    fi

    exit "${'$'}status"
"""

project {
    pipeline {
        id("OpaqueLinuxAmd64Tests")
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
                id("LinuxAmd64${shard.id}")
                name = shard.label

                params {
                    param("env.OPAQUE_PYTHON", "python3.11")
                }

                requirements {
                    equals("teamcity.agent.jvm.os.family", "Linux")
                    equals("teamcity.agent.jvm.os.arch", "amd64")
                }

                steps {
                    script {
                        name = "Set up test environment"
                        scriptContent = setupScript
                    }
                    script {
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
