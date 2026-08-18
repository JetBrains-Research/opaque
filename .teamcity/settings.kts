import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.buildFeatures.XmlReport
import jetbrains.buildServer.configs.kotlin.buildFeatures.xmlReport
import jetbrains.buildServer.configs.kotlin.buildFeatures.PullRequests
import jetbrains.buildServer.configs.kotlin.buildFeatures.pullRequests
import jetbrains.buildServer.configs.kotlin.triggers.gitHubChecks
import java.io.File

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
    val pythonVersion: String,
    val architecture: String,
    val hostedRunnerName: String,
    val timeoutMinutes: Int,
)

private fun testShards() =
    File(DslContext.baseDir, "test-shards.tsv")
        .readLines()
        .filter { it.isNotBlank() }
        .mapIndexed { index, line ->
            val fields = line.split("\t")
            require(fields.size == 3) {
                "test-shards.tsv line ${index + 1} must contain id, label, and path"
            }
            TestShard(fields[0], fields[1], fields[2])
        }

private val testShards = testShards()

private val testLanes = listOf(
    TestLane(
        id = "LinuxAmd64",
        name = "Linux/amd64",
        environmentName = "linux-amd64",
        dependencyInstall = "uv sync --locked --group dev --all-packages --extra all",
        pythonVersion = "3.11",
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
        pythonVersion = "3.11",
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
        pythonVersion = "3.12",
        architecture = "amd64",
        hostedRunnerName = "Linux-Large",
        timeoutMinutes = 30,
    ),
    TestLane(
        id = "LinuxAarch64",
        name = "Linux/aarch64",
        environmentName = "linux-aarch64",
        dependencyInstall = "uv sync --locked --group dev --all-packages --extra all",
        pythonVersion = "3.11",
        architecture = "aarch64",
        hostedRunnerName = "Linux-Large-Arm64",
        timeoutMinutes = 30,
    ),
)

private fun setupScript(lane: TestLane) = """
    set -euo pipefail

    rustc --version
    uv --version

    uv python install ${lane.pythonVersion}
    uv venv --python ${lane.pythonVersion}
    ${lane.dependencyInstall}
"""

private fun reportPath(kind: String, lane: TestLane, shard: String) =
    "$kind-${lane.environmentName}-$shard.xml"

private fun testScript(lane: TestLane) = """
    set -euo pipefail

    test_path="${'$'}TEST_PATH"
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
            name = "Python ${lane.name} tests"
            artifactRules = """
                ${reportPath("coverage", lane, "*")}
                ${reportPath("junit", lane, "*")}
            """.trimIndent()

            vcs {
                root(DslContext.settingsRoot)
                branchFilter = """
                    +:main
                    +:refs/pull/*
                """.trimIndent()
            }

            triggers {
                gitHubChecks {}
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
                        "env.TEST_PATH",
                        testShards.map { shard -> value(shard.path, label = shard.label) },
                    )
                    groupArtifactsByBuild = true
                }
                xmlReport {
                    reportType = XmlReport.XmlReportType.JUNIT
                    rules = "+:${reportPath("junit", lane, "*")}"
                }
                pullRequests {
                    vcsRootExtId = "${DslContext.settingsRoot.id}"
                    provider = github {
                        authType = vcsRoot()
                        filterAuthorRole = PullRequests.GitHubRoleFilter.EVERYBODY
                        filterTargetBranch = "+:refs/heads/main"
                    }
                }
            }

            failureConditions {
                executionTimeoutMin = lane.timeoutMinutes
                testFailure = true
            }
        }
    }
}
