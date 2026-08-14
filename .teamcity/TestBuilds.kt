import jetbrains.buildServer.configs.kotlin.BuildType
import jetbrains.buildServer.configs.kotlin.BuildTypeSettings
import jetbrains.buildServer.configs.kotlin.DslContext
import jetbrains.buildServer.configs.kotlin.FailureAction
import jetbrains.buildServer.configs.kotlin.ReuseBuilds
import jetbrains.buildServer.configs.kotlin.Template
import jetbrains.buildServer.configs.kotlin.buildFeatures.PullRequests
import jetbrains.buildServer.configs.kotlin.buildFeatures.commitStatusPublisher
import jetbrains.buildServer.configs.kotlin.buildFeatures.perfmon
import jetbrains.buildServer.configs.kotlin.buildFeatures.pullRequests
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.triggers.vcs

private enum class TestDevice(
    val displayName: String,
    val markerPr: String,
    val markerMain: String,
    val osName: String,
    val xdist: String,
) {
    CPU("CPU", "not cuda and not mps and not slow", "not cuda and not mps", "Linux", "-n auto --dist loadscope"),
    MPS("MPS", "not cuda and not slow", "not cuda", "Mac OS X", "-n auto --dist loadscope"),
    CUDA("CUDA", "cuda and not slow", "cuda", "Linux", "-rs"),
}

private data class TestShard(val id: String, val path: String)

private val testShards = listOf(
    TestShard("Accounting", "packages/opaque-accounting"),
    TestShard("Alignment", "packages/opaque-alignment"),
    TestShard("Auditing", "packages/opaque-auditing"),
    TestShard("Base", "packages/opaque-base"),
    TestShard("DpFtrl", "packages/opaque-dpftrl"),
    TestShard("DpSgd", "packages/opaque-dpsgd"),
    TestShard("Engine", "packages/opaque-engine"),
    TestShard("Optimizers", "packages/opaque-optimizers"),
    TestShard("Patches", "packages/opaque-patches"),
    TestShard("Transformers", "packages/opaque-transformers"),
    TestShard("Repo", "tests"),
)

object PythonTestTemplate : Template({
    id("Opaque_PythonTestTemplate")
    name = "Python test template"

    artifactRules = "coverage.xml => coverage"
    failureConditions {
        executionTimeoutMin = 40
    }
    requirements {
        exists("teamcity.agent.jvm.os.name")
        exists("teamcity.agent.jvm.os.arch")
    }
    features {
        perfmon { }
    }
})

private fun pythonTest(device: TestDevice, shard: TestShard?): BuildType = BuildType {
    val suffix = shard?.id ?: "All"
    id("Opaque_Python${device.name.lowercase().replaceFirstChar(Char::uppercase)}$suffix")
    name = "Python ${device.displayName}: ${shard?.id ?: "all packages"}"
    templates(PythonTestTemplate)

    vcs {
        root(DslContext.settingsRoot)
        cleanCheckout = true
        showDependenciesChanges = true
    }
    params {
        param("opaque.pytest.paths", shard?.path ?: "packages tests")
        param("opaque.pytest.marker", device.markerPr)
        param("opaque.pytest.marker.main", device.markerMain)
        param("opaque.pytest.xdist", device.xdist)
    }
    requirements {
        equals("teamcity.agent.jvm.os.name", device.osName)
        if (device == TestDevice.CUDA) {
            equals("opaque.agent.cuda", "true")
        }
    }
    steps {
        script {
            name = "Sync test environment"
            scriptContent = "uv sync --group dev --all-packages --extra all"
        }
        if (device == TestDevice.CUDA) {
            script {
                name = "Assert CUDA availability"
                scriptContent = "uv run python -c \"import torch; assert torch.cuda.is_available(), 'CUDA not available on GPU agent'\""
            }
        }
        script {
            name = "Run pytest"
            scriptContent = """
                MARKER="%opaque.pytest.marker%"
                if [ "%teamcity.build.branch%" = "main" ]; then
                  MARKER="%opaque.pytest.marker.main%"
                fi
                uv run pytest %opaque.pytest.paths% -m "${'$'}MARKER" %opaque.pytest.xdist% --cov=opaque --cov-report=xml:coverage.xml --durations=25 -q
            """.trimIndent()
        }
    }
}

private object RustTests : BuildType({
    id("Opaque_RustTests")
    name = "Rust tests"
    vcs { root(DslContext.settingsRoot) }
    requirements { equals("teamcity.agent.jvm.os.name", "Linux") }
    steps {
        script {
            name = "Run Rust tests and doctests"
            scriptContent = "cargo test --manifest-path packages/opaque-accounting/Cargo.toml"
        }
    }
})

private object StrictDocs : BuildType({
    id("Opaque_StrictDocs")
    name = "Strict documentation build"
    vcs { root(DslContext.settingsRoot) }
    artifactRules = "site/** => site.zip"
    requirements { equals("teamcity.agent.jvm.os.name", "Linux") }
    steps {
        script {
            name = "Sync documentation environment"
            scriptContent = "uv sync --group docs --all-packages"
        }
        script {
            name = "Build documentation"
            scriptContent = "uv run mkdocs build --strict"
        }
    }
})

private val cpuTests = testShards.map { pythonTest(TestDevice.CPU, it) }
private val mpsTests = testShards.map { pythonTest(TestDevice.MPS, it) }
private val cudaTest = pythonTest(TestDevice.CUDA, null)

val opaqueTestBuildTypes = cpuTests + mpsTests + cudaTest + RustTests + StrictDocs

private fun aggregateBuild(
    buildType: BuildType,
    buildId: String,
    buildName: String,
    branchFilter: String,
    dependencies: List<BuildType>,
    cancelRunning: Boolean,
) = buildType.apply {
    id(buildId)
    name = buildName
    type = BuildTypeSettings.Type.COMPOSITE

    vcs {
        root(DslContext.settingsRoot)
        this.branchFilter = branchFilter
    }
    dependencies {
        dependencies.forEach {
            snapshot(it) {
                onDependencyFailure = FailureAction.FAIL_TO_START
                onDependencyCancel = FailureAction.CANCEL
                synchronizeRevisions = true
                reuseBuilds = ReuseBuilds.NO
            }
        }
    }
    triggers {
        vcs {
            this.branchFilter = branchFilter
            enableQueueOptimization = cancelRunning
        }
    }
    features {
        commitStatusPublisher {
            publisher = github {
                githubUrl = "https://api.github.com"
                authType = vcsRoot()
            }
        }
    }
}

object OpaqueTestsPr : BuildType({
    aggregateBuild(
        this,
        buildId = "Opaque_TestsPr",
        buildName = "Opaque tests (PR)",
        branchFilter = "+:pull/*",
        dependencies = cpuTests + mpsTests + RustTests + StrictDocs,
        cancelRunning = true,
    )
    features {
        pullRequests {
            provider = github {
                authType = vcsRoot()
                filterTargetBranch = "+:main"
            }
        }
    }
})

object OpaqueCudaTrustedPr : BuildType({
    aggregateBuild(
        this,
        buildId = "Opaque_CudaTrustedPr",
        buildName = "Opaque CUDA tests (trusted PR)",
        branchFilter = "+:pull/*",
        dependencies = listOf(cudaTest),
        cancelRunning = true,
    )
    features {
        pullRequests {
            provider = github {
                authType = vcsRoot()
                filterTargetBranch = "+:main"
                filterAuthorRole = PullRequests.GitHubRoleFilter.MEMBER
            }
        }
    }
})

object OpaqueTestsMain : BuildType({
    aggregateBuild(
        this,
        buildId = "Opaque_TestsMain",
        buildName = "Opaque tests (main)",
        branchFilter = "+:<default>",
        dependencies = cpuTests + mpsTests + cudaTest + RustTests + StrictDocs,
        cancelRunning = false,
    )
})