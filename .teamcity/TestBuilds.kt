import jetbrains.buildServer.configs.kotlin.BuildType
import jetbrains.buildServer.configs.kotlin.BuildTypeSettings
import jetbrains.buildServer.configs.kotlin.DslContext
import jetbrains.buildServer.configs.kotlin.FailureAction
import jetbrains.buildServer.configs.kotlin.ReuseBuilds
import jetbrains.buildServer.configs.kotlin.buildFeatures.commitStatusPublisher
import jetbrains.buildServer.configs.kotlin.buildFeatures.perfmon
import jetbrains.buildServer.configs.kotlin.buildFeatures.pullRequests
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.matrix
import jetbrains.buildServer.configs.kotlin.triggers.vcs

private fun pythonTestMatrix(device: CiModel.TestDevice): BuildType = BuildType {
    id("Opaque_Python${device.name.lowercase().replaceFirstChar(Char::uppercase)}")
    name = "Python ${device.displayName} tests"
    templates(PythonTestTemplate)

    params {
        param("opaque.pytest.path", "packages")
        param("opaque.pytest.marker", device.markerForPullRequest)
        param("opaque.pytest.marker.main", device.markerForMain)
        param("opaque.pytest.xdist", device.xdistArguments)
    }
    features {
        matrix {
            param(
                "opaque.pytest.path",
                CiModel.testShards.map { value(it.path, it.label) },
            )
        }
    }
    useAgent(device.agentClass)
    steps {
        script {
            name = "Sync test environment"
            scriptContent = "uv sync --group dev --all-packages --extra all"
        }
        if (device == CiModel.TestDevice.CUDA) {
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
                uv run pytest %opaque.pytest.path% -m "${'$'}MARKER" %opaque.pytest.xdist% --cov=opaque --cov-report=xml:coverage.xml --junitxml=test-results.xml --durations=25 -q
            """.trimIndent()
        }
    }
}

private object RustTests : BuildType({
    id("Opaque_RustTests")
    name = "Rust tests"
    configureCheckout()
    configureCleanup()
    configureEphemeralCachePolicy()
    useAgent(CiModel.AgentClass.LINUX_LARGE)
    failureConditions {
        executionTimeoutMin = CiModel.TEST_TIMEOUT_MINUTES
    }
    features { perfmon { } }
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
    templates(PythonUtilityTemplate)
    artifactRules = "site/** => site.zip"
    useAgent(CiModel.AgentClass.LINUX_SMALL)
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

private val cpuTests = pythonTestMatrix(CiModel.TestDevice.CPU)
private val mpsTests = pythonTestMatrix(CiModel.TestDevice.MPS)
private val cudaTest = BuildType {
    id("Opaque_PythonCuda")
    name = "Python CUDA tests"
    paused = true
    configurePythonTooling()
    artifactRules = "coverage.xml => coverage"
    failureConditions {
        executionTimeoutMin = CiModel.TEST_TIMEOUT_MINUTES
    }
    useAgent(CiModel.AgentClass.CUDA)
    configurePythonTestReporting()
    params {
        param("opaque.pytest.marker", CiModel.TestDevice.CUDA.markerForPullRequest)
        param("opaque.pytest.marker.main", CiModel.TestDevice.CUDA.markerForMain)
    }
    steps {
        script {
            name = "Sync test environment"
            scriptContent = "uv sync --group dev --all-packages --extra all"
        }
        script {
            name = "Assert CUDA availability"
            scriptContent = "uv run python -c \"import torch; assert torch.cuda.is_available(), 'CUDA not available on GPU agent'\""
        }
        script {
            name = "Run pytest"
            scriptContent = "uv run pytest packages tests -m \"%opaque.pytest.marker%\" -rs --cov=opaque --cov-report=xml:coverage.xml --junitxml=test-results.xml --durations=25 -q"
        }
    }
}

val verificationBuildTypes = listOf(cpuTests, mpsTests, cudaTest, RustTests, StrictDocs)

private fun aggregateBuild(
    buildType: BuildType,
    buildId: String,
    buildName: String,
    branchKind: CiModel.BranchKind,
    dependencies: List<BuildType>,
    cancelRunning: Boolean,
) = buildType.apply {
    id(buildId)
    name = buildName
    type = BuildTypeSettings.Type.COMPOSITE

    vcs {
        root(DslContext.settingsRoot)
        this.branchFilter = branchKind.branchFilter
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
            this.branchFilter = branchKind.branchFilter
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
        branchKind = CiModel.BranchKind.PULL_REQUEST,
        dependencies = listOf(cpuTests, mpsTests, RustTests, StrictDocs),
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
    id("Opaque_CudaTrustedPr")
    name = "CUDA verification"
    type = BuildTypeSettings.Type.COMPOSITE
    paused = true
    vcs {
        root(DslContext.settingsRoot)
        branchFilter = CiModel.BranchKind.PULL_REQUEST.branchFilter
    }
    dependencies {
        snapshot(cudaTest) {
            onDependencyFailure = FailureAction.FAIL_TO_START
            onDependencyCancel = FailureAction.CANCEL
            synchronizeRevisions = true
            reuseBuilds = ReuseBuilds.NO
        }
    }
})

object OpaqueTestsMain : BuildType({
    aggregateBuild(
        this,
        buildId = "Opaque_TestsMain",
        buildName = "Opaque tests (main)",
        branchKind = CiModel.BranchKind.MAIN,
        dependencies = listOf(cpuTests, mpsTests, RustTests, StrictDocs),
        cancelRunning = false,
    )
})