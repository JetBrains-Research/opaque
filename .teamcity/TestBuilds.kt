import jetbrains.buildServer.configs.kotlin.BuildType
import jetbrains.buildServer.configs.kotlin.BuildTypeSettings
import jetbrains.buildServer.configs.kotlin.DslContext
import jetbrains.buildServer.configs.kotlin.FailureAction
import jetbrains.buildServer.configs.kotlin.ReuseBuilds
import jetbrains.buildServer.configs.kotlin.buildFeatures.perfmon
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.matrix

private fun pythonTestMatrix(device: CiModel.TestDevice): BuildType = BuildType {
    id("Opaque_Python${device.name.lowercase().replaceFirstChar(Char::uppercase)}")
    name = "Python ${device.displayName} tests"
    templates(PythonTestTemplate)

    params {
        param("opaque.pytest.path", "packages")
        param("opaque.pytest.marker", device.markerForMain)
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
            scriptContent = "uv sync --python ${CiModel.PYTHON_VERSION} --group dev --all-packages --extra all"
        }
        if (device == CiModel.TestDevice.CUDA) {
            script {
                name = "Assert CUDA availability"
                scriptContent = "uv run python -c \"import torch; assert torch.cuda.is_available(), 'CUDA not available on GPU agent'\""
            }
        }
        script {
            name = "Run pytest"
            scriptContent = "uv run pytest %opaque.pytest.path% -m \"%opaque.pytest.marker%\" %opaque.pytest.xdist% --cov=opaque --cov-report=xml:coverage.xml --junitxml=test-results.xml --durations=25 -q"
        }
    }
}

private object RustTests : BuildType({
    id("Opaque_RustTests")
    name = "Rust tests"
    configureCheckout()
    configureCleanup()
    configureHostedCachePolicy()
    configureAgentDiagnostics()
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
private val mpsTests = pythonTestMatrix(CiModel.TestDevice.MPS).apply {
    paused = true
    description = "Disabled until TeamCity provides a larger compatible macOS hosted-agent type."
}
private val cudaTest = BuildType {
    id("Opaque_PythonCuda")
    name = "Python CUDA tests"
    configurePythonTooling()
    artifactRules = "coverage.xml => coverage"
    failureConditions {
        executionTimeoutMin = CiModel.TEST_TIMEOUT_MINUTES
    }
    useAgent(CiModel.AgentClass.CUDA)
    configurePythonTestReporting()
    params {
        param("opaque.pytest.marker", CiModel.TestDevice.CUDA.markerForMain)
    }
    steps {
        script {
            name = "Sync test environment"
            scriptContent = "uv sync --python ${CiModel.PYTHON_VERSION} --group dev --all-packages --extra all"
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

private fun verificationBuild(
    buildType: BuildType,
    buildId: String,
    buildName: String,
    branchKind: CiModel.BranchKind,
) = buildType.apply {
    id(buildId)
    name = buildName
    type = BuildTypeSettings.Type.COMPOSITE

    vcs {
        root(DslContext.settingsRoot)
        this.branchFilter = branchKind.branchFilter
    }
    params {
        if (branchKind == CiModel.BranchKind.PULL_REQUEST) {
            param(
                "override.dep.Opaque_PythonCpu.opaque.pytest.marker",
                CiModel.TestDevice.CPU.markerForPullRequest,
            )
            param(
                "override.dep.Opaque_PythonCuda.opaque.pytest.marker",
                CiModel.TestDevice.CUDA.markerForPullRequest,
            )
        }
    }
    dependencies {
        listOf(cpuTests, cudaTest).forEach { matrix ->
            snapshot(matrix) {
                onDependencyFailure = FailureAction.FAIL_TO_START
                onDependencyCancel = FailureAction.CANCEL
                synchronizeRevisions = true
                reuseBuilds = ReuseBuilds.SUCCESSFUL
            }
        }
        listOf(RustTests, StrictDocs).forEach {
            snapshot(it) {
                onDependencyFailure = FailureAction.FAIL_TO_START
                onDependencyCancel = FailureAction.CANCEL
                synchronizeRevisions = true
                reuseBuilds = ReuseBuilds.SUCCESSFUL
            }
        }
    }
}

object OpaqueTestsPr : BuildType({
    verificationBuild(
        this,
        buildId = "Opaque_TestsPr",
        buildName = "PR verification",
        branchKind = CiModel.BranchKind.PULL_REQUEST,
    )
})

object OpaqueTestsMain : BuildType({
    verificationBuild(
        this,
        buildId = "Opaque_TestsMain",
        buildName = "Main verification",
        branchKind = CiModel.BranchKind.MAIN,
    )
})

object OpaqueTestsTag : BuildType({
    verificationBuild(
        this,
        buildId = "Opaque_TestsTag",
        buildName = "Tag verification",
        branchKind = CiModel.BranchKind.RELEASE_TAG,
    )
})

val verificationBuildTypes = listOf(
    cpuTests,
    mpsTests,
    cudaTest,
    RustTests,
    StrictDocs,
    OpaqueTestsPr,
    OpaqueTestsMain,
    OpaqueTestsTag,
)