import jetbrains.buildServer.configs.kotlin.BuildType
import jetbrains.buildServer.configs.kotlin.BuildTypeSettings
import jetbrains.buildServer.configs.kotlin.FailureAction
import jetbrains.buildServer.configs.kotlin.ReuseBuilds
import jetbrains.buildServer.configs.kotlin.buildFeatures.perfmon
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.matrix

private fun CiModel.VerificationProfile.projectDirectory() =
    if (dependencyResolution.isolated) "%teamcity.build.tempDir%/opaque-$id" else "%teamcity.build.checkoutDir%"

private fun CiModel.VerificationProfile.prepareProjectScript(): String {
    if (!dependencyResolution.isolated) {
        return "uv sync --python $pythonVersion --group dev --all-packages --extra all ${dependencyResolution.syncArguments}"
    }
    return """
        set -eu
        PROJECT_DIR="${projectDirectory()}"
        rm -rf "${'$'}PROJECT_DIR"
        mkdir -p "${'$'}PROJECT_DIR"
        rsync -a \
          --exclude .git \
          --exclude .teamcity-cache \
          --exclude .venv \
          --exclude dist \
          --exclude site \
          --exclude target \
          ./ "${'$'}PROJECT_DIR/"
        cd "${'$'}PROJECT_DIR"
        uv sync --python $pythonVersion --group dev --all-packages --extra all ${dependencyResolution.syncArguments}
        test "${'$'}(git -C %teamcity.build.checkoutDir% status --porcelain -- uv.lock)" = ""
    """.trimIndent()
}

private fun pythonTestMatrix(profile: CiModel.VerificationProfile): BuildType = BuildType {
    id("Opaque_Python${profile.id}")
    name = "Python ${profile.displayName} tests"
    templates(PythonTestTemplate)

    params {
        param("teamcity.build.tempDir", "%teamcity.build.checkoutDir%/.teamcity-tmp")
        param("opaque.pytest.path", "packages")
        param("opaque.pytest.marker", profile.pytestMarker)
        param("opaque.pytest.xdist", CiModel.TestDevice.CPU.xdistArguments)
    }
    features {
        matrix {
            param(
                "opaque.pytest.path",
                CiModel.testShards.map { value(it.path, it.label) },
            )
        }
    }
    useAgent(profile.agentClass)
    steps {
        script {
            name = "Sync test environment"
            scriptContent = profile.prepareProjectScript()
        }
        script {
            name = "Run pytest"
            scriptContent = """
                set -eu
                cd ${profile.projectDirectory()}
                trap 'cp coverage.xml test-results.xml %teamcity.build.checkoutDir%/ 2>/dev/null || true' EXIT
                uv run --frozen pytest %opaque.pytest.path% -m "%opaque.pytest.marker%" %opaque.pytest.xdist% --cov=opaque --cov-report=xml:coverage.xml --junitxml=test-results.xml --durations=25 -q
            """.trimIndent()
        }
    }
    pruneUvCacheForCi()
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
    pruneUvCacheForCi()
})

private val profileTestMatrices = CiModel.verificationProfiles.associateWith(::pythonTestMatrix)
private val mpsTests = BuildType {
    id("Opaque_PythonMps")
    name = "Python MPS tests"
    paused = true
    description = "Disabled until TeamCity provides a larger compatible macOS hosted-agent type."
    templates(PythonTestTemplate)
    useAgent(CiModel.TestDevice.MPS.agentClass)
}
private val cudaTest = BuildType {
    id("Opaque_PythonCuda")
    name = "Python CUDA tests"
    paused = true
    description = "Disabled until TeamCity provides a compatible GPU hosted-agent type."
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
            scriptContent = "uv sync --python ${CiModel.PYTHON_311} --group dev --all-packages --extra all --locked"
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
    pruneUvCacheForCi()
}

private fun verificationProfile(profile: CiModel.VerificationProfile) = BuildType {
    id("Opaque_Verification${profile.id}")
    name = "${profile.displayName} verification"
    type = BuildTypeSettings.Type.COMPOSITE
    dependencies {
        snapshot(profileTestMatrices.getValue(profile)) {
            onDependencyFailure = FailureAction.FAIL_TO_START
            onDependencyCancel = FailureAction.CANCEL
            synchronizeRevisions = true
            reuseBuilds = ReuseBuilds.SUCCESSFUL
        }
        if (profile == CiModel.pinnedVerification) {
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
}

val PinnedVerification = verificationProfile(CiModel.pinnedVerification)
val MinimumVerification = verificationProfile(CiModel.minimumVerification)
val MaximumVerification = verificationProfile(CiModel.maximumVerification)

val verificationBuildTypes = listOf(
    *profileTestMatrices.values.toTypedArray(),
    mpsTests,
    cudaTest,
    RustTests,
    StrictDocs,
    PinnedVerification,
    MinimumVerification,
    MaximumVerification,
)