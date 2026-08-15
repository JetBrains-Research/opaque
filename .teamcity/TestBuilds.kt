import jetbrains.buildServer.configs.kotlin.BuildType
import jetbrains.buildServer.configs.kotlin.BuildTypeSettings
import jetbrains.buildServer.configs.kotlin.FailureAction
import jetbrains.buildServer.configs.kotlin.ReuseBuilds
import jetbrains.buildServer.configs.kotlin.buildFeatures.perfmon
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.matrix

private fun CiModel.VerificationProfile.pythonVersionPreparationScript() = """
    PYTHON_VERSION=${'$'}(
      uv run --isolated --no-project --with packaging==25.0 python - ${pythonVersionSelection.name.lowercase()} <<'PY'
    import sys
    import tomllib
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    selection = sys.argv[1]
    with open("pyproject.toml", "rb") as project_file:
        requires_python = tomllib.load(project_file)["project"]["requires-python"]
    supported = [
        Version(f"{major}.{minor}")
        for major in range(2, 10)
        for minor in range(100)
        if Version(f"{major}.{minor}") in SpecifierSet(requires_python)
    ]
    if not supported:
        raise SystemExit(f"No Python version satisfies {requires_python!r}")
    print(min(supported) if selection == "minimum" else max(supported))
    PY
    )
    echo "Using Python ${'$'}PYTHON_VERSION"
""".trimIndent()

private fun CiModel.VerificationProfile.projectDirectory() =
    if (dependencyResolution.isolated) "%teamcity.build.tempDir%/opaque-$id" else "%teamcity.build.checkoutDir%"

private fun CiModel.VerificationProfile.prepareProjectScript(): String {
    if (!dependencyResolution.isolated) {
        return """
            ${pythonVersionPreparationScript()}
            uv sync --python "${'$'}PYTHON_VERSION" --group dev --all-packages --extra all ${dependencyResolution.syncArguments}
        """.trimIndent()
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
        ${pythonVersionPreparationScript()}
        uv sync --python "${'$'}PYTHON_VERSION" --group dev --all-packages --extra all ${dependencyResolution.syncArguments}
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

object StrictDocs : BuildType({
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

private fun tests() = BuildType {
    id("Opaque_Tests")
    name = "Tests"
    type = BuildTypeSettings.Type.COMPOSITE
    dependencies {
        snapshot(profileTestMatrices.getValue(CiModel.pinnedVerification)) {
            onDependencyFailure = FailureAction.FAIL_TO_START
            onDependencyCancel = FailureAction.CANCEL
            synchronizeRevisions = true
            reuseBuilds = ReuseBuilds.SUCCESSFUL
        }
        snapshot(RustTests) {
            onDependencyFailure = FailureAction.FAIL_TO_START
            onDependencyCancel = FailureAction.CANCEL
            synchronizeRevisions = true
            reuseBuilds = ReuseBuilds.SUCCESSFUL
        }
    }
}

private fun dependencyVersionVerification() = BuildType {
    id("Opaque_DependencyVersionVerification")
    name = "Dependency version verification"
    type = BuildTypeSettings.Type.COMPOSITE
    dependencies {
        listOf(CiModel.minimumVerification, CiModel.maximumVerification).forEach { profile ->
            snapshot(profileTestMatrices.getValue(profile)) {
                onDependencyFailure = FailureAction.FAIL_TO_START
                onDependencyCancel = FailureAction.CANCEL
                synchronizeRevisions = true
                reuseBuilds = ReuseBuilds.SUCCESSFUL
            }
        }
    }
}

private fun platformVerification() = BuildType {
    id("Opaque_PlatformVerification")
    name = "Platform verification"
    type = BuildTypeSettings.Type.COMPOSITE
    dependencies {
        platformWheelBuildTypes.forEach { buildType ->
            snapshot(buildType) {
                onDependencyFailure = FailureAction.FAIL_TO_START
                onDependencyCancel = FailureAction.CANCEL
                synchronizeRevisions = true
                reuseBuilds = ReuseBuilds.SUCCESSFUL
            }
        }
    }
}

val Tests = tests()
val DependencyVersionVerification = dependencyVersionVerification()
val PlatformVerification = platformVerification()

val verificationBuildTypes = listOf(
    *profileTestMatrices.values.toTypedArray(),
    mpsTests,
    cudaTest,
    RustTests,
    StrictDocs,
    Tests,
    DependencyVersionVerification,
    PlatformVerification,
)