import jetbrains.buildServer.configs.kotlin.BuildType
import jetbrains.buildServer.configs.kotlin.buildFeatures.perfmon
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.matrix

object PythonTests : BuildType({
    id("Opaque_PythonTests")
    name = "Python tests"
    templates(PythonTestTemplate)

    params {
        param("teamcity.build.tempDir", "%teamcity.build.checkoutDir%/.teamcity-tmp")
        param("opaque.pytest.path", "packages")
        param("opaque.pytest.marker.pullRequest", CiModel.PULL_REQUEST_CPU_MARKER)
        param("opaque.pytest.marker.main", CiModel.MAIN_CPU_MARKER)
        param("opaque.pytest.xdist", "-n auto --dist loadscope")
    }
    features {
        matrix {
            param(
                "opaque.pytest.path",
                CiModel.testShards.map { value(it.path, it.label) },
            )
            groupArtifactsByBuild = true
        }
    }
    useAgent(CiModel.AgentClass.LINUX_LARGE)
    steps {
        script {
            name = "Sync test environment"
            scriptContent = """
                PYTHON_VERSION=${'$'}(
                  uv run --isolated --no-project --with packaging==25.0 python - <<'PY'
                import tomllib
                from packaging.specifiers import SpecifierSet
                from packaging.version import Version

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
                print(min(supported))
                PY
                )
                echo "Using Python ${'$'}PYTHON_VERSION"
                uv sync --python "${'$'}PYTHON_VERSION" --group dev --all-packages --extra all --locked
            """.trimIndent()
        }
        script {
            name = "Run pytest"
            scriptContent = """
                set -eu
                trap 'for output in coverage.xml .coverage test-results.xml; do test -f "${'$'}output" && cp "${'$'}output" %teamcity.build.checkoutDir%/ || true; done' EXIT
                printf '[run]\nrelative_files = true\n' > .teamcity-coverage.ini
                case '%teamcity.build.branch%' in
                  pull/*) PYTEST_MARKER='%opaque.pytest.marker.pullRequest%' ;;
                  *) PYTEST_MARKER='%opaque.pytest.marker.main%' ;;
                esac
                uv run --frozen pytest %opaque.pytest.path% -m "${'$'}PYTEST_MARKER" %opaque.pytest.xdist% --cov=opaque --cov-config=.teamcity-coverage.ini --cov-report=xml:coverage.xml --junitxml=test-results.xml --durations=25 -q
            """.trimIndent()
        }
    }
    pruneUvCacheForCi()
})

object RustTests : BuildType({
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

val verificationBuildTypes = listOf(PythonTests, RustTests)
