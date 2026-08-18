import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildFeatures.PullRequests
import jetbrains.buildServer.configs.kotlin.buildFeatures.XmlReport
import jetbrains.buildServer.configs.kotlin.buildFeatures.buildCache
import jetbrains.buildServer.configs.kotlin.buildFeatures.perfmon
import jetbrains.buildServer.configs.kotlin.buildFeatures.pullRequests
import jetbrains.buildServer.configs.kotlin.buildFeatures.xmlReport
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.matrix
import jetbrains.buildServer.configs.kotlin.projectFeatures.UntrustedBuildsSettings
import jetbrains.buildServer.configs.kotlin.projectFeatures.kubernetesConnection
import jetbrains.buildServer.configs.kotlin.projectFeatures.untrustedBuildsSettings
import jetbrains.buildServer.configs.kotlin.triggers.gitHubChecks
import jetbrains.buildServer.configs.kotlin.vcs.GitVcsRoot

/*
The settings script is an entry point for defining a TeamCity
project hierarchy. The script should contain a single call to the
project() function with a Project instance or an init function as
an argument.

VcsRoots, BuildTypes, Templates, and subprojects can be
registered inside the project using the vcsRoot(), buildType(),
template(), and subProject() methods respectively.

To debug settings scripts in command-line, run the

    mvnDebug org.jetbrains.teamcity:teamcity-configs-maven-plugin:generate

command and attach your debugger to the port 8000.

To debug in IntelliJ Idea, open the 'Maven Projects' tool window (View
-> Tool Windows -> Maven Projects), find the generate task node
(Plugins -> teamcity-configs -> teamcity-configs:generate), the
'Debug' option is available in the context menu for the task.
*/

version = "2026.1"

project {
    description = "TeamCity runs Linux CPU Python shards and accounting Rust tests; GitHub Actions owns all other CI/CD work."

    vcsRoot(HttpsGithubComJetBrainsResearchOpaqueRefsHeadsMain)

    buildType(Opaque_LinuxTests)

    features {
        untrustedBuildsSettings {
            id = "Opaque_UntrustedBuildApproval"
            defaultAction = UntrustedBuildsSettings.DefaultAction.APPROVE
            enableLog = true
            approvalRules = "user:evgeny.grigorenko@jetbrains.com"
            timeoutMinutes = 60
        }
        kubernetesConnection {
            id = "PROJECT_EXT_13"
            name = "k8s-eqx / agents-jbr-fed"
            apiServerUrl = "https://eqx.k8s.intellij.net:6443"
            caCertificate = "credentialsJSON:e3d2e14e-fade-49de-a8d0-89b5ebe305e0"
            namespace = "agents-jbr-fed"
            authStrategy = token {
                token = "credentialsJSON:e425f8e7-fe04-40b8-b5bc-b35f38405e35"
            }
        }
    }

    subProject(Opaque_Verification)
}

object Opaque_LinuxTests : BuildType({
    name = "TeamCity Linux tests"

    type = BuildTypeSettings.Type.COMPOSITE

    vcs {
        root(DslContext.settingsRoot)

        branchFilter = """
            +:main
            +:pull/*
        """.trimIndent()
    }

    triggers {
        gitHubChecks {
        }
    }

    features {
        pullRequests {
            provider = github {
                authType = vcsRoot()
                filterTargetBranch = "+:main"
                filterAuthorRole = PullRequests.GitHubRoleFilter.EVERYBODY
            }
        }
    }

    dependencies {
        snapshot(Opaque_PythonTests) {
            onDependencyFailure = FailureAction.FAIL_TO_START
            onDependencyCancel = FailureAction.CANCEL
        }
        snapshot(Opaque_RustTests) {
            onDependencyFailure = FailureAction.FAIL_TO_START
            onDependencyCancel = FailureAction.CANCEL
        }
    }
})

object HttpsGithubComJetBrainsResearchOpaqueRefsHeadsMain : GitVcsRoot({
    name = "https://github.com/JetBrains-Research/opaque#refs/heads/teamcity-prototype-stabilization"
    url = "https://github.com/JetBrains-Research/opaque"
    branch = "refs/heads/teamcity-prototype-stabilization"
    branchSpec = "+:refs/heads/*"
    authMethod = password {
        userName = "oauth2"
        password = "credentialsJSON:b5969cfb-b453-45e1-b5c6-02690cef6977"
    }
    param("tokenId", "tc_token_id:CID_8f0b5e59348f7f723191b83227c8a1e1:-1:92908327-48e1-4c13-b5ca-b4f0f7e6e947")
})


object Opaque_Verification : Project({
    name = "Verification"

    buildType(Opaque_PythonTests)
    buildType(Opaque_RustTests)

    template(Opaque_PythonTestTemplate)
})

object Opaque_PythonTests : BuildType({
    templates(Opaque_PythonTestTemplate)
    name = "Python tests"

    params {
        param("opaque.pytest.xdist", "-n auto --dist loadscope")
        param("opaque.pytest.marker.pullRequest", "not cuda and not mps and not slow")
        param("teamcity.build.tempDir", "%teamcity.build.checkoutDir%/.teamcity-tmp")
        param("opaque.pytest.marker.main", "not cuda and not mps")
        param("opaque.pytest.path", "packages")
    }

    steps {
        script {
            name = "Sync test environment"
            id = "RUNNER_1"
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
            id = "RUNNER_2"
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
        script {
            name = "Prune uv cache for CI"
            id = "RUNNER_3"
            scriptContent = "uv cache prune --ci"
        }
    }

    features {
        matrix {
            id = "BUILD_EXT_1"
            param("opaque.pytest.path", listOf(
                value("packages/opaque-accounting", label = "accounting"),
                value("packages/opaque-alignment", label = "alignment"),
                value("packages/opaque-auditing", label = "auditing"),
                value("packages/opaque-base", label = "base"),
                value("packages/opaque-dpftrl", label = "dpftrl"),
                value("packages/opaque-dpsgd", label = "dpsgd"),
                value("packages/opaque-engine", label = "engine"),
                value("packages/opaque-optimizers", label = "optimizers"),
                value("packages/opaque-patches", label = "patches"),
                value("packages/opaque-transformers", label = "transformers"),
                value("tests", label = "repository")
            ))
        }
    }

    requirements {
        startsWith("teamcity.agent.name", "Linux-Large", "RQ_1")
        equals("teamcity.agent.jvm.os.name", "Linux", "RQ_2")
        equals("teamcity.agent.jvm.os.arch", "amd64", "RQ_3")
    }
})

object Opaque_RustTests : BuildType({
    name = "Rust tests"

    params {
        param("env.UV_CACHE_DIR", "%teamcity.build.checkoutDir%/.teamcity-cache/uv")
        param("env.CARGO_HOME", "%teamcity.build.checkoutDir%/.teamcity-cache/cargo")
        param("opaque.cache.scope", "branch-%teamcity.build.branch%")
    }

    vcs {
        root(HttpsGithubComJetBrainsResearchOpaqueRefsHeadsMain)

        cleanCheckout = true
        showDependenciesChanges = true
    }

    steps {
        script {
            name = "Report agent environment"
            scriptContent = """
                printf 'agent=%%s\nos=%%s\narchitecture=%%s\nmemory_mb=%%s\n' \
                  '%teamcity.agent.name%' \
                  '%teamcity.agent.jvm.os.name%' \
                  '%teamcity.agent.jvm.os.arch%' \
                  '%teamcity.agent.hardware.memorySizeMb%'
                uname -a
            """.trimIndent()
        }
        script {
            name = "Run Rust tests and doctests"
            scriptContent = "cargo test --manifest-path packages/opaque-accounting/Cargo.toml"
        }
    }

    failureConditions {
        executionTimeoutMin = 40
    }

    features {
        buildCache {
            name = "opaque-uv-%opaque.cache.scope%-%teamcity.agent.jvm.os.name%-%teamcity.agent.jvm.os.arch%"
            rules = ".teamcity-cache/uv"
        }
        buildCache {
            name = "opaque-cargo-%opaque.cache.scope%-%teamcity.agent.jvm.os.name%-%teamcity.agent.jvm.os.arch%"
            rules = ".teamcity-cache/cargo"
        }
        perfmon {
        }
    }

    requirements {
        startsWith("teamcity.agent.name", "Linux-Large")
        equals("teamcity.agent.jvm.os.name", "Linux")
        equals("teamcity.agent.jvm.os.arch", "amd64")
    }

    cleanup {
        baseRule {
            history(days = 90)
            artifacts(days = 90, artifactPatterns = "+:**/*")
            preventDependencyCleanup = true
        }
    }
})

object Opaque_PythonTestTemplate : Template({
    name = "Python test template"

    artifactRules = """
        coverage.xml => coverage
        .coverage => coverage-data
    """.trimIndent()

    params {
        param("env.UV_CACHE_DIR", "%teamcity.build.checkoutDir%/.teamcity-cache/uv")
        param("env.CARGO_HOME", "%teamcity.build.checkoutDir%/.teamcity-cache/cargo")
        param("opaque.cache.scope", "branch-%teamcity.build.branch%")
    }

    vcs {
        root(HttpsGithubComJetBrainsResearchOpaqueRefsHeadsMain)

        cleanCheckout = true
        showDependenciesChanges = true
    }

    steps {
        script {
            name = "Bootstrap uv"
            id = "TEMPLATE_RUNNER_1"
            scriptContent = """
                set -eu
                if ! command -v uv >/dev/null 2>&1; then
                  curl -LsSf https://astral.sh/uv/install.sh | sh
                  export PATH="${'$'}HOME/.local/bin:${'$'}PATH"
                  echo "##teamcity[setParameter name='env.PATH' value='${'$'}PATH']"
                fi
                uv --version
            """.trimIndent()
        }
        script {
            name = "Report agent environment"
            id = "TEMPLATE_RUNNER_2"
            scriptContent = """
                printf 'agent=%%s\nos=%%s\narchitecture=%%s\nmemory_mb=%%s\n' \
                  '%teamcity.agent.name%' \
                  '%teamcity.agent.jvm.os.name%' \
                  '%teamcity.agent.jvm.os.arch%' \
                  '%teamcity.agent.hardware.memorySizeMb%'
                uname -a
            """.trimIndent()
        }
    }

    failureConditions {
        executionTimeoutMin = 40
    }

    features {
        buildCache {
            id = "TEMPLATE_BUILD_EXT_1"
            name = "opaque-uv-%opaque.cache.scope%-%teamcity.agent.jvm.os.name%-%teamcity.agent.jvm.os.arch%"
            rules = ".teamcity-cache/uv"
        }
        buildCache {
            id = "TEMPLATE_BUILD_EXT_2"
            name = "opaque-cargo-%opaque.cache.scope%-%teamcity.agent.jvm.os.name%-%teamcity.agent.jvm.os.arch%"
            rules = ".teamcity-cache/cargo"
        }
        perfmon {
            id = "TEMPLATE_BUILD_EXT_3"
        }
        xmlReport {
            id = "TEMPLATE_BUILD_EXT_4"
            reportType = XmlReport.XmlReportType.JUNIT
            rules = "test-results.xml"
        }
    }

    cleanup {
        baseRule {
            history(days = 90)
            artifacts(days = 90, artifactPatterns = "+:**/*")
            preventDependencyCleanup = true
        }
    }
})
