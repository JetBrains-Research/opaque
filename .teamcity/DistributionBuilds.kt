import jetbrains.buildServer.configs.kotlin.BuildType
import jetbrains.buildServer.configs.kotlin.BuildTypeSettings
import jetbrains.buildServer.configs.kotlin.DslContext
import jetbrains.buildServer.configs.kotlin.FailureAction
import jetbrains.buildServer.configs.kotlin.ReuseBuilds
import jetbrains.buildServer.configs.kotlin.buildFeatures.commitStatusPublisher
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.matrix
import jetbrains.buildServer.configs.kotlin.triggers.vcs

private data class PythonDistribution(val label: String, val path: String)

private data class AccountingVariant(val id: String, val label: String)

private val pythonDistributions = listOf(
    PythonDistribution("opaque", "."),
    PythonDistribution("opaque-alignment", "packages/opaque-alignment"),
    PythonDistribution("opaque-auditing", "packages/opaque-auditing"),
    PythonDistribution("opaque-base", "packages/opaque-base"),
    PythonDistribution("opaque-dpftrl", "packages/opaque-dpftrl"),
    PythonDistribution("opaque-dpsgd", "packages/opaque-dpsgd"),
    PythonDistribution("opaque-engine", "packages/opaque-engine"),
    PythonDistribution("opaque-optimizers", "packages/opaque-optimizers"),
    PythonDistribution("opaque-patches", "packages/opaque-patches"),
    PythonDistribution("opaque-transformers", "packages/opaque-transformers"),
)

private val accountingVariants = listOf(
    AccountingVariant("native-linux-x64", "native Linux x86_64"),
    AccountingVariant("native-linux-arm64", "native Linux ARM64"),
    AccountingVariant("native-macos-arm64", "native macOS ARM64"),
    AccountingVariant("sdist-linux-x64", "sdist Linux x86_64"),
)

private val versionedPyprojects = listOf(
    "pyproject.toml",
    "packages/opaque-accounting/pyproject.toml",
    *pythonDistributions.filter { it.path != "." }.map { "${it.path}/pyproject.toml" }.toTypedArray(),
)

private fun ensureUv() = """
    set -eu
    if ! command -v uv >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/install.sh | sh
      echo "##teamcity[setParameter name='env.PATH' value='${'$'}HOME/.local/bin:${'$'}PATH']"
    fi
    uv --version
""".trimIndent()

private fun versionPreflight(): String {
    val files = versionedPyprojects.joinToString(" ") { "'$it'" }
    return """
        set -eu
        BRANCH='%teamcity.build.branch%'
        if printf '%s' "${'$'}BRANCH" | grep -Eq '^v[0-9]+\.[0-9]+\.[0-9]+\.(post|alpha|beta|rc)[0-9]+${'$'}|^v[0-9]+\.[0-9]+\.[0-9]+${'$'}'; then
          VERSION="${'$'}{BRANCH#v}"
        else
          LAST="${'$'}(git tag --list 'v*' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+${'$'}' | sort -V -r | head -1 || true)"
          if [[ -z "${'$'}LAST" ]]; then
            BASE=0.0.1
            DISTANCE=0
          else
            IFS=. set -- ${'$'}{LAST#v}
            BASE="${'$'}1.${'$'}2.${'$'}(( ${'$'}3 + 1 ))"
            DISTANCE="${'$'}(git rev-list --count "${'$'}LAST..HEAD")"
          fi
          SHA="${'$'}(git rev-parse --short HEAD)"
          case "${'$'}BRANCH" in
            pull/[0-9]*) VERSION="${'$'}BASE.dev${'$'}DISTANCE+pr.${'$'}{BRANCH#pull/}.g${'$'}SHA" ;;
            *) VERSION="${'$'}BASE.dev${'$'}DISTANCE+g${'$'}SHA" ;;
          esac
        fi
        CARGO_VERSION="${'$'}(printf '%s' "${'$'}VERSION" | sed -E -e 's/\.(alpha|beta|rc|post)([0-9]+)\.(dev)([0-9]+)/-\1.\2.\3.\4/' -e 's/\.(alpha|beta|rc)([0-9]+)/-\1.\2/' -e 's/\.post([0-9]+)\+([0-9A-Za-z.-]+)/+post.\1.\2/' -e 's/\.post([0-9]+)/+post.\1/' -e 's/\.dev([0-9]+)/-dev.\1/')"
        perl -pi -e "s/^version = \"[^\"]+\"/version = \"${'$'}VERSION\"/" packages/opaque-accounting/pyproject.toml
        perl -pi -e "s/^version = \"[^\"]+\"/version = \"${'$'}CARGO_VERSION\"/" Cargo.toml
        perl -pi -e "s/>=0\.0\.0\.dev0/==${'$'}VERSION/g" $files
        echo "##teamcity[setParameter name='env.SETUPTOOLS_SCM_PRETEND_VERSION' value='${'$'}VERSION']"
        echo "##teamcity[setParameter name='opaque.version' value='${'$'}VERSION']"
        echo "##teamcity[buildNumber '${'$'}VERSION']"
    """.trimIndent()
}

private fun baseDistributionBuild(buildType: BuildType, buildId: String, buildName: String) = buildType.apply {
    id(buildId)
    name = buildName
    buildNumberPattern = "%opaque.version%"
    params { param("opaque.version", "pending") }
    outputParams { param("opaque.version", "%opaque.version%") }
    cleanup {
        baseRule {
            history(days = 90)
            artifacts(days = 90)
            preventDependencyCleanup = true
        }
    }
    vcs {
        root(DslContext.settingsRoot)
        cleanCheckout = true
    }
    steps {
        script {
            name = "Ensure uv is available"
            scriptContent = ensureUv()
        }
        script {
            name = "Set synchronized distribution version"
            scriptContent = versionPreflight()
        }
    }
}

private object PythonWheels : BuildType({
    baseDistributionBuild(this, "Opaque_BuildPythonWheels", "Build pure-Python wheels")
    artifactRules = "dist/*.whl => distributions"
    requirements { equals("teamcity.agent.jvm.os.name", "Linux") }
    steps {
        script {
            name = "Build platform-agnostic wheels"
            scriptContent = pythonDistributions.joinToString("\n") {
                "(cd ${it.path} && uv build --wheel --out-dir %teamcity.build.checkoutDir%/dist)"
            }
        }
    }
})

private object AccountingDistributions : BuildType({
    baseDistributionBuild(this, "Opaque_BuildAccounting", "Build opaque-accounting distributions")
    artifactRules = "dist/*.whl dist/*.tar.gz => distributions"
    params { param("opaque.accounting.variant", "native-linux-x64") }
    requirements {
        exists("opaque.agent.accounting.%opaque.accounting.variant%")
    }
    features {
        matrix {
            param(
                "opaque.accounting.variant",
                accountingVariants.map { value(it.id, it.label) },
            )
        }
    }
    steps {
        script {
            name = "Build selected accounting distribution"
            scriptContent = """
                case "%opaque.accounting.variant%" in
                  native-linux-x64)
                    uv run --isolated --no-project --with maturin maturin build --manifest-path packages/opaque-accounting/Cargo.toml --release --target x86_64-unknown-linux-gnu --manylinux 2_28 --interpreter python3.11 --out dist --compatibility pypi
                    ;;
                  native-linux-arm64)
                    uv run --isolated --no-project --with maturin maturin build --manifest-path packages/opaque-accounting/Cargo.toml --release --target aarch64-unknown-linux-gnu --manylinux 2_28 --interpreter python3.11 --out dist --compatibility pypi
                    ;;
                  native-macos-arm64)
                    uv run --isolated --no-project --with maturin maturin build --manifest-path packages/opaque-accounting/Cargo.toml --release --target aarch64-apple-darwin --interpreter python3.11 --out dist --compatibility pypi
                    ;;
                  sdist-linux-x64)
                    cd packages/opaque-accounting && uv build --sdist --out-dir %teamcity.build.checkoutDir%/dist
                    ;;
                esac
            """.trimIndent()
        }
    }
})

private val distributionBuilders = listOf(PythonWheels, AccountingDistributions)

object ValidateDistributions : BuildType({
    id("Opaque_ValidateDistributions")
    name = "Validate exact distribution bundle"
    vcs { root(DslContext.settingsRoot) }
    artifactRules = "dist/** => validated-distributions.zip"
    requirements { equals("teamcity.agent.jvm.os.name", "Linux") }
    dependencies {
        distributionBuilders.forEach { builder ->
            snapshot(builder) {
                onDependencyFailure = FailureAction.FAIL_TO_START
                onDependencyCancel = FailureAction.CANCEL
                synchronizeRevisions = true
                reuseBuilds = ReuseBuilds.NO
            }
            artifacts(builder) {
                artifactRules = "+:distributions/* => dist"
            }
        }
    }
    steps {
        script {
            name = "Validate internal pins"
            scriptContent = "uv run --isolated --no-project --with packaging==25.0 python .github/scripts/check_wheel_internal_pins.py --wheel-dir dist"
        }
        script {
            name = "Validate accounting artifact policy"
            scriptContent = "python3 .github/scripts/check_accounting_artifact_policy.py --wheel-dir dist"
        }
        script {
            name = "Verify complete artifact set"
            scriptContent = "test \"${'$'}(find dist -maxdepth 1 -name '*.whl' | wc -l | tr -d ' ')\" = 13 && test \"${'$'}(find dist -maxdepth 1 -name '*.tar.gz' | wc -l | tr -d ' ')\" = 1"
        }
        script {
            name = "Record artifact identity"
            scriptContent = "printf 'commit=%s\\nversion=%s\\nbuild_id=%s\\n' '%build.vcs.number%' '%dep.Opaque_BuildPythonWheels.opaque.version%' '%teamcity.build.id%' > dist/teamcity-artifact-identity.txt"
        }
    }
    cleanup {
        baseRule {
            history(days = 90)
            artifacts(days = 90)
            preventDependencyCleanup = true
        }
    }
})

val opaqueDistributionBuildTypes = distributionBuilders + ValidateDistributions

private fun distributionChain(buildType: BuildType, buildId: String, buildName: String, branchFilter: String) = buildType.apply {
    id(buildId)
    name = buildName
    type = BuildTypeSettings.Type.COMPOSITE
    vcs {
        root(DslContext.settingsRoot)
        this.branchFilter = branchFilter
    }
    dependencies {
        snapshot(ValidateDistributions) {
            onDependencyFailure = FailureAction.FAIL_TO_START
            onDependencyCancel = FailureAction.CANCEL
            synchronizeRevisions = true
            reuseBuilds = ReuseBuilds.NO
        }
    }
    triggers { vcs { this.branchFilter = branchFilter } }
    features {
        commitStatusPublisher {
            publisher = github {
                githubUrl = "https://api.github.com"
                authType = vcsRoot()
            }
        }
    }
}

object PreviewDistributions : BuildType({
    distributionChain(this, "Opaque_PreviewDistributions", "Preview distributions", "+:pull/*")
})

object DevDistributions : BuildType({
    distributionChain(this, "Opaque_DevDistributions", "Dev distributions", "+:<default>")
})

object ReleaseDistributions : BuildType({
    distributionChain(this, "Opaque_ReleaseDistributions", "Release distributions", "+:v*")
})