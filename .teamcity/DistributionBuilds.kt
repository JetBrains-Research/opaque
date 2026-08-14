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

private data class NativeTarget(
    val id: String,
    val displayName: String,
    val osName: String,
    val arch: String,
    val rustTarget: String,
    val manylinux: String?,
)

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

private val nativeTargets = listOf(
    NativeTarget("LinuxX64", "Linux x86_64", "Linux", "x86_64", "x86_64-unknown-linux-gnu", "2_28"),
    NativeTarget("LinuxArm64", "Linux ARM64", "Linux", "aarch64", "aarch64-unknown-linux-gnu", "2_28"),
    NativeTarget("MacArm64", "macOS ARM64", "Mac OS X", "aarch64", "aarch64-apple-darwin", null),
)

private val versionedPyprojects = listOf(
    "pyproject.toml",
    "packages/opaque-accounting/pyproject.toml",
    *pythonDistributions.filter { it.path != "." }.map { "${it.path}/pyproject.toml" }.toTypedArray(),
)

private fun versionPreflight(): String {
    val files = versionedPyprojects.joinToString(" ") { "'$it'" }
    return """
        set -euo pipefail
        BRANCH='%teamcity.build.branch%'
        if [[ "${'$'}BRANCH" =~ ^v([0-9]+\.[0-9]+\.[0-9]+(\.(post|alpha|beta|rc)[0-9]+)?)${'$'} ]]; then
          VERSION="${'$'}{BASH_REMATCH[1]}"
        else
          LAST="${'$'}(git tag --list 'v*' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+${'$'}' | sort -V -r | head -1 || true)"
          if [[ -z "${'$'}LAST" ]]; then
            BASE=0.0.1
            DISTANCE=0
          else
            IFS=. read -r MAJOR MINOR PATCH <<< "${'$'}{LAST#v}"
            BASE="${'$'}MAJOR.${'$'}MINOR.${'$'}((PATCH + 1))"
            DISTANCE="${'$'}(git rev-list --count "${'$'}LAST..HEAD")"
          fi
          SHA="${'$'}(git rev-parse --short HEAD)"
          if [[ "${'$'}BRANCH" =~ ^pull/([0-9]+) ]]; then
            VERSION="${'$'}BASE.dev${'$'}DISTANCE+pr.${'$'}{BASH_REMATCH[1]}.g${'$'}SHA"
          else
            VERSION="${'$'}BASE.dev${'$'}DISTANCE+g${'$'}SHA"
          fi
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
            name = "Set synchronized distribution version"
            scriptContent = versionPreflight()
        }
    }
}

private object PythonWheels : BuildType({
    baseDistributionBuild(this, "Opaque_BuildPythonWheels", "Build pure-Python wheels")
    artifactRules = "dist/*.whl => distributions"
    requirements { equals("teamcity.agent.jvm.os.name", "Linux") }
    params { param("opaque.distribution.path", ".") }
    features {
        matrix {
            param(
                "opaque.distribution.path",
                pythonDistributions.map { value(it.path, it.label) },
            )
        }
    }
    steps {
        script {
            name = "Build wheel"
            scriptContent = "cd %opaque.distribution.path% && uv build --wheel --out-dir %teamcity.build.checkoutDir%/dist"
        }
    }
})

private fun nativeWheel(target: NativeTarget) = BuildType {
    baseDistributionBuild(this, "Opaque_BuildAccounting${target.id}", "Build opaque-accounting (${target.displayName})")
    artifactRules = "dist/*.whl => distributions"
    requirements {
        equals("teamcity.agent.jvm.os.name", target.osName)
        equals("teamcity.agent.jvm.os.arch", target.arch)
    }
    steps {
        script {
            name = "Build native wheel"
            val manylinux = target.manylinux?.let { "--manylinux $it" } ?: ""
            scriptContent = "uv run --isolated --no-project --with maturin maturin build --manifest-path packages/opaque-accounting/Cargo.toml --release --target ${target.rustTarget} $manylinux --interpreter python3.11 --out dist --compatibility pypi"
        }
    }
}

private object AccountingSdist : BuildType({
    baseDistributionBuild(this, "Opaque_BuildAccountingSdist", "Build opaque-accounting sdist")
    artifactRules = "dist/*.tar.gz => distributions"
    requirements { equals("teamcity.agent.jvm.os.name", "Linux") }
    steps {
        script {
            name = "Build sdist"
            scriptContent = "cd packages/opaque-accounting && uv build --sdist --out-dir %teamcity.build.checkoutDir%/dist"
        }
    }
})

private val nativeWheelBuilds = nativeTargets.map(::nativeWheel)
private val distributionBuilders = listOf(PythonWheels) + nativeWheelBuilds + AccountingSdist

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