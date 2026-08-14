import jetbrains.buildServer.configs.kotlin.BuildType
import jetbrains.buildServer.configs.kotlin.BuildTypeSettings
import jetbrains.buildServer.configs.kotlin.DslContext
import jetbrains.buildServer.configs.kotlin.FailureAction
import jetbrains.buildServer.configs.kotlin.ReuseBuilds
import jetbrains.buildServer.configs.kotlin.buildSteps.script

object PrepareVersion : BuildType({
    id("Opaque_PrepareVersion")
    name = "Prepare immutable version"

    buildNumberPattern = "%opaque.version%"
    params {
        param("opaque.version", "pending")
        param("opaque.source.build.id", "%teamcity.build.id%")
    }
    outputParams {
        param("opaque.version", "%opaque.version%")
        param("opaque.source.build.id", "%opaque.source.build.id%")
    }
    failureConditions {
        executionTimeoutMin = CiModel.BUILD_TIMEOUT_MINUTES
    }
    configureCheckout()
    configureCleanup()
    configureAgentDiagnostics()
    artifactRules = "${CiModel.Artifacts.VERSION_MANIFEST} => metadata"
    useAgent(CiModel.AgentClass.LINUX_SMALL)
    steps {
        script {
            name = "Compute immutable version"
            scriptContent = """
                ${prepareVersionScript()}
                printf 'commit=%s\nversion=%s\nsource_build_id=%s\ntarget=source\n' '%build.vcs.number%' "${'$'}VERSION" '%teamcity.build.id%' > ${CiModel.Artifacts.VERSION_MANIFEST}
            """.trimIndent()
        }
    }
})

private fun distributionBuild(
    buildType: BuildType,
    buildId: String,
    buildName: String,
    target: String,
) = buildType.apply {
    id(buildId)
    name = buildName
    templates(DistributionBuildTemplate)
    params {
        param("opaque.version", "%dep.Opaque_PrepareVersion.opaque.version%")
        param("opaque.source.build.id", "%dep.Opaque_PrepareVersion.opaque.source.build.id%")
        param("opaque.artifact.target", target)
    }
    dependencies {
        snapshot(PrepareVersion) {
            onDependencyFailure = FailureAction.FAIL_TO_START
            onDependencyCancel = FailureAction.CANCEL
            synchronizeRevisions = true
            reuseBuilds = ReuseBuilds.NO
        }
    }
}

private fun BuildType.recordArtifactManifest() {
    steps {
        script {
            name = "Record artifact manifest"
            scriptContent = "printf 'commit=%s\nversion=%s\nsource_build_id=%s\ntarget=%s\n' '%build.vcs.number%' '%opaque.version%' '%opaque.source.build.id%' '%opaque.artifact.target%' > ${CiModel.Artifacts.VERSION_MANIFEST}"
        }
    }
    pruneUvCacheForCi()
}

object PythonWheels : BuildType({
    distributionBuild(this, "Opaque_BuildPythonWheels", "Build pure-Python wheels", "pure-python")
    artifactRules = "${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}/*.whl => distributions\n${CiModel.Artifacts.VERSION_MANIFEST} => metadata"
    useAgent(CiModel.AgentClass.LINUX_SMALL)
    steps {
        script {
            name = "Build platform-agnostic wheels"
            scriptContent = CiModel.pythonDistributions.joinToString("\n") {
                "(cd ${it.path} && uv build --wheel --out-dir %teamcity.build.checkoutDir%/${CiModel.Artifacts.DISTRIBUTION_DIRECTORY})"
            }
        }
    }
    recordArtifactManifest()
})

private fun accountingWheel(target: CiModel.AccountingTarget) = BuildType {
    distributionBuild(this, "Opaque_BuildAccounting${target.id}", "Build opaque-accounting (${target.label})", "accounting-${target.id}")
    artifactRules = "${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}/*.whl => distributions\n${CiModel.Artifacts.VERSION_MANIFEST} => metadata"
    useAgent(target.agentClass)
    steps {
        script {
            name = "Build native wheel"
            val manylinux = target.manylinux?.let { "--manylinux $it" } ?: ""
            scriptContent = "uv run --isolated --no-project --with maturin maturin build --manifest-path packages/opaque-accounting/Cargo.toml --release --target ${target.rustTarget} $manylinux --interpreter python3.11 --out ${CiModel.Artifacts.DISTRIBUTION_DIRECTORY} --compatibility pypi"
        }
    }
    recordArtifactManifest()
}

object AccountingSdist : BuildType({
    distributionBuild(this, "Opaque_BuildAccountingSdist", "Build opaque-accounting sdist", "accounting-sdist")
    artifactRules = "${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}/*.tar.gz => distributions\n${CiModel.Artifacts.VERSION_MANIFEST} => metadata"
    useAgent(CiModel.AgentClass.LINUX_SMALL)
    steps {
        script {
            name = "Build accounting sdist"
            scriptContent = "cd packages/opaque-accounting && uv build --sdist --out-dir %teamcity.build.checkoutDir%/${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}"
        }
    }
    recordArtifactManifest()
})

private data class DistributionBuilder(val target: String, val buildType: BuildType)

private val accountingWheelBuilds = CiModel.accountingTargets.map { target ->
    DistributionBuilder("accounting-${target.id}", accountingWheel(target))
}
private val distributionBuilders = listOf(DistributionBuilder("pure-python", PythonWheels)) +
    accountingWheelBuilds +
    DistributionBuilder("accounting-sdist", AccountingSdist)

object ValidateDistributions : BuildType({
    id("Opaque_ValidateDistributions")
    name = "Validate exact distribution bundle"
    configurePythonTooling()
    artifactRules = "${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}/** => ${CiModel.Artifacts.VALIDATED_BUNDLE}"
    useAgent(CiModel.AgentClass.LINUX_SMALL)
    params {
        param("opaque.version", "%dep.Opaque_PrepareVersion.opaque.version%")
        param("opaque.source.build.id", "%dep.Opaque_PrepareVersion.opaque.source.build.id%")
    }
    dependencies {
        snapshot(PrepareVersion) {
            onDependencyFailure = FailureAction.FAIL_TO_START
            onDependencyCancel = FailureAction.CANCEL
            synchronizeRevisions = true
            reuseBuilds = ReuseBuilds.NO
        }
        distributionBuilders.forEach { builder ->
            snapshot(builder.buildType) {
                onDependencyFailure = FailureAction.FAIL_TO_START
                onDependencyCancel = FailureAction.CANCEL
                synchronizeRevisions = true
                reuseBuilds = ReuseBuilds.SUCCESSFUL
            }
            artifacts(builder.buildType) {
                buildRule = sameChain()
                artifactRules = """
                    +:distributions/* => ${CiModel.Artifacts.STAGED_INPUT_DIRECTORY}/${builder.target}/artifacts
                    +:metadata/${CiModel.Artifacts.VERSION_MANIFEST} => ${CiModel.Artifacts.STAGED_INPUT_DIRECTORY}/${builder.target}/metadata
                """.trimIndent()
            }
        }
    }
    steps {
        script {
            name = "Stage target-qualified artifacts"
            scriptContent = """
                set -eu
                mkdir -p ${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}
                find ${CiModel.Artifacts.STAGED_INPUT_DIRECTORY} -type f \( -name '*.whl' -o -name '*.tar.gz' \) -exec cp {} ${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}/ \;
            """.trimIndent()
        }
        script {
            name = "Verify source manifests"
            scriptContent = """
                set -eu
                for manifest in ${CiModel.Artifacts.STAGED_INPUT_DIRECTORY}/*/metadata/${CiModel.Artifacts.VERSION_MANIFEST}; do
                  test "${'$'}(sed -n 's/^commit=//p' "${'$'}manifest")" = "%build.vcs.number%"
                  test "${'$'}(sed -n 's/^version=//p' "${'$'}manifest")" = "%opaque.version%"
                  test "${'$'}(sed -n 's/^source_build_id=//p' "${'$'}manifest")" = "%opaque.source.build.id%"
                done
            """.trimIndent()
        }
        script {
            name = "Validate internal pins"
            scriptContent = "uv run --isolated --no-project --with packaging==25.0 python .github/scripts/check_wheel_internal_pins.py --wheel-dir ${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}"
        }
        script {
            name = "Validate accounting artifact policy"
            scriptContent = "python3 .github/scripts/check_accounting_artifact_policy.py --wheel-dir ${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}"
        }
        script {
            name = "Verify complete artifact set"
            scriptContent = "test \"${'$'}(find ${CiModel.Artifacts.DISTRIBUTION_DIRECTORY} -maxdepth 1 -name '*.whl' | wc -l | tr -d ' ')\" = ${CiModel.Artifacts.COMPLETE_WHEEL_COUNT} && test \"${'$'}(find ${CiModel.Artifacts.DISTRIBUTION_DIRECTORY} -maxdepth 1 -name '*.tar.gz' | wc -l | tr -d ' ')\" = ${CiModel.Artifacts.SDIST_COUNT}"
        }
        script {
            name = "Record artifact identity"
            scriptContent = "printf 'commit=%s\\nversion=%s\\nsource_build_id=%s\\nbuild_id=%s\\n' '%build.vcs.number%' '%opaque.version%' '%opaque.source.build.id%' '%teamcity.build.id%' > ${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}/${CiModel.Artifacts.IDENTITY_MANIFEST}"
        }
    }
    pruneUvCacheForCi()
})

private fun distributionChain(
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
    dependencies {
        snapshot(ValidateDistributions) {
            onDependencyFailure = FailureAction.FAIL_TO_START
            onDependencyCancel = FailureAction.CANCEL
            synchronizeRevisions = true
            reuseBuilds = ReuseBuilds.SUCCESSFUL
        }
    }
}

object PreviewDistributions : BuildType({
    distributionChain(this, "Opaque_PreviewDistributions", "Preview artifacts", CiModel.BranchKind.PULL_REQUEST)
})

object DevDistributions : BuildType({
    distributionChain(this, "Opaque_DevDistributions", "Dev artifacts", CiModel.BranchKind.MAIN)
})

object ReleaseDistributions : BuildType({
    distributionChain(this, "Opaque_ReleaseDistributions", "Release artifacts", CiModel.BranchKind.RELEASE_TAG)
})

val artifactBuildTypes = listOf(PrepareVersion) +
    distributionBuilders.map { it.buildType } +
    listOf(ValidateDistributions, PreviewDistributions, DevDistributions, ReleaseDistributions)