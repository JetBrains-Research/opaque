import jetbrains.buildServer.configs.kotlin.BuildType
import jetbrains.buildServer.configs.kotlin.BuildTypeSettings
import jetbrains.buildServer.configs.kotlin.DslContext
import jetbrains.buildServer.configs.kotlin.FailureAction
import jetbrains.buildServer.configs.kotlin.ReuseBuilds
import jetbrains.buildServer.configs.kotlin.buildFeatures.commitStatusPublisher
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.triggers.vcs

private fun distributionBuild(buildType: BuildType, buildId: String, buildName: String) = buildType.apply {
    id(buildId)
    name = buildName
    templates(DistributionBuildTemplate)
}

object PythonWheels : BuildType({
    distributionBuild(this, "Opaque_BuildPythonWheels", "Build pure-Python wheels")
    artifactRules = "${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}/*.whl => distributions"
    useAgent(CiModel.AgentClass.LINUX_SMALL)
    steps {
        script {
            name = "Build platform-agnostic wheels"
            scriptContent = CiModel.pythonDistributions.joinToString("\n") {
                "(cd ${it.path} && uv build --wheel --out-dir %teamcity.build.checkoutDir%/${CiModel.Artifacts.DISTRIBUTION_DIRECTORY})"
            }
        }
    }
})

private fun accountingWheel(target: CiModel.AccountingTarget) = BuildType {
    distributionBuild(this, "Opaque_BuildAccounting${target.id}", "Build opaque-accounting (${target.label})")
    artifactRules = "${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}/*.whl => distributions"
    useAgent(target.agentClass)
    steps {
        script {
            name = "Build native wheel"
            val manylinux = target.manylinux?.let { "--manylinux $it" } ?: ""
            scriptContent = "uv run --isolated --no-project --with maturin maturin build --manifest-path packages/opaque-accounting/Cargo.toml --release --target ${target.rustTarget} $manylinux --interpreter python3.11 --out ${CiModel.Artifacts.DISTRIBUTION_DIRECTORY} --compatibility pypi"
        }
    }
}

object AccountingSdist : BuildType({
    distributionBuild(this, "Opaque_BuildAccountingSdist", "Build opaque-accounting sdist")
    artifactRules = "${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}/*.tar.gz => distributions"
    useAgent(CiModel.AgentClass.LINUX_SMALL)
    steps {
        script {
            name = "Build accounting sdist"
            scriptContent = "cd packages/opaque-accounting && uv build --sdist --out-dir %teamcity.build.checkoutDir%/${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}"
        }
    }
})

private val accountingWheelBuilds = CiModel.accountingTargets.map(::accountingWheel)
private val distributionBuilders = listOf(PythonWheels) + accountingWheelBuilds + AccountingSdist

object ValidateDistributions : BuildType({
    id("Opaque_ValidateDistributions")
    name = "Validate exact distribution bundle"
    configurePythonTooling()
    artifactRules = "${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}/** => ${CiModel.Artifacts.VALIDATED_BUNDLE}"
    useAgent(CiModel.AgentClass.LINUX_SMALL)
    dependencies {
        distributionBuilders.forEach { builder ->
            snapshot(builder) {
                onDependencyFailure = FailureAction.FAIL_TO_START
                onDependencyCancel = FailureAction.CANCEL
                synchronizeRevisions = true
                reuseBuilds = ReuseBuilds.NO
            }
            artifacts(builder) {
                artifactRules = "+:distributions/* => ${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}"
            }
        }
    }
    steps {
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
            scriptContent = "printf 'commit=%s\\nversion=%s\\nbuild_id=%s\\n' '%build.vcs.number%' '%dep.Opaque_BuildPythonWheels.opaque.version%' '%teamcity.build.id%' > ${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}/${CiModel.Artifacts.IDENTITY_MANIFEST}"
        }
    }
})

val artifactBuildTypes = distributionBuilders + ValidateDistributions

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
            reuseBuilds = ReuseBuilds.NO
        }
    }
    triggers { vcs { this.branchFilter = branchKind.branchFilter } }
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
    distributionChain(this, "Opaque_PreviewDistributions", "Preview distributions", CiModel.BranchKind.PULL_REQUEST)
})

object DevDistributions : BuildType({
    distributionChain(this, "Opaque_DevDistributions", "Dev distributions", CiModel.BranchKind.MAIN)
})

object ReleaseDistributions : BuildType({
    distributionChain(this, "Opaque_ReleaseDistributions", "Release distributions", CiModel.BranchKind.RELEASE_TAG)
})