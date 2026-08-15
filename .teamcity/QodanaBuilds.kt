import jetbrains.buildServer.configs.kotlin.BuildType
import jetbrains.buildServer.configs.kotlin.DslContext
import jetbrains.buildServer.configs.kotlin.FailureAction
import jetbrains.buildServer.configs.kotlin.ReuseBuilds
import jetbrains.buildServer.configs.kotlin.buildSteps.Qodana
import jetbrains.buildServer.configs.kotlin.buildSteps.qodana

private fun qodanaGate(
    buildType: BuildType,
    buildId: String,
    buildName: String,
    branchKind: CiModel.BranchKind,
    verificationBuilds: List<BuildType>,
) = buildType.apply {
    id(buildId)
    name = buildName
    vcs {
        root(DslContext.settingsRoot)
        branchFilter = branchKind.branchFilter
    }
    useAgent(CiModel.AgentClass.LINUX_SMALL)
    dependencies {
        verificationBuilds.forEach { verificationBuild ->
            snapshot(verificationBuild) {
                onDependencyFailure = FailureAction.FAIL_TO_START
                onDependencyCancel = FailureAction.CANCEL
                synchronizeRevisions = true
                reuseBuilds = ReuseBuilds.SUCCESSFUL
            }
        }
    }
    steps {
        qodana {
            name = "Qodana Python analysis"
            linter = python {
                version = Qodana.PythonVersion.LATEST
            }
            inspectionProfile = embedded {
                name = "qodana.starter"
            }
            cloudToken = "%opaque.qodana.cloud.token%"
            reportAsTests = true
            additionalQodanaArguments = "--baseline qodana.sarif.json --fail-threshold 0"
        }
    }
}

object QodanaPr : BuildType({
    qodanaGate(
        this,
        buildId = "Opaque_QodanaPr",
        buildName = "Qodana PR gate",
        branchKind = CiModel.BranchKind.PULL_REQUEST,
        verificationBuilds = listOf(PinnedVerification),
    )
})

object QodanaMain : BuildType({
    qodanaGate(
        this,
        buildId = "Opaque_QodanaMain",
        buildName = "Qodana main gate",
        branchKind = CiModel.BranchKind.MAIN,
        verificationBuilds = listOf(PinnedVerification, MinimumVerification, MaximumVerification),
    )
})

object QodanaRelease : BuildType({
    qodanaGate(
        this,
        buildId = "Opaque_QodanaRelease",
        buildName = "Qodana release gate",
        branchKind = CiModel.BranchKind.RELEASE_TAG,
        verificationBuilds = listOf(PinnedVerification, MinimumVerification, MaximumVerification),
    )
})

val qodanaBuildTypes = listOf(QodanaPr, QodanaMain, QodanaRelease)