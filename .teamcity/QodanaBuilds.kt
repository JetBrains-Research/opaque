import jetbrains.buildServer.configs.kotlin.BuildType
import jetbrains.buildServer.configs.kotlin.DslContext
import jetbrains.buildServer.configs.kotlin.FailureAction
import jetbrains.buildServer.configs.kotlin.ReuseBuilds
import jetbrains.buildServer.configs.kotlin.buildSteps.Qodana
import jetbrains.buildServer.configs.kotlin.buildSteps.qodana

private fun qodanaGate(
    buildType: BuildType,
) = buildType.apply {
    id("Opaque_Qodana")
    name = "Qodana"
    vcs {
        root(DslContext.settingsRoot)
    }
    useAgent(CiModel.AgentClass.LINUX_LARGE)
    dependencies {
        snapshot(Tests) {
            onDependencyFailure = FailureAction.FAIL_TO_START
            onDependencyCancel = FailureAction.CANCEL
            synchronizeRevisions = true
            reuseBuilds = ReuseBuilds.SUCCESSFUL
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

object Qodana : BuildType({ qodanaGate(this) })

val qodanaBuildTypes = listOf(Qodana)