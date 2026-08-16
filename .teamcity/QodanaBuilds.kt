import jetbrains.buildServer.configs.kotlin.BuildType
import jetbrains.buildServer.configs.kotlin.DslContext
import jetbrains.buildServer.configs.kotlin.FailureAction
import jetbrains.buildServer.configs.kotlin.ParameterDisplay
import jetbrains.buildServer.configs.kotlin.ReuseBuilds
import jetbrains.buildServer.configs.kotlin.buildSteps.Qodana
import jetbrains.buildServer.configs.kotlin.buildSteps.qodana
import jetbrains.buildServer.configs.kotlin.buildSteps.script

private fun coverageMerge(
    buildType: BuildType,
    buildId: String,
    buildName: String,
    testMatrices: List<BuildType>,
) = buildType.apply {
    id(buildId)
    name = buildName
    templates(PythonUtilityTemplate)
    artifactRules = "coverage-reports/** => coverage-reports"
    useAgent(CiModel.AgentClass.LINUX_SMALL)
    dependencies {
        testMatrices.forEach { testMatrix ->
            snapshot(testMatrix) {
                onDependencyFailure = FailureAction.IGNORE
                onDependencyCancel = FailureAction.CANCEL
                synchronizeRevisions = true
                reuseBuilds = ReuseBuilds.SUCCESSFUL
            }
            artifacts(testMatrix) {
                buildRule = sameChain()
                artifactRules = "+:coverage-data/** => coverage-inputs/${testMatrix.id}"
                cleanDestination = true
            }
        }
    }
    steps {
        script {
            name = "Merge Python coverage"
            scriptContent = """
                set -eu
                mkdir -p coverage-reports
                find coverage-inputs -type f -name '.coverage*' -print0 > coverage-files
                if [ -s coverage-files ]; then
                  xargs -0 uv run --isolated --no-project --with coverage==7.11.2 coverage combine --keep < coverage-files
                  uv run --isolated --no-project --with coverage==7.11.2 coverage xml -o coverage-reports/coverage.xml
                else
                  echo 'No coverage data was produced by the test matrix.'
                fi
            """.trimIndent()
        }
    }
}

private fun qodanaGate(
    buildType: BuildType,
    buildId: String,
    buildName: String,
    coverageMerge: BuildType,
) = buildType.apply {
    id(buildId)
    name = buildName
    params {
        password(
            "opaque.qodana.cloud.token",
            "credentialsJSON:bab0ea16-4051-43f8-aab7-8c15d0c75e21",
            display = ParameterDisplay.HIDDEN,
        )
    }
    vcs {
        root(DslContext.settingsRoot)
    }
    useAgent(CiModel.AgentClass.LINUX_LARGE)
    dependencies {
        snapshot(coverageMerge) {
            onDependencyFailure = FailureAction.IGNORE
            onDependencyCancel = FailureAction.CANCEL
            synchronizeRevisions = true
            reuseBuilds = ReuseBuilds.SUCCESSFUL
        }
        artifacts(coverageMerge) {
            buildRule = sameChain()
            artifactRules = "+:coverage-reports/** => coverage-reports"
            cleanDestination = true
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
            reportAsTests = false
            additionalQodanaArguments = "--baseline qodana.sarif.json --coverage-dir coverage-reports"
        }
    }
}

object PrCoverageMerge : BuildType({
    coverageMerge(
        this,
        buildId = "Opaque_CoverageMergePr",
        buildName = "Merge PR coverage",
        testMatrices = listOf(PythonTests),
    )
})

object CoverageMerge : BuildType({
    coverageMerge(
        this,
        buildId = "Opaque_CoverageMerge",
        buildName = "Merge coverage",
        testMatrices = listOf(PythonTests, PlatformTests),
    )
})

object QodanaPr : BuildType({
    qodanaGate(
        this,
        buildId = "Opaque_QodanaPr",
        buildName = "Qodana PR",
        coverageMerge = PrCoverageMerge,
    )
})

object Qodana : BuildType({
    qodanaGate(
        this,
        buildId = "Opaque_Qodana",
        buildName = "Qodana",
        coverageMerge = CoverageMerge,
    )
})

val qodanaBuildTypes = listOf(PrCoverageMerge, CoverageMerge, QodanaPr, Qodana)