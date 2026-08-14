import jetbrains.buildServer.configs.kotlin.BuildType
import jetbrains.buildServer.configs.kotlin.DslContext
import jetbrains.buildServer.configs.kotlin.FailureAction
import jetbrains.buildServer.configs.kotlin.ParameterDisplay
import jetbrains.buildServer.configs.kotlin.ReuseBuilds
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.triggers.vcs

private fun publicationBuild(
    buildType: BuildType,
    buildId: String,
    buildName: String,
    branchFilter: String,
    releaseTagRequired: Boolean = false,
) = buildType.apply {
    id(buildId)
    name = buildName
    vcs {
        root(DslContext.settingsRoot)
        this.branchFilter = branchFilter
    }
    params {
        text(
            "env.TWINE_USERNAME",
            "%opaque.packages.username%",
            display = ParameterDisplay.HIDDEN,
            readOnly = true,
        )
        password(
            "env.TWINE_PASSWORD",
            "%opaque.packages.token%",
            display = ParameterDisplay.HIDDEN,
            readOnly = true,
        )
    }
    requirements { equals("teamcity.agent.jvm.os.name", "Linux") }
    dependencies {
        listOf(ValidateDistributions, OpaqueTestsMain).forEach { dependency ->
            snapshot(dependency) {
                onDependencyFailure = FailureAction.FAIL_TO_START
                onDependencyCancel = FailureAction.CANCEL
                synchronizeRevisions = true
                reuseBuilds = ReuseBuilds.NO
            }
        }
        artifacts(ValidateDistributions) {
            artifactRules = "+:validated-distributions.zip!/** => dist"
            cleanDestination = true
        }
    }
    triggers { vcs { this.branchFilter = branchFilter } }
    steps {
        if (releaseTagRequired) {
            script {
                name = "Verify stable release tag"
                scriptContent = "[[ '%teamcity.build.branch%' =~ ^v[0-9]+\\.[0-9]+\\.[0-9]+(\\.(post|alpha|beta|rc)[0-9]+)?${'$'} ]]"
            }
        }
        script {
            name = "Verify validated artifact identity"
            scriptContent = """
                set -euo pipefail
                test "${'$'}(sed -n 's/^commit=//p' dist/teamcity-artifact-identity.txt)" = "%build.vcs.number%"
                test "${'$'}(find dist -maxdepth 1 -name '*.whl' | wc -l | tr -d ' ')" = 13
                test "${'$'}(find dist -maxdepth 1 -name '*.tar.gz' | wc -l | tr -d ' ')" = 1
            """.trimIndent()
        }
        script {
            name = "Publish validated distributions"
            scriptContent = "uv run --isolated --no-project --with twine==6.2.0 python .github/scripts/twine_upload.py upload --repository-url https://packages.jetbrains.team/pypi/p/fed/python/legacy/ dist/*.whl dist/*.tar.gz"
        }
    }
}

object PublishDevDistributions : BuildType({
    publicationBuild(
        this,
        buildId = "Opaque_PublishDevDistributions",
        buildName = "Publish dev distributions",
        branchFilter = "+:<default>",
    )
})

object PublishReleaseDistributions : BuildType({
    publicationBuild(
        this,
        buildId = "Opaque_PublishReleaseDistributions",
        buildName = "Publish release distributions",
        branchFilter = "+:v*",
        releaseTagRequired = true,
    )
})