import jetbrains.buildServer.configs.kotlin.BuildType
import jetbrains.buildServer.configs.kotlin.BuildTypeSettings
import jetbrains.buildServer.configs.kotlin.DslContext
import jetbrains.buildServer.configs.kotlin.FailureAction
import jetbrains.buildServer.configs.kotlin.ParameterDisplay
import jetbrains.buildServer.configs.kotlin.ReuseBuilds
import jetbrains.buildServer.configs.kotlin.buildFeatures.approval
import jetbrains.buildServer.configs.kotlin.buildSteps.script

private fun publicationBuild(
    buildType: BuildType,
    buildId: String,
    buildName: String,
    branchKind: CiModel.BranchKind,
    verificationBuilds: List<BuildType>,
    releaseTagRequired: Boolean = false,
) = buildType.apply {
    id(buildId)
    name = buildName
    templates(PublicationTemplate)
    vcs {
        root(DslContext.settingsRoot)
        this.branchFilter = branchKind.branchFilter
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
    useAgent(CiModel.AgentClass.LINUX_SMALL)
    dependencies {
        (listOf(ValidateDistributions) + verificationBuilds).forEach { dependency ->
            snapshot(dependency) {
                onDependencyFailure = FailureAction.FAIL_TO_START
                onDependencyCancel = FailureAction.CANCEL
                synchronizeRevisions = true
                reuseBuilds = ReuseBuilds.SUCCESSFUL
            }
        }
        artifacts(ValidateDistributions) {
            buildRule = sameChain()
            artifactRules = "+:${CiModel.Artifacts.VALIDATED_BUNDLE}!/** => ${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}"
            cleanDestination = true
        }
    }
    steps {
        if (releaseTagRequired) {
            script {
                name = "Verify stable release tag"
                scriptContent = "printf '%%s' '%teamcity.build.branch%' | grep -Eq '^v[0-9]+\\.[0-9]+\\.[0-9]+(\\.(post|alpha|beta|rc)[0-9]+)?${'$'}'"
            }
        }
        script {
            name = "Verify validated artifact identity"
            scriptContent = """
                set -euo pipefail
                test "${'$'}(sed -n 's/^commit=//p' ${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}/${CiModel.Artifacts.IDENTITY_MANIFEST})" = "%build.vcs.number%"
                test "${'$'}(find ${CiModel.Artifacts.DISTRIBUTION_DIRECTORY} -maxdepth 1 -name '*.whl' | wc -l | tr -d ' ')" = ${CiModel.Artifacts.COMPLETE_WHEEL_COUNT}
                test "${'$'}(find ${CiModel.Artifacts.DISTRIBUTION_DIRECTORY} -maxdepth 1 -name '*.tar.gz' | wc -l | tr -d ' ')" = ${CiModel.Artifacts.SDIST_COUNT}
            """.trimIndent()
        }
        script {
            name = "Publish validated distributions"
            scriptContent = "uv run --isolated --no-project --with twine==6.2.0 python .github/scripts/twine_upload.py upload --repository-url https://packages.jetbrains.team/pypi/p/fed/python/legacy/ ${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}/*.whl ${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}/*.tar.gz"
        }
    }
}

object PublishDevDistributions : BuildType({
    publicationBuild(
        this,
        buildId = "Opaque_PublishDevDistributions",
        buildName = "Publish dev distributions",
        branchKind = CiModel.BranchKind.MAIN,
        verificationBuilds = listOf(MainCi),
    )
    type = BuildTypeSettings.Type.DEPLOYMENT
    enablePersonalBuilds = false
    maxRunningBuilds = 1
})

object PublishReleaseDistributions : BuildType({
    publicationBuild(
        this,
        buildId = "Opaque_PublishReleaseDistributions",
        buildName = "Publish release distributions",
        branchKind = CiModel.BranchKind.RELEASE_TAG,
        verificationBuilds = listOf(ReleaseCandidate),
        releaseTagRequired = true,
    )
    type = BuildTypeSettings.Type.DEPLOYMENT
    enablePersonalBuilds = false
    maxRunningBuilds = 1
    features {
        approval {
            approvalRules = DslContext.getParameter("opaque.release.approvalRules")
            timeout = 60
            manualRunsApproved = false
        }
    }
})