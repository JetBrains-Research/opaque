import jetbrains.buildServer.configs.kotlin.BuildType
import jetbrains.buildServer.configs.kotlin.BuildTypeSettings
import jetbrains.buildServer.configs.kotlin.DslContext
import jetbrains.buildServer.configs.kotlin.FailureAction
import jetbrains.buildServer.configs.kotlin.ReuseBuilds
import jetbrains.buildServer.configs.kotlin.buildFeatures.PullRequests
import jetbrains.buildServer.configs.kotlin.buildFeatures.commitStatusPublisher
import jetbrains.buildServer.configs.kotlin.buildFeatures.pullRequests
import jetbrains.buildServer.configs.kotlin.triggers.vcs

private fun entryChain(
    buildType: BuildType,
    buildId: String,
    buildName: String,
    branchKind: CiModel.BranchKind,
    dependencies: List<BuildType>,
    nonBlockingDependencies: Set<BuildType> = emptySet(),
    cancelObsoletePullRequestBuilds: Boolean,
) = buildType.apply {
    id(buildId)
    name = buildName
    type = BuildTypeSettings.Type.COMPOSITE
    vcs {
        root(DslContext.settingsRoot)
        branchFilter = branchKind.branchFilter
    }
    dependencies {
        dependencies.forEach { dependency ->
            snapshot(dependency) {
                onDependencyFailure = if (dependency in nonBlockingDependencies) {
                    FailureAction.IGNORE
                } else {
                    FailureAction.FAIL_TO_START
                }
                onDependencyCancel = FailureAction.CANCEL
                synchronizeRevisions = true
                reuseBuilds = ReuseBuilds.SUCCESSFUL
            }
        }
    }
    triggers {
        vcs {
            branchFilter = branchKind.branchFilter
            enableQueueOptimization = cancelObsoletePullRequestBuilds
        }
    }
    features {
        commitStatusPublisher {
            publisher = github {
                githubUrl = "https://api.github.com"
                authType = vcsRoot()
            }
        }
    }
}

object PrGate : BuildType({
    entryChain(
        this,
        buildId = "Opaque_PrGate",
        buildName = CiModel.Status.PR_GATE,
        branchKind = CiModel.BranchKind.PULL_REQUEST,
        dependencies = listOf(PythonTests, QodanaPr, RustTests, StrictDocs, PreviewDistributions),
        nonBlockingDependencies = setOf(QodanaPr),
        cancelObsoletePullRequestBuilds = true,
    )
    params {
        param("override.dep.*.opaque.cache.scope", "pr-%teamcity.pullRequest.number%")
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
})

object MainCi : BuildType({
    entryChain(
        this,
        buildId = "Opaque_MainCi",
        buildName = CiModel.Status.MAIN_CI,
        branchKind = CiModel.BranchKind.MAIN,
        dependencies = listOf(PythonTests, Qodana, RustTests, StrictDocs, DependencyVersionVerification, PlatformVerification, DevDistributions),
        nonBlockingDependencies = setOf(Qodana),
        cancelObsoletePullRequestBuilds = false,
    )
})

object ReleaseCandidate : BuildType({
    entryChain(
        this,
        buildId = "Opaque_ReleaseCandidate",
        buildName = CiModel.Status.RELEASE_CANDIDATE,
        branchKind = CiModel.BranchKind.RELEASE_TAG,
        dependencies = listOf(PythonTests, Qodana, RustTests, StrictDocs, DependencyVersionVerification, PlatformVerification, ReleaseDistributions),
        nonBlockingDependencies = setOf(Qodana),
        cancelObsoletePullRequestBuilds = false,
    )
})