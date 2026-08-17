import jetbrains.buildServer.configs.kotlin.BuildType
import jetbrains.buildServer.configs.kotlin.BuildTypeSettings
import jetbrains.buildServer.configs.kotlin.DslContext
import jetbrains.buildServer.configs.kotlin.FailureAction
import jetbrains.buildServer.configs.kotlin.ReuseBuilds
import jetbrains.buildServer.configs.kotlin.buildFeatures.PullRequests
import jetbrains.buildServer.configs.kotlin.buildFeatures.pullRequests
import jetbrains.buildServer.configs.kotlin.triggers.gitHubChecks

object LinuxTests : BuildType({
    id("Opaque_LinuxTests")
    name = CiModel.Status.LINUX_TESTS
    type = BuildTypeSettings.Type.COMPOSITE
    vcs {
        root(DslContext.settingsRoot)
        branchFilter = CiModel.BranchKind.MAIN_AND_PULL_REQUEST.branchFilter
    }
    dependencies {
        listOf(PythonTests, RustTests).forEach { dependency ->
            snapshot(dependency) {
                onDependencyFailure = FailureAction.FAIL_TO_START
                onDependencyCancel = FailureAction.CANCEL
                synchronizeRevisions = true
                reuseBuilds = ReuseBuilds.SUCCESSFUL
            }
        }
    }
    triggers {
        gitHubChecks { }
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