import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.projectFeatures.UntrustedBuildsSettings
import jetbrains.buildServer.configs.kotlin.projectFeatures.untrustedBuildsSettings

version = "2026.1"

object Verification : Project({
    id("Opaque_Verification")
    name = "Verification"

    template(PythonTestTemplate)
    verificationBuildTypes.forEach(::buildType)
})

project {
    description = "TeamCity runs Linux CPU Python shards and accounting Rust tests; GitHub Actions owns all other CI/CD work."

    features {
        untrustedBuildsSettings {
            id = "Opaque_UntrustedBuildApproval"
            defaultAction = UntrustedBuildsSettings.DefaultAction.APPROVE
            approvalRules = DslContext.getParameter("opaque.untrustedBuilds.approvalRules")
            enableLog = true
            timeoutMinutes = 60
        }
    }

    subProject(Verification)

    buildType(LinuxTests)
}
