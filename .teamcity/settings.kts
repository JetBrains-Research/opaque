import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.projectFeatures.UntrustedBuildsSettings
import jetbrains.buildServer.configs.kotlin.projectFeatures.untrustedBuildsSettings

version = "2026.1"

object Verification : Project({
    id("Opaque_Verification")
    name = "Verification"

    template(PythonTestTemplate)
    template(PythonUtilityTemplate)
    verificationBuildTypes.forEach(::buildType)
    qodanaBuildTypes.forEach(::buildType)
})

object Artifacts : Project({
    id("Opaque_Artifacts")
    name = "Artifacts"

    template(DistributionBuildTemplate)
    artifactBuildTypes.forEach(::buildType)
})

object Delivery : Project({
    id("Opaque_Delivery")
    name = "Delivery"

    params {
        text(
            "opaque.packages.username",
            "37ed2d18-f9be-4e61-9ca1-288f4bde66f8",
            display = ParameterDisplay.HIDDEN,
            readOnly = true,
        )
        password(
            "opaque.packages.token",
            "credentialsJSON:5222e228-5d60-492b-9ddf-889a02e0805e",
            display = ParameterDisplay.HIDDEN,
            readOnly = true,
        )
    }

    template(PublicationTemplate)
    buildType(PublishDevDistributions)
    buildType(PublishReleaseDistributions)
})

project {
    description = "Hybrid CI prototype; GitHub Actions remains required until parity is demonstrated"

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
    subProject(Artifacts)
    subProject(Delivery)

    buildType(PrGate)
    buildType(MainCi)
    buildType(ReleaseCandidate)
}
