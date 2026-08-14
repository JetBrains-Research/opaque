import jetbrains.buildServer.configs.kotlin.*

version = "2026.1"

object Verification : Project({
    id("Opaque_Verification")
    name = "Verification"

    template(PythonTestTemplate)
    template(PythonUtilityTemplate)
    verificationBuildTypes.forEach(::buildType)
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

    template(PublicationTemplate)
    deliveryBuildTypes.forEach(::buildType)
})

project {
    description = "Hybrid CI prototype; GitHub Actions remains required until parity is demonstrated"

    subProject(Verification)
    subProject(Artifacts)
    subProject(Delivery)

    buildType(PrGate)
    buildType(OpaqueCudaTrustedPr)
    buildType(MainCi)
    buildType(ReleaseCandidate)
    buildType(PublishDevDistributions)
    buildType(PublishReleaseDistributions)
}
