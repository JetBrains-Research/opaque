import jetbrains.buildServer.configs.kotlin.*

version = "2026.1"

project {
    description = "Hybrid CI prototype; GitHub Actions remains required until parity is demonstrated"

    template(PythonTestTemplate)
    opaqueTestBuildTypes.forEach(::buildType)
    buildType(OpaqueTestsPr)
    buildType(OpaqueCudaTrustedPr)
    buildType(OpaqueTestsMain)
    opaqueDistributionBuildTypes.forEach(::buildType)
    buildType(PreviewDistributions)
    buildType(DevDistributions)
    buildType(ReleaseDistributions)
    buildType(PublishDevDistributions)
    buildType(PublishReleaseDistributions)
}
