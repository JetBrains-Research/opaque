object CiModel {
    data class PackageShard(val label: String, val path: String)

    data class PythonDistribution(val label: String, val path: String)

    enum class DependencyResolution(val syncArguments: String, val isolated: Boolean) {
        PINNED("--locked", false),
        MINIMUM("--resolution lowest-direct --upgrade", true),
        MAXIMUM("--resolution highest --upgrade", true),
    }

    data class VerificationProfile(
        val id: String,
        val displayName: String,
        val pythonVersion: String,
        val agentClass: AgentClass,
        val dependencyResolution: DependencyResolution,
        val pytestMarker: String,
    )

    data class AccountingTarget(
        val id: String,
        val label: String,
        val agentClass: AgentClass,
        val rustTarget: String,
        val manylinux: String?,
        val manylinuxContainer: String?,
    )

    enum class BranchKind(val branchFilter: String) {
        PULL_REQUEST("+:pull/*"),
        MAIN("+:main"),
        RELEASE_TAG("+:v*"),
    }

    enum class AgentClass(
        val displayName: String,
        val image: String?,
        val osName: String,
        val architecture: String,
        val requiredCapability: Pair<String, String>? = null,
    ) {
        LINUX_LARGE("Linux Large", "Linux-Large", "Linux", "amd64"),
        LINUX_SMALL("Linux Small", "Linux-Small", "Linux", "amd64"),
        UBUNTU_22_LINUX_SMALL("Ubuntu 22.04 Linux Small", "Ubuntu-22.04-Small", "Linux", "amd64"),
        UBUNTU_22_LINUX_ARM64_LARGE("Ubuntu 22.04 Linux ARM64 Large", "Ubuntu-22.04-Large-Arm64", "Linux", "aarch64"),
        UBUNTU_22_LINUX_ARM64_SMALL("Ubuntu 22.04 Linux ARM64 Small", "Ubuntu-22.04-Small-Arm64", "Linux", "aarch64"),
        MACOS_ARM64("macOS ARM64", "macOS-14-Sonoma-Medium-Arm64", "Mac OS X", "aarch64"),
        CUDA("CUDA", null, "Linux", "amd64", "opaque.agent.cuda" to "true"),
    }

    enum class TestDevice(
        val displayName: String,
        val markerForPullRequest: String,
        val markerForMain: String,
        val agentClass: AgentClass,
        val xdistArguments: String,
    ) {
        CPU(
            "CPU",
            "not cuda and not mps and not slow",
            "not cuda and not mps",
            AgentClass.LINUX_LARGE,
            "-n auto --dist loadscope",
        ),
        MPS(
            "MPS",
            "not cuda and not slow",
            "not cuda",
            AgentClass.MACOS_ARM64,
            "",
        ),
        CUDA("CUDA", "cuda and not slow", "cuda", AgentClass.CUDA, "-rs"),
        ;

        fun markerFor(branchKind: BranchKind) = when (branchKind) {
            BranchKind.PULL_REQUEST -> markerForPullRequest
            BranchKind.MAIN, BranchKind.RELEASE_TAG -> markerForMain
        }
    }

    object Artifacts {
        const val DISTRIBUTION_DIRECTORY = "dist"
        const val STAGED_INPUT_DIRECTORY = "inputs"
        const val COMPLETE_WHEEL_COUNT = 13
        const val SDIST_COUNT = 1
        const val VALIDATED_BUNDLE = "validated-distributions.zip"
        const val IDENTITY_MANIFEST = "teamcity-artifact-identity.txt"
        const val VERSION_MANIFEST = "teamcity-version-manifest.txt"
    }

    object Status {
        const val PR_GATE = "PR Gate"
        const val MAIN_CI = "Main CI"
        const val RELEASE_CANDIDATE = "Release Candidate"
    }

    const val ARTIFACT_RETENTION_DAYS = 90
    const val PYTHON_311 = "3.11"
    const val PYTHON_312 = "3.12"
    const val TEST_TIMEOUT_MINUTES = 40
    const val BUILD_TIMEOUT_MINUTES = 30

    val testShards = listOf(
        PackageShard("accounting", "packages/opaque-accounting"),
        PackageShard("alignment", "packages/opaque-alignment"),
        PackageShard("auditing", "packages/opaque-auditing"),
        PackageShard("base", "packages/opaque-base"),
        PackageShard("dpftrl", "packages/opaque-dpftrl"),
        PackageShard("dpsgd", "packages/opaque-dpsgd"),
        PackageShard("engine", "packages/opaque-engine"),
        PackageShard("optimizers", "packages/opaque-optimizers"),
        PackageShard("patches", "packages/opaque-patches"),
        PackageShard("transformers", "packages/opaque-transformers"),
        PackageShard("repository", "tests"),
    )

    val pinnedVerification = VerificationProfile(
        id = "Tests",
        displayName = "Tests",
        pythonVersion = PYTHON_311,
        agentClass = AgentClass.LINUX_LARGE,
        dependencyResolution = DependencyResolution.PINNED,
        pytestMarker = TestDevice.CPU.markerForPullRequest,
    )

    val minimumVerification = VerificationProfile(
        id = "MinimumAmd64",
        displayName = "Minimum dependencies (amd64)",
        pythonVersion = PYTHON_311,
        agentClass = AgentClass.LINUX_LARGE,
        dependencyResolution = DependencyResolution.MINIMUM,
        pytestMarker = TestDevice.CPU.markerForMain,
    )

    val maximumVerification = VerificationProfile(
        id = "MaximumAarch64",
        displayName = "Maximum dependencies (aarch64)",
        pythonVersion = PYTHON_312,
        agentClass = AgentClass.UBUNTU_22_LINUX_ARM64_LARGE,
        dependencyResolution = DependencyResolution.MAXIMUM,
        pytestMarker = TestDevice.CPU.markerForMain,
    )

    val verificationProfiles = listOf(pinnedVerification, minimumVerification, maximumVerification)

    val pythonDistributions = listOf(
        PythonDistribution("opaque", "."),
        PythonDistribution("opaque-alignment", "packages/opaque-alignment"),
        PythonDistribution("opaque-auditing", "packages/opaque-auditing"),
        PythonDistribution("opaque-base", "packages/opaque-base"),
        PythonDistribution("opaque-dpftrl", "packages/opaque-dpftrl"),
        PythonDistribution("opaque-dpsgd", "packages/opaque-dpsgd"),
        PythonDistribution("opaque-engine", "packages/opaque-engine"),
        PythonDistribution("opaque-optimizers", "packages/opaque-optimizers"),
        PythonDistribution("opaque-patches", "packages/opaque-patches"),
        PythonDistribution("opaque-transformers", "packages/opaque-transformers"),
    )

    val accountingTargets = listOf(
        AccountingTarget(
            "LinuxX64",
            "Linux x86_64",
            AgentClass.UBUNTU_22_LINUX_SMALL,
            "x86_64-unknown-linux-gnu",
            "2_28",
            "quay.io/pypa/manylinux_2_28_x86_64:latest",
        ),
        AccountingTarget(
            "LinuxArm64",
            "Linux ARM64",
            AgentClass.UBUNTU_22_LINUX_ARM64_SMALL,
            "aarch64-unknown-linux-gnu",
            "2_28",
            "quay.io/pypa/manylinux_2_28_aarch64:latest",
        ),
        AccountingTarget(
            "MacArm64",
            "macOS ARM64",
            AgentClass.MACOS_ARM64,
            "aarch64-apple-darwin",
            null,
            null,
        ),
    )

    val versionedPyprojectPaths = listOf(
        "pyproject.toml",
        "packages/opaque-accounting/pyproject.toml",
        *pythonDistributions.filter { it.path != "." }.map { "${it.path}/pyproject.toml" }.toTypedArray(),
    )
}