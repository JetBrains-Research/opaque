object CiModel {
    data class PackageShard(val label: String, val path: String)

    enum class BranchKind(val branchFilter: String) {
        MAIN_AND_PULL_REQUEST("+:main\n+:pull/*"),
    }

    enum class AgentClass(
        val displayName: String,
        val image: String?,
        val osName: String,
        val architecture: String,
    ) {
        LINUX_LARGE("Linux Large", "Linux-Large", "Linux", "amd64"),
    }

    object Status {
        const val LINUX_TESTS = "TeamCity Linux tests"
    }

    const val ARTIFACT_RETENTION_DAYS = 90
    const val TEST_TIMEOUT_MINUTES = 40
    const val PULL_REQUEST_CPU_MARKER = "not cuda and not mps and not slow"
    const val MAIN_CPU_MARKER = "not cuda and not mps"

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
}
