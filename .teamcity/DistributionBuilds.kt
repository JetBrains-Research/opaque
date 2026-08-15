import jetbrains.buildServer.configs.kotlin.BuildType
import jetbrains.buildServer.configs.kotlin.BuildTypeSettings
import jetbrains.buildServer.configs.kotlin.DslContext
import jetbrains.buildServer.configs.kotlin.FailureAction
import jetbrains.buildServer.configs.kotlin.ReuseBuilds
import jetbrains.buildServer.configs.kotlin.buildSteps.ScriptBuildStep
import jetbrains.buildServer.configs.kotlin.buildSteps.script

object PrepareVersion : BuildType({
    id("Opaque_PrepareVersion")
    name = "Prepare immutable version"

    buildNumberPattern = "%opaque.version%"
    params {
        param("opaque.version", "pending")
        param("opaque.source.build.id", "%teamcity.build.id%")
    }
    outputParams {
        param("opaque.version", "%opaque.version%")
        param("opaque.source.build.id", "%opaque.source.build.id%")
    }
    failureConditions {
        executionTimeoutMin = CiModel.BUILD_TIMEOUT_MINUTES
    }
    configureCheckout()
    configureCleanup()
    configureAgentDiagnostics()
    artifactRules = "${CiModel.Artifacts.VERSION_MANIFEST} => metadata"
    useAgent(CiModel.AgentClass.LINUX_SMALL)
    steps {
        script {
            name = "Compute immutable version"
            scriptContent = """
                ${prepareVersionScript()}
                printf 'commit=%%s\nversion=%%s\nsource_build_id=%%s\ntarget=source\n' '%build.vcs.number%' "${'$'}VERSION" '%teamcity.build.id%' > ${CiModel.Artifacts.VERSION_MANIFEST}
            """.trimIndent()
        }
    }
})

private fun distributionBuild(
    buildType: BuildType,
    buildId: String,
    buildName: String,
    target: String,
) = buildType.apply {
    id(buildId)
    name = buildName
    buildNumberPattern = "%opaque.version%"
    templates(DistributionBuildTemplate)
    params {
        param("opaque.version", "${PrepareVersion.depParamRefs["opaque.version"]}")
        param("opaque.source.build.id", "${PrepareVersion.depParamRefs["opaque.source.build.id"]}")
        param("opaque.artifact.target", target)
    }
    dependencies {
        snapshot(PrepareVersion) {
            onDependencyFailure = FailureAction.FAIL_TO_START
            onDependencyCancel = FailureAction.CANCEL
            synchronizeRevisions = true
            reuseBuilds = ReuseBuilds.SUCCESSFUL
        }
    }
}

private fun BuildType.recordArtifactManifest() {
    steps {
        script {
            name = "Record artifact manifest"
            scriptContent = "printf 'commit=%%s\nversion=%%s\nsource_build_id=%%s\ntarget=%%s\n' '%build.vcs.number%' '%opaque.version%' '%opaque.source.build.id%' '%opaque.artifact.target%' > ${CiModel.Artifacts.VERSION_MANIFEST}"
        }
    }
    pruneUvCacheForCi()
}

private fun BuildType.validateInternalPins() {
    steps {
        script {
            name = "Validate internal pins"
            scriptContent = "uv run --isolated --no-project --with packaging==25.0 python .github/scripts/check_wheel_internal_pins.py --wheel-dir ${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}"
        }
    }
}

private fun BuildType.restoreVersionedFiles() {
    steps {
        script {
            name = "Restore source metadata"
            scriptContent = """
                git restore --source=HEAD -- Cargo.toml ${CiModel.versionedPyprojectPaths.joinToString(" ")}
            """.trimIndent()
        }
    }
}

private fun BuildType.smokeAccountingWheel() {
    steps {
        script {
            name = "Smoke installed native wheel"
            scriptContent = """
                set -eu
                WHEEL="${'$'}(find %teamcity.build.checkoutDir%/${CiModel.Artifacts.DISTRIBUTION_DIRECTORY} -maxdepth 1 -name 'opaque_accounting-*.whl' -print -quit)"
                test -n "${'$'}WHEEL"
                SMOKE_DIR="${'$'}(mktemp -d)"
                trap 'rm -rf "${'$'}SMOKE_DIR"' EXIT
                uv venv --python python3.11 "${'$'}SMOKE_DIR/venv"
                uv pip install --no-deps --python "${'$'}SMOKE_DIR/venv/bin/python" packages/opaque-base
                uv pip install --no-deps --python "${'$'}SMOKE_DIR/venv/bin/python" "${'$'}WHEEL"
                cd "${'$'}SMOKE_DIR"
                "${'$'}SMOKE_DIR/venv/bin/python" - <<'PY'
                import opaque.accounting as acc
                from opaque.api.accounting.core import opaque_accounting as native

                assert native is not None
                assert acc.eps_delta(1.0, 1e-5).pld() is not None
                PY
            """.trimIndent()
        }
    }
}

private fun BuildType.bootstrapRustToolchain() {
    steps {
        script {
            name = "Bootstrap Rust"
            scriptContent = """
                set -eu
                if ! command -v cargo >/dev/null 2>&1 && [ -x "${'$'}HOME/.cargo/bin/cargo" ]; then
                  export PATH="${'$'}HOME/.cargo/bin:${'$'}PATH"
                fi
                if ! command -v cargo >/dev/null 2>&1; then
                  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | env CARGO_HOME="${'$'}HOME/.cargo" RUSTUP_HOME="${'$'}HOME/.rustup" sh -s -- -y --profile minimal --default-toolchain stable
                  export PATH="${'$'}HOME/.cargo/bin:${'$'}PATH"
                fi
                echo "##teamcity[setParameter name='env.PATH' value='${'$'}PATH']"
                cargo --version
                rustc --version
            """.trimIndent()
        }
    }
}

private fun CiModel.AccountingTarget.nativeWheelBuildScript(): String {
    val maturinArguments = "--manifest-path packages/opaque-accounting/Cargo.toml --release --target $rustTarget --interpreter python3.11 --out ${CiModel.Artifacts.DISTRIBUTION_DIRECTORY} --compatibility pypi"
    if (manylinuxContainer == null) {
        return "uv run --isolated --no-project --with maturin maturin build $maturinArguments"
    }
    return """
        set -eu
        curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="${'$'}HOME/.local/bin:${'$'}CARGO_HOME/bin:/opt/python/cp311-cp311/bin:${'$'}PATH"
        uv run --isolated --no-project --with maturin maturin build $maturinArguments --manylinux $manylinux
    """.trimIndent()
}

private fun pythonWheel(distribution: CiModel.PythonDistribution) = BuildType {
    val buildSuffix = distribution.label.split("-").joinToString("") { part ->
        part.replaceFirstChar(Char::uppercaseChar)
    }
    distributionBuild(
        this,
        "Opaque_BuildPython$buildSuffix",
        "Build ${distribution.label}",
        "python-${distribution.label}",
    )
    artifactRules = "${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}/*.whl => distributions\n${CiModel.Artifacts.VERSION_MANIFEST} => metadata"
    useAgent(CiModel.AgentClass.LINUX_SMALL)
    steps {
        script {
            name = "Build wheel"
            scriptContent = """
                bash .github/scripts/set_build_versions.sh '%opaque.version%'
                export SETUPTOOLS_SCM_PRETEND_VERSION='%opaque.version%'
                (cd ${distribution.path} && uv build --wheel --out-dir %teamcity.build.checkoutDir%/${CiModel.Artifacts.DISTRIBUTION_DIRECTORY})
            """.trimIndent()
        }
    }
    validateInternalPins()
    restoreVersionedFiles()
    recordArtifactManifest()
}

private fun accountingWheel(target: CiModel.AccountingTarget) = BuildType {
    distributionBuild(this, "Opaque_BuildAccounting${target.id}", "Build opaque-accounting (${target.label})", "accounting-${target.id}")
    artifactRules = "${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}/*.whl => distributions\n${CiModel.Artifacts.VERSION_MANIFEST} => metadata"
    useAgent(target.agentClass)
    if (target.manylinuxContainer == null) {
        bootstrapRustToolchain()
    }
    steps {
        script {
            name = "Build native wheel"
            scriptContent = """
                bash .github/scripts/set_build_versions.sh '%opaque.version%'
                ${target.nativeWheelBuildScript()}
            """.trimIndent()
            target.manylinuxContainer?.let {
                dockerImage = it
                dockerImagePlatform = ScriptBuildStep.ImagePlatform.Linux
            }
        }
    }
    smokeAccountingWheel()
    validateInternalPins()
    restoreVersionedFiles()
    recordArtifactManifest()
}

object AccountingSdist : BuildType({
    distributionBuild(this, "Opaque_BuildAccountingSdist", "Build opaque-accounting sdist", "accounting-sdist")
    artifactRules = "${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}/*.tar.gz => distributions\n${CiModel.Artifacts.VERSION_MANIFEST} => metadata"
    useAgent(CiModel.AgentClass.LINUX_SMALL)
    steps {
        script {
            name = "Build accounting sdist"
            scriptContent = "cd packages/opaque-accounting && uv build --sdist --out-dir %teamcity.build.checkoutDir%/${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}"
        }
    }
    recordArtifactManifest()
})

private data class DistributionBuilder(val target: String, val buildType: BuildType)

private val pythonWheelBuilds = CiModel.pythonDistributions.map { distribution ->
    DistributionBuilder("python-${distribution.label}", pythonWheel(distribution))
}
private val accountingWheelBuilds = CiModel.accountingTargets.map { target ->
    DistributionBuilder("accounting-${target.id}", accountingWheel(target))
}
private val distributionBuilders = pythonWheelBuilds +
    accountingWheelBuilds +
    DistributionBuilder("accounting-sdist", AccountingSdist)

object ValidateDistributions : BuildType({
    id("Opaque_ValidateDistributions")
    name = "Validate exact distribution bundle"
    buildNumberPattern = "%opaque.version%"
    configurePythonTooling()
    artifactRules = "${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}/** => ${CiModel.Artifacts.VALIDATED_BUNDLE}"
    useAgent(CiModel.AgentClass.LINUX_SMALL)
    params {
        param("opaque.version", "${PrepareVersion.depParamRefs["opaque.version"]}")
        param("opaque.source.build.id", "${PrepareVersion.depParamRefs["opaque.source.build.id"]}")
    }
    dependencies {
        snapshot(PrepareVersion) {
            onDependencyFailure = FailureAction.FAIL_TO_START
            onDependencyCancel = FailureAction.CANCEL
            synchronizeRevisions = true
            reuseBuilds = ReuseBuilds.SUCCESSFUL
        }
        distributionBuilders.forEach { builder ->
            snapshot(builder.buildType) {
                onDependencyFailure = FailureAction.FAIL_TO_START
                onDependencyCancel = FailureAction.CANCEL
                synchronizeRevisions = true
                reuseBuilds = ReuseBuilds.SUCCESSFUL
            }
            artifacts(builder.buildType) {
                buildRule = sameChain()
                artifactRules = """
                    +:distributions/* => ${CiModel.Artifacts.STAGED_INPUT_DIRECTORY}/${builder.target}/artifacts
                    +:metadata/${CiModel.Artifacts.VERSION_MANIFEST} => ${CiModel.Artifacts.STAGED_INPUT_DIRECTORY}/${builder.target}/metadata
                """.trimIndent()
            }
        }
    }
    steps {
        script {
            name = "Stage target-qualified artifacts"
            scriptContent = """
                set -eu
                mkdir -p ${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}
                find ${CiModel.Artifacts.STAGED_INPUT_DIRECTORY} -type f \( -name '*.whl' -o -name '*.tar.gz' \) -exec cp {} ${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}/ \;
            """.trimIndent()
        }
        script {
            name = "Verify source manifests"
            scriptContent = """
                set -eu
                for manifest in ${CiModel.Artifacts.STAGED_INPUT_DIRECTORY}/*/metadata/${CiModel.Artifacts.VERSION_MANIFEST}; do
                  test "${'$'}(sed -n 's/^commit=//p' "${'$'}manifest")" = "%build.vcs.number%"
                  test "${'$'}(sed -n 's/^version=//p' "${'$'}manifest")" = "%opaque.version%"
                  test "${'$'}(sed -n 's/^source_build_id=//p' "${'$'}manifest")" = "%opaque.source.build.id%"
                done
            """.trimIndent()
        }
        script {
            name = "Validate accounting artifact policy"
            scriptContent = "python3 .github/scripts/check_accounting_artifact_policy.py --wheel-dir ${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}"
        }
        script {
            name = "Verify complete artifact set"
            scriptContent = "test \"${'$'}(find ${CiModel.Artifacts.DISTRIBUTION_DIRECTORY} -maxdepth 1 -name '*.whl' | wc -l | tr -d ' ')\" = ${CiModel.Artifacts.COMPLETE_WHEEL_COUNT} && test \"${'$'}(find ${CiModel.Artifacts.DISTRIBUTION_DIRECTORY} -maxdepth 1 -name '*.tar.gz' | wc -l | tr -d ' ')\" = ${CiModel.Artifacts.SDIST_COUNT}"
        }
        script {
            name = "Install and import complete bundle"
            scriptContent = """
                set -eu
                bundle_venv=%teamcity.build.tempDir%/opaque-bundle
                rm -rf "${'$'}bundle_venv"
                uv venv --python ${CiModel.PYTHON_311} "${'$'}bundle_venv"
                uv pip install --python "${'$'}bundle_venv/bin/python" --find-links %teamcity.build.checkoutDir%/${CiModel.Artifacts.DISTRIBUTION_DIRECTORY} --prerelease=allow "opaque[all]==%opaque.version%"
                (
                  cd "${'$'}bundle_venv"
                  PYTHONNOUSERSITE=1 "${'$'}bundle_venv/bin/python" -c 'import importlib; [importlib.import_module(module) for module in ("opaque.accounting", "opaque.alignment", "opaque.auditing", "opaque.distributed", "opaque.dpftrl", "opaque.dpsgd", "opaque.functional", "opaque.optimizers", "opaque.patches", "opaque.precision", "opaque.profiling", "opaque.pytree", "opaque.random", "opaque.scheduling", "opaque.serialization", "opaque.transformers", "opaque.types")]'
                )
            """.trimIndent()
        }
        script {
            name = "Record artifact identity"
            scriptContent = "printf 'commit=%%s\\nversion=%%s\\nsource_build_id=%%s\\nbuild_id=%%s\\n' '%build.vcs.number%' '%opaque.version%' '%opaque.source.build.id%' '%teamcity.build.id%' > ${CiModel.Artifacts.DISTRIBUTION_DIRECTORY}/${CiModel.Artifacts.IDENTITY_MANIFEST}"
        }
    }
    pruneUvCacheForCi()
})

private fun distributionChain(
    buildType: BuildType,
    buildId: String,
    buildName: String,
    branchKind: CiModel.BranchKind,
) = buildType.apply {
    id(buildId)
    name = buildName
    type = BuildTypeSettings.Type.COMPOSITE
    vcs {
        root(DslContext.settingsRoot)
        this.branchFilter = branchKind.branchFilter
    }
    dependencies {
        snapshot(ValidateDistributions) {
            onDependencyFailure = FailureAction.FAIL_TO_START
            onDependencyCancel = FailureAction.CANCEL
            synchronizeRevisions = true
            reuseBuilds = ReuseBuilds.SUCCESSFUL
        }
    }
}

object PreviewDistributions : BuildType({
    distributionChain(this, "Opaque_PreviewDistributions", "Preview artifacts", CiModel.BranchKind.PULL_REQUEST)
})

object DevDistributions : BuildType({
    distributionChain(this, "Opaque_DevDistributions", "Dev artifacts", CiModel.BranchKind.MAIN)
})

object ReleaseDistributions : BuildType({
    distributionChain(this, "Opaque_ReleaseDistributions", "Release artifacts", CiModel.BranchKind.RELEASE_TAG)
})

val artifactBuildTypes = listOf(PrepareVersion) +
    distributionBuilders.map { it.buildType } +
    listOf(ValidateDistributions, PreviewDistributions, DevDistributions, ReleaseDistributions)