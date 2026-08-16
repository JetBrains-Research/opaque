import jetbrains.buildServer.configs.kotlin.BuildTypeSettings
import jetbrains.buildServer.configs.kotlin.DslContext
import jetbrains.buildServer.configs.kotlin.Template
import jetbrains.buildServer.configs.kotlin.buildFeatures.buildCache
import jetbrains.buildServer.configs.kotlin.buildFeatures.XmlReport
import jetbrains.buildServer.configs.kotlin.buildFeatures.perfmon
import jetbrains.buildServer.configs.kotlin.buildFeatures.xmlReport
import jetbrains.buildServer.configs.kotlin.buildSteps.script

fun BuildTypeSettings.useAgent(agentClass: CiModel.AgentClass) {
    requirements {
        agentClass.image?.let { startsWith("teamcity.agent.name", it) }
        equals("teamcity.agent.jvm.os.name", agentClass.osName)
        equals("teamcity.agent.jvm.os.arch", agentClass.architecture)
        agentClass.requiredCapability?.let { (key, value) -> equals(key, value) }
    }
}

fun BuildTypeSettings.configureCheckout() {
    vcs {
        root(DslContext.settingsRoot)
        cleanCheckout = true
        showDependenciesChanges = true
    }
}

fun BuildTypeSettings.configureCleanup() {
    cleanup {
        baseRule {
            history(days = CiModel.ARTIFACT_RETENTION_DAYS)
            artifacts(days = CiModel.ARTIFACT_RETENTION_DAYS)
            preventDependencyCleanup = true
        }
    }
}

fun BuildTypeSettings.configureHostedCachePolicy() {
    params {
        param("opaque.cache.scope", "trusted")
        param("env.UV_CACHE_DIR", "%teamcity.build.checkoutDir%/.teamcity-cache/uv")
        param("env.CARGO_HOME", "%teamcity.build.checkoutDir%/.teamcity-cache/cargo")
    }
    features {
        buildCache {
            name = "opaque-uv-%opaque.cache.scope%-%teamcity.agent.jvm.os.name%-%teamcity.agent.jvm.os.arch%"
            rules = ".teamcity-cache/uv"
            use = true
            publish = true
            publishOnlyChanged = true
        }
        buildCache {
            name = "opaque-cargo-%opaque.cache.scope%-%teamcity.agent.jvm.os.name%-%teamcity.agent.jvm.os.arch%"
            rules = ".teamcity-cache/cargo"
            use = true
            publish = true
            publishOnlyChanged = true
        }
    }
}

fun ensureUvScript() = """
    set -eu
    if ! command -v uv >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/install.sh | sh
      export PATH="${'$'}HOME/.local/bin:${'$'}PATH"
      echo "##teamcity[setParameter name='env.PATH' value='${'$'}PATH']"
    fi
    uv --version
""".trimIndent()

fun BuildTypeSettings.configureAgentDiagnostics() {
    steps {
        script {
            name = "Report agent environment"
            scriptContent = """
                printf 'agent=%%s\nos=%%s\narchitecture=%%s\nmemory_mb=%%s\n' \
                  '%teamcity.agent.name%' \
                  '%teamcity.agent.jvm.os.name%' \
                  '%teamcity.agent.jvm.os.arch%' \
                  '%teamcity.agent.hardware.memorySizeMb%'
                uname -a
            """.trimIndent()
        }
    }
}

fun BuildTypeSettings.configurePythonTooling(useHostedCache: Boolean = true) {
    configureCheckout()
    configureCleanup()
    if (useHostedCache) {
        configureHostedCachePolicy()
    }
    steps {
        script {
            name = "Bootstrap uv"
            scriptContent = ensureUvScript()
        }
    }
    configureAgentDiagnostics()
}

fun BuildTypeSettings.configurePythonTestReporting() {
    features {
        perfmon { }
        xmlReport {
            reportType = XmlReport.XmlReportType.JUNIT
            rules = "test-results.xml"
        }
    }
}

fun BuildTypeSettings.pruneUvCacheForCi() {
    steps {
        script {
            name = "Prune uv cache for CI"
            scriptContent = "uv cache prune --ci"
        }
    }
}

fun prepareVersionScript(): String {
    return """
        set -eu
        BRANCH='%teamcity.build.branch%'
        if printf '%%s' "${'$'}BRANCH" | grep -Eq '^v[0-9]+\.[0-9]+\.[0-9]+(\.(post|alpha|beta|rc)[0-9]+)?${'$'}'; then
          VERSION="${'$'}{BRANCH#v}"
        else
          LAST="${'$'}(git tag --list 'v[0-9]*' --sort=-v:refname | sed -n '1p')"
          if [ -z "${'$'}LAST" ]; then
            BASE=0.0.1
            DISTANCE=0
          else
            BASE="${'$'}(printf '%%s' "${'$'}{LAST#v}" | awk -F. '{printf "%%s.%%s.%%d", ${'$'}1, ${'$'}2, ${'$'}3 + 1}')"
            DISTANCE="${'$'}(git rev-list --count "${'$'}LAST..HEAD")"
          fi
          SHA="${'$'}(git rev-parse --short HEAD)"
          case "${'$'}BRANCH" in
            pull/[0-9]*) VERSION="${'$'}BASE.dev${'$'}DISTANCE+pr.${'$'}{BRANCH#pull/}.g${'$'}SHA" ;;
            *) VERSION="${'$'}BASE.dev${'$'}DISTANCE+g${'$'}SHA" ;;
          esac
        fi
        echo "##teamcity[setParameter name='env.SETUPTOOLS_SCM_PRETEND_VERSION' value='${'$'}VERSION']"
        echo "##teamcity[setParameter name='opaque.version' value='${'$'}VERSION']"
        echo "##teamcity[buildNumber '${'$'}VERSION']"
    """.trimIndent()
}

fun applyPreparedVersionScript(): String {
    val files = CiModel.versionedPyprojectPaths.joinToString(" ") { "'$it'" }
    return """
        set -eu
        VERSION='%opaque.version%'
        CARGO_VERSION="${'$'}(printf '%%s' "${'$'}VERSION" | sed -E -e 's/\.(alpha|beta|rc|post)([0-9]+)\.(dev)([0-9]+)/-\1.\2.\3.\4/' -e 's/\.(alpha|beta|rc)([0-9]+)/-\1.\2/' -e 's/\.post([0-9]+)\+([0-9A-Za-z.-]+)/+post.\1.\2/' -e 's/\.post([0-9]+)/+post.\1/' -e 's/\.dev([0-9]+)/-dev.\1/')"
        perl -pi -e "s/^version = \"[^\"]+\"/version = \"${'$'}VERSION\"/" packages/opaque-accounting/pyproject.toml
        perl -pi -e "s/^version = \"[^\"]+\"/version = \"${'$'}CARGO_VERSION\"/" Cargo.toml
        perl -pi -e "s/>=0\.0\.0\.dev0/==${'$'}VERSION/g" $files
    """.trimIndent()
}

object PythonTestTemplate : Template({
    id("Opaque_PythonTestTemplate")
    name = "Python test template"

    artifactRules = "coverage.xml => coverage\n.coverage => coverage-data"
    failureConditions {
        executionTimeoutMin = CiModel.TEST_TIMEOUT_MINUTES
    }
    configurePythonTooling()
    configurePythonTestReporting()
})

object PythonUtilityTemplate : Template({
    id("Opaque_PythonUtilityTemplate")
    name = "Python utility template"

    failureConditions {
        executionTimeoutMin = CiModel.BUILD_TIMEOUT_MINUTES
    }
    configurePythonTooling()
})

object DistributionBuildTemplate : Template({
    id("Opaque_DistributionBuildTemplate")
    name = "Distribution build template"

    buildNumberPattern = "%opaque.version%"
    params {
        param("opaque.version", "pending")
    }
    outputParams {
        param("opaque.version", "%opaque.version%")
    }
    failureConditions {
        executionTimeoutMin = CiModel.BUILD_TIMEOUT_MINUTES
    }
    configurePythonTooling()
    steps {
        script {
            name = "Apply prepared distribution version"
            scriptContent = applyPreparedVersionScript()
        }
    }
})

object PublicationTemplate : Template({
    id("Opaque_PublicationTemplate")
    name = "Publication template"

    failureConditions {
        executionTimeoutMin = CiModel.BUILD_TIMEOUT_MINUTES
    }
    configurePythonTooling(useHostedCache = false)
})