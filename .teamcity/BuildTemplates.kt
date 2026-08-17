import jetbrains.buildServer.configs.kotlin.BuildTypeSettings
import jetbrains.buildServer.configs.kotlin.DslContext
import jetbrains.buildServer.configs.kotlin.Template
import jetbrains.buildServer.configs.kotlin.buildFeatures.XmlReport
import jetbrains.buildServer.configs.kotlin.buildFeatures.buildCache
import jetbrains.buildServer.configs.kotlin.buildFeatures.perfmon
import jetbrains.buildServer.configs.kotlin.buildFeatures.xmlReport
import jetbrains.buildServer.configs.kotlin.buildSteps.script

fun BuildTypeSettings.useAgent(agentClass: CiModel.AgentClass) {
    requirements {
        agentClass.image?.let { startsWith("teamcity.agent.name", it) }
        equals("teamcity.agent.jvm.os.name", agentClass.osName)
        equals("teamcity.agent.jvm.os.arch", agentClass.architecture)
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
        param("opaque.cache.scope", "branch-%teamcity.build.branch%")
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

private fun ensureUvScript() = """
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

private fun BuildTypeSettings.configurePythonTooling() {
    configureCheckout()
    configureCleanup()
    configureHostedCachePolicy()
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
