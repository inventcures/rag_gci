// Android client for Palli Sahayak.
//
// The include list is explicit. A wildcard scan would pull in the Python tree at
// the repository root, which is not a Gradle project and must never be part of
// the build.
pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "palli-sahayak-android"

include(":app")
include(":core:api")
include(":core:safety")
include(":core:data")
