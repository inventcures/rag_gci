// The API contract module.
//
// This is a plain Kotlin library, not an Android one, and deliberately has no
// Android UI dependencies. The generated client is the boundary between the app
// and the FastAPI server, and keeping it in its own module means the contract can
// be regenerated and verified without the presentation layer participating.
//
// The client is generated from the server's OpenAPI schema rather than written by
// hand. A hand-written client drifts: the app compiles, the server renames a
// field, and answers arrive silently empty. That failure is invisible in review
// and fatal in the field.

plugins {
    alias(libs.plugins.kotlin.jvm)
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(17))
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

dependencies {
    api(libs.retrofit)
    api(libs.retrofit.converter.moshi)
    api(libs.okhttp)
    api(libs.moshi)
    implementation(libs.moshi.kotlin)
    implementation(libs.okhttp.logging)
    implementation(libs.kotlinx.coroutines.android)

    testImplementation(libs.junit)
    testImplementation(libs.truth)
    testImplementation(libs.mockwebserver)
    testImplementation(libs.kotlinx.coroutines.test)
}

// ---------------------------------------------------------------------------
// Contract verification
// ---------------------------------------------------------------------------

// The committed schema snapshot. Its location is here so a reviewer looking at
// this module can find both the client and the contract it is checked against.
val apiContractDir = layout.projectDirectory.dir("contract")
// The committed snapshot is the contract. If the server's schema has moved, this
// fails the build and names the field that changed, rather than letting the app
// ship a client that compiles against a schema nobody serves any more.
//
// The comparison itself lives in scripts/export_openapi_contract.py. It has to:
// the Gradle scripting classpath has no JSON parser available, and the script
// already owns schema export, so keeping both halves in one place means the
// snapshot and the check cannot diverge.

val verifyApiContract by tasks.registering(Exec::class) {
    group = "verification"
    description = "Fails when the server's OpenAPI schema differs from the committed contract."

    val script = rootProject.layout.projectDirectory
        .file("../scripts/export_openapi_contract.py").asFile
    val contract = apiContractDir.file("openapi.json")

    inputs.file(contract).withPathSensitivity(PathSensitivity.RELATIVE)
    inputs.file(script).withPathSensitivity(PathSensitivity.RELATIVE)
    outputs.upToDateWhen { false }

    commandLine("python3", script.absolutePath, "--check")
    // The script prints the drift itself and exits 1, so Gradle adds no detail.
    isIgnoreExitValue = false
}

tasks.named("check") {
    dependsOn(verifyApiContract)
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
