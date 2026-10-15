// Local persistence and the repository implementation.
//
// Depends on :core:api for the generated client and :core:safety for the
// invariants, so the data layer cannot apply a response the safety layer has not
// cleared. Nothing here decides anything: it stores what the safety layer produced
// and reports it back unchanged.
plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.ksp)
}

// The server the app talks to and the study site it registers participants at.
// Passed per build (`./gradlew :app:assembleRelease -PpalliBaseUrl=https://... -PpalliSiteId=...`)
// so a study deployment points at its own server without a code change.
//
// Debug defaults to the host machine as the Android emulator sees it. The release
// default is the production host named in the Android specification; a release
// build refuses to configure with anything that is not https, or with the old
// `.invalid` placeholder, so it cannot ship pointing nowhere.
val debugBaseUrl: String = providers.gradleProperty("palliDebugBaseUrl")
    .getOrElse("http://10.0.2.2:8000/")
val releaseBaseUrl: String = providers.gradleProperty("palliBaseUrl")
    .getOrElse("https://api.pallisahayak.org/")
val studySiteId: String = providers.gradleProperty("palliSiteId").getOrElse("unassigned")

check(releaseBaseUrl.startsWith("https://") && !releaseBaseUrl.contains(".invalid")) {
    "palliBaseUrl must be an https URL of a real server, was '$releaseBaseUrl'"
}

fun quoted(value: String) = "\"" + value.replace("\"", "\\\"") + "\""

android {
    namespace = "org.inventcures.pallisahayak.data"
    compileSdk = 35

    defaultConfig {
        minSdk = 26
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        buildConfigField("String", "SITE_ID", quoted(studySiteId))
    }

    buildFeatures {
        buildConfig = true
    }

    buildTypes {
        debug {
            buildConfigField("String", "BASE_URL", quoted(debugBaseUrl))
        }
        release {
            buildConfigField("String", "BASE_URL", quoted(releaseBaseUrl))
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    testOptions {
        unitTests {
            // Plain JVM tests reach android.util.Log on the offline and sync
            // failure paths; let the stub return instead of throwing.
            isReturnDefaultValues = true
        }
    }

    sourceSets {
        getByName("androidTest").assets.srcDir("$projectDir/schemas")
    }
}

ksp {
    arg("room.schemaLocation", "$projectDir/schemas")
}

dependencies {
    api(project(":core:api"))
    api(project(":core:safety"))

    implementation(libs.androidx.core.ktx)
    implementation(libs.kotlinx.coroutines.android)
    // Retrofit and OkHttp are declared as `implementation` in :core:api, so they
    // are not on this module's compile classpath. Declared here rather than
    // widening the api dependency, which would leak them to every consumer.
    implementation(libs.retrofit)
    implementation(libs.retrofit.converter.moshi)
    implementation(libs.okhttp)
    implementation(libs.okhttp.logging)
    // Required for KotlinJsonAdapterFactory, without which Moshi cannot read
    // Kotlin default parameter values in the generated data classes.
    implementation(libs.moshi.kotlin)
    // ViewModel and viewModelScope. lifecycle-runtime alone does not bring them,
    // so the ViewModel would not compile without this.
    implementation(libs.androidx.lifecycle.viewmodel.ktx)

    implementation(libs.room.runtime)
    implementation(libs.room.ktx)
    ksp(libs.room.compiler)

    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
    testImplementation(libs.robolectric)
    testImplementation(libs.androidx.test.core)
    testImplementation(libs.truth)
    testImplementation(libs.mockwebserver)
    testImplementation(libs.kotlinx.coroutines.test)
}
