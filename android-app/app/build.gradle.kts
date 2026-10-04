plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.compose.compiler)
}

android {
    namespace = "org.inventcures.pallisahayak"
    compileSdk = 35

    defaultConfig {
        applicationId = "org.inventcures.pallisahayak"
        minSdk = 26          // §3.1 device constraints
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        debug {
            applicationIdSuffix = ".debug"
        }
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        compose = true
    }

    testOptions {
        unitTests {
            // Robolectric needs the Android resources and manifest on the unit
            // test classpath.
            isIncludeAndroidResources = true
            isReturnDefaultValues = true
        }
    }

    // The APK budget is a field constraint, not a hard limit: 2G connectivity and
    // a 2 GB reference device. Raising the ceiling does not make a 2G download
    // viable, so the number is measured rather than enforced blindly.
    bundle {
        abi { enableSplit = true }
    }
}

dependencies {
    // :core:api comes transitively through :core:data, which is where the
    // repository and the generated client live.
    implementation(project(":core:data"))
    implementation(project(":core:safety"))

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.viewmodel.ktx)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.kotlinx.coroutines.android)

    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
    testImplementation(libs.robolectric)
    testImplementation(libs.androidx.test.core)
    testImplementation(libs.androidx.test.junit)
    testImplementation(libs.truth)

    // Compose UI tests. The BOM is repeated here because the test source set does not
    // inherit the main one, and without it ui-test-junit4 has no version to resolve
    // to and the import fails as "unresolved" rather than as a version problem.
    testImplementation(platform(libs.androidx.compose.bom))
    testImplementation(libs.androidx.compose.ui.test.junit4)
    debugImplementation(libs.androidx.compose.ui.test.manifest)
    testImplementation(libs.mockwebserver)
    testImplementation(libs.kotlinx.coroutines.test)
    // The test builds a real in-memory Room database rather than mocking it, so
    // Room has to be on this module's test classpath.
    testImplementation(libs.room.runtime)
    // The test builds its own Retrofit, and Moshi cannot read Kotlin default
    // parameter values without this factory.
    testImplementation(libs.moshi.kotlin)
}
