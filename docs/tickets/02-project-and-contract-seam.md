# 02: Project and contract seam

**What to build:** The Android project builds, and the client's API contract cannot
silently drift from the server's. An app and a server that share a repository drift
more often than not, and the failure — a renamed field producing silently empty
answers — is invisible in review and fatal in the field.

**Blocked by:** None (can start immediately)

**Status:** complete — 5 of 5 acceptance criteria met

- [x] The Android project builds a debug APK on a clean checkout
- [x] The API client is generated from the server's OpenAPI schema rather than
      hand-written
- [x] Regenerating the client and running the suite fails when the server schema
      changes, naming the changed field
- [x] Gradle files exist only under the Android project directory; none at repository
      root, and the Gradle build never scans the Python tree
- [x] The build pins the JDK and Android Gradle Plugin versions explicitly rather
      than inheriting whatever the machine has

## Verification done

- Debug APK builds: 10.4 MB, `BUILD SUCCESSFUL`.
- Contract gate proven to fail, not assumed. Renaming `emergency_level` to
  `emergencyLevel`, and `VoiceQueryResponse.audio_base64` to `audioBase64`, each
  fail the build naming the field.
- Toolchain: JDK 17, Gradle 8.11.1 plus wrapper, SDK platform-35, build-tools
  35.0.0, platform-tools, all user-local.

## Known limitations

- **The generated client itself is not generated yet.** `:core:api` holds the
  contract, the gate, and the Retrofit/Moshi/OkHttp dependencies. Wiring an
  OpenAPI generator is the first thing in ticket 03; until then "generated" is a
  seam rather than an output.
- **The Gradle wrapper has not been exercised through `./gradlew`**, only through
  a local Gradle install. `gradle-wrapper.properties` points at the standard
  services.gradle.org URL, so it should work, but that path is untested here.
