# R8 rules for the release build (app/build.gradle.kts names this file).
#
# The generated API models are read and written by Moshi's reflective Kotlin
# adapter, which needs their constructors, fields and Kotlin metadata as they are
# in source. Without this, R8 renames or strips them and every call, sign-in
# included, fails to (de)serialise in the release build only.
-keep class org.inventcures.pallisahayak.api.generated.** { *; }
-keep class kotlin.Metadata { *; }

# The sync push models are written by hand (sync is not a generated operation)
# and read by the same reflective adapter, so they need the same rule.
-keep class org.inventcures.pallisahayak.data.sync.wire.** { *; }
