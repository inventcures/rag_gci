package org.inventcures.pallisahayak

import android.content.Context
import android.content.ContextWrapper
import android.content.res.Configuration
import android.content.res.Resources
import android.os.LocaleList
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import java.util.Locale
import org.inventcures.pallisahayak.data.model.Language

/**
 * The language the screens are shown in.
 *
 * The participant's chosen language, from the first-run picker or settings, so a
 * Hindi speaker reads Hindi whatever the phone is set to. Before a language is
 * chosen, the phone's own (a phone in Hindi shows the Hindi picker). Resources are
 * looked up under the language's table code (`values-hi`, and Odia under `or`).
 *
 * Applied inside the composition rather than by recreating the Activity: a
 * recreate re-runs the release check, which a release build offline cannot pass,
 * so switching language without signal would lock the app. Nor through AppCompat's
 * per-app locales, which would add AppCompat to an APK with a 2G size budget.
 */
object AppLocale {

    /** The locale a Supported Language's screens are shown in, e.g. `hi-IN`, `or-IN`. */
    fun localeFor(language: Language): Locale = Locale.forLanguageTag(language.tag)

    /**
     * [base] with its resources in [language], or [base] itself when none is chosen.
     *
     * A wrapper rather than the configuration context itself, so the Activity is
     * still reachable from it.
     */
    fun localized(base: Context, language: Language?): Context {
        if (language == null) return base
        val config = Configuration(base.resources.configuration)
        config.setLocales(LocaleList(localeFor(language)))
        val resources = base.createConfigurationContext(config).resources
        return object : ContextWrapper(base) {
            override fun getResources(): Resources = resources
        }
    }
}

/** Show [content] with its string resources in [language]. */
@Composable
fun InLanguage(language: Language?, content: @Composable () -> Unit) {
    val base = LocalContext.current
    val localized = remember(base, language) { AppLocale.localized(base, language) }
    CompositionLocalProvider(
        LocalContext provides localized,
        LocalConfiguration provides localized.resources.configuration,
        content = content,
    )
}
