package org.inventcures.pallisahayak.app

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import com.google.common.truth.Truth.assertThat
import com.google.common.truth.Truth.assertWithMessage
import java.util.Locale
import org.inventcures.pallisahayak.AppLocale
import org.inventcures.pallisahayak.R
import org.inventcures.pallisahayak.data.model.Language
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * With the phone in Hindi, every string the screens use is Hindi.
 *
 * Resolved through Android's own resource lookup for a hi-IN device, so it checks
 * what a participant would read rather than the XML files. Every app string resource
 * is covered by reflection over `R.string` (non-transitive, so only the app's own),
 * which means a new English string with no Hindi fails here. The source-level half,
 * that no screen holds a literal, is `scripts/check_no_hardcoded_strings.py`.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33], qualifiers = "hi-rIN")
class HindiScreensTest {

    private val context: Context = ApplicationProvider.getApplicationContext()

    private val placeholder = Regex("%(\\d+\\$)?[sd]")

    private fun appStrings(): Map<String, String> =
        R.string::class.java.fields.associate { field -> field.name to context.getString(field.getInt(null)) }

    @Test
    fun `every screen string resolves to Hindi on a phone in Hindi`() {
        val strings = appStrings()

        assertThat(strings).isNotEmpty()
        strings.forEach { (name, text) ->
            val letters = placeholder.replace(text, "").filter { it.isLetter() }
            assertWithMessage(name).that(letters).isNotEmpty()
            assertWithMessage("$name: $text")
                .that(letters.all { Character.UnicodeScript.of(it.code) == Character.UnicodeScript.DEVANAGARI })
                .isTrue()
        }
    }

    @Test
    fun `the emergency line reads in Hindi`() {
        assertThat(context.getString(R.string.emergency_call_108)).contains("108")
        assertThat(context.getString(R.string.emergency_call_108)).contains("आपातकाल")
    }

    @Test
    fun `a chosen language is shown in the locale of its table code`() {
        assertThat(AppLocale.localeFor(Language.HINDI)).isEqualTo(Locale.forLanguageTag("hi-IN"))
        // Odia's resources and locale are "or", never Sarvam's "od".
        assertThat(AppLocale.localeFor(Language.ODIA).language).isEqualTo("or")
    }
}
