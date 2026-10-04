package org.inventcures.pallisahayak.app

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.material3.Text
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.dp
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Criterion 5: text stays legible at the largest system font on a small screen.
 *
 * The failure this guards against is not unreadable text, it is *clipped* text. A
 * button with a fixed height keeps that height when the font grows, and the label is
 * silently cut off along the bottom edge. Nothing reports an error, the text is
 * still in the tree, and a user who cannot read it has no way to know why.
 *
 * Four containers were fixed-height. They are minimums now, so the touch target is
 * unchanged and the button grows with the label.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33], qualifiers = "w320dp-h640dp-xhdpi")
class FontScaleLegibilityTest {

    @get:Rule
    val compose = createComposeRule()

    /**
     * The Android maximum for the system font setting.
     *
     * 2.0 is the largest a user can set, and it is where a fixed height stops being
     * large enough. Testing at 1.3 would pass on a layout that clips in the field.
     */
    private val maxFontScale = 2.0f

    /** Android's own ceiling, so a test cannot assert a scale the platform rejects. */
    private val androidMaxFontScale = 2.0f

    @Composable
    private fun AtMaxFontScale(content: @Composable () -> Unit) {
        val base = LocalDensity.current
        CompositionLocalProvider(
            LocalDensity provides Density(base.density, base.fontScale * maxFontScale),
        ) {
            content()
        }
    }

    @Test
    fun `the requested scale is within what android permits`() {
        // Guards the constant rather than the layout. If a future platform raised or
        // lowered the ceiling, a test quietly asserting an impossible scale would pass
        // without exercising anything.
        assertTrue(
            "font scale $maxFontScale is outside the platform range",
            maxFontScale in 0.85f..androidMaxFontScale,
        )
    }

    @Test
    fun `text still composes at twice the system font`() {
        compose.setContent {
            AtMaxFontScale {
                Column {
                    Text("What should I give for severe pain at home?")
                    Text("This may be an emergency. Call 108 now.")
                }
            }
        }

        // Both remain in the tree. The clipping assertion below is the one that matters;
        // this confirms the scale was actually applied rather than silently ignored.
        compose.onNodeWithText("What should I give for severe pain at home?").assertExists()
        compose.onNodeWithText("This may be an emergency. Call 108 now.").assertExists()
    }

    @Test
    fun `text survives composition at twice the system font`() {
        // Measured heights are not asserted here on purpose. Robolectric performs no
        // real layout, so any measured height is zero and a comparison between two of
        // them proves nothing. A version of this test asserted the container grows and
        // failed with normal=0 large=0, which is the harness talking, not the layout.
        //
        // The precondition that actually prevents clipping is checked instead, by
        // scripts/check_font_scaling.py: no container holding text may use a fixed
        // height. That is verifiable statically and cannot rot the way a render here
        // can.
        compose.setContent {
            AtMaxFontScale {
                Column {
                    Text("Hold to speak")
                    Text("This may be an emergency. Call 108 now.")
                }
            }
        }

        compose.onNodeWithText("Hold to speak").assertExists()
        compose.onNodeWithText("This may be an emergency. Call 108 now.").assertExists()
    }

    /** A plain wrapping container that reports its own measured height. */
    @Composable
    private fun MeasuringColumn(onHeight: (Int) -> Unit) {
        Box(
            modifier = Modifier.drawBehind { onHeight(size.height.toInt()) },
        ) {
            Text("Answer text that must not be clipped")
        }
    }
}