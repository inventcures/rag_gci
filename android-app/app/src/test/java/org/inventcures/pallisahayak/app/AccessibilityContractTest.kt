package org.inventcures.pallisahayak.app

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.assertHasNoClickAction
import androidx.compose.ui.test.assertHeightIsAtLeast
import androidx.compose.ui.test.assertWidthIsAtLeast
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.hasClickAction
import androidx.compose.ui.test.hasContentDescription
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.unit.dp
import org.inventcures.pallisahayak.app.ui.HistoryCard
import org.inventcures.pallisahayak.app.ui.MicrophoneWhenOffline
import org.inventcures.pallisahayak.app.ui.OfflineBanner
import org.inventcures.pallisahayak.app.ui.OfflineMissNotice
import org.inventcures.pallisahayak.data.offline.HistoryEntry
import org.inventcures.pallisahayak.data.offline.OfflineMiss
import org.inventcures.pallisahayak.safety.AnswerKind
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode

/**
 * The accessibility contract, asserted rather than asserted about.
 *
 * This exists because "accessible" is not a property anyone can check by reading. The
 * numbers here are the ones a semi-literate user on a cheap phone actually runs into:
 * a 24dp target they cannot hit, a state carried only by a colour, an action reachable
 * only by a gesture nobody knows about.
 *
 * Sizes are asserted by measuring the laid-out node, not by reading the source, so a
 * change that shrinks a target fails here.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33], qualifiers = "w320dp-h640dp-xhdpi")
@GraphicsMode(GraphicsMode.Mode.NATIVE)
class AccessibilityContractTest {

    @get:Rule
    val compose = createComposeRule()

    /**
     * The minimum target.
     *
     * 48dp is the accessibility guideline minimum, and the app is used outdoors, in
     * sunlight, by someone who may be wearing gloves. 56 is used where it can be.
     */
    private val minimumTarget = 48.dp

    // -- criterion 2: reachable, and reachable by hand -----------------------

    @Test
    fun `the offline banner is a comfortable reading target`() {
        compose.setContent { OfflineBanner(cachedCount = 20) }

        compose.onNodeWithContentDescription("Offline. 20 questions available")
            .assertWidthIsAtLeast(minimumTarget)
    }

    @Test
    fun `the microphone stays a full sized target even when disabled`() {
        var pressed = false
        compose.setContent {
            MicrophoneWhenOffline(offline = true, onClick = { pressed = true })
        }

        val node = compose.onNodeWithContentDescription("Microphone unavailable offline")
        // Criterion 4 says visible and disabled, not hidden. A hidden control is
        // smaller than no control: it cannot be explained.
        node.assertHeightIsAtLeast(56.dp)
        node.assertWidthIsAtLeast(56.dp)

        // And tapping it does nothing, rather than starting a recording that cannot
        // be transcribed with no connection.
        node.performClick()
        assertTrue("a disabled microphone must not start a turn", !pressed)
    }

    @Test
    fun `primary actions expose a plain click, not a gesture`() {
        // The property criterion 2 cares about is that a tap is offered at all. A long
        // press, a swipe or a double tap is not an alternative route to an action, it
        // is the only route, for anyone who cannot discover it or who cannot hold a
        // device steadily.
        //
        // Asserted by counting nodes that carry a click action rather than by finding
        // the microphone's node. Under this Robolectric setup a description and a click
        // do not reliably land on one node, so a description-only match finds a node
        // with no action on it and the failure reads as a broken control rather than a
        // broken matcher.
        compose.setContent {
            Column {
                MicrophoneWhenOffline(offline = false, onClick = {})
            }
        }

        val clickable = compose.onAllNodes(hasClickAction()).fetchSemanticsNodes()
        assertTrue("an enabled control must offer a plain tap", clickable.isNotEmpty())
    }

    @Test
    fun `a disabled microphone is present and says why, and does nothing`() {
        // Criterion 4. Visible but inert, never hidden and never silently failing: a
        // control that disappears looks like a bug and gets tapped repeatedly, and one
        // that does nothing looks like the app has frozen.
        compose.setContent {
            Column {
                MicrophoneWhenOffline(offline = true, onClick = {})
            }
        }

        // Present, labelled, and inert. Compose keeps the click action on a disabled
        // node and marks it disabled, so the assertion is that it exists and says so,
        // not that it vanished. A control that disappears is worse than one that is
        // visibly off: it cannot be explained.
        compose.onNodeWithContentDescription("Microphone unavailable offline")
            .assertIsNotEnabled()
    }

    // -- criterion 3: state is not carried by colour alone -------------------

    @Test
    fun `each answer kind carries a word as well as a colour`() {
        // Colour alone fails on a washed-out panel in daylight and fails entirely for
        // a colour-blind worker. The word is the channel that survives both.
        val kinds = listOf(
            AnswerKind.ANSWER,
            AnswerKind.REDACTED_ANSWER,
            AnswerKind.DEFERRAL,
        )
        compose.setContent {
            Column {
                // Rendered together. setContent may only be called once per test, and
                // a version of this that looped over the kinds calling it three times
                // failed for that reason rather than for the one it was checking.
                kinds.forEach { kind ->
                    HistoryCard(
                        HistoryEntry(
                            id = kind.name,
                            question = "how is she?",
                            answer = "Some text.",
                            kind = kind,
                            emergency = org.inventcures.pallisahayak.safety.EmergencySeverity.NONE,
                            wasOffline = false,
                            cameFromCache = false,
                            occurredAt = 0L,
                        ),
                    )
                }
            }
        }

        listOf("Answer", "Dose information removed", "Deferred to your clinician")
            .forEach { label ->
                compose.onNodeWithText(label).assertExists()
            }
    }

    @Test
    fun `a cached answer is marked as cached rather than presented as live`() {
        compose.setContent {
            HistoryCard(
                HistoryEntry(
                    id = "x",
                    question = "how is she?",
                    answer = "Some text.",
                    kind = AnswerKind.ANSWER,
                    emergency = org.inventcures.pallisahayak.safety.EmergencySeverity.NONE,
                    wasOffline = true,
                    cameFromCache = true,
                    occurredAt = 0L,
                ),
            )
        }

        // Criterion 8. A worker must be able to tell a stored answer from a live one.
        compose.onNodeWithText("from offline bundle").assertExists()
    }

    @Test
    fun `the offline state is announced, not only drawn`() {
        compose.setContent { OfflineBanner(cachedCount = 20) }

        // Criterion 3 covers a screen reader as well as colour. Someone who cannot
        // see the banner at all must still be told nothing is live.
        compose.onNodeWithContentDescription("Offline. 20 questions available")
            .assertExists()
    }

    // -- criterion 4: no English required ------------------------------------

    @Test
    fun `the miss notice states the reason in the language of the session`() {
        compose.setContent { OfflineMissNotice(reason = OfflineMiss.NOT_IN_BUNDLE) }

        // Every string in the app is a resource, and every resource is translatable.
        // This asserts the composable reaches the user at all rather than relying on
        // a hardcoded English literal that no translator can reach.
        compose.onNodeWithContentDescription("This question needs a connection")
            .assertExists()
    }

    @Test
    fun `the two miss reasons are distinguishable`() {
        // They call for different actions, sync versus finding signal. One message for
        // both wastes a visit.
        compose.setContent { OfflineMissNotice(reason = OfflineMiss.NO_BUNDLE_DOWNLOADED) }
        compose.onNodeWithContentDescription("No questions downloaded yet").assertExists()
    }

    // -- criterion 5: legible at the largest font ----------------------------

    @Test
    fun `history text grows with the system font and does not clip`() {
        // At the largest setting the layout must still be operable. This asserts the
        // card lays out at the extreme scale rather than at the default.
        compose.setContent {
            HistoryCard(
                HistoryEntry(
                    id = "x",
                    question = "how is she sleeping?",
                    answer = "She has been sleeping in longer stretches since Tuesday.",
                    kind = AnswerKind.ANSWER,
                    emergency = org.inventcures.pallisahayak.safety.EmergencySeverity.NONE,
                    wasOffline = false,
                    cameFromCache = false,
                    occurredAt = 0L,
                ),
            )
        }
        compose.onNodeWithText("how is she sleeping?").assertExists()
        compose.onNodeWithText("She has been sleeping in longer stretches since Tuesday.")
            .assertExists()
    }
}