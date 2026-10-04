package org.inventcures.pallisahayak.app.ui

import androidx.compose.ui.test.assertCountEquals
import androidx.compose.ui.test.assertHeightIsAtLeast
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.unit.dp
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.inventcures.pallisahayak.data.model.CareRole
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The role switcher and the Active Role badge.
 *
 * Language coverage lives in `LanguageAndRoleTest`, which compares the enum against
 * `offline/questions.py`. Two picker tests were written here and removed: a
 * LazyColumn composes nothing under this Robolectric setup, so they could not be
 * made to pass honestly. Shipping a test that asserts something the harness cannot
 * render would be a test that proves nothing and reads as though it proves
 * something. The picker itself is therefore unverified at the UI level and needs an
 * instrumented test or a host-side render.
 *
 * Two properties are load-bearing and neither can be checked by reading the code.
 *
 * Switching must cost one tap. A confirmation dialog or a PIN field would signal that
 * something is protected, which ADR 0006 says is false, so the test asserts the
 * selection lands on the first click with nothing in the way.
 *
 * The badge must be readable without reading. It is asserted by measuring the drawn
 * icon rather than by checking that a label exists, because a label is exactly what
 * a semi-literate user cannot use.
 */
@RunWith(AndroidJUnit4::class)
class RoleAndLanguageUiTest {

    @get:Rule
    val compose = createComposeRule()

    @Test
    fun switchingRoleTakesOneTapAndAsksForNothing() {
        val selections = mutableListOf<CareRole>()
        compose.setContent {
            RoleSwitcher(active = CareRole.ASHA_WORKER, onSelect = { selections += it })
        }

        compose.onNodeWithContentDescription(CareRole.PATIENT.label).performClick()

        assertEquals(listOf(CareRole.PATIENT), selections)
    }

    @Test
    fun theSwitcherNeverAsksForAPin() {
        compose.setContent {
            RoleSwitcher(active = CareRole.ASHA_WORKER, onSelect = {})
        }

        // No PIN, password or confirmation may exist anywhere on this screen.
        // An earlier version of this test fetched the nodes and asserted nothing,
        // which passed whether or not a PIN field was present. It is asserted now,
        // or it is not a test.
        listOf("PIN", "Password", "Confirm", "Verify").forEach { forbidden ->
            compose.onAllNodes(hasText(forbidden, substring = true))
                .assertCountEquals(0)
        }
    }

    @Test
    fun theActiveRoleIsDrawnLargeEnoughToSeeWithoutReading() {
        compose.setContent { ActiveRoleBadge(role = CareRole.FAMILY_CAREGIVER) }

        // Measured, not assumed. A badge that shrank to a 16dp dot would still have
        // a correct content description and still be useless in the field.
        compose.onNodeWithContentDescription("Active role. ${CareRole.FAMILY_CAREGIVER.label}")
            .assertHeightIsAtLeast(56.dp)
    }

    @Test
    fun theBadgeAnnouncesFramingRatherThanProtection() {
        compose.setContent { ActiveRoleBadge(role = CareRole.PATIENT) }

        // "Speaking as" is framing. Anything about signing in or privacy would be a
        // claim the app cannot keep, and the announcement is where a screen reader
        // user would hear that claim made.
        compose.onNodeWithContentDescription("Active role. Speaking as a patient")
            .assertExists()
    }
}
