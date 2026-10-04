package org.inventcures.pallisahayak.app.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import org.inventcures.pallisahayak.R
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import org.inventcures.pallisahayak.data.model.CareRole
import org.inventcures.pallisahayak.data.model.Language

/**
 * The Active Role badge.
 *
 * Persistent and large, because whoever is holding the phone has to see whose view
 * they are looking at without reading anything. That is the whole function of this
 * component: a person who cannot read the label still needs to know.
 *
 * Colour is a second channel, not the only one. Two of the three roles would be hard
 * to tell apart by hue alone on a cheap panel in daylight.
 *
 * There is deliberately no lock, no shield and no "secure" affordance anywhere in
 * this file. ADR 0006 settled that the Care Role is not an access boundary, and an
 * icon that implies protection it cannot deliver teaches people to stop checking.
 */
@Composable
fun ActiveRoleBadge(
    role: CareRole,
    modifier: Modifier = Modifier,
    /** Large enough to read across a room. The badge is meant to be seen, not read. */
    size: androidx.compose.ui.unit.Dp = 56.dp,
) {
    Box(
        modifier = modifier
            .size(size + 16.dp)
            .background(role.tint.copy(alpha = 0.12f), CircleShape)
            .semantics {
                // Announced as context, never as protection. The wording matters as
                // much as the glyph: "Speaking as" is framing, "Signed in as" is not.
                contentDescription = "Active role. ${role.label}"
                stateDescription = role.label
            },
        contentAlignment = Alignment.Center,
    ) {
        androidx.compose.material3.Icon(
            imageVector = RoleIcons.forRole(role),
            contentDescription = null,
            tint = role.tint,
            modifier = Modifier.size(size),
        )
    }
}

/**
 * The role switcher.
 *
 * One tap, no PIN, no confirmation dialog. A dialog would signal that something is
 * being protected, which is the one thing this switch must not do.
 *
 * Tapping the role that is already active does nothing. It is not disabled either:
 * a disabled control invites the reader to wonder what is wrong with it, and this
 * switch is never locked.
 */
@Composable
fun RoleSwitcher(
    active: CareRole,
    onSelect: (CareRole) -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(modifier = modifier) {
        Text(
            text = active.label,
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold,
        )
        Text(
            text = stringResource(R.string.role_switch_explainer),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(bottom = 8.dp),
        )
        Row(
            horizontalArrangement = Arrangement.spacedBy(20.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            CareRole.ALL.forEach { role ->
                val isActive = role == active
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    modifier = Modifier
                        .clickable { onSelect(role) }
                        .padding(8.dp)
                        .semantics {
                            contentDescription = role.label
                            stateDescription =
                                if (isActive) "currently speaking as" else "switch to"
                        },
                ) {
                    androidx.compose.material3.Icon(
                        imageVector = RoleIcons.forRole(role),
                        contentDescription = null,
                        // The active role is drawn larger as well as tinted, because
                        // tint alone is not enough on a washed-out screen.
                        modifier = Modifier.size(if (isActive) 56.dp else 40.dp),
                        tint = if (isActive) role.tint else MaterialTheme.colorScheme.outline,
                    )
                    Text(
                        text = role.shortLabel,
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = if (isActive) FontWeight.Bold else FontWeight.Normal,
                    )
                }
            }
        }
    }
}

/**
 * Language selection, shown once.
 *
 * Endonyms, so a worker who reads Devanagari or Tamil finds their language without
 * hunting through English names. Ordering is the enum's, which matches the server's
 * `SUPPORTED_LANGUAGES`, and a mismatch there would pair the wrong offline question
 * with the wrong answer.
 *
 * Tapping a language selects it and finishes. There is no second confirmation step
 * and no back button, because a choice that must be confirmed is a choice the app
 * does not trust the user to have made.
 */
@Composable
fun LanguagePicker(
    selected: Language?,
    onSelect: (Language) -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(modifier = modifier.padding(16.dp)) {
        Text(
            text = stringResource(R.string.language_picker_title),
            style = MaterialTheme.typography.headlineSmall,
            modifier = Modifier.padding(bottom = 4.dp),
        )
        Text(
            text = stringResource(R.string.language_picker_explainer),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(bottom = 12.dp),
        )
        LazyColumn {
            items(Language.ALL, key = { it.tag }) { language ->
                val isSelected = language == selected
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { onSelect(language) }
                        // A generous row, because this is tapped once and for all by
                        // someone who may be navigating by shape rather than by eye.
                        .padding(vertical = 12.dp)
                        .semantics {
                            contentDescription = language.endonym
                            stateDescription =
                                if (isSelected) "selected" else "not selected"
                        },
                ) {
                    RadioButton(selected = isSelected, onClick = { onSelect(language) })
                    Text(
                        text = language.endonym,
                        style = MaterialTheme.typography.bodyLarge,
                        modifier = Modifier.padding(start = 12.dp),
                    )
                }
            }
        }
    }
}

/** A short label for the switcher, where the full framing text would not fit. */
private val CareRole.shortLabel: String
    get() = when (this) {
        CareRole.ASHA_WORKER -> "ASHA"
        CareRole.FAMILY_CAREGIVER -> "Family"
        CareRole.PATIENT -> "Patient"
    }

/**
 * Role colours.
 *
 * Distinct hues, and each passes against both its own tint and a white card. Chosen
 * for separation in hue and in lightness, so the three do not collapse into one
 * another on a low-quality panel in bright daylight.
 */
val CareRole.tint: Color
    get() = when (this) {
        CareRole.ASHA_WORKER -> Color(0xFF1B6EC2)
        CareRole.FAMILY_CAREGIVER -> Color(0xFFB4531A)
        CareRole.PATIENT -> Color(0xFF2E7D4F)
    }