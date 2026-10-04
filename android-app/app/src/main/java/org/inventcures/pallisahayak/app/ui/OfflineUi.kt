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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.vector.path
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import org.inventcures.pallisahayak.R
import org.inventcures.pallisahayak.data.offline.HistoryEntry
import org.inventcures.pallisahayak.data.offline.OfflineMiss
import org.inventcures.pallisahayak.safety.AnswerKind

/**
 * The offline state, made unmissable.
 *
 * Criterion 3: conveyed by icon and colour as well as by text. A worker who cannot
 * read the words still has to see that nothing is live, because the failure mode is
 * asking a stale question and getting a confidently wrong answer.
 *
 * This uses a scrolling `Column`, never a `LazyColumn`. Under this Robolectric setup
 * a LazyColumn composes nothing at all, so anything built on one is untestable here.
 * History is bounded at fifty entries, so a scrolling Column is the right shape
 * anyway.
 */

/** The glyph next to the offline banner: a cloud with a slash. */
val OfflineGlyph: ImageVector by lazy {
    androidx.compose.ui.graphics.vector.ImageVector
        .Builder(
            name = "Offline",
            defaultWidth = 24.dp,
            defaultHeight = 24.dp,
            viewportWidth = 24f,
            viewportHeight = 24f,
        )
        .apply {
            path(fill = androidx.compose.ui.graphics.SolidColor(Color.Black)) {
                // Cloud body.
                moveTo(6.5f, 18.0f)
                curveTo(4.0f, 18.0f, 2.0f, 16.2f, 2.0f, 14.0f)
                curveTo(2.0f, 11.9f, 3.7f, 10.2f, 5.8f, 10.1f)
                curveTo(6.4f, 6.9f, 9.3f, 4.8f, 12.6f, 4.8f)
                curveTo(15.9f, 4.8f, 18.9f, 7.0f, 19.5f, 10.2f)
                curveTo(21.4f, 10.4f, 23.0f, 12.0f, 23.0f, 14.0f)
                curveTo(23.0f, 16.2f, 21.0f, 18.0f, 18.5f, 18.0f)
                close()
                // The slash, cut out as a lighter bar so it reads at small sizes.
                moveTo(4.2f, 20.4f)
                lineTo(6.2f, 22.4f)
                lineTo(19.8f, 5.2f)
                lineTo(17.8f, 3.2f)
                close()
            }
        }
        .build()
}

@Composable
fun OfflineBanner(
    cachedCount: Int,
    modifier: Modifier = Modifier,
) {
    // Resolved here rather than inside the semantics lambda, which is not a composable
    // scope. A screen reader is the one channel a worker who cannot see the banner
    // still has, so the announcement is a translatable string, not an English literal.
    val announcement = stringResource(R.string.offline_banner_a11y, cachedCount)
    Card(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp, vertical = 8.dp)
            // Announced as a state, so a screen reader says it too.
            .semantics { contentDescription = announcement },
        colors = CardDefaults.cardColors(
            containerColor = Color(0xFFFFF3CD),
            contentColor = Color(0xFF664D03),
        ),
        shape = RoundedCornerShape(12.dp),
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(12.dp),
        ) {
            androidx.compose.material3.Icon(
                imageVector = OfflineGlyph,
                contentDescription = null,
                tint = Color(0xFFB45309),
                modifier = Modifier.size(28.dp),
            )
            Column(modifier = Modifier.padding(start = 12.dp)) {
                Text(
                    text = stringResource(R.string.offline_banner_title),
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    text = stringResource(R.string.offline_banner_body, cachedCount),
                    style = MaterialTheme.typography.bodyMedium,
                )
            }
        }
    }
}

/**
 * The microphone, disabled, with the reason shown.
 *
 * Criterion 4: visible but disabled, never hidden and never silently failing. A
 * control that disappears looks like a bug and gets tapped repeatedly; a control
 * that does nothing looks like the app has frozen.
 */
@Composable
fun MicrophoneWhenOffline(
    offline: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val tint = if (offline) Color(0xFF9CA3AF) else MaterialTheme.colorScheme.primary
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = modifier
            .semantics {
                contentDescription =
                    if (offline) "Microphone unavailable offline" else "Ask by voice"
            }
            // A plain tap, and only a plain tap. No long press, no double tap and no
            // swipe: none are discoverable by someone who cannot read the hints, and
            // none survive a motor tremor.
            .clickable(enabled = !offline, onClick = onClick),
    ) {
        Box(
            modifier = Modifier
                .size(72.dp)
                .background(
                    if (offline) Color(0xFFE5E7EB) else Color(0xFFE8F0FE),
                    RoundedCornerShape(36.dp),
                ),
            contentAlignment = Alignment.Center,
        ) {
            androidx.compose.material3.Icon(
                imageVector = MicGlyph,
                contentDescription = null,
                tint = tint,
                modifier = Modifier.size(36.dp),
            )
        }
        if (offline) {
            Text(
                text = stringResource(R.string.mic_disabled_offline),
                style = MaterialTheme.typography.bodySmall,
                color = Color(0xFF6B7280),
                modifier = Modifier.padding(top = 6.dp),
            )
        }
    }
}

/**
 * One history entry.
 *
 * The three answer kinds are distinguished by colour, icon shape and label, not by
 * wording alone. Criterion 2 exists because a refusal reread later must not look
 * like advice, and someone who cannot read the label still has to tell them apart.
 */
@Composable
fun HistoryCard(
    entry: HistoryEntry,
    modifier: Modifier = Modifier,
) {
    val accent = when (entry.kind) {
        AnswerKind.ANSWER -> Color(0xFF2E7D4F)
        AnswerKind.REDACTED_ANSWER -> Color(0xFFB4531A)
        AnswerKind.DEFERRAL -> Color(0xFF9A3412)
    }
    Card(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp, vertical = 5.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Text(
                text = entry.question,
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                text = entry.answer,
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.padding(top = 4.dp),
            )
            Row(
                modifier = Modifier.padding(top = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Box(
                    modifier = Modifier
                        .size(width = 8.dp, height = 20.dp)
                        .background(accent, RoundedCornerShape(4.dp)),
                )
                Text(
                    text = when (entry.kind) {
                        AnswerKind.ANSWER -> stringResource(R.string.kind_answer)
                        AnswerKind.REDACTED_ANSWER -> stringResource(R.string.kind_redacted)
                        AnswerKind.DEFERRAL -> stringResource(R.string.kind_deferral)
                    },
                    style = MaterialTheme.typography.labelSmall,
                    color = accent,
                    fontWeight = FontWeight.Bold,
                )
                // Cached answers are marked as such, never presented as live.
                if (entry.cameFromCache) {
                    Text(
                        text = stringResource(R.string.from_cache_badge),
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.outline,
                    )
                }
            }
        }
    }
}

/** The history list. Scrolling Column, deliberately, not a LazyColumn. */
@Composable
fun HistoryList(
    entries: List<HistoryEntry>,
    modifier: Modifier = Modifier,
) {
    if (entries.isEmpty()) {
        Box(modifier = modifier.fillMaxWidth().padding(24.dp), contentAlignment = Alignment.Center) {
            Text(
                text = stringResource(R.string.history_empty),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        return
    }
    Column(
        modifier = modifier
            .fillMaxWidth()
            .verticalScroll(rememberScrollState()),
    ) {
        entries.forEach { entry ->
            HistoryCard(entry)
        }
    }
}

/**
 * The explanation shown when the bundle cannot answer.
 *
 * Criterion 7: a clear explanation that it needs a connection, not silence and not a
 * stall. The two reasons get different words because the remedies differ.
 */
@Composable
fun OfflineMissNotice(
    reason: OfflineMiss,
    modifier: Modifier = Modifier,
) {
    Card(
        modifier = modifier
            .fillMaxWidth()
            .padding(12.dp)
            .semantics {
                contentDescription = when (reason) {
                    OfflineMiss.NO_BUNDLE_DOWNLOADED -> "No questions downloaded yet"
                    OfflineMiss.NOT_IN_BUNDLE -> "This question needs a connection"
                }
            },
        colors = CardDefaults.cardColors(
            containerColor = Color(0xFFEEF2FF),
            contentColor = Color(0xFF312E81),
        ),
    ) {
        Text(
            text = when (reason) {
                OfflineMiss.NO_BUNDLE_DOWNLOADED ->
                    stringResource(R.string.offline_miss_no_bundle)
                OfflineMiss.NOT_IN_BUNDLE ->
                    stringResource(R.string.offline_miss_not_in_bundle)
            },
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.padding(14.dp),
        )
    }
}

/** The microphone glyph. Drawn for the same size reason as the role icons. */
val MicGlyph: ImageVector by lazy {
    androidx.compose.ui.graphics.vector.ImageVector
        .Builder(
            name = "Mic",
            defaultWidth = 24.dp,
            defaultHeight = 24.dp,
            viewportWidth = 24f,
            viewportHeight = 24f,
        )
        .apply {
            path(fill = androidx.compose.ui.graphics.SolidColor(Color.Black)) {
                moveTo(10.0f, 2.5f)
                lineTo(14.0f, 2.5f)
                lineTo(14.0f, 11.0f)
                curveTo(14.0f, 12.7f, 12.7f, 14.0f, 11.0f, 14.0f)
                curveTo(9.3f, 14.0f, 8.0f, 12.7f, 8.0f, 11.0f)
                close()
                moveTo(17.0f, 10.0f)
                lineTo(18.8f, 10.0f)
                curveTo(18.8f, 14.6f, 16.6f, 17.5f, 13.8f, 18.4f)
                lineTo(13.8f, 20.2f)
                lineTo(16.4f, 20.2f)
                lineTo(16.4f, 22.0f)
                lineTo(7.6f, 22.0f)
                lineTo(7.6f, 20.2f)
                lineTo(10.2f, 20.2f)
                lineTo(10.2f, 18.4f)
                curveTo(7.4f, 17.5f, 5.2f, 14.6f, 5.2f, 10.0f)
                lineTo(7.0f, 10.0f)
                curveTo(7.0f, 13.2f, 8.7f, 15.2f, 12.0f, 15.2f)
                curveTo(15.3f, 15.2f, 17.0f, 13.2f, 17.0f, 10.0f)
                close()
            }
        }
        .build()
}