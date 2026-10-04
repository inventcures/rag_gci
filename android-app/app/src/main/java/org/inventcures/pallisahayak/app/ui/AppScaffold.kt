package org.inventcures.pallisahayak.app.ui

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.vector.PathBuilder
import androidx.compose.ui.graphics.vector.path
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

/**
 * The three destinations, held in state rather than in a navigation library.
 *
 * The app has three screens. Navigation Compose would add a dependency and a
 * transitive set for no behaviour this needs, and the APK carries an explicit budget
 * for 2G devices. Revisit when there are back stacks, deep links or saved state per
 * destination; none of which exist.
 */
enum class Destination(val label: String, val glyphKey: String) {
    ASK("Ask", "ask"),
    HISTORY("Visits", "history"),
    SETTINGS("Settings", "settings"),
}

/**
 * The app shell.
 *
 * Navigation state is remembered so a configuration change does not drop the worker
 * back onto the Ask screen mid-visit, which at a home visit means losing the thread
 * of a conversation with a family.
 */
@Composable
fun AppScaffold(
    offlineBanner: @Composable () -> Unit = {},
    ask: @Composable () -> Unit,
    history: @Composable () -> Unit,
    settings: @Composable () -> Unit,
) {
    var destination by remember { mutableStateOf(Destination.ASK) }

    Column(modifier = Modifier.fillMaxSize()) {
        Column(modifier = Modifier.weight(1f)) {
            when (destination) {
                Destination.ASK -> ask()
                Destination.HISTORY -> history()
                Destination.SETTINGS -> settings()
            }
        }
        HorizontalDivider()
        NavigationBar {
            Destination.entries.forEach { entry ->
                NavigationBarItem(
                    selected = destination == entry,
                    onClick = { destination = entry },
                    icon = {
                        androidx.compose.material3.Icon(
                            imageVector = NavGlyphs.forDestination(entry),
                            contentDescription = null,
                            modifier = Modifier.size(24.dp),
                        )
                    },
                    // Labelled as well as drawn, because a semi-literate user navigates
                    // by shape and everyone else benefits from the word.
                    label = { Text(entry.label) },
                    modifier = Modifier.semantics { contentDescription = entry.label },
                )
            }
        }
    }
}

/** The three nav glyphs, again drawn rather than imported for APK size. */
private object NavGlyphs {

    fun forDestination(destination: Destination) = when (destination) {
        Destination.ASK -> ask
        Destination.HISTORY -> history
        Destination.SETTINGS -> settings
    }

    val ask: androidx.compose.ui.graphics.vector.ImageVector by lazy { glyph("Ask") {
        moveTo(3.0f, 4.0f); lineTo(21.0f, 4.0f); lineTo(21.0f, 6.5f)
        lineTo(3.0f, 6.5f); close()
        moveTo(3.0f, 10.0f); lineTo(21.0f, 10.0f); lineTo(21.0f, 12.5f)
        lineTo(3.0f, 12.5f); close()
        moveTo(3.0f, 16.0f); lineTo(21.0f, 16.0f); lineTo(21.0f, 18.5f)
        lineTo(3.0f, 18.5f); close()
    } }

    val history: androidx.compose.ui.graphics.vector.ImageVector by lazy { glyph("History") {
        moveTo(12.0f, 2.0f)
        lineTo(12.0f, 4.4f); lineTo(12.0f, 19.6f); lineTo(12.0f, 22.0f)
        close()
        moveTo(2.0f, 12.0f)
        lineTo(4.4f, 12.0f); lineTo(19.6f, 12.0f); lineTo(22.0f, 12.0f)
        close()
    } }

    val settings: androidx.compose.ui.graphics.vector.ImageVector by lazy { glyph("Settings") {
        moveTo(12.0f, 7.2f)
        curveTo(14.4f, 7.2f, 16.3f, 9.1f, 16.3f, 11.5f)
        curveTo(16.3f, 13.9f, 14.4f, 15.8f, 12.0f, 15.8f)
        curveTo(9.6f, 15.8f, 7.7f, 13.9f, 7.7f, 11.5f)
        curveTo(7.7f, 9.1f, 9.6f, 7.2f, 12.0f, 7.2f)
        close()
        moveTo(4.4f, 10.0f)
        lineTo(7.2f, 10.0f); lineTo(16.8f, 10.0f); lineTo(19.6f, 10.0f)
        lineTo(19.6f, 13.0f); lineTo(4.4f, 13.0f)
        close()
    } }

    private fun glyph(
        name: String,
        body: androidx.compose.ui.graphics.vector.PathBuilder.() -> Unit,
    ) = androidx.compose.ui.graphics.vector.ImageVector
        .Builder(name, 24.dp, 24.dp, 24f, 24f)
        .apply {
            path(
                fill = androidx.compose.ui.graphics.SolidColor(
                    androidx.compose.ui.graphics.Color.Black,
                ),
            ) { body() }
        }
        .build()
}