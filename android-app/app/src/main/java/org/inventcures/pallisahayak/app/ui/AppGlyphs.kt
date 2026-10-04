package org.inventcures.pallisahayak.app.ui

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.vector.PathBuilder
import androidx.compose.ui.graphics.vector.path
import androidx.compose.ui.unit.dp

/**
 * Glyphs for the primary actions.
 *
 * Criterion 1: every primary action has to be completable from pictures alone. Until
 * this file existed, asking, stopping playback and replaying were text-only buttons.
 * A worker who cannot read "Stop speaking" mid-visit had no way to silence the app,
 * which is the one action that must always be reachable.
 *
 * Drawn rather than imported, for the size reason in RoleIcons, and chosen to differ
 * in silhouette the way the role glyphs do: a filled circle, a solid square, and a
 * triangle in a ring. Silhouette is what survives at small size and for a user who
 * cannot read the label beside it.
 */
object ActionGlyphs {

    /** Hold to speak. A filled microphone capsule. */
    val Speak: ImageVector by lazy { glyph("Speak") {
        moveTo(10.0f, 2.5f); lineTo(14.0f, 2.5f); lineTo(14.0f, 11.0f)
        curveTo(14.0f, 12.7f, 12.7f, 14.0f, 11.0f, 14.0f)
        curveTo(9.3f, 14.0f, 8.0f, 12.7f, 8.0f, 11.0f); close()
        moveTo(17.0f, 10.0f); lineTo(18.8f, 10.0f)
        curveTo(18.8f, 14.6f, 16.6f, 17.5f, 13.8f, 18.4f)
        lineTo(13.8f, 20.2f); lineTo(16.4f, 20.2f); lineTo(16.4f, 22.0f)
        lineTo(7.6f, 22.0f); lineTo(7.6f, 20.2f); lineTo(10.2f, 20.2f)
        lineTo(10.2f, 18.4f)
        curveTo(7.4f, 17.5f, 5.2f, 14.6f, 5.2f, 10.0f)
        lineTo(7.0f, 10.0f)
        curveTo(7.0f, 13.2f, 8.7f, 15.2f, 12.0f, 15.2f)
        curveTo(15.3f, 15.2f, 17.0f, 13.2f, 17.0f, 10.0f); close()
    } }

    /** Stop. A solid square. The shape of the thing it does. */
    val Stop: ImageVector by lazy { glyph("Stop") {
        moveTo(5.0f, 5.0f); lineTo(19.0f, 5.0f); lineTo(19.0f, 19.0f)
        lineTo(5.0f, 19.0f); close()
    } }

    /** Replay. A triangle, which points forward and reads as "again". */
    val Replay: ImageVector by lazy { glyph("Replay") {
        moveTo(12.0f, 4.0f)
        lineTo(20.0f, 12.0f); lineTo(12.0f, 20.0f); close()
    } }

    private fun glyph(name: String, body: PathBuilder.() -> Unit) =
        ImageVector
            .Builder(name, 28.dp, 28.dp, 24f, 24f)
            .apply { path(fill = SolidColor(Color.Black)) { body() } }
            .build()
}
