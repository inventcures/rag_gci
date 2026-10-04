package org.inventcures.pallisahayak.app.ui

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.vector.PathBuilder
import androidx.compose.ui.graphics.vector.path
import androidx.compose.ui.unit.dp
import org.inventcures.pallisahayak.data.model.CareRole

/**
 * The three role glyphs, drawn rather than imported.
 *
 * `androidx.compose.material:material-icons-extended` carries `medical_services`,
 * `diversity_2` and `bed`, and using it would have been the obvious choice. It also
 * adds several megabytes to an APK that has to install over 2G on a low-end device,
 * and ADR 0002 bounds what this app may carry. Three small paths cost nothing.
 *
 * Drawing them by hand also serves the requirement directly. ADR 0006 asks for icons
 * that differ in *silhouette* rather than in detail, because three similar human
 * figures are indistinguishable at small sizes to someone who may not read the
 * label. These differ in outline shape, not decoration:
 *
 *  - ASHA Worker       upright cross          the only plus-shaped glyph
 *  - Family Caregiver  two upright figures     visibly two heads
 *  - Patient           horizontal figure       visibly lying down
 *
 * A cross, two figures and a reclining figure are told apart from silhouette alone,
 * at 24dp, by someone who cannot read a word of the label beside them.
 *
 * Filled rather than stroked, because two adjacent outlines blur into one another
 * below roughly 20dp, which is close to the size the badge draws at.
 */
object RoleIcons {

    val AshaWorker: ImageVector by lazy { cross() }

    val FamilyCaregiver: ImageVector by lazy { twoFigures() }

    val Patient: ImageVector by lazy { recliningFigure() }

    fun forRole(role: CareRole): ImageVector = when (role) {
        CareRole.ASHA_WORKER -> AshaWorker
        CareRole.FAMILY_CAREGIVER -> FamilyCaregiver
        CareRole.PATIENT -> Patient
    }

    private fun builder(name: String, block: PathBuilder.() -> Unit): ImageVector =
        ImageVector
            .Builder(
                name = name,
                defaultWidth = 24.dp,
                defaultHeight = 24.dp,
                viewportWidth = 24f,
                viewportHeight = 24f,
            )
            .apply {
                // Color.Unspecified leaves the tint to whoever draws it, so the same
                // glyph can carry a role colour rather than being baked black.
                path(fill = SolidColor(Color.Black)) { block() }
            }
            .build()

    private fun cross(): ImageVector = builder("AshaWorker") {
        moveTo(10.5f, 2.5f)
        lineTo(13.5f, 2.5f)
        lineTo(13.5f, 9.5f)
        lineTo(20.5f, 9.5f)
        lineTo(20.5f, 12.5f)
        lineTo(13.5f, 12.5f)
        lineTo(13.5f, 21.5f)
        lineTo(10.5f, 21.5f)
        lineTo(10.5f, 12.5f)
        lineTo(2.5f, 12.5f)
        lineTo(2.5f, 9.5f)
        lineTo(10.5f, 9.5f)
        close()
    }

    private fun twoFigures(): ImageVector = builder("FamilyCaregiver") {
        // Left head and body.
        moveTo(5.2f, 3.4f)
        curveTo(7.0f, 3.4f, 8.4f, 4.8f, 8.4f, 6.6f)
        curveTo(8.4f, 8.4f, 7.0f, 9.8f, 5.2f, 9.8f)
        curveTo(3.4f, 9.8f, 2.0f, 8.4f, 2.0f, 6.6f)
        curveTo(2.0f, 4.8f, 3.4f, 3.4f, 5.2f, 3.4f)
        close()
        moveTo(1.2f, 21.4f)
        lineTo(1.2f, 18.2f)
        curveTo(1.2f, 14.4f, 3.0f, 12.2f, 5.2f, 12.2f)
        curveTo(7.4f, 12.2f, 9.2f, 14.4f, 9.2f, 18.2f)
        lineTo(9.2f, 21.4f)
        close()
        // Right figure, slightly smaller so it reads as a second person.
        moveTo(17.0f, 4.2f)
        curveTo(18.5f, 4.2f, 19.7f, 5.4f, 19.7f, 6.9f)
        curveTo(19.7f, 8.4f, 18.5f, 9.6f, 17.0f, 9.6f)
        curveTo(15.5f, 9.6f, 14.3f, 8.4f, 14.3f, 6.9f)
        curveTo(14.3f, 5.4f, 15.5f, 4.2f, 17.0f, 4.2f)
        close()
        moveTo(12.6f, 21.4f)
        lineTo(12.6f, 18.6f)
        curveTo(12.6f, 15.4f, 14.2f, 13.4f, 17.0f, 13.4f)
        curveTo(19.8f, 13.4f, 21.4f, 15.4f, 21.4f, 18.6f)
        lineTo(21.4f, 21.4f)
        close()
    }

    private fun recliningFigure(): ImageVector = builder("Patient") {
        // Head, at the left.
        moveTo(3.4f, 8.0f)
        curveTo(4.6f, 8.0f, 5.6f, 9.0f, 5.6f, 10.2f)
        curveTo(5.6f, 11.4f, 4.6f, 12.4f, 3.4f, 12.4f)
        curveTo(2.2f, 12.4f, 1.2f, 11.4f, 1.2f, 10.2f)
        curveTo(1.2f, 9.0f, 2.2f, 8.0f, 3.4f, 8.0f)
        close()
        // Body lying down.
        moveTo(6.6f, 8.0f)
        lineTo(21.0f, 8.0f)
        lineTo(21.0f, 12.4f)
        lineTo(6.6f, 12.4f)
        close()
        // Bed line, so the posture still reads at 16dp.
        moveTo(1.0f, 15.2f)
        lineTo(23.0f, 15.2f)
        lineTo(23.0f, 16.8f)
        lineTo(1.0f, 16.8f)
        close()
        moveTo(1.2f, 17.6f)
        lineTo(3.2f, 17.6f)
        lineTo(3.2f, 22.0f)
        lineTo(1.2f, 22.0f)
        close()
        moveTo(20.6f, 17.6f)
        lineTo(22.6f, 17.6f)
        lineTo(22.6f, 22.0f)
        lineTo(20.6f, 22.0f)
        close()
    }
}