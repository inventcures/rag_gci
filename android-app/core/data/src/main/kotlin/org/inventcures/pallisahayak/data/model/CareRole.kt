package org.inventcures.pallisahayak.data.model

/**
 * The three Care Roles, as an analysis label.
 *
 * ADR 0006 settled what this is not. It is not an access boundary. On a shared
 * phone a caregiver can read the household record anyway, so a rule that appeared to
 * separate them would teach a false model of the device and make people stop
 * checking. A switch that looks like a protection and is not worse than no switch.
 *
 * So the icon is a *context* glyph, not a person, and no lock or shield appears
 * anywhere near it. The label reads as framing.
 *
 * The three icons differ in silhouette and posture rather than in detail, because
 * three similar human figures are indistinguishable at small sizes to a user who
 * may not read the label:
 *
 *  - ASHA Worker      medical_services   a cross inside a rounded form
 *  - Family Caregiver diversity_2        two figures, shoulder to shoulder
 *  - Patient          bed                a figure lying down
 *
 * These are Material icon names, resolved in [iconKey]. They differ in outline
 * shape at 24dp, which is the size the persistent badge draws at.
 */
enum class CareRole(
    /** What the study reports. The server and every metric use this. */
    val wireName: String,
    /** How the app frames it. Never implies access control. */
    val label: String,
    /** Material icon name, drawn persistently so it is recognisable unread. */
    val iconKey: String,
) {
    ASHA_WORKER("asha_worker", "Speaking as an ASHA Worker", "medical_services"),
    FAMILY_CAREGIVER(
        "family_caregiver",
        "Speaking as a family caregiver",
        "diversity_2",
    ),
    PATIENT("patient", "Speaking as a patient", "bed");

    companion object {
        val ALL: List<CareRole> = entries

        /**
         * Resolve a wire name, defaulting to the ASHA Worker.
         *
         * A default here is safe because a role is an analysis attribute and cannot
         * expose anything. An unknown value means a server we do not understand, and
         * continuing as the common case beats refusing to speak. This is deliberately
         * different from [Language.fromTag], which returns null, because a wrong
         * language misleads the user about being understood and a wrong role misleads
         * no one.
         */
        fun fromWire(value: String?): CareRole =
            ALL.firstOrNull { it.wireName == value } ?: ASHA_WORKER
    }
}