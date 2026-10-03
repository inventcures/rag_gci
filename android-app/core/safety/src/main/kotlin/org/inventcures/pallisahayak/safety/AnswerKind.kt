package org.inventcures.pallisahayak.safety

/**
 * What kind of answer the user is looking at.
 *
 * The distinction is load-bearing. A [REDACTED_ANSWER] and a [DEFERRAL] are not
 * answers: a refusal saved weeks ago must never be read back as clinical guidance,
 * so it renders differently, is recorded differently, and is counted differently in
 * every metric the study reports.
 */
enum class AnswerKind {
    /** An answer the system stands behind. */
    ANSWER,

    /** Dose-bearing sentences were removed; the remainder is genuine guidance. */
    REDACTED_ANSWER,

    /** Nothing usable survived the dose restriction, or the question was entirely about dosing. */
    DEFERRAL,
}

/**
 * Severity of a detected emergency.
 *
 * Deliberately an enum with an ordering rather than a boolean. ADR 0004 grants the
 * Emergency Override only to [CRITICAL]; a HIGH alert must not switch the dose
 * restriction off. "Severe pain" is HIGH, and treating it as critical would let the
 * most common phrasing of a dose question bypass the control entirely.
 */
enum class EmergencySeverity(val level: Int) {
    NONE(0),
    MEDIUM(1),
    HIGH(2),
    CRITICAL(3);

    /** True only for CRITICAL, which is the sole severity that overrides the dose boundary. */
    val overridesDoseBoundary: Boolean get() = this == CRITICAL

    companion object {
        fun fromLabel(value: String?): EmergencySeverity = when (value?.lowercase()) {
            "critical" -> CRITICAL
            "high" -> HIGH
            "medium", "moderate" -> MEDIUM
            else -> NONE
        }
    }
}

/**
 * The outcome of applying the client safety layer to a response.
 */
data class SafetyResult(
    val text: String,
    val kind: AnswerKind,
    val emergency: EmergencySeverity,
    val doseFindings: List<DoseFinding> = emptyList(),
    val emergencyOverrodeDose: Boolean = false,
) {
    val isDoseRestricted: Boolean
        get() = kind == AnswerKind.REDACTED_ANSWER || kind == AnswerKind.DEFERRAL
}

/**
 * A dose-shaped span found in a response.
 */
data class DoseFinding(
    val reason: String,
    val pattern: String,
    val snippet: String,
)
