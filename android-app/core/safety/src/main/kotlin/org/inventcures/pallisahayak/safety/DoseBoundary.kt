package org.inventcures.pallisahayak.safety

/**
 * The client dose boundary and emergency ordering.
 *
 * Applies, in order:
 *   1. Emergency detection, on the question and never on the response. A CRITICAL
 *      alert outranks the dose boundary and the response passes through untouched.
 *      The alert is the higher of the level the caller passes in (the server's,
 *      which the server computes from the question) and the on-device detector's
 *      reading of the question in the participant's language. An answer that
 *      merely mentions "unconscious" or "call 108" must not lift the boundary:
 *      that is the response-text fallback ticket 02 removed from the server.
 *   2. Dose restriction. Dose-bearing sentences are removed and the remainder kept;
 *      if too little survives, the whole answer becomes a deferral.
 *   3. Verification. The redacted text is re-checked, and anything still matching
 *      escalates to a full deferral.
 *
 * Step 3 is the reason this can be trusted: redaction must never be the thing
 * that lets a dose through, so it fails closed rather than open.
 */
class DoseBoundary(
    private val emergencyDetector: EmergencyDetector = EmergencyDetector(),
) {

    /**
     * Apply the boundary to [response].
     *
     * [transcript] is the question as the participant asked it (typed, or as
     * transcribed), in [language]; never the response. [alert] is the Emergency
     * level already known for that question, such as the server's.
     */
    fun apply(
        response: String?,
        language: String = "en-IN",
        transcript: String? = null,
        alert: EmergencySeverity = EmergencySeverity.NONE,
    ): SafetyResult {
        val detected = emergencyDetector.detect(transcript, language)
        val severity = if (alert.level >= detected.level) alert else detected

        if (severity.overridesDoseBoundary) {
            return SafetyResult(
                text = response.orEmpty(),
                kind = AnswerKind.ANSWER,
                emergency = severity,
                emergencyOverrodeDose = true,
            )
        }

        val findings = DoseDetector.detect(response)
        if (findings.isEmpty()) {
            return SafetyResult(text = response.orEmpty(), kind = AnswerKind.ANSWER, emergency = severity)
        }

        val (redacted, redactedFindings, _) = Redactor.redact(response.orEmpty())
        if (Redactor.isWorthKeeping(redacted) && !DoseDetector.containsDose(redacted)) {
            return SafetyResult(
                text = redacted + DeferralCopy.redactionNote(language),
                kind = AnswerKind.REDACTED_ANSWER,
                emergency = severity,
                doseFindings = redactedFindings,
            )
        }

        // Either nothing survived, or the redaction still contains a dose. In both
        // cases the safe outcome is the same: refuse the whole thing.
        return SafetyResult(
            text = DeferralCopy.forLanguage(language),
            kind = AnswerKind.DEFERRAL,
            emergency = severity,
            doseFindings = findings,
        )
    }
}
