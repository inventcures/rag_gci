package org.inventcures.pallisahayak.safety

/**
 * The client dose boundary and emergency ordering.
 *
 * Applies, in order:
 *   1. Emergency detection. A CRITICAL alert outranks the dose boundary and the
 *      response passes through untouched.
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

    fun apply(
        response: String?,
        language: String = "en-IN",
        transcript: String? = null,
    ): SafetyResult {
        val severity = emergencyDetector.detect(transcript, language)

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
