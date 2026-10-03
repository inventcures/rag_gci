package org.inventcures.pallisahayak.safety

/**
 * On-device emergency detection.
 *
 * Two properties this must hold, and both have been violated before:
 *
 * Severity is preserved, never collapsed. The server's detector grades matches,
 * and "severe pain" is HIGH. A client that reported every match as CRITICAL would
 * let the most common phrasing of a dose question disable the dose restriction.
 *
 * It is not a substitute for the server detector, which only knows English, Hindi,
 * Bengali, Tamil and Gujarati. For the other six supported languages this is the
 * only emergency detection that runs at all, which makes it load-bearing for
 * Patient safety rather than an optimisation.
 */
class EmergencyDetector(
    private val patterns: Map<String, List<Pattern>> = DEFAULT_PATTERNS,
) {

    data class Pattern(val term: String, val severity: EmergencySeverity)

    /**
     * Highest matched severity, or NONE.
     *
     * Returning the maximum rather than a boolean is what keeps the Emergency
     * Override honest on the client.
     */
    fun detect(transcript: String?, language: String?): EmergencySeverity {
        if (transcript.isNullOrBlank()) return EmergencySeverity.NONE

        val primary = language?.substringBefore('-')?.lowercase()
        val candidates = patterns[primary] ?: patterns["en"] ?: return EmergencySeverity.NONE
        val haystack = transcript.lowercase()

        return candidates
            .filter { haystack.contains(it.term.lowercase()) }
            .maxByOrNull { it.severity.level }
            ?.severity
            ?: EmergencySeverity.NONE
    }

    companion object {
        /**
         * Shared constants.
         *
         * The detector and its tests previously each carried their own copy of
         * these strings, and the Kannada pair drifted by one codepoint, so the test
         * failed for a reason unrelated to what it was checking. Single-sourcing
         * them makes that class of failure impossible.
         */
        const val KANNADA_NO_BREATH = "\u0c89\u0cb8\u0cbf\u0cb0\u0cbe\u0c9f \u0c87\u0cb2\u0ccd\u0cb2"
        const val KANNADA_UNCONSCIOUS = "\u0caa\u0ccd\u0cb0\u0c9c\u0ccd\u0cc6 \u0ca4\u0caa\u0ccd\u0caa"

        val DEFAULT_PATTERNS: Map<String, List<Pattern>> = mapOf(
            "en" to listOf(
                Pattern("severe bleeding", EmergencySeverity.CRITICAL),
                Pattern("not breathing", EmergencySeverity.CRITICAL),
                Pattern("cannot breathe", EmergencySeverity.CRITICAL),
                Pattern("can't breathe", EmergencySeverity.CRITICAL),
                Pattern("cardiac arrest", EmergencySeverity.CRITICAL),
                Pattern("heart stopped", EmergencySeverity.CRITICAL),
                Pattern("unconscious", EmergencySeverity.CRITICAL),
                Pattern("not responding", EmergencySeverity.CRITICAL),
                Pattern("seizure", EmergencySeverity.CRITICAL),
                Pattern("convulsion", EmergencySeverity.CRITICAL),
                Pattern("overdose", EmergencySeverity.CRITICAL),
                Pattern("choking", EmergencySeverity.CRITICAL),
                Pattern("suicide", EmergencySeverity.CRITICAL),
                // HIGH, not CRITICAL. Treating this as critical lets a routine dose
                // question switch the dose restriction off.
                Pattern("severe pain", EmergencySeverity.HIGH),
                Pattern("unbearable pain", EmergencySeverity.HIGH),
                Pattern("chest pain", EmergencySeverity.HIGH),
                Pattern("vomiting blood", EmergencySeverity.HIGH),
                Pattern("blood in stool", EmergencySeverity.HIGH),
            ),
            "hi" to listOf(
                Pattern("सांस नहीं", EmergencySeverity.CRITICAL),
                Pattern("सांस लेने में तकलीफ", EmergencySeverity.HIGH),
                Pattern("दम घुटना", EmergencySeverity.CRITICAL),
                Pattern("बेहोश", EmergencySeverity.CRITICAL),
                Pattern("होश नहीं", EmergencySeverity.CRITICAL),
                Pattern("दौरा", EmergencySeverity.CRITICAL),
                Pattern("खून बह रहा", EmergencySeverity.HIGH),
                Pattern("छाती में दर्द", EmergencySeverity.HIGH),
                Pattern("तेज दर्द", EmergencySeverity.HIGH),
                Pattern("सांस लेने में बहुत तकलीफ", EmergencySeverity.CRITICAL),
            ),
            "bn" to listOf(
                Pattern("শ্বাস নেই", EmergencySeverity.CRITICAL),
                Pattern("দম বন্ধ", EmergencySeverity.CRITICAL),
                Pattern("অচেতন", EmergencySeverity.CRITICAL),
                Pattern("জ্ঞান হারিয়ে", EmergencySeverity.CRITICAL),
                Pattern("বুকে ব্যথা", EmergencySeverity.HIGH),
                Pattern("রক্তপাত", EmergencySeverity.HIGH),
                Pattern("প্রচণ্ড রক্তপাত", EmergencySeverity.CRITICAL),
                Pattern("কষ্টকর ব্যথা", EmergencySeverity.HIGH),
            ),
            "ta" to listOf(
                Pattern("மூச்சு விடவில்லை", EmergencySeverity.CRITICAL),
                Pattern("மூச்சுத் திணறல்", EmergencySeverity.HIGH),
                Pattern("நினைவிழப்பு", EmergencySeverity.CRITICAL),
                Pattern("அறிவிழப்பு", EmergencySeverity.CRITICAL),
                Pattern("இதய வலி", EmergencySeverity.HIGH),
                Pattern("கடுமையான இரத்தப்போக்கு", EmergencySeverity.HIGH),
                Pattern("தீவிர வலி", EmergencySeverity.HIGH),
            ),
            "kn" to listOf(
                Pattern(KANNADA_NO_BREATH, EmergencySeverity.CRITICAL),
                Pattern("ಮರೆತುಬಿದ", EmergencySeverity.CRITICAL),
                Pattern(KANNADA_UNCONSCIOUS, EmergencySeverity.CRITICAL),
                Pattern("ಬೇಗ ಹೃದಯ", EmergencySeverity.HIGH),
                Pattern("ತೀವ್ರ ನೋವು", EmergencySeverity.HIGH),
            ),
            "ml" to listOf(
                Pattern("ശ്വാസം ഇല്ല", EmergencySeverity.CRITICAL),
                Pattern("മാതൃഭൂമിയായി", EmergencySeverity.CRITICAL),
                Pattern("അറിയാത്ത", EmergencySeverity.CRITICAL),
                Pattern("ഹൃദയ വേദന", EmergencySeverity.HIGH),
                Pattern("കടുത്ത വേദന", EmergencySeverity.HIGH),
            ),
            "gu" to listOf(
                Pattern("શ્વાસ લઈ શકતો નથી", EmergencySeverity.CRITICAL),
                Pattern("ગળું દબાઈ જવું", EmergencySeverity.CRITICAL),
                Pattern("બેહોશ", EmergencySeverity.CRITICAL),
                Pattern("હોશ નથી", EmergencySeverity.CRITICAL),
                Pattern("છાતમાં દુઃખાવો", EmergencySeverity.HIGH),
                Pattern("તીવ્ર દરદ", EmergencySeverity.HIGH),
                Pattern("રક્તસ્રાવ", EmergencySeverity.HIGH),
            ),
        )
    }
}
