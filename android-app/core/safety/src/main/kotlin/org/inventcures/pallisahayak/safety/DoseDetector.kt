package org.inventcures.pallisahayak.safety

/**
 * Detects dose-shaped text in a response.
 *
 * Mirrors the server-side guard in `dosage_guard.py`. The duplication is
 * deliberate and is the defence in depth ADR 0004 calls for: the server filters
 * every response it generates, and the client filters again, because a route that
 * ever bypasses the server filter must not be able to put a dose on screen.
 *
 * Indic numerals are normalised before matching. A dose written as "१० मि.ग्रा."
 * or "๑๐ มิลลิกรัม" is exactly as much a dose as "10 mg", and a regex over \d alone
 * misses all of them.
 */
object DoseDetector {

    private val INDIC_DIGITS = mapOf(
        '०' to '0', '१' to '1', '२' to '2', '३' to '3', '४' to '4',
        '५' to '5', '६' to '6', '७' to '7', '८' to '8', '९' to '9',
        '౦' to '0', '౧' to '1', '౨' to '2', '౩' to '3', '౪' to '4',
        '౫' to '5', '౬' to '6', '౭' to '7', '౮' to '8', '౯' to '9',
        '೦' to '0', '೧' to '1', '೨' to '2', '೩' to '3', '೪' to '4',
        '೫' to '5', '೬' to '6', '೭' to '7', '೮' to '8', '೯' to '9',
        '൦' to '0', '൧' to '1', '൨' to '2', '൩' to '3', '൪' to '4',
        '൫' to '5', '൬' to '6', '൭' to '7', '൮' to '8', '൯' to '9',
        '੦' to '0', '੧' to '1', '੨' to '2', '੩' to '3', '੪' to '4',
        '੫' to '5', '੬' to '6', '੭' to '7', '੮' to '8', '੯' to '9',
        // Thai. Absent from the first draft, so a dose written as
        // "๑๐ มิลลิกรัม" passed straight through the guard.
        '๐' to '0', '๑' to '1', '๒' to '2', '๓' to '3', '๔' to '4',
        '๕' to '5', '๖' to '6', '๗' to '7', '๘' to '8', '๙' to '9',
        '૦' to '0', '૧' to '1', '૨' to '2', '૩' to '3', '૪' to '4',
        '૫' to '5', '૬' to '6', '૭' to '7', '૮' to '8', '૯' to '9',
        '௦' to '0', '௧' to '1', '௨' to '2', '௩' to '3', '௪' to '4',
        '௫' to '5', '௬' to '6', '௭' to '7', '௮' to '8', '௯' to '9',
        '০' to '0', '১' to '1', '২' to '2', '৩' to '3', '৪' to '4',
        '৫' to '5', '৬' to '6', '৭' to '7', '৮' to '8', '৯' to '9',
        '໐' to '0', '໑' to '1', '໒' to '2', '໓' to '3', '໔' to '4',
        '໕' to '5', '໖' to '6', '໗' to '7', '໘' to '8', '໙' to '9',
    )

    private const val UNIT_ALTERNATION =
        "(?:mg|mcg|µg|ug|gm|g|ml|ML|IU|cc|%|units?)"

    private const val INDIC_UNITS =
        "(?:मि\\.?\\s?ग्रा\\.?|मिलीग्रा\\.?|ग्राम|मि\\.?\\s?ली\\.?|मिली" +
            "|ಮಿ\\.?\\s?ಗ್ರಾ\\.?|ಗ್ರಾಂ|ಮಿ\\.?\\s?ಲೀ" +
            "|மி\\.?\\s?கி\\.?|மிலி\\.?\\s?கிராம்|மி\\.?\\s?லி\\.?" +
            "|మి\\.?\\s?గ్రా\\.?|గ్రామ్|మి\\.?\\s?లీ" +
            "|ਮਿਗ੍ਰਾ|ਗ੍ਰਾਮ|ਮਿਲੀ" +
            "|મિ\\.?\\s?ગ્રા\\.?|ગ્રામ|મિ\\.?\\s?લી" +
            "|ମିଗ୍ରା|ଗ୍ରାମ|ମିଲୀ" +
            "|മിഗ്രാ|ഗ്രാം|മിലി" +
            "|ਮਿਲੀਗ੍ਰਾਮ|ਮਿਲੀਲੀਟਰ" +
            "|มิลลิกรัม|มิลลิลิตร)"

    private const val FRACTION_DOSE =
        "(?:half|quarter|third|one|a|an|\\d+(?:\\.\\d+)?)\\s+" +
            "(?:tablet|tab|tablets|capsule|cap|capsules|pill|pills|" +
            "dose|drop|drops|spoonful|teaspoon|tsp|tablespoon|tbsp|syrup|sachet|satchet)"

    private const val FREQUENCY =
        "(?:once|twice|thrice|four\\s+times|three\\s+times)\\s+" +
            "(?:a\\s+|per\\s+|every\\s+)?(?:day|daily|night|morning|evening|" +
            "bedtime|meals?|mealtime|week|hourly|hours)" +
            "|\\bevery\\s+\\d+\\s*(?:hours?|hrs?|minutes?|mins?|days?)\\b" +
            "|\\b\\d+\\s*times\\s+(?:a|per)\\s+(?:day|daily|week)\\b" +
            "|\\bprn\\b|\\bas\\s+needed\\b|\\bstat\\b|\\bod\\b|\\bpo\\b|\\bIV\\b|\\bIM\\b|\\bSC\\b"

    private const val INDIC_FREQUENCY =
        "(?:दिन|रोज़|रोज|दिनों|सुबह|शाम|रात|बार)" +
            "|(?:ದಿನ|ದೀನ|ಬಾರ)" +
            "|(?:தினம்|தின|நாள்|வாரம்|முறை)" +
            "|(?:రోజు|రోజ|తరువాత|సార్లు)" +
            "|(?:ਰੋਜ਼|ਰੋਜ|ਵਾਰ)" +
            "|(?:રોજ|દિવસ|વાર)" +
            "|(?:ଦିନ|ରୋଜ)"

    private const val TITRATION =
        "\\btitrate\\b|\\btitrating\\b|\\bstep\\s*up\\b|\\bstep\\s*down\\b" +
            "|\\bescalate\\b|\\bincrease\\s+the\\s+(?:dose|dosage)\\b" +
            "|\\bdecrease\\s+the\\s+(?:dose|dosage)\\b|\\bstart\\s+at\\b.*\\b(?:mg|ml)\\b" +
            "|\\bmaximum\\s+dose\\b|\\bmax\\s+dose\\b"

    private const val START_STOP =
        "\\bstart\\s+(?:the\\s+)?\\w+\\s+(?:at\\s+)?\\d" +
            "|\\bstop\\s+(?:the\\s+)?\\w+\\b|\\bdiscontinue\\b" +
            "|\\bswitch\\s+to\\b|\\bsubstitute\\s+with\\b"

    // No trailing word boundary on the unit alternation: Indic units commonly end
    // in a full stop or a combining mark, where a boundary would fail to match.
    private val NUMERIC_DOSE = Regex("\\d+(?:\\.\\d+)?\\s*(?:$UNIT_ALTERNATION|$INDIC_UNITS)", RegexOption.IGNORE_CASE)
    private val FRACTION = Regex("\\b$FRACTION_DOSE\\b", RegexOption.IGNORE_CASE)
    private val FREQUENCY_RE = Regex("(?:$FREQUENCY)|(?:$INDIC_FREQUENCY)", RegexOption.IGNORE_CASE)
    private val TITRATION_RE = Regex(TITRATION, RegexOption.IGNORE_CASE)
    private val START_STOP_RE = Regex(START_STOP, RegexOption.IGNORE_CASE)

    /**
     * Non-dose numbers that must not trip the guard.
     *
     * Without these the guard would fire on "Chapter 5", "2 hour intervals" and
     * "1-2 hourly", and over-firing destroys the tool: a family asking about
     * turning every two hours would be told to raise it at the next consultation.
     */
    private val FALSE_POSITIVES = listOf(
        Regex("\\b\\d+\\s*(?:chapter|section|page|step|grade|level|site)s?\\b", RegexOption.IGNORE_CASE),
        Regex("\\b(?:age|aged)\\s*\\d+\\b", RegexOption.IGNORE_CASE),
        Regex("\\b\\d+\\s*(?:mm|cm)\\s*(?:diameter|width|length|depth)\\b", RegexOption.IGNORE_CASE),
    )

    private val PATTERNS = listOf(
        "numeric_dose" to NUMERIC_DOSE,
        "fraction_dose" to FRACTION,
        "frequency" to FREQUENCY_RE,
        "titration" to TITRATION_RE,
        "start_stop" to START_STOP_RE,
    )

    fun normalise(text: String): String =
        text.map { INDIC_DIGIT_TO_NORMAL[it] ?: it }.joinToString("")

    private val INDIC_DIGIT_TO_NORMAL: Map<Char, Char> =
        INDIC_DIGITS.entries.associate { (from, to) -> from to to }

    fun detect(text: String?): List<DoseFinding> {
        if (text.isNullOrBlank()) return emptyList()

        val normalised = normalise(text)
        val findings = mutableListOf<DoseFinding>()
        val seen = mutableSetOf<String>()

        for ((reason, pattern) in PATTERNS) {
            for (match in pattern.findAll(normalised)) {
                val value = match.value.trim()
                if (value.lowercase() in seen) continue
                val start = (match.range.first - 40).coerceAtLeast(0)
                val end = (match.range.last + 41).coerceAtMost(normalised.length)
                if (isFalsePositive(normalised.substring(start, end))) continue
                seen.add(value.lowercase())
                findings.add(DoseFinding(reason, value, normalised.substring(start, end).trim()))
            }
        }
        return findings
    }

    private fun isFalsePositive(sentence: String): Boolean =
        FALSE_POSITIVES.any { it.containsMatchIn(sentence) }

    fun containsDose(text: String?): Boolean = detect(text).isNotEmpty()
}
