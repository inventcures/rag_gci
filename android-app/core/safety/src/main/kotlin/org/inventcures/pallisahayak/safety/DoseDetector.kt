package org.inventcures.pallisahayak.safety

/**
 * Detects dose-shaped text in a response.
 *
 * Mirrors the server's one dose detector, `dose_detector.py`, pattern for
 * pattern and with its vocabulary ([DoseVocabulary]). The duplication is
 * deliberate and is the defence in depth ADR 0004 calls for: the server filters
 * every response it generates, and the client filters again, because a route
 * that ever bypasses the server filter must not be able to put a dose on screen.
 *
 * What counts as a dose (one reason per family, the names the study rows store):
 *
 *     numeric_dose   a number, in digits or in English or Hindi words, followed
 *                    by a unit: "mg", "milligrams", or the unit word of any
 *                    Supported Language in its own script
 *     fraction_dose  a count of a dose form: "half a tablet", "2 puffs", "३ गोली"
 *     frequency      a schedule: "twice daily", "every four hours", "4-hourly",
 *                    "q4h", "BD", and the Indic day and time words
 *     titration      dose changes: "titrate", "step up", "maximum dose"
 *     start_stop     start, stop and switch decisions
 *
 * A few phrases contain a number without being a dose ("aged 70", "chapter 5",
 * "Stage IV"). Each exception covers only its own span: a dose elsewhere in the
 * same sentence is still a dose.
 *
 * Indic words match whole words only: "बार" is a schedule word, "बारे" ("about")
 * is not. Digits are not word letters, so "3गोली" still reads as a count.
 *
 * Written with explicit character classes rather than `\d`, `\w` and `\b`,
 * because those mean ASCII on the JVM, where the unit tests run, and Unicode on
 * Android, where ICU implements the regex engine. `\p{Nd}` matches every Indic
 * digit on both, so "१० मि.ग्रा." is caught with no digit normalisation.
 */
object DoseDetector {

    /** Python's `\w`: a Unicode letter, number or underscore. */
    private const val WORD = "[\\p{L}\\p{N}_]"

    /** Python's `\b`: a position between a word character and anything else. */
    private const val B = "(?:(?<=$WORD)(?!$WORD)|(?<!$WORD)(?=$WORD))"

    /** Any decimal digit, in any script. */
    private const val DIGIT = "\\p{Nd}"

    private const val SNIPPET_CONTEXT = 40

    private val LETTER_OR_MARK: String = letterOrMarkClass()
    private val INDIC_START = "(?<!$LETTER_OR_MARK)"
    private val INDIC_END = "(?!$LETTER_OR_MARK)"

    // Latin-script units, longest first so a spelled-out unit is reported whole.
    private val LATIN_UNITS = listOf(
        "microgrammes?", "micrograms?", "milligrammes?", "milligrams?",
        "millilitres?", "milliliters?", "grammes?", "grams?",
        "international\\s+units?", "units?",
        "mcg", "µg", "μg", "ug", "mg", "gm", "g", "ml", "iu", "cc", "gy",
    )

    // Counts of a dose form. These may follow "a"/"an"/"half"; the others need a
    // number, so "it can be given as an injection" is not a dose.
    private val COUNT_NOUNS_AFTER_ARTICLE = listOf(
        "tablets?", "tabs?", "capsules?", "caps?", "pills?", "doses?",
        "drops?", "spoonfuls?", "teaspoons?(?:ful)?s?", "tsp",
        "tablespoons?(?:ful)?s?", "tbsp", "syrup", "sachets?", "satchets?",
    )
    private val COUNT_NOUNS_AFTER_NUMBER = COUNT_NOUNS_AFTER_ARTICLE + listOf(
        "puffs?", "sprays?", "inhalations?", "patch(?:es)?", "ampoules?",
        "ampules?", "vials?", "injections?", "suppositor(?:y|ies)",
        "lozenges?",
    )

    private val EN_WORD = alternation(DoseVocabulary.EN_NUMBER_WORDS)
    private val EN_NUMBER_PHRASE =
        "$B$EN_WORD(?:(?:\\s+|-)(?:and\\s+)?(?:a\\s+)?$EN_WORD)*$B"
    private val HI_NUMBER_PHRASE = wholeWords(DoseVocabulary.HI_NUMBER_WORDS)

    // Not preceded by a letter, so "SpO2 drop" or "B12" does not start a dose.
    private const val DIGITS = "(?<![A-Za-z])$DIGIT+(?:[.,/]$DIGIT+)?"
    private val NUMBER = "(?:$DIGITS|$EN_NUMBER_PHRASE|$HI_NUMBER_PHRASE)"

    private val LATIN_UNIT = LATIN_UNITS.joinToString("|", "(?:", ")") + "(?![A-Za-z])"

    // Unit words are not closed on the right: they follow a number, which already
    // anchors them, and they take suffixes (Telugu plural "-లు", Tamil case
    // endings) that a closing boundary would cut off.
    private val INDIC_UNIT = alternation(DoseVocabulary.UNIT_WORDS.values.flatten())

    private val NUMERIC_DOSE_RE = "$NUMBER\\s*(?:$LATIN_UNIT|%|$INDIC_UNIT)"

    private val COUNT_DOSE_RE =
        "$B(?:half|quarter|third|one|a|an)\\s+(?:of\\s+a\\s+)?" +
            COUNT_NOUNS_AFTER_ARTICLE.joinToString("|", "(?:", ")") + B +
            "|$NUMBER\\s+(?:of\\s+a\\s+)?" +
            COUNT_NOUNS_AFTER_NUMBER.joinToString("|", "(?:", ")") + B +
            "|(?:$DIGITS|$HI_NUMBER_PHRASE)\\s*" + wholeWords(DoseVocabulary.HI_COUNT_NOUNS)

    private val INTERVAL = "(?:$DIGIT+(?:\\s*(?:-|–|to)\\s*$DIGIT+)?|$EN_NUMBER_PHRASE)"

    private val FREQUENCY_RE =
        "$B(?:once|twice|thrice|three\\s+times|four\\s+times)\\s+" +
            "(?:a\\s+|per\\s+|every\\s+)?(?:day|daily|night|morning|evening|" +
            // No trailing boundary: "twice weekly" must match as "twice week".
            "bedtime|meals?|mealtime|week|hourly|hours)" +
            "|${B}every\\s+$INTERVAL\\s*(?:hours?|hrs?|minutes?|mins?|days?)$B" +
            "|$B$INTERVAL\\s*times\\s+(?:a|per|each)\\s+(?:day|daily|week)$B" +
            "|$B$INTERVAL\\s*[-–]?\\s*hourly$B" +
            "|${B}q\\s?$DIGIT+\\s?h(?:rs?|ourly)?$B" +
            "|${B}prn$B|${B}as\\s+needed$B|${B}stat$B|${B}od$B|${B}po$B|${B}IV$B|${B}IM$B|${B}SC$B" +
            "|${B}nocte$B|${B}mane$B" +
            // Upper case only: "bid" and "bd" are ordinary words in lower case.
            "|(?-i:$B(?:BD|BID|TDS|TID|QDS|QID)$B)" +
            "|" + wholeWords(DoseVocabulary.INDIC_FREQUENCY_WORDS.values.flatten())

    private const val TITRATION_RE =
        "${B}titrate$B|${B}titrating$B|${B}step\\s*up$B|${B}step\\s*down$B" +
            "|${B}escalate$B|${B}increase\\s+the\\s+(?:dose|dosage)$B" +
            "|${B}decrease\\s+the\\s+(?:dose|dosage)$B|${B}start\\s+at$B.*$B(?:mg|ml)$B" +
            "|${B}maximum\\s+dose$B|${B}max\\s+dose$B"

    private const val START_STOP_RE =
        "${B}start\\s+(?:the\\s+)?$WORD+\\s+(?:at\\s+)?$DIGIT" +
            "|${B}stop\\s+(?:the\\s+)?$WORD+$B|${B}discontinue$B" +
            "|${B}switch\\s+to$B|${B}substitute\\s+with$B"

    private val PATTERNS: List<Pair<String, Regex>> = listOf(
        "numeric_dose" to Regex(NUMERIC_DOSE_RE, RegexOption.IGNORE_CASE),
        "fraction_dose" to Regex(COUNT_DOSE_RE, RegexOption.IGNORE_CASE),
        "frequency" to Regex(FREQUENCY_RE, RegexOption.IGNORE_CASE),
        "titration" to Regex(TITRATION_RE, RegexOption.IGNORE_CASE),
        "start_stop" to Regex(START_STOP_RE, RegexOption.IGNORE_CASE),
    )

    /**
     * Spans that contain a number but are not doses. A match is excused only
     * where it overlaps one of these spans; the rest of the sentence is still
     * checked, so "For patients aged 70, give morphine 10 mg" is still a dose.
     */
    private val NOT_A_DOSE: List<Regex> = listOf(
        Regex(
            "$B$DIGIT+\\s*(?:chapter|section|page|step|grade|level|site visit|site)s?$B" +
                "|$B(?:chapter|section|page|step|grade|level|stage|class|type)\\s*(?:$DIGIT+|[IV]+)[a-c]?$B",
            RegexOption.IGNORE_CASE,
        ),
        Regex("$B(?:age|aged)\\s*(?:$DIGIT+|$EN_NUMBER_PHRASE)", RegexOption.IGNORE_CASE),
        Regex("$B$DIGIT+\\s*(?:mm|cm)\\s*(?:diameter|width|length|depth)$B", RegexOption.IGNORE_CASE),
    )

    /**
     * Every dose-shaped span in [text], strongest reason first, one per distinct
     * matched text. Empty for null or empty text.
     */
    fun detect(text: String?): List<DoseFinding> {
        if (text.isNullOrEmpty()) return emptyList()

        val excused = NOT_A_DOSE.flatMap { pattern ->
            pattern.findAll(text).map { it.range.first to it.range.last + 1 }.toList()
        }
        val findings = mutableListOf<DoseFinding>()
        val seen = mutableSetOf<String>()
        for ((reason, pattern) in PATTERNS) {
            for (match in pattern.findAll(text)) {
                val start = match.range.first
                val end = match.range.last + 1
                if (excused.any { (s, e) -> start < e && s < end }) continue
                if (!seen.add(match.value.lowercase().trim())) continue
                val snippet = text.substring(
                    (start - SNIPPET_CONTEXT).coerceAtLeast(0),
                    (end + SNIPPET_CONTEXT).coerceAtMost(text.length),
                ).trim()
                findings.add(DoseFinding(reason, match.value, snippet))
            }
        }
        return findings
    }

    fun containsDose(text: String?): Boolean = detect(text).isNotEmpty()

    /** Indic words that match only where no letter or combining mark adjoins. */
    private fun wholeWords(words: Collection<String>): String =
        INDIC_START + alternation(words) + INDIC_END

    /**
     * Literal words as a regex alternation, longest first. A full stop in an
     * abbreviation is often dropped or followed by a space ("मि.ग्रा.",
     * "मि. ग्रा.", "मि.ग्रा"), and a space may be dropped ("एम जी", "एमजी").
     */
    private fun alternation(words: Collection<String>): String =
        words.toSet().sortedByDescending { it.length }.joinToString("|", "(?:", ")") { word ->
            buildString {
                for (ch in word) {
                    when {
                        ch == '.' -> append("\\.?\\s?")
                        ch == ' ' -> append("\\s?")
                        ch in REGEX_META -> append('\\').append(ch)
                        else -> append(ch)
                    }
                }
            }
        }

    private const val REGEX_META = "\\^$|?*+()[]{}"

    /**
     * A character class of every letter and combining mark in the Latin and
     * Indic blocks (U+0900 to U+0DFF), plus the zero-width joiners that sit
     * inside Indic words. Built the way the server builds it, from the Unicode
     * categories, because a vowel sign such as "ि" is a mark rather than a
     * letter and must still count as part of a word.
     */
    private fun letterOrMarkClass(): String {
        val inWord = setOf(
            Character.UPPERCASE_LETTER, Character.LOWERCASE_LETTER, Character.TITLECASE_LETTER,
            Character.MODIFIER_LETTER, Character.OTHER_LETTER,
            Character.NON_SPACING_MARK, Character.COMBINING_SPACING_MARK,
        ).map { it.toInt() }.toSet()
        val ranges = mutableListOf<IntRange>()
        for (code in 0x0900 until 0x0E00) {
            if (Character.getType(code) !in inWord) continue
            val last = ranges.lastOrNull()
            if (last != null && last.last + 1 == code) {
                ranges[ranges.size - 1] = last.first..code
            } else {
                ranges.add(code..code)
            }
        }
        val body = ranges.joinToString("") { range ->
            if (range.first == range.last) hex(range.first) else "${hex(range.first)}-${hex(range.last)}"
        }
        return "[A-Za-z\\u200C\\u200D$body]"
    }

    private fun hex(code: Int): String = "\\u%04X".format(code)
}
