package org.inventcures.pallisahayak.data.repository

/**
 * Removes direct identifiers from free text before it is written to disk.
 *
 * Mirrors `study_logging.py` on the server. The duplication is deliberate: text
 * is scrubbed on the device before storage *and* again on the server before
 * aggregation, so neither side has to trust the other to have done it.
 *
 * What this cannot do is remove personal names. There is no reliable way to spot
 * a name in Marathi or Tamil without a gazetteer of people, and a false positive
 * would corrupt clinical text. So [Scrubbed] carries whether anything was found,
 * and the caller records that names may survive, so an auditor is not misled about
 * what the store holds.
 */
object TextScrubber {

    private val PATTERNS = listOf(
        Pattern("aadhaar", Regex("""\b\d{4}\s?\d{4}\s?\d{4}\b""")),
        // Indian mobile numbers, with or without the +91 prefix.
        Pattern("phone", Regex("""(?<![\d.])(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}\b""")),
        Pattern("email", Regex("""\b[\w.+-]+@[\w-]+\.[\w.]+\b""")),
        Pattern("abha", Regex("""\b\d{2}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{2}\b""")),
        Pattern("url", Regex("""\bhttps?://\S+\b""")),
        // Catch-all for bare digit runs. The patterns above only match well-formed
        // numbers, so an 11-digit string or a malformed one would otherwise reach
        // the database intact. Runs last so recognised formats are labelled first.
        // Word characters only in the guards: a full stop after a number is normal
        // punctuation, and excluding it let exactly those numbers through.
        Pattern("digits", Regex("""(?<![\w])\+?\d[\d\s-]{7,}\d(?![\w])""")),
    )

    private data class Pattern(val label: String, val regex: Regex)

    data class Scrubbed(
        val text: String,
        val labels: List<String>,
    ) {
        val wasScrubbed: Boolean get() = labels.isNotEmpty()

        /**
         * True when structured identifiers were found, meaning the text is
         * trustworthy for analysis. False is the common case and says nothing
         * about names.
         */
        val isClean: Boolean get() = labels.isEmpty()
    }

    fun scrub(text: String?): Scrubbed {
        if (text.isNullOrEmpty()) return Scrubbed("", emptyList())

        val found = mutableListOf<String>()
        // Non-null because the empty and null cases returned above. Without the
        // explicit type, reassignment through the loop makes Kotlin widen this back
        // to String? and every call site then fails to compile.
        var out: String = text
        for (pattern in PATTERNS) {
            if (pattern.regex.containsMatchIn(out)) {
                out = pattern.regex.replace(out, "[${pattern.label.uppercase()}_REDACTED]")
                found.add(pattern.label)
            }
        }
        return Scrubbed(out, found.distinct())
    }

    /**
     * Over-firing would destroy the corpus the study's topic analysis depends on,
     * so these have to survive untouched.
     */
    private val FALSE_POSITIVES = listOf(
        Regex("""\b\d+\s*(?:chapter|section|page|step|grade|level|site)s?\b""", RegexOption.IGNORE_CASE),
        Regex("""\b(?:age|aged)\s*\d+\b""", RegexOption.IGNORE_CASE),
        Regex("""\b\d+\s*(?:mm|cm)\s*(?:diameter|width|length|depth)\b""", RegexOption.IGNORE_CASE),
        Regex("""\b\d+\s*(?:hours?|hrs?|minutes?|mins?|days?|weeks?)\b""", RegexOption.IGNORE_CASE),
    )

    /**
     * Numeric clinical phrasing that is not an identifier.
     *
     * Used by the repository before recording, so a study question like "turn the
     * patient every 2 hours" is not stored as `[DIGITS_REDACTED]`.
     */
    fun looksLikeIdentifier(candidate: String): Boolean {
        val digits = Regex("""[\d\s+-]{4,}""").find(candidate)?.value.orEmpty()
        if (digits.isBlank()) return false
        if (FALSE_POSITIVES.any { it.containsMatchIn(candidate) }) return false
        return digits.any { it.isDigit() }
    }
}
