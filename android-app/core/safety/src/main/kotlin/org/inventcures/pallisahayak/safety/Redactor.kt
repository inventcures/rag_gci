package org.inventcures.pallisahayak.safety

/**
 * Splits a response into independently removable units and drops the dose-bearing
 * ones.
 *
 * Sentence-level rather than whole-response: a general question about a medicine
 * should not lose its useful half because one sentence contained a dose. Measured
 * against the evaluation corpus, every response that crossed the boundary retained
 * enough surviving content to be worth keeping, so discarding wholesale would have
 * thrown away good guidance for nothing.
 */
object Redactor {

    /** Below this, the survivor is an orphaned clause and a clean deferral reads better. */
    private const val MIN_REMAINING_CHARS = 80

    private val HEADING = Regex("^[\\s#*\\-–—]*[^\\n:]{0,60}:\\s*$")

    private val SENTENCE_SPLIT = Regex("(?<=[.!?।!?])\\s+")
    private val LIST_MARKER = Regex("^((?:[-•*]|\\d+[.)])\\s+)")

    fun redact(text: String): Triple<String, List<DoseFinding>, Int> {
        val units = splitUnits(text)
        val kept = mutableListOf<String>()
        val findings = mutableListOf<DoseFinding>()
        var removed = 0

        for (unit in units) {
            if (unit.isBlank()) {
                kept.add(unit)
                continue
            }
            val unitFindings = DoseDetector.detect(unit)
            if (unitFindings.isNotEmpty()) {
                findings.addAll(unitFindings)
                removed++
            } else {
                kept.add(unit)
            }
        }

        val collapsed = dropOrphanHeadings(kept).joinToString("\n").trim()
        val deduped = findings.distinctBy { it.pattern.lowercase() }
        return Triple(collapsed, deduped, removed)
    }

    /** True when enough survived for the result to be worth showing. */
    fun isWorthKeeping(redacted: String): Boolean = redacted.length >= MIN_REMAINING_CHARS

    private fun splitUnits(text: String): List<String> {
        val units = mutableListOf<String>()
        for (line in text.split("\n")) {
            val stripped = line.trim()
            if (stripped.isEmpty()) {
                units.add("")
                continue
            }
            // A list marker is re-attached to its first sentence so removing one
            // numbered item does not leave an orphaned "1." behind.
            val marker = LIST_MARKER.find(stripped)?.value ?: ""
            val body = if (marker.isNotEmpty()) stripped.substring(marker.length) else stripped
            val sentences = SENTENCE_SPLIT.split(body).map { it.trim() }.filter { it.isNotEmpty() }
            if (sentences.isEmpty()) {
                units.add(stripped)
                continue
            }
            units.add(marker + sentences.first())
            units.addAll(sentences.drop(1))
        }
        return units
    }

    /**
     * Remove a heading whose entire block beneath it was redacted.
     *
     * A heading is orphaned when the next non-blank unit is another heading or
     * nothing at all, meaning everything it introduced is gone.
     */
    private fun dropOrphanHeadings(units: List<String>): List<String> {
        val out = units.toMutableList()
        var index = 0
        while (index < out.size) {
            if (out[index].isNotBlank() && HEADING.matches(out[index])) {
                var scan = index + 1
                while (scan < out.size && out[scan].isBlank()) scan++
                val gone = scan >= out.size ||
                    (out[scan].isNotBlank() && HEADING.matches(out[scan]))
                if (gone) {
                    repeat(scan - index) { out.removeAt(index) }
                    continue
                }
            }
            index++
        }
        return collapseBlankRuns(out)
    }

    private fun collapseBlankRuns(units: List<String>): List<String> {
        val out = mutableListOf<String>()
        for (unit in units) {
            if (unit.isBlank() && out.isNotEmpty() && out.last().isBlank()) continue
            out.add(unit)
        }
        return out
    }
}
