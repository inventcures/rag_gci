package org.inventcures.pallisahayak.safety

import com.google.common.truth.Truth.assertThat
import java.io.File
import org.junit.jupiter.api.Test

/**
 * The client dose vocabulary is the server's, and clean.
 *
 * The Android detector drifted from `dose_detector.py` once already: its unit
 * words for Punjabi, Gujarati and Bengali mixed scripts and could never match
 * real text, and it lacked every word ticket 01 added. This reads the server's
 * vocabulary from its source and fails on any difference, so the next change on
 * one side cannot go unnoticed on the other.
 */
class DoseVocabularyTest {

    @Test
    fun `every vocabulary word is in a single script and none is corrupted`() {
        val broken = DoseVocabulary.all().filter { word ->
            '\uFFFD' in word || scriptsOf(word).size > 1
        }
        assertThat(broken).isEmpty()
    }

    @Test
    fun `the Indic vocabulary is the server's, word for word`() {
        val source = serverDetector().readText()
        assertThat(DoseVocabulary.HI_NUMBER_WORDS.toSet())
            .isEqualTo(pythonTuple(source, "_HI_NUMBER_WORDS"))
        assertThat(DoseVocabulary.HI_COUNT_NOUNS.toSet())
            .isEqualTo(pythonTuple(source, "_HI_COUNT_NOUNS"))
        assertThat(DoseVocabulary.UNIT_WORDS.mapValues { it.value.toSet() })
            .isEqualTo(pythonDict(source, "UNIT_WORDS"))
        assertThat(DoseVocabulary.INDIC_FREQUENCY_WORDS.mapValues { it.value.toSet() })
            .isEqualTo(pythonDict(source, "_INDIC_FREQUENCY_WORDS"))
    }

    private fun scriptsOf(word: String): Set<Character.UnicodeScript> =
        word.codePoints().toArray()
            .filter { it >= 128 && (Character.isLetter(it) || Character.getType(it).let { t ->
                t == Character.NON_SPACING_MARK.toInt() || t == Character.COMBINING_SPACING_MARK.toInt()
            }) }
            .map { Character.UnicodeScript.of(it) }
            .filter { it != Character.UnicodeScript.INHERITED && it != Character.UnicodeScript.COMMON }
            .toSet()

    /** dose_detector.py, found by walking up from the module directory. */
    private fun serverDetector(): File {
        var dir: File? = File(System.getProperty("user.dir") ?: ".").absoluteFile
        while (dir != null && !File(dir, "dose_detector.py").exists()) dir = dir.parentFile
        checkNotNull(dir) { "dose_detector.py not found above ${System.getProperty("user.dir")}" }
        return File(dir, "dose_detector.py")
    }

    /** The source of one top-level assignment, with comment lines removed. */
    private fun block(source: String, name: String): String {
        val start = Regex("(?m)^$name\\b[^=]*=").find(source)
            ?: error("$name not found in dose_detector.py")
        val rest = source.substring(start.range.last + 1)
        val end = Regex("(?m)^[)}]").find(rest) ?: error("$name is not closed")
        return rest.substring(0, end.range.first)
            .lines()
            .filterNot { it.trimStart().startsWith("#") }
            .joinToString("\n")
    }

    private val quoted = Regex("\"([^\"]*)\"")

    private fun pythonTuple(source: String, name: String): Set<String> =
        quoted.findAll(block(source, name)).map { it.groupValues[1] }.toSet()

    private fun pythonDict(source: String, name: String): Map<String, Set<String>> =
        Regex("\"(\\w+)\":\\s*\\(([^)]*)\\)").findAll(block(source, name)).associate { entry ->
            entry.groupValues[1] to quoted.findAll(entry.groupValues[2]).map { it.groupValues[1] }.toSet()
        }
}
