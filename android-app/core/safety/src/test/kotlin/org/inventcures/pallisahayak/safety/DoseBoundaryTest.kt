package org.inventcures.pallisahayak.safety

import com.google.common.truth.Truth.assertThat
import org.junit.jupiter.api.Test

/**
 * Tests for the client safety invariants.
 *
 * A regression in any of these is a protocol breach rather than a bug, so each is
 * named for the invariant it protects and fails with that name in the message.
 *
 * These run on the JVM with no Android dependency, because the invariants are
 * clinical controls and must be testable without an emulator.
 */
class DoseBoundaryTest {

    private val boundary = DoseBoundary()

    // -- SI-1: the dose boundary -------------------------------------------

    @Test
    fun `SI-1 a response containing a dose is not returned verbatim`() {
        val result = boundary.apply(
            "Give morphine 10 mg orally every 4 hours and titrate upward.",
            language = "en-IN",
        )
        assertThat(result.isDoseRestricted).isTrue()
        assertThat(result.text).doesNotContain("10 mg")
    }

    @Test
    fun `SI-1 general drug information is answered normally`() {
        val result = boundary.apply(
            "Morphine is a strong opioid used for severe cancer pain.",
            language = "en-IN",
        )
        assertThat(result.kind).isEqualTo(AnswerKind.ANSWER)
        assertThat(result.text).contains("strong opioid")
    }

    @Test
    fun `SI-1 over-firing destroys the tool and is treated as a defect`() {
        // Chapter references, intervals and grades are not doses. If these trip
        // the guard, a family asking how often to turn a patient gets a refusal.
        val safe = listOf(
            "See Chapter 5 of the handbook for details.",
            "Turn and reposition the patient every 1-2 hour intervals.",
            "This intervention is supported by Grade C evidence.",
            "Age 84 and under 2 years of care.",
        )
        for (text in safe) {
            assertThat(DoseDetector.containsDose(text)).isFalse()
        }
    }

    @Test
    fun `SI-1 doses written in Indic numerals are caught`() {
        val samples = listOf("१० मि.ग्रा. दें", "১০ মিলিগ্রাম", "๑๐ มิลลิกรัม")
        for (text in samples) {
            assertThat(DoseDetector.containsDose(text)).isTrue()
        }
    }

    // -- SI-2: redacted answer, not silent omission --------------------------

    @Test
    fun `SI-2 useful content survives when only one sentence carries a dose`() {
        val result = boundary.apply(
            "Morphine is a strong opioid used for severe cancer pain. " +
                "It is the gold standard for severe pain control. " +
                "Give 10 mg orally every 4 hours.",
            language = "en-IN",
        )
        assertThat(result.kind).isEqualTo(AnswerKind.REDACTED_ANSWER)
        assertThat(result.text).contains("strong opioid")
        assertThat(result.text).contains("gold standard")
        assertThat(result.text).doesNotContain("10 mg")
    }

    @Test
    fun `SI-2 a redacted answer is never reported as an answer`() {
        val result = boundary.apply(
            "Morphine is a strong opioid. Give 10 mg every 4 hours.",
            language = "en-IN",
        )
        assertThat(result.kind).isNotEqualTo(AnswerKind.ANSWER)
    }

    @Test
    fun `SI-2 an entirely dosing answer becomes a deferral`() {
        val result = boundary.apply("Give 10 mg morphine now.", language = "en-IN")
        assertThat(result.kind).isEqualTo(AnswerKind.DEFERRAL)
    }

    @Test
    fun `SI-2 a deferral names next steps and the emergency number`() {
        val result = boundary.apply("Give 10 mg morphine now.", language = "en-IN")
        assertThat(result.text).contains("next consultation")
        assertThat(result.text).contains("108")
    }

    @Test
    fun `SI-2 a deferral never implies a person has been contacted`() {
        val result = boundary.apply("Give 10 mg morphine now.", language = "en-IN")
        val text = result.text.lowercase()
        for (claim in listOf("will contact", "will call", "within 5 minutes", "within 10 minutes", "a counselor will")) {
            assertThat(text).doesNotContain(claim)
        }
    }

    @Test
    fun `SI-2 redaction is verified and fails closed`() {
        // Whatever the redactor does, the returned text must contain no dose.
        val hostile = listOf(
            "Constipation relief: 1. Take senna 15 mg at night. Give plenty of fluids.",
            "Actions: 1. Rest. 2. Titrate morphine upward. 3. Keep a record.",
        )
        for (text in hostile) {
            val result = boundary.apply(text, language = "en-IN")
            assertThat(DoseDetector.containsDose(result.text)).isFalse()
        }
    }

    @Test
    fun `SI-2 localised deferral is used when available`() {
        val hindi = boundary.apply("Give 10 mg morphine.", language = "hi-IN")
        assertThat(hindi.kind).isEqualTo(AnswerKind.DEFERRAL)
        assertThat(hindi.text).contains("108")
    }

    @Test
    fun `SI-2 unsupported language falls back to English rather than failing`() {
        val result = boundary.apply("Give 10 mg morphine.", language = "zu-IN")
        assertThat(result.kind).isEqualTo(AnswerKind.DEFERRAL)
        assertThat(result.text).contains("next consultation")
    }

    // -- SI-4: emergency override is CRITICAL only ---------------------------

    @Test
    fun `SI-4 a critical emergency is never withheld`() {
        val result = boundary.apply(
            "Call 108 immediately. The patient cannot breathe.",
            language = "en-IN",
            transcript = "patient is choking and cannot breathe",
        )
        assertThat(result.emergency).isEqualTo(EmergencySeverity.CRITICAL)
        assertThat(result.isDoseRestricted).isFalse()
        assertThat(result.emergencyOverrodeDose).isTrue()
        assertThat(result.text).contains("108")
    }

    @Test
    fun `SI-4 severe pain is high and does not override the dose boundary`() {
        // The single most important test here. If "severe pain" were treated as
        // critical, every dose question phrased as pain would bypass the guard.
        val result = boundary.apply(
            "Give morphine 10 mg orally every 4 hours.",
            language = "en-IN",
            transcript = "patient has severe pain, what dose of morphine?",
        )
        assertThat(result.emergency).isEqualTo(EmergencySeverity.HIGH)
        assertThat(result.isDoseRestricted).isTrue()
        assertThat(result.text).doesNotContain("10 mg")
    }

    @Test
    fun `SI-4 only critical severity overrides`() {
        assertThat(EmergencySeverity.CRITICAL.overridesDoseBoundary).isTrue()
        assertThat(EmergencySeverity.HIGH.overridesDoseBoundary).isFalse()
        assertThat(EmergencySeverity.MEDIUM.overridesDoseBoundary).isFalse()
        assertThat(EmergencySeverity.NONE.overridesDoseBoundary).isFalse()
    }

    @Test
    fun `SI-4 severity survives parsing a server label`() {
        assertThat(EmergencySeverity.fromLabel("critical")).isEqualTo(EmergencySeverity.CRITICAL)
        assertThat(EmergencySeverity.fromLabel("high")).isEqualTo(EmergencySeverity.HIGH)
        assertThat(EmergencySeverity.fromLabel("none")).isEqualTo(EmergencySeverity.NONE)
        assertThat(EmergencySeverity.fromLabel(null)).isEqualTo(EmergencySeverity.NONE)
        assertThat(EmergencySeverity.fromLabel("garbage")).isEqualTo(EmergencySeverity.NONE)
    }

    @Test
    fun `SI-4 highest severity wins when several patterns match`() {
        val detector = EmergencyDetector()
        val severity = detector.detect("severe pain and cannot breathe", "en-IN")
        assertThat(severity).isEqualTo(EmergencySeverity.CRITICAL)
    }

    @Test
    fun `SI-4 detection works in the languages the server cannot cover`() {
        // Server-side emergency detection exists only in en, hi, bn, ta, gu. For the
        // other supported languages this detector is the only one that runs.
        // These strings are the phrases the detector actually carries. Asserting
        // against invented translations would fail for a reason unrelated to the
        // property under test.
        val detector = EmergencyDetector()
        assertThat(detector.detect("ಉಸಿರಾಟ ಇಲ್ಲ", "kn-IN")).isEqualTo(EmergencySeverity.CRITICAL)
        assertThat(detector.detect("ശ്വാസം ഇല്ല", "ml-IN")).isEqualTo(EmergencySeverity.CRITICAL)
        assertThat(detector.detect("ಪ್ರಜ್ಞೆ തಪ್ಪಿದ", "kn-IN")).isEqualTo(EmergencySeverity.CRITICAL)
        assertThat(detector.detect("ગળું દબાઈ જવું", "gu-IN")).isEqualTo(EmergencySeverity.CRITICAL)
    }

    @Test
    fun `SI-4 an unknown language falls back to English rather than to none`() {
        val detector = EmergencyDetector()
        assertThat(detector.detect("choking", "zu-IN")).isEqualTo(EmergencySeverity.CRITICAL)
    }

    @Test
    fun `SI-4 empty input is not an emergency`() {
        val detector = EmergencyDetector()
        assertThat(detector.detect("", "en-IN")).isEqualTo(EmergencySeverity.NONE)
        assertThat(detector.detect(null, "en-IN")).isEqualTo(EmergencySeverity.NONE)
    }

    // -- general behaviour ---------------------------------------------------

    @Test
    fun `an empty response is passed through untouched`() {
        val result = boundary.apply("", language = "en-IN")
        assertThat(result.kind).isEqualTo(AnswerKind.ANSWER)
        assertThat(result.isDoseRestricted).isFalse()
    }

    @Test
    fun `a null response does not crash`() {
        val result = boundary.apply(null, language = "en-IN")
        assertThat(result.kind).isEqualTo(AnswerKind.ANSWER)
    }
}
