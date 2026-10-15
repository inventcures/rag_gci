package org.inventcures.pallisahayak.data.repository

import kotlinx.coroutines.test.runTest
import org.inventcures.pallisahayak.safety.AnswerKind
import org.inventcures.pallisahayak.safety.EmergencyDetector
import org.inventcures.pallisahayak.safety.EmergencySeverity
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The app keeps the server's Emergency level for the question, and checks the
 * question itself, in the participant's language, on the device.
 *
 * Before ticket 23 the repository threw the server's level away and ran the
 * on-device detector on the server's *answer*, in English. A Bengali emergency
 * lost its CRITICAL level (no ambulance path), and an answer that merely
 * mentioned "unconscious" lifted the Dose Boundary for a routine question: the
 * same "the answer says call 108" fallback ticket 02 removed from the server.
 */
class EmergencySeverityKeptTest {

    private val api = FakeApi()
    private val interactions = FakeInteractionDao()
    private val repository = PalliSahayakRepository(api = api, interactions = interactions)

    private fun session(language: String) = SessionContext(
        participantId = "3f2a8c14-9b7e-4d21-8a55-6c1e0d3b7f92",
        careRole = "asha_worker",
        siteId = "CCHRC",
        language = language,
        releaseId = "study-1@abc1234",
        releaseApproved = true,
    )

    @Test
    fun `a Bengali emergency question with a server CRITICAL level shows the ambulance path`() = runTest {
        // "The patient is having a fit": not a phrase the on-device Bengali list
        // carries, so only the server's level can make this CRITICAL.
        api.answer(
            answer = "অবিলম্বে জরুরি পরিষেবায় (108/102) কল করুন! রোগীকে পাশ ফিরিয়ে শুইয়ে দিন।",
            emergencyLevel = "critical",
        )

        val answered = repository.ask("রোগীর খিঁচুনি হচ্ছে, কী করব?", session("bn-IN"))

        assertEquals(EmergencySeverity.CRITICAL, answered.emergency)
        assertTrue(answered.emergency.overridesDoseBoundary)
        assertTrue(answered.text.contains("108"))
        assertEquals("critical", interactions.rows.single().emergencyLevel)
    }

    @Test
    fun `the question is checked on the device in the participant's language`() = runTest {
        // The server has no Kannada emergency words and reports none.
        api.answer(answer = "Lay the patient on their side and stay with them.", emergencyLevel = "none")

        val answered = repository.ask(EmergencyDetector.KANNADA_NO_BREATH, session("kn-IN"))

        assertEquals(EmergencySeverity.CRITICAL, answered.emergency)
    }

    @Test
    fun `an answer that mentions an emergency does not lift the boundary for a routine question`() = runTest {
        api.answer(
            answer = "Constipation is common with these medicines and can be eased. " +
                "If the patient is unconscious, call 108. Give senna 15 mg at night.",
            emergencyLevel = "none",
        )

        val answered = repository.ask("How can I help with constipation?", session("en-IN"))

        assertEquals(EmergencySeverity.NONE, answered.emergency)
        assertNotEquals(AnswerKind.ANSWER, answered.kind)
        assertFalse(answered.text.contains("15 mg"))
    }

    @Test
    fun `a HIGH server level is kept but does not lift the boundary`() = runTest {
        api.answer(answer = "Give morphine 10 mg orally every 4 hours.", emergencyLevel = "high")

        val answered = repository.ask("He has severe pain, what can I do?", session("en-IN"))

        assertEquals(EmergencySeverity.HIGH, answered.emergency)
        assertNotEquals(AnswerKind.ANSWER, answered.kind)
        assertFalse(answered.text.contains("10 mg"))
    }

    @Test
    fun `the higher of the server's level and the device's is kept`() = runTest {
        // The device finds CRITICAL in the Hindi question; the server said high.
        api.answer(answer = "Call 108 now and keep the airway clear.", emergencyLevel = "high")

        val answered = repository.ask("मरीज़ बेहोश है", session("hi-IN"))

        assertEquals(EmergencySeverity.CRITICAL, answered.emergency)
    }

    @Test
    fun `a spoken emergency keeps the server's level too`() = runTest {
        api.voice(
            transcript = "রোগীর খিঁচুনি হচ্ছে",
            answer = "অবিলম্বে জরুরি পরিষেবায় (108/102) কল করুন!",
            emergencyLevel = "critical",
        )

        val answered = repository.askByVoice(ByteArray(8), session("bn-IN"))

        assertEquals(EmergencySeverity.CRITICAL, answered?.emergency)
    }
}
