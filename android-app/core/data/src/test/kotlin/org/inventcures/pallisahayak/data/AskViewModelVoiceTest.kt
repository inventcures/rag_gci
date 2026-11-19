package org.inventcures.pallisahayak.data

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.runTest
import org.inventcures.pallisahayak.data.repository.FakeApi
import org.inventcures.pallisahayak.data.repository.FakeInteractionDao
import org.inventcures.pallisahayak.data.repository.PalliSahayakRepository
import org.inventcures.pallisahayak.data.repository.SessionContext
import org.inventcures.pallisahayak.data.voice.VoiceEngine
import org.inventcures.pallisahayak.data.voice.WavAudio
import org.inventcures.pallisahayak.safety.AnswerKind
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Base64

/**
 * The server's spoken answer is played only when the on-device Dose Boundary
 * passes the answer whole (round-3 final review, finding 10).
 *
 * Before this the ViewModel played the server's audio whatever the device's
 * boundary decided: the text on screen was redacted or deferred, and the
 * speaker still said the dose. The audio is the server's rendering of the
 * unredacted answer, so when the device changes the text the audio no longer
 * matches it and is not played or kept for replay.
 *
 * The real repository and Dose Boundary, with the server and the speaker faked.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class AskViewModelVoiceTest {

    private val api = FakeApi()
    private val repository = PalliSahayakRepository(api = api, interactions = FakeInteractionDao())
    private val speaker = RecordingVoiceEngine()

    private val session = SessionContext(
        participantId = "3f2a8c14-9b7e-4d21-8a55-6c1e0d3b7f92",
        careRole = "asha_worker",
        siteId = "CCHRC",
        language = "en-IN",
        releaseId = "study-1@abc1234",
        releaseApproved = true,
    )

    /** Sixteen bytes of "PCM", base64 encoded as the server sends it. */
    private val serverAudio = ByteArray(16) { it.toByte() }
    private val serverAudioBase64 = Base64.getEncoder().encodeToString(serverAudio)

    private fun TestScope.viewModel() = AskViewModel(
        repository = repository,
        session = session,
        voiceEngine = speaker,
        scope = CoroutineScope(UnconfinedTestDispatcher(testScheduler)),
    )

    private fun TestScope.askByVoice(): AskViewModel {
        val vm = viewModel()
        vm.onMicPressed()
        vm.onMicReleased()
        testScheduler.advanceUntilIdle()
        return vm
    }

    @Test
    fun `a spoken answer the device passes whole is played`() = runTest {
        api.voice("what is morphine used for?", "Morphine is a strong opioid used for severe cancer pain.",
            audioBase64 = serverAudioBase64)

        val state = askByVoice().state.value

        assertEquals(AnswerKind.ANSWER, state.answerKind)
        assertEquals(1, speaker.played.size)
        assertArrayEquals(serverAudio, speaker.played.single())
        assertTrue(state.canReplay)
    }

    @Test
    fun `a spoken answer the device restricts for a dose is not played`() = runTest {
        api.voice("what dose of morphine?", "Give morphine 10 mg every 4 hours.",
            audioBase64 = serverAudioBase64)

        val state = askByVoice().state.value

        assertNotEquals(AnswerKind.ANSWER, state.answerKind)
        assertFalse(state.answer.contains("10 mg"))
        assertTrue("the server's audio was played: ${speaker.played.size}", speaker.played.isEmpty())
        assertNull(state.spokenAnswer)
        assertFalse(state.canReplay)
    }

    @Test
    fun `a spoken answer with one dosed sentence redacted is not played either`() = runTest {
        api.voice(
            "what is morphine used for?",
            "Morphine is a strong opioid used for severe cancer pain. " +
                "It is the gold standard for severe pain control. Give 10 mg every 4 hours.",
            audioBase64 = serverAudioBase64,
        )

        val state = askByVoice().state.value

        assertEquals(AnswerKind.REDACTED_ANSWER, state.answerKind)
        assertTrue(state.answer.contains("strong opioid"))
        assertTrue(speaker.played.isEmpty())
        assertNull(state.spokenAnswer)
    }

    // -- the Stop button's state (Android review, R1) ----------------------------

    private fun spokenAnswer() = api.voice(
        "what is morphine used for?", "Morphine is a strong opioid used for severe cancer pain.",
        audioBase64 = serverAudioBase64,
    )

    @Test
    fun `the answer is speaking while it plays and not once it ends`() = runTest {
        spokenAnswer()

        val vm = askByVoice()
        assertTrue("Stop must be offered while the answer plays", vm.state.value.isSpeaking)
        assertFalse(vm.state.value.canReplay)

        speaker.finishPlaying()
        assertFalse("Stop must go once the answer has ended", vm.state.value.isSpeaking)
        assertTrue(vm.state.value.canReplay)
    }

    @Test
    fun `stopping an answer leaves it replayable, and the replay is speaking again`() = runTest {
        spokenAnswer()
        val vm = askByVoice()

        vm.stopSpeaking()
        assertFalse(vm.state.value.isSpeaking)

        vm.replayAnswer()
        assertTrue(vm.state.value.isSpeaking)
        assertEquals(2, speaker.played.size)

        speaker.finishPlaying()
        assertFalse(vm.state.value.isSpeaking)
    }

    @Test
    fun `audio the phone cannot play is not shown as speaking`() = runTest {
        spokenAnswer()
        speaker.playable = false

        val state = askByVoice().state.value

        assertFalse(state.isSpeaking)
        assertTrue("the answer is still on screen", state.answer.contains("strong opioid"))
    }

    // -- the microphone permission (Android review, R1) ---------------------------

    @Test
    fun `without the permission the screen is told to ask, and nothing is recorded`() = runTest {
        speaker.permitted = false
        val vm = viewModel()

        assertTrue(vm.needsMicrophonePermission())

        vm.onMicrophonePermission(granted = false)
        assertEquals(AskProblem.MICROPHONE_NEEDED, vm.state.value.problem)
        assertFalse(vm.state.value.isListening)
    }

    @Test
    fun `once the permission is granted the notice goes and the mic is not asked for again`() = runTest {
        speaker.permitted = false
        val vm = viewModel()
        vm.onMicrophonePermission(granted = false)

        speaker.permitted = true
        vm.onMicrophonePermission(granted = true)

        assertNull(vm.state.value.problem)
        assertFalse(vm.needsMicrophonePermission())
    }

    // -- what is uploaded (Android review, R1) -------------------------------------

    @Test
    fun `the recording is uploaded as a named WAV file`() = runTest {
        spokenAnswer()

        askByVoice()

        val part = api.voiceParts.single()
        val disposition = part.headers?.get("Content-Disposition").orEmpty()
        assertTrue(disposition, disposition.contains("name=\"audio\""))
        // Without a filename FastAPI reads the part as text and answers 422.
        assertTrue(disposition, disposition.contains("filename=\"question.wav\""))

        val body = okio.Buffer().also { part.body.writeTo(it) }.readByteArray()
        assertEquals("RIFF", String(body, 0, 4, Charsets.US_ASCII))
        assertEquals("WAVE", String(body, 8, 4, Charsets.US_ASCII))
        // The recording engine's fake gives 16,000 bytes of PCM.
        assertEquals(WavAudio.HEADER_BYTES + 16_000, body.size)
        assertEquals("audio/wav", part.body.contentType().toString())
    }
}

/** Stands in for the microphone and speaker; keeps every buffer it is asked to play. */
private class RecordingVoiceEngine : VoiceEngine {
    val played = mutableListOf<ByteArray>()
    private var listening = false
    private var speaking = false

    override fun canRecord(): Boolean = permitted
    override val isListening: Boolean get() = listening
    override val isSpeaking: Boolean get() = speaking

    override fun startRecording() {
        listening = true
    }

    override fun stopRecording(): ByteArray? {
        if (!listening) return null
        listening = false
        return ByteArray(16_000)
    }

    /** Set false to stand in for a phone that cannot play the file. */
    var playable = true

    /** Set false to stand in for a refused microphone permission. */
    var permitted = true

    private var onFinished: (() -> Unit)? = null

    override fun speak(audio: ByteArray, onFinished: () -> Unit): Boolean {
        if (!playable) return false
        played.add(audio)
        speaking = true
        this.onFinished = onFinished
        return true
    }

    /** The audio reaches its end, as the real player reports it. */
    fun finishPlaying() {
        speaking = false
        onFinished?.also { onFinished = null }?.invoke()
    }

    override fun stopSpeaking() {
        speaking = false
        // As the real engine: a stopped playback never calls back.
        onFinished = null
    }
}
