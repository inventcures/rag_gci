package org.inventcures.pallisahayak.data

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import org.inventcures.pallisahayak.data.repository.AnsweredQuestion
import org.inventcures.pallisahayak.data.repository.AnswerSource
import org.inventcures.pallisahayak.data.repository.PalliSahayakRepository
import org.inventcures.pallisahayak.data.repository.SessionContext
import org.inventcures.pallisahayak.data.voice.VoiceEngine
import org.inventcures.pallisahayak.safety.AnswerKind
import org.inventcures.pallisahayak.safety.EmergencySeverity

/**
 * Everything the screen renders, as one immutable value.
 *
 * A single state object rather than several LiveData fields, so the screen cannot
 * show a half-updated combination such as an answer from one question next to the
 * busy flag of another.
 */
data class AskUiState(
    val draft: String = "",
    val isAsking: Boolean = false,
    val lastQuestion: String = "",
    val answer: String = "",
    val answerKind: AnswerKind = AnswerKind.ANSWER,
    val emergency: EmergencySeverity = EmergencySeverity.NONE,
    val cameFromCache: Boolean = false,
    val sourceCount: Int = 0,
    val history: List<Pair<String, String>> = emptyList(),
    val errorMessage: String? = null,

    // -- voice -------------------------------------------------------------
    /** True while the mic is held. The user needs to see this, not infer it. */
    val isListening: Boolean = false,

    /** True while the answer is being spoken aloud. */
    val isSpeaking: Boolean = false,

    /**
     * The question as transcribed, kept separately from the answer because
     * protocol 4.1 counts interactions by what was asked and a transcript is not
     * always the question as spoken.
     */
    val transcript: String = "",

    /** Held so the user can replay. A spoken answer nobody can re-read is lost. */
    val spokenAnswer: ByteArray? = null,
) {
    /** True when the last answer can be played again. */
    val canReplay: Boolean get() = spokenAnswer != null && !isSpeaking

    /**
     * True when there is nothing to show and the screen should prompt rather than
     * render an empty answer card.
     */
    val hasAnswer: Boolean get() = answer.isNotBlank()

    val isDeferral: Boolean get() = answerKind != AnswerKind.ANSWER
}

/**
 * Holds the screen's state and asks the repository.
 *
 * The ViewModel is the only place that knows an answer came from somewhere; the
 * screen only renders what it is told. Keeping the network, database and safety
 * logic out of the screen is what makes the Robolectric tests in ticket 03
 * possible without an emulator.
 */
class AskViewModel(
    private val repository: PalliSahayakRepository,
    private val session: SessionContext,
    private val voiceEngine: VoiceEngine? = null,
    /**
     * Injected so tests can drive it.
     *
     * viewModelScope is pinned to Dispatchers.Main.immediate, which does not run
     * under Robolectric whatever Main is set to. Injecting the scope makes the
     * class testable by construction rather than by fighting the dispatcher.
     *
     * Nullable with a fallback rather than defaulted to viewModelScope, because a
     * default argument cannot reference an extension property of the object being
     * constructed: the ViewModel does not exist yet at that point.
     */
    private val scope: CoroutineScope? = null,
) : ViewModel() {

    private val workScope: CoroutineScope
        get() = scope ?: viewModelScope

    private val _state = MutableStateFlow(AskUiState())
    val state: StateFlow<AskUiState> = _state.asStateFlow()

    fun onDraftChanged(value: String) {
        _state.update { it.copy(draft = value) }
    }

    /**
     * Send the draft question.
     *
     * The guard on [AskUiState.isAsking] matters on a cheap phone over a slow
     * link, where a double tap sends the same question twice and both land in the
     * study log as separate interactions.
     */
    fun submit() {
        val question = _state.value.draft.trim()
        if (question.isEmpty() || _state.value.isAsking) return

        _state.update { it.copy(isAsking = true, errorMessage = null) }

        workScope.launch {
            try {
                val answered = repository.ask(question, session)
                _state.update { current ->
                    current.copy(
                        draft = "",
                        isAsking = false,
                        lastQuestion = question,
                        answer = answered.text,
                        answerKind = answered.kind,
                        emergency = answered.emergency,
                        cameFromCache = answered.source is AnswerSource.Bundle,
                        sourceCount = answered.sourceCount,
                        errorMessage = unavailableMessage(answered),
                        history = (listOf(question to answered.text) + current.history)
                            .take(HISTORY_LIMIT),
                    )
                }
            } catch (error: Exception) {
                // The repository already absorbs network failures. Reaching here
                // means something the app did not anticipate, and a message beats
                // a crash during a home visit.
                _state.update {
                    it.copy(isAsking = false, errorMessage = "Something went wrong. Please try again.")
                }
            }
        }
    }

    /**
     * Mic pressed. Starts capture.
     *
     * Holding is the interaction. A tap produces too little audio to be a
     * question, so the button is pressed-and-held rather than tapped.
     */
    fun onMicPressed() {
        val engine = voiceEngine ?: return
        if (!engine.canRecord()) {
            _state.update {
                it.copy(errorMessage = "Microphone access is needed to ask by voice.")
            }
            return
        }
        if (_state.value.isAsking || _state.value.isSpeaking) return

        engine.startRecording()
        _state.update { it.copy(isListening = true, errorMessage = null) }
    }

    /**
     * Mic released. Sends what was captured and plays the answer.
     *
     * [AskUiState.isListening] is cleared before the request so the UI stops
     * showing "listening" the instant the user lets go, rather than after a round
     * trip over a slow link.
     */
    fun onMicReleased() {
        val engine = voiceEngine ?: return
        _state.update { it.copy(isListening = false) }

        val audio = engine.stopRecording()
        if (audio == null) {
            // Nothing was captured. Not an error worth a banner; the user simply
            // did not say anything.
            return
        }

        _state.update { it.copy(isAsking = true, errorMessage = null) }
        scope?.launch { sendVoice(audio) }
    }

    private suspend fun sendVoice(audio: ByteArray) {
        try {
            val answered = repository.askByVoice(audio, session)
            if (answered == null) {
                _state.update {
                    it.copy(
                        isAsking = false,
                        answerKind = org.inventcures.pallisahayak.safety.AnswerKind.DEFERRAL,
                        errorMessage = "Could not reach the service. Please try again.",
                    )
                }
                return
            }

            val audioToPlay = (answered.source as? AnswerSource.LiveVoice)?.audioBase64
            val pcm = audioToPlay?.let { decodeBase64(it) }

            _state.update { current ->
                current.copy(
                    isAsking = false,
                    lastQuestion = answered.transcript,
                    transcript = answered.transcript,
                    answer = answered.text,
                    answerKind = answered.kind,
                    emergency = answered.emergency,
                    cameFromCache = false,
                    sourceCount = answered.sourceCount,
                    spokenAnswer = pcm,
                    history = (listOf(answered.transcript to answered.text) + current.history)
                        .take(HISTORY_LIMIT),
                )
            }

            pcm?.let { voiceEngine?.speak(it) }
        } catch (error: Exception) {
            _state.update {
                it.copy(isAsking = false, errorMessage = "Something went wrong. Please try again.")
            }
        }
    }

    /**
     * Cut playback off.
     *
     * Exposed as a single tap because a spoken answer the user cannot interrupt
     * is the failure this app cannot afford.
     */
    fun stopSpeaking() {
        voiceEngine?.stopSpeaking()
        _state.update { it.copy(isSpeaking = false) }
    }

    /** Play the last answer again. */
    fun replayAnswer() {
        val pcm = _state.value.spokenAnswer ?: return
        voiceEngine?.stopSpeaking()
        voiceEngine?.speak(pcm)
        _state.update { it.copy(isSpeaking = true) }
    }

    private fun decodeBase64(value: String): ByteArray? =
        runCatching { android.util.Base64.decode(value, android.util.Base64.DEFAULT) }
            .getOrNull()

    fun clearError() {
        _state.update { it.copy(errorMessage = null) }
    }

    private fun unavailableMessage(answered: AnsweredQuestion): String? =
        if (answered.source is AnswerSource.Unavailable) {
            "No connection, and this question is not one of the saved ones. " +
                "Try again when there is signal."
        } else {
            null
        }

    private companion object {
        const val HISTORY_LIMIT = 20
    }
}
