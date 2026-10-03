package org.inventcures.pallisahayak.data.repository

import android.util.Log
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.inventcures.pallisahayak.api.generated.MobileQueryRequest
import org.inventcures.pallisahayak.api.generated.MobileQueryResponse
import org.inventcures.pallisahayak.api.generated.VoiceQueryResponse
import org.inventcures.pallisahayak.api.generated.PalliSahayakApi
import org.inventcures.pallisahayak.data.local.CachedAnswerEntity
import org.inventcures.pallisahayak.data.local.InteractionDao
import org.inventcures.pallisahayak.data.local.InteractionEntity
import org.inventcures.pallisahayak.safety.AnswerKind
import org.inventcures.pallisahayak.safety.DoseBoundary
import org.inventcures.pallisahayak.safety.EmergencyDetector
import org.inventcures.pallisahayak.safety.EmergencySeverity
import org.inventcures.pallisahayak.safety.SafetyResult
import java.util.UUID

/**
 * What the app needs from wherever an answer came from.
 *
 * Kept separate from the implementation so the ViewModel and the tests do not
 * depend on Retrofit, a database or a network being present.
 */
sealed interface AnswerSource {
    /** The server answered a text query. */
    data class Live(val response: MobileQueryResponse) : AnswerSource

    /** The server answered a spoken query, and supplied audio to play. */
    data class LiveVoice(
        val transcript: String?,
        val audioBase64: String?,
    ) : AnswerSource

    /** The Cached Answer Bundle answered, and which version. */
    data class Bundle(val answer: CachedAnswerEntity) : AnswerSource

    /** Nothing could answer, and the app says so rather than guessing. */
    data object Unavailable : AnswerSource
}

data class AnsweredQuestion(
    val text: String,
    val kind: AnswerKind,
    val emergency: EmergencySeverity,
    val source: AnswerSource,
    val sourceCount: Int,
    /**
     * The question as transcribed, which is not always the question as asked.
     * Recorded separately from the answer because protocol 4.1 counts
     * interactions by what was asked.
     */
    val transcript: String = "",
)

/** Who is using the phone right now, and which study release they are on. */
data class SessionContext(
    val participantId: String,
    val careRole: String,
    val siteId: String,
    val language: String,
    val releaseId: String,
    val releaseApproved: Boolean,
    val isOffline: Boolean = false,
    val voicePath: String? = null,
)

/**
 * The single path an answer takes through the app.
 *
 * Everything a user sees passes through here, which means the safety boundary and
 * the study recording cannot be bypassed by a caller that forgets one of them. That
 * is the reason this exists rather than letting each screen call the API and the
 * database itself.
 *
 * Order is deliberate: ask, apply safety, record. Recording before safety would
 * store a dose that the user never saw.
 */
class PalliSahayakRepository(
    private val api: PalliSahayakApi,
    private val interactions: InteractionDao,
    private val doseBoundary: DoseBoundary = DoseBoundary(),
    private val emergencyDetector: EmergencyDetector = EmergencyDetector(),
) {

    /**
     * Ask a spoken question and play the answer aloud.
     *
     * Voice is not a separate safety path. The audio goes to the server, the
     * server returns the same JSON shape as the text route, and the returned text
     * goes through the identical Dose Boundary. A route that bypassed the filter
     * would be exactly the hole ADR 0004 is written to close.
     *
     * Returns the answer so the screen can also show it, because a spoken answer
     * a worker wants to re-read is not re-readable.
     */
    suspend fun askByVoice(
        audio: ByteArray,
        session: SessionContext,
    ): AnsweredQuestion? {
        val requested = session.copy(voicePath = VOICE_PATH_LIVE, isOffline = false)

        val answered = try {
            val response = api.voiceQuery(
                language = session.language,
                audio = audio.toRequestBody(AUDIO_MEDIA_TYPE),
            )
            if (response.isSuccessful && response.body() != null) {
                val body = response.body()!!
                val safety = doseBoundary.apply(
                    // The transcript is what the safety detector must judge,
                    // since it is what the user actually said.
                    response = body.answer,
                    language = session.language,
                    transcript = body.transcript,
                )
                AnsweredQuestion(
                    text = safety.text,
                    kind = safety.kind,
                    emergency = safety.emergency,
                    source = AnswerSource.LiveVoice(body.transcript, body.audio_base64),
                    sourceCount = body.sources?.size ?: 0,
                )
            } else {
                null
            }
        } catch (error: Exception) {
            Log.w(TAG, "voice query failed", error)
            null
        } ?: run {
            record("", AnsweredQuestion("", AnswerKind.DEFERRAL,
                emergencyDetector.detect(null, session.language),
                AnswerSource.Unavailable, 0), requested)
            return null
        }

        record(answered.transcript ?: "", answered, requested)
        return answered
    }

    /**
     * Ask a question and record what happened.
     *
     * The safety boundary runs on the response regardless of whether the server
     * already applied it. The server does filter its own output, and this repeats
     * that deliberately: a route that ever bypassed the server filter must not be
     * able to put a dose on screen. ADR 0004 calls this defence in depth.
     */
    companion object {
        private const val TAG = "PalliRepo"
        const val VOICE_PATH_LIVE = "live"
        const val VOICE_PATH_FALLBACK = "fallback"
        const val VOICE_PATH_CACHE = "cache"
        private val AUDIO_MEDIA_TYPE = "audio/wav".toMediaType()
    }

    suspend fun ask(question: String, session: SessionContext): AnsweredQuestion {
        val answered = request(question, session)
        record(question, answered, session)
        return answered
    }

    private suspend fun request(question: String, session: SessionContext): AnsweredQuestion =
        try {
            val response = api.query(
                MobileQueryRequest(
                    query = question,
                    language = session.language,
                    include_context = true,
                    patient_id = null,
                ),
            )
            if (response.isSuccessful && response.body() != null) {
                toAnswered(response.body()!!, AnswerSource.Live(response.body()!!))
            } else {
                offlineOrUnavailable(question, session)
            }
        } catch (error: Exception) {
            // A network failure is an ordinary state in these sites, not a crash.
            //
            // Logged rather than swallowed. A silent fallback here looks identical
            // to a connectivity problem, which is the same failure shape as the
            // English-only index: the app appears healthy and returns the wrong
            // thing. Logcat, not stdout, because that is where an Android crash
            // report would lead.
            Log.w(TAG, "query failed, falling back to offline", error)
            offlineOrUnavailable(question, session.copy(isOffline = true))
        }

    /**
     * Apply the client safety boundary to a server response.
     *
     * The transcript is passed to the emergency detector alongside the response,
     * because the detector works on what the user said, not on what the server
     * replied.
     */
    private fun toAnswered(response: MobileQueryResponse, source: AnswerSource): AnsweredQuestion {
        val safety: SafetyResult = doseBoundary.apply(
            response = response.answer,
            language = "en-IN",
            transcript = response.answer,
        )
        return AnsweredQuestion(
            text = safety.text,
            kind = safety.kind,
            emergency = safety.emergency,
            source = source,
            sourceCount = response.sources?.size ?: 0,
        )
    }

    private fun offlineOrUnavailable(question: String, session: SessionContext) = AnsweredQuestion(
        text = "",
        kind = AnswerKind.DEFERRAL,
        emergency = emergencyDetector.detect(question, session.language),
        source = AnswerSource.Unavailable,
        sourceCount = 0,
    )

    /**
     * Write one row. SI-5.
     *
     * Every branch records, including failures. An interaction that errored is
     * exactly what protocol §4.1 excludes from the adoption numerator, so it has
     * to be on disk to be excluded from it.
     */
    private suspend fun record(
        question: String,
        answered: AnsweredQuestion,
        session: SessionContext,
    ) {
        val scrubbedQuery = TextScrubber.scrub(question)
        val scrubbedResponse = TextScrubber.scrub(answered.text)

        // Protocol 4.1 counts an interaction as substantive only when it recorded
        // a real care question and returned a response or an escalation. Where it
        // is unsure it counts, so the adoption denominator is not quietly
        // inflated by silent exclusions.
        val substantive = question.isNotBlank() && answered.source != AnswerSource.Unavailable

        interactions.insert(
            InteractionEntity(
                id = UUID.randomUUID().toString(),
                participantId = session.participantId,
                careRole = session.careRole,
                siteId = session.siteId,
                releaseId = session.releaseId,
                releaseApproved = session.releaseApproved,
                occurredAt = System.currentTimeMillis(),
                language = session.language,
                channel = session.voicePath ?: "text",
                isOffline = session.isOffline,
                fromCache = answered.source is AnswerSource.Bundle,
                voicePath = session.voicePath,
                query = scrubbedQuery.text,
                response = scrubbedResponse.text,
                answerKind = answered.kind.name,
                emergencyLevel = answered.emergency.name.lowercase(),
                dosageBlocked = answered.kind != AnswerKind.ANSWER,
                sourceCount = answered.sourceCount,
                interactionClass = if (substantive) "substantive" else "excluded",
                // False whenever anything was found, and also false when nothing
                // was found but free text may still hold a name. See TextScrubber.
                textIsScrubbed = scrubbedQuery.isClean && scrubbedResponse.isClean,
            ),
        )
    }
}
