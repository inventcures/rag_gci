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
import org.inventcures.pallisahayak.data.model.CareRole
import org.inventcures.pallisahayak.data.model.Language
import org.inventcures.pallisahayak.data.offline.OfflineAnswerResolver
import org.inventcures.pallisahayak.data.offline.OfflineResolution
import org.inventcures.pallisahayak.data.sync.SyncOutbox
import org.inventcures.pallisahayak.data.sync.offlineInteractionLog
import org.inventcures.pallisahayak.safety.AnswerKind
import org.inventcures.pallisahayak.safety.DeferralCopy
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
    /** True when the server could not be reached, whatever answered instead. */
    val offline: Boolean = false,
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
) {
    /**
     * The Active Role as the server's `active_role` field, so its study row
     * records who is speaking. A value that is no Care Role is sent as absent
     * (the server then uses the registered role) rather than refused there.
     */
    fun activeRoleOnTheWire(): String? = CareRole.ALL.firstOrNull { it.wireName == careRole }?.wireName
}

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
    /** The Cached Answer Bundle, asked when the server cannot be reached. */
    private val bundle: OfflineAnswerResolver? = null,
    /** Where an interaction the server never saw waits to be synced. */
    private val outbox: SyncOutbox? = null,
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
                // The Active Role: the server's study row records who is speaking.
                active_role = session.activeRoleOnTheWire(),
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
                    alert = EmergencySeverity.fromLabel(body.emergency_level),
                )
                AnsweredQuestion(
                    text = safety.text,
                    kind = safety.kind,
                    emergency = safety.emergency,
                    source = AnswerSource.LiveVoice(body.transcript, body.audio_base64),
                    sourceCount = body.sources?.size ?: 0,
                    // The transcript is what the user actually said, which is not
                    // always what was asked, and protocol 4.1 counts by what was
                    // asked.
                    transcript = body.transcript,
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

        record(answered.transcript, answered, requested)
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
                    active_role = session.activeRoleOnTheWire(),
                ),
            )
            val body = response.body()
            if (response.isSuccessful && body != null) {
                toAnswered(body, question, session)
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
     * The Emergency level is the server's for the question, kept, together with
     * the on-device check of the question in the participant's language; the
     * higher wins. The answer text never decides it: an answer that mentions
     * "unconscious" must not lift the Dose Boundary for a routine question.
     */
    private fun toAnswered(
        response: MobileQueryResponse,
        question: String,
        session: SessionContext,
    ): AnsweredQuestion {
        val safety: SafetyResult = doseBoundary.apply(
            response = response.answer,
            language = session.language,
            transcript = question,
            alert = EmergencySeverity.fromLabel(response.emergency_level),
        )
        return AnsweredQuestion(
            text = safety.text,
            kind = safety.kind,
            emergency = safety.emergency,
            source = AnswerSource.Live(response),
            sourceCount = response.sources?.size ?: 0,
        )
    }

    /**
     * The server did not answer: the Cached Answer Bundle answers if it holds the
     * question, in the participant's language, and otherwise the app says plainly
     * that it needs a connection (ADR 0005).
     *
     * A bundled answer passes the Dose Boundary again, and is never shown as more
     * of an answer than the kind it was stored with: a stored Deferral stays one.
     * The Emergency level comes from the on-device check of the question, since
     * there is no server level offline.
     */
    private suspend fun offlineOrUnavailable(question: String, session: SessionContext): AnsweredQuestion {
        val emergency = emergencyDetector.detect(question, session.language)
        val cached = findInBundle(question, session.language)
            ?: return AnsweredQuestion(
                text = "",
                kind = AnswerKind.DEFERRAL,
                emergency = emergency,
                source = AnswerSource.Unavailable,
                sourceCount = 0,
                offline = session.isOffline,
            )

        val safety = doseBoundary.apply(
            response = cached.response,
            language = session.language,
            transcript = question,
            alert = emergency,
        )
        val stored = AnswerKind.entries.firstOrNull { it.name == cached.answerKind } ?: AnswerKind.DEFERRAL
        val kind = if (stored.ordinal > safety.kind.ordinal) stored else safety.kind
        return AnsweredQuestion(
            text = if (kind == AnswerKind.DEFERRAL && safety.kind != AnswerKind.DEFERRAL) {
                DeferralCopy.forLanguage(session.language)
            } else {
                safety.text
            },
            kind = kind,
            emergency = safety.emergency,
            source = AnswerSource.Bundle(cached),
            sourceCount = 0,
            offline = session.isOffline,
        )
    }

    private suspend fun findInBundle(question: String, language: String): CachedAnswerEntity? {
        val resolver = bundle ?: return null
        val supported = Language.fromTag(language) ?: return null
        return (resolver.resolve(question, supported) as? OfflineResolution.Found)?.answer
    }

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

        val row = InteractionEntity(
            id = UUID.randomUUID().toString(),
            participantId = session.participantId,
            careRole = session.careRole,
            siteId = session.siteId,
            releaseId = session.releaseId,
            releaseApproved = session.releaseApproved,
            occurredAt = System.currentTimeMillis(),
            language = session.language,
            channel = session.voicePath ?: "text",
            isOffline = session.isOffline || answered.offline,
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
        )
        interactions.insert(row)

        // The server never saw an interaction it could not be reached for, so it
        // waits in the outbox until a sync acknowledges it. Without its text: the
        // study counts it, and free text could still hold a name.
        if (row.isOffline) outbox?.enqueue(offlineInteractionLog(row))
    }
}
