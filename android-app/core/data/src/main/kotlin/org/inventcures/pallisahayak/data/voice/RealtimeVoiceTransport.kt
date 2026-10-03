package org.inventcures.pallisahayak.data.voice

import android.util.Base64
import android.util.Log
import kotlinx.coroutines.suspendCancellableCoroutine
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import org.json.JSONObject
import kotlin.coroutines.resume

/**
 * Real-time voice transport over the server's WebSocket.
 *
 * The app used to post a recording to `/query/voice` and wait. That works, but it
 * is one turn at a time with a full round trip between each, which is not a
 * conversation. This opens one socket and keeps it, so the session survives
 * across turns and the server can hold state the client does not need.
 *
 * The server applies the same safety boundary to a socket turn as to a posted
 * one, so nothing is gained here that would not also be filtered. What is gained
 * is latency and the ability to interrupt.
 */
class RealtimeVoiceTransport(
    private val client: OkHttpClient,
    private val baseWsUrl: String,
    private val releaseId: String = "",
) {

    private var socket: WebSocket? = null
    private var sessionId: String = ""

    /**
     * The language the session opened with.
     *
     * Kept because registration knows the language and the socket may not, and
     * every turn frame carries it so the server never has to guess which language
     * the recording is in.
     */
    @Volatile
    private var lastLanguage: String = "en-IN"

    @Volatile
    var lastVoicePath: String = ""
        private set

    /**
     * Open a session, or reuse the open one.
     *
     * Reconnecting per turn would put a handshake in front of every question,
     * which is exactly the latency the socket exists to avoid.
     */
    suspend fun connect(language: String): String {
        lastLanguage = language
        // onFailure nulls the socket, so a non-null socket means the session is
        // still live. Reconnecting per turn would put a handshake in front of
        // every question.
        socket?.let { return sessionId }

        val url = baseWsUrl.trimEnd('/') + "/ws/voice?language=" + language
        val request = Request.Builder().url(url).build()

        return suspendCancellableCoroutine { continuation ->
            val listener = object : WebSocketListener() {
                override fun onOpen(webSocket: WebSocket, response: Response) {
                    socket = webSocket
                    webSocket.send(
                        JSONObject()
                            .put("type", "start")
                            .put("language", language)
                            .put("release_id", releaseId)
                            .toString(),
                    )
                }

                override fun onMessage(webSocket: WebSocket, text: String) {
                    val frame = runCatching { JSONObject(text) }.getOrNull() ?: return
                    if (frame.optString("type") == "ready" && continuation.isActive) {
                        sessionId = frame.optString("session_id")
                        continuation.resume(sessionId)
                    }
                }

                override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                    Log.w(TAG, "voice socket failed", t)
                    socket = null
                }
            }
            client.newWebSocket(request, listener)
            continuation.invokeOnCancellation { close() }
        }
    }

    /**
     * Send one turn and wait for its answer.
     *
     * Resolves on the server's `response` frame. An `error` frame resolves to null
     * rather than throwing, because a turn that fails is an ordinary state in
     * these sites and the caller already reports it to the user.
     */
    suspend fun sendTurn(audio: ByteArray): VoiceReply? {
        val active = socket ?: return null

        return suspendCancellableCoroutine { continuation ->

            val listener = object : WebSocketListener() {
                override fun onMessage(webSocket: WebSocket, text: String) {
                    if (!continuation.isActive) return
                    val frame = runCatching { JSONObject(text) }.getOrNull() ?: return
                    when (frame.optString("type")) {
                        "response" -> {
                            lastVoicePath = frame.optString("voice_path")
                            continuation.resume(
                                VoiceReply(
                                    transcript = frame.optString("transcript"),
                                    text = frame.optString("text"),
                                    audioBase64 = frame.optStringOrNull("audio_base64"),
                                    answerKind = frame.optString("answer_kind", "ANSWER"),
                                    evidenceLevel = frame.optString("evidence_level", "E"),
                                    emergencyLevel = frame.optString("emergency_level", "none"),
                                    sourceCount = frame.optInt("source_count", 0),
                                    voicePath = lastVoicePath,
                                    fallbackReason = frame.optStringOrNull("fallback_reason"),
                                ),
                            )
                        }

                        "error" -> continuation.resume(null)
                    }
                }

                override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                    Log.w(TAG, "voice turn failed", t)
                    if (continuation.isActive) continuation.resume(null)
                }

                override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                    socket = null
                    if (continuation.isActive) continuation.resume(null)
                }
            }

            // A fresh listener per turn, so one turn's frames cannot resolve the
            // next turn's continuation.
            socket = active
            active.send(
                JSONObject()
                    .put("type", "audio")
                    .put("audio_base64", Base64.encodeToString(audio, Base64.NO_WRAP))
                    .put("language", lastLanguage)
                    .toString(),
            )

            continuation.invokeOnCancellation { close() }
        }
    }

    /**
     * Tell the server the user cut in.
     *
     * The server discards the half-heard answer so the next turn starts clean,
     * which matters because a user who interrupts has usually already understood
     * enough and does not want the rest.
     */
    fun interrupt() {
        socket?.send(JSONObject().put("type", "interrupt").toString())
    }

    fun close() {
        socket?.close(1000, "done")
        socket = null
    }

    private fun JSONObject.optStringOrNull(key: String): String? =
        if (isNull(key)) null else optString(key).takeIf { it.isNotEmpty() }

    private companion object {
        const val TAG = "VoiceSocket"
    }
}

/** One answered turn, as it came off the wire. */
data class VoiceReply(
    val transcript: String,
    val text: String,
    val audioBase64: String?,
    val answerKind: String,
    val evidenceLevel: String,
    val emergencyLevel: String,
    val sourceCount: Int,
    val voicePath: String,
    val fallbackReason: String?,
)
