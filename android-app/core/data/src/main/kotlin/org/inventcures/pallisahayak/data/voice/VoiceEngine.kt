package org.inventcures.pallisahayak.data.voice

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaDataSource
import android.media.MediaPlayer
import android.media.MediaRecorder
import android.util.Log
import androidx.core.content.ContextCompat
import java.io.ByteArrayOutputStream
import java.io.IOException
import kotlin.concurrent.thread

/**
 * Records a spoken question and plays the answer aloud.
 *
 * The question is recorded as 16-bit PCM, 16 kHz, mono, the format the Android
 * specification §10.6 pins. The answer is whatever audio file the server's
 * text-to-speech returned (Sarvam sends a WAV file at its own sample rate), so
 * playback reads the format from the file rather than assuming the recording's.
 *
 * Kept behind an interface so tests can drive the voice path without a microphone.
 * Robolectric has no audio hardware, and a real microphone in a unit test is a
 * flake waiting to happen.
 */
interface VoiceEngine {

    /** True when the app holds the record permission. */
    fun canRecord(): Boolean

    /** True when recording has started and is still running. */
    val isListening: Boolean

    /** True while audio is being played. */
    val isSpeaking: Boolean

    /** Begin capturing. Does not block; call [stopRecording] to get the audio. */
    fun startRecording()

    /** Stop capturing and return 16-bit PCM mono 16 kHz, or null if nothing captured. */
    fun stopRecording(): ByteArray?

    /**
     * Play an audio file held in memory. Returns immediately; playback runs in
     * the background. False when it could not be started.
     *
     * [onFinished] is called once, when the audio has played to its end or failed
     * part-way. It is not called after [stopSpeaking], nor once a later [speak]
     * has replaced this one, so a caller can clear its "speaking" state in it
     * without clearing the state of a newer playback.
     */
    fun speak(audio: ByteArray, onFinished: () -> Unit = {}): Boolean

    /** Stop playback. Safe to call when nothing is playing. */
    fun stopSpeaking()
}

/**
 * The real implementation, using AudioRecord and MediaPlayer.
 *
 * Recording starts on a background thread because AudioRecord.read blocks, and
 * blocking the main thread in a voice-first app is how the UI stops responding
 * while the user is speaking.
 *
 * Playback goes through MediaPlayer from memory: it reads the file's own header,
 * so a 22 kHz WAV plays at 22 kHz, and the answer is never written to disk.
 */
class AndroidVoiceEngine(
    private val context: Context,
) : VoiceEngine {

    private var recorder: AudioRecord? = null
    private var captureThread: Thread? = null

    @Volatile
    private var player: MediaPlayer? = null

    @Volatile
    private var listening = false

    override val isListening: Boolean get() = listening
    override val isSpeaking: Boolean get() = player != null

    override fun canRecord(): Boolean =
        ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED

    override fun startRecording() {
        if (listening || !canRecord()) return

        val minBuffer = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
        if (minBuffer <= 0) {
            Log.w(TAG, "this device cannot record 16 kHz mono")
            return
        }

        val record = try {
            AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                minBuffer * 2,
            )
        } catch (error: SecurityException) {
            // The permission was withdrawn between the check and here.
            Log.w(TAG, "microphone refused", error)
            return
        }
        if (record.state != AudioRecord.STATE_INITIALIZED) {
            record.release()
            return
        }

        recorder = record
        listening = true
        record.startRecording()

        captureThread = thread(name = "voice-capture") {
            val buffer = ByteArray(minBuffer)
            val collected = ByteArrayOutputStream()
            // Bounded: a button held down by a pocket must not grow without limit
            // on a 2 GB phone. What was said in the first minute is still sent.
            while (listening && collected.size() < MAX_AUDIO_BYTES) {
                val read = record.read(buffer, 0, buffer.size)
                if (read > 0) collected.write(buffer, 0, read) else if (read < 0) break
            }
            pendingAudio = collected.toByteArray()
        }
    }

    /**
     * Written by the capture thread and read by [stopRecording] once it joins.
     *
     * Joined before reading, so there is no race on the buffer.
     */
    @Volatile
    private var pendingAudio: ByteArray? = null

    override fun stopRecording(): ByteArray? {
        if (!listening) return null

        listening = false
        captureThread?.join(STOP_JOIN_TIMEOUT_MS)
        captureThread = null

        runCatching { recorder?.stop() }
        recorder?.release()
        recorder = null

        val audio = pendingAudio
        pendingAudio = null
        // Below this the "question" is a tap or a cough, and sending it wastes a
        // request and records a row that protocol 4.1 would exclude anyway.
        return audio?.takeIf { it.size >= MIN_AUDIO_BYTES }
    }

    override fun speak(audio: ByteArray, onFinished: () -> Unit): Boolean {
        if (audio.isEmpty()) return false
        stopSpeaking()

        val media = MediaPlayer()
        return try {
            media.setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build(),
            )
            media.setDataSource(InMemoryAudio(audio))
            media.setOnPreparedListener { prepared -> if (player === prepared) prepared.start() }
            media.setOnCompletionListener { finished(it, onFinished) }
            media.setOnErrorListener { failed, what, extra ->
                Log.w(TAG, "playback failed ($what, $extra)")
                finished(failed, onFinished)
                true
            }
            player = media
            media.prepareAsync()
            true
        } catch (error: IOException) {
            abandon(media, error)
        } catch (error: IllegalStateException) {
            abandon(media, error)
        } catch (error: IllegalArgumentException) {
            abandon(media, error)
        }
    }

    private fun abandon(media: MediaPlayer, error: Exception): Boolean {
        Log.w(TAG, "the answer could not be played", error)
        if (player === media) player = null
        media.release()
        return false
    }

    /** The end of one playback, unless it was already stopped or replaced. */
    private fun finished(media: MediaPlayer, onFinished: () -> Unit) {
        if (player !== media) return
        player = null
        media.release()
        onFinished()
    }

    /**
     * Stop playback immediately.
     *
     * The user must be able to cut the app off mid-answer. In a home visit an
     * answer that has moved on is worse than one that stopped, because the
     * patient is waiting.
     */
    override fun stopSpeaking() {
        val media = player ?: return
        // Cleared first, so the completion or error callback of this playback
        // finds it replaced and stays quiet.
        player = null
        runCatching { media.stop() }
        media.release()
    }

    /** The answer's audio, read by MediaPlayer straight from memory. */
    private class InMemoryAudio(private val bytes: ByteArray) : MediaDataSource() {

        override fun readAt(position: Long, buffer: ByteArray, offset: Int, size: Int): Int {
            if (position >= bytes.size) return END_OF_AUDIO
            val count = minOf(size, bytes.size - position.toInt())
            System.arraycopy(bytes, position.toInt(), buffer, offset, count)
            return count
        }

        override fun getSize(): Long = bytes.size.toLong()

        override fun close() = Unit

        private companion object {
            const val END_OF_AUDIO = -1
        }
    }

    companion object {
        private const val TAG = "PalliVoice"

        /** Pinned by §10.6 and required by Sarvam. */
        const val SAMPLE_RATE = 16_000
        const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT

        /** Bytes a second at this format: 16-bit mono. */
        private const val BYTES_PER_SECOND = SAMPLE_RATE * 2

        /** Roughly a quarter second. Below it, it is a tap. */
        const val MIN_AUDIO_BYTES = BYTES_PER_SECOND / 4

        /** One minute, about 1.9 MB, which is also what one upload over 2G can carry. */
        const val MAX_AUDIO_BYTES = BYTES_PER_SECOND * 60

        const val STOP_JOIN_TIMEOUT_MS = 2_000L
    }
}
