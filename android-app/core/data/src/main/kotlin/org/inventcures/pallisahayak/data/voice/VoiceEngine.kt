package org.inventcures.pallisahayak.data.voice

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import androidx.core.content.ContextCompat
import java.io.ByteArrayOutputStream
import kotlin.concurrent.thread

/**
 * Records a spoken question and plays the answer aloud.
 *
 * 16-bit PCM, 16 kHz, mono. The Android specification §10.6 pins that format, and
 * it is the format Sarvam expects, so the same bytes go to the network and to
 * playback with no conversion.
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

    /** Play PCM audio. Returns immediately; playback runs in the background. */
    fun speak(pcm: ByteArray)

    /** Stop playback. Safe to call when nothing is playing. */
    fun stopSpeaking()
}

/**
 * The real implementation, using AudioRecord and AudioTrack.
 *
 * Recording starts on a background thread because AudioRecord.read blocks, and
 * blocking the main thread in a voice-first app is how the UI stops responding
 * while the user is speaking.
 */
class AndroidVoiceEngine(
    private val context: android.content.Context,
) : VoiceEngine {

    private var recorder: AudioRecord? = null
    private var player: AudioTrack? = null
    private var captureThread: Thread? = null

    @Volatile
    private var listening = false

    @Volatile
    private var speaking = false

    override val isListening: Boolean get() = listening
    override val isSpeaking: Boolean get() = speaking

    override fun canRecord(): Boolean =
        ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED

    override fun startRecording() {
        if (listening || !canRecord()) return

        val minBuffer = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
        check(minBuffer > 0) { "Device does not support 16 kHz mono capture" }

        val record = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            CHANNEL_CONFIG,
            AUDIO_FORMAT,
            minBuffer * 2,
        )
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
            while (listening) {
                val read = record.read(buffer, 0, buffer.size)
                if (read > 0) collected.write(buffer, 0, read)
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

    override fun speak(pcm: ByteArray) {
        if (pcm.isEmpty()) return
        stopSpeaking()

        val track = AudioTrack.Builder()
            .setAudioAttributes(
                android.media.AudioAttributes.Builder()
                    .setUsage(android.media.AudioAttributes.USAGE_MEDIA)
                    .setContentType(android.media.AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build(),
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AUDIO_FORMAT)
                    .setSampleRate(SAMPLE_RATE)
                    .setChannelMask(CHANNEL_CONFIG)
                    .build(),
            )
            .setBufferSizeInBytes(pcm.size)
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()

        player = track
        speaking = true
        track.write(pcm, 0, pcm.size)
        track.play()
    }

    /**
     * Stop playback immediately.
     *
     * The user must be able to cut the app off mid-answer. In a home visit an
     * answer that has moved on is worse than one that stopped, because the
     * patient is waiting.
     */
    override fun stopSpeaking() {
        val track = player ?: return
        runCatching {
            if (track.playState == AudioTrack.PLAYSTATE_PLAYING) track.stop()
            track.release()
        }
        player = null
        speaking = false
    }

    companion object {
        /** Pinned by §10.6 and required by Sarvam. */
        const val SAMPLE_RATE = 16_000
        const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT

        /** Roughly a quarter second at this format. Below it, it is a tap. */
        const val MIN_AUDIO_BYTES = SAMPLE_RATE / 4 * 2

        const val STOP_JOIN_TIMEOUT_MS = 2_000L
    }
}
