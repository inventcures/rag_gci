package org.inventcures.pallisahayak.data.voice

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Puts a WAV header on recorded PCM.
 *
 * The microphone gives bare samples. The server passes the upload to Sarvam as
 * `audio.wav`, and a decoder that is told "WAV" and finds no RIFF header has no
 * sample rate or width to read the samples with. The app sent the bare samples
 * labelled `audio/wav` until the Android review (R1).
 *
 * Plain Kotlin, so the header is tested on the JVM without a device.
 */
object WavAudio {

    const val HEADER_BYTES = 44

    /** [pcm] as a canonical 44-byte-header PCM WAV file. */
    fun fromPcm(
        pcm: ByteArray,
        sampleRate: Int,
        channels: Int = 1,
        bitsPerSample: Int = 16,
    ): ByteArray {
        val blockAlign = channels * bitsPerSample / 8
        val out = ByteBuffer.allocate(HEADER_BYTES + pcm.size).order(ByteOrder.LITTLE_ENDIAN)
        out.put(ascii("RIFF"))
        out.putInt(HEADER_BYTES - 8 + pcm.size)
        out.put(ascii("WAVE"))
        out.put(ascii("fmt "))
        out.putInt(16)
        out.putShort(FORMAT_PCM)
        out.putShort(channels.toShort())
        out.putInt(sampleRate)
        out.putInt(sampleRate * blockAlign)
        out.putShort(blockAlign.toShort())
        out.putShort(bitsPerSample.toShort())
        out.put(ascii("data"))
        out.putInt(pcm.size)
        out.put(pcm)
        return out.array()
    }

    private fun ascii(tag: String): ByteArray = tag.toByteArray(Charsets.US_ASCII)

    private const val FORMAT_PCM: Short = 1
}
