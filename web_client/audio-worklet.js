/**
 * AudioWorklet processor for capturing microphone audio
 *
 * This processor runs in the audio rendering thread for low-latency
 * audio capture. It converts Float32 samples to Int16 PCM and sends
 * to the main thread via MessagePort.
 *
 * Audio Format:
 * - Input: Float32 samples from microphone (typically 128 samples per frame)
 * - Output: Int16 PCM in ArrayBuffer (sent when buffer is full)
 * - Sample Rate: Inherits from AudioContext (should be 16kHz for Gemini)
 * - Channels: Mono (single channel)
 */

class AudioCaptureProcessor extends AudioWorkletProcessor {
    /**
     * Create a new AudioCaptureProcessor
     * @param {Object} options - Processor options
     * @param {Object} options.processorOptions - Custom options
     * @param {number} [options.processorOptions.bufferSize=4096] - Buffer size before sending
     */
    constructor(options) {
        super();

        // Get buffer size from options or use default
        const processorOptions = options.processorOptions || {};
        this.bufferSize = processorOptions.bufferSize || 4096;

        // Internal buffer for accumulating samples
        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;

        // Track if we should continue processing
        this.isActive = true;

        // Listen for messages from main thread
        this.port.onmessage = (event) => {
            if (event.data.type === 'stop') {
                this.isActive = false;
            } else if (event.data.type === 'start') {
                this.isActive = true;
            } else if (event.data.type === 'setBufferSize') {
                this.bufferSize = event.data.bufferSize;
                this.buffer = new Float32Array(this.bufferSize);
                this.bufferIndex = 0;
            }
        };
    }

    /**
     * Process audio samples
     *
     * This method is called for each audio frame (typically 128 samples at 16kHz).
     * We accumulate samples into a larger buffer before sending to reduce overhead.
     *
     * @param {Float32Array[][]} inputs - Input audio channels
     * @param {Float32Array[][]} outputs - Output audio channels (not used)
     * @param {Object} parameters - Audio parameters (not used)
     * @returns {boolean} True to keep processing, false to stop
     */
    process(inputs, outputs, parameters) {
        // Get first input (microphone)
        const input = inputs[0];

        // Check if we have audio data
        if (!input || !input[0] || input[0].length === 0) {
            return this.isActive;
        }

        // Get mono channel data
        const samples = input[0];

        // Add samples to buffer
        for (let i = 0; i < samples.length; i++) {
            this.buffer[this.bufferIndex++] = samples[i];

            // When buffer is full, convert and send
            if (this.bufferIndex >= this.bufferSize) {
                this.sendBuffer();
            }
        }

        // Continue processing
        return this.isActive;
    }

    /**
     * Convert buffer to Int16 PCM and send to main thread
     */
    sendBuffer() {
        // Create Int16 buffer for PCM data
        const int16Buffer = new Int16Array(this.bufferSize);

        // Convert Float32 [-1.0, 1.0] to Int16 [-32768, 32767]
        for (let i = 0; i < this.bufferSize; i++) {
            // Clamp to valid range
            const sample = Math.max(-1, Math.min(1, this.buffer[i]));

            // Convert to Int16
            // For negative values, multiply by 32768 (0x8000)
            // For positive values, multiply by 32767 (0x7FFF)
            int16Buffer[i] = sample < 0
                ? Math.floor(sample * 0x8000)
                : Math.floor(sample * 0x7FFF);
        }

        // Send to main thread
        // Transfer the ArrayBuffer for efficiency (zero-copy)
        this.port.postMessage(
            {
                type: 'audio',
                buffer: int16Buffer.buffer,
                samples: this.bufferSize,
                timestamp: currentTime
            },
            [int16Buffer.buffer]  // Transferable
        );

        // Reset buffer index
        this.bufferIndex = 0;
    }
}

// Register the processor
registerProcessor('audio-capture-processor', AudioCaptureProcessor);
