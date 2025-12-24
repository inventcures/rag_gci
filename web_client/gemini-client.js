/**
 * GeminiLiveClient - WebSocket client for Gemini Live voice conversations
 *
 * Handles:
 * - Audio capture using AudioWorklet (16kHz PCM for input)
 * - WebSocket communication with backend
 * - Audio playback of responses (24kHz PCM from Gemini)
 * - Session management and language switching
 *
 * Requirements:
 * - Modern browser with AudioWorklet support
 * - Microphone permission
 * - WebSocket connection to backend
 */

class GeminiLiveClient {
    /**
     * Create a new GeminiLiveClient
     * @param {Object} options - Configuration options
     * @param {string} options.serverUrl - WebSocket server URL
     * @param {string} [options.language='en-IN'] - Initial language code
     * @param {Function} [options.onAudioReceived] - Callback for audio responses
     * @param {Function} [options.onTranscription] - Callback for transcription updates
     * @param {Function} [options.onError] - Callback for errors
     * @param {Function} [options.onStatusChange] - Callback for status updates
     * @param {Function} [options.onConnectionChange] - Callback for connection state changes
     */
    constructor(options) {
        this.serverUrl = options.serverUrl;
        this.language = options.language || 'en-IN';

        // Callbacks
        this.onAudioReceived = options.onAudioReceived || (() => {});
        this.onTranscription = options.onTranscription || (() => {});
        this.onError = options.onError || console.error;
        this.onStatusChange = options.onStatusChange || (() => {});
        this.onConnectionChange = options.onConnectionChange || (() => {});

        // Internal state
        this.ws = null;
        this.audioContext = null;
        this.mediaStream = null;
        this.workletNode = null;
        this.sourceNode = null;
        this.isInitialized = false;
        this.isRecording = false;

        // Audio playback
        this.audioQueue = [];
        this.isPlaying = false;
        this.playbackContext = null;

        // Reconnection settings
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;

        // Audio settings
        this.inputSampleRate = 16000;  // Gemini input requirement
        this.outputSampleRate = 24000; // Gemini output format
        this.bufferSize = 4096;
    }

    /**
     * Initialize the audio system and request microphone permission
     * @returns {Promise<void>}
     */
    async initialize() {
        try {
            // Check for required browser APIs
            if (!window.AudioContext && !window.webkitAudioContext) {
                throw new Error('AudioContext not supported in this browser');
            }

            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('getUserMedia not supported in this browser');
            }

            // Create AudioContext at input sample rate
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.inputSampleRate
            });

            // Load AudioWorklet processor
            try {
                await this.audioContext.audioWorklet.addModule('audio-worklet.js');
            } catch (workletError) {
                console.warn('AudioWorklet failed, falling back to ScriptProcessor:', workletError);
                // Will use ScriptProcessorNode as fallback
            }

            // Request microphone permission
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: this.inputSampleRate,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });

            this.isInitialized = true;
            console.log('GeminiLiveClient initialized successfully');

        } catch (error) {
            console.error('Initialization failed:', error);
            this.onError(error);
            throw error;
        }
    }

    /**
     * Connect to the WebSocket server
     * @returns {Promise<void>}
     */
    async connect() {
        return new Promise((resolve, reject) => {
            try {
                this.onConnectionChange('connecting');
                this.ws = new WebSocket(this.serverUrl);
                this.ws.binaryType = 'arraybuffer';

                this.ws.onopen = () => {
                    console.log('WebSocket connected');
                    this.reconnectAttempts = 0;

                    // Send initial configuration
                    this.ws.send(JSON.stringify({
                        type: 'config',
                        language: this.language,
                        user_id: this.getUserId()
                    }));

                    this.onConnectionChange('connected');
                    resolve();
                };

                this.ws.onmessage = (event) => {
                    this.handleMessage(event.data);
                };

                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.onError(error);
                    this.onConnectionChange('error');
                    reject(error);
                };

                this.ws.onclose = (event) => {
                    console.log('WebSocket closed:', event.code, event.reason);
                    this.onConnectionChange('disconnected');

                    // Attempt reconnection if not intentionally closed
                    if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
                        this.scheduleReconnect();
                    }
                };

            } catch (error) {
                this.onError(error);
                this.onConnectionChange('error');
                reject(error);
            }
        });
    }

    /**
     * Schedule a reconnection attempt
     */
    scheduleReconnect() {
        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

        console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        this.onStatusChange(`Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

        setTimeout(async () => {
            try {
                await this.connect();
            } catch (error) {
                console.error('Reconnection failed:', error);
            }
        }, delay);
    }

    /**
     * Start recording audio from microphone
     * @returns {Promise<void>}
     */
    async startRecording() {
        if (!this.isInitialized) {
            await this.initialize();
        }

        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            await this.connect();
        }

        // Resume AudioContext if suspended (required after user interaction)
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }

        // Create source from microphone
        this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);

        // Try to use AudioWorklet, fallback to ScriptProcessor
        if (this.audioContext.audioWorklet) {
            try {
                this.workletNode = new AudioWorkletNode(
                    this.audioContext,
                    'audio-capture-processor',
                    {
                        processorOptions: {
                            bufferSize: this.bufferSize
                        }
                    }
                );

                this.workletNode.port.onmessage = (event) => {
                    if (event.data.type === 'audio' && this.ws?.readyState === WebSocket.OPEN) {
                        this.ws.send(event.data.buffer);
                    }
                };

                this.sourceNode.connect(this.workletNode);
                // Don't connect to destination to avoid feedback
            } catch (error) {
                console.warn('AudioWorklet node creation failed, using fallback:', error);
                this.setupScriptProcessor();
            }
        } else {
            this.setupScriptProcessor();
        }

        // Notify server that we're starting
        this.ws.send(JSON.stringify({ type: 'start_audio' }));

        this.isRecording = true;
        this.onStatusChange('Listening...');
    }

    /**
     * Setup ScriptProcessorNode as fallback for older browsers
     */
    setupScriptProcessor() {
        // ScriptProcessorNode is deprecated but provides fallback
        const scriptNode = this.audioContext.createScriptProcessor(this.bufferSize, 1, 1);

        scriptNode.onaudioprocess = (event) => {
            if (!this.isRecording || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
                return;
            }

            const inputData = event.inputBuffer.getChannelData(0);

            // Convert Float32 to Int16
            const int16Buffer = new Int16Array(inputData.length);
            for (let i = 0; i < inputData.length; i++) {
                const s = Math.max(-1, Math.min(1, inputData[i]));
                int16Buffer[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }

            this.ws.send(int16Buffer.buffer);
        };

        this.sourceNode.connect(scriptNode);
        scriptNode.connect(this.audioContext.destination);
        this.workletNode = scriptNode; // Store for cleanup
    }

    /**
     * Stop recording audio
     * @returns {Promise<void>}
     */
    async stopRecording() {
        this.isRecording = false;

        // Disconnect audio nodes
        if (this.workletNode) {
            this.workletNode.disconnect();
            this.workletNode = null;
        }

        if (this.sourceNode) {
            this.sourceNode.disconnect();
            this.sourceNode = null;
        }

        // Notify server that we stopped
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'stop_audio' }));
        }

        this.onStatusChange('Processing...');
    }

    /**
     * Handle incoming WebSocket message
     * @param {ArrayBuffer|string} data - Message data
     */
    handleMessage(data) {
        // Binary data = audio response
        if (data instanceof ArrayBuffer) {
            this.audioQueue.push(data);
            this.playNextAudio();
            return;
        }

        // Text data = JSON message
        try {
            const message = JSON.parse(data);

            switch (message.type) {
                case 'transcription':
                    this.onTranscription({
                        type: message.role,
                        text: message.text
                    });
                    break;

                case 'turn_complete':
                    this.onStatusChange('Tap to speak');
                    break;

                case 'session_created':
                    console.log('Session created:', message.session_id);
                    break;

                case 'error':
                    console.error('Server error:', message.error);
                    this.onError(new Error(message.error));
                    break;

                case 'rag_context':
                    console.log('RAG context injected:', message.chars, 'characters');
                    break;

                default:
                    console.log('Unknown message type:', message.type);
            }
        } catch (e) {
            console.error('Failed to parse message:', e);
        }
    }

    /**
     * Play queued audio responses
     */
    async playNextAudio() {
        if (this.isPlaying || this.audioQueue.length === 0) {
            return;
        }

        this.isPlaying = true;
        this.onStatusChange('Speaking...');

        try {
            // Create playback context at output sample rate
            if (!this.playbackContext || this.playbackContext.state === 'closed') {
                this.playbackContext = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: this.outputSampleRate
                });
            }

            // Resume if suspended
            if (this.playbackContext.state === 'suspended') {
                await this.playbackContext.resume();
            }

            while (this.audioQueue.length > 0) {
                const audioData = this.audioQueue.shift();

                // Convert ArrayBuffer (Int16) to Float32 for Web Audio
                const int16Data = new Int16Array(audioData);
                const float32Data = new Float32Array(int16Data.length);

                for (let i = 0; i < int16Data.length; i++) {
                    float32Data[i] = int16Data[i] / 32768.0;
                }

                // Create AudioBuffer
                const buffer = this.playbackContext.createBuffer(
                    1,                      // mono
                    float32Data.length,     // length
                    this.outputSampleRate   // sample rate
                );
                buffer.getChannelData(0).set(float32Data);

                // Create and play source
                const source = this.playbackContext.createBufferSource();
                source.buffer = buffer;
                source.connect(this.playbackContext.destination);
                source.start();

                // Wait for this chunk to finish
                await new Promise(resolve => {
                    source.onended = resolve;
                });
            }

        } catch (error) {
            console.error('Audio playback error:', error);
            this.onError(error);
        } finally {
            this.isPlaying = false;
            this.onStatusChange('Tap to speak');
        }
    }

    /**
     * Set the conversation language
     * @param {string} language - Language code (e.g., 'en-IN', 'hi-IN')
     */
    setLanguage(language) {
        this.language = language;

        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'set_language',
                language: language
            }));
        }
    }

    /**
     * Send a text message (for testing or hybrid mode)
     * @param {string} text - Text to send
     */
    sendText(text) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'text',
                text: text
            }));
        }
    }

    /**
     * Get or generate a user ID
     * @returns {string} User ID
     */
    getUserId() {
        let userId = localStorage.getItem('gemini_user_id');
        if (!userId) {
            userId = 'web_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('gemini_user_id', userId);
        }
        return userId;
    }

    /**
     * Disconnect and cleanup all resources
     */
    disconnect() {
        this.isRecording = false;

        // Stop audio capture
        if (this.workletNode) {
            this.workletNode.disconnect();
            this.workletNode = null;
        }

        if (this.sourceNode) {
            this.sourceNode.disconnect();
            this.sourceNode = null;
        }

        // Stop microphone
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }

        // Close WebSocket
        if (this.ws) {
            this.ws.close(1000, 'Client disconnecting');
            this.ws = null;
        }

        // Close audio contexts
        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close();
        }

        if (this.playbackContext && this.playbackContext.state !== 'closed') {
            this.playbackContext.close();
        }

        this.isInitialized = false;
        this.audioQueue = [];

        console.log('GeminiLiveClient disconnected');
    }

    /**
     * Get current client state
     * @returns {Object} State object
     */
    getState() {
        return {
            isInitialized: this.isInitialized,
            isRecording: this.isRecording,
            isPlaying: this.isPlaying,
            isConnected: this.ws?.readyState === WebSocket.OPEN,
            language: this.language,
            audioQueueLength: this.audioQueue.length
        };
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GeminiLiveClient;
}
