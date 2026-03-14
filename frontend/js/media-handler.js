/**
 * MediSight — Media Handler
 *
 * Handles webcam video capture, microphone audio capture,
 * and audio playback for the Gemini Live API integration.
 *
 * Video capture uses an OFFSCREEN canvas so the visible
 * webcam feed stays perfectly smooth at full frame rate.
 */

export class MediaHandler {
    constructor() {
        this.stream = null;
        this.audioContext = null;
        this.audioWorklet = null;
        this.videoElement = null;
        this.frameInterval = null;
        this.playbackContext = null;
        this.playbackQueue = [];
        this.isPlaying = false;
        this.nextPlayTime = 0;

        // Offscreen canvas for frame capture (never visible — no flicker)
        this._captureCanvas = document.createElement('canvas');
        this._captureCanvas.width = 512;
        this._captureCanvas.height = 512;
        this._captureCtx = this._captureCanvas.getContext('2d');

        // Callbacks
        this.onAudioData = null;   // (base64: string) => void
        this.onVideoFrame = null;  // (base64: string) => void
    }

    /**
     * Start capturing audio and video from the user's devices.
     */
    async start(videoElement) {
        this.videoElement = videoElement;

        try {
            // Check if mediaDevices API is available
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error(
                    'Camera/microphone access requires HTTPS or localhost. ' +
                    'Make sure you are accessing via http://localhost:8000'
                );
            }

            // Request camera + microphone
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                },
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    frameRate: { ideal: 30 },
                    facingMode: 'environment',
                },
            });

            // Set video source — streams directly, no canvas involvement
            this.videoElement.srcObject = this.stream;
            await this.videoElement.play();

            // Start audio capture
            await this._startAudioCapture();

            // Start video frame capture (1 FPS on offscreen canvas)
            this._startVideoCapture();

            return true;
        } catch (err) {
            console.error('Failed to access media devices:', err);
            throw err;
        }
    }

    /**
     * Stop all media capture.
     */
    stop() {
        if (this.frameInterval) {
            clearInterval(this.frameInterval);
            this.frameInterval = null;
        }

        if (this.audioWorklet) {
            this.audioWorklet.disconnect();
            this.audioWorklet = null;
        }

        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close();
            this.audioContext = null;
        }

        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }

        if (this.videoElement) {
            this.videoElement.srcObject = null;
        }

        this._stopPlayback();
    }

    /**
     * Queue received audio data for playback.
     * @param {string} base64Audio - Base64-encoded PCM audio (24kHz, 16-bit LE)
     */
    playAudio(base64Audio) {
        const binaryStr = atob(base64Audio);
        const bytes = new Uint8Array(binaryStr.length);
        for (let i = 0; i < binaryStr.length; i++) {
            bytes[i] = binaryStr.charCodeAt(i);
        }

        const int16 = new Int16Array(bytes.buffer);
        const float32 = new Float32Array(int16.length);
        for (let i = 0; i < int16.length; i++) {
            float32[i] = int16[i] / 32768.0;
        }

        this.playbackQueue.push(float32);
        this._processPlaybackQueue();
    }

    /**
     * Stop any currently playing audio (for interruption/barge-in).
     */
    stopPlayback() {
        this._stopPlayback();
    }

    // -------------------------------------------------------------------
    // Private: Audio capture
    // -------------------------------------------------------------------

    async _startAudioCapture() {
        this.audioContext = new AudioContext({ sampleRate: 16000 });

        const workletUrl = new URL('./pcm-processor.js', import.meta.url).href;
        await this.audioContext.audioWorklet.addModule(workletUrl);

        const source = this.audioContext.createMediaStreamSource(this.stream);

        this.audioWorklet = new AudioWorkletNode(this.audioContext, 'pcm-processor');

        this.audioWorklet.port.onmessage = (event) => {
            if (event.data.type === 'audio' && this.onAudioData) {
                const pcmBuffer = new Uint8Array(event.data.data);
                const base64 = this._arrayBufferToBase64(pcmBuffer);
                this.onAudioData(base64);
            }
        };

        source.connect(this.audioWorklet);
        // Don't connect to destination to avoid feedback
    }

    // -------------------------------------------------------------------
    // Private: Video capture (OFFSCREEN — never touches the visible feed)
    // -------------------------------------------------------------------

    _startVideoCapture() {
        // Capture frames at 1 FPS using an offscreen canvas.
        // The visible <video> element streams directly from getUserMedia
        // at full frame rate — this capture process is invisible.
        this.frameInterval = setInterval(() => {
            if (!this.videoElement || this.videoElement.readyState < 2) return;

            // Draw to offscreen canvas only
            this._captureCtx.drawImage(
                this.videoElement,
                0, 0,
                this._captureCanvas.width,
                this._captureCanvas.height,
            );

            // Convert to JPEG and send
            this._captureCanvas.toBlob((blob) => {
                if (blob && this.onVideoFrame) {
                    const reader = new FileReader();
                    reader.onloadend = () => {
                        const base64 = reader.result.split(',')[1];
                        this.onVideoFrame(base64);
                    };
                    reader.readAsDataURL(blob);
                }
            }, 'image/jpeg', 0.6);
        }, 1000); // 1 FPS
    }

    // -------------------------------------------------------------------
    // Private: Audio playback
    // -------------------------------------------------------------------

    _processPlaybackQueue() {
        if (this.playbackQueue.length === 0) return;

        if (!this.playbackContext || this.playbackContext.state === 'closed') {
            this.playbackContext = new AudioContext({ sampleRate: 24000 });
            this.nextPlayTime = this.playbackContext.currentTime;
        }

        while (this.playbackQueue.length > 0) {
            const float32Data = this.playbackQueue.shift();

            const audioBuffer = this.playbackContext.createBuffer(
                1,                    // mono
                float32Data.length,   // length
                24000                 // sample rate
            );

            audioBuffer.getChannelData(0).set(float32Data);

            const source = this.playbackContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.playbackContext.destination);

            const startTime = Math.max(
                this.playbackContext.currentTime,
                this.nextPlayTime,
            );
            source.start(startTime);
            this.nextPlayTime = startTime + audioBuffer.duration;

            this.isPlaying = true;
            source.onended = () => {
                if (this.playbackQueue.length === 0) {
                    this.isPlaying = false;
                }
            };
        }
    }

    _stopPlayback() {
        if (this.playbackContext && this.playbackContext.state !== 'closed') {
            this.playbackContext.close();
            this.playbackContext = null;
        }
        this.playbackQueue = [];
        this.isPlaying = false;
        this.nextPlayTime = 0;
    }

    // -------------------------------------------------------------------
    // Utility
    // -------------------------------------------------------------------

    _arrayBufferToBase64(buffer) {
        let binary = '';
        const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }
}
