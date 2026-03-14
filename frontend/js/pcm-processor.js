/**
 * MediSight — PCM Audio Worklet Processor
 *
 * Captures raw PCM audio data from the microphone at 16kHz
 * and posts it to the main thread as Float32Array chunks.
 *
 * This file is loaded as an AudioWorklet module.
 */

class PCMProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this._buffer = new Float32Array(0);
        // Send audio every ~100ms (1600 samples at 16kHz)
        this._chunkSize = 1600;
    }

    process(inputs) {
        const input = inputs[0];
        if (!input || !input[0]) return true;

        const inputData = input[0]; // mono channel

        // Append to buffer
        const newBuffer = new Float32Array(this._buffer.length + inputData.length);
        newBuffer.set(this._buffer);
        newBuffer.set(inputData, this._buffer.length);
        this._buffer = newBuffer;

        // Send chunks when we have enough data
        while (this._buffer.length >= this._chunkSize) {
            const chunk = this._buffer.slice(0, this._chunkSize);
            this._buffer = this._buffer.slice(this._chunkSize);

            // Convert float32 to int16 PCM
            const pcm16 = new Int16Array(chunk.length);
            for (let i = 0; i < chunk.length; i++) {
                const s = Math.max(-1, Math.min(1, chunk[i]));
                pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
            }

            this.port.postMessage({
                type: 'audio',
                data: pcm16.buffer,
            }, [pcm16.buffer]);
        }

        return true;
    }
}

registerProcessor('pcm-processor', PCMProcessor);
