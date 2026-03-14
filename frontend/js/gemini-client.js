/**
 * MediSight — Gemini WebSocket Client
 *
 * Manages the WebSocket connection between the browser and the
 * FastAPI backend, which proxies communication with the Gemini
 * Live API.
 */

export class GeminiClient {
    constructor() {
        this.ws = null;
        this.isConnected = false;

        // Event callbacks
        this.onAudio = null;         // (base64: string) => void
        this.onTranscript = null;    // (role, text) => void
        this.onToolCall = null;      // (name, args) => void
        this.onToolResult = null;    // (name, result) => void
        this.onStatus = null;        // (status, message) => void
        this.onTurnComplete = null;  // () => void
        this.onInterrupted = null;   // () => void
        this.onError = null;         // (message) => void
    }

    /**
     * Connect to the backend WebSocket.
     */
    connect() {
        return new Promise((resolve, reject) => {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const url = `${protocol}//${window.location.host}/ws`;

            console.log('[GeminiClient] Connecting to', url);
            this.ws = new WebSocket(url);

            this.ws.onopen = () => {
                console.log('[GeminiClient] WebSocket connected');
                this.isConnected = true;
                
                // Immediately send the setup message containing any patient context
                const setupMsg = {
                    type: "setup",
                    patient_context: window.currentPatientContext || ""
                };
                this.ws.send(JSON.stringify(setupMsg));
                
                resolve();
            };

            this.ws.onmessage = (event) => {
                this._handleMessage(event.data);
            };

            this.ws.onerror = (event) => {
                console.error('[GeminiClient] WebSocket error:', event);
                if (!this.isConnected) {
                    reject(new Error('WebSocket connection failed'));
                }
                if (this.onError) {
                    this.onError('WebSocket connection error');
                }
            };

            this.ws.onclose = (event) => {
                console.log('[GeminiClient] WebSocket closed:', event.code, event.reason);
                this.isConnected = false;
                if (this.onStatus) {
                    this.onStatus('disconnected', 'Connection closed');
                }
            };
        });
    }

    /**
     * Disconnect from the backend WebSocket.
     */
    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.isConnected = false;
    }

    /**
     * Send base64-encoded audio data to the backend.
     */
    sendAudio(base64Audio) {
        this._send({
            type: 'audio',
            data: base64Audio,
        });
    }

    /**
     * Send a base64-encoded video frame to the backend.
     */
    sendVideo(base64Frame) {
        this._send({
            type: 'video',
            data: base64Frame,
        });
    }

    /**
     * Send a text message to the backend.
     */
    sendText(text) {
        this._send({
            type: 'text',
            text: text,
        });
    }

    // -------------------------------------------------------------------
    // Private
    // -------------------------------------------------------------------

    _send(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        }
    }

    _handleMessage(raw) {
        let msg;
        try {
            msg = JSON.parse(raw);
        } catch (e) {
            console.error('[GeminiClient] Invalid JSON:', raw);
            return;
        }

        switch (msg.type) {
            case 'audio':
                if (this.onAudio) this.onAudio(msg.data);
                break;

            case 'transcript':
                if (this.onTranscript) this.onTranscript(msg.role, msg.text);
                break;

            case 'tool_call':
                if (this.onToolCall) this.onToolCall(msg.name, msg.args);
                break;

            case 'tool_result':
                if (this.onToolResult) this.onToolResult(msg.name, msg.result);
                break;

            case 'status':
                if (this.onStatus) this.onStatus(msg.status, msg.message);
                break;

            case 'turn_complete':
                if (this.onTurnComplete) this.onTurnComplete();
                break;

            case 'interrupted':
                if (this.onInterrupted) this.onInterrupted();
                break;

            case 'error':
                console.error('[GeminiClient] Error from server:', msg.message);
                if (this.onError) this.onError(msg.message);
                break;

            default:
                console.warn('[GeminiClient] Unknown message type:', msg.type);
        }
    }
}
