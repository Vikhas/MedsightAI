/**
 * MediSight — Main Application Controller
 *
 * Orchestrates the UI, media capture, and WebSocket communication
 * for the MediSight clinical decision assistant.
 */

import { GeminiClient } from './gemini-client.js';
import { MediaHandler } from './media-handler.js';

// ---------------------------------------------------------------------------
// DOM Elements
// ---------------------------------------------------------------------------

const elements = {
    // Header
    connectionStatus: document.getElementById('connectionStatus'),
    statusText: document.querySelector('.status-text'),

    // Video panel
    webcamVideo: document.getElementById('webcamVideo'),
    overlayCanvas: document.getElementById('overlayCanvas'),
    videoPlaceholder: document.getElementById('videoPlaceholder'),
    audioVisualizer: document.getElementById('audioVisualizer'),
    visualizerCanvas: document.getElementById('visualizerCanvas'),

    // Transcript panel
    transcriptMessages: document.getElementById('transcriptMessages'),
    connectBtn: document.getElementById('connectBtn'),
    disconnectBtn: document.getElementById('disconnectBtn'),
    textInputContainer: document.getElementById('textInputContainer'),
    textInput: document.getElementById('textInput'),
    sendTextBtn: document.getElementById('sendTextBtn'),
    clearTranscriptBtn: document.getElementById('clearTranscriptBtn'),

    // Insights panel
    riskBanner: document.getElementById('riskBanner'),
    riskDot: document.getElementById('riskDot'),
    riskText: document.getElementById('riskText'),
    riskScore: document.getElementById('riskScore'),
    diagnosisSection: document.getElementById('diagnosisSection'),
    diagnosisList: document.getElementById('diagnosisList'),
    testsSection: document.getElementById('testsSection'),
    testsList: document.getElementById('testsList'),
    redFlagsSection: document.getElementById('redFlagsSection'),
    redFlagsList: document.getElementById('redFlagsList'),
    drugSection: document.getElementById('drugSection'),
    drugInfo: document.getElementById('drugInfo'),
    guidelinesSection: document.getElementById('guidelinesSection'),
    guidelinesInfo: document.getElementById('guidelinesInfo'),
    insightsEmpty: document.getElementById('insightsEmpty'),
    disclaimer: document.getElementById('disclaimer'),
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const client = new GeminiClient();
const media = new MediaHandler();

let state = 'idle'; // idle | connecting | connected
let lastAssistantBubble = null; // tracks the current streaming assistant bubble

// ---------------------------------------------------------------------------
// UI Helpers
// ---------------------------------------------------------------------------

function setConnectionState(newState) {
    state = newState;
    const el = elements.connectionStatus;

    el.classList.remove('disconnected', 'connecting', 'connected');

    switch (newState) {
        case 'idle':
            el.classList.add('disconnected');
            elements.statusText.textContent = 'Disconnected';
            elements.connectBtn.classList.remove('hidden');
            elements.disconnectBtn.classList.add('hidden');
            elements.textInputContainer.classList.add('hidden');
            elements.videoPlaceholder.classList.remove('hidden');
            elements.audioVisualizer.classList.remove('active');
            break;

        case 'connecting':
            el.classList.add('connecting');
            elements.statusText.textContent = 'Connecting…';
            elements.connectBtn.classList.add('hidden');
            elements.disconnectBtn.classList.add('hidden');
            break;

        case 'connected':
            el.classList.add('connected');
            elements.statusText.textContent = 'Connected';
            elements.connectBtn.classList.add('hidden');
            elements.disconnectBtn.classList.remove('hidden');
            elements.textInputContainer.classList.remove('hidden');
            elements.videoPlaceholder.classList.add('hidden');
            elements.audioVisualizer.classList.add('active');
            break;
    }
}

function addMessage(role, text) {
    // Remove welcome message if present
    const welcome = elements.transcriptMessages.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    const wrapper = document.createElement('div');
    wrapper.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';

    if (role === 'user') {
        avatar.textContent = 'DR';
        lastAssistantBubble = null; // reset on user message
    } else if (role === 'assistant') {
        avatar.innerHTML = `<svg width="16" height="16" viewBox="0 0 28 28" fill="none">
            <circle cx="14" cy="14" r="10" stroke="white" stroke-width="1.5" fill="none"/>
            <path d="M14 8 L14 20 M10 12 L18 12 M10 16 L18 16" stroke="white" stroke-width="1.2" stroke-linecap="round"/>
        </svg>`;
    } else {
        // System message
        lastAssistantBubble = null;
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.textContent = text;
        wrapper.appendChild(bubble);
        elements.transcriptMessages.appendChild(wrapper);
        scrollToBottom();
        return;
    }

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    
    // Store the raw text so we can continuously parse it as streaming chunks arrive
    bubble.dataset.rawText = text;
    bubble.innerHTML = parseMarkdown(text);

    wrapper.appendChild(avatar);
    wrapper.appendChild(bubble);
    elements.transcriptMessages.appendChild(wrapper);
    scrollToBottom();

    // Track for streaming accumulation
    if (role === 'assistant') {
        lastAssistantBubble = bubble;
    }
}

/**
 * Append text to the current assistant message bubble,
 * or create a new one if none exists. This handles
 * streaming transcript chunks properly so each response
 * is one continuous message instead of many small bubbles.
 */
function appendAssistantText(text) {
    if (lastAssistantBubble) {
        lastAssistantBubble.dataset.rawText += text;
        lastAssistantBubble.innerHTML = parseMarkdown(lastAssistantBubble.dataset.rawText);
        scrollToBottom();
    } else {
        addMessage('assistant', text);
    }
}

/**
 * Converts basic markdown (like **bold**) into HTML tags.
 */
function parseMarkdown(text) {
    if (!text) return '';
    // Replace **bold** with <strong>bold</strong>
    let html = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Replace *italic* with <em>italic</em>
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    // Replace line breaks with <br>
    html = html.replace(/\n/g, '<br>');
    return html;
}

function addToolIndicator(toolName) {
    lastAssistantBubble = null; // break after tool calls
    const indicator = document.createElement('div');
    indicator.className = 'tool-indicator';
    indicator.id = `tool-${toolName}`;
    indicator.innerHTML = `
        <div class="spinner"></div>
        <span>Running ${formatToolName(toolName)}…</span>
    `;
    elements.transcriptMessages.appendChild(indicator);
    scrollToBottom();
}

function removeToolIndicator(toolName) {
    const el = document.getElementById(`tool-${toolName}`);
    if (el) el.remove();
}

function formatToolName(name) {
    return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function scrollToBottom() {
    elements.transcriptMessages.scrollTop = elements.transcriptMessages.scrollHeight;
}

// ---------------------------------------------------------------------------
// Insights Rendering
// ---------------------------------------------------------------------------

function renderDiagnosis(result) {
    console.warn(">>> renderDiagnosis called. typeof result:", typeof result);
    let parsedResult = result;
    if (typeof result === 'string') {
        try {
            parsedResult = JSON.parse(result);
        } catch (e) {
            console.error("Failed to parse result JSON in renderDiagnosis:", e);
            console.error("String was:", result);
            return;
        }
    }

    console.warn(">>> parsedResult structure:", parsedResult);
    
    if (!parsedResult) {
        console.warn(">>> Exiting early: parsedResult is null or undefined");
        return;
    }
    if (!parsedResult.differentials) {
        console.warn(">>> Exiting early: parsedResult.differentials is undefined. Object keys are:", Object.keys(parsedResult));
        return;
    }

    console.warn(">>> Check passed. Proceeding to render the UI.");
    
    elements.insightsEmpty.classList.add('hidden');
    elements.disclaimer.classList.remove('hidden');

    // Diagnoses
    elements.diagnosisSection.classList.remove('hidden');
    elements.diagnosisList.innerHTML = '';

    parsedResult.differentials.forEach(d => {
        const confidence = Math.round(d.confidence * 100);
        let color;
        if (confidence >= 70) color = 'var(--risk-low)';
        else if (confidence >= 50) color = 'var(--accent-cyan)';
        else if (confidence >= 30) color = 'var(--risk-moderate)';
        else color = 'var(--text-muted)';

        const card = document.createElement('div');
        card.className = 'diagnosis-card';
        card.innerHTML = `
            <div>
                <div class="diagnosis-name">${d.diagnosis}</div>
                <div class="icd-code">${d.icd10 || ''}</div>
            </div>
            <div class="diagnosis-meta">
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${confidence}%; background: ${color};"></div>
                </div>
                <div class="confidence-text" style="color: ${color};">${confidence}%</div>
            </div>
        `;
        elements.diagnosisList.appendChild(card);
    });

    // Recommended tests
    if (parsedResult.recommended_tests && parsedResult.recommended_tests.length) {
        elements.testsSection.classList.remove('hidden');
        elements.testsList.innerHTML = '';
        parsedResult.recommended_tests.forEach(test => {
            const li = document.createElement('li');
            li.textContent = test;
            elements.testsList.appendChild(li);
        });
    }

    // Red flags
    if (parsedResult.red_flags && parsedResult.red_flags.length) {
        elements.redFlagsSection.classList.remove('hidden');
        elements.redFlagsList.innerHTML = '';
        parsedResult.red_flags.forEach(flag => {
            const li = document.createElement('li');
            li.textContent = flag;
            elements.redFlagsList.appendChild(li);
        });
    }
}

function renderDrugInfo(result) {
    if (!result) return;

    elements.insightsEmpty.classList.add('hidden');
    elements.disclaimer.classList.remove('hidden');
    elements.drugSection.classList.remove('hidden');

    const isSafe = result.status === 'SAFE_TO_USE';
    let html = `
        <div class="drug-status ${isSafe ? 'safe' : 'contraindicated'}">
            ${isSafe ? '✓ Safe' : '✕ Contraindicated'}
        </div>
        <div class="info-field">
            <strong>Drug</strong>
            ${result.drug} (${result.drug_class || 'N/A'})
        </div>
    `;

    if (result.allergy_conflict) {
        html += `
            <div class="info-field">
                <strong>Allergy Conflict</strong>
                Patient allergies: ${result.patient_allergies.join(', ')}
            </div>
        `;
    }

    if (result.alternatives && result.alternatives.length) {
        html += `
            <div class="info-field">
                <strong>Alternatives</strong>
                ${result.alternatives.join(', ')}
            </div>
        `;
    }

    if (result.known_interactions && result.known_interactions.length) {
        html += `
            <div class="info-field">
                <strong>Known Interactions</strong>
                ${result.known_interactions.join('; ')}
            </div>
        `;
    }

    elements.drugInfo.innerHTML = html;
}

function renderGuidelines(result) {
    if (!result) return;

    elements.insightsEmpty.classList.add('hidden');
    elements.disclaimer.classList.remove('hidden');
    elements.guidelinesSection.classList.remove('hidden');

    let html = `
        <div class="info-field">
            <strong>Condition</strong>
            ${result.condition}
        </div>
        <div class="info-field">
            <strong>First-Line Treatment</strong>
            ${result.first_line_treatment || 'N/A'}
        </div>
        <div class="info-field">
            <strong>Alternative</strong>
            ${result.alternative || 'N/A'}
        </div>
        <div class="info-field">
            <strong>Follow-Up</strong>
            ${result.follow_up || 'N/A'}
        </div>
        <div class="info-field">
            <strong>Evidence</strong>
            ${result.evidence_level || 'N/A'}
        </div>
        <div class="info-field">
            <strong>Source</strong>
            ${result.source || 'N/A'}
        </div>
    `;

    elements.guidelinesInfo.innerHTML = html;
}

function renderRiskAssessment(result) {
    if (!result) return;

    elements.insightsEmpty.classList.add('hidden');
    elements.disclaimer.classList.remove('hidden');
    elements.riskBanner.classList.remove('hidden', 'critical', 'high', 'moderate', 'low');

    const level = result.risk_level.toLowerCase();
    elements.riskBanner.classList.add(level);
    elements.riskText.textContent = result.risk_level;
    elements.riskScore.textContent = `Severity Score: ${result.severity_score}`;
}

function clearInsights() {
    elements.riskBanner.classList.add('hidden');
    elements.diagnosisSection.classList.add('hidden');
    elements.testsSection.classList.add('hidden');
    elements.redFlagsSection.classList.add('hidden');
    elements.drugSection.classList.add('hidden');
    elements.guidelinesSection.classList.add('hidden');
    elements.disclaimer.classList.add('hidden');
    elements.insightsEmpty.classList.remove('hidden');
}

// ---------------------------------------------------------------------------
// Connection Logic
// ---------------------------------------------------------------------------

async function connect() {
    if (state !== 'idle') return;

    setConnectionState('connecting');
    addMessage('system', 'Connecting to MediSight AI…');

    try {
        // Clean up any previous session
        media.stop();
        client.disconnect();

        // Small delay to let cleanup finish
        await new Promise(r => setTimeout(r, 200));

        // Connect WebSocket
        await client.connect();

        // Start media capture (fresh AudioContext each time)
        await media.start(elements.webcamVideo);

        // Wire up media → client
        media.onAudioData = (base64) => {
            if (state === 'connected') client.sendAudio(base64);
        };

        media.onVideoFrame = (base64) => {
            if (state === 'connected') client.sendVideo(base64);
        };

        setConnectionState('connected');
        addMessage('system', 'Connected — speak naturally or type a message');

    } catch (err) {
        console.error('Connection failed:', err);
        media.stop();
        client.disconnect();
        setConnectionState('idle');
        addMessage('system', `Connection failed: ${err.message}`);
    }
}

function disconnect() {
    media.stop();
    client.disconnect();
    setConnectionState('idle');
    lastAssistantBubble = null;
    addMessage('system', 'Disconnected from MediSight AI');
}

// ---------------------------------------------------------------------------
// Gemini Event Handlers
// ---------------------------------------------------------------------------

client.onAudio = (base64Audio) => {
    media.playAudio(base64Audio);
};

// Use appendAssistantText for streaming — accumulates into one bubble
client.onTranscript = (role, text) => {
    if (!text || !text.trim()) return;

    if (role === 'user') {
        addMessage('user', text.trim());
    } else {
        // Assistant transcripts stream in chunks — accumulate them
        appendAssistantText(text);
    }
};

client.onToolCall = (name, args) => {
    addToolIndicator(name);
    console.log('[ToolCall]', name, args);
};

client.onToolResult = (name, result) => {
    removeToolIndicator(name);
    console.log('[ToolResult]', name, result);
    console.warn('[ToolResult Debug]', JSON.stringify(result, null, 2));

    // Route tool results to the appropriate renderer
    switch (name) {
        case 'analyze_symptom_image':
        case 'analyze_camera_frame':
            renderDiagnosis(result);
            break;
        case 'get_drug_interactions':
            renderDrugInfo(result);
            break;
        case 'get_clinical_guidelines':
            renderGuidelines(result);
            break;
        case 'risk_assessment':
            renderRiskAssessment(result);
            break;
    }
};

client.onStatus = (status, message) => {
    if (status === 'connected') {
        setConnectionState('connected');
        addMessage('system', 'Connected — speak naturally or type a message');
    } else if (status === 'reconnecting') {
        // Backend is auto-reconnecting the Gemini session
        addMessage('system', `🔄 ${message}`);
    } else if (status === 'disconnected') {
        if (state === 'connected') {
            // Backend-initiated disconnect
            media.stop();
            client.disconnect();
            setConnectionState('idle');
            lastAssistantBubble = null;
            addMessage('system', 'Session ended — click Connect to restart');
        }
    }
    console.log('[Status]', status, message);
};

client.onTurnComplete = () => {
    console.log('[TurnComplete]');
    lastAssistantBubble = null; // next transcript starts a fresh bubble
};

client.onInterrupted = () => {
    console.log('[Interrupted] — barge-in detected');
    media.stopPlayback();
    lastAssistantBubble = null;
};

client.onError = (message) => {
    console.error('[Error]', message);
    addMessage('system', `Error: ${message}`);
};

// ---------------------------------------------------------------------------
// UI Event Listeners
// ---------------------------------------------------------------------------

elements.connectBtn.addEventListener('click', connect);
elements.disconnectBtn.addEventListener('click', disconnect);

elements.sendTextBtn.addEventListener('click', () => {
    const text = elements.textInput.value.trim();
    if (text) {
        client.sendText(text);
        addMessage('user', text);
        elements.textInput.value = '';
    }
});

elements.textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        elements.sendTextBtn.click();
    }
});

elements.clearTranscriptBtn.addEventListener('click', () => {
    elements.transcriptMessages.innerHTML = '';
    clearInsights();
    lastAssistantBubble = null;
    if (state === 'idle') {
        elements.transcriptMessages.innerHTML = `
            <div class="welcome-message">
                <div class="welcome-icon">
                    <svg width="40" height="40" viewBox="0 0 28 28" fill="none">
                        <circle cx="14" cy="14" r="12" stroke="url(#grad2)" stroke-width="2" fill="none"/>
                        <path d="M14 6 L14 22 M8 11 L20 11 M8 17 L20 17" stroke="url(#grad2)" stroke-width="1.5" stroke-linecap="round"/>
                        <defs>
                            <linearGradient id="grad2" x1="0" y1="0" x2="28" y2="28">
                                <stop offset="0%" stop-color="#00d4ff"/>
                                <stop offset="100%" stop-color="#7c3aed"/>
                            </linearGradient>
                        </defs>
                    </svg>
                </div>
                <h3>Welcome to MediSight</h3>
                <p>Click <strong>Connect</strong> to start your clinical decision support session. Speak naturally and show symptoms via webcam.</p>
            </div>
        `;
    }
});

// ---------------------------------------------------------------------------
// Audio Visualizer (simple waveform)
// ---------------------------------------------------------------------------

function initVisualizer() {
    const canvas = elements.visualizerCanvas;
    const ctx = canvas.getContext('2d');
    const bars = 20;
    let heights = new Array(bars).fill(2);

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const barWidth = canvas.width / bars - 2;
        const centerY = canvas.height / 2;

        for (let i = 0; i < bars; i++) {
            const x = i * (barWidth + 2);

            if (state === 'connected') {
                heights[i] += (Math.random() - 0.5) * 4;
                heights[i] = Math.max(2, Math.min(canvas.height * 0.8, heights[i]));
            } else {
                heights[i] *= 0.9;
                heights[i] = Math.max(2, heights[i]);
            }

            const h = heights[i];
            const gradient = ctx.createLinearGradient(0, centerY - h / 2, 0, centerY + h / 2);
            gradient.addColorStop(0, 'rgba(0, 212, 255, 0.8)');
            gradient.addColorStop(1, 'rgba(124, 58, 237, 0.8)');
            ctx.fillStyle = gradient;

            ctx.beginPath();
            ctx.roundRect(x, centerY - h / 2, barWidth, h, 2);
            ctx.fill();
        }

        requestAnimationFrame(draw);
    }

    draw();
}

// ---------------------------------------------------------------------------
// Initialize
// ---------------------------------------------------------------------------

initVisualizer();
console.log('[MediSight] Application initialized');
