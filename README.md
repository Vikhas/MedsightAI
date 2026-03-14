# 🏥 MediSight — Real-Time Clinical Decision Assistant

> **Gemini Live Agent Challenge Hackathon** — A multimodal AI assistant that helps doctors analyze patient symptoms during rounds using voice and vision.

[![Built with Gemini](https://img.shields.io/badge/Built%20with-Gemini%20Live%20API-4285F4?style=flat-square&logo=google)](https://ai.google.dev/gemini-api/docs/live)
[![Google Cloud](https://img.shields.io/badge/Hosted%20on-Google%20Cloud%20Run-4285F4?style=flat-square&logo=googlecloud)](https://cloud.google.com/run)
[![Python](https://img.shields.io/badge/Backend-FastAPI%20%2B%20Python-3776AB?style=flat-square&logo=python)](https://fastapi.tiangolo.com)

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution](#-solution)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Features](#-features)
- [Demo Scenario](#-demo-scenario)
- [Getting Started](#-getting-started)
- [Deploying to Google Cloud](#-deploying-to-google-cloud)
- [Google Cloud Services Used](#-google-cloud-services-used)
- [Hackathon Requirements Checklist](#-hackathon-requirements-checklist)
- [Project Structure](#-project-structure)

---

## 🔍 Problem Statement

During hospital rounds, physicians must rapidly assess patient symptoms, recall drug interactions, reference clinical guidelines, and make critical decisions — often under time pressure with limited access to reference materials. Current tools require manual lookup and switching between multiple systems.

## 💡 Solution

**MediSight** is a real-time clinical decision support assistant powered by **Gemini Live API**. A doctor opens the web app and:

1. 🗣️ **Speaks** to the AI assistant naturally
2. 📸 **Shows** symptoms (rash, wound, X-ray) through the webcam
3. 🧠 **AI analyzes** both voice and image inputs in real-time
4. 🔊 **AI responds** with spoken clinical insights
5. ⚡ **Doctor can interrupt** at any time (barge-in support)

The AI provides differential diagnoses, severity assessments, drug interaction checks, and clinical guideline references — all through a natural voice conversation.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Browser (Web App)                 │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │ Webcam   │  │ Mic      │  │ Clinical Insights  │  │
│  │ (JPEG)   │  │ (PCM     │  │ Panel              │  │
│  │          │  │  16kHz)  │  │                    │  │
│  └────┬─────┘  └────┬─────┘  └───────────▲────────┘  │
│       │              │                    │           │
│       └──────┬───────┘                    │           │
│              ▼                            │           │
│       ┌──────────────┐                    │           │
│       │  WebSocket   │◄───────────────────┘           │
│       │  Client      │                                │
│       └──────┬───────┘                                │
└──────────────┼────────────────────────────────────────┘
               │ WebSocket (wss://)
               ▼
┌──────────────────────────────────────────────────────┐
│           FastAPI Backend (Cloud Run)                 │
│  ┌───────────────────────────────────────────────┐   │
│  │              WebSocket Proxy                   │   │
│  └──────────────────┬────────────────────────────┘   │
│                     │                                 │
│  ┌──────────────────▼────────────────────────────┐   │
│  │          GeminiLiveSession                     │   │
│  │  ┌─────────────┐  ┌────────────────────────┐  │   │
│  │  │ google-genai │  │ Tool Call Handler       │  │   │
│  │  │ SDK          │  │ (agent_tools.py)        │  │   │
│  │  └──────┬──────┘  └──────────┬─────────────┘  │   │
│  └─────────┼────────────────────┼────────────────┘   │
└────────────┼────────────────────┼────────────────────┘
             │                    │
             ▼                    ▼
    ┌────────────────┐   ┌───────────────────────┐
    │ Gemini Live API│   │ Clinical Agent Tools   │
    │ (gemini-2.5-   │   │ • analyze_symptom      │
    │  flash-native- │   │ • drug_interactions    │
    │  audio-dialog) │   │ • clinical_guidelines  │
    └────────────────┘   │ • risk_assessment      │
                         └───────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **AI Model** | Gemini 2.5 Flash (Native Audio Dialog) |
| **Live API** | Gemini Live API (WebSocket, real-time multimodal) |
| **SDK** | Google GenAI Python SDK (`google-genai`) |
| **Backend** | Python 3.11 + FastAPI + Uvicorn |
| **Frontend** | Vanilla JS + Web Audio API + MediaDevices API |
| **Styling** | Custom CSS (Dark Medical Theme) |
| **Deployment** | Google Cloud Run |
| **Container** | Docker |
| **CI/CD** | Google Cloud Build |

---

## ✨ Features

### Multimodal Interaction
- 🎤 **Voice input** — speak naturally to the AI
- 📹 **Webcam video** — show symptoms, X-rays, wounds
- 🔊 **Voice output** — AI responds with natural speech
- ⚡ **Barge-in** — interrupt the AI at any time

### Clinical Agent Tools
- 🔬 **Symptom Analysis** — differential diagnoses from visual observation
- 💊 **Drug Interactions** — safety checks with allergy awareness
- 📋 **Clinical Guidelines** — evidence-based treatment protocols
- ⚠️ **Risk Assessment** — severity scoring with triage recommendations

### Premium UI
- 🌙 Dark medical theme with glassmorphism
- 📊 Real-time confidence bars for diagnoses
- 🚦 Color-coded risk level banners
- 💬 Live conversation transcript
- 🎵 Audio waveform visualizer

---

## 🎬 Demo Scenario

### Scene 1: Visual Symptom Analysis

> **Doctor** opens MediSight and points the webcam at a rash.
>
> **Doctor**: *"MediSight, what do you think about this rash on the patient's forearm?"*
>
> **MediSight**: *"I can see what appears to be an erythematous, raised rash on the forearm. Let me run an analysis..."*
>
> The Clinical Insights panel populates with:
> - **Contact Dermatitis** — 75% confidence
> - **Cellulitis** — 60% confidence
> - **Allergic Reaction** — 55% confidence
> - Recommended tests: Skin biopsy, CBC, IgE levels
> - ⚠️ Warning: Watch for rapid spreading, fever

### Scene 2: Drug Interaction Check (Interruption)

> **Doctor** interrupts: *"Wait — the patient is allergic to penicillin. What antibiotics are safe?"*
>
> MediSight immediately stops speaking and responds:
>
> **MediSight**: *"Given the penicillin allergy, amoxicillin is contraindicated. Safe alternatives include azithromycin, doxycycline, or trimethoprim-sulfamethoxazole..."*
>
> The Drug Safety panel shows: **✕ CONTRAINDICATED** for Amoxicillin with alternatives listed.

### Scene 3: Clinical Guidelines

> **Doctor**: *"What are the current guidelines for treating cellulitis?"*
>
> **MediSight**: *"According to IDSA 2024 guidelines, first-line treatment for cellulitis is Cephalexin 500mg orally four times daily for 7 to 10 days. For penicillin-allergic patients, Clindamycin 300mg three times daily is recommended..."*

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- A [Google AI Studio API key](https://aistudio.google.com/apikey)

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/medisight.git
cd medisight
```

### 2. Set up the backend

```bash
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Set your API key
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 3. Run locally

```bash
# From the backend/ directory
python main.py
```

### 4. Open the app

Navigate to **http://localhost:8000** in your browser.

> **Note:** HTTPS is required for camera/microphone access in production. Localhost is exempt from this requirement during development.

### 5. Use MediSight

1. Click **Connect** to start the session
2. Grant camera and microphone permissions
3. Speak to the AI or type a message
4. Show symptoms via webcam for visual analysis

---

## ☁️ Deploying to Google Cloud

### Prerequisites

- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed
- A GCP project with billing enabled
- `GEMINI_API_KEY` environment variable set

### Deploy with the script

```bash
# Set your API key
export GEMINI_API_KEY=your_key_here

# Deploy (uses current gcloud project)
chmod +x infrastructure/deploy.sh
./infrastructure/deploy.sh

# Or specify project and region
./infrastructure/deploy.sh my-project-id us-central1
```

### Deploy manually

```bash
# Build and push container
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/medisight

# Deploy to Cloud Run
gcloud run deploy medisight \
    --image gcr.io/YOUR_PROJECT_ID/medisight \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --set-env-vars "GEMINI_API_KEY=${GEMINI_API_KEY}" \
    --memory 1Gi \
    --timeout 3600 \
    --port 8080
```

---

## 🌐 Google Cloud Services Used

| Service | Purpose |
|---------|---------|
| **Cloud Run** | Hosts the FastAPI backend with auto-scaling |
| **Cloud Build** | CI/CD pipeline for container builds |
| **Container Registry** | Stores the Docker container image |
| **Vertex AI** | Gemini model access via Google AI Studio |

---

## ✅ Hackathon Requirements Checklist

| Requirement | Status | Implementation |
|-------------|--------|---------------|
| Uses a Gemini model | ✅ | `gemini-2.5-flash-preview-native-audio-dialog` |
| Uses Gemini Live API | ✅ | Real-time WebSocket session via `google-genai` SDK |
| Uses Google GenAI SDK | ✅ | `google-genai` Python SDK for session management |
| Uses Google Cloud service | ✅ | Cloud Run, Cloud Build, Container Registry |
| Multimodal interaction | ✅ | Voice (mic) + Vision (webcam) + AI speech output |
| Live Agent with natural speech | ✅ | Full-duplex audio conversation |
| Supports interruption (barge-in) | ✅ | Built-in Gemini Live API barge-in support |
| Backend hosted on Google Cloud | ✅ | Deployed on Cloud Run |
| New project for hackathon | ✅ | Built from scratch |

---

## 📁 Project Structure

```
medisight/
├── backend/
│   ├── main.py              # FastAPI server + WebSocket endpoint
│   ├── gemini_live.py        # Gemini Live API session wrapper
│   ├── agent_tools.py        # Clinical reasoning tool functions
│   ├── requirements.txt      # Python dependencies
│   └── .env.example          # Environment variable template
├── frontend/
│   ├── index.html            # Main web page
│   ├── css/
│   │   └── styles.css        # Premium dark medical theme
│   └── js/
│       ├── main.js           # Application controller
│       ├── gemini-client.js  # WebSocket client
│       ├── media-handler.js  # Audio/video capture & playback
│       └── pcm-processor.js  # AudioWorklet for PCM audio
├── prompts/
│   └── system_prompt.txt     # MediSight clinical system prompt
├── infrastructure/
│   ├── deploy.sh             # Cloud Run deployment script
│   └── cloudbuild.yaml       # Cloud Build CI/CD config
├── Dockerfile                # Container definition
├── .gitignore
└── README.md
```

---

## ⚖️ Disclaimer

MediSight is a **demonstration project** built for the Gemini Live Agent Challenge hackathon. It is **not** a certified medical device and should **not** be used for actual clinical decision-making. All clinical tool responses are mock implementations for demonstration purposes.

---

<p align="center">
  Built with ❤️ for the <strong>Gemini Live Agent Challenge</strong><br>
  Powered by <strong>Google Gemini</strong> and <strong>Google Cloud</strong>
</p>
