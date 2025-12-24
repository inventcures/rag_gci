# Gemini Live API Voice AI Integration - Detailed Specifications

---

## IMPORTANT: Architecture Update (December 2024)

> **Palli Sahayak Voice AI Agent Helpline** now uses a hybrid architecture:
> - **Primary Orchestrator**: [Bolna.ai](https://bolna.ai) - for production telephony (phone calls)
> - **Fallback**: Gemini Live API - for web-based voice and when Bolna is unavailable
>
> **See `bolna_palli-sahayak-helpline_specs.md` for the complete Bolna integration specifications.**
>
> This document covers the Gemini Live API implementation, which serves as the web voice interface and fallback system.

---

## QUICK START: Google Cloud & Vertex AI Setup Guide

This section provides step-by-step instructions to configure Google Cloud Platform (GCP) and Vertex AI for the Gemini Live API integration.

### Prerequisites

- Google Cloud account with billing enabled
- `gcloud` CLI installed ([Install Guide](https://cloud.google.com/sdk/docs/install))
- Python 3.10+ with pip
- Admin access to create projects and enable APIs

---

### Step 1: Create a Google Cloud Project

```bash
# Set your project ID (choose a unique name)
export PROJECT_ID="your-palliative-care-rag"

# Create new project
gcloud projects create $PROJECT_ID --name="Palliative Care RAG"

# Set as default project
gcloud config set project $PROJECT_ID

# Link billing account (required for Vertex AI)
# List available billing accounts
gcloud billing accounts list

# Link billing (replace BILLING_ACCOUNT_ID with your account ID)
gcloud billing projects link $PROJECT_ID --billing-account=BILLING_ACCOUNT_ID
```

---

### Step 2: Enable Required APIs

```bash
# Enable all required APIs
gcloud services enable \
    aiplatform.googleapis.com \
    storage.googleapis.com \
    cloudresourcemanager.googleapis.com \
    iam.googleapis.com \
    serviceusage.googleapis.com

# Verify APIs are enabled
gcloud services list --enabled --filter="aiplatform"
```

**APIs Explained:**
| API | Purpose |
|-----|---------|
| `aiplatform.googleapis.com` | Vertex AI including Gemini Live API |
| `storage.googleapis.com` | Cloud Storage for audio/data |
| `cloudresourcemanager.googleapis.com` | Project management |
| `iam.googleapis.com` | Service account management |

---

### Step 3: Create Service Account & Credentials

```bash
# Create service account for the application
gcloud iam service-accounts create gemini-live-sa \
    --display-name="Gemini Live Service Account" \
    --description="Service account for Gemini Live API access"

# Grant required roles
export SA_EMAIL="gemini-live-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# Vertex AI User role (for Gemini Live API)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/aiplatform.user"

# Storage Object Viewer (if using Cloud Storage for audio)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/storage.objectViewer"

# Create and download key file
gcloud iam service-accounts keys create ./credentials/gcp-key.json \
    --iam-account=${SA_EMAIL}

# Secure the key file
chmod 600 ./credentials/gcp-key.json
```

---

### Step 4: Configure Environment Variables

Create or update your `.env` file with the following:

```bash
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=your-palliative-care-rag
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=./credentials/gcp-key.json

# Alternative: Use API Key instead of service account
# GEMINI_API_KEY=your-api-key-here

# Gemini Live Model (check for latest version)
GEMINI_LIVE_MODEL=gemini-2.0-flash-live-001
```

**Available Regions for Gemini Live:**
| Region | Location Code |
|--------|---------------|
| US Central | `us-central1` |
| US East | `us-east4` |
| Europe West | `europe-west1` |
| Asia Northeast | `asia-northeast1` |

> **Note:** Check [Vertex AI Regional Availability](https://cloud.google.com/vertex-ai/docs/general/locations) for the latest supported regions.

---

### Step 5: Install Python Dependencies

```bash
# Activate your virtual environment
source venv/bin/activate

# Install Google Cloud AI SDK
pip install google-genai>=1.0.0

# Install additional dependencies
pip install websockets>=12.0 pyyaml>=6.0.1 numpy>=1.24.0

# Or install all from requirements
pip install -r requirements_simple.txt
```

---

### Step 6: Authenticate with Google Cloud

**Option A: Service Account Key (Recommended for Production)**
```bash
# Set environment variable to key file path
export GOOGLE_APPLICATION_CREDENTIALS="./credentials/gcp-key.json"

# Verify authentication
python3 -c "from google.auth import default; creds, project = default(); print(f'Authenticated to: {project}')"
```

**Option B: User Account (For Development)**
```bash
# Login with your Google account
gcloud auth application-default login

# This creates credentials at ~/.config/gcloud/application_default_credentials.json
```

**Option C: API Key (Simplest)**
```bash
# Get API key from Google AI Studio: https://aistudio.google.com/apikey
export GEMINI_API_KEY="your-api-key-here"
```

---

### Step 7: Verify Gemini Live API Access

Run this test script to verify your setup:

```python
#!/usr/bin/env python3
"""Test Gemini Live API connection."""
import os
from google import genai

# Initialize client
client = genai.Client(
    # Uses GOOGLE_APPLICATION_CREDENTIALS or GEMINI_API_KEY automatically
)

# Check available models
print("Available Gemini models:")
for model in client.models.list():
    if "live" in model.name.lower() or "flash" in model.name.lower():
        print(f"  - {model.name}")

# Test basic generation
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Say 'Hello, Gemini Live is working!' in Hindi"
)
print(f"\nTest response: {response.text}")

print("\n✅ Gemini API connection successful!")
```

Save as `test_gemini_connection.py` and run:
```bash
python3 test_gemini_connection.py
```

---

### Step 8: Configure Gemini Live in config.yaml

Update your `config.yaml` to enable Gemini Live:

```yaml
# Gemini Live API Configuration
gemini_live:
  enabled: true  # Set to true to enable
  project_id: ${GOOGLE_CLOUD_PROJECT}
  location: "us-central1"

  # Model selection (check for latest versions)
  model: "gemini-2.0-flash-live-001"

  # Voice settings
  default_voice: "Aoede"  # Warm, empathetic voice for healthcare

  # Supported languages for Indian users
  default_language: "en-IN"
  supported_languages:
    - code: "en-IN"
      name: "English (India)"
    - code: "hi-IN"
      name: "Hindi"
    - code: "mr-IN"
      name: "Marathi"
    - code: "ta-IN"
      name: "Tamil"

  # Session settings
  session_timeout_minutes: 14  # Max 15 for Gemini
  max_sessions_per_user: 1

  # Audio settings
  input_sample_rate: 16000   # 16kHz for input
  output_sample_rate: 24000  # 24kHz for output
  chunk_size: 4096

  # RAG integration
  rag_context_enabled: true
  rag_top_k: 3

  # Fallback to STT+LLM+TTS pipeline on error
  fallback_enabled: true
```

---

### Step 9: Test the Full Integration

```bash
# Start the server
python3 simple_rag_server.py --port 8000

# Expected output should include:
# 🎤 Voice UI: http://localhost:8000/voice
# 🔊 Voice WebSocket: ws://localhost:8000/ws/voice
# Gemini Live initialized - WebSocket endpoint /ws/voice available
```

Then:
1. Open http://localhost:8000/voice in your browser
2. Allow microphone access when prompted
3. Click the microphone button and speak
4. The system should respond with voice

---

### Step 10: Production Deployment Checklist

- [ ] **Security**: Store credentials in Secret Manager, not files
  ```bash
  # Create secret
  gcloud secrets create gemini-live-credentials \
      --data-file=./credentials/gcp-key.json

  # Grant access to service account
  gcloud secrets add-iam-policy-binding gemini-live-credentials \
      --member="serviceAccount:${SA_EMAIL}" \
      --role="roles/secretmanager.secretAccessor"
  ```

- [ ] **Monitoring**: Enable Cloud Monitoring
  ```bash
  gcloud services enable monitoring.googleapis.com
  ```

- [ ] **Logging**: Enable Cloud Logging
  ```bash
  gcloud services enable logging.googleapis.com
  ```

- [ ] **Quotas**: Check and request quota increases if needed
  - Go to: https://console.cloud.google.com/iam-admin/quotas
  - Filter by "Vertex AI"
  - Request increases for production workloads

- [ ] **Budget Alerts**: Set up billing alerts
  ```bash
  # Create budget alert (example: $100/month)
  gcloud billing budgets create \
      --billing-account=BILLING_ACCOUNT_ID \
      --display-name="Gemini Live Budget" \
      --budget-amount=100USD \
      --threshold-rule=percent=50 \
      --threshold-rule=percent=90
  ```

- [ ] **HTTPS**: Use HTTPS in production for WebSocket (wss://)

- [ ] **Rate Limiting**: Implement rate limiting for voice endpoints

---

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `Permission denied` | Check IAM roles: `gcloud projects get-iam-policy $PROJECT_ID` |
| `API not enabled` | Enable API: `gcloud services enable aiplatform.googleapis.com` |
| `Quota exceeded` | Request quota increase in Cloud Console |
| `Invalid credentials` | Regenerate key: `gcloud iam service-accounts keys create...` |
| `Model not found` | Check model name: `client.models.list()` |
| `Region not supported` | Try `us-central1` (most features available) |
| `WebSocket connection failed` | Check firewall rules and CORS settings |
| `Audio not playing` | Verify sample rates (16kHz in, 24kHz out) |

### Cost Estimation

| Usage | Estimated Cost |
|-------|----------------|
| Gemini Live API | ~$0.075 per minute of audio |
| Light usage (100 min/day) | ~$225/month |
| Medium usage (500 min/day) | ~$1,125/month |
| Heavy usage (2000 min/day) | ~$4,500/month |

> **Tip:** Use `fallback_enabled: true` to fall back to cheaper STT+LLM+TTS pipeline during high-load periods.

### Useful Links

- [Gemini Live API Documentation](https://ai.google.dev/gemini-api/docs/live)
- [Vertex AI Pricing](https://cloud.google.com/vertex-ai/pricing)
- [Supported Languages](https://cloud.google.com/vertex-ai/generative-ai/docs/live-api/configure-language-voice#languages_supported)
- [Google Cloud Console](https://console.cloud.google.com/)
- [API Key Management](https://aistudio.google.com/apikey)

---

## Document Overview
This document contains the complete specifications for implementing voice AI capabilities using Google's Gemini Live API for the Palliative Care RAG system.

---

## PART 1: USER PROMPT

```
HELP ME IMPLEMENT A NEW MAJOR FEATURE

1. I WANT USERS WANTING PALLIATIVE CARE (OR HEALTH CARE ADVICE / RESPONSES TO
   HEALTH CARE QUERIES IN GENERAL) TO BE ABLE TO TALK TO A VOICE AI AGENT POWERED BY
   GEMINI LIVE API
   https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api/configure-language-voice#languages_supported

   WITH SUPPORT FOR INDIAN ENGLISH, HINDI, MARATHI, TAMIL

2. READ DETAILED DOCUMENTATION ABOUT GEMINI LIVE API
   - https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api/start-manage-session
   - https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api/send-audio-video-streams
   - https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api/configure-language-voice

3. MY HIGH LEVEL ARCHITECTURE UNDERSTANDING IS:
   A. IMPLEMENT CODE FOR CLIENT SIDE SUPPORT IN THE CODEBASE
   B. SERVER SIDE WILL BE HANDLED BY VERTEX

4. WRITE A DETAILED CLAUDE.MD and PLAN+SPECS IN live_call_detailed_specifications.md
   CONTAINING DETAILED SPECS AND PLAN, WHICH WHEN FED TO YOU TO IMPLEMENT,
   PROGRESSIVELY STEP BY STEP, WILL ENABLE THE CODEBASE TO IMPLEMENT THIS
   FUNCTIONALITY IN TOTAL

5. WRITE THIS DETAILED PROMPT AND YOUR RESPONSE TO IT IN live_call_detailed_specifications.md

6. USE CLAUDE DEEP RESEARCH, GOOGLE SEARCH TOOLS, TO LOOK UP ANY & ALL REQUIRED
   GEMINI LIVE AND OTHER DOCUMENTATION, TUTORIALS AND EXAMPLES
```

---

## PART 2: CLAUDE'S RESPONSE - COMPREHENSIVE SPECIFICATIONS

### Executive Summary

After thorough research of the Gemini Live API documentation and analysis of your existing codebase, I've designed a comprehensive integration plan. The system will enable real-time voice conversations for palliative care queries with support for **Indian English (en-IN), Hindi (hi-IN), Marathi (mr-IN), and Tamil (ta-IN)**.

---

## 1. CURRENT CODEBASE ARCHITECTURE ANALYSIS

### Existing System Components

| Component | File | Purpose |
|-----------|------|---------|
| RAG Pipeline | `simple_rag_server.py` | Document retrieval, embedding, query processing |
| WhatsApp Bot | `whatsapp_bot.py` | Twilio integration, message handling |
| Smart Clarification | `smart_clarification_system.py` | Query disambiguation |
| Configuration | `config.yaml` | System settings |

### Current Voice Flow
```
WhatsApp Voice Message → Twilio Webhook → Download Audio →
Groq Whisper STT → RAG Query → LLM Response →
Edge TTS → Send Audio Back
```

### Limitations of Current Approach
1. **High Latency**: Sequential STT → LLM → TTS pipeline
2. **No Real-Time Streaming**: Full message required before response
3. **No Interruption Support**: Cannot handle user barge-in
4. **Limited Emotional Context**: Loses tone/emotion in transcription

---

## 2. GEMINI LIVE API OVERVIEW

### What is Gemini Live API?
Gemini Live API is Google's low-latency, real-time multimodal API that enables:
- **Native audio processing**: Single model handles STT + LLM + TTS
- **Real-time streaming**: Sub-second response latency
- **Barge-in support**: Users can interrupt mid-response
- **Affective dialogue**: Understands tone, emotion, and pace
- **Multi-language**: Supports 24 languages including Indian languages

### Key Technical Specifications

| Parameter | Value |
|-----------|-------|
| Input Audio | Raw 16-bit PCM, 16kHz, mono, little-endian |
| Output Audio | Raw 16-bit PCM, 24kHz, mono, little-endian |
| Connection | WebSocket (WSS) |
| Max Session (Audio) | 15 minutes |
| Context Window | 128k tokens (native audio models) |
| Concurrent Sessions | 1,000 per project (PayGo) |

### Supported Indian Languages

| Language | Code | Supported |
|----------|------|-----------|
| English (India) | en-IN | Yes |
| Hindi (India) | hi-IN | Yes |
| Marathi (India) | mr-IN | Yes |
| Tamil (India) | ta-IN | Yes |
| Telugu (India) | te-IN | Yes (bonus) |
| Bengali (India) | bn-BD | Yes (bonus) |

### Available Voice Options (30 voices)
- **Zephyr** - Bright
- **Kore** - Firm
- **Puck** - Upbeat
- **Fenrir** - Excitable
- **Charon** - Informative
- **Aoede** - Easy-going (recommended for healthcare)

---

## 3. PROPOSED ARCHITECTURE

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACES                               │
├──────────────────┬──────────────────┬───────────────────────────────┤
│   WhatsApp       │   Web Browser    │   Phone (PSTN)                │
│   (Voice Msg)    │   (WebRTC)       │   (via Twilio)                │
└────────┬─────────┴────────┬─────────┴────────────┬──────────────────┘
         │                  │                      │
         ▼                  ▼                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND SERVER                            │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                 GeminiLiveService                            │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │    │
│  │  │ Session Mgr  │  │ Audio Stream │  │ Context Manager  │   │    │
│  │  │              │  │ Handler      │  │ (RAG Integration)│   │    │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Existing RAG Pipeline                           │    │
│  │  ChromaDB │ Embeddings │ Document Store │ Query Engine       │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   GOOGLE VERTEX AI                                   │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Gemini Live API                                 │    │
│  │  gemini-live-2.5-flash-preview-native-audio                  │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │    │
│  │  │ Native STT   │  │ LLM Engine   │  │ Native TTS       │   │    │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow for Voice Conversation
```
1. User speaks (any interface)
           │
           ▼
2. Audio captured as PCM (16kHz, 16-bit)
           │
           ▼
3. Stream to Backend via WebSocket
           │
           ▼
4. Backend fetches RAG context
   ├─ Query vector DB with recent context
   └─ Inject as system instruction
           │
           ▼
5. Forward to Gemini Live API
   ├─ WebSocket connection to Vertex AI
   └─ Audio streaming with RAG context
           │
           ▼
6. Gemini processes audio natively
   ├─ Understands speech (no separate STT)
   ├─ Generates response with RAG context
   └─ Produces speech output (no separate TTS)
           │
           ▼
7. Stream audio response back
           │
           ▼
8. Play to user with <1 second latency
```

---

## 4. IMPLEMENTATION PLAN

### Phase 1: Core Infrastructure (Week 1)

#### 1.1 New Files to Create

```
rag_gci/
├── gemini_live/
│   ├── __init__.py
│   ├── service.py           # GeminiLiveService class
│   ├── session_manager.py   # Session lifecycle management
│   ├── audio_handler.py     # Audio format conversion
│   ├── context_manager.py   # RAG context injection
│   └── config.py            # Gemini-specific configuration
├── web_client/
│   ├── index.html           # Web interface for voice
│   ├── audio-worklet.js     # AudioWorklet for capture/playback
│   └── gemini-client.js     # WebSocket client
└── telephony/
    ├── twilio_voice.py      # PSTN call handling
    └── daily_bridge.py      # WebRTC bridge (optional)
```

#### 1.2 Dependencies to Add

```python
# requirements.txt additions
google-genai>=1.0.0          # Google Gen AI SDK
websockets>=12.0             # WebSocket client
pyaudio>=0.2.14              # Audio I/O (optional, for testing)
numpy>=1.24.0                # Audio processing
fastapi[all]>=0.109.0        # Already present, ensure updated
python-multipart>=0.0.6      # File uploads
```

#### 1.3 Environment Variables

```bash
# .env additions
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GEMINI_API_KEY=your-api-key        # Alternative to ADC
GEMINI_MODEL=gemini-live-2.5-flash-preview-native-audio-09-2025
GEMINI_VOICE=Aoede                 # Default voice
DAILY_API_KEY=your-daily-key       # For telephony (optional)
```

### Phase 2: Gemini Live Service Implementation

#### 2.1 Core Service Class (`gemini_live/service.py`)

```python
"""
GeminiLiveService - Core service for Gemini Live API integration

This module provides the main interface for real-time voice conversations
using Google's Gemini Live API with RAG context injection.
"""

import asyncio
import json
import base64
from typing import Optional, AsyncGenerator, Callable
from google import genai
from google.genai import types

class GeminiLiveService:
    """
    Manages real-time voice conversations with Gemini Live API.

    Features:
    - WebSocket-based streaming
    - RAG context injection
    - Multi-language support (en-IN, hi-IN, mr-IN, ta-IN)
    - Session resumption
    - Barge-in handling
    """

    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        model: str = "gemini-live-2.5-flash-preview-native-audio-09-2025",
        rag_pipeline: Optional[object] = None
    ):
        self.project_id = project_id
        self.location = location
        self.model = model
        self.rag_pipeline = rag_pipeline
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )
        self.active_sessions = {}

    async def create_session(
        self,
        session_id: str,
        language: str = "en-IN",
        voice: str = "Aoede",
        system_instruction: Optional[str] = None
    ) -> "GeminiLiveSession":
        """
        Create a new Gemini Live session.

        Args:
            session_id: Unique identifier for this session
            language: Language code (en-IN, hi-IN, mr-IN, ta-IN)
            voice: Voice name (Aoede, Puck, Kore, etc.)
            system_instruction: Custom system prompt

        Returns:
            GeminiLiveSession object
        """
        # Build system instruction with medical context
        base_instruction = self._build_medical_system_instruction(language)
        if system_instruction:
            base_instruction = f"{base_instruction}\n\n{system_instruction}"

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice
                    )
                ),
                language_code=language
            ),
            system_instruction=types.Content(
                parts=[types.Part(text=base_instruction)]
            ),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig()
        )

        session = GeminiLiveSession(
            service=self,
            session_id=session_id,
            config=config,
            language=language
        )

        self.active_sessions[session_id] = session
        return session

    def _build_medical_system_instruction(self, language: str) -> str:
        """Build the medical/palliative care system instruction."""

        language_instructions = {
            "en-IN": "Respond in Indian English with a warm, empathetic tone.",
            "hi-IN": "हिंदी में जवाब दें। गर्मजोशी और सहानुभूति के साथ बात करें।",
            "mr-IN": "मराठीत उत्तर द्या. सहानुभूती आणि काळजी घेणारा स्वर वापरा.",
            "ta-IN": "தமிழில் பதிலளிக்கவும். அன்பான மற்றும் பரிவான தொனியில் பேசுங்கள்."
        }

        return f"""You are a compassionate palliative care assistant helping patients
and caregivers with healthcare queries.

IMPORTANT GUIDELINES:
1. Be warm, empathetic, and supportive in all interactions
2. Provide accurate medical information from the knowledge base
3. Always recommend consulting healthcare professionals for serious concerns
4. Use simple, clear language appropriate for patients and families
5. Be culturally sensitive to Indian healthcare contexts
6. If unsure, acknowledge uncertainty and suggest professional consultation

LANGUAGE: {language_instructions.get(language, language_instructions["en-IN"])}

SAFETY: Never provide emergency medical advice. For emergencies,
direct users to call emergency services or visit the nearest hospital.
"""

    async def inject_rag_context(
        self,
        session: "GeminiLiveSession",
        query_context: str
    ) -> None:
        """
        Inject RAG-retrieved context into an active session.

        This method queries the RAG pipeline and sends relevant
        context to the Gemini session as a text message.
        """
        if not self.rag_pipeline:
            return

        # Query RAG for relevant documents
        result = await self.rag_pipeline.query(
            question=query_context,
            conversation_id=session.session_id,
            user_id=session.session_id,
            top_k=3
        )

        if result.get("status") == "success" and result.get("context_used"):
            context_message = f"""
[MEDICAL KNOWLEDGE BASE CONTEXT]
The following information from verified medical documents may be relevant:

{result['context_used']}

Use this information to provide accurate, evidence-based responses.
Always cite sources when using this information.
[END CONTEXT]
"""
            await session.send_text(context_message)


class GeminiLiveSession:
    """
    Represents an active Gemini Live session.

    Handles:
    - Audio streaming (send/receive)
    - Text messaging
    - Session lifecycle
    - Transcription capture
    """

    def __init__(
        self,
        service: GeminiLiveService,
        session_id: str,
        config: types.LiveConnectConfig,
        language: str
    ):
        self.service = service
        self.session_id = session_id
        self.config = config
        self.language = language
        self.session = None
        self.is_active = False
        self.transcription_buffer = []
        self.response_buffer = []

    async def connect(self) -> None:
        """Establish connection to Gemini Live API."""
        self.session = await self.service.client.aio.live.connect(
            model=self.service.model,
            config=self.config
        )
        self.is_active = True

    async def disconnect(self) -> None:
        """Close the session."""
        if self.session:
            await self.session.close()
        self.is_active = False
        if self.session_id in self.service.active_sessions:
            del self.service.active_sessions[self.session_id]

    async def send_audio(self, audio_chunk: bytes) -> None:
        """
        Send audio chunk to Gemini.

        Args:
            audio_chunk: Raw PCM audio (16kHz, 16-bit, mono)
        """
        if not self.is_active or not self.session:
            raise RuntimeError("Session not connected")

        await self.session.send_realtime_input(
            audio=types.Blob(
                data=audio_chunk,
                mime_type="audio/pcm;rate=16000"
            )
        )

    async def send_text(self, text: str) -> None:
        """Send text message to Gemini (for context injection)."""
        if not self.is_active or not self.session:
            raise RuntimeError("Session not connected")

        await self.session.send_client_content(
            turns=[types.Content(
                role="user",
                parts=[types.Part(text=text)]
            )],
            turn_complete=True
        )

    async def receive_audio(self) -> AsyncGenerator[bytes, None]:
        """
        Receive audio responses from Gemini.

        Yields:
            Raw PCM audio chunks (24kHz, 16-bit, mono)
        """
        if not self.is_active or not self.session:
            raise RuntimeError("Session not connected")

        async for message in self.session.receive():
            if message.server_content:
                content = message.server_content

                # Handle model turn (audio output)
                if content.model_turn:
                    for part in content.model_turn.parts:
                        if part.inline_data:
                            yield part.inline_data.data

                # Capture transcription
                if content.input_transcription:
                    self.transcription_buffer.append(
                        content.input_transcription.text
                    )
                if content.output_transcription:
                    self.response_buffer.append(
                        content.output_transcription.text
                    )

                # Check for turn complete
                if content.turn_complete:
                    yield b"__TURN_COMPLETE__"

            # Handle interruption
            if message.server_content and message.server_content.interrupted:
                yield b"__INTERRUPTED__"
```

#### 2.2 Audio Handler (`gemini_live/audio_handler.py`)

```python
"""
Audio format conversion utilities for Gemini Live API.

Handles conversion between:
- WhatsApp audio formats (OGG, MP3) → PCM
- PCM → MP3/OGG for responses
- Sample rate conversions (16kHz ↔ 24kHz)
"""

import io
import wave
import struct
import subprocess
from typing import Optional
import numpy as np

class AudioHandler:
    """
    Audio format conversion for Gemini Live API.

    Input Requirements:  16-bit PCM, 16kHz, mono, little-endian
    Output Format:       16-bit PCM, 24kHz, mono, little-endian
    """

    INPUT_SAMPLE_RATE = 16000
    OUTPUT_SAMPLE_RATE = 24000
    CHANNELS = 1
    SAMPLE_WIDTH = 2  # 16-bit

    @staticmethod
    def convert_to_pcm(
        audio_data: bytes,
        input_format: str = "ogg"
    ) -> bytes:
        """
        Convert audio to PCM format for Gemini Live.

        Args:
            audio_data: Input audio bytes
            input_format: Source format (ogg, mp3, wav, m4a)

        Returns:
            PCM audio bytes (16kHz, 16-bit, mono)
        """
        # Use FFmpeg for conversion
        cmd = [
            "ffmpeg",
            "-i", "pipe:0",           # Read from stdin
            "-f", "s16le",            # Output format: signed 16-bit little-endian
            "-acodec", "pcm_s16le",   # PCM codec
            "-ar", "16000",           # Sample rate: 16kHz
            "-ac", "1",               # Mono
            "-"                        # Output to stdout
        ]

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        pcm_data, error = process.communicate(input=audio_data)

        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg conversion failed: {error.decode()}")

        return pcm_data

    @staticmethod
    def convert_from_pcm(
        pcm_data: bytes,
        output_format: str = "mp3",
        sample_rate: int = 24000
    ) -> bytes:
        """
        Convert PCM audio from Gemini Live to playable format.

        Args:
            pcm_data: Raw PCM bytes (24kHz, 16-bit, mono)
            output_format: Target format (mp3, ogg, wav)
            sample_rate: Input sample rate (default 24kHz for Gemini output)

        Returns:
            Converted audio bytes
        """
        cmd = [
            "ffmpeg",
            "-f", "s16le",            # Input format
            "-ar", str(sample_rate),  # Input sample rate
            "-ac", "1",               # Mono
            "-i", "pipe:0",           # Read from stdin
            "-f", output_format,      # Output format
            "-"                        # Output to stdout
        ]

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        output_data, error = process.communicate(input=pcm_data)

        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg conversion failed: {error.decode()}")

        return output_data

    @staticmethod
    def resample(
        audio_data: bytes,
        from_rate: int,
        to_rate: int
    ) -> bytes:
        """Resample audio data using numpy interpolation."""
        samples = np.frombuffer(audio_data, dtype=np.int16)

        # Calculate new length
        new_length = int(len(samples) * to_rate / from_rate)

        # Resample using linear interpolation
        indices = np.linspace(0, len(samples) - 1, new_length)
        resampled = np.interp(indices, np.arange(len(samples)), samples)

        return resampled.astype(np.int16).tobytes()

    @staticmethod
    def chunk_audio(
        audio_data: bytes,
        chunk_size: int = 4096
    ) -> list[bytes]:
        """Split audio into chunks for streaming."""
        chunks = []
        for i in range(0, len(audio_data), chunk_size):
            chunks.append(audio_data[i:i + chunk_size])
        return chunks
```

#### 2.3 Session Manager (`gemini_live/session_manager.py`)

```python
"""
Session lifecycle management for Gemini Live.

Handles:
- Session creation and cleanup
- Session resumption (for long conversations)
- Concurrent session limits
- Timeout handling
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class SessionManager:
    """
    Manages lifecycle of Gemini Live sessions.

    Features:
    - Automatic session cleanup after timeout
    - Session resumption support
    - Concurrent session tracking
    - Health monitoring
    """

    DEFAULT_TIMEOUT = timedelta(minutes=14)  # Gemini max is 15
    MAX_SESSIONS_PER_USER = 1

    def __init__(self, gemini_service):
        self.service = gemini_service
        self.sessions: Dict[str, SessionInfo] = {}
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the session manager background tasks."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self):
        """Stop the session manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def get_or_create_session(
        self,
        user_id: str,
        language: str = "en-IN",
        voice: str = "Aoede"
    ):
        """
        Get existing session or create new one for user.

        Ensures only one session per user.
        """
        # Check for existing session
        if user_id in self.user_sessions:
            session_id = self.user_sessions[user_id]
            if session_id in self.sessions:
                session_info = self.sessions[session_id]
                if session_info.is_valid():
                    session_info.last_activity = datetime.now()
                    return session_info.session

        # Create new session
        session_id = f"{user_id}_{datetime.now().timestamp()}"
        session = await self.service.create_session(
            session_id=session_id,
            language=language,
            voice=voice
        )
        await session.connect()

        # Store session info
        self.sessions[session_id] = SessionInfo(
            session_id=session_id,
            user_id=user_id,
            session=session,
            created_at=datetime.now(),
            language=language
        )
        self.user_sessions[user_id] = session_id

        return session

    async def close_session(self, user_id: str):
        """Close and cleanup a user's session."""
        if user_id in self.user_sessions:
            session_id = self.user_sessions[user_id]
            if session_id in self.sessions:
                session_info = self.sessions[session_id]
                await session_info.session.disconnect()
                del self.sessions[session_id]
            del self.user_sessions[user_id]

    async def _cleanup_loop(self):
        """Background task to cleanup expired sessions."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                now = datetime.now()
                expired = []

                for session_id, info in self.sessions.items():
                    if not info.is_valid():
                        expired.append(info.user_id)

                for user_id in expired:
                    logger.info(f"Cleaning up expired session for {user_id}")
                    await self.close_session(user_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")


class SessionInfo:
    """Container for session metadata."""

    def __init__(
        self,
        session_id: str,
        user_id: str,
        session,
        created_at: datetime,
        language: str
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.session = session
        self.created_at = created_at
        self.last_activity = created_at
        self.language = language
        self.resumption_handle: Optional[str] = None

    def is_valid(self) -> bool:
        """Check if session is still valid (not timed out)."""
        timeout = SessionManager.DEFAULT_TIMEOUT
        return datetime.now() - self.last_activity < timeout
```

### Phase 3: WhatsApp Integration

#### 3.1 Enhanced WhatsApp Bot (`whatsapp_bot.py` modifications)

```python
# Add to whatsapp_bot.py

from gemini_live.service import GeminiLiveService
from gemini_live.session_manager import SessionManager
from gemini_live.audio_handler import AudioHandler

class EnhancedWhatsAppBot:
    """
    Enhanced WhatsApp bot with Gemini Live voice support.

    New capabilities:
    - Real-time voice conversations via Gemini Live
    - Streaming audio responses
    - Multi-language native audio support
    """

    def __init__(self, ...):
        # Existing initialization...

        # Add Gemini Live support
        self.gemini_live_enabled = config.get("gemini_live", {}).get("enabled", False)
        if self.gemini_live_enabled:
            self.gemini_service = GeminiLiveService(
                project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
                location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
                rag_pipeline=self.rag_pipeline
            )
            self.session_manager = SessionManager(self.gemini_service)
            self.audio_handler = AudioHandler()

    async def _handle_twilio_media_message_gemini(
        self,
        from_number: str,
        media_url: str,
        media_type: str
    ) -> dict:
        """
        Handle voice messages using Gemini Live API.

        Flow:
        1. Download audio from Twilio
        2. Convert to PCM format
        3. Get/create Gemini session
        4. Stream audio to Gemini
        5. Collect audio response
        6. Convert and send back to user
        """
        try:
            # 1. Download audio from Twilio
            audio_data = await self.twilio_api.download_media(media_url)

            # 2. Convert to PCM (16kHz, 16-bit, mono)
            pcm_data = self.audio_handler.convert_to_pcm(
                audio_data,
                input_format="ogg"
            )

            # 3. Get user's preferred language
            user_language = self.user_preferences.get(
                from_number, {}
            ).get("language", "en-IN")

            # Map to Gemini language codes
            language_map = {
                "hi": "hi-IN",
                "mr": "mr-IN",
                "ta": "ta-IN",
                "en": "en-IN"
            }
            gemini_language = language_map.get(user_language, "en-IN")

            # 4. Get or create Gemini session
            session = await self.session_manager.get_or_create_session(
                user_id=from_number,
                language=gemini_language,
                voice="Aoede"  # Empathetic voice for healthcare
            )

            # 5. Inject RAG context based on recent conversation
            if self.conversation_history.get(from_number):
                recent_context = " ".join([
                    msg.get("content", "")
                    for msg in self.conversation_history[from_number][-3:]
                ])
                await self.gemini_service.inject_rag_context(
                    session,
                    recent_context
                )

            # 6. Stream audio to Gemini
            audio_chunks = self.audio_handler.chunk_audio(pcm_data)
            for chunk in audio_chunks:
                await session.send_audio(chunk)

            # 7. Collect response audio
            response_audio = bytearray()
            transcription = ""

            async for audio_chunk in session.receive_audio():
                if audio_chunk == b"__TURN_COMPLETE__":
                    break
                elif audio_chunk == b"__INTERRUPTED__":
                    response_audio.clear()
                else:
                    response_audio.extend(audio_chunk)

            # Get transcription for logging
            if session.response_buffer:
                transcription = " ".join(session.response_buffer)
                session.response_buffer.clear()

            # 8. Convert response to MP3
            mp3_audio = self.audio_handler.convert_from_pcm(
                bytes(response_audio),
                output_format="mp3",
                sample_rate=24000
            )

            # 9. Save and send audio response
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_filename = f"gemini_response_{gemini_language}_{timestamp}.mp3"
            audio_path = f"cache/tts/{audio_filename}"

            with open(audio_path, "wb") as f:
                f.write(mp3_audio)

            # 10. Send via Twilio
            public_url = f"{self.ngrok_url}/media/{audio_filename}"
            await self.twilio_api.send_audio_message(
                to=from_number,
                audio_file_path=audio_path,
                public_url=public_url
            )

            # Also send text transcription if available
            if transcription:
                await self.twilio_api.send_text_message(
                    to=from_number,
                    message=f"[Transcript]: {transcription[:1500]}"
                )

            return {
                "status": "success",
                "method": "gemini_live",
                "language": gemini_language,
                "transcription": transcription
            }

        except Exception as e:
            logger.error(f"Gemini Live error: {e}")
            # Fallback to existing STT+RAG+TTS pipeline
            return await self._handle_twilio_media_message_fallback(
                from_number, media_url, media_type
            )
```

### Phase 4: Web Client Implementation

#### 4.1 Web Interface (`web_client/index.html`)

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Palliative Care Voice Assistant</title>
    <style>
        :root {
            --primary: #4A90A4;
            --secondary: #2C5F72;
            --success: #48BB78;
            --danger: #F56565;
            --bg: #F7FAFC;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .header {
            background: var(--primary);
            color: white;
            padding: 1rem;
            text-align: center;
        }

        .container {
            flex: 1;
            max-width: 600px;
            margin: 0 auto;
            padding: 1rem;
            display: flex;
            flex-direction: column;
        }

        .language-selector {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }

        .lang-btn {
            padding: 0.5rem 1rem;
            border: 2px solid var(--primary);
            background: white;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .lang-btn.active {
            background: var(--primary);
            color: white;
        }

        .conversation {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
            background: white;
            border-radius: 8px;
            margin-bottom: 1rem;
            min-height: 300px;
        }

        .message {
            margin-bottom: 1rem;
            padding: 0.75rem 1rem;
            border-radius: 12px;
            max-width: 85%;
        }

        .message.user {
            background: var(--primary);
            color: white;
            margin-left: auto;
        }

        .message.assistant {
            background: #E2E8F0;
            color: #2D3748;
        }

        .controls {
            display: flex;
            gap: 1rem;
            align-items: center;
        }

        .record-btn {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            border: none;
            background: var(--primary);
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        }

        .record-btn:hover {
            transform: scale(1.05);
        }

        .record-btn.recording {
            background: var(--danger);
            animation: pulse 1s infinite;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }

        .status {
            flex: 1;
            text-align: center;
            color: #718096;
        }

        .status.active {
            color: var(--success);
        }

        .waveform {
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 2px;
        }

        .waveform .bar {
            width: 4px;
            background: var(--primary);
            border-radius: 2px;
            transition: height 0.1s;
        }
    </style>
</head>
<body>
    <header class="header">
        <h1>Palliative Care Voice Assistant</h1>
        <p>Powered by Gemini Live</p>
    </header>

    <div class="container">
        <div class="language-selector">
            <button class="lang-btn active" data-lang="en-IN">English</button>
            <button class="lang-btn" data-lang="hi-IN">हिंदी</button>
            <button class="lang-btn" data-lang="mr-IN">मराठी</button>
            <button class="lang-btn" data-lang="ta-IN">தமிழ்</button>
        </div>

        <div class="conversation" id="conversation">
            <div class="message assistant">
                Hello! I'm your palliative care assistant. How can I help you today?
            </div>
        </div>

        <div class="controls">
            <button class="record-btn" id="recordBtn">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm-1 1.93c-3.94-.49-7-3.85-7-7.93V7h2v1c0 2.76 2.24 5 5 5s5-2.24 5-5V7h2v1c0 4.08-3.06 7.44-7 7.93V19h3v2H9v-2h3v-3.07z"/>
                </svg>
            </button>
            <div class="status" id="status">
                Tap to speak
            </div>
            <div class="waveform" id="waveform" style="display: none;">
                <!-- Waveform bars will be added dynamically -->
            </div>
        </div>
    </div>

    <script src="gemini-client.js"></script>
    <script>
        const conversation = document.getElementById('conversation');
        const recordBtn = document.getElementById('recordBtn');
        const status = document.getElementById('status');
        const waveform = document.getElementById('waveform');
        const langBtns = document.querySelectorAll('.lang-btn');

        let isRecording = false;
        let geminiClient = null;
        let currentLanguage = 'en-IN';

        // Initialize Gemini client
        async function init() {
            geminiClient = new GeminiLiveClient({
                serverUrl: `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/voice`,
                language: currentLanguage,
                onAudioReceived: handleAudioResponse,
                onTranscription: handleTranscription,
                onError: handleError,
                onStatusChange: updateStatus
            });

            await geminiClient.initialize();
        }

        // Language selection
        langBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                langBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentLanguage = btn.dataset.lang;
                if (geminiClient) {
                    geminiClient.setLanguage(currentLanguage);
                }
            });
        });

        // Record button handler
        recordBtn.addEventListener('click', async () => {
            if (!geminiClient) {
                await init();
            }

            if (!isRecording) {
                await geminiClient.startRecording();
                isRecording = true;
                recordBtn.classList.add('recording');
                status.textContent = 'Listening...';
                status.classList.add('active');
            } else {
                await geminiClient.stopRecording();
                isRecording = false;
                recordBtn.classList.remove('recording');
                status.textContent = 'Processing...';
            }
        });

        function handleAudioResponse(audioBlob) {
            // Create audio element and play
            const audio = new Audio(URL.createObjectURL(audioBlob));
            audio.play();
            status.textContent = 'Tap to speak';
            status.classList.remove('active');
        }

        function handleTranscription(data) {
            const { type, text } = data;

            if (type === 'user') {
                addMessage(text, 'user');
            } else if (type === 'assistant') {
                addMessage(text, 'assistant');
            }
        }

        function addMessage(text, role) {
            const msg = document.createElement('div');
            msg.className = `message ${role}`;
            msg.textContent = text;
            conversation.appendChild(msg);
            conversation.scrollTop = conversation.scrollHeight;
        }

        function handleError(error) {
            console.error('Gemini error:', error);
            status.textContent = 'Error occurred. Tap to retry.';
            isRecording = false;
            recordBtn.classList.remove('recording');
        }

        function updateStatus(newStatus) {
            status.textContent = newStatus;
        }

        // Initialize on load
        init().catch(console.error);
    </script>
</body>
</html>
```

#### 4.2 Gemini Client (`web_client/gemini-client.js`)

```javascript
/**
 * GeminiLiveClient - WebSocket client for Gemini Live voice conversations
 *
 * Handles:
 * - Audio capture using AudioWorklet
 * - WebSocket communication with backend
 * - Audio playback of responses
 * - Session management
 */

class GeminiLiveClient {
    constructor(options) {
        this.serverUrl = options.serverUrl;
        this.language = options.language || 'en-IN';
        this.onAudioReceived = options.onAudioReceived || (() => {});
        this.onTranscription = options.onTranscription || (() => {});
        this.onError = options.onError || console.error;
        this.onStatusChange = options.onStatusChange || (() => {});

        this.ws = null;
        this.audioContext = null;
        this.mediaStream = null;
        this.workletNode = null;
        this.isInitialized = false;
        this.audioQueue = [];
        this.isPlaying = false;
    }

    async initialize() {
        // Create AudioContext
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 16000  // Gemini input rate
        });

        // Load AudioWorklet for capture
        await this.audioContext.audioWorklet.addModule('audio-worklet.js');

        // Get microphone access
        this.mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true
            }
        });

        this.isInitialized = true;
    }

    async connect() {
        return new Promise((resolve, reject) => {
            this.ws = new WebSocket(this.serverUrl);
            this.ws.binaryType = 'arraybuffer';

            this.ws.onopen = () => {
                // Send initial configuration
                this.ws.send(JSON.stringify({
                    type: 'config',
                    language: this.language
                }));
                resolve();
            };

            this.ws.onmessage = (event) => {
                this.handleMessage(event.data);
            };

            this.ws.onerror = (error) => {
                this.onError(error);
                reject(error);
            };

            this.ws.onclose = () => {
                this.onStatusChange('Disconnected');
            };
        });
    }

    async startRecording() {
        if (!this.isInitialized) {
            await this.initialize();
        }

        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            await this.connect();
        }

        // Resume AudioContext if suspended
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }

        // Create source from microphone
        const source = this.audioContext.createMediaStreamSource(this.mediaStream);

        // Create AudioWorklet node for processing
        this.workletNode = new AudioWorkletNode(this.audioContext, 'audio-capture-processor');

        this.workletNode.port.onmessage = (event) => {
            if (event.data.type === 'audio') {
                // Send audio chunk to server
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(event.data.buffer);
                }
            }
        };

        source.connect(this.workletNode);
        this.workletNode.connect(this.audioContext.destination);

        // Notify server that we're starting
        this.ws.send(JSON.stringify({ type: 'start_audio' }));

        this.onStatusChange('Listening...');
    }

    async stopRecording() {
        if (this.workletNode) {
            this.workletNode.disconnect();
            this.workletNode = null;
        }

        // Notify server that we stopped
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'stop_audio' }));
        }

        this.onStatusChange('Processing...');
    }

    handleMessage(data) {
        // Binary data = audio response
        if (data instanceof ArrayBuffer) {
            this.audioQueue.push(data);
            this.playNextAudio();
            return;
        }

        // JSON messages
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

                case 'error':
                    this.onError(message.error);
                    break;
            }
        } catch (e) {
            console.error('Failed to parse message:', e);
        }
    }

    async playNextAudio() {
        if (this.isPlaying || this.audioQueue.length === 0) {
            return;
        }

        this.isPlaying = true;

        // Create playback context at 24kHz (Gemini output rate)
        const playbackContext = new AudioContext({ sampleRate: 24000 });

        while (this.audioQueue.length > 0) {
            const audioData = this.audioQueue.shift();

            // Convert ArrayBuffer to Float32Array
            const int16Data = new Int16Array(audioData);
            const float32Data = new Float32Array(int16Data.length);
            for (let i = 0; i < int16Data.length; i++) {
                float32Data[i] = int16Data[i] / 32768.0;
            }

            // Create AudioBuffer
            const buffer = playbackContext.createBuffer(1, float32Data.length, 24000);
            buffer.getChannelData(0).set(float32Data);

            // Play
            const source = playbackContext.createBufferSource();
            source.buffer = buffer;
            source.connect(playbackContext.destination);
            source.start();

            // Wait for completion
            await new Promise(resolve => {
                source.onended = resolve;
            });
        }

        this.isPlaying = false;
        await playbackContext.close();
    }

    setLanguage(language) {
        this.language = language;
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'set_language',
                language: language
            }));
        }
    }

    disconnect() {
        if (this.workletNode) {
            this.workletNode.disconnect();
        }
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
        }
        if (this.ws) {
            this.ws.close();
        }
        if (this.audioContext) {
            this.audioContext.close();
        }
    }
}
```

#### 4.3 Audio Worklet (`web_client/audio-worklet.js`)

```javascript
/**
 * AudioWorklet processor for capturing microphone audio
 *
 * Converts Float32 samples to Int16 PCM and sends to main thread
 */

class AudioCaptureProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.bufferSize = 4096;
        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (!input || !input[0]) {
            return true;
        }

        const samples = input[0];

        for (let i = 0; i < samples.length; i++) {
            this.buffer[this.bufferIndex++] = samples[i];

            if (this.bufferIndex >= this.bufferSize) {
                // Convert to Int16
                const int16Buffer = new Int16Array(this.bufferSize);
                for (let j = 0; j < this.bufferSize; j++) {
                    const s = Math.max(-1, Math.min(1, this.buffer[j]));
                    int16Buffer[j] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }

                // Send to main thread
                this.port.postMessage({
                    type: 'audio',
                    buffer: int16Buffer.buffer
                }, [int16Buffer.buffer]);

                this.bufferIndex = 0;
            }
        }

        return true;
    }
}

registerProcessor('audio-capture-processor', AudioCaptureProcessor);
```

### Phase 5: Backend WebSocket Endpoint

#### 5.1 FastAPI WebSocket Handler

```python
# Add to simple_rag_server.py

from fastapi import WebSocket, WebSocketDisconnect
from gemini_live.service import GeminiLiveService
from gemini_live.session_manager import SessionManager
from gemini_live.audio_handler import AudioHandler

# Initialize services
gemini_service = None
session_manager = None

@app.on_event("startup")
async def startup_gemini():
    global gemini_service, session_manager

    if config.get("gemini_live", {}).get("enabled", False):
        gemini_service = GeminiLiveService(
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
            rag_pipeline=rag_pipeline  # Inject existing RAG
        )
        session_manager = SessionManager(gemini_service)
        await session_manager.start()

@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for voice conversations.

    Protocol:
    - Client sends JSON config message first
    - Client sends binary audio chunks
    - Server sends binary audio responses
    - Server sends JSON transcription/status messages
    """
    await websocket.accept()

    session = None
    language = "en-IN"
    user_id = f"web_{datetime.now().timestamp()}"

    try:
        while True:
            data = await websocket.receive()

            # Handle text messages (JSON)
            if "text" in data:
                message = json.loads(data["text"])

                if message["type"] == "config":
                    language = message.get("language", "en-IN")
                    user_id = message.get("user_id", user_id)

                elif message["type"] == "start_audio":
                    # Create/get session
                    session = await session_manager.get_or_create_session(
                        user_id=user_id,
                        language=language,
                        voice="Aoede"
                    )

                    # Start receiving responses in background
                    asyncio.create_task(
                        stream_responses(websocket, session)
                    )

                elif message["type"] == "stop_audio":
                    # Audio stream ended, wait for response
                    pass

                elif message["type"] == "set_language":
                    language = message["language"]
                    # Close existing session to switch language
                    if session:
                        await session_manager.close_session(user_id)
                        session = None

            # Handle binary messages (audio)
            elif "bytes" in data:
                if session:
                    await session.send_audio(data["bytes"])

    except WebSocketDisconnect:
        pass
    finally:
        if session:
            await session_manager.close_session(user_id)


async def stream_responses(websocket: WebSocket, session):
    """Stream audio responses back to client."""
    try:
        async for audio_chunk in session.receive_audio():
            if audio_chunk == b"__TURN_COMPLETE__":
                await websocket.send_json({
                    "type": "turn_complete"
                })

                # Send transcription
                if session.response_buffer:
                    await websocket.send_json({
                        "type": "transcription",
                        "role": "assistant",
                        "text": " ".join(session.response_buffer)
                    })
                    session.response_buffer.clear()

                if session.transcription_buffer:
                    await websocket.send_json({
                        "type": "transcription",
                        "role": "user",
                        "text": " ".join(session.transcription_buffer)
                    })
                    session.transcription_buffer.clear()

            elif audio_chunk == b"__INTERRUPTED__":
                # User interrupted, clear pending audio
                pass

            else:
                # Send audio chunk
                await websocket.send_bytes(audio_chunk)

    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "error": str(e)
        })
```

---

## 5. CONFIGURATION UPDATES

### 5.1 config.yaml Additions

```yaml
# Add to config.yaml

gemini_live:
  enabled: true
  project_id: "${GOOGLE_CLOUD_PROJECT}"
  location: "us-central1"
  model: "gemini-live-2.5-flash-preview-native-audio-09-2025"

  # Voice settings
  default_voice: "Aoede"  # Empathetic voice for healthcare

  # Language support
  supported_languages:
    - code: "en-IN"
      name: "English (India)"
      voice: "Aoede"
    - code: "hi-IN"
      name: "Hindi"
      voice: "Aoede"
    - code: "mr-IN"
      name: "Marathi"
      voice: "Aoede"
    - code: "ta-IN"
      name: "Tamil"
      voice: "Aoede"

  # Session settings
  session_timeout_minutes: 14
  max_sessions_per_user: 1

  # Audio settings
  input_sample_rate: 16000
  output_sample_rate: 24000
  chunk_size: 4096

  # RAG integration
  rag_context_enabled: true
  rag_top_k: 3

  # Fallback to existing pipeline on error
  fallback_enabled: true
```

### 5.2 Environment Variables

```bash
# Add to .env

# Google Cloud / Vertex AI
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GEMINI_API_KEY=your-api-key  # Alternative to ADC

# Optional: Telephony
DAILY_API_KEY=your-daily-api-key
TWILIO_VOICE_SID=your-voice-sid
```

---

## 6. TESTING PLAN

### 6.1 Unit Tests

```python
# tests/test_gemini_live.py

import pytest
import asyncio
from gemini_live.service import GeminiLiveService
from gemini_live.audio_handler import AudioHandler

@pytest.fixture
def audio_handler():
    return AudioHandler()

@pytest.fixture
def gemini_service():
    return GeminiLiveService(
        project_id="test-project",
        location="us-central1"
    )

class TestAudioHandler:
    def test_convert_to_pcm(self, audio_handler):
        # Create test OGG audio
        test_audio = b"..." # Sample OGG bytes
        pcm = audio_handler.convert_to_pcm(test_audio, "ogg")

        # Verify PCM format
        assert len(pcm) > 0
        assert len(pcm) % 2 == 0  # 16-bit samples

    def test_chunk_audio(self, audio_handler):
        test_data = b"x" * 10000
        chunks = audio_handler.chunk_audio(test_data, chunk_size=4096)

        assert len(chunks) == 3
        assert len(chunks[0]) == 4096
        assert len(chunks[1]) == 4096
        assert len(chunks[2]) == 10000 - 8192

class TestGeminiService:
    @pytest.mark.asyncio
    async def test_create_session(self, gemini_service):
        session = await gemini_service.create_session(
            session_id="test-123",
            language="en-IN"
        )

        assert session is not None
        assert session.language == "en-IN"

    @pytest.mark.asyncio
    async def test_language_support(self, gemini_service):
        languages = ["en-IN", "hi-IN", "mr-IN", "ta-IN"]

        for lang in languages:
            session = await gemini_service.create_session(
                session_id=f"test-{lang}",
                language=lang
            )
            assert session.language == lang
```

### 6.2 Integration Tests

```python
# tests/test_integration.py

import pytest
import asyncio
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket

@pytest.mark.asyncio
async def test_websocket_connection():
    """Test WebSocket voice endpoint."""
    async with TestClient(app) as client:
        async with client.websocket_connect("/ws/voice") as ws:
            # Send config
            await ws.send_json({
                "type": "config",
                "language": "en-IN"
            })

            # Start audio
            await ws.send_json({"type": "start_audio"})

            # Send test audio chunk
            test_audio = b"\x00" * 4096  # Silence
            await ws.send_bytes(test_audio)

            # Stop
            await ws.send_json({"type": "stop_audio"})

            # Should receive turn_complete eventually
            # (May timeout if no real Gemini connection)

@pytest.mark.asyncio
async def test_rag_context_injection():
    """Test that RAG context is properly injected."""
    # Add test document
    await rag_pipeline.add_documents(
        file_paths=["test_doc.txt"],
        metadata={"category": "test"}
    )

    # Create session with RAG
    session = await gemini_service.create_session(
        session_id="test-rag",
        language="en-IN"
    )

    # Inject context
    await gemini_service.inject_rag_context(
        session,
        "What is palliative care?"
    )

    # Verify context was sent
    # (Would need mock or capture mechanism)
```

### 6.3 Manual Testing Checklist

```markdown
## Voice Conversation Testing

### Basic Functionality
- [ ] Web client loads correctly
- [ ] Microphone permission requested
- [ ] Audio capture starts on button press
- [ ] Audio stops on button release
- [ ] Response audio plays correctly
- [ ] Transcriptions appear in UI

### Language Support
- [ ] English (en-IN) works
- [ ] Hindi (hi-IN) works
- [ ] Marathi (mr-IN) works
- [ ] Tamil (ta-IN) works
- [ ] Language switching works mid-session

### RAG Integration
- [ ] Medical queries return relevant info
- [ ] Citations are mentioned in responses
- [ ] Context from documents is accurate

### Error Handling
- [ ] Network disconnect handled gracefully
- [ ] Session timeout handled
- [ ] Invalid audio format handled
- [ ] Fallback to STT+TTS works

### WhatsApp Integration
- [ ] Voice messages processed via Gemini Live
- [ ] Audio responses sent back
- [ ] Text transcription included
- [ ] Multi-language support works
```

---

## 7. DEPLOYMENT CHECKLIST

### 7.1 Pre-Deployment

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up Google Cloud authentication
gcloud auth application-default login

# 3. Enable required APIs
gcloud services enable aiplatform.googleapis.com

# 4. Verify environment variables
cat .env | grep GOOGLE_CLOUD

# 5. Run tests
pytest tests/

# 6. Test Gemini connectivity
python -c "from google import genai; print(genai.Client(vertexai=True).models.list())"
```

### 7.2 Deployment Steps

```bash
# 1. Update configuration
cp config.yaml config.yaml.backup
# Edit config.yaml with production settings

# 2. Start server
python simple_rag_server.py --host 0.0.0.0 --port 8000

# 3. Verify endpoints
curl http://localhost:8000/health
curl http://localhost:8000/api/debug

# 4. Test WebSocket
# Use browser developer tools or wscat

# 5. Update Twilio webhook (if using WhatsApp)
# Point to: https://your-domain.com/webhook
```

---

## 8. ARCHITECTURE DECISIONS & RATIONALE

### Why Gemini Live API?

| Factor | Traditional (STT→LLM→TTS) | Gemini Live |
|--------|---------------------------|-------------|
| Latency | 3-5 seconds | <1 second |
| Pipeline | 3 services | 1 service |
| Interruption | Not possible | Native support |
| Emotion | Lost in transcription | Preserved |
| Cost | 3x API calls | 1x API call |

### Architecture Choice: Hybrid Approach

We chose a **hybrid approach** that:
1. Uses Gemini Live for real-time voice conversations
2. Maintains existing RAG pipeline for knowledge grounding
3. Keeps fallback to STT+LLM+TTS for reliability

**Rationale:**
- Leverages Gemini's low-latency audio streaming
- Preserves medical knowledge from RAG documents
- Ensures reliability with fallback option
- Minimizes changes to existing infrastructure

### Language Support Strategy

The system supports 4 Indian languages through:
1. **Gemini's native multilingual capability** - Understands Hindi, Marathi, Tamil natively
2. **Language-specific system prompts** - Tailored instructions per language
3. **Per-user language preference** - Remembers user's language choice

---

## 9. FUTURE ENHANCEMENTS

### 9.1 Telephony Integration (Optional)

```python
# telephony/twilio_voice.py

"""
PSTN call support via Twilio + Daily.co bridge.

Flow:
User dials phone → Twilio → TwiML → Daily SIP → WebRTC → Gemini Live
"""

from pipecat.pipeline import Pipeline
from pipecat.services.gemini import GeminiMultimodalLiveLLMService
from pipecat.transports.services.daily import DailyTransport

# Implementation similar to:
# https://github.com/sa-kanean/gemini-live-voice-ai-agent-with-telephony
```

### 9.2 Video Support

```python
# Future: Add video streaming for visual context
# Gemini Live supports video at 1 FPS, 768x768 resolution

async def send_video_frame(session, frame_data: bytes):
    """Send video frame for visual context."""
    await session.send_realtime_input(
        media=types.Blob(
            data=frame_data,
            mime_type="image/jpeg"
        )
    )
```

### 9.3 Advanced Analytics

```python
# Future: Add conversation analytics
class ConversationAnalytics:
    """Track conversation metrics."""

    async def log_interaction(
        self,
        session_id: str,
        user_audio_duration: float,
        response_audio_duration: float,
        transcription: str,
        language: str,
        sentiment: Optional[str] = None
    ):
        """Log interaction for analytics."""
        pass
```

---

## 10. QUICK START GUIDE

### For Developers

```bash
# 1. Clone and setup
cd rag_gci
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your credentials

# 3. Run
python simple_rag_server.py

# 4. Test web client
open http://localhost:8000/voice
```

### For Users

1. **Web Interface**:
   - Go to `https://your-server.com/voice`
   - Select your language
   - Click the microphone button and speak
   - Release to hear the response

2. **WhatsApp**:
   - Send a voice message to the WhatsApp number
   - Receive audio response in your language
   - Text transcription also provided

---

## APPENDIX A: API Reference

### Gemini Live API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `wss://{location}-aiplatform.googleapis.com/ws/...` | WebSocket connection |
| `POST /v1/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent` | REST fallback |

### Message Types

```typescript
// Client → Server
interface ClientMessage {
  type: 'config' | 'start_audio' | 'stop_audio' | 'set_language';
  language?: string;
  user_id?: string;
}

// Server → Client
interface ServerMessage {
  type: 'transcription' | 'turn_complete' | 'error';
  role?: 'user' | 'assistant';
  text?: string;
  error?: string;
}
```

---

## APPENDIX B: Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Permission denied" | ADC not configured | Run `gcloud auth application-default login` |
| "Model not found" | Wrong model ID | Use `gemini-live-2.5-flash-preview-native-audio-09-2025` |
| "Session timeout" | 15-minute limit | Implement session resumption |
| "Audio format error" | Wrong PCM format | Ensure 16kHz, 16-bit, mono input |
| "Language not supported" | Invalid code | Use exact codes: `en-IN`, `hi-IN`, `mr-IN`, `ta-IN` |

### Debug Logging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("gemini_live").setLevel(logging.DEBUG)
```

---

## APPENDIX C: References

### Codebase Documentation
- **[DeepWiki - RAG GCI System](https://deepwiki.com/inventcures/rag_gci)** - Comprehensive auto-generated documentation of the current codebase including architecture diagrams, data flows, and API specifications.

### Official Documentation
- [Gemini Live API Overview](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api)
- [Start and Manage Sessions](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api/start-manage-session)
- [Send Audio/Video Streams](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api/send-audio-video-streams)
- [Configure Language and Voice](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api/configure-language-voice)
- [Google Gen AI SDK](https://ai.google.dev/gemini-api/docs/live)

### Example Repositories
- [Google Cloud Generative AI Samples](https://github.com/GoogleCloudPlatform/generative-ai)
- [Live API Web Console](https://github.com/google-gemini/live-api-web-console)
- [Gemini Live Voice Agent with Telephony](https://github.com/sa-kanean/gemini-live-voice-ai-agent-with-telephony)

### Related Technologies
- [Pipecat Framework](https://github.com/pipecat-ai/pipecat)
- [Daily.co](https://www.daily.co/)
- [Twilio Voice](https://www.twilio.com/voice)

---

**Document Version:** 1.0
**Created:** December 2024
**Author:** Claude (Anthropic)
**For:** Palliative Care RAG System Voice AI Integration
