# Palli Sahayak Demo Guide - Marathi Voice with Gemini Live API

**Purpose:** Step-by-step instructions for running Palli Sahayak with Gemini Live API in Marathi for the Cipla demo.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Running the Server with Gemini Live (-p g)](#2-running-the-server-with-gemini-live--p-g)
3. [Marathi Language Support](#3-marathi-language-support)
4. [Web Voice Interface](#4-web-voice-interface)
5. [Preventing AI from Talking Over User](#5-preventing-ai-from-talking-over-user)
6. [Demo Checklist](#6-demo-checklist)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Prerequisites

### Required Environment Variables

Create/update your `.env` file with these Gemini Live settings:

```bash
# Google Cloud / Vertex AI Configuration
GOOGLE_CLOUD_PROJECT=palli-sahayak          # Your GCP project ID
VERTEX_AI_LOCATION=asia-south1              # Mumbai region for lower latency
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-credentials.json

# Optional: Use Gemini API Key instead of service account
# GEMINI_API_KEY=your_gemini_api_key
```

### Verify Google Cloud Setup

```bash
# Set environment variables
export GOOGLE_CLOUD_PROJECT="palli-sahayak"
export VERTEX_AI_LOCATION="asia-south1"
export GOOGLE_APPLICATION_CREDENTIALS="/Users/tp53/palli-sahayak-credentials.json"

# Verify authentication
gcloud auth application-default print-access-token
```

### Install Dependencies

```bash
cd /Users/tp53/Documents/tp53_AA/llms4palliative_gci/demo_feb2025/rag_gci
./venv/bin/pip install google-genai  # Gemini Live SDK
```

---

## 2. Running the Server with Gemini Live (-p g)

### Quick Start Command

```bash
cd /Users/tp53/Documents/tp53_AA/llms4palliative_gci/demo_feb2025/rag_gci

# Run with Gemini Live provider
./venv/bin/python simple_rag_server.py -p g
```

### What `-p g` Does

The `-p` (or `--provider`) flag selects the voice AI provider:

| Flag | Provider | Description |
|------|----------|-------------|
| `-p g` or `-p gemini` | Gemini Live API | Google's native voice AI with real-time audio |
| `-p b` or `-p bolna` | Bolna.ai | Third-party voice AI (default) |
| `-p r` or `-p retell` | Retell.ai | Third-party voice AI with Vobiz DID |

### Full Command with All Options

```bash
./venv/bin/python simple_rag_server.py \
    -p g \                    # Use Gemini Live provider
    --port 8000               # Server port (default: 8000)
```

### Expected Output

```
🚀 Starting Palli Sahayak RAG Server...
📂 Data directory: ./data
📁 Upload directory: ./uploads
🔊 Voice Provider: gemini
🎙️  Gemini Live enabled via --provider flag
✅ Gemini Live service available (model: gemini-live-2.5-flash-native-audio)
🎙️  Gemini Live initialized - WebSocket endpoint /ws/voice available
🎤 Voice interface available at /voice
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## 3. Marathi Language Support

### Yes, Marathi is Fully Supported!

Gemini Live API supports **4 Indian languages**:

| Language | Code | Native Name | Voice Fallback |
|----------|------|-------------|----------------|
| English (India) | `en-IN` | English | en-IN-NeerjaNeural |
| Hindi | `hi-IN` | हिन्दी | hi-IN-SwaraNeural |
| **Marathi** | `mr-IN` | **मराठी** | mr-IN-AarohiNeural |
| Tamil | `ta-IN` | தமிழ் | ta-IN-PallaviNeural |

### Configuration in `config.yaml`

The Marathi language is already configured:

```yaml
gemini_live:
  enabled: true
  default_language: "en-IN"  # Can change to "mr-IN" for Marathi-first
  supported_languages:
    - code: "en-IN"
      name: "English (India)"
    - code: "hi-IN"
      name: "Hindi"
    - code: "mr-IN"      # Marathi is supported!
      name: "Marathi"
    - code: "ta-IN"
      name: "Tamil"
```

### Setting Marathi as Default

To make Marathi the default language, update `config.yaml`:

```yaml
gemini_live:
  default_language: "mr-IN"  # Set Marathi as default
```

Or set via environment variable:

```bash
export GEMINI_DEFAULT_LANGUAGE="mr-IN"
```

---

## 4. Web Voice Interface

### Accessing the Voice Interface

Once the server is running with `-p g`, open your browser:

```
http://localhost:8000/voice
```

Or if using ngrok for external access:

```
https://your-ngrok-domain.ngrok-free.app/voice
```

### Interface Features

The web voice interface provides:

1. **Language Selector** - Click buttons to switch between:
   - English
   - हिंदी (Hindi)
   - **मराठी (Marathi)** - Select this for the demo
   - தமிழ் (Tamil)

2. **Microphone Button** - Tap to start/stop recording

3. **Conversation Display** - Shows transcribed user input and AI responses

4. **Connection Status** - Indicates WebSocket connection state

### For Marathi Demo

1. Open `http://localhost:8000/voice`
2. Click the **"मराठी"** button to switch to Marathi
3. The welcome message will change to:
   > "नमस्कार! मी तुमची पॅलिएटिव्ह केअर व्हॉइस असिस्टंट आहे. मी आज तुम्हाला कशी मदत करू शकते?"
4. Tap the microphone and speak in Marathi

### WebSocket Endpoint

For custom integrations, the WebSocket endpoint is:

```
ws://localhost:8000/ws/voice
```

Or with SSL:
```
wss://your-domain/ws/voice
```

---

## 5. Preventing AI from Talking Over User

### Understanding the Problem

Gemini Live API has built-in **Voice Activity Detection (VAD)** that detects when the user starts speaking and can interrupt its own response. However, it may sometimes:

- Start responding too quickly
- Not give the user enough time to finish
- Interrupt during natural pauses

### Solutions Implemented in Palli Sahayak

#### 1. Turn-Taking Detection

The system waits for `turn_complete` signals before responding:

```python
# In gemini_live/service.py
if content.turn_complete:
    await self._audio_out_queue.put(self.TURN_COMPLETE)
    # Only then trigger RAG query
    if self._pending_transcription:
        asyncio.create_task(self._query_rag_and_inject())
```

#### 2. Interrupt Handling

When the user interrupts, the system clears pending responses:

```python
if content.interrupted:
    await self._audio_out_queue.put(self.INTERRUPTED)
    # Clear pending transcription on interrupt
    self._pending_transcription.clear()
```

### Configuration Options to Reduce Over-Eagerness

#### Option A: Adjust System Instruction (Recommended)

Add explicit instructions to the system prompt in `gemini_live/service.py`:

```python
# Find the _build_system_instruction method and add:
PALLIATIVE_CARE_INSTRUCTION = """
...existing instructions...

IMPORTANT CONVERSATION STYLE:
- Wait for the user to fully complete their question before responding
- Do not interrupt the user while they are speaking
- Take brief pauses before responding to ensure the user has finished
- If the user pauses mid-sentence, wait 2-3 seconds before assuming they are done
- Speak slowly and clearly, especially for medical terminology
- Allow natural pauses for the user to process information
"""
```

#### Option B: Voice Selection

Use a calmer voice that speaks more deliberately:

```yaml
# In config.yaml
gemini_live:
  default_voice: "Aoede"  # Warm, easy-going - best for palliative care
```

Available voices:

| Voice | Style | Best For |
|-------|-------|----------|
| **Aoede** | Easy-going | Healthcare, empathetic conversations |
| Charon | Informative | Medical information delivery |
| Kore | Firm | Clear instructions |
| Puck | Upbeat | General engagement |

#### Option C: Client-Side Control (Push-to-Talk)

The web interface uses **push-to-talk** by default, which prevents the AI from detecting ambient noise as user input:

1. User taps microphone to START recording
2. User speaks
3. User taps microphone to STOP recording
4. Only then is audio sent to Gemini

This prevents:
- Background noise triggering responses
- AI responding to partial sentences
- Crosstalk and interruptions

### For the Cipla Demo

**Recommended Settings:**

1. **Use Push-to-Talk Mode** (default in web interface)
   - User controls when they're speaking
   - No accidental interruptions

2. **Set Voice to Aoede** (already default)
   - Warm, empathetic tone
   - Speaks at a measured pace

3. **Select Marathi Language**
   - Click "मराठी" in the web interface
   - System will respond in Marathi

4. **Brief User on Protocol**
   - Instruct demo participant to:
     - Tap mic, speak complete question, tap mic again
     - Wait for full response before asking next question
     - Speak clearly and at normal pace

---

## 6. Demo Checklist

### Before the Demo

- [ ] Environment variables set correctly
- [ ] Google Cloud credentials valid
- [ ] Server running with `-p g` flag
- [ ] Voice interface accessible at `/voice`
- [ ] Microphone permissions granted in browser
- [ ] Tested Marathi language selection
- [ ] Network stable (for Gemini API calls)

### Quick Test Script

```bash
# 1. Set environment
export GOOGLE_CLOUD_PROJECT="palli-sahayak"
export VERTEX_AI_LOCATION="asia-south1"
export GOOGLE_APPLICATION_CREDENTIALS="/Users/tp53/palli-sahayak-credentials.json"

# 2. Start server
cd /Users/tp53/Documents/tp53_AA/llms4palliative_gci/demo_feb2025/rag_gci
./venv/bin/python simple_rag_server.py -p g

# 3. Open browser
open http://localhost:8000/voice

# 4. Test Marathi query
# Click "मराठी" → Tap mic → Say "वेदना कमी करण्यासाठी काय करावे?" → Tap mic
```

### Sample Marathi Test Queries

| Marathi Query | English Translation |
|---------------|---------------------|
| "वेदना कमी करण्यासाठी काय करावे?" | What should be done to reduce pain? |
| "माझ्या आईला खूप वेदना होत आहेत" | My mother is having a lot of pain |
| "औषध कधी द्यावे?" | When should I give medicine? |
| "रात्री झोप येत नाही" | Cannot sleep at night |
| "जेवण जात नाही" | Not able to eat |

---

## 7. Troubleshooting

### Common Issues

#### "Gemini Live service not available"

```bash
# Check authentication
gcloud auth application-default print-access-token

# Verify project ID
echo $GOOGLE_CLOUD_PROJECT
```

#### "WebSocket connection failed"

```bash
# Check if server is running on correct port
curl http://localhost:8000/api/voice/health

# Expected response:
# {"status": "ok", "gemini_live": {...}}
```

#### "No audio output"

1. Check browser microphone permissions
2. Ensure HTTPS is used for production (required for mic access)
3. Test with headphones to avoid echo

#### "AI responds too quickly"

1. Use push-to-talk mode (default)
2. Instruct user to tap mic only after completing their question
3. Consider adding a small delay in the web client before sending

#### "Language not switching to Marathi"

1. Click the "मराठी" button in the web interface
2. Verify `mr-IN` is in `supported_languages` in config
3. Check browser console for errors

### Getting Help

- **Server Logs:** Check terminal output for errors
- **Browser Console:** Press F12 → Console tab
- **WebSocket Inspector:** F12 → Network → WS tab

---

## Quick Reference

### One-Liner to Start Demo

```bash
export GOOGLE_CLOUD_PROJECT="palli-sahayak" && export VERTEX_AI_LOCATION="asia-south1" && export GOOGLE_APPLICATION_CREDENTIALS="/Users/tp53/palli-sahayak-credentials.json" && cd /Users/tp53/Documents/tp53_AA/llms4palliative_gci/demo_feb2025/rag_gci && ./venv/bin/python simple_rag_server.py -p g
```

### URLs

| Resource | URL |
|----------|-----|
| Voice Interface | `http://localhost:8000/voice` |
| Admin Dashboard | `http://localhost:8000/admin` |
| API Health | `http://localhost:8000/api/voice/health` |
| WebSocket | `ws://localhost:8000/ws/voice` |

---

*Last updated: January 2026*
