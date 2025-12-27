# Gemini Live API Setup Guide (Vertex AI)

This guide explains how to set up and run the Palli Sahayak RAG server with Google's Gemini Live API for real-time voice conversations.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Google Cloud Setup](#google-cloud-setup)
3. [Running with Gemini Live](#running-with-gemini-live)
4. [Testing & Debugging](#testing--debugging)
5. [Monitoring & Analytics](#monitoring--analytics)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Google Cloud account with billing enabled
- Python 3.10+
- RAG server dependencies installed (`pip install -r requirements.txt`)
- Microphone & speakers for voice testing

---

## Google Cloud Setup

### Step 1: Create/Select a Google Cloud Project

```bash
# Install Google Cloud CLI if not already installed
# macOS:
brew install google-cloud-sdk

# Or download from: https://cloud.google.com/sdk/docs/install
```

```bash
# Login to Google Cloud
gcloud auth login

# Create a new project (or use existing)
gcloud projects create palli-sahayak-voice --name="Palli Sahayak Voice AI"

# Set as active project
gcloud config set project palli-sahayak-voice

# Enable billing (required for Vertex AI)
# Go to: https://console.cloud.google.com/billing
```

### Step 2: Enable Required APIs

```bash
# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com

# Enable Cloud Speech API (for fallback STT)
gcloud services enable speech.googleapis.com

# Enable Cloud Text-to-Speech API (for fallback TTS)
gcloud services enable texttospeech.googleapis.com

# Verify APIs are enabled
gcloud services list --enabled | grep -E "aiplatform|speech|texttospeech"
```

### Step 3: Create Service Account & Credentials

```bash
# Create service account
gcloud iam service-accounts create palli-sahayak-voice \
    --display-name="Palli Sahayak Voice AI Service Account"

# Grant Vertex AI User role
gcloud projects add-iam-policy-binding palli-sahayak-voice \
    --member="serviceAccount:palli-sahayak-voice@palli-sahayak-voice.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Create and download key
gcloud iam service-accounts keys create ~/palli-sahayak-credentials.json \
    --iam-account=palli-sahayak-voice@palli-sahayak-voice.iam.gserviceaccount.com

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS=~/palli-sahayak-credentials.json
```

### Step 4: Configure Environment Variables

Add to your `.env` file:

```bash
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=palli-sahayak
GOOGLE_APPLICATION_CREDENTIALS=/path/to/palli-sahayak-credentials.json
VERTEX_AI_LOCATION=us-central1

# Gemini Live Configuration
GEMINI_LIVE_ENABLED=true
GEMINI_LIVE_MODEL=gemini-2.0-flash-live-001
GEMINI_LIVE_VOICE=Aoede  # Warm, empathetic voice

# Voice Provider
VOICE_PROVIDER=gemini
```

### Step 5: Verify Vertex AI Access

```bash
# Test Vertex AI access
python3 -c "
from google.cloud import aiplatform
aiplatform.init(project='palli-sahayak-voice', location='us-central1')
print('✅ Vertex AI access verified!')
"
```

---

## Running with Gemini Live

### Option 1: Using --provider Flag (Recommended)

```bash
# Run with Gemini Live API
python simple_rag_server.py --provider g

# Or full form
python simple_rag_server.py --provider gemini

# With custom port
python simple_rag_server.py --provider g --port 8000
```

### Option 2: Using Environment Variable

```bash
# Set provider via environment
export VOICE_PROVIDER=gemini
export GEMINI_LIVE_ENABLED=true

python simple_rag_server.py
```

### Expected Startup Output

```
============================================================
🎙️  VOICE PROVIDER: Google Gemini Live API (Vertex AI)
============================================================
INFO:__main__:🎙️  Gemini Live enabled via --provider flag
INFO:__main__:Gemini Live initialized - WebSocket endpoint /ws/voice available
INFO:__main__:🎤 Voice interface available at /voice
INFO:__main__:🌐 Starting ngrok tunnel...
INFO:__main__:🌐 ngrok tunnel started: https://xxxxx.ngrok-free.app
```

---

## Testing & Debugging

### Test 1: Health Check

```bash
curl http://localhost:8000/health | jq
```

Expected response:
```json
{
  "status": "healthy",
  "gemini_live": "enabled",
  "voice_websocket": "ws://localhost:8000/ws/voice"
}
```

### Test 2: Voice WebSocket Connection

Open in browser: `http://localhost:8000/voice`

This opens the voice interface where you can:
1. Click "Start Recording" to speak
2. Ask a health question in Hindi/English
3. Hear the AI response

### Test 3: Direct WebSocket Test

```python
import asyncio
import websockets
import json

async def test_gemini_voice():
    uri = "ws://localhost:8000/ws/voice?user_id=test&language=hi-IN"

    async with websockets.connect(uri) as ws:
        # Send test message
        await ws.send(json.dumps({
            "type": "text",
            "content": "What helps with pain?"
        }))

        # Receive response
        response = await ws.recv()
        print(f"Response: {response}")

asyncio.run(test_gemini_voice())
```

### Viewing Real-time Logs

```bash
# Terminal 1: Run server with Gemini
python simple_rag_server.py --provider g 2>&1 | tee /tmp/gemini_server.log

# Terminal 2: Filter for Gemini & RAG queries
tail -f /tmp/gemini_server.log | grep -E "GEMINI|RAG|BOLNA|📞|🎙️|✅|❌"
```

### Log Output Example

When a voice query is processed:

```
============================================================
🎙️  GEMINI LIVE - VOICE QUERY
============================================================
🗣️  Transcript: दर्द में क्या मदद करता है?
🌐 Language: hi-IN | Session: gemini_user123_20251225
------------------------------------------------------------
============================================================
📞 BOLNA VOICE CALL - RAG QUERY
============================================================
🗣️  Query: What helps with pain management?
🌐 Language: hi | Source: gemini_live
------------------------------------------------------------
✅ RAG SUCCESS (1.45s)
📚 Sources: Textbook_IndianPrimer.pdf, Twycross-2008.pdf
💬 Answer preview: Pain management involves regular assessment...
🎯 Confidence: 87%
============================================================
```

---

## Monitoring & Analytics

### Google Cloud Console

1. **Vertex AI Dashboard**: https://console.cloud.google.com/vertex-ai
   - View API usage
   - Monitor quotas
   - Check error rates

2. **Cloud Logging**: https://console.cloud.google.com/logs
   ```
   resource.type="aiplatform.googleapis.com/Endpoint"
   ```

3. **Cloud Monitoring**: https://console.cloud.google.com/monitoring
   - Set up alerts for errors
   - Track latency metrics

### Local Analytics Endpoint

```bash
# Get voice session stats
curl http://localhost:8000/api/voice/stats | jq

# Get RAG query stats
curl http://localhost:8000/api/stats | jq
```

---

## Troubleshooting

### Error: "Gemini Live module not available"

```bash
# Install required dependencies
pip install google-cloud-aiplatform>=1.38.0
pip install google-generativeai>=0.3.0
```

### Error: "Permission denied" or "Credentials not found"

```bash
# Verify credentials
echo $GOOGLE_APPLICATION_CREDENTIALS
cat $GOOGLE_APPLICATION_CREDENTIALS | jq .project_id

# Re-authenticate
gcloud auth application-default login
```

### Error: "Quota exceeded"

1. Check quotas: https://console.cloud.google.com/iam-admin/quotas
2. Request quota increase if needed
3. Implement rate limiting in your app

### Error: "Voice WebSocket disabled"

Check these in order:
1. `GEMINI_LIVE_ENABLED=true` in environment
2. `--provider g` flag when starting server
3. Valid Google Cloud credentials
4. Vertex AI API enabled

### Audio Issues

```bash
# Test microphone
python -c "
import pyaudio
p = pyaudio.PyAudio()
print(f'Input devices: {p.get_device_count()}')
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f'  [{i}] {info[\"name\"]}')
"
```

---

## Comparison: Gemini Live vs Bolna

| Feature | Gemini Live | Bolna.ai |
|---------|-------------|----------|
| **Best For** | Web voice interface | Phone calls |
| **Latency** | ~200-400ms | ~300-500ms |
| **Languages** | en-IN, hi-IN, mr-IN, ta-IN | Same + more |
| **Cost** | Pay-per-use (Vertex AI) | Subscription + per-min |
| **Phone Support** | ❌ No | ✅ Yes (Twilio/Plivo) |
| **WebSocket** | ✅ Native | Via webhook |
| **Setup Complexity** | Medium | Easy (dashboard) |

### When to Use Which

- **Use Gemini Live (`-p g`)** for:
  - Web-based voice interface
  - Development/testing
  - Lower latency requirements
  - Custom WebSocket integrations

- **Use Bolna (`-p b`)** for:
  - Production phone calls
  - Toll-free helpline
  - Call analytics & recordings
  - Multi-channel telephony

---

## Quick Reference

```bash
# Start with Bolna (default)
python simple_rag_server.py

# Start with Gemini Live
python simple_rag_server.py --provider g

# Start with Gemini Live + custom port + no ngrok
python simple_rag_server.py -p g --port 8080 --no-ngrok

# Check which provider is active
curl http://localhost:8000/health | jq .voice_provider
```
