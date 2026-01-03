# Palli Sahayak - Complete How-To Guide

## Democratizing Palliative Care Through AI-Powered Voice Assistance

**Version:** 1.0.0
**Last Updated:** December 2025

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Overview](#2-system-overview)
3. [Quick Start Guide](#3-quick-start-guide)
4. [Installation & Setup](#4-installation--setup)
5. [Configuration Reference](#5-configuration-reference)
6. [Features Overview](#6-features-overview)
7. [Bolna.ai Voice Integration](#7-bolnaai-voice-integration)
8. [Gemini Live API Integration](#8-gemini-live-api-integration)
9. [For System Administrators](#9-for-system-administrators)
10. [For End Users](#10-for-end-users)
11. [API Reference](#11-api-reference)
12. [Troubleshooting](#12-troubleshooting)
13. [FAQ](#13-faq)

---

## 1. Introduction

### 1.1 What is Palli Sahayak?

Palli Sahayak (पल्ली सहायक) is a compassionate AI-powered voice assistant designed to democratize palliative care knowledge. It provides 24/7 support for patients and caregivers dealing with serious illness through:

- **Phone Helpline** - Call and speak naturally in Hindi, English, Marathi, or Tamil
- **WhatsApp Bot** - Text or voice messages via WhatsApp
- **Web Interface** - Browser-based voice conversations
- **Admin Dashboard** - Manage documents and monitor the system

### 1.2 Key Capabilities

| Feature | Description |
|---------|-------------|
| **Multi-Language Support** | Hindi, English, Marathi, Tamil, Bengali, Gujarati |
| **Voice Conversations** | Natural speech recognition and synthesis |
| **RAG-Powered Answers** | Responses grounded in medical knowledge base |
| **Citation Tracking** | Every answer includes source references |
| **24/7 Availability** | Always-on AI assistance |
| **Zero Cost for Users** | Free API tiers enable sustainable operation |

### 1.3 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     PALLI SAHAYAK SYSTEM                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│   │   Phone      │  │   WhatsApp   │  │   Web Voice  │         │
│   │   (Bolna)    │  │   (Twilio)   │  │   (Gemini)   │         │
│   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│          │                 │                 │                  │
│          └─────────────────┼─────────────────┘                  │
│                            ▼                                    │
│                   ┌────────────────┐                            │
│                   │  Voice Router  │                            │
│                   └────────┬───────┘                            │
│                            ▼                                    │
│   ┌──────────────────────────────────────────────────────┐     │
│   │                    RAG PIPELINE                       │     │
│   │  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌────────┐  │     │
│   │  │ Query   │→ │ Vector   │→ │ Context │→ │  LLM   │  │     │
│   │  │ Process │  │ Search   │  │ Build   │  │ Answer │  │     │
│   │  └─────────┘  └──────────┘  └─────────┘  └────────┘  │     │
│   └──────────────────────────────────────────────────────┘     │
│                            ▲                                    │
│                   ┌────────┴───────┐                            │
│                   │   Knowledge    │                            │
│                   │   Base (Docs)  │                            │
│                   └────────────────┘                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. System Overview

### 2.1 Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Main Server** | FastAPI + Gradio | API endpoints + Admin UI |
| **Vector DB** | ChromaDB | Semantic search |
| **LLM** | Groq (Llama 3.1) | Answer generation |
| **STT** | Groq Whisper | Speech-to-text |
| **TTS** | Edge TTS | Text-to-speech |
| **Phone Voice** | Bolna.ai | Phone call handling |
| **Web Voice** | Gemini Live | Real-time browser voice |
| **Messaging** | Twilio | WhatsApp integration |

### 2.2 Directory Structure

```
rag_gci/
├── simple_rag_server.py    # Main server (recommended)
├── whatsapp_bot.py         # WhatsApp integration
├── voice_router.py         # Voice provider routing
├── config.yaml             # System configuration
├── .env                    # API keys (not in git)
├── .env.example            # Configuration template
│
├── bolna_integration/      # Phone voice AI
│   ├── client.py           # Bolna API client
│   ├── config.py           # Agent configuration
│   └── webhooks.py         # Event handling
│
├── gemini_live/            # Web voice AI
│   ├── service.py          # Gemini Live service
│   ├── session_manager.py  # Session handling
│   └── audio_handler.py    # Audio processing
│
├── knowledge_graph/        # Neo4j integration
│   ├── neo4j_client.py     # Database client
│   └── entity_extractor.py # Medical entity extraction
│
├── graphrag_integration/   # Microsoft GraphRAG
│   ├── config.py           # Configuration
│   ├── indexer.py          # Document indexing
│   ├── query_engine.py     # Search methods
│   └── utils.py            # Performance utilities
│
├── data/
│   ├── chroma_db/          # Vector database
│   ├── documents/          # Document metadata
│   └── graphrag/           # GraphRAG index
│
├── uploads/                # Uploaded files
├── cache/tts/              # Audio cache
└── docs/                   # Documentation
```

---

## 3. Quick Start Guide

### 3.1 Prerequisites

- Python 3.10 or higher
- 4GB RAM minimum
- Internet connection for API calls

### 3.2 5-Minute Setup

```bash
# 1. Clone the repository
git clone https://github.com/inventcures/rag_gci.git
cd rag_gci

# 2. Run setup script
python setup.py

# 3. Get free Groq API key
# Visit: https://console.groq.com/
# Copy your API key

# 4. Configure environment
cp .env.example .env
# Edit .env and add: GROQ_API_KEY=your_key_here

# 5. Start the server
./run_simple.sh

# 6. Open browser
# Admin UI: http://localhost:8000/admin
```

### 3.3 First Steps

1. **Upload Documents**: Go to "Upload Documents" tab, upload palliative care PDFs
2. **Test Query**: Go to "Test Queries" tab, ask "What helps with pain management?"
3. **Check Stats**: Go to "Index Statistics" to verify documents are indexed

---

## 4. Installation & Setup

### 4.1 System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10+ | 3.11+ |
| RAM | 4GB | 8GB+ |
| Storage | 2GB | 10GB+ |
| OS | Linux/macOS/Windows | Ubuntu 22.04 |

### 4.2 Dependencies Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 4.3 Required API Keys

| Service | Required? | Free Tier | Get Key At |
|---------|-----------|-----------|------------|
| Groq | Yes | 14,400 tokens/day | https://console.groq.com |
| Twilio | For WhatsApp | Trial credits | https://console.twilio.com |
| Bolna.ai | For Phone | Free tier available | https://app.bolna.ai |
| Gemini | For Web Voice | Free tier | https://aistudio.google.com |
| Neo4j | Optional | Free Aura | https://neo4j.com/cloud |

### 4.4 Environment Configuration

Create `.env` file with your API keys:

```bash
# =============================================================================
# REQUIRED - Core API Keys
# =============================================================================
GROQ_API_KEY=gsk_your_groq_api_key_here

# =============================================================================
# OPTIONAL - WhatsApp Integration (Twilio)
# =============================================================================
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886

# =============================================================================
# OPTIONAL - Phone Voice AI (Bolna.ai)
# =============================================================================
BOLNA_API_KEY=your_bolna_api_key
BOLNA_AGENT_NAME=Palli Sahayak
BOLNA_AGENT_ID=your_agent_id
BOLNA_PHONE_NUMBER=+919876543210

# =============================================================================
# OPTIONAL - Web Voice AI (Gemini Live)
# =============================================================================
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_CLOUD_PROJECT=your_project_id

# =============================================================================
# OPTIONAL - Knowledge Graph (Neo4j)
# =============================================================================
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# =============================================================================
# Server Configuration
# =============================================================================
PUBLIC_BASE_URL=https://your-domain.com
PORT=8000
DEBUG=false
```

### 4.5 Running the Server

**Option 1: Using Script (Recommended)**
```bash
./run_simple.sh
```

**Option 2: Direct Python**
```bash
python simple_rag_server.py --host 0.0.0.0 --port 8000
```

**Option 3: With ngrok (for webhooks)**
```bash
python simple_rag_server.py --host 0.0.0.0 --port 8000 --ngrok
```

**Startup Output:**
```
🚀 RAG SERVER WITH WHATSAPP BOT
==================================================
📊 Admin UI: http://localhost:8000/admin
🔗 API Docs: http://localhost:8000/docs
💚 Health Check: http://localhost:8000/health
📱 WhatsApp Webhook: http://localhost:8000/webhook
```

---

## 5. Configuration Reference

### 5.1 config.yaml

The main configuration file controls all system behavior:

```yaml
# =============================================================================
# LLM Configuration
# =============================================================================
llm:
  provider: "groq"
  model: "llama-3.1-8b-instant"  # Fast and capable
  temperature: 0.7               # Creativity level (0-1)
  max_tokens: 1024               # Max response length

# =============================================================================
# Embedding Configuration
# =============================================================================
embedding:
  provider: "sentence_transformers"
  model: "all-MiniLM-L6-v2"      # Lightweight, 384 dimensions
  # Alternative: "embeddinggemma" for higher quality

# =============================================================================
# Vector Store Configuration
# =============================================================================
vectorstore:
  provider: "chroma"
  persist_directory: "./data/chroma_db"
  collection_name: "palli_sahayak"

# =============================================================================
# Speech-to-Text Configuration
# =============================================================================
stt:
  provider: "groq"
  model: "whisper-large-v3"
  supported_languages:
    - hi  # Hindi
    - bn  # Bengali
    - ta  # Tamil
    - gu  # Gujarati
    - en  # English

# =============================================================================
# Text-to-Speech Configuration
# =============================================================================
tts:
  provider: "edge_tts"
  languages:
    hi: "hi-IN-SwaraNeural"
    bn: "bn-IN-TanishaaNeural"
    ta: "ta-IN-PallaviNeural"
    gu: "gu-IN-DhwaniNeural"
    en: "en-IN-NeerjaNeural"

# =============================================================================
# Voice Router Configuration
# =============================================================================
voice_router:
  preferred_provider: "bolna"     # Primary voice provider
  phone_primary: "bolna"          # For phone calls
  web_primary: "gemini_live"      # For web voice
  ultimate_fallback: "stt_rag_tts" # Last resort pipeline

# =============================================================================
# RAG Pipeline Configuration
# =============================================================================
rag:
  chunk_size: 1000               # Text chunk size
  chunk_overlap: 200             # Overlap between chunks
  top_k: 5                       # Number of chunks to retrieve
  citation_format: "grouped"     # Citation style
```

### 5.2 Language Codes Reference

| Code | Language | Voice ID |
|------|----------|----------|
| hi | Hindi | hi-IN-SwaraNeural |
| en | English | en-IN-NeerjaNeural |
| bn | Bengali | bn-IN-TanishaaNeural |
| ta | Tamil | ta-IN-PallaviNeural |
| gu | Gujarati | gu-IN-DhwaniNeural |
| mr | Marathi | mr-IN-AarohiNeural |

---

## 6. Features Overview

### 6.1 Core RAG Pipeline

The RAG (Retrieval-Augmented Generation) pipeline:

1. **Query Processing** - Translate non-English queries to English for better matching
2. **Vector Search** - Find relevant document chunks using semantic similarity
3. **Context Building** - Assemble retrieved chunks with metadata
4. **LLM Generation** - Generate response grounded in retrieved context
5. **Citation Tracking** - Attach source references to response

**Example Query Flow:**
```
User: "दर्द के लिए क्या करें?" (What to do for pain?)
     ↓
[Translate to English]
     ↓
[Search vectors for "pain management"]
     ↓
[Retrieve top 5 chunks from documents]
     ↓
[Generate response with citations]
     ↓
Response: "For pain management, consider... [Sources: pain_guide.pdf: pg 12,15]"
```

### 6.2 Document Management

**Supported Formats:**
- PDF (`.pdf`)
- Word Documents (`.docx`)
- Text Files (`.txt`, `.md`)
- HTML (`.html`)
- Excel (`.xlsx`)

**Upload Process:**
1. Document parsed and text extracted
2. Text split into chunks (1000 chars with 200 overlap)
3. Chunks embedded using sentence-transformers
4. Vectors stored in ChromaDB
5. Metadata saved for citation tracking

### 6.3 Multi-Modal Communication

| Channel | Technology | Use Case |
|---------|------------|----------|
| **Phone** | Bolna.ai | Rural areas, elderly users |
| **WhatsApp** | Twilio | Text/voice messages |
| **Web** | Gemini Live | Browser-based voice |
| **API** | FastAPI | Integration with other systems |

### 6.4 Knowledge Graph (Optional)

Neo4j-powered knowledge graph for:
- **Entity Extraction** - Identify symptoms, medications, treatments
- **Relationship Mapping** - TREATS, CAUSES, MANAGES, SIDE_EFFECT_OF
- **Graph Queries** - "What treats nausea?" → morphine → side effects
- **Visualization** - Interactive graph exploration

### 6.5 GraphRAG (Optional)

Microsoft GraphRAG for advanced queries:
- **Global Search** - Broad, thematic questions
- **Local Search** - Entity-focused questions
- **DRIFT Search** - Multi-hop reasoning
- **Community Detection** - Topic clustering

---

## 7. Bolna.ai Voice Integration

### 7.1 Overview

Bolna.ai provides phone-based voice conversations for the helpline. Users can call a phone number and speak naturally with the AI assistant.

### 7.2 How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    BOLNA.AI ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   User Phone Call                                                │
│        ↓                                                         │
│   ┌─────────────────┐                                           │
│   │  Bolna Platform │  ← Hosted by Bolna.ai                     │
│   │  ┌───────────┐  │                                           │
│   │  │ Deepgram  │  │  ← Speech-to-Text                         │
│   │  │   STT     │  │                                           │
│   │  └─────┬─────┘  │                                           │
│   │        ↓        │                                           │
│   │  ┌───────────┐  │                                           │
│   │  │  OpenAI   │  │  ← LLM with Function Calling              │
│   │  │   LLM     │──┼──→ Calls RAG API                          │
│   │  └─────┬─────┘  │                                           │
│   │        ↓        │                                           │
│   │  ┌───────────┐  │                                           │
│   │  │ ElevenLabs│  │  ← Text-to-Speech                         │
│   │  │   TTS     │  │                                           │
│   │  └───────────┘  │                                           │
│   └────────┬────────┘                                           │
│            ↓                                                     │
│   Voice Response to User                                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 Setup Instructions

#### Step 1: Create Bolna Account
1. Go to https://app.bolna.ai
2. Sign up for a free account
3. Navigate to Settings → API Keys
4. Copy your API key

#### Step 2: Configure Environment
```bash
# Add to .env file
BOLNA_API_KEY=bn-your-api-key-here
BOLNA_AGENT_NAME=Palli Sahayak
```

#### Step 3: Create Agent
Use the API or dashboard to create an agent:

```python
from bolna_integration import BolnaClient, get_agent_config_from_env

# Initialize client
client = BolnaClient()

# Get configuration
config = get_agent_config_from_env(language="hi")

# Create agent
result = await client.create_agent(config)
print(f"Agent ID: {result.agent_id}")
```

#### Step 4: Configure Webhooks
Set webhook URL in Bolna dashboard:
```
https://your-domain.com/api/bolna/webhook
```

#### Step 5: Assign Phone Number
1. Go to Bolna Dashboard → Phone Numbers
2. Purchase or connect a phone number
3. Assign to your agent

### 7.4 Agent Configuration

The agent is configured with a specialized system prompt:

```python
PALLI_SAHAYAK_SYSTEM_PROMPT = """
You are Palli Sahayak (पल्ली सहायक), a compassionate palliative care
voice assistant for the Palli Sahayak Voice AI Agent Helpline.

## YOUR ROLE
- Provide empathetic support for patients and caregivers
- Answer questions about pain management, symptom control
- Offer emotional support and guidance
- Help navigate palliative care services

## LANGUAGE GUIDELINES
- Respond in the SAME LANGUAGE the user speaks
- Use simple, clear language
- Be warm, patient, and understanding

## CRITICAL RULES
1. ALWAYS call the RAG function when user asks health questions
2. NEVER provide specific medication dosages
3. For emergencies, advise calling 108/112
4. Keep voice responses concise (2-3 sentences)
"""
```

### 7.5 RAG Integration

Bolna calls your RAG API via function calling:

```json
{
  "name": "query_rag_knowledge_base",
  "description": "Query palliative care knowledge base",
  "parameters": {
    "user_query": "What helps with pain?",
    "user_language": "hi",
    "conversation_context": "User asked about pain management"
  }
}
```

**Endpoint:** `POST /api/bolna/query`

### 7.6 Webhook Events

Handle call lifecycle events:

| Event | Description | Use Case |
|-------|-------------|----------|
| `call_started` | Call initiated | Log start time |
| `call_ended` | Call completed | Save transcript |
| `extraction_completed` | Post-call analysis | Extract key topics |

**Example Webhook Handler:**
```python
@app.post("/api/bolna/webhook")
async def bolna_webhook(event: dict):
    event_type = event.get("type")

    if event_type == "call_ended":
        # Save call transcript
        transcript = event.get("transcript", [])
        duration = event.get("duration_seconds")
        await save_call_record(transcript, duration)

    return {"status": "ok"}
```

### 7.7 Bolna API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/bolna/query` | POST | RAG query from Bolna |
| `/api/bolna/webhook` | POST | Receive Bolna events |
| `/api/bolna/stats` | GET | Call statistics |
| `/api/bolna/calls` | GET | Call history |

### 7.8 Testing Bolna Integration

```bash
# Test RAG endpoint
curl -X POST http://localhost:8000/api/bolna/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What helps with pain?",
    "language": "en",
    "source": "bolna_call"
  }'

# Check stats
curl http://localhost:8000/api/bolna/stats
```

---

## 8. Gemini Live API Integration

### 8.1 Overview

Gemini Live provides real-time voice conversations in the web browser. Users can speak naturally and receive instant voice responses.

### 8.2 How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                   GEMINI LIVE ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Browser (User)                                                 │
│        ↓ WebSocket                                               │
│   ┌─────────────────────────────────────────┐                   │
│   │           RAG Server                     │                   │
│   │   ┌─────────────────────────────────┐   │                   │
│   │   │      GeminiLiveService          │   │                   │
│   │   │  ┌─────────┐  ┌─────────────┐   │   │                   │
│   │   │  │ Session │  │   Audio     │   │   │                   │
│   │   │  │ Manager │  │   Handler   │   │   │                   │
│   │   │  └────┬────┘  └──────┬──────┘   │   │                   │
│   │   │       ↓              ↓          │   │                   │
│   │   │  ┌────────────────────────┐     │   │                   │
│   │   │  │   Gemini Live API      │     │   │                   │
│   │   │  │   (Google Cloud)       │     │   │                   │
│   │   │  └────────────────────────┘     │   │                   │
│   │   └─────────────────────────────────┘   │                   │
│   │              ↓                          │                    │
│   │   ┌─────────────────┐                   │                   │
│   │   │   RAG Pipeline   │  ← Context       │                   │
│   │   └─────────────────┘    Injection      │                   │
│   └─────────────────────────────────────────┘                   │
│        ↓ WebSocket                                               │
│   Voice Response to Browser                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.3 Setup Instructions

#### Step 1: Get Gemini API Key
1. Go to https://aistudio.google.com/app/apikey
2. Create a new API key
3. Enable the Gemini API

#### Step 2: Configure Environment
```bash
# Add to .env file
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_CLOUD_LOCATION=us-central1
```

#### Step 3: Verify Installation
```python
from gemini_live import GeminiLiveService

service = GeminiLiveService()
available = service.is_available()
print(f"Gemini Live available: {available}")
```

### 8.4 Audio Requirements

| Parameter | Input | Output |
|-----------|-------|--------|
| Sample Rate | 16kHz | 24kHz |
| Channels | Mono | Mono |
| Format | PCM16 | PCM16 |
| Encoding | Linear16 | Linear16 |

### 8.5 WebSocket Protocol

**Connection:** `ws://localhost:8000/ws/voice`

**Client → Server Messages:**
```json
{
  "type": "audio",
  "data": "base64_encoded_pcm_audio",
  "language": "hi"
}
```

**Server → Client Messages:**
```json
{
  "type": "audio",
  "data": "base64_encoded_pcm_audio"
}
```

```json
{
  "type": "transcript",
  "text": "User said this"
}
```

```json
{
  "type": "response",
  "text": "AI response text"
}
```

### 8.6 Session Management

```python
from gemini_live import GeminiLiveService, SessionManager

# Create session
session_id = await session_manager.create_session(
    user_id="user123",
    language="hi"
)

# Send audio
response = await service.process_audio(
    session_id=session_id,
    audio_data=pcm_audio_bytes
)

# End session
await session_manager.end_session(session_id)
```

### 8.7 RAG Context Injection

Gemini Live responses are grounded with RAG context:

```python
# Before generating response
rag_context = await rag_pipeline.query(user_query)

# Inject into Gemini prompt
system_instruction = f"""
You are Palli Sahayak. Use this context to answer:

{rag_context}

Respond compassionately in the user's language.
"""
```

### 8.8 Language Configuration

```python
GEMINI_LANGUAGE_CONFIGS = {
    "hi": {
        "speech_config": {"language_code": "hi-IN"},
        "voice_name": "hi-IN-Wavenet-A"
    },
    "en": {
        "speech_config": {"language_code": "en-IN"},
        "voice_name": "en-IN-Wavenet-A"
    },
    "ta": {
        "speech_config": {"language_code": "ta-IN"},
        "voice_name": "ta-IN-Wavenet-A"
    }
}
```

### 8.9 Fallback Handling

If Gemini Live fails, the system falls back to:

```
Gemini Live → STT (Groq Whisper) → RAG → LLM → TTS (Edge TTS)
```

### 8.10 Testing Gemini Live

```python
# Test script
import asyncio
from gemini_live import GeminiLiveService

async def test():
    service = GeminiLiveService()

    # Read test audio
    with open("test_audio.wav", "rb") as f:
        audio_data = f.read()

    # Process
    response = await service.process_audio(
        session_id="test",
        audio_data=audio_data
    )

    print(f"Response: {response.text}")
    print(f"Audio length: {len(response.audio)} bytes")

asyncio.run(test())
```

---

## 9. For System Administrators

### 9.1 Admin Dashboard Overview

Access the admin dashboard at: `http://localhost:8000/admin`

**Available Tabs:**

| Tab | Purpose |
|-----|---------|
| **📁 Upload Documents** | Add knowledge base documents |
| **💬 Test Queries** | Test RAG responses |
| **📋 Manage Documents** | View/delete documents |
| **📊 Index Statistics** | Monitor system health |
| **🏥 Database Health** | Corruption detection & recovery |
| **🕸️ Knowledge Graph** | Neo4j entity management |
| **📈 GraphRAG** | Microsoft GraphRAG interface |

### 9.2 Document Management

#### Uploading Documents

1. Go to **Upload Documents** tab
2. Click "Upload File" or drag-drop
3. Optionally add metadata as JSON:
   ```json
   {"category": "pain_management", "language": "en"}
   ```
4. Click "Upload & Index"
5. Wait for confirmation

#### Removing Documents

1. Go to **Manage Documents** tab
2. Select document from dropdown
3. Click "Remove Document"
4. Confirm deletion

#### Best Practices

- **Document Quality**: Use well-formatted PDFs with clear text
- **Naming Convention**: Use descriptive filenames
- **Size Limits**: Keep documents under 50MB each
- **Regular Updates**: Re-upload updated documents to refresh index

### 9.3 Database Health Monitoring

#### Health Dashboard

The **Database Health** tab shows:

| Metric | Description | Action if Failed |
|--------|-------------|------------------|
| Connectivity | Can connect to ChromaDB | Restart server |
| Metadata Consistency | Documents match vectors | Auto-rebuild |
| Query Functionality | Search returns results | Check embeddings |
| Embedding Quality | Vectors are valid | Re-embed documents |
| Index Integrity | No corruption detected | Force rebuild |

#### Corruption Detection

The system detects corruption through:

1. **Orphaned Vectors** - Vectors without metadata
2. **Missing Embeddings** - Documents without vectors
3. **Invalid Dimensions** - Wrong embedding size
4. **Query Failures** - Search errors

**Corruption Score:**
- 0-49: Minor (no action needed)
- 50-79: Moderate (auto-rebuild recommended)
- 80+: Critical (force rebuild required)

#### Auto-Rebuild Process

When corruption is detected:

1. Click **"Auto Rebuild Database"**
2. System creates backup
3. Documents reloaded from storage
4. Vectors regenerated
5. Index rebuilt
6. Integrity verified

#### Force Rebuild

For critical issues:

1. Click **"Force Rebuild Database"**
2. Confirm the action
3. All vectors deleted and regenerated
4. May take several minutes for large databases

### 9.4 Monitoring & Logging

#### Log Levels

Set in `.env`:
```bash
DEBUG=true  # Verbose logging
# or
DEBUG=false  # Production logging
```

#### Log Files

```bash
# View logs in real-time
tail -f server.log

# Search for errors
grep "ERROR" server.log
```

#### Metrics to Monitor

| Metric | Normal Range | Alert Threshold |
|--------|--------------|-----------------|
| Query Latency | 1-3 seconds | > 5 seconds |
| Memory Usage | < 80% | > 90% |
| Vector Count | Stable | Unexpected drops |
| Error Rate | < 1% | > 5% |

### 9.5 Backup & Recovery

#### Manual Backup

```bash
# Backup vector database
cp -r data/chroma_db data/backups/chroma_db_$(date +%Y%m%d)

# Backup metadata
cp data/document_metadata.json data/backups/metadata_$(date +%Y%m%d).json
```

#### Automated Backups

Add to crontab:
```bash
# Daily backup at 2 AM
0 2 * * * /path/to/backup_script.sh
```

#### Recovery Procedure

1. Stop the server
2. Remove corrupted database:
   ```bash
   rm -rf data/chroma_db
   ```
3. Restore from backup:
   ```bash
   cp -r data/backups/chroma_db_YYYYMMDD data/chroma_db
   ```
4. Restart server
5. Verify with health check

### 9.6 Performance Tuning

#### Embedding Model Selection

| Model | Quality | Speed | Memory |
|-------|---------|-------|--------|
| all-MiniLM-L6-v2 | Good | Fast | Low |
| all-mpnet-base-v2 | Better | Medium | Medium |
| embeddinggemma | Best | Slow | High |

#### Chunk Size Optimization

```yaml
rag:
  chunk_size: 1000   # Smaller = more precise, slower
  chunk_overlap: 200  # Higher = better context, more storage
  top_k: 5           # More = better recall, higher latency
```

#### Caching Configuration

```python
# Query cache (in query_engine.py)
engine = GraphRAGQueryEngine(
    config,
    enable_cache=True,
    cache_maxsize=100,  # Max cached queries
    cache_ttl=3600      # 1 hour TTL
)
```

### 9.7 Security Considerations

#### API Authentication

Add to `.env`:
```bash
RAG_API_KEY=your_secure_api_key
```

Clients must include header:
```
Authorization: Bearer your_secure_api_key
```

#### Webhook Security

For Bolna webhooks:
```bash
BOLNA_WEBHOOK_SECRET=your_webhook_secret
```

Verify signatures in webhook handler.

#### Network Security

- Run behind reverse proxy (nginx)
- Use HTTPS in production
- Restrict admin UI access
- Rate limit API endpoints

### 9.8 Scaling Considerations

#### Horizontal Scaling

```bash
# Run multiple workers
gunicorn simple_rag_server:app -w 4 -k uvicorn.workers.UvicornWorker
```

#### Database Scaling

For large document collections:
- Consider PostgreSQL with pgvector
- Use Redis for caching
- Implement sharding for millions of documents

---

## 10. For End Users

### 10.1 Getting Help with Palli Sahayak

Palli Sahayak is here to help you with questions about palliative care, pain management, and caregiving. Here's how to use it:

### 10.2 Using the Phone Helpline

#### How to Call

1. **Dial the helpline number** provided by your healthcare provider
2. **Wait for the greeting**: "Namaste! I am Palli Sahayak, your palliative care assistant. How can I help you today?"
3. **Speak naturally** in Hindi, English, Marathi, or Tamil
4. **Ask your question** about pain, symptoms, or caregiving
5. **Listen to the response** and ask follow-up questions

#### Tips for Better Conversations

- **Speak clearly** and at a normal pace
- **One question at a time** for best results
- **Wait for the response** before asking another question
- **Say "repeat"** if you didn't understand the answer

#### What You Can Ask

| Topic | Example Questions |
|-------|-------------------|
| **Pain Management** | "What can help with pain at home?" |
| **Symptoms** | "What causes nausea and how to manage it?" |
| **Medications** | "What are the side effects of morphine?" |
| **Caregiving** | "How do I care for a bedridden patient?" |
| **Emotional Support** | "I'm feeling overwhelmed, can you help?" |
| **Emergency** | "When should I call an ambulance?" |

#### Emergency Situations

If you or your loved one has:
- **Severe breathing difficulty**
- **Uncontrolled severe pain**
- **Loss of consciousness**
- **Seizures**

**Call emergency services immediately: 108 or 112**

### 10.3 Using WhatsApp

#### Getting Started

1. **Save the WhatsApp number** provided by your healthcare provider
2. **Send a message** to start the conversation
3. **Type your question** or **send a voice message**

#### Text Messages

Simply type your question:
```
You: What helps with constipation?

Palli Sahayak: Constipation is common with certain medications.
Here are some helpful tips:
1. Drink plenty of water (6-8 glasses daily)
2. Eat fiber-rich foods like fruits and vegetables
3. Gentle walking if possible
4. Ask your doctor about stool softeners

[Sources: symptom_guide.pdf: pg 23]
```

#### Voice Messages

1. **Hold the microphone button** in WhatsApp
2. **Speak your question** in any supported language
3. **Release to send**
4. **Receive a voice response** in your language

#### Language Commands

Change your preferred language:
- `/lang hi` - Switch to Hindi
- `/lang en` - Switch to English
- `/lang bn` - Switch to Bengali
- `/lang ta` - Switch to Tamil

### 10.4 Using Web Voice

#### Accessing Web Voice

1. **Open your browser** and go to the provided URL
2. **Allow microphone access** when prompted
3. **Click "Start Speaking"** to begin
4. **Speak naturally** and wait for response

#### Browser Requirements

- Chrome (recommended)
- Firefox
- Edge
- Safari (limited support)

**Note:** Internet connection required for web voice.

### 10.5 Understanding Responses

#### What the AI Provides

| Provides | Does NOT Provide |
|----------|------------------|
| General information | Specific diagnoses |
| Comfort measures | Prescription changes |
| Caregiving tips | Emergency treatment |
| Emotional support | Medical decisions |
| Source references | Personalized dosages |

#### Citations

Every response includes sources:
```
[Sources: pain_management_guide.pdf: pg 12, 15; caregiver_handbook.pdf: pg 8]
```

This shows which documents were used to answer your question.

### 10.6 Privacy & Safety

#### Your Privacy

- Conversations are **confidential**
- Personal information is **not stored** permanently
- Call recordings are used only for **quality improvement**

#### Safety Guidelines

- **Always consult your doctor** before changing medications
- **Palli Sahayak is not a replacement** for professional medical care
- **In emergencies**, call 108 or 112 immediately
- **Share concerns** with your healthcare team

### 10.7 Frequently Asked Questions (End Users)

**Q: Can I call at any time?**
A: Yes, Palli Sahayak is available 24 hours a day, 7 days a week.

**Q: Does it understand my language?**
A: Yes, Palli Sahayak understands Hindi, English, Marathi, Tamil, Bengali, and Gujarati.

**Q: Will a human call me back?**
A: Palli Sahayak is an AI assistant. For human support, ask for a callback during the conversation.

**Q: Is it free to use?**
A: The service itself is free. Standard phone/data charges from your provider may apply.

**Q: Can it prescribe medications?**
A: No, Palli Sahayak cannot prescribe medications. It provides information only. Always consult your doctor for prescriptions.

**Q: What if I don't understand the answer?**
A: Say "please explain again" or "can you say that more simply?"

**Q: Can I ask about any health topic?**
A: Palli Sahayak specializes in palliative care. For other health topics, please consult appropriate resources.

---

## 11. API Reference

### 11.1 Core Endpoints

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-01T12:00:00Z",
  "services": {
    "rag": "ok",
    "vectordb": "ok",
    "llm": "ok"
  }
}
```

#### RAG Query
```http
POST /api/query
Content-Type: application/json

{
  "query": "What helps with pain?",
  "language": "en",
  "top_k": 5
}
```

**Response:**
```json
{
  "response": "For pain management, consider...",
  "sources": [
    {
      "document": "pain_guide.pdf",
      "pages": [12, 15],
      "relevance": 0.89
    }
  ],
  "citations": "[Sources: pain_guide.pdf: pg 12,15]"
}
```

### 11.2 Document Endpoints

#### Upload Document
```http
POST /api/upload
Content-Type: multipart/form-data

file: <binary>
metadata: {"category": "medical"}
```

#### List Documents
```http
GET /api/documents
```

#### Delete Document
```http
DELETE /api/documents/{filename}
```

### 11.3 Bolna Endpoints

#### Query from Bolna
```http
POST /api/bolna/query
Content-Type: application/json

{
  "query": "What is morphine?",
  "language": "hi",
  "context": "User asking about pain medication",
  "source": "bolna_call"
}
```

#### Webhook Events
```http
POST /api/bolna/webhook
Content-Type: application/json

{
  "type": "call_ended",
  "call_id": "call_123",
  "transcript": [...],
  "duration_seconds": 180
}
```

### 11.4 Knowledge Graph Endpoints

#### Query Graph
```http
POST /api/kg/query
Content-Type: application/json

{
  "query": "What treats nausea?",
  "max_results": 10
}
```

#### Get Treatments
```http
GET /api/kg/treatments/{symptom}
```

### 11.5 GraphRAG Endpoints

#### Query GraphRAG
```http
POST /api/graphrag/query
Content-Type: application/json

{
  "query": "Overview of pain management approaches",
  "method": "global"
}
```

---

## 12. Troubleshooting

### 12.1 Common Issues

#### Server Won't Start

**Symptom:** Error on startup

**Solutions:**
1. Check Python version: `python --version` (need 3.10+)
2. Verify dependencies: `pip install -r requirements.txt`
3. Check port availability: `lsof -i :8000`
4. Verify .env file exists with GROQ_API_KEY

#### No Response from RAG

**Symptom:** Queries return empty or error

**Solutions:**
1. Check documents are uploaded (Index Statistics tab)
2. Verify ChromaDB health (Database Health tab)
3. Test LLM connection: Check API key in .env
4. Review logs for errors

#### Voice Not Working

**Symptom:** Bolna/Gemini voice fails

**Solutions:**
1. Verify API keys in .env
2. Check webhook URL is publicly accessible
3. Test audio format compatibility
4. Review voice provider status page

#### WhatsApp Not Responding

**Symptom:** No response to WhatsApp messages

**Solutions:**
1. Verify Twilio credentials in .env
2. Check webhook URL in Twilio dashboard
3. Ensure ngrok is running (if using)
4. Test with `/health` endpoint

### 12.2 Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `GROQ_API_KEY not set` | Missing API key | Add key to .env |
| `ChromaDB connection failed` | Database issue | Check data/chroma_db |
| `Embedding dimension mismatch` | Model changed | Force rebuild database |
| `Rate limit exceeded` | Too many API calls | Wait or upgrade plan |
| `WebSocket connection failed` | Network issue | Check firewall/proxy |

### 12.3 Performance Issues

#### Slow Queries

1. **Reduce top_k**: Lower from 5 to 3
2. **Use caching**: Enable query cache
3. **Optimize chunks**: Adjust chunk_size
4. **Check network**: Verify API latency

#### High Memory Usage

1. **Reduce batch size**: Lower concurrent operations
2. **Clear cache**: Remove old TTS files
3. **Limit documents**: Archive unused documents
4. **Restart server**: Clear memory leaks

### 12.4 Getting Help

1. **Check logs**: `tail -f server.log`
2. **Review documentation**: This guide and API docs
3. **Search issues**: https://github.com/inventcures/rag_gci/issues
4. **Report bugs**: Create new issue with details

---

## 13. FAQ

### General Questions

**Q: What languages are supported?**
A: Hindi, English, Marathi, Tamil, Bengali, and Gujarati for voice. Document processing supports English and transliterated content.

**Q: Is this system HIPAA compliant?**
A: The base system does not include HIPAA-specific features. For healthcare deployment, additional security measures are required.

**Q: Can I run this offline?**
A: The core RAG pipeline requires internet for LLM calls. Local models can be configured for offline use.

**Q: How accurate are the answers?**
A: Accuracy depends on the quality of uploaded documents. All answers include source citations for verification.

### Technical Questions

**Q: Can I use different LLM providers?**
A: Yes, the system supports Groq, OpenAI, and local models via LiteLLM.

**Q: How do I add new languages?**
A: Add language configuration to config.yaml under tts.languages and stt.supported_languages.

**Q: Can I customize the voice?**
A: Yes, Edge TTS offers multiple voices per language. Configure in config.yaml.

**Q: How large can the document collection be?**
A: ChromaDB handles millions of vectors. For very large collections, consider PostgreSQL with pgvector.

### Deployment Questions

**Q: Can I deploy on cloud providers?**
A: Yes, the system runs on AWS, GCP, Azure, or any provider supporting Python.

**Q: What are the minimum server requirements?**
A: 4GB RAM, 2 vCPU, 10GB storage for a small deployment.

**Q: How do I enable HTTPS?**
A: Use a reverse proxy (nginx) with SSL certificates, or deploy behind a load balancer.

---

## Appendix A: API Key Setup Guides

### Groq API Key
1. Visit https://console.groq.com
2. Sign up for free account
3. Go to API Keys section
4. Click "Create API Key"
5. Copy and add to .env

### Twilio Setup
1. Visit https://console.twilio.com
2. Create account (free trial available)
3. Get Account SID and Auth Token
4. Set up WhatsApp sandbox or number
5. Configure webhook URL

### Bolna.ai Setup
1. Visit https://app.bolna.ai
2. Create account
3. Go to Settings → API Keys
4. Create and copy API key
5. Create agent with configuration
6. Assign phone number

### Gemini API Setup
1. Visit https://aistudio.google.com/app/apikey
2. Sign in with Google account
3. Create new API key
4. Enable Gemini API in Cloud Console
5. Copy key to .env

### Retell.AI Setup
1. Visit https://dashboard.retellai.com
2. Create account and obtain API key
3. Add to .env: `RETELL_API_KEY=your_key`
4. Create an agent with Custom LLM option
5. Configure Custom Function for RAG queries (see below)

#### Custom Function Configuration

When using Retell.AI with ngrok for local development, configure the Custom Function with these values:

| Field | Value |
|-------|-------|
| **Name** | `query_rag_knowledge_base` |
| **Description** | Query the Palli Sahayak palliative care knowledge base to get accurate medical information about symptoms, medications, caregiving guidance, and emotional support for patients and families. |
| **API Endpoint** | `POST` `https://<your-ngrok-subdomain>.ngrok-free.app/api/bolna/query` |
| **Timeout (ms)** | `120000` |
| **Headers** | `Content-Type`: `application/json` |

**Parameters JSON Schema:**
```json
{
  "type": "object",
  "properties": {
    "user_query": {
      "type": "string",
      "description": "The user's question about palliative care, symptoms, medications, or caregiving"
    },
    "user_language": {
      "type": "string",
      "enum": ["hi", "en", "mr", "ta"],
      "description": "Language code for the response (hi=Hindi, en=English, mr=Marathi, ta=Tamil)"
    },
    "conversation_context": {
      "type": "string",
      "description": "Recent conversation history for context-aware responses"
    }
  },
  "required": ["user_query"]
}
```

**Notes:**
- Replace `<your-ngrok-subdomain>` with your actual ngrok subdomain
- The `/api/bolna/query` endpoint works for all voice providers
- Increase timeout for complex queries that require RAG processing
- Set `user_language` to match the caller's preferred language

#### ngrok Static Domain Setup (Recommended)

By default, ngrok generates a new random URL each time you restart it. To avoid updating Retell every time, use a **static domain**.

**Step 1: Get Your Free Static Domain**
1. Go to https://dashboard.ngrok.com
2. Log in (or create free account)
3. Click **"Domains"** in left sidebar (or visit https://dashboard.ngrok.com/domains)
4. Free accounts get ONE auto-generated "dev domain" like:
   - `untoppable-extraterritorially-nellie.ngrok-free.dev`
5. Copy your domain name

**Note:** Free tier domains have auto-generated names. Custom domain names (e.g., `palli-sahayak.ngrok-free.app`) require a paid subscription.

**Step 2: Use Your Static Domain**
```bash
# Run ngrok with your static domain
ngrok http 8000 --domain=YOUR-DOMAIN.ngrok-free.dev

# Example:
ngrok http 8000 --domain=untoppable-extraterritorially-nellie.ngrok-free.dev
```

**Step 3: Update Retell Custom Function (One Time)**

Set your Retell Custom Function API Endpoint to:
```
https://YOUR-DOMAIN.ngrok-free.dev/api/bolna/query
```

This URL will remain constant across ngrok restarts.

**Step 4: Verify ngrok is Running**
```bash
# Check current ngrok URL
curl -s http://127.0.0.1:4040/api/tunnels | python3 -c "import sys,json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
```

#### Single Command: Start ngrok + Server

Run both ngrok and the RAG server in one command (ngrok runs in background):

```bash
ngrok http 8000 --domain=untoppable-extraterritorially-nellie.ngrok-free.dev > /dev/null 2>&1 & sleep 2 && cd /Users/tp53/Documents/tp53_AA/llms4palliative_gci/demo_feb2025/rag_gci && source ./venv/bin/activate && export GOOGLE_APPLICATION_CREDENTIALS="/Users/tp53/palli-sahayak-credentials.json" GOOGLE_CLOUD_PROJECT="palli-sahayak" VERTEX_AI_LOCATION="asia-south1" && python simple_rag_server.py -p r
```

**To stop everything:**
```bash
# Kill ngrok
pkill -f ngrok

# Kill server (Ctrl+C in terminal, or)
pkill -f simple_rag_server
```

#### Viewing RAG Pipeline Logs

Filter server logs to see RAG-specific activity:

```bash
# RAG queries and voice calls
python simple_rag_server.py -p r 2>&1 | grep -i "rag\|query\|📞|🗣️"

# Document retrieval activity
python simple_rag_server.py -p r 2>&1 | grep -i "retriev\|chunk\|document\|source"

# Embedding and vector operations
python simple_rag_server.py -p r 2>&1 | grep -i "embed\|vector\|chroma"

# All important events (colored)
python simple_rag_server.py -p r 2>&1 | grep --color=always -E "RAG|Query|Error|WARNING|retriev|📞|🗣️"
```

**Tail existing log file:**
```bash
# Follow log file with RAG filter
tail -f server.log | grep -i "rag\|query\|retriev"
```

#### Updating Retell Custom Function URL

When you need to update the ngrok URL in Retell:

1. **Log in** to https://dashboard.retellai.com
2. Click **"Agents"** in left sidebar
3. Select your **Palli Sahayak agent**
4. Scroll to **"Custom Functions"** section
5. Click **Edit** on `query_rag_knowledge_base`
6. Update **"API Endpoint"** URL field with new ngrok URL
7. Click **"Save"**

---

**Document End**

*This guide is maintained by the Palli Sahayak development team.*
*Last updated: December 2025*
