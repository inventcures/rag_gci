# Agent Coding Guidelines for RAG GCI Project

## Commands
- **Run server**: `./run.sh` or `python main.py` (starts FastAPI + Gradio admin UI)
- **Run simple server**: `./run_simple.sh` or `python simple_rag_server.py`
- **Test**: `python test_pipeline.py` (main tests), `python test_corruption_recovery.py` (DB tests)
- **Single test**: `python <test_file>.py` (e.g., `python test_citations.py`)
- **Lint**: `flake8 *.py` (if installed)
- **Format**: `black *.py` (if installed)

## Architecture
- **FastAPI backend** at port 8000 with Gradio admin UI mounted at `/admin`
- **Two RAG implementations**: Full (rag_server.py using Kotaemon) and Simple (simple_rag_server.py, file-based)
- **Vector DB**: ChromaDB (local, at ./data/chroma_db)
- **Document storage**: uploads/ directory with metadata in ./data/documents
- **WhatsApp bot**: whatsapp_bot.py (Twilio integration with STT/TTS)
- **Config**: config.yaml (main), .env (API keys)
- **Key services**: STT (Groq Whisper), TTS (Edge TTS), LLM (Groq llama-3.1-8b-instant)

## Code Style
- **Imports**: Standard library → third-party → local (grouped with blank lines)
- **Types**: Use type hints (Dict, List, Optional, Any from typing)
- **Async**: Use async/await for I/O operations; FastAPI endpoints are async
- **Logging**: Use logger.info/error/warning; logger = logging.getLogger(__name__)
- **Error handling**: Try-except with specific exceptions; return {"status": "error", "error": str(e)}
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Docstrings**: Triple-quoted strings for classes/functions

---

## Gemini Live API Voice Integration

### Overview
This project is being enhanced with **Gemini Live API** for real-time voice conversations. See `live_call_detailed_specifications.md` for complete implementation details.

### New Directory Structure
```
rag_gci/
├── gemini_live/              # NEW: Gemini Live API integration
│   ├── __init__.py
│   ├── service.py            # GeminiLiveService class
│   ├── session_manager.py    # Session lifecycle management
│   ├── audio_handler.py      # Audio format conversion (PCM)
│   ├── context_manager.py    # RAG context injection
│   └── config.py             # Gemini-specific settings
├── web_client/               # NEW: Web voice interface
│   ├── index.html            # Voice conversation UI
│   ├── audio-worklet.js      # AudioWorklet for capture
│   └── gemini-client.js      # WebSocket client
└── telephony/                # OPTIONAL: PSTN support
    ├── twilio_voice.py       # Phone call handling
    └── daily_bridge.py       # WebRTC bridge
```

### Key Technical Specs (Gemini Live API)
| Parameter | Value |
|-----------|-------|
| Input Audio | 16-bit PCM, 16kHz, mono, little-endian |
| Output Audio | 16-bit PCM, 24kHz, mono, little-endian |
| Connection | WebSocket (WSS) |
| Max Session | 15 minutes (audio-only) |
| Supported Languages | en-IN, hi-IN, mr-IN, ta-IN |

### Implementation Phases
1. **Phase 1**: Core Infrastructure (`gemini_live/` module)
2. **Phase 2**: GeminiLiveService implementation
3. **Phase 3**: WhatsApp integration (modify `whatsapp_bot.py`)
4. **Phase 4**: Web client (`web_client/`)
5. **Phase 5**: FastAPI WebSocket endpoint (`/ws/voice`)

### New Dependencies
```
google-genai>=1.0.0
websockets>=12.0
numpy>=1.24.0
```

### New Environment Variables
```bash
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GEMINI_API_KEY=your-api-key
```

### New Endpoints
| Endpoint | Purpose |
|----------|---------|
| `GET /voice` | Web voice interface |
| `WS /ws/voice` | WebSocket for voice streaming |

### Audio Flow
```
User speaks → PCM (16kHz) → WebSocket → Gemini Live API
                                              ↓
User hears ← PCM (24kHz) ← WebSocket ← Audio response
```

### RAG Context Injection
The system injects relevant RAG context into Gemini sessions:
1. Query vector DB with recent conversation context
2. Format as system instruction or client content
3. Gemini uses this for grounded responses

### Fallback Strategy
If Gemini Live fails, fall back to existing pipeline:
```
STT (Groq Whisper) → RAG Query → LLM → TTS (Edge TTS)
```

### Testing Commands
```bash
# Test Gemini Live connection
python -c "from gemini_live.service import GeminiLiveService; ..."

# Test audio conversion
python -c "from gemini_live.audio_handler import AudioHandler; ..."

# Run integration tests
pytest tests/test_gemini_live.py
```

### Documentation
- **Full Specs**: `live_call_detailed_specifications.md`
- **Gemini Docs**: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api
- **Language Support**: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api/configure-language-voice

---

## Palli Sahayak Voice AI Agent Helpline (Bolna.ai Integration)

### Overview
The system has been renamed to **Palli Sahayak Voice AI Agent Helpline** with a hybrid architecture:
- **Primary Orchestrator**: Bolna.ai (phone calls)
- **Fallback**: Gemini Live API (web voice)

See `bolna_palli-sahayak-helpline_specs.md` for complete specifications.

### New Directory Structure (Bolna)
```
rag_gci/
├── bolna_integration/           # NEW: Bolna.ai integration
│   ├── __init__.py
│   ├── client.py               # BolnaClient API class
│   ├── config.py               # Agent configuration
│   └── webhooks.py             # Webhook event handler
├── voice_router.py             # NEW: Provider routing logic
└── bolna_palli-sahayak-helpline_specs.md  # Complete specs
```

### Key Technical Specs (Bolna)
| Component | Provider |
|-----------|----------|
| ASR (Speech-to-Text) | Deepgram (nova-2) |
| LLM | OpenAI (gpt-4o-mini) |
| TTS (Text-to-Speech) | ElevenLabs (eleven_multilingual_v2) |
| Telephony | Twilio/Plivo/Exotel |

### New API Endpoints (Bolna)
| Endpoint | Purpose |
|----------|---------|
| `POST /api/bolna/query` | RAG query endpoint for Bolna custom function calls |
| `POST /api/bolna/webhook` | Webhook for call events |

### New Environment Variables (Bolna)
```bash
BOLNA_API_KEY=your-bolna-api-key
BOLNA_AGENT_ID=your-agent-id
BOLNA_WEBHOOK_SECRET=your-webhook-secret
BOLNA_PHONE_NUMBER=+91XXXXXXXXXX
```

### Custom Function Call Integration
Bolna calls the RAG pipeline via custom function:
```json
{
  "name": "query_rag_knowledge_base",
  "description": "Query palliative care knowledge base",
  "value": {
    "method": "POST",
    "url": "https://YOUR_SERVER/api/bolna/query"
  }
}
```

### Voice Router Logic
```python
# Priority:
# 1. Bolna - for phone calls
# 2. Gemini Live - for web voice
# 3. STT+LLM+TTS - ultimate fallback
```

### Supported Languages
- Hindi (hi-IN) - Primary
- English (en-IN)
- Marathi (mr-IN)
- Tamil (ta-IN)

### Implementation Phases (Bolna)
1. **Phase 1**: Create `bolna_integration/` module
2. **Phase 2**: Add `/api/bolna/query` endpoint to `simple_rag_server.py`
3. **Phase 3**: Implement `voice_router.py`
4. **Phase 4**: Configuration updates (config.yaml, .env)
5. **Phase 5**: Testing & Deployment

### Documentation
- **Complete Bolna Specs**: `bolna_palli-sahayak-helpline_specs.md`
- **Bolna Docs**: https://www.bolna.ai/docs/introduction
- **Custom Functions**: https://www.bolna.ai/docs/tool-calling/custom-function-calls
- **Codebase Wiki**: https://deepwiki.com/inventcures/rag_gci

---

## Knowledge Graph Integration (Neo4j)

### Overview
The RAG system includes a knowledge graph module for enhanced medical entity relationships.
Inspired by OncoGraph (https://github.com/ib565/OncoGraph).

### Directory Structure
```
knowledge_graph/
├── __init__.py           # Module exports
├── neo4j_client.py       # Neo4j connection & queries
├── entity_extractor.py   # LLM + pattern entity extraction
├── graph_builder.py      # Graph construction
├── cypher_generator.py   # NL → Cypher translation
├── visualizer.py         # Cytoscape.js visualization
└── kg_rag.py            # Main KG-RAG integration
```

### Node Types (Palliative Care)
| Node Type | Description |
|-----------|-------------|
| Symptom | Pain, nausea, fatigue, etc. |
| Medication | Morphine, ondansetron, etc. |
| Condition | Cancer, heart failure, etc. |
| Treatment | Chemotherapy, palliative care, etc. |
| SideEffect | Constipation, drowsiness, etc. |

### Relationship Types
| Relationship | Pattern |
|--------------|---------|
| TREATS | (Medication)-[:TREATS]->(Symptom) |
| CAUSES | (Condition)-[:CAUSES]->(Symptom) |
| SIDE_EFFECT_OF | (SideEffect)-[:SIDE_EFFECT_OF]->(Medication) |
| MANAGES | (Treatment)-[:MANAGES]->(Condition) |

### Environment Variables
```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

### Usage
```python
from knowledge_graph import KnowledgeGraphRAG

kg = KnowledgeGraphRAG()
await kg.initialize()

# Query with natural language
result = await kg.query("What medications treat pain?")

# Get visualization
viz_html = kg.get_visualization_html(result["visualization"])
```

### Features
- **Hybrid Retrieval**: Vector search + graph traversal
- **Entity Extraction**: LLM + regex pattern matching
- **Cypher Generation**: Natural language → Cypher queries
- **Visualization**: Interactive Cytoscape.js graphs
- **Safety**: Query validation blocks write operations

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/kg/health` | GET | Health status check |
| `/api/kg/stats` | GET | Graph statistics |
| `/api/kg/query` | POST | Natural language query |
| `/api/kg/extract` | POST | Entity extraction from text |
| `/api/kg/entity/{name}` | GET | Get entity subgraph |
| `/api/kg/treatments/{symptom}` | GET | Find treatments for symptom |
| `/api/kg/side-effects/{medication}` | GET | Get medication side effects |
| `/api/kg/search` | GET | Search entities by name |
| `/api/kg/visualization/{name}` | GET | HTML visualization |
| `/api/kg/import-base` | POST | Import base knowledge |

### Admin UI Tab
The Gradio admin interface includes a **Knowledge Graph** tab with:
- **Extract Entities**: Extract medical entities from free text
- **Query Graph**: Natural language queries
- **Find Treatments**: Lookup treatments for symptoms
- **Side Effects**: Check medication side effects
- **Graph Status**: Health and statistics monitoring
