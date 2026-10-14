# RAG Server with WhatsApp Bot Integration

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/inventcures/rag_gci)

A comprehensive RAG (Retrieval-Augmented Generation) pipeline built with Kotaemon, featuring:
- 📄 Document processing and indexing
- 🖥️ Web-based admin UI for corpus management  
- 📱 WhatsApp bot with voice message support
- 🗣️ Speech-to-Text for Indian languages (Hindi, Bengali, Tamil, Gujarati)
- 🔊 Text-to-Speech responses in multiple languages
- 🌐 ngrok integration for external access

## Features

### 🎯 Core RAG Pipeline
- Built on Kotaemon framework for robust document processing
- Support for PDF, DOCX, TXT, HTML, XLSX files
- Vector search with citation support
- Multiple retrieval strategies and reranking
- **🔧 Automatic corruption detection and database rebuilding**
- **🛡️ Robust error recovery and data integrity monitoring**

### 🌍 Multilingual Support
- **STT Languages**: Hindi, Bengali, Tamil, Gujarati, English
- **TTS Languages**: Same as STT with native voice synthesis
- Automatic language detection from audio
- Language-specific response generation

### 📱 WhatsApp Integration
- Text message queries
- Voice message support with automatic transcription
- Audio responses in user's preferred language
- Language selection commands (`/lang hi`, `/lang bn`, etc.)

### 🖥️ Admin Interface
- Document upload and management
- Query testing interface
- Index statistics and monitoring
- **🏥 Database health monitoring and auto-rebuild**
- Configuration management

## Quick Start

### 1. Setup Environment

```bash
# Clone and navigate to the project
cd rag_gci

# Run the setup script
python setup.py

# Install dependencies (handled by setup)
pip install -r requirements.txt
```

### 2. Configure API Keys

Edit `.env` file with your API keys:

```bash
# LLM (llm/client.py). Either Vertex AI, pinned to asia-south1:
GOOGLE_CLOUD_PROJECT=your-gcp-project
# ...or the Gemini API (not region-pinned):
# GEMINI_API_KEY=your_gemini_api_key

# Speech-to-text and the Indian-hosted voice fallback
SARVAM_API_KEY=your_sarvam_api_key

# Optional, only if llm.fallback.enabled is true (DeepSeek is hosted outside India)
# DEEPSEEK_API_KEY=your_deepseek_api_key

# Optional - for WhatsApp Bot
WHATSAPP_VERIFY_TOKEN=your_verify_token
WHATSAPP_ACCESS_TOKEN=your_access_token
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
```

Every text LLM call (answers, translation, graph extraction, agents) goes through
one client, configured by the `llm:` block in `config.yaml`. See
[docs/llm.md](docs/llm.md) for the model, the fallback, and how to switch.

### 3. Start the Server

#### Option A: Using the run script (Recommended)
```bash
./run.sh
```

#### Option B: Direct Python execution
```bash
python main.py
```

#### Option C: Custom configuration
```bash
python main.py --host 0.0.0.0 --port 8000 --no-ngrok
```

### 4. Access the Application

Once started, you'll see:

```
🚀 RAG SERVER WITH WHATSAPP BOT
==================================================
📊 Admin UI: http://localhost:8000/admin
🔗 API Docs: http://localhost:8000/docs
💚 Health Check: http://localhost:8000/health
📱 WhatsApp Webhook: http://localhost:8000/webhook
🌐 Public URL: https://abc123.ngrok.io (if ngrok enabled)
```

## Usage Guide

### 📄 Document Management

1. Open Admin UI at `http://localhost:8000/admin`
2. Go to "Upload Documents" tab
3. Select files (PDF, DOCX, TXT, etc.)
4. Add metadata (optional): `{"category": "medical", "language": "en"}`
5. Click "Upload & Index"

### 💬 Testing Queries

1. Use "Test Queries" tab in Admin UI
2. Enter questions like:
   - "What is palliative care?"
   - "How to manage pain in cancer patients?"
   - "What are the principles of hospice care?"

### 📱 WhatsApp Bot Setup

1. **Get WhatsApp Business Account**:
   - Create Facebook Business account
   - Set up WhatsApp Business API
   - Get access token and phone number ID

2. **Configure Webhook**:
   - Use the public ngrok URL: `https://your-ngrok-url.ngrok.io/webhook`
   - Set verify token in `.env` file

3. **Test the Bot**:
   - Send text messages
   - Send voice messages in supported languages
   - Use language commands: `/lang hi`, `/lang bn`, `/lang ta`, `/lang gu`

### 🗣️ Voice Features

**Sending Voice Queries**:
1. Send voice message to WhatsApp bot
2. Bot transcribes automatically
3. Detects language from speech
4. Returns text + audio response

**Language Commands**:
- `/lang hi` - Set to Hindi
- `/lang bn` - Set to Bengali  
- `/lang ta` - Set to Tamil
- `/lang gu` - Set to Gujarati
- `/lang en` - Set to English

### 🏥 Database Health & Auto-Rebuild

**Corruption Detection**:
- **Multi-layer health monitoring**: Connectivity, metadata consistency, query functionality
- **Automatic detection**: Identifies corruption from frequent add/remove operations
- **Health scoring**: 0-100 corruption score with severity levels (minor/moderate/critical)
- **Real-time monitoring**: Continuous health checks during operations

**Auto-Rebuild System**:
- **Intelligent recovery**: Automatically rebuilds vector database when corruption detected
- **Safe operation**: Creates backups before rebuilding, with rollback capability
- **Zero-downtime**: Continues serving responses during rebuild process
- **Progress tracking**: Detailed rebuild statistics and real-time status updates

**Admin Interface**:
1. Navigate to "🏥 Database Health" tab
2. **🔍 Check Health**: Scan database for corruption issues
3. **🔧 Auto Rebuild**: Rebuild only if corruption detected
4. **⚡ Force Rebuild**: Manual rebuild regardless of health status
5. **Real-time logs**: Monitor rebuild progress and results

**Programmatic Access**:
```python
# Check database health
health_status = rag.check_database_health()
print(f"Corrupted: {health_status['is_corrupted']}")

# Auto-rebuild if needed  
rebuild_result = await rag.auto_rebuild_database()

# Query with automatic recovery
result = await rag.query_with_auto_recovery("your question")
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   WhatsApp      │───▶│   FastAPI        │───▶│   RAG Pipeline  │
│   Bot           │    │   Server         │    │   (Kotaemon)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   STT Service   │    │   Admin UI       │    │   Vector Store  │
│   (Groq)        │    │   (Gradio)       │    │   (Chroma)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐
│   TTS Service   │
│   (Edge TTS)    │
└─────────────────┘
```

## API Endpoints

### Core Endpoints
- `GET /` - Redirect to admin UI
- `GET /admin` - Admin interface
- `GET /health` - Health check
- `GET /docs` - API documentation

### WhatsApp Webhook
- `GET /webhook` - Webhook verification
- `POST /webhook` - Message handling

### API Routes
- `POST /api/query` - Direct query endpoint
- `POST /api/upload` - File upload endpoint
- `POST /api/set_language` - Set user language preference
- `GET /api/health-check` - Database health monitoring
- `POST /api/rebuild` - Trigger database rebuild

## Configuration

### Settings Files
- `config.yaml` - Main configuration
- `.env` - Environment variables and API keys
- `kotaemon-main/settings.yaml` - Kotaemon-specific settings

### Key Configuration Options

```yaml
# LLM Configuration
llm:
  model: "gemini-3.5-flash"     # LLM_MODEL; gemini-3.8-flash is one line away
  location: "asia-south1"
  fallback:
    enabled: false              # DeepSeek; LLM_FALLBACK_ENABLED
  
# Embedding Configuration  
embedding:
  provider: "fastembed"
  model: "BAAI/bge-small-en-v1.5"

# STT Configuration
stt:
  provider: "groq"
  model: "whisper-large-v3"
  
# TTS Configuration
tts:
  provider: "edge_tts"
  supported_languages:
    hi: "hi-IN-SwaraNeural"
    bn: "bn-IN-TanishaaNeural"
    ta: "ta-IN-PallaviNeural"
    gu: "gu-IN-DhwaniNeural"
```

## Testing

Run the comprehensive test suite:

```bash
# Main pipeline tests
python test_pipeline.py

# Database corruption and recovery tests
python test_corruption_recovery.py
```

Tests include:
- ✅ Document processing and indexing
- ✅ Query processing and responses
- ✅ STT service functionality
- ✅ TTS service functionality  
- ✅ Multilingual support
- ✅ **Database corruption detection**
- ✅ **Auto-rebuild functionality**
- ✅ **Recovery from corruption scenarios**
- ✅ End-to-end pipeline

## Deployment

### Local Development
```bash
# Start with ngrok for external access
./run.sh

# Start without ngrok (local only)
./run.sh --no-ngrok
```

### Production Considerations
1. **Security**: Enable authentication in admin UI
2. **Scaling**: Use multiple workers with gunicorn
3. **Monitoring**: Add logging and metrics
4. **Backup**: Regular database and index backups
5. **Reliability**: Built-in corruption detection and auto-rebuild ensures 99%+ uptime
6. **Maintenance**: Schedule periodic health checks via `/api/health-check` endpoint

### ngrok for External Access
```bash
# Install ngrok
brew install ngrok

# Authenticate (one-time setup)
ngrok authtoken your_auth_token

# ngrok will start automatically with the server
```

## Troubleshooting

### Common Issues

**"Error generating response: gemini: no GOOGLE_CLOUD_PROJECT or GEMINI_API_KEY"**
- Set `GOOGLE_CLOUD_PROJECT` (Vertex AI) or `GEMINI_API_KEY` in `.env`

**"kotaemon-main directory not found"**
- Ensure Kotaemon codebase is in the project directory
- Check the directory structure matches requirements

**"ngrok not found"**
- Install ngrok: `brew install ngrok`
- Or run with `--no-ngrok` flag for local testing

**Audio processing fails**
- Check if edge-tts is installed: `pip install edge-tts`
- Verify audio file permissions and storage

**WhatsApp webhook verification fails**
- Check webhook URL and verify token
- Ensure ngrok tunnel is active
- Verify WhatsApp Business API setup

**Database corruption detected**
- Use "🏥 Database Health" tab for diagnosis
- Run auto-rebuild: Click "🔧 Auto Rebuild" button
- For severe corruption: Use "⚡ Force Rebuild"
- Check logs for rebuild progress and errors

**Queries return "could not find answer" after adding/removing documents**
- This indicates potential vector database corruption
- Navigate to Admin UI → Database Health tab
- Click "🔍 Check Health" to diagnose
- Use "🔧 Auto Rebuild" to fix automatically

**Rebuild process fails**
- Check available disk space for backups
- Ensure source documents are still in uploads/ directory
- Review rebuild logs for specific error messages
- Use manual rebuild script: `python test_corruption_recovery.py`

### Logs and Debugging

Check logs in:
- `logs/main.log` - Main application logs
- `logs/rag_server.log` - RAG pipeline logs
- Console output for real-time debugging

## Free Tier Usage

This solution is designed to work within free tiers:

### Gemini Flash
- **LLM**: paid per token; Flash is the low-cost tier

### Edge TTS (Free)
- **Unlimited**: Text-to-speech generation
- **Cost**: $0

### Local Storage (Free)
- **Vector DB**: Chroma (local files)
- **Document Store**: Local file system
- **Cost**: $0

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is open source. Please check individual component licenses:
- Kotaemon: [MIT License](https://github.com/Cinnamon/kotaemon/blob/main/LICENSE)
- Other dependencies as per their respective licenses

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs for error details
3. Test with the provided test suite
4. Create an issue with detailed information

---

**🚀 Ready to get started? Run `./run.sh` and visit http://localhost:8000/admin**