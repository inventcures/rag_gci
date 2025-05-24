# How to Setup and Run the RAG Server with WhatsApp Bot

This guide provides step-by-step instructions to setup and run the complete RAG pipeline with WhatsApp bot integration on your local machine (macOS).

## 📋 Prerequisites

### System Requirements
- **macOS** (optimized for M3 Pro, but works on Intel Macs too)
- **Python 3.8+** (check with `python3 --version`)
- **8GB+ RAM** (for embedding models and vector storage)
- **Internet connection** (for API calls and dependencies)

### Required Accounts (Free)
1. **Groq Account** - For LLM and STT
   - Sign up at: https://console.groq.com/
   - Get free API key (no credit card required)
   - Free tier: 14,400 tokens/day + 100 min audio transcription

2. **Twilio Account** (Optional - for WhatsApp)
   - Only needed if you want WhatsApp integration
   - Sign up at: https://www.twilio.com/try-twilio
   - Free tier: $15.50 credit + WhatsApp sandbox

## 🚀 Setup Instructions

### Step 1: Environment Setup

```bash
# 1. Navigate to the project directory
cd /Users/tp53/Documents/tp53_AA/llms4palliative_gci/demo_feb2025/rag_gci

# 2. Run the automated setup
python3 setup.py
```

The setup script will:
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Create necessary directories
- ✅ Generate `.env` template
- ✅ Configure Kotaemon settings

### Step 2: Configure API Keys

Edit the `.env` file that was created:

```bash
# Open .env file in your preferred editor
nano .env
# or
code .env
```

Add your API keys:

```bash
# Required - Get from https://console.groq.com/
GROQ_API_KEY=gsk_your_actual_groq_api_key_here

# Optional - Only needed for WhatsApp integration via Twilio
TWILIO_ACCOUNT_SID=your_twilio_account_sid_here
TWILIO_AUTH_TOKEN=your_twilio_auth_token_here
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
PUBLIC_BASE_URL=https://your-ngrok-url.ngrok.io

# Database (uses SQLite by default)
DATABASE_URL=sqlite:///./data/rag_server.db

# Environment
ENVIRONMENT=development
DEBUG=true
```

**🔑 How to get Groq API Key:**
1. Go to https://console.groq.com/
2. Sign up with Google/GitHub (free)
3. Navigate to "API Keys" section
4. Click "Create API Key"
5. Copy the key and paste in `.env` file

**🔑 How to get Twilio Credentials:**
1. Go to https://www.twilio.com/try-twilio
2. Sign up for free account (no credit card needed)
3. Go to Console Dashboard
4. Copy Account SID and Auth Token
5. Go to Messaging → Try WhatsApp
6. Note the sandbox number and join code
7. See `twilio_setup_guide.md` for detailed instructions

### Step 3: Start the Server

#### Option A: Using the Run Script (Recommended)

```bash
# Make script executable (if not already)
chmod +x run.sh

# Start the server with ngrok (for external access)
./run.sh

# Or start without ngrok (local only)
./run.sh --no-ngrok
```

#### Option B: Direct Python Execution

```bash
# Activate virtual environment
source venv/bin/activate

# Start the server
python main.py
```

### Step 4: Verify Installation

After starting, you should see:

```
🚀 RAG SERVER WITH WHATSAPP BOT
==================================================
📊 Admin UI: http://localhost:8000/admin
🔗 API Docs: http://localhost:8000/docs
💚 Health Check: http://localhost:8000/health
📱 WhatsApp Webhook: http://localhost:8000/webhook
🌐 Public URL: https://abc123.ngrok.io
📱 Public Webhook: https://abc123.ngrok.io/webhook
```

## 📄 Adding Documents to the RAG System

### Step 1: Access Admin UI

1. Open your browser
2. Go to: `http://localhost:8000/admin`
3. You'll see the admin interface with multiple tabs

### Step 2: Upload Documents

1. Click on **"📁 Upload Documents"** tab
2. Click **"Select files to upload"**
3. Choose your files (PDF, DOCX, TXT, HTML, XLSX)
4. Optionally add metadata in JSON format:
   ```json
   {
     "category": "palliative_care",
     "topic": "pain_management",
     "language": "en"
   }
   ```
5. Click **"Upload & Index"**
6. Wait for processing to complete

### Step 3: Test the System

1. Go to **"💬 Test Queries"** tab
2. Enter test questions like:
   - "What is palliative care?"
   - "How to manage pain in cancer patients?"
   - "What are the signs of approaching death?"
3. Click **"Submit Query"**
4. Review the response and sources

## 📱 WhatsApp Bot Setup and Usage

### Quick Testing (Without Real WhatsApp)

You can test the pipeline using the API directly:

```bash
# Test text query
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "query=What is palliative care?&language=hi"

# Test file upload
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@your_document.pdf" \
  -F "metadata={\"category\": \"test\"}"
```

### WhatsApp Integration Setup (via Twilio)

For detailed Twilio setup instructions, see: **`twilio_setup_guide.md`**

#### Quick Setup Summary:

1. **Create Twilio Account**:
   - Go to https://www.twilio.com/try-twilio
   - Sign up for free (no credit card required)
   - Get Account SID and Auth Token from console

2. **Setup WhatsApp Sandbox**:
   - Go to Messaging → Try WhatsApp  
   - Note sandbox number: `+1 415 523 8886`
   - Note join code: `join [your-code]`

3. **Configure Webhook**:
   - Set webhook URL to: `https://your-ngrok-url.ngrok.io/webhook`
   - Set method to: `POST`

4. **Update .env File**:
   ```bash
   TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   TWILIO_AUTH_TOKEN=your_auth_token_here
   TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
   PUBLIC_BASE_URL=https://your-ngrok-url.ngrok.io
   ```

5. **Restart Server**:
   ```bash
   ./run.sh
   ```

#### Test Integration:

1. **Join Sandbox**:
   - Send WhatsApp message to: `+1 415 523 8886`
   - Send: `join [your-code]`

2. **Test Bot**:
   - Send: "What is palliative care?"
   - Send voice message in any language

## 💬 How Users Interact with WhatsApp Bot

### Text Message Interaction

**User sends**: "What is palliative care?"

**Bot responds**: 
```
Palliative care is specialized medical care focused on providing relief from pain and other symptoms of serious illness. The goal is to improve quality of life for both patients and families.

[Sources: document1.pdf, page 3]
```

### Voice Message Interaction

**User**: *Sends voice message in Hindi saying "दर्द का इलाज कैसे करें?"*

**Bot responds**:
1. **Text**: "🎯 Understood (Hindi): दर्द का इलाज कैसे करें?"
2. **Text Answer**: "दर्द का इलाज कई तरीकों से किया जा सकता है..."
3. **Audio Response**: *Sends audio file with Hindi TTS*

### Language Selection

**User sends**: `/lang hi`

**Bot responds**: 
```
✅ Language set to Hindi (hi)
अब मैं आपको हिंदी में जवाब दूंगा।
```

**Available language commands**:
- `/lang hi` - Hindi (हिंदी)
- `/lang bn` - Bengali (বাংলা) 
- `/lang ta` - Tamil (தமிழ்)
- `/lang gu` - Gujarati (ગુજરાતી)
- `/lang en` - English

### Typical User Journey

```
1. User: *Sends voice message in Tamil*
   Bot: 🎯 Understood (Tamil): [transcribed text]

2. Bot: *Sends text answer + Tamil audio response*

3. User: "/lang hi" 
   Bot: ✅ Language set to Hindi

4. User: "What about pain management?"
   Bot: *Responds in Hindi text + Hindi audio*

5. User: *Sends another voice message*
   Bot: *Automatically responds in Hindi (user's preference)*
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. "GROQ_API_KEY not configured"

**Problem**: API key missing or invalid

**Solution**:
```bash
# Check if .env file exists
ls -la .env

# Edit and add your key
nano .env

# Restart server
./run.sh
```

#### 2. "kotaemon-main directory not found"

**Problem**: Missing Kotaemon codebase

**Solution**:
```bash
# Check if directory exists
ls -la kotaemon-main/

# If missing, ensure you have the complete project structure
```

#### 3. "ngrok not found"

**Problem**: ngrok not installed

**Solution**:
```bash
# Install ngrok on macOS
brew install ngrok

# Or run without ngrok
./run.sh --no-ngrok
```

#### 4. WhatsApp webhook verification fails

**Problem**: Webhook URL or token mismatch

**Solution**:
1. Check ngrok is running: `curl http://127.0.0.1:4040/api/tunnels`
2. Verify webhook URL in Meta console matches ngrok URL
3. Check WHATSAPP_VERIFY_TOKEN in `.env` matches Meta console

#### 5. Audio processing fails

**Problem**: TTS/STT not working

**Solution**:
```bash
# Install audio dependencies
pip install edge-tts whisper-openai

# Check Groq API key for STT
echo $GROQ_API_KEY

# Test TTS manually
edge-tts --voice hi-IN-SwaraNeural --text "Test" --write-media test.mp3
```

### Performance Optimization

#### For M3 Pro Macs

```bash
# Set environment variables for optimal performance
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Start server
./run.sh
```

#### Memory Management

```bash
# Monitor memory usage
top -pid $(pgrep -f "python main.py")

# Reduce batch size if needed (edit config.yaml)
embedding:
  batch_size: 16  # Reduce from 32 if low memory
```

## 🧪 Testing Your Setup

### Automated Tests

```bash
# Run comprehensive test suite
python test_pipeline.py
```

### Manual Testing Checklist

- [ ] ✅ Server starts without errors
- [ ] ✅ Admin UI loads at localhost:8000/admin
- [ ] ✅ Document upload works
- [ ] ✅ Query testing returns answers
- [ ] ✅ Health check returns "healthy"
- [ ] ✅ ngrok tunnel is active (if enabled)
- [ ] ✅ Webhook verification passes (if WhatsApp configured)

## 📊 Monitoring and Logs

### Log Files

```bash
# Main application logs
tail -f logs/main.log

# RAG pipeline logs  
tail -f logs/rag_server.log

# Real-time console output
# Visible when running ./run.sh
```

### Health Monitoring

```bash
# Check system health
curl http://localhost:8000/health

# Check ngrok status
curl http://127.0.0.1:4040/api/tunnels

# Monitor resource usage
htop
```

## 🎯 Usage Tips

### Best Practices

1. **Document Organization**:
   - Use consistent metadata tags
   - Organize files by topic/category
   - Keep file names descriptive

2. **Query Optimization**:
   - Start with simple questions
   - Use specific medical terminology
   - Reference document context when possible

3. **Language Usage**:
   - Set user language preferences early
   - Use voice messages for natural interaction
   - Test different accents and speech patterns

4. **Performance**:
   - Upload documents in batches
   - Monitor API rate limits
   - Clean up cache files regularly

### Sample Documents to Test

Create test documents with content like:

**palliative_care_basics.md**:
```markdown
# Palliative Care Fundamentals

## Definition
Palliative care is specialized medical care for people living with serious illness...

## Core Principles
1. Pain and symptom management
2. Emotional support
3. Spiritual care
4. Family involvement
```

**pain_management.pdf**: Include content about different pain management strategies, medications, and non-pharmacological approaches.

## 🆘 Getting Help

### If Something Goes Wrong

1. **Check Logs**: Always start with log files
2. **Test Components**: Use test script to isolate issues
3. **Verify Config**: Double-check API keys and settings
4. **Restart Clean**: Stop server, restart, check again

### Common Questions

**Q: Can I use this without WhatsApp?**  
A: Yes! Use the admin UI for document management and testing.

**Q: What file types are supported?**  
A: PDF, DOCX, TXT, HTML, XLSX, MD files.

**Q: How many documents can I upload?**  
A: Limited by storage space and processing time, typically hundreds of documents.

**Q: Can I use other languages?**  
A: Currently supports Hindi, Bengali, Tamil, Gujarati, and English. Code can be extended for other languages.

---

🚀 **You're all set!** Start with `./run.sh` and begin uploading your documents to create your RAG-powered WhatsApp bot.