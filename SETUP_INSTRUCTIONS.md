# Complete Setup Instructions

This guide sets up the study server, `simple_rag_server.py`, with its API keys and configuration. It is the only server: the Kotaemon "full" version was removed because it had no Dose Boundary and no study release.

## 🎯 **The Server**

### 🚀 **Study server**
- **File**: `simple_rag_server.py`
- **Run**: `./start.sh`, `./run_simple.sh`, `./run.sh` or `python simple_rag_server.py`
- **Database**: None (file-based storage)
- **Setup Time**: 2-5 minutes
- **Dependencies**: ~15 packages

---

## ⚡ **Quick Start (Recommended)**

### Super Simple Start:
```bash
# Navigate to project directory
cd rag_gci

# One-command startup
./start.sh

# Or explicitly choose simple version
./start.sh --simple

# With validation
./start.sh --simple --validate
```

**That's it!** The script will:
- ✅ Auto-create virtual environment
- ✅ Install only missing dependencies (smart caching)
- ✅ Create .env template
- ✅ Start the server

---

## 🚀 **Manual Simple Version Setup**

### Step 1: Smart Dependency Installation

```bash
# Navigate to project directory
cd rag_gci

# Smart install (only installs missing packages)
./run_simple.sh

# Force reinstall all (if needed)
./run_simple.sh --force-install
```

### Step 2: Get API Keys

#### 🔑 **Groq API Key** (Required)

1. **Go to Groq Console**: https://console.groq.com/
2. **Sign up for free** (no credit card required)
3. **Create API Key**:
   - Click on your profile (top right)
   - Select "API Keys"
   - Click "Create API Key"
   - Give it a name: "RAG Server"
   - Copy the key (starts with `gsk_`)

#### 🔑 **Twilio Credentials** (Optional - for WhatsApp)

1. **Go to Twilio**: https://www.twilio.com/try-twilio
2. **Sign up for free** (no credit card required)
3. **Get Credentials**:
   - Go to Console Dashboard
   - Copy **Account SID** (starts with `AC`)
   - Copy **Auth Token** (click to reveal)
4. **Setup WhatsApp Sandbox**:
   - Go to "Messaging" → "Try it out" → "Send a WhatsApp message"
   - Note the sandbox number: `+1 415 523 8886`
   - Note your join code: `join [your-code]`

### Step 3: Configure API Keys

The script auto-creates a `.env` template. Just edit it with your keys:

```bash
# Edit the auto-generated .env file
nano .env

# Or use any text editor
code .env
open -a TextEdit .env
```

**Just replace these lines with your actual keys:**

```bash
# Replace this line:
GROQ_API_KEY=gsk_your_actual_groq_api_key_here_it_starts_with_gsk

# With your actual key:
GROQ_API_KEY=gsk_abc123your_real_groq_key_here

# For WhatsApp (optional):
TWILIO_ACCOUNT_SID=AC123your_real_twilio_sid_here
TWILIO_AUTH_TOKEN=your_real_32_char_auth_token_here
```

**💡 Tip:** The template shows exactly what format your keys should be in!

### Step 4: Start the Server

```bash
# Smart startup (checks dependencies, only installs missing ones)
./run_simple.sh

# Or use the universal starter
./start.sh --simple
```

**🚀 On subsequent runs**, the script will be much faster:
```
✅ All dependencies are already installed  # <- No reinstall!
✅ .env file exists
🌟 Starting server...
```

### Step 5: Verify Setup

**First run output:**
```
🚀 Starting Simple RAG Server (No Database)
==================================================
✅ Python3 found
⚙️ Installing missing packages: fastapi gradio chromadb
✅ All packages installed successfully
📝 Created .env template. Please update with your actual API keys:
   1. Get Groq API key from: https://console.groq.com/
   2. Replace 'gsk_your_actual_groq_api_key_here_it_starts_with_gsk' with your key
🌟 Starting server...
```

**Subsequent runs:**
```
✅ All dependencies are already installed
✅ .env file exists
🚀 Starting Simple RAG Server...
```

**Server ready:**
```
📊 Admin UI: http://localhost:8000/admin
🔗 API Docs: http://localhost:8000/docs
💚 Health Check: http://localhost:8000/health
🏥 Database Health: Auto-monitoring & rebuild enabled
🗄️ Storage: File-based (no database required)
```

---

## 📱 **WhatsApp Bot Setup**

### Step 1: Configure Twilio Webhook

1. **Get your ngrok URL** (from server startup):
   ```
   🌐 Public URL: https://abc123.ngrok.io
   📱 Public Webhook: https://abc123.ngrok.io/webhook
   ```

2. **Set Webhook in Twilio Console**:
   - Go to WhatsApp Sandbox settings
   - Set webhook URL: `https://your-ngrok-url.ngrok.io/webhook`
   - Set HTTP method: `POST`
   - Save configuration

### Step 2: Test WhatsApp Integration

1. **Join the Sandbox**:
   - From your phone, send WhatsApp message to: `+1 415 523 8886`
   - Send: `join [your-code]` (replace with your actual join code)
   - Wait for confirmation

2. **Test the Bot**:
   ```
   You: "What is palliative care?"
   Bot: [Response with answer]
   
   You: /lang hi
   Bot: "✅ Language set to Hindi (hi)"
   
   You: [Send voice message in Hindi]
   Bot: "🎯 Understood (Hindi): [transcription]"
   Bot: [Text response + Hindi audio]
   ```

---

## 🔧 **Troubleshooting**

### Performance Issues:

#### 🐌 **"Why does first run take so long?"**
**Answer**: Only the first run installs dependencies. Subsequent runs are much faster!

```bash
# First run: ~2-5 minutes (downloads packages)
./run_simple.sh

# Later runs: ~10-30 seconds (skips installation)
./run_simple.sh
```

#### 🔄 **"How to force reinstall packages?"**
```bash
# Force reinstall all dependencies
./run_simple.sh --force-install

# Or delete virtual environment and start fresh
rm -rf venv
./run_simple.sh
```

### Configuration Issues:

#### ❌ "GROQ_API_KEY not configured"
**Problem**: API key missing or incorrect

**Solution**:
```bash
# Check your .env file
cat .env | grep GROQ

# Should show:
GROQ_API_KEY=gsk_actual_key_here

# If missing or wrong, edit:
nano .env
# Replace with: GROQ_API_KEY=gsk_your_actual_key

# Restart server
./run_simple.sh
```

#### ❌ "Twilio credentials not configured"
**Problem**: Twilio keys missing (only affects WhatsApp)

**Solution**:
```bash
# Check Twilio config
cat .env | grep TWILIO

# Should show:
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_token_here

# If missing, add them to .env
nano .env
```

#### ❌ "ngrok not found"
**Problem**: ngrok not installed

**Solution**:
```bash
# Install ngrok on macOS
brew install ngrok

# Or run without external access
./run_simple.sh --help
# Use local-only mode if you don't need WhatsApp
```

#### ❌ "Webhook verification failed"
**Problem**: Twilio can't reach your webhook

**Solution**:
1. Check ngrok is running: `curl http://127.0.0.1:4040/api/tunnels`
2. Update PUBLIC_BASE_URL in .env with ngrok URL
3. Update webhook URL in Twilio console
4. Restart server

#### ❌ "Permission denied" on run scripts
**Problem**: Scripts not executable

**Solution**:
```bash
chmod +x run_simple.sh
chmod +x run.sh
```

#### ⚠️ **Database Corruption Issues**
**Problem**: Queries return "could not find answer" after adding/removing documents

**Solution**:
```bash
# Check database health via Admin UI
# 1. Open http://localhost:8000/admin
# 2. Go to "🏥 Database Health" tab
# 3. Click "🔍 Check Health"
# 4. If corrupted, click "🔧 Auto Rebuild"

# Or rebuild with the study release's embedding model
python scripts/reindex_with_configured_model.py --force

# Manual rebuild if needed
# Admin UI → Database Health → ⚡ Force Rebuild
```

#### 🔧 **Auto-Rebuild Features**
**New**: Built-in corruption detection and recovery

**Features**:
- **Automatic detection**: Monitors database health during operations
- **Smart rebuilds**: Only rebuilds when corruption detected
- **Zero downtime**: Continues serving during rebuild process
- **Backup protection**: Creates backups before any rebuild operation
- **Progress tracking**: Real-time status in Admin UI

**Usage**:
1. Check health: Admin UI → Database Health → 🔍 Check Health
2. Auto-rebuild: Admin UI → Database Health → 🔧 Auto Rebuild  
3. Force rebuild: Admin UI → Database Health → ⚡ Force Rebuild

---

## ✅ **Verification Checklist**

### For the study server:
- [ ] Python 3.8+ installed
- [ ] `pip install -r requirements_simple.txt` completed
- [ ] `.env` file created with GROQ_API_KEY
- [ ] `./run_simple.sh` starts without errors
- [ ] Can access http://localhost:8000/admin
- [ ] Can upload a test document
- [ ] Can query and get responses
- [ ] "🏥 Database Health" tab shows healthy status

### For WhatsApp:
- [ ] Twilio account created
- [ ] WhatsApp sandbox configured
- [ ] Webhook URL set in Twilio
- [ ] ngrok tunnel active
- [ ] Can join sandbox with phone
- [ ] Bot responds to test messages

---

## 🎯 **Command Reference**

### Smart Startup (Recommended):
```bash
# Start the study server
./start.sh

# With validation
./start.sh --simple --validate

# Force reinstall dependencies  
./start.sh --simple --force-install

# Custom host/port
./start.sh --simple --host 127.0.0.1 --port 9000
```

### Direct Version Commands:
```bash
# Simple version (smart dependency management)
./run_simple.sh                    # Normal start
./run_simple.sh --force-install    # Force reinstall
./run_simple.sh --port 9000        # Custom port

# The same server, installing requirements.txt first
./run.sh                           # Normal start
./run.sh --host 127.0.0.1          # Custom host

# Validation
python validate_setup.py           # Check configuration
```

### Quick Setup:
```bash
# Fastest way to get started
./start.sh --simple

# If you want to test everything first
./start.sh --simple --validate
```

---

## 🆘 **Getting Help**

### If you're stuck:

1. **Check Logs**: Look at the terminal output for error messages
2. **Verify API Keys**: Make sure they're correctly copied to .env
3. **Test Components**: Start with simple text queries before trying WhatsApp
4. **Check Network**: Ensure your machine can reach external APIs

### Sample .env for Testing:
```bash
# Minimal working .env for testing
GROQ_API_KEY=gsk_your_actual_groq_key_here
ENVIRONMENT=development
DEBUG=true
```

**🚀 Ready to start? Run `./start.sh`.**