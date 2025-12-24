# Palli Sahayak Voice AI Agent Helpline - Deployment Guide

This guide covers deploying the Bolna.ai integration for the Palli Sahayak palliative care voice helpline.

## Prerequisites

Before deploying, ensure you have:

- [ ] Python 3.10+ installed
- [ ] ngrok installed (`brew install ngrok` on macOS)
- [ ] RAG server with indexed palliative care documents
- [ ] Credit card for Bolna.ai (pay-as-you-go pricing)

## Step 1: Get API Keys

### 1.1 Bolna.ai Account

1. Sign up at https://app.bolna.ai
2. Go to **Settings → API Keys**
3. Create a new API key
4. Copy the key for `.env`

### 1.2 OpenAI API Key (for Bolna LLM)

1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Add billing information if needed

### 1.3 ElevenLabs API Key (for Bolna TTS)

1. Go to https://elevenlabs.io
2. Sign up for an account
3. Get API key from profile settings

### 1.4 Twilio Account (for phone numbers)

1. Go to https://console.twilio.com
2. Get Account SID and Auth Token
3. Purchase a phone number with voice capability

## Step 2: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit with your keys
nano .env
```

Required variables:
```bash
# Core
GROQ_API_KEY=your_groq_key

# Bolna
BOLNA_API_KEY=your_bolna_key
BOLNA_WEBHOOK_SECRET=generate_a_secret_string

# For Bolna's LLM/TTS
OPENAI_API_KEY=your_openai_key
ELEVENLABS_API_KEY=your_elevenlabs_key
```

## Step 3: Start the Server

```bash
# Install dependencies
pip install -r requirements.txt

# Start with ngrok for external access
./run_simple.sh

# Or without ngrok (local only)
python simple_rag_server.py --no-ngrok
```

Note the public ngrok URL (e.g., `https://abc123.ngrok.io`)

## Step 4: Create Bolna Agent

### 4.1 Using Python Script

```python
import asyncio
from bolna_integration import BolnaClient, get_palli_sahayak_agent_config

async def create_agent():
    client = BolnaClient()

    # Get your ngrok URL
    server_url = "https://your-ngrok-url.ngrok.io"

    # Generate config
    config = get_palli_sahayak_agent_config(
        server_url=server_url,
        language="hi"  # Primary language
    )

    # Create agent
    result = await client.create_agent(config)

    if result.success:
        print(f"✅ Agent created: {result.agent_id}")
        print(f"   Save this ID to .env as BOLNA_AGENT_ID")
    else:
        print(f"❌ Failed: {result.error}")

asyncio.run(create_agent())
```

### 4.2 Using Bolna Dashboard

1. Go to https://app.bolna.ai/agents
2. Click **Create Agent**
3. Use these settings:

| Setting | Value |
|---------|-------|
| Name | Palli Sahayak |
| Type | Free Flowing |
| ASR | Deepgram (nova-2) |
| LLM | OpenAI (gpt-4o-mini) |
| TTS | ElevenLabs (eleven_multilingual_v2) |

4. Add the system prompt from `bolna_integration/config.py`
5. Add custom function:

```json
{
  "name": "query_rag_knowledge_base",
  "description": "Query palliative care knowledge base",
  "parameters": {
    "type": "object",
    "properties": {
      "user_query": {"type": "string"},
      "user_language": {"type": "string", "enum": ["en", "hi", "mr", "ta"]}
    },
    "required": ["user_query", "user_language"]
  },
  "value": {
    "method": "POST",
    "url": "https://YOUR_NGROK_URL/api/bolna/query"
  }
}
```

## Step 5: Configure Webhook

1. In Bolna dashboard, go to agent settings
2. Set webhook URL: `https://YOUR_NGROK_URL/api/bolna/webhook`
3. Select events:
   - `call_started`
   - `call_ended`
   - `extraction_completed`

## Step 6: Set Up Phone Number

### Option A: Use Bolna's Phone Numbers

1. Go to Bolna dashboard → Phone Numbers
2. Purchase a number
3. Link to your agent

### Option B: Bring Your Own Twilio Number

1. In Twilio console, configure your number
2. Set Voice webhook to Bolna's SIP endpoint
3. Follow Bolna's Twilio integration guide

## Step 7: Test the Integration

### 7.1 Run Test Suite

```bash
# Full test suite
python test_bolna_integration.py

# Quick smoke test
python test_bolna_integration.py --quick

# With live API tests
python test_bolna_integration.py --live
```

### 7.2 Test API Endpoints

```bash
# Test RAG query endpoint
curl -X POST https://YOUR_NGROK_URL/api/bolna/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is palliative care?", "language": "en"}'

# Check call stats
curl https://YOUR_NGROK_URL/api/bolna/stats
```

### 7.3 Make a Test Call

1. Call the phone number
2. Speak in Hindi or English
3. Ask about palliative care
4. Verify response uses RAG knowledge base

## Step 8: Monitor and Debug

### View Logs

```bash
# Server logs
tail -f logs/rag_server.log

# Watch for Bolna events
grep "Bolna" logs/rag_server.log
```

### Check Call History

```bash
curl https://YOUR_NGROK_URL/api/bolna/calls
```

### Health Check

```bash
curl https://YOUR_NGROK_URL/health
curl https://YOUR_NGROK_URL/api/bolna/stats
```

## Production Deployment

### For Production Use

1. **Use a persistent URL** instead of ngrok:
   - Deploy to cloud (AWS, GCP, Azure)
   - Use a custom domain with SSL

2. **Enable webhook signature verification**:
   ```bash
   BOLNA_WEBHOOK_SECRET=your_secure_secret
   ```

3. **Set up monitoring**:
   - Add logging to external service
   - Set up alerts for failed calls
   - Monitor RAG query latency

4. **Configure auto-scaling** if expecting high call volume

### Environment Variables for Production

```bash
# Required
BOLNA_API_KEY=prod_key
BOLNA_AGENT_ID=prod_agent_id
BOLNA_WEBHOOK_SECRET=secure_secret_here
PUBLIC_BASE_URL=https://your-production-domain.com

# Recommended
DEBUG=false
LOG_LEVEL=WARNING
```

## Troubleshooting

### "BOLNA_API_KEY not configured"

- Check `.env` file has the key
- Restart the server after adding

### Webhook not receiving events

- Verify webhook URL is accessible from internet
- Check ngrok is running
- Verify webhook URL in Bolna dashboard

### RAG responses not being used

- Check `/api/bolna/query` endpoint is working
- Verify custom function URL is correct
- Check server logs for errors

### Call quality issues

- Check internet connection
- Try different TTS voice
- Adjust LLM temperature

## Cost Estimation

| Component | Estimated Cost |
|-----------|----------------|
| Bolna Platform | $0.05-0.10 per minute |
| OpenAI (gpt-4o-mini) | ~$0.001 per call |
| ElevenLabs TTS | ~$0.01 per call |
| Twilio Phone Number | $1-2 per month |
| **Total per call** | **~$0.10-0.15** |

## Support

- **Bolna Docs**: https://www.bolna.ai/docs
- **Project Wiki**: https://deepwiki.com/inventcures/rag_gci
- **Issues**: https://github.com/inventcures/rag_gci/issues
