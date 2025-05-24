# Twilio WhatsApp Integration Setup Guide

This guide explains how to set up the WhatsApp bot using Twilio's free sandbox account, which is perfect for development and testing.

## 🆓 Why Twilio Free Tier?

### Benefits:
- ✅ **Completely Free**: No credit card required for sandbox
- ✅ **Easy Setup**: 5-minute configuration
- ✅ **No Business Verification**: Unlike Meta WhatsApp Business API
- ✅ **Instant Testing**: Start testing immediately
- ✅ **Voice Message Support**: Full audio message support

### Limitations:
- 📱 **Sandbox Only**: Limited to pre-verified phone numbers
- 🔢 **Limited Numbers**: Can add up to 5 phone numbers for testing
- 💬 **Prefix Required**: Users must send "join [sandbox-name]" first

## 🚀 Step-by-Step Setup

### Step 1: Create Twilio Account

1. **Sign up for Twilio**:
   - Go to: https://www.twilio.com/try-twilio
   - Click "Sign up for free"
   - Enter your email and create password
   - Verify your phone number

2. **Get Your Credentials**:
   - After signup, go to Twilio Console
   - Note down your **Account SID** and **Auth Token**
   - These are on the main dashboard

### Step 2: Setup WhatsApp Sandbox

1. **Navigate to WhatsApp Sandbox**:
   - In Twilio Console, go to "Messaging" → "Try it out" → "Send a WhatsApp message"
   - Or directly visit: https://console.twilio.com/us1/develop/sms/try-it-out/whatsapp-learn

2. **Get Sandbox Information**:
   - You'll see a sandbox number like: `+1 415 523 8886`
   - You'll see a join code like: `join yellow-tiger`
   - Note these down

3. **Configure Webhook**:
   - In the sandbox settings, set webhook URL
   - Use your ngrok URL: `https://your-ngrok-url.ngrok.io/webhook`

### Step 3: Configure Your Application

1. **Update .env File**:
   ```bash
   # Edit your .env file
   nano .env
   ```

   Add your Twilio credentials:
   ```bash
   # Required - Get from https://console.groq.com/
   GROQ_API_KEY=your_groq_api_key_here

   # Required - Get from Twilio Console
   TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   TWILIO_AUTH_TOKEN=your_auth_token_here
   TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
   PUBLIC_BASE_URL=https://your-ngrok-url.ngrok.io
   ```

2. **Start Your Server**:
   ```bash
   ./run.sh
   ```

3. **Note Your Webhook URL**:
   - When the server starts, you'll see:
   ```
   🌐 Public URL: https://abc123.ngrok.io
   📱 Public Webhook: https://abc123.ngrok.io/webhook
   ```

### Step 4: Configure Twilio Webhook

1. **Back in Twilio Console**:
   - Go to WhatsApp Sandbox settings
   - Set webhook URL to: `https://your-ngrok-url.ngrok.io/webhook`
   - Set HTTP method to: `POST`
   - Save configuration

### Step 5: Test Your Bot

1. **Join the Sandbox**:
   - From your phone, send a WhatsApp message to: `+1 415 523 8886`
   - Send message: `join yellow-tiger` (replace with your join code)
   - You should get a confirmation message

2. **Test Text Queries**:
   ```
   You: "What is palliative care?"
   Bot: [Response with answer and optional audio]
   ```

3. **Test Voice Messages**:
   - Send a voice message in any supported language
   - Bot will transcribe and respond with text + audio

4. **Test Language Commands**:
   ```
   You: "/lang hi"
   Bot: "✅ Language set to Hindi (hi)"
   Bot: "अब मैं आपको हिंदी में जवाब दूंगा।"
   ```

## 📱 User Experience with Twilio

### First Time Setup (One-time per user):

1. **User joins sandbox**:
   ```
   User sends: "join yellow-tiger"
   Twilio: "You are now connected to the sandbox..."
   ```

2. **User starts chatting**:
   ```
   User: "What is pain management?"
   Bot: [Detailed response with sources]
   Bot: [Audio response in user's preferred language]
   ```

### Ongoing Usage:

- **Text Messages**: Instant responses with optional audio
- **Voice Messages**: Automatic transcription → RAG query → text + audio response  
- **Language Switching**: `/lang hi`, `/lang bn`, `/lang ta`, `/lang gu`
- **Multilingual**: Automatic language detection from voice

## 🔧 Troubleshooting

### Common Issues:

#### "Webhook not responding"
**Problem**: Twilio can't reach your webhook

**Solution**:
1. Check if ngrok is running: `curl http://127.0.0.1:4040/api/tunnels`
2. Verify webhook URL in Twilio matches ngrok URL
3. Ensure your server is running on the correct port

#### "User not in sandbox"
**Problem**: User hasn't joined the sandbox

**Solution**:
1. User must send "join [code]" to sandbox number first
2. Check Twilio console for verified numbers
3. Add user's number to sandbox if needed

#### "Audio messages not working"
**Problem**: Voice messages not being processed

**Solution**:
1. Check Groq API key for STT
2. Verify PUBLIC_BASE_URL in .env points to ngrok URL
3. Check audio file permissions and storage

#### "No response from bot"
**Problem**: Bot not responding to messages

**Solution**:
1. Check server logs for errors
2. Verify RAG pipeline is initialized
3. Test with simple text query first

### Testing Commands:

```bash
# Test webhook locally
curl -X POST http://localhost:8000/webhook \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "From=whatsapp:+1234567890&To=whatsapp:+14155238886&Body=test"

# Check ngrok status
curl http://127.0.0.1:4040/api/tunnels

# Test audio serving
curl http://localhost:8000/media/test.mp3
```

## 📊 Free Tier Limits

### Twilio Free Tier:
- **Messages**: $15.50 free credit (≈ 500+ messages)
- **WhatsApp Sandbox**: Unlimited for testing
- **Phone Numbers**: 5 verified numbers for sandbox
- **Voice**: Included in free credit

### Groq Free Tier:
- **LLM**: 14,400 tokens/day
- **STT**: 100 minutes/day audio transcription
- **Cost**: $0

### Total Cost: **$0** for development and testing

## 🔄 Upgrading to Production

When ready for production:

1. **Upgrade Twilio Account**:
   - Add payment method
   - Request WhatsApp Business Profile
   - Get dedicated phone number

2. **Update Configuration**:
   - Change from sandbox to production number
   - Update webhook URLs
   - Remove join requirement

3. **Scale Infrastructure**:
   - Use proper hosting instead of ngrok
   - Add monitoring and logging
   - Implement user management

## 💡 Best Practices

### Development:
- Start with text messages for initial testing
- Test one language at a time initially
- Use sandbox with 2-3 test numbers first
- Monitor logs for debugging

### User Onboarding:
- Provide clear instructions for joining sandbox
- Test with different devices and languages
- Document common user questions
- Create user guide with examples

### Production Readiness:
- Test extensively in sandbox first
- Prepare user documentation
- Set up monitoring and alerts
- Plan for scaling and load testing

---

🚀 **Ready to start?** Follow the steps above and you'll have a working WhatsApp bot in under 10 minutes!