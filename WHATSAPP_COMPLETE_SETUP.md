# Complete WhatsApp Setup Guide - Step by Step

This guide walks you through the complete setup process for WhatsApp integration with your RAG system.

## 🎯 **Quick Answer: Where Do Users Send Messages?**

**Users send WhatsApp messages to**: `+1 415 523 8886`

This is **Twilio's WhatsApp Sandbox number**, not your personal number.

## 📋 **Prerequisites**

1. ✅ **ngrok installed**: `brew install ngrok`
2. ✅ **Twilio account**: Free signup at https://www.twilio.com/try-twilio
3. ✅ **RAG server running**: With documents uploaded

## 🚀 **Complete Setup Process**

### **Step 1: Install ngrok (if not already installed)**

```bash
# Install ngrok
brew install ngrok

# Test it works
ngrok version
```

### **Step 2: Get Twilio Credentials**

1. **Sign up for Twilio**:
   - Go to: https://www.twilio.com/try-twilio
   - Sign up with email (no credit card required)
   - Verify your phone number

2. **Get Your Credentials**:
   - After login, you'll see the **Console Dashboard**
   - Copy **Account SID** (starts with `AC`)
   - Click "Show" next to **Auth Token** and copy it

### **Step 3: Configure Environment**

Edit your `.env` file:

```bash
nano .env
```

Add your Twilio credentials:

```bash
# Required - Your Groq API key
GROQ_API_KEY=gsk_your_actual_groq_key_here

# Required for WhatsApp - Your Twilio credentials
TWILIO_ACCOUNT_SID=AC1234567890abcdef1234567890abcdef
TWILIO_AUTH_TOKEN=your_32_character_auth_token_here
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
PUBLIC_BASE_URL=https://will-be-set-automatically.ngrok.io

# Application settings
ENVIRONMENT=development
DEBUG=true
```

### **Step 4: Start the Server**

```bash
# Start with WhatsApp support (includes ngrok)
./run_simple.sh
```

**You should see output like:**

```
🚀 Starting Simple RAG Server...
📊 Admin UI: http://localhost:8000/admin
🔗 API Docs: http://localhost:8000/docs  
💚 Health Check: http://localhost:8000/health
🗄️ Storage: File-based (no database required)
🌐 Public URL: https://abc123.ngrok.io
📱 WhatsApp Webhook: https://abc123.ngrok.io/webhook
📋 WhatsApp Setup:
   1. Configure webhook in Twilio Console
   2. Set webhook URL to: https://abc123.ngrok.io/webhook
   3. Join sandbox: send 'join [code]' to +1 415 523 8886
================================================================================
```

**Important**: Copy the webhook URL from the output!

### **Step 5: Configure Twilio Webhook**

1. **Go to Twilio Console WhatsApp Sandbox**:
   - Navigate to: **Messaging** → **Try it out** → **Send a WhatsApp message**
   - Or direct link: https://console.twilio.com/us1/develop/sms/try-it-out/whatsapp-learn

2. **You'll see a page like this:**
   ```
   ┌─────────────────────────────────────────────────────┐
   │ Try WhatsApp                                        │
   │                                                     │
   │ Sandbox participants send messages to:              │
   │ +1 415 523 8886                                    │
   │                                                     │
   │ To join sandbox, participants need to send:         │
   │ join happy-elephant                                 │
   │                                                     │
   │ Sandbox Configuration                               │
   │ When a message comes in:                            │
   │ [https://abc123.ngrok.io/webhook    ] [POST ▼]    │
   │                                                     │
   │ [ Save Configuration ]                              │
   └─────────────────────────────────────────────────────┘
   ```

3. **Configure the Webhook**:
   - In the **"When a message comes in"** field, paste: `https://abc123.ngrok.io/webhook`
   - Make sure the dropdown is set to **"POST"**
   - Click **"Save Configuration"**
   - You should see: ✅ **"Configuration saved successfully"**

4. **Note Your Join Code**:
   - Write down the join code (e.g., `join happy-elephant`)
   - This is what users need to send first

### **Step 6: Test the Integration**

#### **6.1 Join the Sandbox (You)**

From your phone:
1. Open WhatsApp
2. Start a new chat with: `+1 415 523 8886`
3. Send: `join happy-elephant` (use your actual join code)
4. Wait for confirmation from Twilio

#### **6.2 Test Basic Functionality**

```
You: "hello"
Bot: [Should respond - if not, check webhook configuration]

You: "What is palliative care?"
Bot: [Should provide detailed response from your uploaded documents]
```

#### **6.3 Test Voice Messages**

```
You: [Send voice message in Hindi saying "दर्द का इलाज कैसे करें?"]
Bot: "🎯 Understood (Hindi): दर्द का इलाज कैसे करें?"
Bot: [Detailed response about pain treatment]
Bot: [Audio response in Hindi]
```

#### **6.4 Test Language Commands**

```
You: "/lang hi"
Bot: "✅ Language set to Hindi (hi)"
Bot: "अब मैं आपको हिंदी में जवाब दूंगा।"

You: "/lang bn"
Bot: "✅ Language set to Bengali (bn)"
Bot: "এখন আমি আপনাকে বাংলায় উত্তর দেব।"
```

## 👥 **For Other Users**

### **Adding More Users to Your RAG System**

Other people who want to use your RAG system need to:

1. **Send WhatsApp message to**: `+1 415 523 8886`
2. **First message must be**: `join happy-elephant` (your join code)
3. **Wait for confirmation** from Twilio
4. **Start asking questions** about your documents

### **User Instructions Template**

Give this to people who want to use your system:

```
📱 How to Access the Medical RAG System

1. Open WhatsApp on your phone
2. Start a new chat with: +1 415 523 8886
3. Send this exact message: join happy-elephant
4. Wait for confirmation message from Twilio
5. Now you can ask medical questions!

Examples:
• "What is palliative care?"
• "How to manage cancer pain?"
• Send voice messages in Hindi/Bengali/Tamil/Gujarati
• Use "/lang hi" to switch to Hindi responses

Note: This is NOT a personal phone number - it's a special 
WhatsApp service number managed by Twilio.
```

## 🔧 **Troubleshooting**

### **Problem: No ngrok URL printed**

**Check:**
```bash
# 1. Is ngrok installed?
ngrok version

# 2. Test ngrok manually (then open http://127.0.0.1:4040)
ngrok http 8000

# 3. Start server and check for errors
./run_simple.sh
```

**If ngrok is not installed:**
```bash
brew install ngrok
```

### **Problem: Webhook verification fails**

**Solutions:**
```bash
# 1. Check if webhook URL is accessible
curl https://your-ngrok-url.ngrok.io/webhook

# 2. Verify ngrok is running
curl http://127.0.0.1:4040/api/tunnels

# 3. Restart server
./run_simple.sh
```

### **Problem: Bot doesn't respond**

**Check:**
1. ✅ Webhook URL is correct in Twilio console
2. ✅ Join code was sent and confirmed
3. ✅ Server is running and showing ngrok URL
4. ✅ TWILIO_ACCOUNT_SID and AUTH_TOKEN are correct in .env

### **Problem: Voice messages don't work**

**Check:**
1. ✅ GROQ_API_KEY is set (for speech-to-text)
2. ✅ Check server logs for errors
3. ✅ Try text messages first to isolate the issue

## 🎯 **Complete Working Example**

Here's what a complete working setup looks like:

### **.env file:**
```bash
GROQ_API_KEY=gsk_abc123your_real_groq_key_here
TWILIO_ACCOUNT_SID=AC1234567890abcdef1234567890abcdef
TWILIO_AUTH_TOKEN=your_32_character_auth_token_here
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
PUBLIC_BASE_URL=https://abc123.ngrok.io
ENVIRONMENT=development
DEBUG=true
```

### **Server startup output:**
```
🌐 Public URL: https://abc123.ngrok.io
📱 WhatsApp Webhook: https://abc123.ngrok.io/webhook
```

### **Twilio webhook configuration:**
```
When a message comes in: https://abc123.ngrok.io/webhook [POST]
```

### **User interaction:**
```
User → +1 415 523 8886: "join happy-elephant"
Twilio → User: "You are now connected to the sandbox..."

User → +1 415 523 8886: "What is palliative care?"
RAG Bot → User: "Palliative care is specialized medical care..."
```

## 🏁 **Success Checklist**

- [ ] ✅ ngrok installed and working
- [ ] ✅ Twilio account created and credentials copied
- [ ] ✅ .env file configured with real credentials
- [ ] ✅ Server starts and shows ngrok URL
- [ ] ✅ Webhook configured in Twilio console
- [ ] ✅ Joined sandbox successfully
- [ ] ✅ Bot responds to test messages
- [ ] ✅ Voice messages work (if needed)
- [ ] ✅ Language switching works (if needed)

Once all items are checked, your WhatsApp RAG system is fully operational! 🎉

## 📞 **Support**

If you're still having issues:

1. **Run diagnostics**: `ngrok version`, then `curl http://127.0.0.1:4040/api/tunnels` while the server runs
2. **Check server health**: Visit `http://localhost:8000/health`
3. **Verify webhook**: Test the webhook URL directly
4. **Check logs**: Look at the terminal output for error messages

The system is designed to be robust and provide clear error messages to help you troubleshoot any issues.