#!/usr/bin/env python3
"""
Palli Sahayak - Voice AI Demo with Actual TTS Audio
====================================================

Demonstrates:
1. Gemini Live API - Web-based voice conversations
2. Bolna.ai - Phone calls with custom voice agents
3. Retell.ai + Vobiz.ai - Indian PSTN (+91) with SIP-REFER warm handoff
4. WhatsApp Bot - Twilio sandbox integration

Generates ACTUAL audio files using Edge TTS for the demo.
"""

import os
import sys
import asyncio
import json
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    print(f"\n{Colors.HEADER}{'='*70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.END}")
    print(f"{Colors.HEADER}{'='*70}{Colors.END}\n")

def print_section(text):
    print(f"\n{Colors.CYAN}{Colors.BOLD}▶ {text}{Colors.END}")
    print(f"{Colors.CYAN}{'─'*60}{Colors.END}")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")

def print_audio(text):
    print(f"{Colors.GREEN}🔊 {text}{Colors.END}")

# ============================================================================
# AUDIO GENERATION USING EDGE TTS
# ============================================================================

async def generate_voice_audio(text, language, output_file):
    """Generate actual audio using Edge TTS"""
    try:
        import edge_tts
        
        voices = {
            "hi": "hi-IN-SwaraNeural",
            "en": "en-IN-NeerjaNeural",
            "gu": "gu-IN-DhwaniNeural",
            "bn": "bn-IN-TanishaaNeural",
            "ta": "ta-IN-PallaviNeural",
        }
        
        voice = voices.get(language, "en-IN-NeerjaNeural")
        
        print_info(f"Generating audio with voice: {voice}")
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)
        
        file_size = os.path.getsize(output_file)
        print_success(f"Audio generated: {output_file} ({file_size} bytes)")
        return True
        
    except Exception as e:
        print_warning(f"Audio generation failed: {e}")
        return False

# ============================================================================
# DEMO 1: GEMINI LIVE API - WEB VOICE
# ============================================================================

async def demo_gemini_live():
    """Demo Gemini Live API for web-based voice"""
    print_header("DEMO 1: GEMINI LIVE API - WEB VOICE")
    
    print_section("Session Initialization")
    
    session = {
        "session_id": "gemini-live-001",
        "provider": "Gemini Live API",
        "audio_format": "PCM 16-bit 16kHz (input) / 24kHz (output)",
        "language": "hi-IN",
        "max_duration": "15 minutes",
        "connection": "WebSocket (WSS)"
    }
    
    print(f"  Session ID: {session['session_id']}")
    print(f"  Provider: {session['provider']}")
    print(f"  Audio Format: {session['audio_format']}")
    print(f"  Language: {session['language']}")
    
    print_section("WebSocket Connection")
    
    websocket_log = """
    CONNECTING wss://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:streamGenerateContent?key=***
    
    ⬆️  SEND: {
      "setup": {
        "model": "models/gemini-2.0-flash-exp",
        "generation_config": {
          "response_modalities": ["AUDIO"],
          "speech_config": {
            "voice_config": {"prebuilt_voice_config": {"voice_name": "Aoede"}}
          }
        }
      }
    }
    
    ⬇️  RECEIVE: {"setupComplete": {}}
    
    ✅ WebSocket connection established
    """
    
    print(websocket_log)
    
    print_section("Patient Query (Voice Input)")
    
    patient_query = "माँ को दर्द है, क्या करूं?"  # "Mother has pain, what should I do?"
    print(f"  Patient (Hindi): {patient_query}")
    print(f"  Translated: \"Mother has pain, what should I do?\"")
    
    # Generate audio for patient query
    audio_file = "cache/demo_gemini_patient_query.mp3"
    await generate_voice_audio(
        "माँ को दर्द है, क्या करूं?", 
        "hi", 
        audio_file
    )
    print_audio(f"Patient audio: {audio_file}")
    
    print_section("RAG Context Injection")
    
    rag_context = """
    Querying ChromaDB vector store...
    
    Retrieved 3 relevant documents:
    1. WHO Cancer Pain Guidelines (relevance: 0.94)
    2. Max Healthcare Pain Management SOP (relevance: 0.91)
    3. Pallium India Home Care Protocol (relevance: 0.88)
    
    Context injected into Gemini session.
    """
    print(rag_context)
    
    print_section("Gemini Response (Voice Output)")
    
    response_text = """
    आपकी माँ को दर्द के लिए ये दवाई दें:
    
    मोर्फिन 5 से 10 मिलीग्राम,
    हर 4 घंटे में एक बार।
    
    अगर दर्द बहुत ज़्यादा हो,
    तो डॉक्टर से बात करें।
    """
    
    print(f"  AI Response (Hindi):")
    for line in response_text.strip().split('\n'):
        print(f"    {line}")
    
    # Generate audio for AI response
    audio_file = "cache/demo_gemini_ai_response.mp3"
    await generate_voice_audio(
        "आपकी माँ को दर्द के लिए मोर्फिन 5 से 10 मिलीग्राम, हर 4 घंटे में एक बार दें। अगर दर्द बहुत ज़्यादा हो, तो डॉक्टर से बात करें।",
        "hi",
        audio_file
    )
    print_audio(f"AI response audio: {audio_file}")
    
    print_success("Gemini Live demo completed")

# ============================================================================
# DEMO 2: BOLNA.AI - PHONE CALLS
# ============================================================================

async def demo_bolna():
    """Demo Bolna.ai for phone calls"""
    print_header("DEMO 2: BOLNA.AI - PHONE CALLS")
    
    print_section("Call Configuration")
    
    config = {
        "agent_name": "Palli Sahayak - Hindi",
        "phone_number": "+91-XXXX-NH-HELP",
        "language": "hi-IN",
        "asr_provider": "Deepgram (nova-2)",
        "llm_provider": "OpenAI (gpt-4o-mini)",
        "tts_provider": "ElevenLabs (eleven_multilingual_v2)",
        "telephony": "Twilio"
    }
    
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print_section("Initiating Outbound Call")
    
    call_log = """
    POST https://api.bolna.ai/call
    {
      "agent_id": "palli-sahayak-hi",
      "phone_number": "+919876543210",
      "patient_id": "PT-ONCO-2026-001",
      "context": {
        "name": "Mrs. Lakshmi Devi",
        "medication": "Ondansetron 8mg",
        "purpose": "Chemotherapy nausea prevention"
      }
    }
    
    ⬇️  RESPONSE: {
      "call_id": "call-bol-001",
      "status": "initiated",
      "webhook_url": "https://api.pallisahayak.io/webhook/bolna"
    }
    
    ✅ Call initiated successfully
    """
    print(call_log)
    
    print_section("Call Flow (Voice Conversation)")
    
    conversation = [
        ("AI", "नमस्ते लक्ष्मी जी, मैं पल्ली सहायक बोल रहा हूं।"),
        ("Patient", "नमस्ते..."),
        ("AI", "यह आपकी दवाई का समय है। कृपया ऑन्डेसेट्रॉन 8 मिलीग्राम लें।"),
        ("Patient", "ठीक है, मैं अभी ले लेती हूं।"),
        ("AI", "बहुत अच्छा। दवाई लेने के बाद फोन पर 1 दबाएं।"),
        ("System", "[DTMF tone: 1]"),
        ("AI", "धन्यवाद लक्ष्मी जी। अगली दवाई शाम 8 बजे है।"),
    ]
    
    for speaker, text in conversation:
        if speaker == "AI":
            print(f"  {Colors.CYAN}🤖 AI: {text}{Colors.END}")
            # Generate audio for AI lines
            audio_file = f"cache/demo_bolna_ai_{hash(text) % 1000}.mp3"
            await generate_voice_audio(text, "hi", audio_file)
            print_audio(f"      Audio: {audio_file}")
        elif speaker == "Patient":
            print(f"  {Colors.WARNING}👤 Patient: {text}{Colors.END}")
        else:
            print(f"  {Colors.GREEN}📞 {speaker}: {text}{Colors.END}")
    
    print_section("Call Summary")
    
    summary = {
        "call_id": "call-bol-001",
        "duration": "45 seconds",
        "patient_confirmed": True,
        "dtmf_input": "1",
        "adherence_logged": True,
        "caregiver_notified": True
    }
    
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print_success("Bolna.ai demo completed")

# ============================================================================
# DEMO 3: RETELL + VOBIZ.AI WITH SIP-REFER WARM HANDOFF
# ============================================================================

async def demo_retell_vobiz_handoff():
    """Demo Retell with Vobiz.ai and SIP-REFER warm handoff"""
    print_header("DEMO 3: RETELL + VOBIZ.AI WITH SIP-REFER WARM HANDOFF")
    
    print_section("Incoming Emergency Call")
    
    emergency = {
        "call_id": "ret-urg-001",
        "caller": "+91-98765-12345",
        "patient": "Mr. Ramesh Patel (PT-COPD-2026-042)",
        "condition": "Severe COPD - Breathlessness",
        "language": "Gujarati",
        "provider": "Vobiz.ai (Indian PSTN)"
    }
    
    for key, value in emergency.items():
        print(f"  {key}: {value}")
    
    print_section("Emergency Detection")
    
    # Patient query in Gujarati
    patient_query_gu = "મને શ્વાસ લેવામાં બહુ તકલીફ થઈ રહી છે... હું શ્વાસ લઈ નથી શકતો..."
    patient_query_en = "I am having a lot of difficulty breathing... I cannot breathe..."
    
    print(f"  Patient (Gujarati): {patient_query_gu}")
    print(f"  Translation: \"{patient_query_en}\"")
    
    print_warning("\n  🚨 CRITICAL EMERGENCY DETECTED")
    print("  Keywords: 'cannot breathe', 'breathlessness'")
    print("  Severity: CRITICAL")
    
    # Generate patient audio
    audio_file = "cache/demo_retell_patient_emergency.mp3"
    await generate_voice_audio(
        "મને શ્વાસ લેવામાં બહુ તકલીફ થઈ રહી છે, હું શ્વાસ લઈ નથી શકતો",
        "gu",
        audio_file
    )
    print_audio(f"Emergency audio: {audio_file}")
    
    print_section("SIP-REFER Warm Handoff to Human Agent")
    
    sip_message = """
    SIP/2.0 302 Moved Temporarily
    Via: SIP/2.0/WSS retell.palliative.care;branch=z9hG4bK776asdhds
    From: <sip:ai-agent@palliative.care>;tag=1928301774
    To: <sip:patient@palliative.care>;tag=a6c85cf
    Call-ID: a84b4c76e66710@pc33.palliative.care
    CSeq: 314159 INVITE
    Contact: <sip:dr-priya@palliative.care>
    Refer-To: <sip:copd-emergency@palliative.care>
    Referred-By: <sip:ai-agent@palliative.care>
    Content-Type: application/sdp
    Content-Length: 0
    
    X-Context-Transfer: {
      "patient_id": "PT-COPD-2026-042",
      "emergency_type": "respiratory_distress",
      "ai_summary": "72yo COPD patient, severe breathlessness, cyanosis risk",
      "conversation_history": "...",
      "recommended_action": "Immediate bronchodilator + oxygen assessment"
    }
    """
    
    print(sip_message)
    
    print_section("Human Agent Connection")
    
    agent = {
        "name": "Dr. Priya Sharma",
        "specialization": "Palliative Care Physician",
        "availability": "Online",
        "connection_time": "< 5 seconds"
    }
    
    print(f"  Agent: {agent['name']}")
    print(f"  Specialization: {agent['specialization']}")
    print(f"  Status: {agent['availability']}")
    print(f"  Connection: {agent['connection_time']}")
    
    print_section("Warm Handoff Message")
    
    handoff_msg_gu = """
    રમેશભાઈ, કૃપા કરીને ચિંતા ન કરો.
    
    હું તમને તરત જ ડૉક્ટર પ્રિયા શર્મા સાથે જોડી રહ્યો છું.
    
    તેઓ તમારા શ્વાસની તકલીફ સમજે છે અને તમને મદદ કરશે।
    
    શાંત રહો, ડૉક્ટર આવી રહ્યા છે।
    """
    
    print(f"  AI Handoff Message (Gujarati):")
    for line in handoff_msg_gu.strip().split('\n'):
        if line.strip():
            print(f"    {line}")
    
    # Generate handoff audio
    audio_file = "cache/demo_retell_handoff.mp3"
    await generate_voice_audio(
        "રમેશભાઈ, કૃપા કરીને ચિંતા ન કરો. હું તમને તરત જ ડૉક્ટર પ્રિયા શર્મા સાથે જોડી રહ્યો છું।",
        "gu",
        audio_file
    )
    print_audio(f"Handoff audio: {audio_file}")
    
    print_success("SIP-REFER warm handoff completed")

# ============================================================================
# DEMO 4: WHATSAPP BOT - TWILIO SANDBOX
# ============================================================================

async def demo_whatsapp_twilio():
    """Demo WhatsApp bot using Twilio sandbox"""
    print_header("DEMO 4: WHATSAPP BOT - TWILIO SANDBOX")
    
    print_section("Twilio Sandbox Configuration")
    
    config = {
        "sandbox_number": "+1-415-523-8886",
        "join_code": "join <unique-code>",
        "webhook_url": "https://api.pallisahayak.io/webhook/whatsapp",
        "supported_features": ["Text", "Voice Notes", "Images", "Location"]
    }
    
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print_section("User Joins Sandbox")
    
    join_flow = """
    User sends: "join pallisahayak"
    
    Twilio Webhook: POST /webhook/whatsapp
    {
      "From": "whatsapp:+919876543210",
      "Body": "join pallisahayak",
      "ProfileName": "Rajesh Kumar"
    }
    
    System Response: 
    "Welcome to Palli Sahayak! 🙏\n\nSend your health query in Hindi, English, 
    or your preferred language. You can also send voice messages."
    """
    
    print(join_flow)
    
    print_section("Sample Conversation")
    
    conversation = [
        ("User", "My father has severe back pain. He is on chemotherapy."),
        ("System", "I understand. Let me check our clinical guidelines..."),
        ("System", """
        Based on WHO Cancer Pain Guidelines (Evidence Level A):
        
        1. Morphine 5-10mg every 4 hours as needed
        2. If pain persists, consult your oncologist
        3. Monitor for constipation
        
        🟢 Confidence: 94%
        📚 Source: WHO + Max Healthcare protocols
        """),
        ("User", "/remind Morphine 08:00,20:00 10mg"),
        ("System", "✅ Reminder set for Morphine 10mg at 08:00 and 20:00."),
        ("System", "You'll receive a voice call reminder."),
    ]
    
    for sender, message in conversation:
        if sender == "System":
            print(f"\n  {Colors.CYAN}🤖 {sender}:{Colors.END}")
            for line in message.strip().split('\n'):
                print(f"    {line}")
        else:
            print(f"\n  {Colors.WARNING}👤 {sender}:{Colors.END}")
            print(f"    {message}")
    
    print_section("Voice Note Feature")
    
    voice_flow = """
    User sends: [Voice Note - 15 seconds in Hindi]
    
    System Processing:
    1. Download audio from Twilio Media URL
    2. Transcribe using Groq Whisper (hi-IN)
    3. Query RAG pipeline
    4. Generate response
    5. Convert to voice using Edge TTS
    6. Send voice note back
    
    Total latency: ~3-4 seconds
    """
    
    print(voice_flow)
    
    print_success("WhatsApp Twilio demo completed")

# ============================================================================
# SUMMARY
# ============================================================================

async def print_summary():
    """Print summary of all demos"""
    print_header("VOICE AI DEMO SUMMARY")
    
    providers = {
        "Gemini Live API": {
            "use_case": "Web-based voice conversations",
            "best_for": "Real-time streaming, natural conversations",
            "audio_format": "PCM 16kHz/24kHz",
            "languages": "hi-IN, en-IN, ta-IN, mr-IN"
        },
        "Bolna.ai": {
            "use_case": "Phone calls via Twilio",
            "best_for": "Production telephony, custom voice agents",
            "stack": "Deepgram → GPT-4o → ElevenLabs",
            "languages": "7+ Indian languages"
        },
        "Retell + Vobiz.ai": {
            "use_case": "Indian PSTN (+91) with SIP-REFER",
            "best_for": "Warm handoff to human agents",
            "feature": "SIP-REFER for seamless transfer",
            "compliance": "Indian telecom regulations"
        },
        "WhatsApp + Twilio": {
            "use_case": "Text and voice messaging",
            "best_for": "Async communication, reminders",
            "features": "Voice notes, images, location",
            "sandbox": "Easy testing environment"
        }
    }
    
    for provider, details in providers.items():
        print(f"\n  {Colors.BOLD}{provider}{Colors.END}")
        for key, value in details.items():
            print(f"    {key}: {value}")
    
    print(f"\n{Colors.GREEN}{'='*70}{Colors.END}")
    print(f"{Colors.GREEN}{Colors.BOLD}🎯 All Voice AI Demos Completed Successfully!{Colors.END}")
    print(f"{Colors.GREEN}{'='*70}{Colors.END}\n")

# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Run all voice AI demos"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔" + "="*68 + "╗")
    print("║" + " PALLI SAHAYAK - VOICE AI DEMO SUITE ".center(68) + "║")
    print("║" + " Gemini Live | Bolna | Retell+Vobiz | WhatsApp ".center(68) + "║")
    print("╚" + "="*68 + "╝")
    print(f"{Colors.END}\n")
    
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Audio Generation: Edge TTS (Microsoft)")
    print(f"Languages: Hindi, Gujarati, English, Bengali, Tamil")
    print()
    
    # Ensure cache directory exists
    Path("cache").mkdir(exist_ok=True)
    
    try:
        # Run all demos
        await demo_gemini_live()
        await demo_bolna()
        await demo_retell_vobiz_handoff()
        await demo_whatsapp_twilio()
        await print_summary()
        
    except Exception as e:
        print(f"\n{Colors.FAIL}Demo failed: {e}{Colors.END}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
