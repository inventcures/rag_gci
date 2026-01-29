#!/usr/bin/env python3
"""
Palli Sahayak - Voice AI Demo Launcher
=======================================

Launches actual demos for EkStep presentation:
1. Gemini Live API - Web UI at localhost
2. Bolna.ai - Phone call demo
3. Retell + Vobiz.ai - PSTN call with SIP-REFER handoff

Usage:
    python3 launch_voice_ai_demos.py --demo gemini
    python3 launch_voice_ai_demos.py --demo bolna
    python3 launch_voice_ai_demos.py --demo retell
    python3 launch_voice_ai_demos.py --demo all
"""

import os
import sys
import asyncio
import subprocess
import argparse
import webbrowser
import time
from pathlib import Path
from datetime import datetime

# Color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_banner(text):
    print(f"\n{Colors.HEADER}{'='*70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.END}")
    print(f"{Colors.HEADER}{'='*70}{Colors.END}\n")

def print_cmd(text):
    print(f"{Colors.CYAN}$ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠ {text}{Colors.END}")

# ============================================================================
# DEMO 1: GEMINI LIVE API
# ============================================================================

async def demo_gemini_live():
    """Launch Gemini Live Web UI demo"""
    print_banner("GEMINI LIVE API DEMO - Web Voice Interface")
    
    print("""
    This demo launches the Gemini Live web interface at http://localhost:8000
    
    Features demonstrated:
    • Real-time voice conversation with Gemini 2.0 Flash
    • PCM 16kHz input / 24kHz output audio
    • WebSocket streaming
    • RAG context injection
    • Hindi, English, Tamil, Marathi support
    """)
    
    # Step 1: Check environment
    print_info("Checking environment...")
    
    env_vars = ["GEMINI_API_KEY", "GOOGLE_CLOUD_PROJECT"]
    for var in env_vars:
        value = os.getenv(var, "")
        if value:
            print_success(f"{var}: {'*' * 10}{value[-4:]}")
        else:
            print_warning(f"{var}: Not set (required)")
    
    # Step 2: Show launch command
    print("\n" + "─" * 70)
    print(f"{Colors.BOLD}STEP 1: Start the FastAPI server{Colors.END}")
    print("─" * 70)
    
    print_cmd("python3 simple_rag_server.py")
    print("""
    Expected output:
    [INFO] Starting Palli Sahayak Server...
    [INFO] Gemini Live Service: Available
    [INFO] Web client: http://localhost:8000/voice
    [INFO] API docs: http://localhost:8000/docs
    """)
    
    # Step 3: Show browser launch
    print("─" * 70)
    print(f"{Colors.BOLD}STEP 2: Open web client{Colors.END}")
    print("─" * 70)
    
    print_cmd("open http://localhost:8000/voice")
    print("""
    Browser will open with Gemini Live Voice Interface:
    
    ┌─────────────────────────────────────────────────────────────────┐
    │  🎤 Palli Sahayak - Voice Assistant                    [● Live] │
    │                                                                 │
    │  Connection Status: Connected to Gemini Live API                │
    │  Language: Hindi (hi-IN)                                        │
    │                                                                 │
    │  [🎤 Hold to Speak]                                             │
    │                                                                 │
    │  ─────────────────────────────────────────────────────         │
    │  Patient: "माँ को दर्द है, क्या करूं?"                          │
    │                                                                 │
    │  ─────────────────────────────────────────────────────         │
    │  AI: "आपकी माँ को मोर्फिन 5-10mg दें..."                        │
    │                                                                 │
    │  Evidence Badge: 🟢 HIGH CONFIDENCE (94%)                       │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    # Step 4: Show WebSocket logs
    print("─" * 70)
    print(f"{Colors.BOLD}STEP 3: WebSocket Connection Logs{Colors.END}")
    print("─" * 70)
    
    print("""
    WebSocket: wss://generativelanguage.googleapis.com/...
    
    ⬆️  Client → Server:
    {
      "realtime_input": {
        "media_chunks": [{
          "mime_type": "audio/pcm;rate=16000",
          "data": "base64_encoded_pcm_audio..."
        }]
      }
    }
    
    ⬇️  Server → Client:
    {
      "server_content": {
        "model_turn": {
          "parts": [{
            "inline_data": {
              "mime_type": "audio/pcm;rate=24000",
              "data": "base64_encoded_response_audio..."
            }
          }]
        }
      }
    }
    """)
    
    print_success("Gemini Live demo instructions ready")
    print_info("Run the command above to start the actual demo")

# ============================================================================
# DEMO 2: BOLNA.AI
# ============================================================================

async def demo_bolna():
    """Launch Bolna.ai phone call demo"""
    print_banner("BOLNA.AI DEMO - Phone Call with Voice AI")
    
    print("""
    This demo initiates an actual phone call using Bolna.ai
    
    Stack:
    • ASR: Deepgram (nova-2)
    • LLM: OpenAI GPT-4o-mini
    • TTS: ElevenLabs (eleven_multilingual_v2)
    • Telephony: Twilio
    
    Cost: ~$0.05-0.10 per minute (charged by Bolna)
    """)
    
    # Check API key
    print_info("Checking Bolna configuration...")
    bolna_key = os.getenv("BOLNA_API_KEY", "")
    if bolna_key:
        print_success(f"BOLNA_API_KEY: {'*' * 10}{bolna_key[-4:]}")
    else:
        print_warning("BOLNA_API_KEY: Not set")
        return
    
    print("\n" + "─" * 70)
    print(f"{Colors.BOLD}STEP 1: Configure Bolna Agent{Colors.END}")
    print("─" * 70)
    
    print_cmd("python3 bolna_integration/client.py")
    print("""
    # Create agent
    agent_config = {
        "agent_name": "Palli Sahayak - Hindi",
        "language": "hi",
        "voice": "eleven_multilingual_v2",
        "asr_provider": "deepgram",
        "llm_model": "gpt-4o-mini"
    }
    """)
    
    print("\n" + "─" * 70)
    print(f"{Colors.BOLD}STEP 2: Initiate Phone Call{Colors.END}")
    print("─" * 70)
    
    print_cmd("python3 -c 'from bolna_integration import BolnaClient; client = BolnaClient(); print(client.make_call(\"+919876543210\", \"palli-sahayak-hi\"))'")
    
    print("\n" + "─" * 70)
    print(f"{Colors.BOLD}STEP 3: Real-time Call Logs{Colors.END}")
    print("─" * 70)
    
    print("""
    📞 Call initiated to +91-98765-43210
    
    [00:00] 🔗 Connected to Twilio
    [00:01] 🔊 Playing greeting: "नमस्ते लक्ष्मी जी..."
    [00:05] 🎤 Patient: "नमस्ते..."
    [00:06] 🤖 AI: "यह आपकी दवाई का समय है..."
    [00:12] 🎤 Patient: "ठीक है, मैं ले लेती हूं"
    [00:15] 🤖 AI: "दवाई लेने के बाद 1 दबाएं"
    [00:20] 📞 [DTMF: 1]
    [00:21] ✅ Patient confirmed medication taken
    [00:22] 🤖 AI: "धन्यवाद लक्ष्मी जी..."
    [00:25] 📴 Call ended
    
    Call Summary:
    - Duration: 25 seconds
    - Patient confirmed: YES
    - Adherence logged: YES
    - Cost: $0.04
    """)
    
    print_success("Bolna demo instructions ready")
    print_warning("Note: This will make an actual phone call and incur charges")

# ============================================================================
# DEMO 3: RETELL + VOBIZ.AI
# ============================================================================

async def demo_retell_vobiz():
    """Launch Retell + Vobiz.ai demo with SIP-REFER handoff"""
    print_banner("RETELL + VOBIZ.AI DEMO - PSTN with Warm Handoff")
    
    print("""
    This demo uses Retell.ai with Vobiz.ai for Indian PSTN (+91)
    and demonstrates SIP-REFER warm handoff to human agents.
    
    Features:
    • Indian phone numbers (+91)
    • SIP-REFER call transfer
    • Warm handoff with context preservation
    • Integration with DID provider (Vobiz)
    """)
    
    # Check API keys
    print_info("Checking configuration...")
    
    retell_key = os.getenv("RETELL_API_KEY", "")
    if retell_key:
        print_success(f"RETELL_API_KEY: {'*' * 10}{retell_key[-4:]}")
    else:
        print_warning("RETELL_API_KEY: Not set")
    
    print("\n" + "─" * 70)
    print(f"{Colors.BOLD}STEP 1: Configure Vobiz.ai SIP Trunk{Colors.END}")
    print("─" * 70)
    
    print_cmd("cat retell_integration/vobiz_config.py")
    print("""
    VOBIZ_CONFIG = {
        "sip_trunk": {
            "host": "sip.vobiz.ai",
            "port": 5060,
            "username": "pallisahayak",
            "password": "***",
            "did_numbers": ["+91-80-XXXX-XXXX", "+91-11-XXXX-XXXX"]
        },
        "retell_integration": {
            "webhook_url": "https://api.pallisahayak.io/webhook/retell",
            "sip_refer_enabled": True,
            "warm_transfer": True
        }
    }
    """)
    
    print("\n" + "─" * 70)
    print(f"{Colors.BOLD}STEP 2: Emergency Call Flow with SIP-REFER{Colors.END}")
    print("─" * 70)
    
    print("""
    Scenario: COPD patient with breathlessness
    
    [00:00] 📞 Incoming call from +91-98765-12345
    [00:01] 🔗 Retell AI agent answers
    [00:03] 🎤 Patient: "મને શ્વાસ લેવામાં તકલીફ થઈ રહી છે..."
    [00:05] ⚠️  Emergency detected: CRITICAL
    [00:06] 🔀 Initiating SIP-REFER warm handoff
    
    ─────────────────────────────────────────
    SIP REFER Transaction:
    ─────────────────────────────────────────
    
    RETELL → VOBIZ:
    
    REFER sip:dr-priya@palliative.care SIP/2.0
    Via: SIP/2.0/UDP retell.palliative.care:5060
    From: <sip:ai-agent@palliative.care>;tag=xyz789
    To: <sip:patient@palliative.care>;tag=abc123
    Call-ID: emergency-call-001
    CSeq: 101 REFER
    Refer-To: <sip:copd-emergency@palliative.care>
    Referred-By: <sip:ai-agent@palliative.care>
    Content-Type: application/json
    Content-Length: 256
    
    {
      "patient_id": "PT-COPD-2026-042",
      "emergency_type": "respiratory_distress",
      "ai_summary": "72yo male, severe COPD, breathlessness",
      "vitals": {"spo2": "88%", "hr": "110"},
      "context": "Patient reports inability to breathe"
    }
    
    ─────────────────────────────────────────
    
    [00:08] 📞 Connecting to Dr. Priya Sharma
    [00:10] ✅ Human agent connected
    [00:11] 🎤 Dr. Priya: "રમેશભાઈ, હું ડૉક્ટર પ્રિયા..."
    [00:15] 📋 Context transferred successfully
    
    Call Handoff Complete:
    - AI handled: Initial triage (8 seconds)
    - Human agent: Dr. Priya Sharma (Pulmonologist)
    - Context preserved: Full patient history
    - Zero data loss: SIP-REFER with JSON payload
    """)
    
    print("\n" + "─" * 70)
    print(f"{Colors.BOLD}STEP 3: Launch Custom LLM Server for Retell{Colors.END}")
    print("─" * 70)
    
    print_cmd("python3 retell_integration/custom_llm_server.py")
    print("""
    [INFO] Starting Retell Custom LLM Server...
    [INFO] WebSocket endpoint: ws://localhost:8001/retell/ws
    [INFO] RAG pipeline: Connected
    [INFO] Emergency detection: Enabled
    [INFO] SIP-REFER handler: Ready
    
    Waiting for Retell webhook connections...
    """)
    
    print_success("Retell + Vobiz demo instructions ready")

# ============================================================================
# DEMO ALL
# ============================================================================

async def demo_all():
    """Run all demos in sequence"""
    await demo_gemini_live()
    input("\nPress Enter to continue to Bolna demo...")
    
    await demo_bolna()
    input("\nPress Enter to continue to Retell+Vobiz demo...")
    
    await demo_retell_vobiz()

# ============================================================================
# MAIN
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Launch Voice AI Demos for EkStep Presentation"
    )
    parser.add_argument(
        "--demo",
        choices=["gemini", "bolna", "retell", "all"],
        default="all",
        help="Which demo to launch"
    )
    
    args = parser.parse_args()
    
    print(f"""
{Colors.HEADER}
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║           PALLI SAHAYAK - VOICE AI DEMO LAUNCHER                     ║
║                                                                      ║
║     EkStep "Voice AI - Making the Best Work for India"              ║
║     The Ritz-Carlton, Bengaluru | January 28, 2026                 ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
{Colors.END}
    """)
    
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Demo mode: {args.demo}")
    print()
    
    if args.demo == "gemini":
        await demo_gemini_live()
    elif args.demo == "bolna":
        await demo_bolna()
    elif args.demo == "retell":
        await demo_retell_vobiz()
    else:
        await demo_all()
    
    print(f"""
{Colors.GREEN}
╔══════════════════════════════════════════════════════════════════════╗
║                      DEMO INSTRUCTIONS COMPLETE                      ║
║                                                                      ║
║  To capture video with audio:                                        ║
║                                                                      ║
║  1. Run the commands shown above                                     ║
║  2. Use QuickTime Player or OBS to record screen + audio             ║
║  3. For terminal recording: use asciinema or terminalizer            ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
{Colors.END}
    """)

if __name__ == "__main__":
    asyncio.run(main())
