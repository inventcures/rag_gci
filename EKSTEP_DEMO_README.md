# Palli Sahayak - EkStep Voice AI Demo Package
## Voice AI - Making the Best Work for India
### The Ritz-Carlton, Bengaluru | January 28, 2026

---

## 📦 Deliverables

### 1. Presentation (43MB)
**File:** `Palli_Sahayak_EkStep_V2_Final.pptx`

**13 Slides:**
1. Title - पल्ली सहायक • Palli Sahayak
2. The Challenge - 10M+ patients, 1-2% access
3. GCI Grant - Dr Anurag Agrawal (PI), Ashish Makani (Co-I)
4. Clinical Partners - Max Healthcare + Pallium India
5. Marathi Demo Video - Dr Sachin & Dr Prakash (Cipla Foundation)
6. Solution Overview
7. Architecture
8. Safety Features
9. **Terminal Demo Video** - Actual code running
10. Demo Flow
11. Impact Stats
12. Open Source / DPG
13. Thank You

**Embedded Videos:**
- Slide 5: Marathi interaction video (36MB)
- Slide 9: Terminal demo (7.1MB)

---

### 2. Demo Videos

#### A. Terminal Demo with Real Code (7.1MB)
**File:** `palli_sahayak_terminal_demo.mp4`
- Duration: ~137 seconds
- Shows actual terminal output from running test cases
- Realistic clinical scenarios with colored terminal output

**Test Cases Shown:**
1. Oncology patient on chemotherapy - Medication reminders
2. COPD patient - Chronic management
3. COPD emergency - Auto-escalation with SIP-REFER
4. Evidence badges with RAG citations

#### B. System Demo (Mock UI) (653KB)
**File:** `palli_sahayak_system_demo.mp4`
- Screen recording style with realistic UIs
- Shows WhatsApp, Admin Dashboard, SIP-REFER flow

#### C. Feature Demo (Mock) (609KB)
**File:** `palli_sahayak_features_demo.mp4`
- Original feature demonstration

---

### 3. Demo Launcher Script
**File:** `launch_voice_ai_demos.py`

Shows actual commands to run live demos:

```bash
# Show all demo commands
python3 launch_voice_ai_demos.py --demo all

# Individual demos
python3 launch_voice_ai_demos.py --demo gemini
python3 launch_voice_ai_demos.py --demo bolna
python3 launch_voice_ai_demos.py --demo retell
```

**Demonstrates:**
- Gemini Live API - Web UI at localhost:8000/voice
- Bolna.ai - Phone calls with Deepgram → GPT-4o → ElevenLabs
- Retell + Vobiz.ai - Indian PSTN with SIP-REFER warm handoff

---

### 4. Clinical Unit Tests
**File:** `test_realistic_clinical_scenarios.py`

Run actual system tests with:
```bash
python3 test_realistic_clinical_scenarios.py
```

**Test Cases:**
1. PT-ONCO-2026-001: Mrs. Lakshmi Devi, Stage III Breast Cancer
2. PT-COPD-2026-042: Mr. Ramesh Patel, Severe COPD
3. Emergency escalation for breathlessness
4. Evidence badges with WHO/Max Healthcare citations

---

## 🎤 Voice AI Providers Highlighted

### 1. Gemini Live API
- **Use Case:** Web-based voice conversations
- **Audio:** PCM 16kHz input / 24kHz output
- **Languages:** hi-IN, en-IN, ta-IN, mr-IN
- **Web UI:** http://localhost:8000/voice
- **Tech:** WebSocket streaming, native audio

### 2. Bolna.ai
- **Use Case:** Phone calls via Twilio
- **Stack:** Deepgram (ASR) → GPT-4o-mini (LLM) → ElevenLabs (TTS)
- **Languages:** 7+ Indian languages
- **Features:** Custom voice agents, DTMF handling

### 3. Retell.ai + Vobiz.ai
- **Use Case:** Indian PSTN (+91) with warm handoff
- **Feature:** SIP-REFER call transfer
- **Integration:** DID provider for Indian numbers
- **Handoff:** AI → Human with full context preservation

### 4. WhatsApp (Twilio Sandbox)
- **Use Case:** Text and voice messaging
- **Number:** +1-415-523-8886
- **Join:** "join pallisahayak"
- **Features:** Voice notes, medication reminders

---

## 🚀 How to Run Live Demos

### Prerequisites
```bash
# API Keys (check .env)
- BOLNA_API_KEY=bn-db24586d5a8b4374a0edf7f9be224a7e
- RETELL_API_KEY=key_cea5017a481500c563aba16ca29a
- GEMINI_API_KEY (if using Gemini Live)
```

### Start Gemini Live Demo
```bash
# 1. Start server
python3 simple_rag_server.py

# 2. Open browser
open http://localhost:8000/voice

# 3. Record with QuickTime/OBS
```

### Start Bolna Demo
```bash
# Shows actual commands
python3 launch_voice_ai_demos.py --demo bolna
```

### Start Retell+Vobiz Demo
```bash
# Shows actual commands  
python3 launch_voice_ai_demos.py --demo retell
```

---

## 📊 Key Statistics in Presentation

- **10M+** Indians need palliative care
- **1-2%** currently have access
- **7+** Indian languages supported
- **3** Voice AI providers (Gemini, Bolna, Retell)
- **GCI Grant:** Awarded Nov 2024
- **Clinical Partners:** Max Healthcare, Pallium India

---

## 👥 Team

- **PI:** Dr. Anurag Agrawal
- **Co-I:** Ashish Makani
- **Institution:** Ashoka University & KCDHA (Karnataka)
- **Funders:** Gates Foundation (India) & BIRAC-DBT
- **Clinical Partners:** Max Healthcare (Delhi), Pallium India (Kerala)

---

## 🔗 Links

- **Website:** https://inventcures.github.io/palli-sahayak/
- **GitHub:** github.com/inventcures/rag_gci
- **Docs:** deepwiki.com/inventcures/rag_gci

---

## ✅ Checklist for Presentation

- [x] Presentation with 13 slides
- [x] Marathi demo video embedded
- [x] Terminal demo video embedded
- [x] GCI Grant details
- [x] Clinical partner logos referenced
- [x] All 3 Voice AI providers documented
- [x] SIP-REFER warm handoff explained
- [x] Demo launcher script ready

---

**Ready for EkStep Voice AI Event! 🚀**
