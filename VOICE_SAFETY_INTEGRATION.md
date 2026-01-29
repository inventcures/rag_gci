# Voice Safety Integration Guide

## Overview

All 5 safety enhancements are now integrated into **every voice AI flow** in Palli Sahayak:

1. ✅ **Gemini Live API** - Web voice with native audio
2. ✅ **Bolna.ai** - Phone calls via Twilio
3. ✅ **Retell.AI + Vobiz** - Phone calls via Indian PSTN (+91)
4. ✅ **Voice Router** - Centralized routing with safety checks

---

## Safety Features for Voice

### 1. Real-Time Emergency Detection 🚨

**How it works:**
- Voice transcript is analyzed **immediately** upon reception
- Emergency keywords detected in **5 languages** (en, hi, bn, ta, gu)
- Critical emergencies trigger **instant escalation**

**Integration Points:**

```python
# Gemini Live - gemini_live/service.py
# In _query_rag_and_inject() method:
safety_result = await safety_wrapper.check_voice_query(
    user_id=self.session_id,
    transcript=query_text,
    language=self.language
)
if safety_result.should_escalate:
    # Inject safety message to Gemini
    await self._session.send_client_content(
        turns=[types.Content(
            role="user",
            parts=[types.Part(text=safety_result.safety_message)]
        )]
    )
```

```python
# Retell - retell_integration/custom_llm_server.py
# In _handle_response_required() method:
safety_result = await safety_wrapper.check_voice_query(
    user_id=session.from_number,
    transcript=user_query,
    language=session.language
)
if safety_result.should_escalate:
    await self._send_response(
        session.websocket,
        response_id,
        safety_result.safety_message
    )
```

```python
# Bolna - simple_rag_server.py
# In /api/bolna/query endpoint:
safety_result = await safety_wrapper.check_voice_query(
    user_id=f"bolna_{source}",
    transcript=query,
    language=language
)
if safety_result.should_escalate:
    return JSONResponse({
        "status": "safety_escalation",
        "answer": safety_result.safety_message
    })
```

---

### 2. Evidence Badges in Voice 🔬

**How it works:**
- Evidence quality assessed after RAG query
- Low confidence triggers spoken disclaimer
- Voice-optimized evidence warnings

**Voice Response Format:**
```
[Normal response]... 

Please note: I recommend consulting a physician for this matter.
```

**Implementation:**
```python
# Add to voice response if needed
answer = safety_wrapper.add_evidence_to_voice(
    answer, 
    evidence_badge, 
    language
)
```

---

### 3. Response Length Optimization 📏

**How it works:**
- Responses optimized for **30-second voice duration**
- Comprehension level adapted per user
- Markdown/formatting stripped for speech

**Optimization Rules:**
| Aspect | Rule |
|--------|------|
| Max Duration | 30 seconds (~130 words) |
| Formatting | Remove markdown, bullets → spoken |
| Citations | Moved to end or removed |
| URLs | Stripped |

**Implementation:**
```python
answer = safety_wrapper.optimize_for_voice(
    answer,
    user_id=user_id,
    language=language,
    max_duration_seconds=30
)
```

---

### 4. Human Handoff for Voice 👤

**How it works:**
- Detects handoff requests ("talk to human", "doctor please")
- Creates handoff ticket automatically
- Provides warm transfer message

**Voice Handoff Flow:**
```
User: "I want to talk to a human"
AI: "I understand you'd like to speak with a human caregiver. 
    I'm connecting you now. Your request ID is ABC123. 
    A healthcare professional will be with you shortly."
    [Handoff ticket created]
```

**Trigger Phrases by Language:**
- **English**: "talk to doctor", "human please", "real person"
- **Hindi**: "डॉक्टर से बात", "इंसान से", "असली आदमी"
- **Bengali**: "ডাক্তারের সাথে কথা", "মানুষের সাথে"
- **Tamil**: "மருத்துவரிடம்", "மனிதரிடம்"

---

### 5. Medication Reminders via Voice 💊

**How it works:**
- Voice commands recognized for reminders
- Guided to WhatsApp for detailed setup
- "TAKEN" confirmation after reminders

**Voice Commands:**
```
User: "Remind me to take Paracetamol at 8 AM"
AI: "I can help you set up medication reminders. 
    Please use the WhatsApp chat to set reminders 
    with the command: /remind followed by medication 
    name, times, and dosage."
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VOICE SAFETY WRAPPER                      │
├─────────────────────────────────────────────────────────────┤
│  check_voice_query()                                         │
│    ├── Emergency Detection (all languages)                   │
│    ├── Human Handoff Detection                              │
│    ├── Medication Command Recognition                       │
│    └── Comprehension Analysis                               │
├─────────────────────────────────────────────────────────────┤
│  optimize_for_voice()                                        │
│    ├── Length optimization (30s max)                        │
│    ├── Text cleaning (no markdown)                          │
│    └── Evidence warnings                                    │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Gemini Live │    │   Bolna.ai   │    │ Retell.AI    │
│   (Web)      │    │   (Phone)    │    │  (Phone)     │
└──────────────┘    └──────────────┘    └──────────────┘
```

---

## Provider-Specific Integration

### Gemini Live API

**File:** `gemini_live/service.py`

**Integration Point:** `_query_rag_and_inject()` method

**Flow:**
1. User speaks → Audio streamed to Gemini
2. Transcription received → **SAFETY CHECK**
3. If emergency → Inject safety alert to Gemini context
4. If normal → Continue with RAG context injection

**Code:**
```python
# In gemini_live/service.py
from voice_safety_wrapper import get_voice_safety_wrapper

safety_wrapper = get_voice_safety_wrapper()
safety_result = await safety_wrapper.check_voice_query(
    user_id=self.session_id,
    transcript=query_text,
    language=self.language
)

if safety_result.should_escalate:
    # Override normal flow with safety message
    await self._session.send_client_content(
        turns=[types.Content(
            role="user",
            parts=[types.Part(text=safety_result.safety_message)]
        )]
    )
```

---

### Bolna.ai

**File:** `simple_rag_server.py` ( `/api/bolna/query` endpoint)

**Integration Point:** Bolna custom function call handler

**Flow:**
1. Bolna LLM detects need for RAG
2. Calls `/api/bolna/query` endpoint
3. **SAFETY CHECK** before RAG query
4. If emergency → Return safety escalation response
5. If normal → Query RAG, optimize for voice, return

**Code:**
```python
# In simple_rag_server.py - /api/bolna/query
from voice_safety_wrapper import get_voice_safety_wrapper

safety_wrapper = get_voice_safety_wrapper()
safety_result = await safety_wrapper.check_voice_query(
    user_id=f"bolna_{source}",
    transcript=query,
    language=language
)

if safety_result.should_escalate:
    return JSONResponse({
        "status": "safety_escalation",
        "answer": safety_result.safety_message
    })
```

---

### Retell.AI

**File:** `retell_integration/custom_llm_server.py`

**Integration Point:** `_handle_response_required()` method

**Flow:**
1. Retell sends transcript via WebSocket
2. **SAFETY CHECK** before processing
3. If emergency → Send safety response via WebSocket
4. If normal → Query RAG, optimize, send response

**Code:**
```python
# In retell_integration/custom_llm_server.py
from voice_safety_wrapper import get_voice_safety_wrapper

safety_wrapper = get_voice_safety_wrapper()
safety_result = await safety_wrapper.check_voice_query(
    user_id=session.from_number,
    transcript=user_query,
    language=session.language
)

if safety_result.should_escalate:
    await self._send_response(
        session.websocket,
        response_id,
        safety_result.safety_message
    )
    return
```

---

### Voice Router

**File:** `voice_router.py`

**Integration Point:** Centralized safety check methods

**Methods Added:**
```python
async def check_voice_safety(self, transcript, user_id, language, call_id)
    """Check safety before routing to any provider"""
    
async def handle_voice_escalation(self, safety_result, provider)
    """Handle escalation across all providers"""
```

**Usage:**
```python
router = VoiceRouter()

# Check safety before routing
safety = await router.check_voice_safety(
    transcript="I can't breathe",
    user_id="+919876543210",
    language="en"
)

if safety["should_escalate"]:
    await router.handle_voice_escalation(safety, "phone")
```

---

## Testing

### Run Voice Safety Tests

```bash
# Test all voice safety features
python tests/test_voice_safety_integration.py

# Test base safety features
python tests/test_safety_enhancements.py
```

### Expected Output

```
======================================================================
VOICE SAFETY INTEGRATION TESTS
======================================================================
✅ VoiceSafetyWrapper initialized
✅ Text cleaned for voice
✅ Critical emergency detected in voice
✅ Hindi emergency detected
✅ Handoff request detected
✅ Normal query passes safety check
✅ Gemini Live safety integration works
✅ Bolna safety integration works
======================================================================
```

---

## Emergency Response Examples

### Critical Emergency (English)

**User:** "I can't breathe"

**AI Response:**
```
🚨 CRITICAL EMERGENCY DETECTED. Call emergency services (108/102) immediately!

Actions:
1. Call 108 (ambulance) or 102 immediately
2. Stay with the patient
3. Do not give food or water if unconscious
4. A human caregiver has been notified
```

### Critical Emergency (Hindi)

**User:** "सांस नहीं आ रही"

**AI Response:**
```
🚨 गंभीर आपातकाल! तुरंत आपातकालीन सेवाओं (108/102) को कॉल करें!

1. तुरंत 108 (एम्बुलेंस) या 102 पर कॉल करें
2. मरीज के पास रहें
3. बेहोश होने पर खाना-पानी न दें
4. एक मानव देखभाल करने वाले को सूचित किया गया है
```

---

## Configuration

### Environment Variables

```bash
# Voice safety (enabled by default if safety_enhancements available)
VOICE_SAFETY_ENABLED=true

# Emergency notification webhook (optional)
EMERGENCY_WEBHOOK_URL=https://your-hospital.com/api/emergency

# Caregiver phone numbers for alerts (comma-separated)
CAREGIVER_PHONES=+919876543210,+919876543211
```

### Disabling Voice Safety

To disable, set in code:
```python
VOICE_SAFETY_AVAILABLE = False
```

Or remove the import from each voice provider file.

---

## Monitoring & Logging

All voice safety events are logged:

```python
# Emergency detected
logger.critical(f"🚨 CRITICAL EMERGENCY detected for {user_id}")

# Handoff requested
logger.info(f"👤 Handoff requested by {user_id}")

# Safety escalation
logger.warning(f"🚨 Voice safety escalation: {event_type}")
```

### Log Locations

- Main log: `logs/rag_server.log`
- Voice safety events tagged with emoji for easy grep:
  - `🚨` - Emergency
  - `👤` - Human handoff
  - `💊` - Medication reminder

---

## Files Modified

| File | Changes |
|------|---------|
| `gemini_live/service.py` | Added safety check in `_query_rag_and_inject()` |
| `retell_integration/custom_llm_server.py` | Added safety check in `_handle_response_required()` |
| `simple_rag_server.py` | Added safety check in `/api/bolna/query` endpoint |
| `voice_router.py` | Added safety check methods |

## Files Created

| File | Purpose |
|------|---------|
| `safety_enhancements.py` | Core safety features (5 enhancements) |
| `voice_safety_wrapper.py` | Voice-specific safety integration |
| `tests/test_safety_enhancements.py` | Unit tests for safety features |
| `tests/test_voice_safety_integration.py` | Integration tests for voice |
| `SAFETY_ENHANCEMENTS.md` | General safety documentation |
| `VOICE_SAFETY_INTEGRATION.md` | This file - voice-specific docs |

---

## Support

For issues:
1. Check logs: `grep "🚨\|👤\|💊" logs/rag_server.log`
2. Run tests: `python tests/test_voice_safety_integration.py`
3. Verify integrations are loading in startup logs

---

**Last Updated:** January 2025  
**Version:** 1.0.0  
**Status:** ✅ Integrated in all voice flows
