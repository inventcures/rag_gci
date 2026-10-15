# Palli Sahayak Safety Enhancements

## Overview

This document describes the 5 quick-win safety enhancement features implemented for the Palli Sahayak palliative care AI system. These features enhance patient safety, improve trust through transparency, and ensure warm handoffs to human caregivers when needed.

---

## 1. Evidence Badges 🔬

### What It Does
Every AI response now includes an **evidence badge** that shows:
- **Confidence Level** (0-100%): How confident the AI is in the answer
- **Evidence Grade** (A-E): Quality of medical sources
  - 🟢 **A**: RCT/Meta-analysis (highest quality)
  - 🟡 **B**: Well-designed controlled studies
  - 🟠 **C**: Observational/limited evidence
  - 🔵 **D**: Expert opinion
  - 🔴 **E**: Consult physician (insufficient evidence)
- **Source Quality**: Description of the medical sources used
- **Consult Physician Warning**: When the answer requires medical review

### Example Badge
```
────────────────────────────────────────
🟢 Confidence: 85%
📚 Source Quality: Excellent - Authoritative medical sources
────────────────────────────────────────
```

### How It Works
- Analyzes vector similarity scores (distance metrics)
- Checks source document quality (WHO guidelines > blog posts)
- Detects high-stakes medical queries (emergencies)
- Identifies uncertainty indicators in AI responses
- Dynamically calculates evidence grade and recommendations

### User Impact
- **Builds trust**: Users see the quality backing each answer
- **Promotes safety**: Clear warnings when physician consultation is needed
- **Transparency**: Users understand the limitations of AI responses

---

## 2. Emergency Detection & Escalation 🚨

### What It Does
Automatically detects emergency situations from user messages in **5 languages**:
- **English**
- **Hindi** (हिन्दी)
- **Bengali** (বাংলা)
- **Tamil** (தமிழ்)
- **Gujarati** (ગુજરાતી)

### Emergency Levels
| Level | Trigger Words | Action |
|-------|--------------|--------|
| 🔴 **CRITICAL** | "can't breathe", "unconscious", "heart attack", "suicide" | Immediate escalation + emergency services notification |
| 🟠 **HIGH** | "severe pain", "high fever", "can't move" | Urgent caregiver notification |
| 🟡 **MEDIUM** | "worried", "nausea", "rash" | Advice to consult doctor soon |

### Example Triggers

#### English
- "I can't breathe" → CRITICAL
- "Having severe chest pain" → CRITICAL
- "High fever won't go down" → HIGH

#### Hindi
- "सांस नहीं आ रही" → CRITICAL
- "छाती में दर्द" → CRITICAL
- "बहुत बुखार है" → HIGH

#### Bengali
- "শ্বাস নিতে পারছি না" → CRITICAL
- "বুকে ব্যথা" → CRITICAL

### Response Actions
1. **Immediate Safety Message**: Clear instructions in user's language
2. **Emergency Numbers**: Displays 108/102 for ambulance services
3. **Caregiver Notification**: Alerts registered family members/caregivers
4. **Human Handoff**: Creates urgent ticket for healthcare professional

---

## 3. Medication Reminder Scheduler 💊

### What It Does
Allows patients and caregivers to schedule medication reminders via WhatsApp.

### Commands

#### Set a Reminder
```
/remind <medicine_name> <times>
```

**Examples:**
```
/remind Paracetamol 08:00,20:00
/remind Ondansetron 06:00,18:00
```

A reminder names the medicine and the time, never a dose (ADR 0004, decision
D10 of the round-3 review). Anything typed after the times is stored with the
reminder as its dosage but is never repeated back, sent or spoken.

#### View Your Reminders
```
/myreminders
```

**Output:**
```
💊 Your Medication Reminders

1. Paracetamol
   🕐 08:00, 20:00
   ✅ Active (ID: abc123...)

2. Ondansetron
   🕐 06:00, 18:00
   ✅ Active (ID: def456...)
```

#### Delete a Reminder
```
/deletereminder <reminder_id>
```

#### Mark as Taken
Reply `TAKEN` after receiving a reminder:
```
✅ Great! I've recorded that you took your Paracetamol.
📊 Your adherence: 85%
```

### Features
- **Flexible Scheduling**: Multiple times per day (up to 5)
- **Multi-language**: Reminders in user's preferred language
- **Adherence Tracking**: Calculates medication adherence percentage
- **Persistent Storage**: Reminders survive server restarts
- **Smart Notifications**: Gentle reminders with dosage instructions

---

## 4. Response Length Optimization 📏

### What It Does
Automatically adapts response length and complexity based on the user's communication style.

### Comprehension Levels

#### Simple (🟢)
- **Max Length**: 500 characters
- **Style**: 4 short sentences, bullet points
- **Vocabulary**: 8th-grade level, no jargon
- **For**: Brief messages, simple questions

#### Moderate (🟡)
- **Max Length**: 1000 characters
- **Style**: 8 sentences, paragraph + key points
- **Vocabulary**: Explains medical terms
- **For**: Average users, standard questions

#### Detailed (🔵)
- **Max Length**: 2000 characters
- **Style**: Comprehensive, technical
- **Vocabulary**: Medical terminology with citations
- **For**: Healthcare professionals, complex questions

### How It Adapts
The system analyzes:
- Message length and sentence complexity
- Use of medical terminology
- Question types (what/how/why)
- Conversation history

### Example Adaptation

**User Query**: "What is morphine?"

**AI Analysis**: Short question, simple vocabulary → Level: SIMPLE

**Response**:
```
Morphine is a strong pain medicine.

• It helps severe pain
• Take as doctor prescribed
• Can cause drowsiness

⚠️ Never take more than prescribed
```

**User Query**: "Explain the pharmacodynamics of morphine sulfate in palliative care"

**AI Analysis**: Complex vocabulary, technical terms → Level: DETAILED

**Response**:
```
Morphine sulfate is an opioid analgesic that works primarily as a μ-opioid receptor agonist...
[Full technical explanation with mechanisms]
```

---

## 5. Human Handoff System 👤

### What It Does
Provides seamless, warm handoffs to human caregivers when the AI cannot adequately help.

### Trigger Conditions

| Reason | Trigger | Auto-Handoff |
|--------|---------|--------------|
| **Emergency** | Critical symptoms detected | ✅ Yes |
| **User Request** | "/human" or "talk to doctor" | ✅ Yes |
| **Complex Medical** | Drug interactions, rare conditions | ❌ No (offered) |
| **Emotional Support** | Grief, depression, anxiety | ❌ No (offered) |
| **AI Uncertain** | Low confidence (<50%) | ❌ No (offered) |
| **Safety Escalation** | Potential harm detected | ✅ Yes |

### Handoff Commands

#### Request Human Help
```
/human
/talktohuman
```

**Response:**
```
👤 Connecting you to a human caregiver as requested.

Request ID: ABC123XYZ
Someone will be with you shortly.

For emergencies, call 108 immediately.
```

### Handoff Workflow

```
User Request
    ↓
AI Detects Handoff Need
    ↓
Create Handoff Ticket
    ↓
Notify Available Caregivers
    ↓
Send User Confirmation (with Request ID)
    ↓
Caregiver Accepts → Status: "assigned"
    ↓
Caregiver Contacts User
    ↓
Resolution → Status: "resolved"
```

### Caregiver Dashboard (Future)
Pending handoff requests are stored with:
- User ID and context
- Conversation history
- Priority level
- Timestamp

Caregivers can:
1. View pending requests
2. Accept assignments
3. Mark as in-progress
4. Resolve with notes

---

## Integration with WhatsApp Bot

### New WhatsApp Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/human` | Request human caregiver |
| `/remind <medicine> <times>` | Set medication reminder |
| `/myreminders` | List all reminders |
| `/deletereminder <id>` | Delete a reminder |
| `TAKEN` | Mark medication as taken |
| `/lang <code>` | Change language (hi/en/bn/ta/gu) |

### Automatic Safety Features

Every incoming message is automatically checked for:
1. **Emergency keywords** → Immediate escalation if detected
2. **Handoff triggers** → Offer human help if appropriate
3. **User comprehension** → Adapt response style

### Response Flow

```
Incoming Message
    ↓
[Emergency Check] → If CRITICAL: Immediate escalation
    ↓
[Handoff Check] → If triggered: Offer human help
    ↓
[Comprehension Analysis] → Determine user level
    ↓
[RAG Query] → Search medical documents
    ↓
[Generate Response] + [Safety Prompt]
    ↓
[Add Evidence Badge]
    ↓
[Optimize Length]
    ↓
Send to User
```

---

## Data Storage

All safety enhancement data is stored locally:

```
data/
├── medication_reminders/
│   └── reminders.json
├── comprehension_profiles/
│   └── profiles.json
└── handoff_requests/
    ├── pending.json
    └── resolved.json
```

**Privacy**: No medical data is sent to external services beyond the core RAG pipeline.

---

## Testing

Run the safety enhancements test suite:

```bash
python tests/test_safety_enhancements.py
```

Or with pytest:

```bash
pytest tests/test_safety_enhancements.py -v
```

### Test Coverage
- ✅ Evidence badge calculation
- ✅ Emergency detection (all languages)
- ✅ Medication reminder CRUD operations
- ✅ Response length adaptation
- ✅ Human handoff workflows
- ✅ Integration scenarios

---

## Configuration

### Environment Variables

Add to your `.env` file:

```bash
# Safety Enhancements (optional - works without)
SAFETY_ENHANCEMENTS_ENABLED=true

# Caregiver notification numbers (comma-separated)
CAREGIVER_PHONES=+919876543210,+919876543211

# Emergency escalation webhook (optional)
EMERGENCY_WEBHOOK_URL=https://your-hospital.com/api/emergency
```

### Disabling Features

To disable safety enhancements, set in code:

```python
SAFETY_ENHANCEMENTS_AVAILABLE = False
```

---

## Future Enhancements

### Planned Features
1. **Proactive Crisis Prediction**: ML model to predict pain crises
2. **Caregiver Mobile App**: Real-time alerts and patient monitoring
3. **Video Consultation**: Integrated telemedicine for handoffs
4. **FHIR Integration**: Sync with hospital EHR systems
5. **Medication Interactions**: Check for drug-drug interactions

### Research Directions
- Federated learning for privacy-preserving model improvement
- Emotion detection from voice for better emotional support
- Predictive analytics for patient deterioration

---

## Support

For issues or questions about safety enhancements:

1. Check logs: `logs/rag_server.log`
2. Run diagnostics: `python validate_setup.py`
3. Review test results: `pytest tests/test_safety_enhancements.py -v`

---

## License

These safety enhancements are part of the Palli Sahayak project and follow the same open-source license.

---

**Last Updated**: January 2025  
**Version**: 1.0.0  
**Maintainer**: Palli Sahayak AI Team
