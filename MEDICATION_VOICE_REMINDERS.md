# Medication Voice Reminder System

## Overview

The Medication Voice Reminder System makes **automated voice calls** to patients to remind them to take their medications. This is especially helpful for:
- Elderly patients who prefer phone calls over text
- Patients with low literacy
- Critical medications that need confirmation
- Building better adherence through personal connection

## Features

### 📞 Voice Call Reminders
- Scheduled automated calls at medication times
- Multi-language support (English, Hindi, Bengali, Tamil, Gujarati)
- Integration with **Bolna.ai** and **Retell.AI**

### ✅ Patient Confirmation
Patients can confirm taking medication via:
- **DTMF**: Press 1 on phone keypad
- **Voice**: Say "yes", "हां", "হ্যাঁ", etc.
- **Missed call**: Call back to confirm

### 🔄 Smart Retry Logic
- Up to 3 retry attempts for failed calls
- Retry after 15 minutes if call missed
- Automatic fallback to WhatsApp text if voice fails

### 📊 Adherence Tracking
- Tracks confirmation rates
- Calculates adherence percentage
- Dashboard for healthcare providers

---

## How It Works

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Patient sets   │───▶│  WhatsApp Bot    │───▶│ Voice Reminder  │
│  reminder via   │    │  /remind command │    │ System          │
│  WhatsApp       │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Patient        │◀───│  Patient answers │◀───│ Scheduled time  │
│  confirms       │    │  phone call      │    │ triggers call   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │
        ▼
┌─────────────────┐
│ WhatsApp        │
│ confirmation    │
│ message sent    │
└─────────────────┘
```

---

## Setting Up Voice Reminders

### Via WhatsApp (Automatic)

When a patient sets a medication reminder via WhatsApp, voice calls are **automatically scheduled**:

```
Patient: /remind Paracetamol 08:00,20:00

Bot: ✅ Medication Reminder Set!

💊 Paracetamol
🕐 Times: 08:00, 20:00

I'll remind you when it's time to take your medication.

📞 Voice call reminders also set up for: 08:00, 20:00
```

A reminder names the medicine and the time, never a dose (ADR 0004, decision
D10 of the round-3 review). A `dosage` typed or sent is stored with the
reminder but is never repeated back, sent or spoken.

### Via API

Admin token required (`auth/policy.py`). A request without `phone_number`
is refused with 422 and nothing is scheduled.

```bash
POST /api/medication/voice-reminder
Content-Type: application/json

{
    "user_id": "+919876543210",
    "phone_number": "+919876543210",
    "medication_name": "Morphine",
    "dosage": "10mg with water",
    "reminder_time": "2025-01-28T14:00:00",
    "language": "hi",
    "provider": "bolna"
}
```

**Response:**
```json
{
    "status": "success",
    "reminder_id": "abc123def456",
    "scheduled_time": "2025-01-28T14:00:00"
}
```

---

## Voice Call Flow

### 1. Call Initiated
```
📞 Calling +919876543210...
```

### 2. Patient Answers
**Voice Message (English):**
> "Hello, this is Palli Sahayak, your healthcare assistant.
> 
> This is your 08:00 reminder to take Paracetamol.
> 
> Please take your medication now.
> 
> Press 1 or say 'yes' after taking your medication.
> 
> Thank you. Have a good day."

**Voice Message (Hindi):**
> "नमस्ते, मैं पल्ली सहायक हूं, आपका स्वास्थ्य सहायक।
>
> यह 08:00 बजे Paracetamol लेने का reminder है।
>
> कृपया अभी अपनी दवा लें।
>
> दवा लेने के बाद 1 दबाएं या 'हां' कहें।
>
> धन्यवाद। अपना ध्यान रखें।"

### 3. Patient Confirms
- Presses **1** on keypad, OR
- Says **"yes"** / **"हां"** / **"হ্যাঁ"**, etc.

### 4. Confirmation Recorded
- Status updated to "confirmed"
- WhatsApp confirmation message sent
- Adherence stats updated

---

## API Endpoints

### Create Voice Reminder
```bash
POST /api/medication/voice-reminder
```

### Get User's Voice Reminders
```bash
GET /api/medication/voice-reminders/{user_id}
```

### Get Adherence Stats
```bash
GET /api/medication/adherence/{user_id}?days=7
```

**Response:**
```json
{
    "user_id": "+919876543210",
    "period_days": 7,
    "total_reminders": 14,
    "confirmed": 12,
    "missed": 2,
    "adherence_rate": 85.7
}
```

### Get Pending Calls (Dashboard)
```bash
GET /api/medication/pending-calls
```

### Webhook Callback
```bash
POST /api/medication/voice-reminder/callback
```

Called by voice providers when call completes.

---

## Integration with Voice Providers

### Bolna.ai

Calls are made via Bolna's outbound calling API:

```python
from medication_voice_reminders import get_medication_voice_reminder_system

system = get_medication_voice_reminder_system()

# Bolna client is automatically used for calls
reminder = system.create_voice_reminder(
    user_id="+919876543210",
    phone_number="+919876543210",
    medication_name="Paracetamol",
    dosage="500mg",
    reminder_time=datetime(2025, 1, 28, 8, 0),
    language="hi",
    preferred_provider="bolna"
)
```

### Retell.AI

Alternative provider for voice calls:

```python
reminder = system.create_voice_reminder(
    ...,
    preferred_provider="retell"
)
```

---

## Voice Message Templates

Messages are localized for each language:

| Language | Greeting |
|----------|----------|
| **English** | "Hello, this is Palli Sahayak..." |
| **Hindi** | "नमस्ते, मैं पल्ली सहायक हूं..." |
| **Bengali** | "হ্যালো, আমি পল্লী সহায়ক..." |
| **Tamil** | "வணக்கம், நான் பல்லி சகாயக்..." |
| **Gujarati** | "નમસ્તે, હું પલ્લી સહાયક છું..." |

---

## Confirmation Methods

### DTMF (Phone Keypad)
- Press **1** = Yes, medication taken
- Press **2** = No, will take now

### Voice Recognition
| Language | "Yes" Variants |
|----------|---------------|
| English | "yes", "yeah", "taken", "done" |
| Hindi | "हां", "हाँ", "haan", "ले ली" |
| Bengali | "হ্যাঁ", "খেয়েছি" |
| Tamil | "ஆம்", "எடுத்தேன்" |
| Gujarati | "હા", "ले lidhi" |

---

## Retry Logic

```python
if call_failed or call_missed:
    if attempts < max_attempts:
        schedule_retry(delay_minutes=15)
    else:
        send_whatsapp_fallback_message()
```

**Example:**
1. 08:00 AM - Call attempted, no answer
2. 08:15 AM - Retry call
3. 08:30 AM - Final retry
4. 08:31 AM - WhatsApp message sent

---

## Dashboard & Monitoring

### Pending Calls
```bash
GET /api/medication/pending-calls
```

Shows:
- Scheduled time
- Patient phone number
- Medication name
- Call status

### Adherence Report
```
┌─────────────────────────────────────┐
│  Medication Adherence Report        │
├─────────────────────────────────────┤
│  Patient: +919876543210             │
│  Period: Last 7 days                │
│                                     │
│  Total Reminders: 14                │
│  Confirmed: 12 ✅                   │
│  Missed: 2 ⚠️                       │
│                                     │
│  Adherence Rate: 85.7%              │
└─────────────────────────────────────┘
```

---

## Configuration

### Environment Variables

```bash
# Voice providers (at least one required)
BOLNA_API_KEY=your_bolna_key
RETELL_API_KEY=your_retell_key

# Default provider
MEDICATION_VOICE_PROVIDER=bolna

# Retry settings
MEDICATION_MAX_RETRIES=3
MEDICATION_RETRY_DELAY_MINUTES=15
```

### Storage

Voice reminders stored in:
```
data/medication_voice_reminders/voice_reminders.json
```

---

## Testing

### Test Voice Reminder

```python
import asyncio
from datetime import datetime, timedelta
from medication_voice_reminders import get_medication_voice_reminder_system

async def test():
    system = get_medication_voice_reminder_system()
    
    reminder = system.create_voice_reminder(
        user_id="test_user",
        phone_number="+919876543210",
        medication_name="Test Medicine",
        dosage="1 tablet",
        reminder_time=datetime.now() + timedelta(minutes=5),
        language="en"
    )
    
    print(f"Reminder created: {reminder.reminder_id}")

asyncio.run(test())
```

### Simulate Call Completion

```bash
curl -X POST http://localhost:8000/api/medication/voice-reminder/callback \
  -H "Content-Type: application/json" \
  -d '{
    "call_id": "call_123",
    "status": "completed",
    "duration": 45,
    "patient_confirmed": true,
    "confirmation_method": "dtmf_1"
  }'
```

---

## Troubleshooting

### Calls Not Being Made
1. Check voice provider credentials (Bolna/Retell)
2. Verify phone number format (+91...)
3. Check scheduler is running

### Patient Not Receiving Calls
1. Check phone number is correct
2. Verify call status in `/api/medication/pending-calls`
3. Check retry attempts haven't exceeded max

### Confirmations Not Recording
1. Check webhook endpoint is accessible
2. Verify callback URL is configured in provider
3. Check logs for webhook errors

---

## Future Enhancements

1. **Smart Scheduling**: Adjust times based on patient response patterns
2. **Family Notification**: Alert caregivers if medication missed
3. **Refill Reminders**: Remind when medication running low
4. **Interactive Voice**: Allow patients to ask questions during call
5. **Voice Biometrics**: Verify patient identity via voice

---

**Last Updated:** January 2025  
**Status:** ✅ Production Ready
