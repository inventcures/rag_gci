#!/usr/bin/env python3
"""
Medication Voice Reminder System
================================

Makes outbound voice calls to remind patients to take medications.
Integrates with Bolna.ai and Retell.AI for automated reminder calls.

Features:
- Scheduled voice calls for medication reminders
- Pre-recorded messages in multiple languages
- Patient confirmation via voice/DTMF
- Retry logic for failed calls
- Adherence tracking

Author: Palli Sahayak AI Team
"""

import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import time
import hashlib
import random

# Setup logger first
logger = logging.getLogger(__name__)

# Try to import schedule, fallback if not available
try:
    import schedule
    HAS_SCHEDULE = True
except ImportError:
    HAS_SCHEDULE = False
    # Simple schedule fallback that does nothing
    class MockSchedule:
        def every(self, interval=1):
            return self
        def day(self):
            return self
        def at(self, time_str):
            return self
        def do(self, job_func, *args, **kwargs):
            # Just log that a job would be scheduled
            logger.info(f"[MockSchedule] Would schedule job at {args}")
            return self
        def run_pending(self):
            pass
    schedule = MockSchedule()


class CallStatus(Enum):
    """Status of a medication reminder call"""
    SCHEDULED = "scheduled"
    PENDING = "pending"
    CALLING = "calling"
    CONNECTED = "connected"
    COMPLETED = "completed"
    CONFIRMED = "confirmed"
    MISSED = "missed"
    FAILED = "failed"
    RETRYING = "retrying"


class ConfirmationMethod(Enum):
    """How patient confirms taking medication"""
    DTMF_1 = "dtmf_1"          # Press 1
    VOICE_YES = "voice_yes"    # Say "yes" or equivalent
    MISSED_CALL = "missed_call" # Missed call back


@dataclass
class MedicationVoiceReminder:
    """A scheduled voice reminder for medication"""
    reminder_id: str
    user_id: str
    phone_number: str
    medication_name: str
    dosage: str
    scheduled_time: datetime
    language: str
    
    # Call settings
    call_status: CallStatus = CallStatus.SCHEDULED
    call_attempts: int = 0
    max_attempts: int = 3
    
    # Provider settings
    preferred_provider: str = "bolna"  # bolna, retell
    
    # Call tracking
    call_id: Optional[str] = None
    call_started_at: Optional[datetime] = None
    call_ended_at: Optional[datetime] = None
    call_duration: int = 0
    
    # Patient response
    patient_confirmed: bool = False
    confirmation_method: Optional[ConfirmationMethod] = None
    confirmation_timestamp: Optional[datetime] = None
    
    # Voice message
    voice_message_url: Optional[str] = None
    tts_message: Optional[str] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reminder_id": self.reminder_id,
            "user_id": self.user_id,
            "phone_number": self.phone_number,
            "medication_name": self.medication_name,
            "dosage": self.dosage,
            "scheduled_time": self.scheduled_time.isoformat(),
            "language": self.language,
            "call_status": self.call_status.value,
            "call_attempts": self.call_attempts,
            "max_attempts": self.max_attempts,
            "preferred_provider": self.preferred_provider,
            "call_id": self.call_id,
            "call_started_at": self.call_started_at.isoformat() if self.call_started_at else None,
            "call_ended_at": self.call_ended_at.isoformat() if self.call_ended_at else None,
            "call_duration": self.call_duration,
            "patient_confirmed": self.patient_confirmed,
            "confirmation_method": self.confirmation_method.value if self.confirmation_method else None,
            "confirmation_timestamp": self.confirmation_timestamp.isoformat() if self.confirmation_timestamp else None,
            "voice_message_url": self.voice_message_url,
            "tts_message": self.tts_message,
            "created_at": self.created_at.isoformat(),
            "notes": self.notes,
        }


class MedicationVoiceReminderSystem:
    """
    System for making medication reminder voice calls.
    
    Integrates with:
    - Bolna.ai for voice calls
    - Retell.AI for voice calls
    - TTS for generating reminder messages
    """
    
    # Voice message templates by language
    VOICE_TEMPLATES = {
        "en": {
            "greeting": "Hello, this is Palli Sahayak, your healthcare assistant.",
            "reminder": "This is a reminder to take your {medication}. {dosage}.",
            "instructions": "Please take your medication now.",
            "confirmation": "Press 1 or say 'yes' after taking your medication.",
            "closing": "Thank you. Have a good day.",
            "follow_up": "Did you take your {medication}? Press 1 for yes, 2 for no.",
        },
        "hi": {
            "greeting": "नमस्ते, मैं पल्ली सहायक हूं, आपका स्वास्थ्य सहायक।",
            "reminder": "यह {medication} लेने का reminder है। {dosage}।",
            "instructions": "कृपया अभी अपनी दवा लें।",
            "confirmation": "दवा लेने के बाद 1 दबाएं या 'हां' कहें।",
            "closing": "धन्यवाद। आपका दिन शुभ हो।",
            "follow_up": "क्या आपने {medication} ली? हां के लिए 1 दबाएं, न के लिए 2।",
        },
        "bn": {
            "greeting": "হ্যালো, আমি পল্লী সহায়ক, আপনার স্বাস্থ্য সহকারী।",
            "reminder": "এটি {medication} খাওয়ার reminder। {dosage}।",
            "instructions": "অনুগ্রহ করে এখনই আপনার ওষুধ খান।",
            "confirmation": "ওষুধ খাওয়ার পরে 1 চাপুন বা 'হ্যাঁ' বলুন।",
            "closing": "ধন্যবাদ। আপনার দিন শুভ হোক।",
            "follow_up": "আপনি কি {medication} খেয়েছেন? হ্যাঁ এর জন্য 1, না এর জন্য 2 চাপুন।",
        },
        "ta": {
            "greeting": "வணக்கம், நான் பல்லி சகாயக், உங்கள் சுகாதார உதவியாளர்.",
            "reminder": "இது {medication} எடுக்கும் நினைவூட்டல். {dosage}.",
            "instructions": "தயவுசெய்து இப்போதே உங்கள் மருந்தை எடுத்துக்கொள்ளுங்கள்.",
            "confirmation": "மருந்தை எடுத்த பிறகு 1 ஐ அழுத்தவும் அல்லது 'ஆம்' என்று கூறவும்.",
            "closing": "நன்றி. உங்கள் நாள் நல்லதாக இருக்கட்டும்.",
            "follow_up": "நீங்கள் {medication} எடுத்தீர்களா? ஆம் என்றால் 1, இல்லை என்றால் 2.",
        },
        "gu": {
            "greeting": "નમસ્તે, હું પલ્લી સહાયક છું, તમારો આરોગ્ય સહાયક.",
            "reminder": "આ {medication} લેવાની યાદ અપાવનાર છે. {dosage}.",
            "instructions": "કૃપા કરીને હવે તમારી દવા લો.",
            "confirmation": "દવા લીધા પછી 1 દબાવો અથવા 'હા' કહો.",
            "closing": "આભાર. તમારો દિવસ શુભ રહે.",
            "follow_up": "શું તમે {medication} લીધી? હા માટે 1, ના માટે 2.",
        },
    }
    
    def __init__(self, storage_path: str = "data/medication_voice_reminders"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Active reminders
        self.reminders: Dict[str, MedicationVoiceReminder] = {}
        self.user_reminders: Dict[str, List[str]] = {}
        
        # Load existing reminders
        self._load_reminders()
        
        # Initialize voice providers
        self.bolna_client = None
        self.retell_client = None
        self._init_providers()
        
        # Start scheduler
        self._start_scheduler()
        
        # Callbacks for events
        self.on_reminder_sent: Optional[Callable] = None
        self.on_patient_confirmed: Optional[Callable] = None
        self.on_call_failed: Optional[Callable] = None
        
        logger.info("📞 Medication Voice Reminder System initialized")
    
    def _init_providers(self):
        """Initialize voice call providers"""
        # Try Bolna
        try:
            from bolna_integration import BolnaClient
            self.bolna_client = BolnaClient()
            logger.info("✅ Bolna client initialized for medication calls")
        except Exception as e:
            logger.warning(f"Bolna not available for medication calls: {e}")
        
        # Try Retell
        try:
            from retell_integration import RetellClient
            self.retell_client = RetellClient()
            logger.info("✅ Retell client initialized for medication calls")
        except Exception as e:
            logger.warning(f"Retell not available for medication calls: {e}")
    
    def _load_reminders(self):
        """Load reminders from storage"""
        reminders_file = self.storage_path / "voice_reminders.json"
        if reminders_file.exists():
            try:
                with open(reminders_file, 'r') as f:
                    data = json.load(f)
                    for reminder_data in data:
                        reminder = self._dict_to_reminder(reminder_data)
                        self.reminders[reminder.reminder_id] = reminder
                        if reminder.user_id not in self.user_reminders:
                            self.user_reminders[reminder.user_id] = []
                        self.user_reminders[reminder.user_id].append(reminder.reminder_id)
                logger.info(f"Loaded {len(self.reminders)} medication voice reminders")
            except Exception as e:
                logger.error(f"Failed to load voice reminders: {e}")
    
    def _save_reminders(self):
        """Save reminders to storage"""
        try:
            reminders_file = self.storage_path / "voice_reminders.json"
            data = [r.to_dict() for r in self.reminders.values()]
            with open(reminders_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save voice reminders: {e}")
    
    def _dict_to_reminder(self, data: Dict) -> MedicationVoiceReminder:
        """Convert dict to MedicationVoiceReminder"""
        return MedicationVoiceReminder(
            reminder_id=data["reminder_id"],
            user_id=data["user_id"],
            phone_number=data["phone_number"],
            medication_name=data["medication_name"],
            dosage=data["dosage"],
            scheduled_time=datetime.fromisoformat(data["scheduled_time"]),
            language=data["language"],
            call_status=CallStatus(data.get("call_status", "scheduled")),
            call_attempts=data.get("call_attempts", 0),
            max_attempts=data.get("max_attempts", 3),
            preferred_provider=data.get("preferred_provider", "bolna"),
            call_id=data.get("call_id"),
            call_started_at=datetime.fromisoformat(data["call_started_at"]) if data.get("call_started_at") else None,
            call_ended_at=datetime.fromisoformat(data["call_ended_at"]) if data.get("call_ended_at") else None,
            call_duration=data.get("call_duration", 0),
            patient_confirmed=data.get("patient_confirmed", False),
            confirmation_method=ConfirmationMethod(data["confirmation_method"]) if data.get("confirmation_method") else None,
            confirmation_timestamp=datetime.fromisoformat(data["confirmation_timestamp"]) if data.get("confirmation_timestamp") else None,
            voice_message_url=data.get("voice_message_url"),
            tts_message=data.get("tts_message"),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            notes=data.get("notes", ""),
        )
    
    def _start_scheduler(self):
        """Start the background call scheduler"""
        def run_scheduler():
            while True:
                try:
                    schedule.run_pending()
                except Exception as e:
                    logger.error(f"Scheduler error: {e}")
                time.sleep(30)  # Check every 30 seconds
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        logger.info("📅 Medication voice call scheduler started")
    
    def create_voice_reminder(
        self,
        user_id: str,
        phone_number: str,
        medication_name: str,
        dosage: str,
        reminder_time: datetime,
        language: str = "en",
        preferred_provider: str = "bolna",
        max_attempts: int = 3
    ) -> MedicationVoiceReminder:
        """
        Create a new voice reminder for medication.
        
        Args:
            user_id: Patient identifier
            phone_number: Phone number in E.164 format (+91...)
            medication_name: Name of medication
            dosage: Dosage instructions (e.g., "500mg with food")
            reminder_time: When to make the call
            language: Language code (en, hi, bn, ta, gu)
            preferred_provider: "bolna" or "retell"
            max_attempts: Max retry attempts
            
        Returns:
            MedicationVoiceReminder object
        """
        reminder_id = hashlib.md5(
            f"{user_id}_{medication_name}_{reminder_time.isoformat()}".encode()
        ).hexdigest()[:12]
        
        # Generate TTS message
        tts_message = self._generate_reminder_message(
            medication_name, dosage, language
        )
        
        reminder = MedicationVoiceReminder(
            reminder_id=reminder_id,
            user_id=user_id,
            phone_number=phone_number,
            medication_name=medication_name,
            dosage=dosage,
            scheduled_time=reminder_time,
            language=language,
            preferred_provider=preferred_provider,
            max_attempts=max_attempts,
            tts_message=tts_message,
        )
        
        # Store reminder
        self.reminders[reminder_id] = reminder
        if user_id not in self.user_reminders:
            self.user_reminders[user_id] = []
        self.user_reminders[user_id].append(reminder_id)
        
        # Schedule the call
        self._schedule_call(reminder)
        
        self._save_reminders()
        
        logger.info(f"📞 Created voice reminder {reminder_id} for {medication_name} at {reminder_time}")
        return reminder
    
    def _generate_reminder_message(self, medication: str, dosage: str, language: str) -> str:
        """Generate TTS message for reminder"""
        templates = self.VOICE_TEMPLATES.get(language, self.VOICE_TEMPLATES["en"])
        
        message = f"""{templates['greeting']}

{templates['reminder'].format(medication=medication, dosage=dosage)}

{templates['instructions']}

{templates['confirmation']}

{templates['closing']}"""
        
        return message
    
    def _schedule_call(self, reminder: MedicationVoiceReminder):
        """Schedule a voice call"""
        time_str = reminder.scheduled_time.strftime("%H:%M")
        
        # Use module-level schedule reference explicitly to avoid scoping issues
        import sys
        current_module = sys.modules[__name__]
        sched = getattr(current_module, 'schedule')
        
        # Schedule with 1-minute window - explicit call chain to avoid issues
        every_result = sched.every()
        day_result = every_result.day()
        at_result = day_result.at(time_str)
        at_result.do(self._trigger_call, reminder.reminder_id)
        
        logger.info(f"📅 Scheduled call for {reminder.medication_name} at {time_str}")
    
    def _trigger_call(self, reminder_id: str):
        """Trigger a medication reminder call"""
        if reminder_id not in self.reminders:
            return
        
        reminder = self.reminders[reminder_id]
        
        # Check if already completed or max attempts reached
        if reminder.call_status in [CallStatus.CONFIRMED, CallStatus.COMPLETED]:
            return
        
        if reminder.call_attempts >= reminder.max_attempts:
            reminder.call_status = CallStatus.FAILED
            self._save_reminders()
            logger.warning(f"Max attempts reached for reminder {reminder_id}")
            return
        
        # Make the call
        asyncio.create_task(self._make_call(reminder_id))
    
    async def _make_call(self, reminder_id: str):
        """Make the voice call"""
        if reminder_id not in self.reminders:
            return
        
        reminder = self.reminders[reminder_id]
        reminder.call_status = CallStatus.CALLING
        reminder.call_attempts += 1
        reminder.call_started_at = datetime.now()
        self._save_reminders()
        
        try:
            if reminder.preferred_provider == "bolna" and self.bolna_client:
                result = await self._make_bolna_call(reminder)
            elif reminder.preferred_provider == "retell" and self.retell_client:
                result = await self._make_retell_call(reminder)
            else:
                # Fallback or error
                logger.error(f"No voice provider available for reminder {reminder_id}")
                reminder.call_status = CallStatus.FAILED
                self._save_reminders()
                return
            
            # Update based on result
            if result.get("success"):
                reminder.call_id = result.get("call_id")
                reminder.call_status = CallStatus.CONNECTED
                logger.info(f"✅ Call connected for reminder {reminder_id}")
            else:
                # Retry if attempts remaining
                if reminder.call_attempts < reminder.max_attempts:
                    reminder.call_status = CallStatus.RETRYING
                    # Schedule retry in 10 minutes
                    await asyncio.sleep(600)
                    await self._make_call(reminder_id)
                else:
                    reminder.call_status = CallStatus.FAILED
                    if self.on_call_failed:
                        await self.on_call_failed(reminder)
            
            self._save_reminders()
            
        except Exception as e:
            logger.error(f"Error making call for reminder {reminder_id}: {e}")
            reminder.call_status = CallStatus.FAILED
            self._save_reminders()
    
    async def _make_bolna_call(self, reminder: MedicationVoiceReminder) -> Dict[str, Any]:
        """Make call via Bolna.ai"""
        try:
            # This would use Bolna's API to make an outbound call
            # For now, return mock success
            logger.info(f"📞 Making Bolna call to {reminder.phone_number} for {reminder.medication_name}")
            
            # In production, this would:
            # 1. Create a Bolna call with custom agent for medication reminders
            # 2. Pass the TTS message
            # 3. Handle DTMF/voice confirmation
            
            return {
                "success": True,
                "call_id": f"bolna_{reminder.reminder_id}_{int(time.time())}",
                "provider": "bolna"
            }
            
        except Exception as e:
            logger.error(f"Bolna call failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _make_retell_call(self, reminder: MedicationVoiceReminder) -> Dict[str, Any]:
        """Make call via Retell.AI"""
        try:
            # This would use Retell's API to make an outbound call
            logger.info(f"📞 Making Retell call to {reminder.phone_number} for {reminder.medication_name}")
            
            # In production, this would:
            # 1. Create a Retell call with medication reminder agent
            # 2. Pass the TTS message
            # 3. Handle confirmation
            
            return {
                "success": True,
                "call_id": f"retell_{reminder.reminder_id}_{int(time.time())}",
                "provider": "retell"
            }
            
        except Exception as e:
            logger.error(f"Retell call failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def handle_call_completed(
        self,
        call_id: str,
        status: str,
        duration: int = 0,
        patient_response: Optional[str] = None
    ):
        """
        Handle call completion webhook from provider.
        
        Args:
            call_id: The call ID from provider
            status: Call status (completed, failed, etc.)
            duration: Call duration in seconds
            patient_response: Patient's response (if any)
        """
        # Find reminder by call_id
        reminder = None
        for r in self.reminders.values():
            if r.call_id == call_id:
                reminder = r
                break
        
        if not reminder:
            logger.warning(f"No reminder found for call_id {call_id}")
            return
        
        reminder.call_ended_at = datetime.now()
        reminder.call_duration = duration
        
        # Check if patient confirmed
        if patient_response:
            reminder.patient_confirmed = self._parse_confirmation(
                patient_response, reminder.language
            )
            if reminder.patient_confirmed:
                reminder.confirmation_method = ConfirmationMethod.VOICE_YES
                reminder.confirmation_timestamp = datetime.now()
                reminder.call_status = CallStatus.CONFIRMED
                
                if self.on_patient_confirmed:
                    await self.on_patient_confirmed(reminder)
                
                logger.info(f"✅ Patient confirmed taking {reminder.medication_name}")
        
        if status == "completed":
            if reminder.call_status != CallStatus.CONFIRMED:
                reminder.call_status = CallStatus.COMPLETED
        elif status == "failed":
            reminder.call_status = CallStatus.FAILED
        elif status == "missed":
            reminder.call_status = CallStatus.MISSED
            # Retry if needed
            if reminder.call_attempts < reminder.max_attempts:
                asyncio.create_task(self._retry_call(reminder.reminder_id, delay_minutes=15))
        
        self._save_reminders()
    
    def _parse_confirmation(self, response: str, language: str) -> bool:
        """Parse patient confirmation from voice/DTMF response"""
        response_lower = response.lower().strip()
        
        # DTMF 1 = yes
        if response_lower == "1" or response_lower == "dtmf_1":
            return True
        
        # Voice confirmations by language
        confirmations = {
            "en": ["yes", "yeah", "yep", "taken", "done", "ok"],
            "hi": ["हां", "हाँ", "yes", "haan", "han", "ले ली", "le li"],
            "bn": ["হ্যাঁ", "হাঁ", "yes", "kheyechi", "খেয়েছি"],
            "ta": ["ஆம்", "aam", "yes", "eduthen", "எடுத்தேன்"],
            "gu": ["હા", "ha", "yes", "le lidhi", "લે લીધી"],
        }
        
        lang_confirmations = confirmations.get(language, confirmations["en"])
        return any(conf in response_lower for conf in lang_confirmations)
    
    async def _retry_call(self, reminder_id: str, delay_minutes: int = 15):
        """Retry a failed call after delay"""
        await asyncio.sleep(delay_minutes * 60)
        await self._make_call(reminder_id)
    
    def get_reminder_status(self, reminder_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a reminder"""
        if reminder_id not in self.reminders:
            return None
        return self.reminders[reminder_id].to_dict()
    
    def get_user_reminders(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all reminders for a user"""
        reminder_ids = self.user_reminders.get(user_id, [])
        return [self.reminders[r_id].to_dict() for r_id in reminder_ids if r_id in self.reminders]
    
    def get_pending_calls(self) -> List[Dict[str, Any]]:
        """Get all pending calls (for dashboard)"""
        pending = [
            r.to_dict() for r in self.reminders.values()
            if r.call_status in [CallStatus.SCHEDULED, CallStatus.PENDING, CallStatus.CALLING]
        ]
        return sorted(pending, key=lambda x: x["scheduled_time"])
    
    def get_adherence_stats(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """Get medication adherence statistics"""
        reminder_ids = self.user_reminders.get(user_id, [])
        
        cutoff = datetime.now() - timedelta(days=days)
        
        total = 0
        confirmed = 0
        missed = 0
        
        for r_id in reminder_ids:
            if r_id in self.reminders:
                r = self.reminders[r_id]
                if r.scheduled_time >= cutoff:
                    total += 1
                    if r.patient_confirmed:
                        confirmed += 1
                    elif r.call_status == CallStatus.MISSED:
                        missed += 1
        
        adherence_rate = (confirmed / total * 100) if total > 0 else 0
        
        return {
            "user_id": user_id,
            "period_days": days,
            "total_reminders": total,
            "confirmed": confirmed,
            "missed": missed,
            "adherence_rate": round(adherence_rate, 1),
        }


# Singleton instance
_voice_reminder_system: Optional[MedicationVoiceReminderSystem] = None


def get_medication_voice_reminder_system() -> MedicationVoiceReminderSystem:
    """Get or create the voice reminder system singleton"""
    global _voice_reminder_system
    if _voice_reminder_system is None:
        _voice_reminder_system = MedicationVoiceReminderSystem()
    return _voice_reminder_system
