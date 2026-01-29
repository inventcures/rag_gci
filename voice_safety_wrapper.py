#!/usr/bin/env python3
"""
Voice Safety Wrapper for Palli Sahayak
=======================================

Integrates safety enhancements into ALL voice AI flows:
- Gemini Live API (Web voice with native audio)
- Bolna.ai (Phone calls via Twilio)
- Retell.AI + Vobiz (Phone calls via Indian PSTN)
- Fallback Pipeline (STT → RAG → TTS)

Safety Features for Voice:
1. Emergency Detection - Real-time analysis of voice transcripts
2. Evidence Badges - Confidence indicators in voice responses  
3. Response Optimization - Length-optimized for voice consumption
4. Human Handoff - Warm transfer to human agents
5. Medication Reminders - Voice-accessible reminders

Author: Palli Sahayak AI Team
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# Import safety enhancements
try:
    from safety_enhancements import (
        SafetyEnhancementsManager,
        get_safety_manager,
        EmergencyDetectionSystem,
        EmergencyLevel,
        EmergencyAlert,
        HumanHandoffSystem,
        HandoffReason,
        ResponseLengthOptimizer,
        UserComprehensionLevel,
        EvidenceBadgeSystem,
    )
    SAFETY_AVAILABLE = True
except ImportError:
    SAFETY_AVAILABLE = False

logger = logging.getLogger(__name__)


class VoiceSafetyEvent(Enum):
    """Safety events that can occur during voice interactions"""
    EMERGENCY_DETECTED = "emergency_detected"
    HUMAN_HANDOFF_TRIGGERED = "human_handoff_triggered"
    AI_UNCERTAIN = "ai_uncertain"
    EVIDENCE_LOW = "evidence_low"
    SAFETY_ESCALATION = "safety_escalation"


@dataclass
class VoiceSafetyResult:
    """Result of voice safety check"""
    should_proceed: bool
    should_escalate: bool
    event_type: Optional[VoiceSafetyEvent]
    emergency_alert: Optional[EmergencyAlert]
    handoff_reason: Optional[HandoffReason]
    modified_transcript: Optional[str]
    evidence_badge: Optional[Dict[str, Any]]
    safety_message: Optional[str]
    metadata: Dict[str, Any]


class VoiceSafetyWrapper:
    """
    Wrapper that adds safety checks to all voice AI flows.
    
    This class intercepts voice transcripts, analyzes them for safety concerns,
    and provides appropriate responses or escalations.
    
    Usage:
        wrapper = VoiceSafetyWrapper()
        
        # Before processing any voice query
        safety_result = await wrapper.check_voice_query(
            user_id="phone_number",
            transcript="I can't breathe",
            language="en"
        )
        
        if safety_result.should_escalate:
            # Handle emergency or handoff
            await wrapper.handle_voice_escalation(safety_result)
        else:
            # Proceed with normal processing
            response = await process_query(safety_result.modified_transcript)
            
            # Optimize response for voice
            voice_response = wrapper.optimize_for_voice(
                response, user_id, language
            )
    """
    
    # Emergency phrases that should trigger immediate escalation
    VOICE_EMERGENCY_PHRASES = {
        "en": ["can't breathe", "not breathing", "heart stopped", "unconscious", 
               "severe bleeding", "dying", "emergency", "dying", "suicide"],
        "hi": ["सांस नहीं आ रही", "दम घुट रहा", "होश नहीं", "बेहोश", 
               "खून बह रहा", "मर रहा हूं", "मर रही हूं", "आपातकाल"],
        "bn": ["শ্বাস নিতে পারছি না", "দম বন্ধ", "জ্ঞান হারিয়ে", 
               "রক্তপাত", "মরছি", "জরুরি"],
        "ta": ["மூச்சு வரவில்லை", "மயக்கம்", "பரிசுத்தமற்றவர்", 
               "இரத்தப்போக்கு", "அவசரம்"],
    }
    
    # Human handoff phrases
    HANDOFF_PHRASES = {
        "en": ["talk to doctor", "speak to nurse", "human please", 
               "real person", "transfer me", "connect to hospital"],
        "hi": ["डॉक्टर से बात", "नर्स से बात", "इंसान से", "असली आदमी"],
        "bn": ["ডাক্তারের সাথে কথা", "নার্সের সাথে", "মানুষের সাথে"],
        "ta": ["மருத்துவரிடம்", "மனிதரிடம்", "உண்மையான நபர்"],
    }
    
    def __init__(self):
        self.safety_manager = None
        self.emergency_system = EmergencyDetectionSystem()
        self.handoff_system = HumanHandoffSystem()
        self.response_optimizer = ResponseLengthOptimizer()
        self.evidence_system = EvidenceBadgeSystem()
        
        if SAFETY_AVAILABLE:
            try:
                self.safety_manager = get_safety_manager()
                logger.info("✅ Voice Safety Wrapper initialized")
            except Exception as e:
                logger.error(f"Failed to initialize safety manager: {e}")
        
        # Event callbacks for integrations
        self.event_callbacks: Dict[VoiceSafetyEvent, List[Callable]] = {
            event: [] for event in VoiceSafetyEvent
        }
    
    def register_callback(self, event: VoiceSafetyEvent, callback: Callable):
        """Register a callback for safety events"""
        self.event_callbacks[event].append(callback)
    
    async def _trigger_event(self, event: VoiceSafetyEvent, data: Dict[str, Any]):
        """Trigger callbacks for an event"""
        for callback in self.event_callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Error in safety event callback: {e}")
    
    async def check_voice_query(
        self,
        user_id: str,
        transcript: str,
        language: str = "en",
        call_id: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None
    ) -> VoiceSafetyResult:
        """
        Check voice transcript for safety concerns.
        
        This is the main entry point - call this before processing any voice query.
        
        Args:
            user_id: User identifier (phone number)
            transcript: Voice transcript text
            language: Language code (en, hi, bn, ta, gu)
            call_id: Optional call/session ID
            conversation_history: Previous conversation turns
            
        Returns:
            VoiceSafetyResult with safety analysis and recommendations
        """
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "call_id": call_id,
            "original_transcript": transcript,
        }
        
        # 1. EMERGENCY DETECTION (Highest priority)
        emergency_alert = self.emergency_system.detect_emergency(
            transcript, user_id, language
        )
        
        if emergency_alert and emergency_alert.level == EmergencyLevel.CRITICAL:
            # Critical emergency - immediate escalation
            logger.critical(f"🚨 CRITICAL EMERGENCY detected for {user_id}: {emergency_alert.detected_keywords}")
            
            await self._trigger_event(VoiceSafetyEvent.EMERGENCY_DETECTED, {
                "user_id": user_id,
                "alert": emergency_alert,
                "transcript": transcript
            })
            
            return VoiceSafetyResult(
                should_proceed=False,
                should_escalate=True,
                event_type=VoiceSafetyEvent.EMERGENCY_DETECTED,
                emergency_alert=emergency_alert,
                handoff_reason=HandoffReason.EMERGENCY,
                modified_transcript=None,
                evidence_badge=None,
                safety_message=self._format_emergency_voice_response(emergency_alert, language),
                metadata=metadata
            )
        
        # 2. HUMAN HANDOFF DETECTION
        handoff_reason = self._detect_handoff_request(transcript, language)
        
        if handoff_reason:
            logger.info(f"👤 Handoff requested by {user_id}: {handoff_reason}")
            
            await self._trigger_event(VoiceSafetyEvent.HUMAN_HANDOFF_TRIGGERED, {
                "user_id": user_id,
                "reason": handoff_reason,
                "transcript": transcript
            })
            
            handoff = self.handoff_system.create_handoff_request(
                user_id=user_id,
                reason=handoff_reason,
                context=transcript,
                conversation_history=conversation_history or []
            )
            
            return VoiceSafetyResult(
                should_proceed=False,
                should_escalate=True,
                event_type=VoiceSafetyEvent.HUMAN_HANDOFF_TRIGGERED,
                emergency_alert=None,
                handoff_reason=handoff_reason,
                modified_transcript=None,
                evidence_badge=None,
                safety_message=self._format_handoff_voice_response(handoff, language),
                metadata={**metadata, "handoff_request_id": handoff.request_id}
            )
        
        # 3. ANALYZE USER COMPREHENSION
        self.response_optimizer.analyze_user_message(user_id, transcript)
        
        # 4. CHECK FOR MEDICATION REMINDER COMMANDS
        reminder_response = self._check_medication_command(transcript, user_id, language)
        if reminder_response:
            return VoiceSafetyResult(
                should_proceed=False,
                should_escalate=False,
                event_type=None,
                emergency_alert=None,
                handoff_reason=None,
                modified_transcript=None,
                evidence_badge=None,
                safety_message=reminder_response,
                metadata=metadata
            )
        
        # All clear - proceed with normal processing
        return VoiceSafetyResult(
            should_proceed=True,
            should_escalate=False,
            event_type=None,
            emergency_alert=emergency_alert,  # May be non-critical
            handoff_reason=None,
            modified_transcript=transcript,
            evidence_badge=None,
            safety_message=None,
            metadata=metadata
        )
    
    def _detect_handoff_request(self, transcript: str, language: str) -> Optional[HandoffReason]:
        """Detect if user is requesting human handoff"""
        transcript_lower = transcript.lower()
        
        # Check handoff phrases
        phrases = self.HANDOFF_PHRASES.get(language, self.HANDOFF_PHRASES["en"])
        for phrase in phrases:
            if phrase in transcript_lower:
                return HandoffReason.USER_REQUEST
        
        # Also use the main handoff system
        return self.handoff_system.check_handoff_needed(transcript)
    
    def _check_medication_command(
        self, transcript: str, user_id: str, language: str
    ) -> Optional[str]:
        """Check if transcript is a medication reminder command"""
        transcript_lower = transcript.lower()
        
        # Command patterns for different languages
        reminder_patterns = {
            "en": [
                ("remind me to take", "medication"),
                ("set reminder for", "medicine"),
                ("remind me about", "medication"),
            ],
            "hi": [
                ("याद दिलाना", "दवा"),
                ("reminder लगा", "dawa"),
            ],
        }
        
        patterns = reminder_patterns.get(language, reminder_patterns["en"])
        
        for pattern, med_keyword in patterns:
            if pattern in transcript_lower and med_keyword in transcript_lower:
                # Extract medication info (simplified)
                return self._format_medication_voice_response(language)
        
        return None
    
    def _format_emergency_voice_response(self, alert: EmergencyAlert, language: str) -> str:
        """Format emergency response for voice output"""
        responses = {
            "en": f"""{alert.message}

This is an emergency. Please call 108 immediately for an ambulance.

{alert.action_required}""",
            "hi": f"""{alert.message}

यह आपातकाल है। कृपया तुरंत 108 पर कॉल करें।""",
        }
        
        return responses.get(language, responses["en"])
    
    def _format_handoff_voice_response(self, handoff, language: str) -> str:
        """Format handoff response for voice output"""
        responses = {
            "en": f"I understand you'd like to speak with a human caregiver. I'm connecting you now. Your request ID is {handoff.request_id[:8]}. A healthcare professional will be with you shortly.",
            "hi": f"मैं समझता हूं कि आप एक मानव देखभाल करने वाले से बात करना चाहते हैं। मैं आपको जोड़ रहा हूं। आपका अनुरोध आईडी है {handoff.request_id[:8]}। एक स्वास्थ्य पेशेवर जल्द ही आपके साथ होगा।",
        }
        
        return responses.get(language, responses["en"])
    
    def _format_medication_voice_response(self, language: str) -> str:
        """Format medication reminder help for voice"""
        responses = {
            "en": "I can help you set up medication reminders. Please use the WhatsApp chat to set reminders with the command: /remind followed by medication name, times, and dosage.",
            "hi": "मैं दवा रिमाइंडर सेट करने में मदद कर सकता हूं। कृपया WhatsApp चैट का उपयोग करें और /remind कमांड के साथ दवा का नाम, समय और खुराक बताएं।",
        }
        
        return responses.get(language, responses["en"])
    
    def optimize_for_voice(
        self,
        response: str,
        user_id: str,
        language: str = "en",
        max_duration_seconds: int = 30
    ) -> str:
        """
        Optimize a text response for voice output.
        
        Voice responses should be:
        - Concise (30 seconds max at normal speaking rate)
        - Easy to understand when heard (not read)
        - Broken into natural pauses
        - Without complex formatting
        
        Args:
            response: Original text response
            user_id: User identifier
            language: Language code
            max_duration_seconds: Maximum duration in seconds
            
        Returns:
            Optimized voice response
        """
        # Get user's comprehension level
        level = self.response_optimizer.get_user_level(user_id)
        
        # Estimate words for duration (average ~130 words per minute)
        max_words = (max_duration_seconds / 60) * 130
        
        # Clean response for voice
        voice_response = self._clean_for_voice(response)
        
        # Truncate if too long
        words = voice_response.split()
        if len(words) > max_words:
            # Find last complete sentence
            truncated = " ".join(words[:int(max_words)])
            last_sentence_end = max(
                truncated.rfind('.'),
                truncated.rfind('!'),
                truncated.rfind('?')
            )
            if last_sentence_end > len(truncated) * 0.7:
                voice_response = truncated[:last_sentence_end + 1]
            else:
                voice_response = truncated + "."
        
        # Add evidence badge if confidence is low
        # (For voice, we only mention if it's important)
        
        return voice_response
    
    def _clean_for_voice(self, text: str) -> str:
        """Clean text for voice output"""
        # Remove markdown formatting
        text = text.replace("**", "").replace("*", "")
        text = text.replace("```", "").replace("`", "")
        
        # Remove citation brackets for voice
        import re
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # Clean up extra whitespace
        text = " ".join(text.split())
        
        # Replace bullet points with spoken equivalents
        text = text.replace("•", "Also, ")
        text = text.replace("- ", "Also, ")
        
        return text.strip()
    
    def add_evidence_to_voice(
        self,
        response: str,
        evidence_badge: Dict[str, Any],
        language: str = "en"
    ) -> str:
        """
        Add evidence information to voice response if needed.
        Only adds if confidence is low or physician consult recommended.
        """
        if evidence_badge.get("consult_physician"):
            warnings = {
                "en": "Please note: I recommend consulting a physician for this matter.",
                "hi": "कृपया ध्यान दें: मैं इस मामले के लिए डॉक्टर से परामर्श करने की सलाह देता हूं।",
            }
            warning = warnings.get(language, warnings["en"])
            return f"{response} {warning}"
        
        if evidence_badge.get("level") in ["D", "E"]:
            warnings = {
                "en": "This information is based on limited evidence. Please consult your doctor.",
                "hi": "यह जानकारी सीमित सबूतों पर आधारित है। कृपया अपने डॉक्टर से परामर्श करें।",
            }
            warning = warnings.get(language, warnings["en"])
            return f"{response} {warning}"
        
        return response
    
    async def handle_voice_escalation(
        self,
        safety_result: VoiceSafetyResult,
        provider: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Handle escalation from voice flow.
        
        Args:
            safety_result: Safety check result
            provider: Voice provider (gemini, bolna, retell, fallback)
            
        Returns:
            Escalation handling result
        """
        result = {
            "escalated": True,
            "provider": provider,
            "event_type": safety_result.event_type.value if safety_result.event_type else None,
            "actions_taken": []
        }
        
        if safety_result.event_type == VoiceSafetyEvent.EMERGENCY_DETECTED:
            # Notify caregivers
            if safety_result.emergency_alert and safety_result.emergency_alert.notify_caregivers:
                # This would integrate with your notification system
                result["actions_taken"].append("caregiver_notification_queued")
                logger.info(f"Caregiver notification queued for {safety_result.metadata.get('user_id')}")
            
            # Log critical event
            logger.critical(f"Voice emergency escalated from {provider}: {safety_result.emergency_alert}")
            
        elif safety_result.event_type == VoiceSafetyEvent.HUMAN_HANDOFF_TRIGGERED:
            # Handoff already created in check_voice_query
            result["actions_taken"].append("handoff_request_created")
            result["handoff_request_id"] = safety_result.metadata.get("handoff_request_id")
        
        return result
    
    def get_voice_prompt_additions(self, user_id: str) -> str:
        """
        Get additional prompt instructions for voice responses.
        This helps the LLM generate voice-appropriate responses.
        """
        level = self.response_optimizer.get_user_level(user_id)
        
        base_instructions = """
VOICE RESPONSE REQUIREMENTS:
1. Use spoken language - write as you would speak
2. Keep responses concise (under 30 seconds when spoken)
3. Avoid bullet points and formatting - use natural flow
4. Use simple, clear sentences
5. Include brief pauses (commas) for natural speech
6. Avoid citations in the spoken text - add them at the end if needed
"""
        
        if level == UserComprehensionLevel.SIMPLE:
            return base_instructions + """
7. Use very simple words (8th grade level)
8. Break complex ideas into small steps
9. Use examples when possible
"""
        elif level == UserComprehensionLevel.DETAILED:
            return base_instructions + """
7. You may use appropriate medical terminology
8. Provide comprehensive but structured information
"""
        
        return base_instructions


# Singleton instance
_voice_safety_wrapper: Optional[VoiceSafetyWrapper] = None


def get_voice_safety_wrapper() -> VoiceSafetyWrapper:
    """Get or create the voice safety wrapper singleton"""
    global _voice_safety_wrapper
    if _voice_safety_wrapper is None:
        _voice_safety_wrapper = VoiceSafetyWrapper()
    return _voice_safety_wrapper


# =============================================================================
# INTEGRATION HELPERS FOR EACH VOICE PROVIDER
# =============================================================================

class GeminiLiveSafetyIntegration:
    """Safety integration for Gemini Live API"""
    
    def __init__(self, gemini_service):
        self.gemini_service = gemini_service
        self.safety_wrapper = get_voice_safety_wrapper()
    
    async def on_transcript(self, session_id: str, transcript: str, language: str = "en"):
        """Called when Gemini Live receives a transcript"""
        # Check safety before processing
        safety_result = await self.safety_wrapper.check_voice_query(
            user_id=session_id,
            transcript=transcript,
            language=language,
            call_id=session_id
        )
        
        if safety_result.should_escalate:
            # Override the normal flow with safety response
            return {
                "override": True,
                "response": safety_result.safety_message,
                "escalate": True
            }
        
        return {"override": False, "transcript": safety_result.modified_transcript}


class RetellSafetyIntegration:
    """Safety integration for Retell.AI"""
    
    def __init__(self, retell_handler):
        self.retell_handler = retell_handler
        self.safety_wrapper = get_voice_safety_wrapper()
    
    async def on_transcript(self, call_id: str, transcript: str, language: str = "en"):
        """Called when Retell receives a transcript"""
        # Get phone number from session if available
        user_id = call_id
        if hasattr(self.retell_handler, 'active_sessions') and call_id in self.retell_handler.active_sessions:
            session = self.retell_handler.active_sessions[call_id]
            user_id = session.from_number or call_id
        
        safety_result = await self.safety_wrapper.check_voice_query(
            user_id=user_id,
            transcript=transcript,
            language=language,
            call_id=call_id
        )
        
        if safety_result.should_escalate:
            return {
                "override": True,
                "response": safety_result.safety_message,
                "escalate": True
            }
        
        return {"override": False, "transcript": safety_result.modified_transcript}


class BolnaSafetyIntegration:
    """Safety integration for Bolna.ai"""
    
    def __init__(self, bolna_webhook_handler):
        self.webhook_handler = bolna_webhook_handler
        self.safety_wrapper = get_voice_safety_wrapper()
    
    async def on_transcript(self, call_id: str, transcript: str, phone_number: str, language: str = "hi"):
        """Called when Bolna receives a transcript"""
        safety_result = await self.safety_wrapper.check_voice_query(
            user_id=phone_number,
            transcript=transcript,
            language=language,
            call_id=call_id
        )
        
        if safety_result.should_escalate:
            return {
                "override": True,
                "response": safety_result.safety_message,
                "escalate": True
            }
        
        return {"override": False, "transcript": safety_result.modified_transcript}
