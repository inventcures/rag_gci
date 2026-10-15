"""
Enhanced WhatsApp Bot with Indian Language Support using Twilio
Supports text queries, voice messages, and multilingual responses

Now with Gemini Live API support for real-time voice conversations.
"""

import os
import json
import asyncio
import functools
import uuid
from collections import OrderedDict
import tempfile
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import aiofiles
import aiohttp
from datetime import datetime, timedelta
import base64
from urllib.parse import unquote

import requests
from fastapi import Depends, FastAPI, Request, HTTPException, Form
from fastapi.responses import JSONResponse, Response
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

from observability import span, trace_request
from post_work import after_reply
from reminder_text import ensure_dose_free, medicine_label
from supported_languages import (
    ENGLISH_CODE,
    PROVIDER_EDGE,
    PROVIDER_GEMINI_LIVE,
    PROVIDER_SARVAM,
    SUPPORTED_LANGUAGES,
    UNSUPPORTED_LANGUAGE_MESSAGE,
    Voice,
    detect_language,
    get_language,
)
from webhook_auth import require_twilio
import edge_tts

# Import the smart clarification system
from smart_clarification_system import SmartClarificationSystem, ClarityLevel

# Import Safety Enhancements
try:
    from safety_enhancements import (
        SafetyEnhancementsManager,
        get_safety_manager,
        MedicationReminderScheduler,
        HandoffReason,
    )
    SAFETY_ENHANCEMENTS_AVAILABLE = True
except ImportError:
    SAFETY_ENHANCEMENTS_AVAILABLE = False
    HandoffReason = None

# Import Medication Voice Reminders
try:
    from medication_voice_reminders import (
        MedicationVoiceReminderSystem,
        get_medication_voice_reminder_system,
    )
    MEDICATION_VOICE_AVAILABLE = True
except ImportError:
    MEDICATION_VOICE_AVAILABLE = False

# Import Gemini Live API integration
try:
    from gemini_live import (
        GeminiLiveService,
        GeminiLiveSession,
        GeminiLiveError,
        SessionManager,
        AudioHandler,
        get_config as get_gemini_config,
        SUPPORTED_LANGUAGES as GEMINI_LANGUAGES,
    )
    GEMINI_LIVE_AVAILABLE = True
except ImportError:
    GEMINI_LIVE_AVAILABLE = False
    GeminiLiveService = None
    SessionManager = None
    AudioHandler = None

logger = logging.getLogger(__name__)


class EnhancedSTTService:
    """Speech-to-Text for WhatsApp voice notes, via Sarvam Saaras (India-hosted).

    Saaras detects the spoken language when asked for "unknown". If that call
    fails, the note is transcribed as Hindi (Sarvam needs a language code) and
    the language is judged from the transcript's script. The language names
    and codes come from the Supported Language table.
    """

    DEFAULT_LANGUAGE = "hi"

    def __init__(self):
        from sarvam_integration.client import SarvamClient

        self.sarvam = SarvamClient()
        self.supported_languages = {lang.code: lang.name for lang in SUPPORTED_LANGUAGES}

    @staticmethod
    def _to_bcp47(language: str) -> str:
        """Sarvam's code for a language: the table's Sarvam spelling for a
        Supported Language, another code Saaras transcribes as is, and
        "unknown" (Saaras detects it) for anything else. Never Hindi."""
        from sarvam_integration.config import SARVAM_STT_LANGUAGES

        lang = get_language(language)
        if lang is not None:
            return lang.voice(PROVIDER_SARVAM).language
        bcp47 = language if "-" in language else f"{language}-IN"
        return bcp47 if bcp47 in SARVAM_STT_LANGUAGES else "unknown"

    async def transcribe_audio(self, audio_file_path: str, detected_language: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio with language detection"""
        try:
            if not self.sarvam.is_available():
                return {"status": "error", "error": "SARVAM_API_KEY not configured"}

            async with aiofiles.open(audio_file_path, "rb") as f:
                audio_bytes = await f.read()

            if detected_language:
                result = await self._transcribe_with_sarvam(audio_bytes, detected_language)
            else:
                # Saaras detects the spoken language when given "unknown" and
                # returns the code it heard. Hindi with a script check is the
                # fallback if detection fails.
                result = await self._transcribe_with_sarvam(audio_bytes, "unknown")
                if result["status"] == "success":
                    detected_language = result["language"]
                else:
                    result = await self._transcribe_with_sarvam(audio_bytes, self.DEFAULT_LANGUAGE)
                    if result["status"] == "success":
                        detected_language = self._detect_language(result["text"])

            if result["status"] == "success":
                result["detected_language"] = detected_language
                lang = get_language(detected_language)
                result["language_name"] = lang.name if lang else "Unknown"
            
            return result
            
        except Exception as e:
            logger.error(f"STT error: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _transcribe_with_sarvam(self, audio_bytes: bytes, language: str) -> Dict[str, Any]:
        """Transcribe using Sarvam Saaras"""
        try:
            code = "unknown" if language == "unknown" else self._to_bcp47(language)
            stt = await self.sarvam.speech_to_text(audio_bytes, language=code, with_timestamps=False)
            if stt.success:
                heard = language
                if language == "unknown":
                    # "ta-IN" -> "ta", "od-IN" -> "or"; a language outside
                    # the eleven keeps its own code, so it is never taken for
                    # Hindi.
                    lang = get_language(stt.language_code)
                    heard = lang.code if lang else (stt.language_code or "").split("-")[0]
                return {"status": "success", "text": stt.transcript or "", "language": heard}
            return {"status": "error", "error": f"Sarvam STT error: {stt.error}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _detect_language(self, text: str) -> str:
        """The Supported Language a transcript is written in, by its script;
        English when it has no letters to judge by."""
        lang = detect_language(text)
        return lang.code if lang else ENGLISH_CODE


class EnhancedTTSService:
    """Text-to-Speech for WhatsApp replies, in the voice the Supported Language
    table names for each language.

    The India-hosted voice (Sarvam Bulbul) is tried first; Edge is the fallback
    for a language the table gives an Edge voice. A language with no voice that
    works is never spoken in another language: the result has status
    "unavailable" and carries the message to send instead, in that language
    (or in English, for a language outside the eleven).
    """

    # The order WhatsApp tries the table's voices in: India-hosted first.
    PROVIDER_ORDER = (PROVIDER_SARVAM, PROVIDER_EDGE)

    def __init__(self):
        # Short code -> the voice each Supported Language is spoken in first
        # (informational; synthesis reads the table).
        first = self.PROVIDER_ORDER[0]
        self.supported_voices: Dict[str, str] = {
            lang.code: lang.voice(first).name for lang in SUPPORTED_LANGUAGES if lang.voice(first)}
        self.output_dir = Path("cache/tts")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def synthesize_speech(self, text: str, language: str = ENGLISH_CODE) -> Dict[str, Any]:
        """Speak ``text`` in ``language``, or say why it cannot be spoken."""
        lang = get_language(language)
        logger.info("🎤 TTS synthesis: %d chars, language %s", len(text or ""), language)
        if lang is None:
            logger.warning("  ⚠️ %s is not a Supported Language; nothing is spoken", language)
            return self._unavailable(text, language, UNSUPPORTED_LANGUAGE_MESSAGE)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = self.output_dir / f"tts_{lang.code}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        for provider in self.PROVIDER_ORDER:
            voice = lang.voice(provider)
            if voice is None:
                continue
            audio_file = await self._synthesize_with(voice, text, stem)
            if audio_file is not None:
                return {
                    "status": "success",
                    "text": text,
                    "language": lang.code,
                    "provider": voice.provider,
                    "voice": voice.name,
                    "audio_file": str(audio_file),
                    "audio_available": True,
                    "file_size": audio_file.stat().st_size,
                }
        logger.warning("  ⚠️ No voice could speak %s; sending the unavailable message", lang.name)
        return self._unavailable(text, lang.code, lang.voice_unavailable)

    @staticmethod
    def _unavailable(text: str, language: str, message: str) -> Dict[str, Any]:
        """The result for speech that cannot be produced, with what to send instead."""
        return {
            "status": "unavailable",
            "text": text,
            "language": language,
            "audio_available": False,
            "unavailable_message": message,
        }

    async def _synthesize_with(self, voice: Voice, text: str, stem: Path) -> Optional[Path]:
        """The audio file ``voice`` produced for ``text``, or None if it failed."""
        try:
            if voice.provider == PROVIDER_SARVAM:
                return await self._sarvam(voice, text, stem)
            if voice.provider == PROVIDER_EDGE:
                audio_file = stem.with_suffix(".mp3")
                await edge_tts.Communicate(text, voice.name).save(str(audio_file))
                return audio_file if audio_file.exists() else None
        except Exception as e:  # one provider failing must not stop the next
            logger.warning("  ⚠️ %s TTS failed for %s: %s", voice.provider, voice.language, e)
        return None

    @staticmethod
    async def _sarvam(voice: Voice, text: str, stem: Path) -> Optional[Path]:
        """Sarvam Bulbul (India-hosted). Its WAV is converted to MP3 for WhatsApp
        where pydub and ffmpeg allow, and kept as WAV otherwise."""
        from sarvam_integration import SarvamClient

        result = await SarvamClient().text_to_speech(text, voice.language, voice=voice.name)
        if not (result and result.success and result.audio_base64):
            logger.warning("  ⚠️ Sarvam returned no audio: %s", getattr(result, "error", None))
            return None
        wav = base64.b64decode(result.audio_base64)
        try:
            import io
            from pydub import AudioSegment

            mp3 = stem.with_suffix(".mp3")
            AudioSegment.from_file(io.BytesIO(wav), format="wav").export(str(mp3), format="mp3")
            return mp3
        except Exception as e:  # pydub or ffmpeg missing, or undecodable audio
            logger.warning("  ⚠️ WAV to MP3 conversion failed (%s); keeping WAV", e)
            path = stem.with_suffix(".wav")
            path.write_bytes(wav)
            return path


class TwilioWhatsAppAPI:
    """Twilio WhatsApp API integration using free sandbox"""
    
    def __init__(self):
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")  # Twilio Sandbox
        
        if self.account_sid and self.auth_token:
            self.client = Client(self.account_sid, self.auth_token)
        else:
            self.client = None
            logger.warning("Twilio credentials not configured")
    
    async def send_text_message(self, to: str, message: str) -> Dict[str, Any]:
        """Send text message via Twilio WhatsApp"""
        logger.info("📤 SEND_TEXT_MESSAGE CALLED:")
        logger.debug(f"  📱 To: {to}")
        logger.info(f"  💬 Message length: {len(message)} chars")
        logger.debug(f"  💬 Message preview: {message[:100]}...")
        
        try:
            if not self.client:
                logger.error("  ❌ Twilio client not configured")
                return {"status": "error", "error": "Twilio not configured"}
            
            # Ensure the 'to' number has whatsapp: prefix
            if not to.startswith("whatsapp:"):
                to = f"whatsapp:{to}"
            logger.debug(f"  📱 Formatted to: {to}")
            
            logger.info(f"  📋 Twilio text message params:")
            logger.debug(f"    From: {self.from_number}")
            logger.debug(f"    To: {to}")
            logger.info(f"    Body length: {len(message)}")
            
            # Send message using Twilio
            logger.info("  📤 Sending text message via Twilio...")
            # The Twilio client is synchronous HTTP: off the event loop.
            message_obj = await asyncio.to_thread(
                self.client.messages.create,
                body=message,
                from_=self.from_number,
                to=to,
            )
            logger.info(f"  ✅ Twilio text message created: {message_obj.sid}")
            logger.info(f"  📊 Message status: {message_obj.status}")
            
            result = {
                "status": "success",
                "message_sid": message_obj.sid,
                "result": {
                    "sid": message_obj.sid,
                    "status": message_obj.status,
                    "to": to,
                    "from": self.from_number
                }
            }
            logger.debug(f"  ✅ Text message sent successfully: {result}")
            return result
                        
        except Exception as e:
            logger.error(f"  ❌ Twilio text send error: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}
    
    async def send_audio_message(self, to: str, audio_file_path: str, public_url: str = None) -> Dict[str, Any]:
        """Send audio message via Twilio WhatsApp"""
        logger.info("📤 SEND_AUDIO_MESSAGE CALLED:")
        logger.debug(f"  📱 To: {to}")
        logger.info(f"  📄 Audio file: {audio_file_path}")
        logger.info(f"  🌐 Public URL: {public_url}")
        
        try:
            if not self.client:
                logger.error("  ❌ Twilio client not configured")
                return {"status": "error", "error": "Twilio not configured"}
            
            # Verify audio file exists
            if not Path(audio_file_path).exists():
                logger.error(f"  ❌ Audio file does not exist: {audio_file_path}")
                return {"status": "error", "error": "Audio file not found"}
            
            file_size = Path(audio_file_path).stat().st_size
            logger.info(f"  📊 Audio file size: {file_size} bytes")
            
            # Ensure the 'to' number has whatsapp: prefix
            if not to.startswith("whatsapp:"):
                to = f"whatsapp:{to}"
            logger.debug(f"  📱 Formatted to: {to}")
            
            # For Twilio, we need a publicly accessible URL for the media
            if not public_url:
                # Generate a URL that will be served by our FastAPI app
                filename = Path(audio_file_path).name
                base_url = os.getenv('PUBLIC_BASE_URL') or os.getenv('NGROK_URL') or 'http://localhost:8001'
                public_url = f"{base_url}/media/{filename}"
                logger.info(f"  🌐 Generated public URL: {public_url}")
            
            logger.info(f"  📋 Twilio message params:")
            logger.debug(f"    From: {self.from_number}")
            logger.debug(f"    To: {to}")
            logger.info(f"    Media URL: {public_url}")
            
            # Send message with media using Twilio
            logger.info("  📤 Sending message via Twilio...")
            message_obj = await asyncio.to_thread(
                self.client.messages.create,
                media_url=[public_url],
                from_=self.from_number,
                to=to,
            )
            logger.info(f"  ✅ Twilio message created: {message_obj.sid}")
            logger.info(f"  📊 Message status: {message_obj.status}")
            
            result = {
                "status": "success",
                "message_sid": message_obj.sid,
                "result": {
                    "sid": message_obj.sid,
                    "status": message_obj.status,
                    "to": to,
                    "from": self.from_number,
                    "media_url": public_url
                }
            }
            logger.debug(f"  ✅ Audio message sent successfully: {result}")
            return result
                        
        except Exception as e:
            logger.error(f"  ❌ Twilio audio send error: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}
    
    async def download_media(self, media_url: str) -> Optional[str]:
        """Download media file from Twilio URL"""
        logger.info("📥 DOWNLOAD_MEDIA CALLED:")
        logger.info(f"  🔗 URL: {media_url}")
        
        try:
            if not media_url:
                logger.error("  ❌ No media URL provided")
                return None
            
            # Check credentials
            logger.info(f"  🔑 Checking credentials:")
            logger.info(f"    Client: {'✅ Present' if self.client else '❌ Missing'}")
            logger.info(f"    Account SID: {'✅ Present' if self.account_sid else '❌ Missing'}")
            logger.info(f"    Auth Token: {'✅ Present' if self.auth_token else '❌ Missing'} ({'*' * 10 if self.auth_token else 'None'})")
            
            if not self.client or not self.account_sid or not self.auth_token:
                logger.error("  ❌ Twilio client not configured for media download")
                return None
            
            # Use Twilio credentials for authentication
            import base64
            auth_string = base64.b64encode(f"{self.account_sid}:{self.auth_token}".encode()).decode()
            logger.info(f"  🔐 Auth string created (length: {len(auth_string)})")
            
            headers = {
                'Authorization': f'Basic {auth_string}',
                'User-Agent': 'python-requests/2.28.1'
            }
            logger.info(f"  📋 Headers prepared: {list(headers.keys())}")
            
            # Download the media file from Twilio's URL with authentication
            logger.info(f"  🌐 Making HTTP request to: {media_url}")
            async with aiohttp.ClientSession() as session:
                async with session.get(media_url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    logger.info(f"  📊 Response received: HTTP {response.status}")
                    logger.info(f"  📋 Response headers: {dict(response.headers)}")
                    
                    if response.status == 200:
                        # Create temp file
                        temp_dir = Path("cache/downloads")
                        temp_dir.mkdir(parents=True, exist_ok=True)
                        logger.info(f"  📁 Temp directory: {temp_dir}")
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        # Determine file extension from content type or URL
                        content_type = response.headers.get('content-type', '')
                        logger.info(f"  📋 Content type: '{content_type}'")
                        
                        if 'audio/ogg' in content_type or media_url.endswith('.ogg'):
                            ext = '.ogg'
                        elif 'audio/mpeg' in content_type or media_url.endswith('.mp3'):
                            ext = '.mp3'
                        elif 'audio/wav' in content_type or media_url.endswith('.wav'):
                            ext = '.wav'
                        else:
                            ext = '.ogg'  # Default for WhatsApp voice messages
                        
                        logger.info(f"  📎 File extension determined: {ext}")
                        temp_file = temp_dir / f"download_{timestamp}_{uuid.uuid4().hex[:8]}{ext}"
                        logger.info(f"  📄 Target file: {temp_file}")
                        
                        # Save file
                        logger.info("  💾 Reading response content...")
                        content = await response.read()
                        logger.info(f"  📊 Content size: {len(content)} bytes")
                        
                        logger.info("  💾 Writing to file...")
                        async with aiofiles.open(temp_file, 'wb') as f:
                            await f.write(content)
                        
                        # Verify file was created
                        if temp_file.exists():
                            file_size = temp_file.stat().st_size
                            logger.info(f"  ✅ File saved successfully: {temp_file} ({file_size} bytes)")
                            return str(temp_file)
                        else:
                            logger.error("  ❌ File was not created on disk")
                            return None
                            
                    elif response.status == 401:
                        logger.error("  ❌ Authentication failed (401 Unauthorized)")
                        error_text = await response.text()
                        logger.error(f"  📄 Error response: {error_text}")
                    elif response.status == 403:
                        logger.error("  ❌ Access forbidden (403 Forbidden)")
                        error_text = await response.text()
                        logger.error(f"  📄 Error response: {error_text}")
                    elif response.status == 404:
                        logger.error("  ❌ Media not found (404 Not Found)")
                        error_text = await response.text()
                        logger.error(f"  📄 Error response: {error_text}")
                    else:
                        logger.error(f"  ❌ HTTP error: {response.status} {response.reason}")
                        error_text = await response.text()
                        logger.error(f"  📄 Error response: {error_text}")
                        
        except aiohttp.ClientTimeout as e:
            logger.error(f"  ⏰ Request timeout: {e}")
        except aiohttp.ClientError as e:
            logger.error(f"  🌐 HTTP client error: {e}")
        except Exception as e:
            logger.error(f"  ❌ Unexpected error: {e}", exc_info=True)
            
        logger.error("  ❌ Download failed, returning None")
        return None


class EnhancedWhatsAppBot:
    """Enhanced WhatsApp bot with full Indian language support and Gemini Live voice"""

    def __init__(self, rag_pipeline, stt_service: EnhancedSTTService, tts_service: EnhancedTTSService):
        self.rag_pipeline = rag_pipeline
        self.stt_service = stt_service
        self.tts_service = tts_service
        self.twilio_api = TwilioWhatsAppAPI()

        # Language preferences per user
        self.user_preferences = {}

        # Store for serving audio files
        self.media_files = {}

        # Twilio WhatsApp message limit
        self.whatsapp_char_limit = 1550

        # Smart clarification system (its LLM calls go through the shared llm client)
        self.clarification_system = None
        try:
            from llm import load_llm_settings

            llm_settings = load_llm_settings()
            if llm_settings.transport != "unconfigured" or llm_settings.fallback_enabled:
                self.clarification_system = SmartClarificationSystem()
        except Exception as e:
            logger.warning(f"Smart clarification system unavailable: {e}")

        # Simple conversation history (last 5 messages per user)
        self.conversation_history = {}

        # Twilio retries a webhook it did not see acknowledged; message ids
        # already accepted are not processed twice.
        self._seen_message_ids: "OrderedDict[str, None]" = OrderedDict()
        
        # Initialize Safety Enhancements
        self.safety_manager = None
        self.reminder_scheduler = None
        if SAFETY_ENHANCEMENTS_AVAILABLE:
            try:
                self.safety_manager = get_safety_manager()
                self.reminder_scheduler = MedicationReminderScheduler()
                logger.info("✅ Safety enhancements initialized in WhatsApp bot")
            except Exception as e:
                logger.warning(f"Failed to initialize safety enhancements: {e}")
        
        # Initialize Medication Voice Reminders
        self.voice_reminder_system = None
        if MEDICATION_VOICE_AVAILABLE:
            try:
                self.voice_reminder_system = get_medication_voice_reminder_system()
                # Set up callbacks
                self.voice_reminder_system.on_patient_confirmed = self._on_medication_confirmed
                self.voice_reminder_system.on_call_failed = self._on_medication_call_failed
                logger.info("✅ Medication voice reminder system initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize voice reminders: {e}")

        # Initialize Gemini Live service (if available and enabled)
        self.gemini_live_enabled = False
        self.gemini_service = None
        self.gemini_session_manager = None
        self.audio_handler = None

        self._init_gemini_live()

    def _init_gemini_live(self):
        """Initialize Gemini Live service if available and enabled."""
        if not GEMINI_LIVE_AVAILABLE:
            logger.info("Gemini Live module not available - using fallback STT+LLM+TTS pipeline")
            return

        try:
            gemini_config = get_gemini_config()

            if not gemini_config.enabled:
                logger.info("Gemini Live is disabled in config - using fallback pipeline")
                return

            # Initialize the Gemini Live service with RAG pipeline
            self.gemini_service = GeminiLiveService(
                rag_pipeline=self.rag_pipeline
            )

            if not self.gemini_service.is_available():
                logger.warning(
                    "Gemini Live service not available (check credentials) - "
                    "using fallback pipeline"
                )
                return

            # Initialize session manager
            self.gemini_session_manager = SessionManager(self.gemini_service)

            # Initialize audio handler
            self.audio_handler = AudioHandler()

            self.gemini_live_enabled = True
            logger.info(
                "Gemini Live initialized successfully - "
                f"languages: {gemini_config.supported_languages}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Gemini Live: {e}")
            self.gemini_live_enabled = False

    async def start_gemini_session_manager(self):
        """Start the Gemini session manager (call this after app startup)."""
        if self.gemini_session_manager:
            await self.gemini_session_manager.start()
            logger.info("Gemini session manager started")

    async def stop_gemini_session_manager(self):
        """Stop the Gemini session manager (call this before app shutdown)."""
        if self.gemini_session_manager:
            await self.gemini_session_manager.stop()
            logger.info("Gemini session manager stopped")

    @staticmethod
    def _live_voice(lang_code: Optional[str]) -> Optional[Voice]:
        """The table's Live voice for a language, or None where Live cannot
        speak it (Odia, Punjabi, anything outside the eleven)."""
        lang = get_language(lang_code)
        return lang.voice(PROVIDER_GEMINI_LIVE) if lang else None

    @staticmethod
    def _answer_language(lang_code: Optional[str]) -> Tuple[str, Optional[str]]:
        """The Supported Language to answer in, and the notice to send first.

        A Supported Language is answered in itself. Anything else is answered
        in English, with the plain notice that the language is unavailable;
        it is never guessed to be Hindi.
        """
        lang = get_language(lang_code)
        if lang is not None:
            return lang.code, None
        logger.warning("Language %r is not a Supported Language; answering in English", lang_code)
        return ENGLISH_CODE, UNSUPPORTED_LANGUAGE_MESSAGE

    def _text_language(self, from_number: str, text: str) -> str:
        """The language to answer a typed message in.

        The script the message is written in decides, unless it matches the
        participant's chosen language (Hindi and Marathi share a script). A
        message in Latin script keeps the chosen language, or English when the
        participant has chosen none. Never a Hindi default.
        """
        preferred = get_language(self.user_preferences.get(from_number, {}).get("language"))
        written = detect_language(text)
        if written is not None and written.code != ENGLISH_CODE:
            if preferred is not None and preferred.script == written.script:
                return preferred.code
            return written.code
        return preferred.code if preferred is not None else ENGLISH_CODE

    def _update_conversation_history(self, user_id: str, message: str):
        """Update conversation history for a user"""
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        self.conversation_history[user_id].append(message)
        
        # Keep only last 5 messages
        if len(self.conversation_history[user_id]) > 5:
            self.conversation_history[user_id] = self.conversation_history[user_id][-5:]
    
    def _get_conversation_history(self, user_id: str) -> List[str]:
        """Get conversation history for a user"""
        return self.conversation_history.get(user_id, [])
    
    def _ensure_whatsapp_length_limit(self, message: str, limit: Optional[int] = None) -> str:
        """Ensure message fits `limit` (default: Twilio's WhatsApp character limit).

        A caller that adds a header and footer passes the room left for the body,
        so the truncation notice is not itself cut off by the header's own limit.
        """
        limit = self.whatsapp_char_limit if limit is None else limit
        if len(message) <= limit:
            return message
        
        # If message is too long, truncate it intelligently
        truncated = message[:limit - 50]  # Leave room for truncation note
        
        # Try to break at sentence boundary
        last_sentence_end = max(
            truncated.rfind('.'),
            truncated.rfind('!'),
            truncated.rfind('?'),
            truncated.rfind('।')  # Hindi sentence ending
        )
        
        if last_sentence_end > len(truncated) * 0.7:  # If sentence break is reasonably close to end
            truncated = truncated[:last_sentence_end + 1]
        
        # Add truncation notice
        truncated += "... (message truncated due to length limit)"
        
        logger.info(f"📏 Message truncated from {len(message)} to {len(truncated)} characters")
        return truncated
    
    def create_webhook_app(self) -> FastAPI:
        """Create FastAPI app for Twilio WhatsApp webhook"""
        
        app = FastAPI(title="Enhanced WhatsApp RAG Bot with Twilio")
        
        @app.post("/webhook", dependencies=[Depends(require_twilio)])
        async def handle_twilio_webhook(
            From: str = Form(...),
            To: str = Form(...),
            Body: str = Form(None),
            MediaUrl0: str = Form(None),
            MediaContentType0: str = Form(None),
            NumMedia: str = Form("0"),
            MessageSid: str = Form(None),
        ):
            """Acknowledge Twilio at once; the reply is sent from the background."""
            self.accept_twilio_message(MessageSid, From, To, Body, MediaUrl0, MediaContentType0, NumMedia)
            return Response(content=str(MessagingResponse()), media_type="application/xml")
        
        @app.get("/media/{filename}")
        async def serve_media(filename: str):
            """Serve audio files for Twilio"""
            try:
                if filename in self.media_files:
                    file_path = self.media_files[filename]
                    if Path(file_path).exists():
                        with open(file_path, 'rb') as f:
                            content = f.read()
                        
                        # Determine content type based on file extension
                        if filename.endswith('.mp3'):
                            media_type = "audio/mpeg"
                        elif filename.endswith('.ogg'):
                            media_type = "audio/ogg"
                        elif filename.endswith('.wav'):
                            media_type = "audio/wav"
                        else:
                            media_type = "audio/mpeg"
                        
                        return Response(content=content, media_type=media_type)
                
                raise HTTPException(status_code=404, detail="Media not found")
                
            except Exception as e:
                logger.error(f"Media serve error: {e}")
                raise HTTPException(status_code=500, detail="Media error")
        
        @app.post("/api/set_language", dependencies=[Depends(require_twilio)])
        async def set_user_language(phone_number: str, language: str):
            """Set a user's preferred language; Twilio-signed requests only."""
            lang = get_language(language)
            if lang is not None:
                self.user_preferences[phone_number] = {"language": lang.code}
                return {"status": "success", "language": lang.code}
            return {"status": "error", "error": "Unsupported language"}
        
        return app
    
    SEEN_MESSAGE_IDS = 4096

    def accept_twilio_message(self, message_sid: Optional[str], from_number: str, to_number: str,
                              body: Optional[str], media_url: Optional[str],
                              media_content_type: Optional[str], num_media: Optional[str]) -> bool:
        """Schedule one webhook's message for processing and return at once.

        Twilio waits on the webhook; answering inside it held Twilio for the
        whole RAG, translation and TTS, and a slow answer was retried as a
        duplicate. Returns False for a message id already accepted."""
        if message_sid:
            if message_sid in self._seen_message_ids:
                logger.info("Duplicate WhatsApp webhook ignored")
                return False
            self._seen_message_ids[message_sid] = None
            while len(self._seen_message_ids) > self.SEEN_MESSAGE_IDS:
                self._seen_message_ids.popitem(last=False)
        after_reply("whatsapp_message", functools.partial(
            self._process_twilio_message, from_number, to_number, body,
            media_url, media_content_type, num_media or "0"))
        return True

    async def _process_twilio_message(self, from_number: str, to_number: str, body: str, 
                                    media_url: str, media_content_type: str, num_media: str):
        """Process incoming Twilio WhatsApp message"""
        try:
            logger.info("🔄 PROCESSING MESSAGE:")
            logger.debug(f"  📱 From: {from_number}")
            logger.info(f"  📍 To: {to_number}")
            logger.debug(f"  💬 Body: '{body}' (present: {bool(body and body.strip())})")
            logger.info(f"  🎵 Media URL: {media_url} (present: {bool(media_url)})")
            logger.info(f"  📋 Content Type: {media_content_type}")
            logger.info(f"  🔢 Num Media: {num_media} (as int: {int(num_media) if num_media else 0})")
            
            # Remove whatsapp: prefix if present
            clean_from = from_number.replace("whatsapp:", "")
            logger.debug(f"  🧹 Clean From: {clean_from}")
            
            # Decision logic with detailed logging
            has_text = body and body.strip()
            has_media = media_url and int(num_media) > 0
            
            logger.info(f"  🎯 Decision: has_text={bool(has_text)}, has_media={bool(has_media)}")
            
            with trace_request("whatsapp", user_id=clean_from, name="message",
                               kind="text" if has_text else "media" if has_media else "other"):
                if has_text:
                    await self._handle_twilio_text_message(clean_from, body.strip())
                elif has_media:
                    await self._handle_twilio_media_message(clean_from, media_url, media_content_type)
                else:
                    await self.twilio_api.send_text_message(
                        from_number,
                        "I can only process text messages and audio messages. Please send a text question or voice message."
                    )
                
        except Exception as e:
            logger.error(f"❌ Error processing Twilio message: {e}", exc_info=True)
            await self._send_error_message(from_number)
    
    async def _handle_twilio_text_message(self, from_number: str, text: str):
        """Handle text message from Twilio with smart clarification and safety enhancements"""
        try:
            if not text:
                return

            # Check for special commands
            text_lower = text.lower().strip()
            
            if text_lower.startswith("/lang"):
                await self._handle_language_command(text, from_number)
                return
            elif text_lower in ["/skip", "/clear", "/cancel"]:
                await self._handle_clarification_command(text, from_number)
                return
            
            # MEDICATION REMINDER COMMANDS
            elif text_lower.startswith("/remind") or text_lower.startswith("/setreminder"):
                await self._handle_medication_reminder_command(from_number=from_number, text=text)
                return
            elif text_lower == "/myreminders":
                await self._handle_list_reminders_command(from_number)
                return
            elif text_lower.startswith("/deletereminder"):
                await self._handle_delete_reminder_command(from_number=from_number, text=text)
                return
            elif text_lower == "/help":
                await self._send_help_message(from_number)
                return
            elif text_lower == "/human" or text_lower == "/talktohuman":
                await self._handle_human_handoff_request(from_number, text)
                return
            elif text_lower == "taken":
                await self._handle_medication_taken(from_number)
                return
            
            # Update conversation history
            self._update_conversation_history(from_number, f"User: {text}")

            # The language the message is written in (or the chosen one)
            user_lang = self._text_language(from_number, text)

            # SMART CLARIFICATION SYSTEM
            if self.clarification_system:
                logger.debug(f"  🧠 Analyzing query clarity: '{text}'")
                
                # Get conversation history for context
                history = self._get_conversation_history(from_number)
                
                # Analyze if clarification is needed
                clarity_result = await self.clarification_system.analyze_query_clarity(
                    text, from_number, history
                )
                
                logger.debug(f"  📊 Clarity analysis: {clarity_result}")
                
                # Handle different clarification scenarios
                if clarity_result.get("needs_clarification", False):
                    
                    if clarity_result.get("next_question"):
                        # Continue existing clarification flow
                        question = clarity_result["next_question"]
                        remaining = clarity_result.get("questions_remaining", 0)
                        
                        clarification_msg = f"❓ {question}"
                        if remaining > 0:
                            clarification_msg += f"\n\n({remaining} more question{'s' if remaining > 1 else ''} to help you better)"
                        
                        await self.twilio_api.send_text_message(from_number, clarification_msg)
                        logger.debug(f"  ❓ Sent clarification question: {question}")
                        return
                    
                    elif clarity_result.get("suggested_questions"):
                        # Start new clarification flow
                        questions = clarity_result["suggested_questions"]
                        if questions:
                            first_question = questions[0]
                            total_questions = len(questions)
                            
                            intro_msg = f"🤔 I need a bit more information to help you better.\n\n❓ {first_question}"
                            if total_questions > 1:
                                intro_msg += f"\n\n({total_questions} questions total to give you the best answer)"
                            
                            await self.twilio_api.send_text_message(from_number, intro_msg)
                            logger.info(f"  ❓ Started clarification flow with {total_questions} questions")
                            return
                
                # If clarification was completed, use enhanced query
                if clarity_result.get("clarification_complete", False):
                    enhanced_query = clarity_result.get("enhanced_query", text)
                    logger.debug(f"  ✅ Using enhanced query: {enhanced_query}")
                    text = enhanced_query
                    
                    # Send acknowledgment
                    await self.twilio_api.send_text_message(
                        from_number, 
                        "✅ Thank you for the additional information! Let me find the best answer for you..."
                    )
                
            # Query RAG pipeline (with potentially enhanced query)
            logger.debug(f"  🔍 About to query RAG pipeline with text: '{text}', user_id: '{from_number}', language: '{user_lang}'")
            result = await self._guarded_query(text, from_number, user_lang, answer_language="en")
            logger.info(f"  📊 RAG pipeline result status: {result.get('status')}, answer length: {len(result.get('answer', ''))}")

            # Send the response using the extracted method
            await self._send_rag_response(from_number, result, user_lang)


        except Exception as e:
            logger.error(f"Error handling Twilio text message: {e}")
            await self._send_error_message(from_number)
    
    async def _handle_twilio_media_message(self, from_number: str, media_url: str, content_type: str):
        """Handle media message from Twilio - routes to Gemini Live or fallback pipeline"""
        try:
            logger.info("🎵 HANDLING MEDIA MESSAGE:")
            logger.debug(f"  📱 From: {from_number}")
            logger.info(f"  🔗 Media URL: {media_url}")
            logger.info(f"  📋 Content Type: {content_type}")
            logger.info(f"  🤖 Gemini Live enabled: {self.gemini_live_enabled}")

            # Validate media URL
            if not media_url:
                logger.error("  ❌ No media URL provided")
                await self.twilio_api.send_text_message(
                    from_number,
                    "Sorry, I couldn't process your audio message."
                )
                return

            # Check if it's an audio message
            is_audio = content_type and content_type.startswith('audio/')
            logger.info(f"  🎯 Is Audio: {is_audio} (content_type='{content_type}')")

            if not is_audio:
                logger.error(f"  ❌ Not an audio message: {content_type}")
                await self.twilio_api.send_text_message(
                    from_number,
                    "I can only process audio messages. Please send a voice message."
                )
                return

            # Check Twilio API configuration before attempting download
            if not self.twilio_api.client:
                logger.error("  ❌ Twilio client not configured")
                await self.twilio_api.send_text_message(
                    from_number,
                    "Sorry, WhatsApp integration is not properly configured."
                )
                return

            # Try Gemini Live if enabled, with fallback to traditional pipeline
            if self.gemini_live_enabled:
                logger.info("  🚀 Using Gemini Live for voice processing")
                try:
                    await self._handle_twilio_media_message_gemini(from_number, media_url, content_type)
                    return
                except Exception as gemini_error:
                    logger.error(f"  ⚠️ Gemini Live failed, falling back to traditional pipeline: {gemini_error}")
                    # Continue to fallback below

            logger.info("  📞 Using traditional STT+LLM+TTS pipeline")

            # Download audio file
            logger.info(f"  📥 Attempting to download media from: {media_url}")
            audio_file_path = await self.twilio_api.download_media(media_url)

            if not audio_file_path:
                logger.error(f"  ❌ Failed to download audio file from {media_url}")
                await self.twilio_api.send_text_message(
                    from_number,
                    "Sorry, I couldn't download your audio message. Please try sending it again or check your connection."
                )
                return
            
            logger.info(f"  ✅ Audio file downloaded successfully: {audio_file_path}")

            # Transcribe audio
            logger.info(f"  🎤 Starting transcription of: {audio_file_path}")
            stt_result = await self.stt_service.transcribe_audio(audio_file_path)
            logger.debug(f"  🎤 Transcription result: {stt_result}")

            # Clean up audio file
            try:
                os.remove(audio_file_path)
                logger.info(f"  🗑️ Cleaned up audio file: {audio_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"  ⚠️ Failed to cleanup audio file: {cleanup_error}")

            if stt_result["status"] != "success":
                logger.error(f"  ❌ Transcription failed: {stt_result}")
                await self.twilio_api.send_text_message(
                    from_number,
                    "Sorry, I couldn't understand your audio message. Please try again."
                )
                return

            # Extract transcribed text and detected language
            text = stt_result["text"]
            detected_language, notice = self._answer_language(stt_result.get("detected_language"))

            # Update user's language preference
            self.user_preferences[from_number] = {"language": detected_language}

            # The answer is drafted while the confirmation is being sent.
            answer_task = asyncio.create_task(self._guarded_query(
                text, from_number, detected_language, answer_language="en"
            ))
            lang_name = stt_result.get("language_name", "Unknown")
            confirmation_msg = self._ensure_whatsapp_length_limit(f"🎯 Understood ({lang_name}): {text}")
            try:
                with span("send.text"):
                    await self.twilio_api.send_text_message(from_number, confirmation_msg)
                    if notice:
                        await self.twilio_api.send_text_message(from_number, notice)
            except BaseException:
                answer_task.cancel()
                raise
            result = await answer_task

            if result["status"] == "success":
                await self._deliver_answer(from_number, result, detected_language, speak=True)
            else:
                logger.error(f"❌ RAG pipeline returned status={result.get('status')}: {result.get('error')}")
                error_msg = "Sorry, I encountered an error processing your question."
                await self.twilio_api.send_text_message(from_number, error_msg)
                if detected_language != "en":
                    speech = await self._speak(error_msg, detected_language)
                    if speech.get("audio_available"):
                        await self._send_audio_file(from_number, speech["audio_file"])

        except Exception as e:
            logger.error(f"Error handling Twilio media message: {e}")
            await self._send_error_message(from_number)

    async def _handle_twilio_media_message_gemini(
        self, from_number: str, media_url: str, content_type: str
    ):
        """
        Handle media message using Gemini Live for real-time voice processing.

        1. Downloads the note and converts it to PCM
        2. Transcribes it (Sarvam), for grounding and for the Dose Boundary
        3. Concurrently: confirms what was heard, opens the Live session, and
           drafts the guarded answer, which is injected as grounding
        4. Sends the note to Live and collects its reply up to turn completion,
           within the voice deadline
        5. Sends Live's audio only if its transcript passes the Dose Boundary;
           otherwise sends and speaks the guarded answer
        """
        logger.info("🚀 GEMINI LIVE VOICE PROCESSING:")
        logger.debug(f"  📱 From: {from_number}")
        logger.info(f"  🔗 Media URL: {media_url}")

        # Step 1: Download audio from Twilio
        logger.info("  📥 Step 1: Downloading audio from Twilio...")
        audio_file_path = await self.twilio_api.download_media(media_url)

        if not audio_file_path:
            raise GeminiLiveError("Failed to download audio from Twilio")

        logger.info(f"  ✅ Audio downloaded: {audio_file_path}")

        try:
            # Step 2: Convert audio to PCM for Gemini
            logger.info("  🔄 Step 2: Converting audio to PCM...")
            pcm_data = await asyncio.to_thread(
                self.audio_handler.convert_to_pcm, audio_file_path
            )
            logger.info(f"  ✅ Audio converted to PCM: {len(pcm_data)} bytes")

            # Step 3: transcribe (for grounding and for the Dose Boundary on
            # what Live will say). Without a transcript nothing can be checked,
            # so the traditional pipeline takes over.
            stt_result = await self.stt_service.transcribe_audio(audio_file_path)
            if stt_result.get("status") != "success":
                raise GeminiLiveError("transcription failed; cannot ground or check the reply")
            query_text = stt_result["text"]
            detected_lang = stt_result.get("detected_language")

            # Step 4: Live only in a language it speaks. Otherwise, before
            # anything is sent, the traditional pipeline answers in the
            # participant's own language (or says plainly it cannot).
            live_voice = self._live_voice(detected_lang)
            if live_voice is None:
                raise GeminiLiveError(f"Live cannot speak {detected_lang!r}; using the Sarvam path")
            detected_lang = get_language(detected_lang).code
            gemini_lang = live_voice.language
            logger.info(f"  🌐 Language: {detected_lang} -> Gemini: {gemini_lang}")
            self.user_preferences[from_number] = {"language": detected_lang}
            self._update_conversation_history(from_number, f"User (voice): {query_text}")

            # Step 5, concurrently: confirm what was heard, open the Live
            # session, and draft the guarded answer that grounds it.
            lang_name = stt_result.get("language_name", "Unknown")
            confirmation_msg = self._ensure_whatsapp_length_limit(f"🎯 Understood ({lang_name}): {query_text}")

            async def confirm():
                with span("send.text"):
                    await self.twilio_api.send_text_message(from_number, confirmation_msg)

            async def open_session():
                with span("live.session"):
                    return await self.gemini_session_manager.get_or_create_session(
                        user_id=from_number, language=gemini_lang, voice=live_voice.name)

            _, session, grounding = await asyncio.gather(
                confirm(), open_session(),
                self._guarded_query(query_text, from_number, detected_lang, answer_language="en"),
            )
            if grounding.get("status") == "success":
                await self.gemini_service.inject_guarded_answer(session, grounding)

            # Step 6: the voice note to Live, then its reply, which ends on
            # turn completion (not when the session closes) or at the deadline.
            for chunk in self.audio_handler.chunk_audio(pcm_data):
                await session.send_audio(chunk)
            await session.end_audio()
            with span("live.reply") as reply_span:
                try:
                    reply_audio = await asyncio.wait_for(
                        self._collect_live_turn(session), timeout=self.live_reply_deadline_s())
                except asyncio.TimeoutError:
                    # Past the deadline the guarded answer is sent instead.
                    reply_audio = b""
                    reply_span.set(deadline_missed=True)
            spoken = session.get_response_transcription(clear=True).strip()

            # Step 7: the Dose Boundary on what Live said. Live's audio is sent
            # only when its transcript passes; otherwise the guarded answer is
            # sent and spoken instead, never the Live audio.
            from safety_enhancements import guard_query_result

            check = {"status": "success", "answer": spoken, "sources": grounding.get("sources", [])}
            safety = guard_query_result(check, query=query_text, language=detected_lang) if spoken else None
            if not reply_audio or safety is None or safety.dosage_blocked:
                logger.warning("Live reply %s; sending the guarded answer instead",
                               "redacted by the Dose Boundary" if safety is not None and safety.dosage_blocked
                               else "unverifiable (no audio or no transcript)")
                fallback = check if safety is not None and safety.dosage_blocked else grounding
                if fallback.get("status") != "success" or not fallback.get("answer"):
                    raise GeminiLiveError("no guarded answer to send")
                await self._deliver_answer(from_number, fallback, detected_lang, speak=True)
                return

            # Step 8: the reply text goes out while the audio is converted.
            mp3_task = asyncio.create_task(asyncio.to_thread(
                self.audio_handler.convert_from_pcm, reply_audio, "mp3"))
            try:
                with span("send.text"):
                    await self.twilio_api.send_text_message(
                        from_number, self._ensure_whatsapp_length_limit(check["answer"]))
            except BaseException:
                mp3_task.cancel()
                raise
            await self._send_audio_file(from_number, await mp3_task)
            self._update_conversation_history(from_number, f"Bot (Gemini Live): {check['answer'][:100]}...")

        finally:
            # Cleanup downloaded audio file
            try:
                os.remove(audio_file_path)
                logger.info(f"  🗑️ Cleaned up: {audio_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"  ⚠️ Cleanup failed: {cleanup_error}")

    @staticmethod
    def live_reply_deadline_s() -> float:
        """How long a voice note waits for Live's reply before the traditional
        pipeline answers instead."""
        return float(os.getenv("WHATSAPP_LIVE_REPLY_DEADLINE_S", "20"))

    @staticmethod
    async def _collect_live_turn(session) -> bytes:
        """Audio of Live's reply, up to its turn completion (or interruption)."""
        chunks = []
        stream = session.receive_audio()
        try:
            async for item in stream:
                if item == session.TURN_COMPLETE or item == session.INTERRUPTED:
                    break
                if isinstance(item, bytes):
                    chunks.append(item)
        finally:
            await stream.aclose()
        return b"".join(chunks)

    async def _handle_text_message(self, message: dict, from_number: str):
        """Handle text message"""
        try:
            text = message.get("text", {}).get("body", "").strip()
            
            if not text:
                return
            
            # Check for special commands
            if text.lower().startswith("/lang"):
                await self._handle_language_command(text, from_number)
                return
            
            # The language the message is written in (or the chosen one)
            user_lang = self._text_language(from_number, text)
            
            # Query RAG pipeline
            await self._send_typing_indicator(from_number)
            
            result = await self._guarded_query(
                text, from_number, user_lang, answer_language=user_lang
            )
            
            if result["status"] == "success":
                response_text = result["answer"]
                
                # Send text response
                await self.whatsapp_api.send_text_message(from_number, response_text)
                
                # Generate and send audio response
                tts_result = await self.tts_service.synthesize_speech(response_text, user_lang)
                
                if tts_result.get("audio_available"):
                    await self.whatsapp_api.send_audio_message(
                        from_number, 
                        tts_result["audio_file"]
                    )
                    
                    # Clean up audio file
                    try:
                        os.remove(tts_result["audio_file"])
                    except:
                        pass
            else:
                rag_error = result.get("error", "Unknown error")
                rag_traceback = result.get("traceback", "")
                logger.error(f"❌ RAG pipeline returned status={result.get('status')}: {rag_error}")
                if rag_traceback:
                    logger.error(f"❌ RAG traceback:\n{rag_traceback}")
                await self.whatsapp_api.send_text_message(
                    from_number,
                    "Sorry, I encountered an error processing your question. Please try again."
                )

        except Exception as e:
            logger.error(f"Error handling text message: {e}", exc_info=True)
            await self._send_error_message(from_number)
    
    async def _handle_audio_message(self, message: dict, from_number: str):
        """Handle audio/voice message"""
        try:
            audio_data = message.get("audio", {})
            media_id = audio_data.get("id")
            
            if not media_id:
                await self.whatsapp_api.send_text_message(
                    from_number,
                    "Sorry, I couldn't process your audio message."
                )
                return
            
            # Download audio file
            audio_file_path = await self.whatsapp_api.download_media(media_id)
            
            if not audio_file_path:
                await self.whatsapp_api.send_text_message(
                    from_number,
                    "Sorry, I couldn't download your audio message."
                )
                return
            
            # Send typing indicator
            await self._send_typing_indicator(from_number)
            
            # Transcribe audio
            stt_result = await self.stt_service.transcribe_audio(audio_file_path)
            
            # Clean up audio file
            try:
                os.remove(audio_file_path)
            except:
                pass
            
            if stt_result["status"] != "success":
                await self.whatsapp_api.send_text_message(
                    from_number,
                    "Sorry, I couldn't understand your audio message. Please try again."
                )
                return
            
            # Extract transcribed text and detected language
            text = stt_result["text"]
            detected_language, _ = self._answer_language(stt_result.get("detected_language"))
            
            # Update user's language preference
            self.user_preferences[from_number] = {"language": detected_language}
            
            # Send transcription confirmation
            lang_name = stt_result.get("language_name", "Unknown")
            await self.whatsapp_api.send_text_message(
                from_number,
                f"🎯 Understood ({lang_name}): {text}"
            )
            
            # Query RAG pipeline
            result = await self._guarded_query(
                text, from_number, detected_language, answer_language=detected_language
            )
            
            if result["status"] == "success":
                response_text = result["answer"]
                
                # Send text response
                await self.whatsapp_api.send_text_message(from_number, response_text)
                
                # Generate and send audio response in the same language
                tts_result = await self.tts_service.synthesize_speech(
                    response_text, detected_language
                )
                
                if tts_result.get("audio_available"):
                    await self.whatsapp_api.send_audio_message(
                        from_number,
                        tts_result["audio_file"]
                    )
                    
                    # Clean up audio file
                    try:
                        os.remove(tts_result["audio_file"])
                    except:
                        pass
            else:
                # Log the actual underlying error from the RAG pipeline
                rag_error = result.get("error", "Unknown error")
                rag_traceback = result.get("traceback", "")
                logger.error(f"❌ RAG pipeline returned status={result.get('status')}: {rag_error}")
                if rag_traceback:
                    logger.error(f"❌ RAG traceback:\n{rag_traceback}")

                error_msg = "Sorry, I encountered an error processing your question."
                await self.whatsapp_api.send_text_message(from_number, error_msg)

                # Send error in detected language
                if detected_language != "en":
                    tts_result = await self.tts_service.synthesize_speech(
                        error_msg, detected_language
                    )
                    if tts_result.get("audio_available"):
                        await self.whatsapp_api.send_audio_message(
                            from_number,
                            tts_result["audio_file"]
                        )
                        try:
                            os.remove(tts_result["audio_file"])
                        except:
                            pass

        except Exception as e:
            logger.error(f"Error handling audio message: {e}", exc_info=True)
            await self._send_error_message(from_number)

    async def _handle_voice_message(self, message: dict, from_number: str):
        """Handle voice message (same as audio)"""
        # Voice messages are handled the same way as audio messages
        voice_data = message.get("voice", {})
        # Convert voice to audio format for processing
        audio_message = {"audio": voice_data}
        await self._handle_audio_message({"audio": voice_data}, from_number)
    
    async def _handle_language_command(self, text: str, from_number: str):
        """Handle /lang: list the eleven Supported Languages, or set one and
        confirm it in that language."""
        try:
            # Parse command: /lang hi, /lang bn, etc.
            parts = text.lower().split()

            if len(parts) < 2:
                lines = "\n".join(
                    f"• /lang {lang.code} - {lang.name}"
                    + (f" ({lang.native_name})" if lang.native_name != lang.name else "")
                    for lang in SUPPORTED_LANGUAGES
                )
                help_text = (
                    "🌐 Bilingual Response Mode:\nSend: /lang [code]\n\n"
                    f"Supported languages:\n{lines}\n\nExample: /lang hi"
                )
                help_text = self._ensure_whatsapp_length_limit(help_text)
                await self.twilio_api.send_text_message(from_number, help_text)
                return

            lang = get_language(parts[1])

            if lang is not None:
                self.user_preferences[from_number] = {"language": lang.code}
                await self.twilio_api.send_text_message(
                    from_number, f"✅ Language set to {lang.name} ({lang.code})")
                # Confirmation in the selected language, from the table
                await self.twilio_api.send_text_message(from_number, lang.language_set)
            else:
                await self.twilio_api.send_text_message(
                    from_number,
                    f"❌ Unsupported language code: {parts[1]}"
                )

        except Exception as e:
            logger.error(f"Error handling language command: {e}")
            await self._send_error_message(from_number)

    async def _handle_clarification_command(self, command: str, from_number: str):
        """Handle clarification flow commands like /skip, /clear, /cancel"""
        try:
            command = command.lower().strip()
            
            if command == "/skip":
                # Skip current clarification and proceed with original query
                if self.clarification_system:
                    status = self.clarification_system.get_clarification_status(from_number)
                    if status and status.get("in_clarification"):
                        # Get the original query
                        original_query = status.get("original_query", "")
                        self.clarification_system.clear_user_state(from_number)
                        
                        await self.twilio_api.send_text_message(
                            from_number, 
                            "⏭️ Skipping clarification questions. Let me answer based on your original question..."
                        )
                        
                        # Process original query without clarification
                        if original_query:
                            # Re-trigger processing but skip clarification
                            await self._process_query_directly(from_number, original_query)
                        return
                
                await self.twilio_api.send_text_message(
                    from_number, 
                    "ℹ️ No active clarification to skip. Send any health question to start."
                )
            
            elif command in ["/clear", "/cancel"]:
                # Clear clarification state
                if self.clarification_system:
                    self.clarification_system.clear_user_state(from_number)
                
                await self.twilio_api.send_text_message(
                    from_number, 
                    "✅ Clarification cleared. You can ask a new question now."
                )
            
        except Exception as e:
            logger.error(f"Error handling clarification command: {e}")
            await self._send_error_message(from_number)
    
    async def _guarded_query(
        self,
        text: str,
        from_number: str,
        source_language: str,
        answer_language: str,
    ) -> Dict[str, Any]:
        """
        Query the pipeline and pass the answer through the safety boundary.

        Every WhatsApp answer, text or spoken, goes through here, so the dosage
        guard runs before any message is sent, translated or synthesised. It is
        the same `process_response` step the mobile and voice routes apply.

        `answer_language` is the language of the text the user will be sent as-is:
        "en" where the bot sends the English answer and then translates it (the
        translation then carries the guarded wording), the user's language where
        the answer is sent and spoken directly.

        If the guard cannot be imported or fails, the exception propagates to the
        handler, which sends the generic error message rather than the draft.
        """
        from safety_enhancements import guard_query_result

        result = await self.rag_pipeline.query(
            text, user_id=from_number, source_language=source_language
        )
        guard_query_result(result, query=text, language=answer_language)
        return result

    async def _process_query_directly(self, from_number: str, query: str):
        """Process query directly without clarification checks"""
        try:
            user_lang = self._text_language(from_number, query)
            
            logger.debug(f"  🔍 Processing query directly: '{query}', language: '{user_lang}'")
            result = await self._guarded_query(query, from_number, user_lang, answer_language="en")
            
            # Process the result normally (same logic as in main handler)
            await self._send_rag_response(from_number, result, user_lang)
            
        except Exception as e:
            logger.error(f"Error processing query directly: {e}")
            await self._send_error_message(from_number)
    
    async def _send_rag_response(self, from_number: str, result: Dict[str, Any], user_lang: str):
        """Send RAG response with bilingual support"""
        try:
            if result["status"] == "success":
                await self._deliver_answer(from_number, result, user_lang, speak=False)
            else:
                logger.error(f"  ❌ RAG Query failed: {result.get('error')}")
                await self.twilio_api.send_text_message(
                    from_number,
                    "I'm having trouble finding information right now. Please try again or ask a different question."
                )
        except Exception as e:
            logger.error(f"Error sending RAG response: {e}")
            await self._send_error_message(from_number)

    def _with_header(self, header: str, text: str, footer: str) -> str:
        """header + text + footer within Twilio's WhatsApp limit."""
        room = self.whatsapp_char_limit - len(header) - len(footer)
        if len(text) > room:
            text = text[:room - 3] + "..."
        return f"{header}{text}{footer}"

    async def _speak(self, text: str, language: str) -> Dict[str, Any]:
        with span("tts", language=language):
            return await self.tts_service.synthesize_speech(text, language)

    async def _send_audio_file(self, from_number: str, audio_file: str) -> None:
        filename = Path(audio_file).name
        self.media_files[filename] = audio_file
        public_url = f"{os.getenv('PUBLIC_BASE_URL', 'http://localhost:8000')}/media/{filename}"
        with span("send.audio"):
            await self.twilio_api.send_audio_message(from_number, audio_file, public_url)

    async def _deliver_answer(self, from_number: str, result: Dict[str, Any], language: str,
                              speak: bool) -> None:
        """English text, then the user's language, then (speak) audio.

        Messages go out in that order, but each next step's work (translation,
        speech synthesis) starts while the previous message is being sent,
        instead of after it and a fixed pause. `result` has already passed the
        Dose Boundary, so the translation and the audio carry guarded text.
        """
        answer = result["answer"]
        footer = f"\n\n(model {result.get('model_used', 'unknown')} used)"
        header = "🇬🇧 English:\n"
        room = self.whatsapp_char_limit - len(header) - len(footer)
        english = self._with_header(header, self._ensure_whatsapp_length_limit(answer, room), footer)

        translation_task = None
        speech_task = None
        if language != "en":
            translation_task = asyncio.create_task(self._translate(answer, language))
        elif speak:
            speech_task = asyncio.create_task(self._speak(answer, language))
        try:
            with span("send.text"):
                await self.twilio_api.send_text_message(from_number, english)

            if translation_task is not None:
                translation = await translation_task
                translated = translation.get("status") == "success"
                # Untranslated, the answer is English, so it is spoken in
                # English rather than read out by another language's voice.
                spoken_text = translation["translated_text"] if translated else answer
                if speak:
                    speech_task = asyncio.create_task(
                        self._speak(spoken_text, language if translated else ENGLISH_CODE))
                if translated:
                    lang = get_language(language)
                    name = lang.name if lang else language
                    with span("send.text"):
                        await self.twilio_api.send_text_message(
                            from_number, self._with_header(f"🇮🇳 {name}:\n", spoken_text, footer))
                else:
                    logger.error(f"  ❌ Translation failed: {translation.get('error')}")

            if speech_task is not None:
                speech = await speech_task
                if speech.get("audio_available"):
                    await self._send_audio_file(from_number, speech["audio_file"])
                elif speech.get("unavailable_message"):
                    # No voice can speak it: say so plainly, never another language.
                    with span("send.text"):
                        await self.twilio_api.send_text_message(from_number, speech["unavailable_message"])
        finally:
            for task in (translation_task, speech_task):
                if task is not None and not task.done():
                    task.cancel()

        self._update_conversation_history(from_number, f"Bot: {answer[:100]}...")

    async def _translate(self, text: str, language: str) -> Dict[str, Any]:
        with span("translate", language=language):
            return await self.rag_pipeline.translate_text(text, language)

    async def _send_typing_indicator(self, to_number: str):
        """Send typing indicator (Twilio doesn't support this)"""
        # Twilio WhatsApp doesn't support typing indicators
        pass
    
    async def _send_unsupported_message(self, from_number: str, message_type: str):
        """Send unsupported message type response"""
        msg = f"Sorry, I don't support {message_type} messages yet. Please send text or voice messages."
        await self.twilio_api.send_text_message(from_number, msg)
    
    async def _send_error_message(self, from_number: str):
        """Send generic error message"""
        msg = "Sorry, I'm experiencing technical difficulties. Please try again later."
        await self.twilio_api.send_text_message(from_number, msg)

    # =========================================================================
    # SAFETY ENHANCEMENTS - Command Handlers
    # =========================================================================
    
    async def _handle_medication_reminder_command(self, from_number: str, text: str):
        """
        Handle /remind: set a reminder that names the medicine and the time.

        Anything typed after the times (often the dose) is stored on the
        reminder as its dosage, and never repeated back or spoken (ADR 0004,
        decision D10).
        """
        if not self.reminder_scheduler:
            await self.twilio_api.send_text_message(
                from_number,
                "Medication reminder feature is not available right now. Please try again later."
            )
            return

        try:
            # Parse command: /remind <medication> <times> [anything else, stored only]
            # Example: /remind Paracetamol 08:00,20:00
            parts = text.split(maxsplit=3)
            if len(parts) < 3:
                help_msg = """💊 Medication Reminder Setup

Usage: /remind <medicine_name> <times>

Examples:
• /remind Paracetamol 08:00,20:00
• /remind Ondansetron 06:00,18:00

Times should be in 24-hour format (HH:MM) separated by commas.
Take your medicine as your doctor or nurse advised."""
                await self.twilio_api.send_text_message(from_number, help_msg)
                return
            
            _, medication, times_str, *rest = parts
            dosage_and_instructions = rest[0] if rest else ""
            
            # Parse times
            times = [t.strip() for t in times_str.split(",")]
            
            # Get user language
            user_lang = self.user_preferences.get(from_number, {}).get("language", "en")
            
            # Create reminder
            reminder = self.reminder_scheduler.create_reminder(
                user_id=from_number,
                medication_name=medication,
                dosage=dosage_and_instructions,
                frequency="daily" if len(times) >= 1 else "custom",
                times=times,
                instructions="",
                language=user_lang
            )
            
            # Also set up voice call reminders if available
            voice_reminders_set = []
            if self.voice_reminder_system and from_number.startswith("+"):
                try:
                    for time_str in times:
                        # Parse time and create voice reminder
                        hour, minute = map(int, time_str.split(':'))
                        reminder_time = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
                        if reminder_time < datetime.now():
                            reminder_time += timedelta(days=1)  # Schedule for tomorrow
                        
                        voice_reminder = self.voice_reminder_system.create_voice_reminder(
                            user_id=from_number,
                            phone_number=from_number,
                            medication_name=medication,
                            dosage=dosage_and_instructions,
                            reminder_time=reminder_time,
                            language=user_lang,
                            preferred_provider="bolna"
                        )
                        voice_reminders_set.append(time_str)
                        
                except Exception as e:
                    logger.warning(f"Failed to create voice reminder: {e}")
            
            # Send confirmation
            times_formatted = ", ".join(times)
            confirm_msg = f"""✅ Medication Reminder Set!

💊 {medicine_label(medication)}
🕐 Times: {times_formatted}

I'll remind you when it's time to take your medication.
Reply 'TAKEN' after taking it."""
            
            # Add voice call info if set up
            if voice_reminders_set:
                confirm_msg += f"\n\n📞 Voice call reminders also set up for: {', '.join(voice_reminders_set)}"
            
            confirm_msg += f"""\n\nTo view all reminders: /myreminders
To delete: /deletereminder {reminder.reminder_id}"""
            
            await self.twilio_api.send_text_message(from_number, ensure_dose_free(confirm_msg))
            logger.debug(f"Created medication reminder {reminder.reminder_id} for {from_number}")
            
        except Exception as e:
            logger.error(f"Error creating medication reminder: {e}")
            await self.twilio_api.send_text_message(
                from_number,
                "Sorry, I couldn't set up that reminder. Please check the format and try again."
            )
    
    async def _handle_list_reminders_command(self, from_number: str):
        """Handle /myreminders command to list all reminders"""
        if not self.reminder_scheduler:
            await self.twilio_api.send_text_message(
                from_number,
                "Medication reminder feature is not available right now."
            )
            return
        
        try:
            reminders = self.reminder_scheduler.get_user_reminders(from_number)
            
            if not reminders:
                await self.twilio_api.send_text_message(
                    from_number,
                    "💊 You don't have any medication reminders set up.\n\nTo create one:\n/remind <medicine> <times>"
                )
                return
            
            lines = ["💊 Your Medication Reminders\n"]
            for i, r in enumerate(reminders, 1):
                times = ", ".join(r.scheduled_times)
                status = "✅ Active" if r.active else "⏸️ Paused"
                lines.append(f"{i}. {medicine_label(r.medication_name)}")
                lines.append(f"   🕐 {times}")
                lines.append(f"   {status} (ID: {r.reminder_id[:8]}...)")
                lines.append("")

            lines.append("To delete a reminder:\n/deletereminder <ID>")

            message = ensure_dose_free("\n".join(lines))
            message = self._ensure_whatsapp_length_limit(message)
            await self.twilio_api.send_text_message(from_number, message)
            
        except Exception as e:
            logger.error(f"Error listing reminders: {e}")
            await self.twilio_api.send_text_message(from_number, "Sorry, I couldn't retrieve your reminders.")
    
    async def _handle_delete_reminder_command(self, from_number: str, text: str):
        """Handle /deletereminder command"""
        if not self.reminder_scheduler:
            await self.twilio_api.send_text_message(
                from_number,
                "Medication reminder feature is not available right now."
            )
            return
        
        try:
            parts = text.split(maxsplit=1)
            if len(parts) < 2:
                await self.twilio_api.send_text_message(
                    from_number,
                    "Usage: /deletereminder <reminder_id>\n\nUse /myreminders to see your reminder IDs."
                )
                return
            
            reminder_id = parts[1].strip()
            
            # Try to delete
            success = self.reminder_scheduler.delete_reminder(reminder_id)
            
            if success:
                await self.twilio_api.send_text_message(
                    from_number,
                    "✅ Medication reminder deleted successfully."
                )
            else:
                await self.twilio_api.send_text_message(
                    from_number,
                    "❌ Could not find that reminder. Please check the ID using /myreminders"
                )
                
        except Exception as e:
            logger.error(f"Error deleting reminder: {e}")
            await self.twilio_api.send_text_message(from_number, "Sorry, I couldn't delete that reminder.")
    
    async def _handle_medication_taken(self, from_number: str):
        """Handle 'TAKEN' response after reminder"""
        if not self.reminder_scheduler:
            return
        
        try:
            # Find the most recent reminder for this user
            reminders = self.reminder_scheduler.get_user_reminders(from_number)
            
            if not reminders:
                await self.twilio_api.send_text_message(
                    from_number,
                    "I don't see any active reminders for you. Use /myreminders to check."
                )
                return
            
            # Find the reminder that was most recently triggered
            recent_reminder = None
            for r in reminders:
                if r.last_reminded:
                    if not recent_reminder or r.last_reminded > recent_reminder.last_reminded:
                        recent_reminder = r
            
            if recent_reminder:
                self.reminder_scheduler.mark_taken(recent_reminder.reminder_id)
                
                # Calculate adherence
                total_scheduled = len(recent_reminder.taken_history) + 1
                taken_count = len(recent_reminder.taken_history)
                
                msg = f"✅ Great! I've recorded that you took {medicine_label(recent_reminder.medication_name)}."
                if total_scheduled > 1:
                    adherence = (taken_count / total_scheduled) * 100
                    msg += f"\n📊 Your adherence: {adherence:.0f}%"
                
                await self.twilio_api.send_text_message(from_number, msg)
            else:
                await self.twilio_api.send_text_message(from_number, "✅ Recorded!")
                
        except Exception as e:
            logger.error(f"Error marking medication taken: {e}")
    
    async def _handle_human_handoff_request(self, from_number: str, text: str):
        """Handle request to talk to a human"""
        if not self.safety_manager:
            await self.twilio_api.send_text_message(
                from_number,
                "I'm connecting you to a human caregiver. Someone will contact you shortly."
            )
            return
        
        try:
            # Get conversation history
            history = []
            if from_number in self.conversation_history:
                history = [{"message": m} for m in self.conversation_history[from_number][-10:]]
            
            # Create handoff request
            handoff = self.safety_manager.handoff_system.create_handoff_request(
                user_id=from_number,
                reason=HandoffReason.USER_REQUEST,
                context="User requested to speak with a human",
                conversation_history=history,
                priority="medium"
            )
            
            # Send confirmation
            message = self.safety_manager.handoff_system.get_handoff_message(handoff, "en")
            await self.twilio_api.send_text_message(from_number, message)
            
            logger.debug(f"Human handoff requested by {from_number}, request ID: {handoff.request_id}")
            
        except Exception as e:
            logger.error(f"Error creating handoff request: {e}")
            await self.twilio_api.send_text_message(
                from_number,
                "I'm having trouble connecting you right now. Please call your care team directly."
            )
    
    async def _send_help_message(self, from_number: str):
        """Send help message with available commands"""
        help_text = """🙏 Welcome to Palli Sahayak!

I can help you with palliative care information and reminders.

📱 Commands:
• Ask any question - I'll search medical documents
• /lang <code> - Set language (hi, en, bn, ta, gu)
• /remind <medicine> <times> - Set medication reminder
• /myreminders - View your reminders
• /deletereminder <id> - Delete a reminder
• /human - Talk to a human caregiver
• /help - Show this message

📞 Voice Call Reminders:
When you set a medication reminder, I'll also call you at the scheduled time to remind you to take your medicine. You can confirm by pressing 1 or saying "yes".

🎙️ You can also send voice messages!

For emergencies, call 108 or 102 immediately."""
        
        await self.twilio_api.send_text_message(from_number, help_text)


# =========================================================================
# MEDICATION VOICE REMINDER CALLBACKS
# =========================================================================

async def _on_medication_confirmed(self, reminder):
    """Callback when patient confirms taking medication via voice call"""
    try:
        message = f"""✅ Voice Confirmation Received

Thank you for confirming that you took {medicine_label(reminder.medication_name)}.

Keep up the good work with your medication schedule! 💊"""
        
        await self.twilio_api.send_text_message(reminder.user_id, message)
        logger.debug(f"Voice confirmation received for {reminder.medication_name} from {reminder.user_id}")
        
    except Exception as e:
        logger.error(f"Error sending confirmation message: {e}")


async def _on_medication_call_failed(self, reminder):
    """Callback when voice call fails"""
    try:
        message = f"""⚠️ Medication Reminder Call

We tried calling you to remind you about {medicine_label(reminder.medication_name)} ({reminder.scheduled_time.strftime("%H:%M")}), but couldn't reach you.

Please take your medication as your doctor or nurse advised, if you haven't already.

Reply 'TAKEN' after taking it."""

        await self.twilio_api.send_text_message(reminder.user_id, ensure_dose_free(message))
        logger.warning("Voice reminder call failed")
        logger.debug(f"Voice call failed for {reminder.medication_name} to {reminder.user_id}")
        
    except Exception as e:
        logger.error(f"Error sending failed call message: {e}")


# Add callbacks to class
EnhancedWhatsAppBot._on_medication_confirmed = _on_medication_confirmed
EnhancedWhatsAppBot._on_medication_call_failed = _on_medication_call_failed
