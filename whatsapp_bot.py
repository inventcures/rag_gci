"""
Enhanced WhatsApp Bot with Indian Language Support using Twilio
Supports text queries, voice messages, and multilingual responses

Now with Gemini Live API support for real-time voice conversations.
"""

import os
import json
import asyncio
import tempfile
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import aiofiles
import aiohttp
from datetime import datetime
import base64
from urllib.parse import unquote

import requests
from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import JSONResponse, Response
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
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
    """Enhanced Speech-to-Text service with Indian language support"""
    
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.supported_languages = {
            "hi": "Hindi",
            "bn": "Bengali", 
            "ta": "Tamil",
            "gu": "Gujarati",
            "en": "English"
        }
        
        # Language detection patterns
        self.language_patterns = {
            "hi": ["हैलो", "नमस्ते", "कैसे", "क्या", "मैं"],
            "bn": ["হ্যালো", "নমস্কার", "কেমন", "কি", "আমি"],
            "ta": ["வணக்கம்", "எப்படி", "என்ன", "நான்"],
            "gu": ["નમસ્તે", "કેમ", "શું", "હું"],
            "en": ["hello", "how", "what", "the", "is"]
        }
    
    async def transcribe_audio(self, audio_file_path: str, detected_language: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio with language detection"""
        try:
            if not self.groq_api_key:
                return {"status": "error", "error": "GROQ_API_KEY not configured"}
            
            # First, try transcription without language specification
            result = await self._transcribe_with_groq(audio_file_path)
            
            if result["status"] == "success":
                text = result["text"]
                
                # Detect language from transcribed text
                if not detected_language:
                    detected_language = self._detect_language(text)
                
                # Re-transcribe with detected language for better accuracy
                if detected_language != "en":
                    refined_result = await self._transcribe_with_groq(
                        audio_file_path, detected_language
                    )
                    if refined_result["status"] == "success":
                        result = refined_result
                
                result["detected_language"] = detected_language
                result["language_name"] = self.supported_languages.get(detected_language, "Unknown")
            
            return result
            
        except Exception as e:
            logger.error(f"STT error: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _transcribe_with_groq(self, audio_file_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe using Groq Whisper API"""
        try:
            url = "https://api.groq.com/openai/v1/audio/transcriptions"
            headers = {"Authorization": f"Bearer {self.groq_api_key}"}
            
            async with aiohttp.ClientSession() as session:
                with open(audio_file_path, 'rb') as audio_file:
                    data = aiohttp.FormData()
                    data.add_field('file', audio_file, filename='audio.wav')
                    data.add_field('model', 'whisper-large-v3')
                    data.add_field('response_format', 'json')
                    
                    if language:
                        data.add_field('language', language)
                    
                    async with session.post(url, headers=headers, data=data) as response:
                        if response.status == 200:
                            result = await response.json()
                            return {
                                "status": "success",
                                "text": result.get("text", ""),
                                "language": language
                            }
                        else:
                            error_text = await response.text()
                            return {
                                "status": "error",
                                "error": f"Groq API error: {response.status} - {error_text}"
                            }
                            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _detect_language(self, text: str) -> str:
        """Detect language from text using simple pattern matching"""
        text_lower = text.lower()
        
        # Count matches for each language
        language_scores = {}
        for lang, patterns in self.language_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            if score > 0:
                language_scores[lang] = score
        
        # Return language with highest score, default to Hindi
        if language_scores:
            return max(language_scores, key=language_scores.get)
        
        return "hi"  # Default to Hindi


class EnhancedTTSService:
    """Enhanced Text-to-Speech service with Indian language support"""
    
    def __init__(self):
        self.supported_voices = {
            "hi": "hi-IN-SwaraNeural",      # Hindi (Female)
            "bn": "bn-IN-TanishaaNeural",   # Bengali (Female) 
            "ta": "ta-IN-PallaviNeural",    # Tamil (Female)
            "gu": "gu-IN-DhwaniNeural",     # Gujarati (Female)
            "en": "en-IN-NeerjaNeural"      # English Indian (Female)
        }
        
        self.output_dir = Path("cache/tts")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def synthesize_speech(self, text: str, language: str = "hi") -> Dict[str, Any]:
        """Convert text to speech using Edge TTS"""
        logger.info(f"🎤 TTS SYNTHESIS CALLED:")
        logger.info(f"  📝 Text: {text[:100]}...")
        logger.info(f"  🌐 Language: {language}")
        
        try:
            if language not in self.supported_voices:
                logger.warning(f"  ⚠️ Language {language} not supported, defaulting to Hindi")
                language = "hi"  # Default to Hindi
            
            voice = self.supported_voices[language]
            logger.info(f"  🗣️ Using voice: {voice}")
            
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"  📁 Output directory: {self.output_dir}")
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_file = self.output_dir / f"tts_{language}_{timestamp}.mp3"
            logger.info(f"  📄 Target file: {audio_file}")
            
            # Generate speech
            logger.info("  🎵 Generating speech with Edge TTS...")
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(audio_file))
            logger.info("  ✅ Edge TTS synthesis completed")
            
            # Verify file was created
            if audio_file.exists():
                file_size = audio_file.stat().st_size
                logger.info(f"  ✅ Audio file created successfully: {file_size} bytes")
                return {
                    "status": "success",
                    "text": text,
                    "language": language,
                    "voice": voice,
                    "audio_file": str(audio_file),
                    "audio_available": True,
                    "file_size": file_size
                }
            else:
                logger.error("  ❌ Audio file was not created")
                return {
                    "status": "error",
                    "error": "Failed to generate audio file",
                    "audio_available": False
                }
                
        except Exception as e:
            logger.error(f"  ❌ TTS synthesis error: {e}", exc_info=True)
            return {
                "status": "error",  # Changed from "success" to "error"
                "text": text,
                "language": language,
                "audio_available": False,
                "error": f"TTS failed: {str(e)}"
            }


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
        logger.info(f"  📱 To: {to}")
        logger.info(f"  💬 Message length: {len(message)} chars")
        logger.info(f"  💬 Message preview: {message[:100]}...")
        
        try:
            if not self.client:
                logger.error("  ❌ Twilio client not configured")
                return {"status": "error", "error": "Twilio not configured"}
            
            # Ensure the 'to' number has whatsapp: prefix
            if not to.startswith("whatsapp:"):
                to = f"whatsapp:{to}"
            logger.info(f"  📱 Formatted to: {to}")
            
            logger.info(f"  📋 Twilio text message params:")
            logger.info(f"    From: {self.from_number}")
            logger.info(f"    To: {to}")
            logger.info(f"    Body length: {len(message)}")
            
            # Send message using Twilio
            logger.info("  📤 Sending text message via Twilio...")
            message_obj = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=to
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
            logger.info(f"  ✅ Text message sent successfully: {result}")
            return result
                        
        except Exception as e:
            logger.error(f"  ❌ Twilio text send error: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}
    
    async def send_audio_message(self, to: str, audio_file_path: str, public_url: str = None) -> Dict[str, Any]:
        """Send audio message via Twilio WhatsApp"""
        logger.info("📤 SEND_AUDIO_MESSAGE CALLED:")
        logger.info(f"  📱 To: {to}")
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
            logger.info(f"  📱 Formatted to: {to}")
            
            # For Twilio, we need a publicly accessible URL for the media
            if not public_url:
                # Generate a URL that will be served by our FastAPI app
                filename = Path(audio_file_path).name
                base_url = os.getenv('PUBLIC_BASE_URL') or os.getenv('NGROK_URL') or 'http://localhost:8001'
                public_url = f"{base_url}/media/{filename}"
                logger.info(f"  🌐 Generated public URL: {public_url}")
            
            logger.info(f"  📋 Twilio message params:")
            logger.info(f"    From: {self.from_number}")
            logger.info(f"    To: {to}")
            logger.info(f"    Media URL: {public_url}")
            
            # Send message with media using Twilio
            logger.info("  📤 Sending message via Twilio...")
            message_obj = self.client.messages.create(
                media_url=[public_url],
                from_=self.from_number,
                to=to
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
            logger.info(f"  ✅ Audio message sent successfully: {result}")
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
            logger.info(f"    Account SID: {'✅ Present' if self.account_sid else '❌ Missing'} ({self.account_sid[:10]}... if present)")
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
                        temp_file = temp_dir / f"download_{timestamp}{ext}"
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

    # Language code mapping: existing codes -> Gemini Live codes
    GEMINI_LANGUAGE_MAP = {
        "hi": "hi-IN",  # Hindi
        "en": "en-IN",  # English (India)
        "mr": "mr-IN",  # Marathi
        "ta": "ta-IN",  # Tamil
        "bn": "en-IN",  # Bengali -> fallback to English (not in Gemini Live)
        "gu": "en-IN",  # Gujarati -> fallback to English (not in Gemini Live)
    }

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

        # Smart clarification system
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.clarification_system = SmartClarificationSystem(groq_api_key) if groq_api_key else None

        # Simple conversation history (last 5 messages per user)
        self.conversation_history = {}
        
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

    def _get_gemini_language(self, lang_code: str) -> str:
        """Convert existing language code to Gemini Live language code."""
        return self.GEMINI_LANGUAGE_MAP.get(lang_code, "en-IN")

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
    
    def _ensure_whatsapp_length_limit(self, message: str) -> str:
        """Ensure message is under Twilio's WhatsApp character limit"""
        if len(message) <= self.whatsapp_char_limit:
            return message
        
        # If message is too long, truncate it intelligently
        truncated = message[:self.whatsapp_char_limit - 50]  # Leave room for truncation note
        
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
        
        @app.post("/webhook")
        async def handle_twilio_webhook(
            From: str = Form(...),
            To: str = Form(...),
            Body: str = Form(None),
            MediaUrl0: str = Form(None),
            MediaContentType0: str = Form(None),
            NumMedia: str = Form("0")
        ):
            """Handle incoming Twilio WhatsApp messages"""
            try:
                logger.info(f"Received message from {From} to {To}")
                
                # Create TwiML response
                resp = MessagingResponse()
                
                # Process the message
                await self._process_twilio_message(From, To, Body, MediaUrl0, MediaContentType0, NumMedia)
                
                # Return empty TwiML response (we'll send responses separately)
                return Response(content=str(resp), media_type="application/xml")
                
            except Exception as e:
                logger.error(f"Webhook error: {e}")
                resp = MessagingResponse()
                resp.message("Sorry, I'm experiencing technical difficulties.")
                return Response(content=str(resp), media_type="application/xml")
        
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
        
        @app.post("/api/set_language")
        async def set_user_language(phone_number: str, language: str):
            """Set user's preferred language"""
            if language in self.stt_service.supported_languages:
                self.user_preferences[phone_number] = {"language": language}
                return {"status": "success", "language": language}
            return {"status": "error", "error": "Unsupported language"}
        
        return app
    
    async def _process_twilio_message(self, from_number: str, to_number: str, body: str, 
                                    media_url: str, media_content_type: str, num_media: str):
        """Process incoming Twilio WhatsApp message"""
        try:
            logger.info("🔄 PROCESSING MESSAGE:")
            logger.info(f"  📱 From: {from_number}")
            logger.info(f"  📍 To: {to_number}")
            logger.info(f"  💬 Body: '{body}' (present: {bool(body and body.strip())})")
            logger.info(f"  🎵 Media URL: {media_url} (present: {bool(media_url)})")
            logger.info(f"  📋 Content Type: {media_content_type}")
            logger.info(f"  🔢 Num Media: {num_media} (as int: {int(num_media) if num_media else 0})")
            
            # Remove whatsapp: prefix if present
            clean_from = from_number.replace("whatsapp:", "")
            logger.info(f"  🧹 Clean From: {clean_from}")
            
            # Decision logic with detailed logging
            has_text = body and body.strip()
            has_media = media_url and int(num_media) > 0
            
            logger.info(f"  🎯 Decision: has_text={has_text}, has_media={has_media}")
            
            if has_text:
                logger.info("  ➡️ Routing to TEXT message handler")
                await self._handle_twilio_text_message(clean_from, body.strip())
            
            elif has_media:
                logger.info("  ➡️ Routing to MEDIA message handler")
                await self._handle_twilio_media_message(clean_from, media_url, media_content_type)
            
            else:
                logger.info("  ➡️ Unknown message type, sending help message")
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
                await self._handle_medication_reminder_command(text, from_number)
                return
            elif text_lower == "/myreminders":
                await self._handle_list_reminders_command(from_number)
                return
            elif text_lower.startswith("/deletereminder"):
                await self._handle_delete_reminder_command(text, from_number)
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

            # Get user's preferred language (default to Hindi for text)
            user_lang = self.user_preferences.get(from_number, {}).get("language", "hi")

            # SMART CLARIFICATION SYSTEM
            if self.clarification_system:
                logger.info(f"  🧠 Analyzing query clarity: '{text}'")
                
                # Get conversation history for context
                history = self._get_conversation_history(from_number)
                
                # Analyze if clarification is needed
                clarity_result = await self.clarification_system.analyze_query_clarity(
                    text, from_number, history
                )
                
                logger.info(f"  📊 Clarity analysis: {clarity_result}")
                
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
                        logger.info(f"  ❓ Sent clarification question: {question}")
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
                    logger.info(f"  ✅ Using enhanced query: {enhanced_query}")
                    text = enhanced_query
                    
                    # Send acknowledgment
                    await self.twilio_api.send_text_message(
                        from_number, 
                        "✅ Thank you for the additional information! Let me find the best answer for you..."
                    )
                    await asyncio.sleep(1)
                
            # Query RAG pipeline (with potentially enhanced query)
            logger.info(f"  🔍 About to query RAG pipeline with text: '{text}', user_id: '{from_number}', language: '{user_lang}'")
            result = await self.rag_pipeline.query(text, user_id=from_number, source_language=user_lang)
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
            logger.info(f"  📱 From: {from_number}")
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
            logger.info(f"  🎤 Transcription result: {stt_result}")

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
            detected_language = stt_result.get("detected_language", "hi")

            # Update user's language preference
            self.user_preferences[from_number] = {"language": detected_language}

            # Send transcription confirmation
            lang_name = stt_result.get("language_name", "Unknown")
            confirmation_msg = f"🎯 Understood ({lang_name}): {text}"
            confirmation_msg = self._ensure_whatsapp_length_limit(confirmation_msg)
            await self.twilio_api.send_text_message(from_number, confirmation_msg)

            # Query RAG pipeline
            result = await self.rag_pipeline.query(text, user_id=from_number, source_language=detected_language)

            if result["status"] == "success":
                response_text = result["answer"]

                # Add model indicator to response text but keep under 1550 characters
                model_used = result.get("model_used", "unknown")
                model_indicator = f"\n\n(model {model_used} used)"
                
                # Ensure total message stays under 1550 chars for English
                max_response_length = 1550 - len("🇬🇧 English:\n") - len(model_indicator)
                display_text = response_text
                if len(display_text) > max_response_length:
                    display_text = display_text[:max_response_length-3] + "..."
                
                response_with_model = f"🇬🇧 English:\n{display_text}{model_indicator}"
                
                # ALWAYS LOG ENGLISH RESPONSE (regardless of Twilio success/failure)
                logger.info("  📤 MEDIA STEP 1: Preparing English response...")
                logger.info(f"  📄 ENGLISH TEXT BEING SENT:")
                logger.info(f"  📄 ****************************************************")
                logger.info(f"  📄 {response_with_model}")
                logger.info(f"  📄 ****************************************************")
                
                # Send English response first
                await self.twilio_api.send_text_message(from_number, response_with_model)
                await asyncio.sleep(1)

                # Translate and send in detected language
                if detected_language != "en":
                    translation_result = await self.rag_pipeline.translate_text(result["answer"], detected_language)  # Use original answer for translation
                    
                    if translation_result["status"] == "success":
                        translated_text = translation_result["translated_text"]
                        
                        # Language flag mapping
                        flag_map = {"hi": "🇮🇳", "bn": "🇧🇩", "ta": "🇮🇳", "gu": "🇮🇳"}
                        flag = flag_map.get(detected_language, "🌐")
                        lang_name = self.stt_service.supported_languages.get(detected_language, detected_language)
                        
                        # Add model indicator and ensure under 1550 chars for translated response
                        header = f"{flag} {lang_name}:\n"
                        max_translated_length = 1550 - len(header) - len(model_indicator)
                        if len(translated_text) > max_translated_length:
                            translated_text = translated_text[:max_translated_length-3] + "..."
                        
                        translated_with_model = f"{header}{translated_text}{model_indicator}"
                        
                        # ALWAYS LOG TRANSLATED RESPONSE (regardless of Twilio success/failure)
                        logger.info("  📤 MEDIA STEP 2: Preparing translated response...")
                        logger.info(f"  📄 TRANSLATED TEXT BEING SENT ({lang_name}):")
                        logger.info(f"  📄 ****************************************************")
                        logger.info(f"  📄 {translated_with_model}")
                        logger.info(f"  📄 ****************************************************")
                        
                        await self.twilio_api.send_text_message(from_number, translated_with_model)
                        await asyncio.sleep(1)
                        
                        # Use original translated text for audio (without model indicator)
                        audio_text = translation_result["translated_text"]
                    else:
                        # Fallback to original text if translation fails (without model indicator)
                        audio_text = result["answer"]
                else:
                    # Use original answer for English audio (without model indicator)
                    audio_text = result["answer"]

                # Generate and send audio response in the detected language
                tts_result = await self.tts_service.synthesize_speech(
                    audio_text, detected_language
                )

                if tts_result.get("audio_available"):
                    # Store audio file for serving
                    filename = Path(tts_result["audio_file"]).name
                    self.media_files[filename] = tts_result["audio_file"]
                    
                    # Set public URL
                    public_url = f"{os.getenv('PUBLIC_BASE_URL', 'http://localhost:8000')}/media/{filename}"
                    
                    await self.twilio_api.send_audio_message(
                        from_number,
                        tts_result["audio_file"],
                        public_url
                    )
            else:
                error_msg = "Sorry, I encountered an error processing your question."
                await self.twilio_api.send_text_message(from_number, error_msg)

                # Send error in detected language
                if detected_language != "en":
                    tts_result = await self.tts_service.synthesize_speech(
                        error_msg, detected_language
                    )
                    if tts_result.get("audio_available"):
                        filename = Path(tts_result["audio_file"]).name
                        self.media_files[filename] = tts_result["audio_file"]
                        public_url = f"{os.getenv('PUBLIC_BASE_URL', 'http://localhost:8000')}/media/{filename}"
                        await self.twilio_api.send_audio_message(
                            from_number,
                            tts_result["audio_file"],
                            public_url
                        )

        except Exception as e:
            logger.error(f"Error handling Twilio media message: {e}")
            await self._send_error_message(from_number)

    async def _handle_twilio_media_message_gemini(
        self, from_number: str, media_url: str, content_type: str
    ):
        """
        Handle media message using Gemini Live for real-time voice processing.

        This method provides native voice-to-voice AI processing:
        1. Downloads audio from Twilio
        2. Converts to PCM format for Gemini
        3. Gets/creates a Gemini Live session
        4. Injects RAG context for grounded responses
        5. Streams audio to Gemini and collects response
        6. Converts response audio to MP3 for WhatsApp
        7. Sends back audio and transcription
        """
        logger.info("🚀 GEMINI LIVE VOICE PROCESSING:")
        logger.info(f"  📱 From: {from_number}")
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

            # Step 3: Get user's language preference
            user_lang = self.user_preferences.get(from_number, {}).get("language", "hi")
            gemini_lang = self._get_gemini_language(user_lang)
            logger.info(f"  🌐 Language: {user_lang} -> Gemini: {gemini_lang}")

            # Step 4: Get or create Gemini session for this user
            logger.info("  🤖 Step 4: Getting/creating Gemini session...")
            session = await self.gemini_session_manager.get_or_create_session(
                user_id=from_number,
                language=gemini_lang,
                voice="Aoede"  # Warm, empathetic voice for healthcare
            )
            logger.info(f"  ✅ Session ready, is_active: {session.is_active}")

            # Step 5: Inject RAG context (get relevant medical documents)
            logger.info("  📚 Step 5: Injecting RAG context...")
            # First, we need to transcribe to get the query text for RAG
            # We'll use the STT service for this
            stt_result = await self.stt_service.transcribe_audio(audio_file_path)

            if stt_result["status"] == "success":
                query_text = stt_result["text"]
                detected_lang = stt_result.get("detected_language", user_lang)
                logger.info(f"  🎤 Transcribed query: '{query_text}'")
                logger.info(f"  🌐 Detected language: {detected_lang}")

                # Update user's language preference based on detection
                self.user_preferences[from_number] = {"language": detected_lang}
                gemini_lang = self._get_gemini_language(detected_lang)

                # Send transcription confirmation
                lang_name = stt_result.get("language_name", "Unknown")
                confirmation_msg = f"🎯 Understood ({lang_name}): {query_text}"
                confirmation_msg = self._ensure_whatsapp_length_limit(confirmation_msg)
                await self.twilio_api.send_text_message(from_number, confirmation_msg)

                # Inject RAG context using query text
                context_injected = await self.gemini_service.inject_rag_context(
                    session, query_text
                )
                if context_injected:
                    logger.info("  ✅ RAG context injected successfully")
                else:
                    logger.info("  ℹ️ No relevant RAG context found or injection skipped")
            else:
                query_text = None
                detected_lang = user_lang
                logger.warning(f"  ⚠️ STT failed, proceeding without RAG context: {stt_result}")

            # Step 6: Send audio to Gemini and collect response
            logger.info("  🎙️ Step 6: Streaming audio to Gemini...")

            # Split audio into chunks for streaming
            chunks = self.audio_handler.chunk_audio(pcm_data)
            logger.info(f"  📦 Audio split into {len(chunks)} chunks")

            # Send all audio chunks
            for i, chunk in enumerate(chunks):
                await session.send_audio(chunk)
                if (i + 1) % 10 == 0:
                    logger.info(f"  📤 Sent chunk {i + 1}/{len(chunks)}")

            logger.info("  ✅ All audio chunks sent")

            # Step 7: Receive audio response from Gemini
            logger.info("  🎧 Step 7: Receiving audio response from Gemini...")
            response_audio_chunks = []
            response_text = None

            async for response in session.receive_audio():
                if isinstance(response, bytes):
                    response_audio_chunks.append(response)
                elif isinstance(response, dict):
                    # Handle text/transcription response
                    if "text" in response:
                        response_text = response["text"]
                    elif "transcription" in response:
                        response_text = response["transcription"]

            if response_audio_chunks:
                response_audio = b"".join(response_audio_chunks)
                logger.info(f"  ✅ Received response audio: {len(response_audio)} bytes")
            else:
                logger.warning("  ⚠️ No audio response received from Gemini")
                response_audio = None

            # Step 8: Convert response audio to MP3 and send
            if response_audio:
                logger.info("  🔄 Step 8: Converting response to MP3...")
                mp3_file_path = await asyncio.to_thread(
                    self.audio_handler.convert_from_pcm,
                    response_audio,
                    "mp3"
                )
                logger.info(f"  ✅ MP3 created: {mp3_file_path}")

                # Store file for serving
                filename = Path(mp3_file_path).name
                self.media_files[filename] = mp3_file_path

                # Send audio response
                public_url = f"{os.getenv('PUBLIC_BASE_URL', 'http://localhost:8000')}/media/{filename}"
                await self.twilio_api.send_audio_message(
                    from_number,
                    mp3_file_path,
                    public_url
                )
                logger.info("  ✅ Audio response sent to WhatsApp")

            # Step 9: Send text transcription if available
            if response_text:
                logger.info("  📝 Step 9: Sending text transcription...")
                # Ensure under WhatsApp limit
                response_text = self._ensure_whatsapp_length_limit(response_text)
                await self.twilio_api.send_text_message(from_number, response_text)
                logger.info("  ✅ Text response sent")

                # Also send translation if not in English
                if detected_lang != "en":
                    translation_result = await self.rag_pipeline.translate_text(
                        response_text, detected_lang
                    )
                    if translation_result["status"] == "success":
                        translated_text = translation_result["translated_text"]
                        flag_map = {"hi": "🇮🇳", "bn": "🇧🇩", "ta": "🇮🇳", "gu": "🇮🇳", "mr": "🇮🇳"}
                        flag = flag_map.get(detected_lang, "🌐")
                        lang_name = self.stt_service.supported_languages.get(
                            detected_lang, detected_lang
                        )
                        translated_msg = f"{flag} {lang_name}:\n{translated_text}"
                        translated_msg = self._ensure_whatsapp_length_limit(translated_msg)
                        await self.twilio_api.send_text_message(from_number, translated_msg)
                        logger.info(f"  ✅ Translated response sent ({lang_name})")

            # Update conversation history
            if query_text:
                self._update_conversation_history(from_number, f"User (voice): {query_text}")
            if response_text:
                self._update_conversation_history(
                    from_number, f"Bot (Gemini Live): {response_text[:100]}..."
                )

            logger.info("  🎉 Gemini Live processing complete!")

        finally:
            # Cleanup downloaded audio file
            try:
                os.remove(audio_file_path)
                logger.info(f"  🗑️ Cleaned up: {audio_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"  ⚠️ Cleanup failed: {cleanup_error}")

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
            
            # Get user's preferred language (default to Hindi for text)
            user_lang = self.user_preferences.get(from_number, {}).get("language", "hi")
            
            # Query RAG pipeline
            await self._send_typing_indicator(from_number)
            
            result = await self.rag_pipeline.query(text, user_id=from_number, source_language=user_lang)
            
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
                await self.whatsapp_api.send_text_message(
                    from_number,
                    "Sorry, I encountered an error processing your question. Please try again."
                )
                
        except Exception as e:
            logger.error(f"Error handling text message: {e}")
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
            detected_language = stt_result.get("detected_language", "hi")
            
            # Update user's language preference
            self.user_preferences[from_number] = {"language": detected_language}
            
            # Send transcription confirmation
            lang_name = stt_result.get("language_name", "Unknown")
            await self.whatsapp_api.send_text_message(
                from_number,
                f"🎯 Understood ({lang_name}): {text}"
            )
            
            # Query RAG pipeline
            result = await self.rag_pipeline.query(text, user_id=from_number, source_language=detected_language)
            
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
            logger.error(f"Error handling audio message: {e}")
            await self._send_error_message(from_number)
    
    async def _handle_voice_message(self, message: dict, from_number: str):
        """Handle voice message (same as audio)"""
        # Voice messages are handled the same way as audio messages
        voice_data = message.get("voice", {})
        # Convert voice to audio format for processing
        audio_message = {"audio": voice_data}
        await self._handle_audio_message({"audio": voice_data}, from_number)
    
    async def _handle_language_command(self, text: str, from_number: str):
        """Handle language selection command"""
        try:
            # Parse command: /lang hi, /lang bn, etc.
            parts = text.lower().split()
            
            if len(parts) < 2:
                help_text = """🌐 Bilingual Response Mode:
Send: /lang [code]

Supported languages:
• /lang hi - Hindi (हिंदी)
• /lang bn - Bengali (বাংলা) 
• /lang ta - Tamil (தமிழ්)
• /lang gu - Gujarati (ગુજરાતી)
• /lang en - English

Example: /lang hi"""
                
                help_text = self._ensure_whatsapp_length_limit(help_text)
                await self.twilio_api.send_text_message(from_number, help_text)
                return
            
            lang_code = parts[1]
            
            if lang_code in self.stt_service.supported_languages:
                self.user_preferences[from_number] = {"language": lang_code}
                lang_name = self.stt_service.supported_languages[lang_code]
                
                success_msg = f"✅ Language set to {lang_name} ({lang_code})"
                await self.twilio_api.send_text_message(from_number, success_msg)
                
                # Send confirmation in selected language
                welcome_msgs = {
                    "hi": "अब मैं आपको हिंदी में जवाब दूंगा।",
                    "bn": "এখন আমি আপনাকে বাংলায় উত্তর দেব।",
                    "ta": "இப்போது நான் உங்களுக்கு தமிழில் பதিলளிப்পேন்।",
                    "gu": "હવે હું તમને ગુજરાતીમાં જવાબ આપીશ।",
                    "en": "I will now respond to you in English."
                }
                
                if lang_code in welcome_msgs:
                    await self.twilio_api.send_text_message(
                        from_number,
                        welcome_msgs[lang_code]
                    )
            else:
                await self.twilio_api.send_text_message(
                    from_number,
                    f"❌ Unsupported language code: {lang_code}"
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
                            await asyncio.sleep(1)
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
    
    async def _process_query_directly(self, from_number: str, query: str):
        """Process query directly without clarification checks"""
        try:
            user_lang = self.user_preferences.get(from_number, {}).get("language", "hi")
            
            logger.info(f"  🔍 Processing query directly: '{query}', language: '{user_lang}'")
            result = await self.rag_pipeline.query(query, user_id=from_number, source_language=user_lang)
            
            # Process the result normally (same logic as in main handler)
            await self._send_rag_response(from_number, result, user_lang)
            
        except Exception as e:
            logger.error(f"Error processing query directly: {e}")
            await self._send_error_message(from_number)
    
    async def _send_rag_response(self, from_number: str, result: Dict[str, Any], user_lang: str):
        """Send RAG response with bilingual support"""
        try:
            if result["status"] == "success":
                response_text = result["answer"]
                logger.info(f"  ✅ RAG Query successful, response length: {len(response_text)}")

                # Ensure response fits Twilio's WhatsApp character limit
                response_text = self._ensure_whatsapp_length_limit(response_text)

                logger.info(f"  🌐 User language preference: {user_lang}")

                # Add model indicator to response text but keep under 1550 characters
                model_used = result.get("model_used", "unknown")
                model_indicator = f"\n\n(model {model_used} used)"
                
                # Ensure total message stays under 1550 chars
                max_response_length = 1550 - len("🇬🇧 English:\n") - len(model_indicator)
                if len(response_text) > max_response_length:
                    response_text = response_text[:max_response_length-3] + "..."
                
                response_with_model = f"🇬🇧 English:\n{response_text}{model_indicator}"
                
                # Send English response FIRST  
                logger.info("  📤 STEP 1: Sending English response...")
                text_result = await self.twilio_api.send_text_message(from_number, response_with_model)
                logger.info(f"  📤 Text message result: {text_result}")
                
                # Check if text was sent successfully
                text_sent_successfully = text_result.get("status") == "success"
                if text_sent_successfully:
                    logger.info("  ✅ English message sent successfully!")
                else:
                    logger.error(f"  ❌ English message failed: {text_result}")

                # Add delay between messages
                await asyncio.sleep(1)

                # If user language is not English, translate and send in target language
                if user_lang != "en":
                    logger.info(f"  🌐 STEP 2: Translating to {user_lang}...")
                    translation_result = await self.rag_pipeline.translate_text(response_text, user_lang)
                    
                    if translation_result["status"] == "success":
                        translated_text = translation_result["translated_text"]
                        
                        # Language flag mapping
                        flag_map = {
                            "hi": "🇮🇳", "bn": "🇧🇩", "ta": "🇮🇳", "gu": "🇮🇳"
                        }
                        flag = flag_map.get(user_lang, "🌐")
                        lang_name = self.stt_service.supported_languages.get(user_lang, user_lang)
                        
                        # Add model indicator and ensure under 1550 chars
                        header = f"{flag} {lang_name}:\n"
                        max_translated_length = 1550 - len(header) - len(model_indicator)
                        if len(translated_text) > max_translated_length:
                            translated_text = translated_text[:max_translated_length-3] + "..."
                        
                        translated_with_model = f"{header}{translated_text}{model_indicator}"
                        
                        # Send translated response
                        logger.info(f"  📤 STEP 2: Sending {lang_name} response...")
                        trans_result = await self.twilio_api.send_text_message(from_number, translated_with_model)
                        logger.info(f"  📤 Translation result: {trans_result}")
                    else:
                        logger.error(f"  ❌ Translation failed: {translation_result}")

                # Update conversation history with response
                self._update_conversation_history(from_number, f"Bot: {response_text[:100]}...")
                
            else:
                logger.error(f"  ❌ RAG Query failed: {result}")
                await self.twilio_api.send_text_message(
                    from_number,
                    "I'm having trouble finding information right now. Please try again or ask a different question."
                )
        
        except Exception as e:
            logger.error(f"Error sending RAG response: {e}")
            await self._send_error_message(from_number)
    
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
        """Handle /remind command to set up medication reminders"""
        if not self.reminder_scheduler:
            await self.twilio_api.send_text_message(
                from_number,
                "Medication reminder feature is not available right now. Please try again later."
            )
            return
        
        try:
            # Parse command: /remind <medication> <time> <dosage>
            # Example: /remind Paracetamol 08:00,20:00 500mg after food
            parts = text.split(maxsplit=3)
            if len(parts) < 3:
                help_msg = """💊 Medication Reminder Setup

Usage: /remind <medication_name> <times> <dosage> [instructions]

Examples:
• /remind Paracetamol 08:00,20:00 500mg after food
• /remind Morphine 08:00,14:00,20:00 10mg with water
• /remind Ondansetron 06:00,18:00 4mg before meals

Times should be in 24-hour format (HH:MM) separated by commas."""
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

💊 {medication}
🕐 Times: {times_formatted}
📋 Dosage: {dosage_and_instructions}

I'll remind you when it's time to take your medication.
Reply 'TAKEN' after taking it."""
            
            # Add voice call info if set up
            if voice_reminders_set:
                confirm_msg += f"\n\n📞 Voice call reminders also set up for: {', '.join(voice_reminders_set)}"
            
            confirm_msg += f"""\n\nTo view all reminders: /myreminders
To delete: /deletereminder {reminder.reminder_id}"""
            
            await self.twilio_api.send_text_message(from_number, confirm_msg)
            logger.info(f"Created medication reminder {reminder.reminder_id} for {from_number}")
            
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
                    "💊 You don't have any medication reminders set up.\n\nTo create one:\n/remind <medication> <times> <dosage>"
                )
                return
            
            lines = ["💊 Your Medication Reminders\n"]
            for i, r in enumerate(reminders, 1):
                times = ", ".join(r.scheduled_times)
                status = "✅ Active" if r.active else "⏸️ Paused"
                lines.append(f"{i}. {r.medication_name}")
                lines.append(f"   🕐 {times}")
                lines.append(f"   📋 {r.dosage}")
                lines.append(f"   {status} (ID: {r.reminder_id[:8]}...)")
                lines.append("")
            
            lines.append("To delete a reminder:\n/deletereminder <ID>")
            
            message = "\n".join(lines)
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
                
                msg = f"✅ Great! I've recorded that you took your {recent_reminder.medication_name}."
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
            
            logger.info(f"Human handoff requested by {from_number}, request ID: {handoff.request_id}")
            
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
• /remind <med> <times> <dose> - Set medication reminder
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

Thank you for confirming that you took your {reminder.medication_name}.

Keep up the good work with your medication schedule! 💊"""
        
        await self.twilio_api.send_text_message(reminder.user_id, message)
        logger.info(f"Voice confirmation received for {reminder.medication_name} from {reminder.user_id}")
        
    except Exception as e:
        logger.error(f"Error sending confirmation message: {e}")


async def _on_medication_call_failed(self, reminder):
    """Callback when voice call fails"""
    try:
        message = f"""⚠️ Medication Reminder Call

We tried calling you to remind you about your {reminder.medication_name}, but couldn't reach you.

Please take your medication if you haven't already:
📋 {reminder.dosage}

Reply 'TAKEN' after taking it."""
        
        await self.twilio_api.send_text_message(reminder.user_id, message)
        logger.warning(f"Voice call failed for {reminder.medication_name} to {reminder.user_id}")
        
    except Exception as e:
        logger.error(f"Error sending failed call message: {e}")


# Add callbacks to class
EnhancedWhatsAppBot._on_medication_confirmed = _on_medication_confirmed
EnhancedWhatsAppBot._on_medication_call_failed = _on_medication_call_failed
