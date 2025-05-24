"""
Enhanced WhatsApp Bot with Indian Language Support using Twilio
Supports text queries, voice messages, and multilingual responses
"""

import os
import json
import asyncio
import tempfile
import logging
from typing import Dict, Any, Optional
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
    """Enhanced WhatsApp bot with full Indian language support"""
    
    def __init__(self, rag_pipeline, stt_service: EnhancedSTTService, tts_service: EnhancedTTSService):
        self.rag_pipeline = rag_pipeline
        self.stt_service = stt_service
        self.tts_service = tts_service
        self.twilio_api = TwilioWhatsAppAPI()
        
        # Language preferences per user
        self.user_preferences = {}
        
        # Store for serving audio files
        self.media_files = {}
    
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
        """Handle text message from Twilio"""
        try:
            if not text:
                return

            # Check for special commands
            if text.lower().startswith("/lang"):
                await self._handle_language_command(text, from_number)
                return

            # Get user's preferred language (default to Hindi for text)
            user_lang = self.user_preferences.get(from_number, {}).get("language", "hi")

            # Query RAG pipeline
            result = await self.rag_pipeline.query(text, user_id=from_number)

            if result["status"] == "success":
                response_text = result["answer"]
                logger.info(f"  ✅ RAG Query successful, response length: {len(response_text)}")

                # Send text response FIRST
                logger.info("  📤 STEP 1: Sending text response...")
                text_result = await self.twilio_api.send_text_message(from_number, response_text)
                logger.info(f"  📤 Text message result: {text_result}")
                
                # Check if text was sent successfully
                text_sent_successfully = text_result.get("status") == "success"
                if text_sent_successfully:
                    logger.info("  ✅ Text message sent successfully!")
                else:
                    logger.error(f"  ❌ Text message failed: {text_result}")
                    # Still continue with audio, but log the issue

                # Add delay to ensure text arrives before audio
                import asyncio
                logger.info("  ⏳ Waiting 2 seconds before sending audio...")
                await asyncio.sleep(2)

                # Generate and send audio response
                logger.info(f"  📤 STEP 2: Starting TTS synthesis for language: {user_lang}")
                logger.info(f"  📝 TTS input text: {response_text[:100]}...")
                
                tts_result = await self.tts_service.synthesize_speech(response_text, user_lang)
                logger.info(f"  🎤 TTS synthesis result: {tts_result}")

                if tts_result.get("audio_available"):
                    logger.info("  ✅ TTS audio available, preparing to send...")
                    
                    # Store audio file for serving
                    filename = Path(tts_result["audio_file"]).name
                    self.media_files[filename] = tts_result["audio_file"]
                    logger.info(f"  📁 Audio file stored: {filename} -> {tts_result['audio_file']}")
                    
                    # Set public URL - check for ngrok URL first
                    base_url = os.getenv('PUBLIC_BASE_URL') or os.getenv('NGROK_URL') or 'http://localhost:8001'
                    public_url = f"{base_url}/media/{filename}"
                    logger.info(f"  🌐 Public audio URL: {public_url}")
                    
                    logger.info("  📤 Sending audio message...")
                    audio_result = await self.twilio_api.send_audio_message(
                        from_number, 
                        tts_result["audio_file"],
                        public_url
                    )
                    logger.info(f"  📤 Audio message result: {audio_result}")
                    
                    if audio_result.get("status") == "success":
                        logger.info("  ✅ Audio message sent successfully!")
                    else:
                        logger.error(f"  ❌ Audio message failed: {audio_result}")
                else:
                    logger.warning(f"  ⚠️ TTS audio not available: {tts_result}")
                
                # Summary
                logger.info(f"  📊 SUMMARY: Text sent: {text_sent_successfully}, Audio sent: {tts_result.get('audio_available', False)}")
            else:
                logger.error(f"  ❌ RAG Query failed: {result}")
                await self.twilio_api.send_text_message(
                    from_number,
                    "Sorry, I encountered an error processing your question. Please try again."
                )

        except Exception as e:
            logger.error(f"Error handling Twilio text message: {e}")
            await self._send_error_message(from_number)
    
    async def _handle_twilio_media_message(self, from_number: str, media_url: str, content_type: str):
        """Handle media message from Twilio"""
        try:
            logger.info("🎵 HANDLING MEDIA MESSAGE:")
            logger.info(f"  📱 From: {from_number}")
            logger.info(f"  🔗 Media URL: {media_url}")
            logger.info(f"  📋 Content Type: {content_type}")
            
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
            await self.twilio_api.send_text_message(
                from_number,
                f"🎯 Understood ({lang_name}): {text}"
            )

            # Query RAG pipeline
            result = await self.rag_pipeline.query(text, user_id=from_number)

            if result["status"] == "success":
                response_text = result["answer"]

                # Send text response
                await self.twilio_api.send_text_message(from_number, response_text)

                # Generate and send audio response in the same language
                tts_result = await self.tts_service.synthesize_speech(
                    response_text, detected_language
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
            
            result = await self.rag_pipeline.query(text, user_id=from_number)
            
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
            result = await self.rag_pipeline.query(text, user_id=from_number)
            
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
                help_text = """🌐 Language Selection:
Send: /lang [code]

Supported languages:
• /lang hi - Hindi (हिंदी)
• /lang bn - Bengali (বাংলা) 
• /lang ta - Tamil (தமிழ்)
• /lang gu - Gujarati (ગુજરાતી)
• /lang en - English

Example: /lang hi"""
                
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