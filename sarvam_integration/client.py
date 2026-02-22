"""
Sarvam AI API Client for Palli Sahayak Voice AI Helpline

Handles:
- Speech-to-Text (Saaras v3): 22 Indian languages
- Text-to-Speech (Bulbul v3): 11 Indian languages, 30+ voices
- Translation: 22 language pairs
- Batch STT: Files up to 1 hour with diarization

Documentation: https://docs.sarvam.ai/api-reference-docs/introduction
"""

import os
import io
import base64
import logging
import aiohttp
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SARVAM_API_BASE = "https://api.sarvam.ai"


@dataclass
class SarvamSTTResult:
    """Result of a Sarvam speech-to-text request."""
    success: bool
    transcript: str = ""
    language_code: str = ""
    timestamps: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


@dataclass
class SarvamTTSResult:
    """Result of a Sarvam text-to-speech request."""
    success: bool
    audio_base64: str = ""
    sample_rate: int = 22050
    error: Optional[str] = None

    def get_audio_bytes(self) -> bytes:
        """Decode base64 audio to bytes."""
        if self.audio_base64:
            return base64.b64decode(self.audio_base64)
        return b""


@dataclass
class SarvamTranslateResult:
    """Result of a Sarvam translation request."""
    success: bool
    translated_text: str = ""
    source_language: str = ""
    target_language: str = ""
    error: Optional[str] = None


class SarvamClient:
    """
    Client for Sarvam AI API.

    Handles STT (Saaras v3), TTS (Bulbul v3), and Translation.

    Usage:
        client = SarvamClient()
        result = await client.speech_to_text(audio_bytes, language="hi-IN")
        result = await client.text_to_speech("namaste", language="hi-IN", voice="meera")
        result = await client.translate("hello", source="en-IN", target="hi-IN")
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY")
        if not self.api_key:
            logger.warning("Sarvam API key not configured - set SARVAM_API_KEY env var")

        self.base_url = base_url or os.getenv("SARVAM_BASE_URL", SARVAM_API_BASE)
        self.headers = {
            "api-subscription-key": self.api_key or ""
        }

    def is_available(self) -> bool:
        """Check if Sarvam client is configured with API key."""
        return bool(self.api_key)

    async def speech_to_text(
        self,
        audio_data: bytes,
        language: str = "hi-IN",
        model: str = "saaras:v3",
        mode: str = "formal",
        with_timestamps: bool = True,
    ) -> SarvamSTTResult:
        """
        Convert speech to text using Saaras v3.

        Args:
            audio_data: Audio bytes (WAV, MP3, FLAC, OGG, WebM)
            language: BCP-47 language code (e.g. "hi-IN")
            model: "saaras:v3" (accurate) or "saaras:flash" (fast)
            mode: "formal", "code-mixed", or "spoken-form"
            with_timestamps: Include word-level timestamps
        """
        if not self.is_available():
            return SarvamSTTResult(success=False, error="Sarvam API key not configured")

        try:
            form_data = aiohttp.FormData()
            form_data.add_field(
                "file", io.BytesIO(audio_data),
                filename="audio.wav",
                content_type="audio/wav"
            )
            form_data.add_field("language_code", language)
            form_data.add_field("model", model)
            form_data.add_field("mode", mode)
            form_data.add_field("with_timestamps", str(with_timestamps).lower())

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/speech-to-text",
                    headers=self.headers,
                    data=form_data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    data = await response.json()

                    if response.status == 200:
                        return SarvamSTTResult(
                            success=True,
                            transcript=data.get("transcript", ""),
                            language_code=data.get("language_code", language),
                            timestamps=data.get("timestamps"),
                        )
                    else:
                        error_msg = data.get("error", data.get("message", f"HTTP {response.status}"))
                        logger.error(f"Sarvam STT failed: {error_msg}")
                        return SarvamSTTResult(success=False, error=error_msg)

        except aiohttp.ClientError as e:
            logger.error(f"Network error in Sarvam STT: {e}")
            return SarvamSTTResult(success=False, error=f"Network error: {e}")
        except Exception as e:
            logger.error(f"Sarvam STT failed: {e}")
            return SarvamSTTResult(success=False, error=str(e))

    async def text_to_speech(
        self,
        text: str,
        language: str = "hi-IN",
        voice: str = "meera",
        model: str = "bulbul:v3",
        pace: float = 1.0,
        pitch: int = 0,
        loudness: float = 1.5,
        sample_rate: int = 22050,
        enable_preprocessing: bool = True,
    ) -> SarvamTTSResult:
        """
        Convert text to speech using Bulbul v3.

        Args:
            text: Text to synthesize
            language: BCP-47 language code
            voice: Speaker name (e.g. "meera", "amol")
            model: TTS model name
            pace: Speech rate 0.5-2.0 (lower = slower, good for elderly)
            pitch: Pitch adjustment -10 to 10
            loudness: Volume 0.5-3.0
            sample_rate: Output sample rate (8000-48000)
            enable_preprocessing: Normalize numbers/dates/abbreviations
        """
        if not self.is_available():
            return SarvamTTSResult(success=False, error="Sarvam API key not configured")

        try:
            payload = {
                "inputs": [text],
                "target_language_code": language,
                "speaker": voice,
                "model": model,
                "pace": pace,
                "pitch": pitch,
                "loudness": loudness,
                "sample_rate": sample_rate,
                "enable_preprocessing": enable_preprocessing,
            }

            headers = {**self.headers, "Content-Type": "application/json"}

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/text-to-speech",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    data = await response.json()

                    if response.status == 200:
                        audios = data.get("audios", [])
                        audio_b64 = audios[0] if audios else ""
                        return SarvamTTSResult(
                            success=True,
                            audio_base64=audio_b64,
                            sample_rate=sample_rate,
                        )
                    else:
                        error_msg = data.get("error", data.get("message", f"HTTP {response.status}"))
                        logger.error(f"Sarvam TTS failed: {error_msg}")
                        return SarvamTTSResult(success=False, error=error_msg)

        except aiohttp.ClientError as e:
            logger.error(f"Network error in Sarvam TTS: {e}")
            return SarvamTTSResult(success=False, error=f"Network error: {e}")
        except Exception as e:
            logger.error(f"Sarvam TTS failed: {e}")
            return SarvamTTSResult(success=False, error=str(e))

    async def translate(
        self,
        text: str,
        source_language: str = "en-IN",
        target_language: str = "hi-IN",
        mode: str = "formal",
        enable_preprocessing: bool = True,
    ) -> SarvamTranslateResult:
        """
        Translate text between Indian languages.

        Args:
            text: Text to translate
            source_language: Source BCP-47 code
            target_language: Target BCP-47 code
            mode: "formal" or "colloquial"
            enable_preprocessing: Normalize input text
        """
        if not self.is_available():
            return SarvamTranslateResult(success=False, error="Sarvam API key not configured")

        try:
            payload = {
                "input": text,
                "source_language_code": source_language,
                "target_language_code": target_language,
                "mode": mode,
                "enable_preprocessing": enable_preprocessing,
            }

            headers = {**self.headers, "Content-Type": "application/json"}

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/translate",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    data = await response.json()

                    if response.status == 200:
                        return SarvamTranslateResult(
                            success=True,
                            translated_text=data.get("translated_text", ""),
                            source_language=source_language,
                            target_language=target_language,
                        )
                    else:
                        error_msg = data.get("error", data.get("message", f"HTTP {response.status}"))
                        logger.error(f"Sarvam translate failed: {error_msg}")
                        return SarvamTranslateResult(success=False, error=error_msg)

        except Exception as e:
            logger.error(f"Sarvam translate failed: {e}")
            return SarvamTranslateResult(success=False, error=str(e))

    async def batch_speech_to_text(
        self,
        file_url: str,
        language: str = "hi-IN",
        model: str = "saaras:v3",
        with_diarization: bool = False,
        num_speakers: int = 2,
    ) -> Dict[str, Any]:
        """
        Batch STT for long audio files (up to 1 hour).

        Args:
            file_url: Public URL of the audio file
            language: BCP-47 language code
            model: STT model
            with_diarization: Enable speaker diarization
            num_speakers: Expected number of speakers

        Returns:
            Dict with job_id for polling
        """
        if not self.is_available():
            return {"success": False, "error": "Sarvam API key not configured"}

        try:
            payload = {
                "url": file_url,
                "language_code": language,
                "model": model,
                "with_diarization": with_diarization,
                "num_speakers": num_speakers,
            }

            headers = {**self.headers, "Content-Type": "application/json"}

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/speech-to-text-batch",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    data = await response.json()

                    if response.status in (200, 202):
                        return {"success": True, "job_id": data.get("job_id"), "data": data}
                    else:
                        error_msg = data.get("error", f"HTTP {response.status}")
                        return {"success": False, "error": error_msg}

        except Exception as e:
            logger.error(f"Sarvam batch STT failed: {e}")
            return {"success": False, "error": str(e)}

    async def health_check(self) -> bool:
        """Check if Sarvam API is accessible."""
        if not self.is_available():
            return False

        try:
            headers = {**self.headers, "Content-Type": "application/json"}
            payload = {
                "inputs": ["test"],
                "target_language_code": "en-IN",
                "speaker": "sita",
                "model": "bulbul:v3",
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/text-to-speech",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200
        except Exception:
            return False
