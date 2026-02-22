# V47: Add Sarvam AI as a Voice Provider

**Version**: 47
**Date**: 2026-02-23
**Status**: Implementation Ready
**Author**: Palli Sahayak AI Team

---

## 1. Overview

Add Sarvam AI as a new voice provider in Palli Sahayak's multi-provider voice architecture. Sarvam is India's leading voice AI company with purpose-built Indian language ASR (Saaras v3) and TTS (Bulbul v3).

### Why Sarvam

| Dimension | Current (Bolna/Deepgram) | Sarvam |
|-----------|--------------------------|--------|
| ASR languages | 7 (hi, en, mr, ta, pa, ml, hinglish) | 22 (all scheduled Indian languages) |
| TTS languages | 4 native + Hindi fallback | 11 native with dedicated voices |
| Hosting | US-hosted (Deepgram, Cartesia) | India-hosted (lower latency from India) |
| Cost | Deepgram ~$0.60/hr | Rs. 30/hr (~$0.36/hr) |
| Punjabi/Malayalam TTS | Hindi fallback (no native) | Native voices (pa-IN, ml-IN via Bulbul v3) |

### Scope

- Additive change: Sarvam joins existing providers (Bolna, Gemini Live, Retell, Fallback)
- Sarvam provides STT + TTS + Translation (not a full telephony platform like Bolna/Retell)
- Primary use: **fallback pipeline enhancement** (replace Groq Whisper + Edge TTS with Sarvam STT + TTS) and **standalone API endpoints**
- Can also be used as the STT/TTS backend within Bolna via Pipecat integration (future)

---

## 2. Sarvam API Reference

### 2.1 Authentication

All requests use header: `api-subscription-key: {SARVAM_API_KEY}`

Exception: Chat completions endpoint uses `Authorization: Bearer {token}` (not used in this integration).

### 2.2 Speech-to-Text (Saaras v3)

**Endpoint**: `POST https://api.sarvam.ai/speech-to-text`
**Content-Type**: `multipart/form-data`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | binary | yes | Audio file (WAV, MP3, FLAC, OGG, WebM) |
| `language_code` | string | yes | BCP-47 code (e.g. `hi-IN`) |
| `model` | string | no | `saaras:v3` (default) or `saaras:flash` (faster, lower accuracy) |
| `with_timestamps` | boolean | no | Word-level timestamps |
| `mode` | string | no | `formal` (default), `code-mixed`, `spoken-form` |

**Response**:
```json
{
  "transcript": "string",
  "language_code": "hi-IN",
  "timestamps": [{"word": "namaste", "start": 0.0, "end": 0.5}]
}
```

**Supported Languages** (22):
```
hi-IN, bn-IN, kn-IN, ml-IN, mr-IN, od-IN, pa-IN, ta-IN, te-IN, en-IN, gu-IN,
as-IN, brx-IN, doi-IN, gom-IN, ks-IN, kok-IN, mai-IN, mni-IN, ne-IN, sa-IN, sd-IN
```

### 2.3 Speech-to-Text Streaming

**Endpoint**: `wss://api.sarvam.ai/speech-to-text/streaming`

Query params: `language_code`, `model`
Header: `api-subscription-key`
Send: PCM16 audio chunks (16kHz mono)
Receive: JSON frames with `transcript`, `is_final`

### 2.4 Speech-to-Text Batch

**Endpoint**: `POST https://api.sarvam.ai/speech-to-text-batch`
**Content-Type**: `application/json`

```json
{
  "url": "https://example.com/audio.wav",
  "language_code": "hi-IN",
  "model": "saaras:v3",
  "with_diarization": true,
  "num_speakers": 2
}
```

For files up to 1 hour. Returns job ID for polling.

### 2.5 Text-to-Speech (Bulbul v3)

**Endpoint**: `POST https://api.sarvam.ai/text-to-speech`
**Content-Type**: `application/json`

```json
{
  "inputs": ["Text to speak"],
  "target_language_code": "hi-IN",
  "speaker": "meera",
  "model": "bulbul:v3",
  "pace": 1.0,
  "pitch": 0,
  "loudness": 1.5,
  "sample_rate": 22050,
  "enable_preprocessing": true
}
```

**Response**:
```json
{
  "audios": ["base64_encoded_wav_audio"]
}
```

**Supported Languages** (11): `hi-IN, bn-IN, kn-IN, ml-IN, mr-IN, od-IN, pa-IN, ta-IN, te-IN, en-IN, gu-IN`

**Voices by Language**:

| Language | Female Voice | Male Voice |
|----------|-------------|------------|
| hi-IN | meera | amol |
| bn-IN | diya | arnab |
| kn-IN | pavithra | raghav |
| ml-IN | ammu | arjun |
| mr-IN | ananya | advait |
| od-IN | shruti | manash |
| pa-IN | divya | gurpreet |
| ta-IN | thara | aravind |
| te-IN | lakshmi | karthik |
| en-IN | sita | neel |
| gu-IN | riddhi | chirag |

**TTS Controls**:
- `pace`: 0.5-2.0 (default 1.0, slower for elderly patients)
- `pitch`: -10 to 10 (default 0)
- `loudness`: 0.5-3.0 (default 1.5)
- `sample_rate`: 8000, 16000, 22050, 24000, 44100, 48000
- `enable_preprocessing`: Normalizes numbers, dates, abbreviations

### 2.6 Text-to-Speech Streaming

**Endpoint**: `wss://api.sarvam.ai/text-to-speech/streaming`

Send: JSON `{"text": "chunk", "language_code": "hi-IN", "speaker": "meera"}`
Receive: Binary audio chunks (WAV)

### 2.7 Translation

**Endpoint**: `POST https://api.sarvam.ai/translate`
**Content-Type**: `application/json`

```json
{
  "input": "How are you feeling today?",
  "source_language_code": "en-IN",
  "target_language_code": "hi-IN",
  "mode": "formal",
  "enable_preprocessing": true
}
```

**Response**:
```json
{
  "translated_text": "आज आप कैसा महसूस कर रहे हैं?"
}
```

### 2.8 Pricing

| Service | Cost |
|---------|------|
| STT (Saaras v3) | Rs. 30/hr (~$0.36/hr) |
| STT (Saaras Flash) | Rs. 15/hr (~$0.18/hr) |
| TTS (Bulbul v3) | Rs. 30/10K chars |
| Translation | Rs. 0.5/request |
| LLM (Sarvam-M) | Free |
| Free credits | Rs. 1000 on signup |

### 2.9 Rate Limits

| Tier | Rate Limit |
|------|------------|
| Starter (free) | 60 req/min |
| Pro | 200 req/min |
| Business | 1000 req/min |

### 2.10 Error Codes

| Code | Meaning |
|------|---------|
| 401 | Invalid API key |
| 402 | Insufficient credits |
| 413 | Audio file too large (>30MB for STT) |
| 422 | Invalid parameters |
| 429 | Rate limit exceeded |
| 500 | Server error |

---

## 3. Module Structure

```
sarvam_integration/
  __init__.py          # Module exports
  client.py            # SarvamClient: STT, TTS, translate, batch STT
  config.py            # Language configs, voice map, system prompt, env config
  streaming.py         # SarvamStreamingClient: WebSocket STT/TTS streaming
  webhooks.py          # SarvamWebhookHandler: event processing, call records
```

---

## 4. File Specifications

### 4.1 `sarvam_integration/__init__.py`

```python
"""
Sarvam AI Integration Module for Palli Sahayak Voice AI Helpline

Provides:
- SarvamClient: API client for Sarvam AI (STT, TTS, Translation)
- SarvamStreamingClient: WebSocket streaming for real-time STT/TTS
- SarvamWebhookHandler: Webhook event processing

Documentation: https://docs.sarvam.ai/api-reference-docs/introduction
"""

from .client import SarvamClient, SarvamSTTResult, SarvamTTSResult, SarvamTranslateResult
from .config import (
    get_sarvam_config_from_env,
    SARVAM_SYSTEM_PROMPT,
    SARVAM_LANGUAGE_CONFIGS,
    SARVAM_VOICE_MAP,
    SARVAM_STT_LANGUAGES,
    SARVAM_TTS_LANGUAGES,
)
from .streaming import SarvamStreamingClient
from .webhooks import SarvamWebhookHandler, SarvamCallRecord

__all__ = [
    "SarvamClient",
    "SarvamSTTResult",
    "SarvamTTSResult",
    "SarvamTranslateResult",
    "SarvamStreamingClient",
    "get_sarvam_config_from_env",
    "SARVAM_SYSTEM_PROMPT",
    "SARVAM_LANGUAGE_CONFIGS",
    "SARVAM_VOICE_MAP",
    "SARVAM_STT_LANGUAGES",
    "SARVAM_TTS_LANGUAGES",
    "SarvamWebhookHandler",
    "SarvamCallRecord",
]

__version__ = "1.0.0"
```

### 4.2 `sarvam_integration/client.py`

**Pattern**: Follows `bolna_integration/client.py`

```python
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

        # Speech-to-text
        result = await client.speech_to_text(audio_bytes, language="hi-IN")

        # Text-to-speech
        result = await client.text_to_speech("namaste", language="hi-IN", voice="meera")

        # Translation
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

        Returns:
            SarvamSTTResult
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
            model: "bulbul:v3"
            pace: Speech rate 0.5-2.0 (lower = slower, good for elderly)
            pitch: Pitch adjustment -10 to 10
            loudness: Volume 0.5-3.0
            sample_rate: Output sample rate (8000-48000)
            enable_preprocessing: Normalize numbers/dates/abbreviations

        Returns:
            SarvamTTSResult with base64-encoded WAV audio
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

        Returns:
            SarvamTranslateResult
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
            # Use a minimal TTS request as health check
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
```

### 4.3 `sarvam_integration/config.py`

**Pattern**: Follows `bolna_integration/config.py`

Key constants:

```python
SARVAM_STT_LANGUAGES = [
    "hi-IN", "bn-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN", "pa-IN",
    "ta-IN", "te-IN", "en-IN", "gu-IN", "as-IN", "brx-IN", "doi-IN",
    "gom-IN", "ks-IN", "kok-IN", "mai-IN", "mni-IN", "ne-IN", "sa-IN", "sd-IN",
]

SARVAM_TTS_LANGUAGES = [
    "hi-IN", "bn-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN", "pa-IN",
    "ta-IN", "te-IN", "en-IN", "gu-IN",
]

SARVAM_VOICE_MAP = {
    "hi-IN": {"female": "meera", "male": "amol"},
    "bn-IN": {"female": "diya", "male": "arnab"},
    "kn-IN": {"female": "pavithra", "male": "raghav"},
    "ml-IN": {"female": "ammu", "male": "arjun"},
    "mr-IN": {"female": "ananya", "male": "advait"},
    "od-IN": {"female": "shruti", "male": "manash"},
    "pa-IN": {"female": "divya", "male": "gurpreet"},
    "ta-IN": {"female": "thara", "male": "aravind"},
    "te-IN": {"female": "lakshmi", "male": "karthik"},
    "en-IN": {"female": "sita", "male": "neel"},
    "gu-IN": {"female": "riddhi", "male": "chirag"},
}
```

`SARVAM_LANGUAGE_CONFIGS` maps all 22 STT languages + 11 TTS languages with:
- `name`: Display name
- `stt_supported`: bool
- `tts_supported`: bool
- `default_voice`: From SARVAM_VOICE_MAP (female default for warmth)
- `welcome_message`: Native script greeting
- `tts_fallback_language`: For STT-only languages, which TTS language to fall back to (e.g. `mai-IN` -> `hi-IN`)

`SARVAM_SYSTEM_PROMPT`: Reuse `PALLI_SAHAYAK_SYSTEM_PROMPT` from `bolna_integration/config.py` with Sarvam-specific additions (22 language support note).

`get_sarvam_config_from_env()`: Reads `SARVAM_API_KEY`, `SARVAM_BASE_URL`, `SARVAM_DEFAULT_LANGUAGE`, `SARVAM_TTS_VOICE`, `SARVAM_TTS_PACE`, `SARVAM_STT_MODEL`.

### 4.4 `sarvam_integration/streaming.py`

WebSocket streaming client for real-time STT and TTS.

```python
class SarvamStreamingClient:
    """WebSocket streaming for real-time Sarvam STT/TTS."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        # Same init pattern as SarvamClient

    async def stream_stt(
        self,
        audio_chunks: AsyncIterator[bytes],
        language: str = "hi-IN",
        model: str = "saaras:v3",
    ) -> AsyncGenerator[SarvamSTTResult, None]:
        """
        Stream audio for real-time transcription.

        Connects to wss://api.sarvam.ai/speech-to-text/streaming
        Sends PCM16 16kHz mono audio chunks.
        Yields partial and final transcripts.
        """

    async def stream_tts(
        self,
        text_chunks: AsyncIterator[str],
        language: str = "hi-IN",
        voice: str = "meera",
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream text for real-time TTS.

        Connects to wss://api.sarvam.ai/text-to-speech/streaming
        Sends text chunks.
        Yields audio bytes.
        """
```

Uses `aiohttp.ClientSession().ws_connect()` for WebSocket connections.

### 4.5 `sarvam_integration/webhooks.py`

**Pattern**: Follows `bolna_integration/webhooks.py`

```python
@dataclass
class SarvamCallRecord:
    """Record of a Sarvam-powered voice session."""
    session_id: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_seconds: int = 0
    transcript: str = ""
    language: str = "hi-IN"
    stt_model: str = "saaras:v3"
    tts_voice: str = "meera"
    status: str = "in_progress"


class SarvamWebhookHandler:
    """Handler for Sarvam-related events (call analytics, session tracking)."""

    def __init__(self):
        self.active_sessions: Dict[str, SarvamCallRecord] = {}
        self.completed_sessions: Dict[str, SarvamCallRecord] = {}
        self.event_handlers: Dict[str, Callable] = {}

    def register_handler(self, event_type: str, handler: Callable): ...
    async def handle_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]: ...
    async def _handle_session_started(self, data): ...
    async def _handle_session_ended(self, data): ...
    async def _handle_transcription(self, data): ...
    def get_session_stats(self) -> Dict[str, Any]: ...
    def get_recent_sessions(self, limit: int = 10) -> list: ...
```

---

## 5. Integration Points

### 5.1 `voice_router.py` Changes

**1. Add SARVAM to VoiceProvider enum** (line 24-29):
```python
class VoiceProvider(Enum):
    BOLNA = "bolna"
    GEMINI_LIVE = "gemini_live"
    RETELL = "retell"
    SARVAM = "sarvam"                    # NEW
    FALLBACK_PIPELINE = "fallback_pipeline"
```

**2. Add SARVAM case to `route_voice_request()`** (after line 282):
```python
elif provider == VoiceProvider.SARVAM:
    return await self._handle_sarvam_request(
        user_id=user_id,
        language=language,
        **kwargs
    )
```

**3. Add `_handle_sarvam_request()` method** (follows `_handle_fallback_request` pattern):

The Sarvam handler implements a STT -> RAG -> LLM -> TTS pipeline using Sarvam for STT and TTS:
1. Receive audio from user
2. Sarvam STT (Saaras v3) -> transcript
3. Feed transcript to RAG pipeline (reuse existing `rag_pipeline`)
4. LLM generates response
5. Sarvam TTS (Bulbul v3) -> audio response

**4. Update `select_provider()`**: Add Sarvam as an option in the priority chain. Sarvam slots between Retell and Fallback.

**5. Add Sarvam availability check in `__init__`**: Try to import and instantiate `SarvamClient`, set `self.sarvam_available`.

### 5.2 `voice_safety_wrapper.py` Changes

**Add `SarvamSafetyIntegration` class** (follows `BolnaSafetyIntegration` at line 603):

```python
class SarvamSafetyIntegration:
    """Safety integration for Sarvam voice provider."""

    def __init__(self, safety_wrapper: 'VoiceSafetyWrapper'):
        self.safety_wrapper = safety_wrapper

    async def process_stt_result(self, transcript: str, language: str) -> Dict[str, Any]:
        """Run safety checks on STT transcript before RAG."""
        return await self.safety_wrapper.check_voice_query(transcript, language)

    async def process_tts_input(self, text: str, language: str) -> str:
        """Optimize response text for TTS output."""
        return await self.safety_wrapper.optimize_for_voice(text, language)
```

### 5.3 `config.yaml` Changes

Add after `bolna:` section (line ~215):

```yaml
# Sarvam AI Voice Configuration (Indian Language STT/TTS)
sarvam:
  enabled: true
  api_key: ${SARVAM_API_KEY}
  base_url: "https://api.sarvam.ai"

  # STT (Saaras v3) settings
  stt:
    model: "saaras:v3"
    mode: "formal"
    with_timestamps: true

  # TTS (Bulbul v3) settings
  tts:
    model: "bulbul:v3"
    default_voice: "meera"
    voice_gender: "female"
    sample_rate: 22050
    pace: 1.0
    pitch: 0
    loudness: 1.5
    enable_preprocessing: true

  # Language settings
  default_language: "hi-IN"
  supported_languages:
    - "hi-IN"
    - "en-IN"
    - "bn-IN"
    - "kn-IN"
    - "ml-IN"
    - "mr-IN"
    - "od-IN"
    - "pa-IN"
    - "ta-IN"
    - "te-IN"
    - "gu-IN"
```

Update `voice_router:` section to include sarvam:

```yaml
voice_router:
  preferred_provider: "bolna"
  phone_primary: "bolna"
  phone_fallback: "retell"
  web_primary: "gemini_live"
  web_fallback: "sarvam"          # NEW: Sarvam as web fallback
  stt_tts_provider: "sarvam"      # NEW: Sarvam for STT/TTS in fallback pipeline
  ultimate_fallback: "stt_rag_tts"
```

### 5.4 `.env.example` Changes

Add after Bolna section (line ~41):

```bash
# SARVAM AI (Optional - Indian language voice AI with 22 language ASR)
# Sign up: https://dashboard.sarvam.ai/
# Docs: https://docs.sarvam.ai/api-reference-docs/introduction
SARVAM_API_KEY=your_sarvam_api_key_here
SARVAM_BASE_URL=https://api.sarvam.ai
SARVAM_DEFAULT_LANGUAGE=hi-IN
SARVAM_TTS_VOICE=meera
SARVAM_TTS_PACE=1.0
SARVAM_STT_MODEL=saaras:v3
```

### 5.5 `simple_rag_server.py` Changes

**1. Initialization at startup** (follows Bolna pattern ~line 4472):

```python
# Initialize Sarvam AI
sarvam_enabled = False
sarvam_client = None
sarvam_webhook_handler = None
try:
    from sarvam_integration import SarvamClient, SarvamWebhookHandler
    sarvam_client = SarvamClient()
    if sarvam_client.is_available():
        sarvam_enabled = True
        sarvam_webhook_handler = SarvamWebhookHandler()
        logger.info("Sarvam AI initialized successfully")
    else:
        logger.info("Sarvam AI not configured (no API key)")
except ImportError:
    logger.info("Sarvam AI module not available")
except Exception as e:
    logger.warning(f"Sarvam AI init failed: {e}")
```

**2. API endpoints**:

```
POST /api/sarvam/stt          - Speech-to-text (accepts audio file upload)
POST /api/sarvam/tts          - Text-to-speech (returns base64 audio)
POST /api/sarvam/translate    - Text translation
GET  /api/sarvam/health       - Health check
POST /api/sarvam/webhook      - Webhook handler
GET  /api/sarvam/stats        - Session statistics
GET  /api/sarvam/languages    - List supported languages
```

All endpoints gated behind `sarvam_enabled` flag, returning 503 if not enabled.

---

## 6. Language Code Mapping

Sarvam uses BCP-47 codes (e.g. `hi-IN`), while Bolna uses short codes (e.g. `hi`). The voice router needs a mapping:

| Short Code | Sarvam Code | Language | STT | TTS |
|------------|-------------|----------|-----|-----|
| hi | hi-IN | Hindi | Y | Y |
| en | en-IN | English | Y | Y |
| mr | mr-IN | Marathi | Y | Y |
| ta | ta-IN | Tamil | Y | Y |
| bn | bn-IN | Bengali | Y | Y |
| kn | kn-IN | Kannada | Y | Y |
| ml | ml-IN | Malayalam | Y | Y |
| te | te-IN | Telugu | Y | Y |
| gu | gu-IN | Gujarati | Y | Y |
| pa | pa-IN | Punjabi | Y | Y |
| od | od-IN | Odia | Y | Y |
| as | as-IN | Assamese | Y | N* |
| mai | mai-IN | Maithili | Y | N* |
| ne | ne-IN | Nepali | Y | N* |
| kok | kok-IN | Konkani | Y | N* |
| doi | doi-IN | Dogri | Y | N* |
| brx | brx-IN | Bodo | Y | N* |
| gom | gom-IN | Goan Konkani | Y | N* |
| ks | ks-IN | Kashmiri | Y | N* |
| mni | mni-IN | Manipuri | Y | N* |
| sa | sa-IN | Sanskrit | Y | N* |
| sd | sd-IN | Sindhi | Y | N* |

*N = TTS not supported, falls back to Hindi TTS

Add helper function in `sarvam_integration/config.py`:

```python
def sarvam_language_code(short_code: str) -> str:
    """Convert short language code to Sarvam BCP-47 format."""
    mapping = {"hi": "hi-IN", "en": "en-IN", "mr": "mr-IN", ...}
    return mapping.get(short_code, f"{short_code}-IN")

def short_language_code(sarvam_code: str) -> str:
    """Convert Sarvam BCP-47 code to short format."""
    return sarvam_code.split("-")[0]
```

---

## 7. Implementation Order

Execute in this order (each step depends on the previous):

1. `sarvam_integration/__init__.py` - Module shell
2. `sarvam_integration/config.py` - Language configs, voice map, env loading
3. `sarvam_integration/client.py` - Core API client (STT, TTS, translate, batch)
4. `sarvam_integration/streaming.py` - WebSocket streaming
5. `sarvam_integration/webhooks.py` - Event handling
6. `voice_router.py` - Add SARVAM provider + routing
7. `voice_safety_wrapper.py` - Add SarvamSafetyIntegration
8. `config.yaml` - Add sarvam section + update voice_router section
9. `.env.example` - Add Sarvam env vars
10. `simple_rag_server.py` - Add Sarvam startup init + endpoints

---

## 8. Testing

### Import test
```bash
python -c "from sarvam_integration import SarvamClient, SARVAM_LANGUAGE_CONFIGS; print(f'OK: {len(SARVAM_LANGUAGE_CONFIGS)} languages')"
```

### Config test
```bash
python -c "from sarvam_integration.config import SARVAM_STT_LANGUAGES, SARVAM_TTS_LANGUAGES, SARVAM_VOICE_MAP; assert len(SARVAM_STT_LANGUAGES) == 22; assert len(SARVAM_TTS_LANGUAGES) == 11; assert len(SARVAM_VOICE_MAP) == 11; print('Config OK')"
```

### Voice router test
```bash
python -c "from voice_router import VoiceProvider; assert VoiceProvider.SARVAM.value == 'sarvam'; print('Router OK')"
```

### Safety wrapper test
```bash
python -c "from voice_safety_wrapper import SarvamSafetyIntegration; print('Safety OK')"
```

### Server import test
```bash
python -c "import simple_rag_server; print('Server OK')"
```
