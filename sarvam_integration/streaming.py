"""
Sarvam AI WebSocket Streaming Client for Palli Sahayak

Real-time STT and TTS streaming via WebSocket connections.

STT Streaming: Send PCM16 audio chunks, receive transcripts
TTS Streaming: Send text chunks, receive audio bytes

Documentation: https://docs.sarvam.ai/api-reference-docs/introduction
"""

import os
import json
import logging
import asyncio
import aiohttp
from typing import Optional, AsyncIterator, AsyncGenerator

from .client import SarvamSTTResult, SARVAM_API_BASE

logger = logging.getLogger(__name__)

SARVAM_WS_STT = "wss://api.sarvam.ai/speech-to-text/streaming"
SARVAM_WS_TTS = "wss://api.sarvam.ai/text-to-speech/streaming"


class SarvamStreamingClient:
    """
    WebSocket streaming client for real-time Sarvam STT/TTS.

    Usage:
        client = SarvamStreamingClient()

        # Stream STT
        async for result in client.stream_stt(audio_chunks, language="hi-IN"):
            print(result.transcript)

        # Stream TTS
        async for audio_chunk in client.stream_tts(text_chunks, language="hi-IN"):
            play_audio(audio_chunk)
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY")
        if not self.api_key:
            logger.warning("Sarvam API key not configured for streaming")

        self.base_url = base_url or os.getenv("SARVAM_BASE_URL", SARVAM_API_BASE)

    def is_available(self) -> bool:
        """Check if streaming client is configured."""
        return bool(self.api_key)

    async def stream_stt(
        self,
        audio_chunks: AsyncIterator[bytes],
        language: str = "hi-IN",
        model: str = "saaras:v3",
    ) -> AsyncGenerator[SarvamSTTResult, None]:
        """
        Stream audio for real-time transcription.

        Connects to Sarvam STT WebSocket endpoint.
        Sends PCM16 16kHz mono audio chunks.
        Yields partial and final transcripts.

        Args:
            audio_chunks: Async iterator of PCM16 audio bytes
            language: BCP-47 language code
            model: STT model ("saaras:v3" or "saaras:flash")
        """
        if not self.is_available():
            yield SarvamSTTResult(success=False, error="Sarvam API key not configured")
            return

        ws_url = f"{SARVAM_WS_STT}?language_code={language}&model={model}"
        headers = {"api-subscription-key": self.api_key}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    ws_url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as ws:

                    async def send_audio():
                        async for chunk in audio_chunks:
                            await ws.send_bytes(chunk)
                        await ws.send_str(json.dumps({"type": "end"}))

                    send_task = asyncio.create_task(send_audio())

                    try:
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                yield SarvamSTTResult(
                                    success=True,
                                    transcript=data.get("transcript", ""),
                                    language_code=language,
                                )
                            elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                                break
                    finally:
                        send_task.cancel()
                        try:
                            await send_task
                        except asyncio.CancelledError:
                            pass

        except aiohttp.ClientError as e:
            logger.error(f"Sarvam STT streaming error: {e}")
            yield SarvamSTTResult(success=False, error=f"WebSocket error: {e}")
        except Exception as e:
            logger.error(f"Sarvam STT streaming failed: {e}")
            yield SarvamSTTResult(success=False, error=str(e))

    async def stream_tts(
        self,
        text_chunks: AsyncIterator[str],
        language: str = "hi-IN",
        voice: str = "meera",
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream text for real-time TTS audio generation.

        Connects to Sarvam TTS WebSocket endpoint.
        Sends text chunks, yields audio bytes.

        Args:
            text_chunks: Async iterator of text strings
            language: BCP-47 language code
            voice: Speaker voice name
        """
        if not self.is_available():
            return

        headers = {"api-subscription-key": self.api_key}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    SARVAM_WS_TTS,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as ws:

                    async def send_text():
                        async for text in text_chunks:
                            await ws.send_str(json.dumps({
                                "text": text,
                                "language_code": language,
                                "speaker": voice,
                            }))
                        await ws.send_str(json.dumps({"type": "end"}))

                    send_task = asyncio.create_task(send_text())

                    try:
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.BINARY:
                                yield msg.data
                            elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                                break
                    finally:
                        send_task.cancel()
                        try:
                            await send_task
                        except asyncio.CancelledError:
                            pass

        except aiohttp.ClientError as e:
            logger.error(f"Sarvam TTS streaming error: {e}")
        except Exception as e:
            logger.error(f"Sarvam TTS streaming failed: {e}")
