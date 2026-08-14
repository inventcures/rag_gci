"""
Live end-to-end test for the new Gemini Live models.

Requires GEMINI_API_KEY in .env and network access. Makes real Live API calls:
1. gemini-3.1-flash-live-preview  - text seed in, audio + transcript out
2. gemini-3.5-live-translate-preview - Hindi speech in, English audio/transcript out

Usage:
    ffmpeg -y -i cache/tts/tts_hi_20250530_010521.mp3 -ar 16000 -ac 1 \
        -f s16le /tmp/test_hi_16k.pcm
    ./venv/bin/python tests/test_new_live_models.py
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from gemini_live.service import GeminiLiveService
from gemini_live.config import GeminiLiveConfig

HINDI_PCM = Path("/tmp/test_hi_16k.pcm")
CHUNK_BYTES = 4096
MAX_INPUT_SECONDS = 6
RECEIVE_TIMEOUT = 30


async def collect_responses(session, timeout: float) -> dict:
    """Drain session output until turn completes or timeout."""
    audio_bytes = 0
    chunks = 0
    turn_complete = False

    async def _drain():
        nonlocal audio_bytes, chunks, turn_complete
        async for data in session.receive_audio():
            if data == session.TURN_COMPLETE:
                turn_complete = True
                break
            if data == session.INTERRUPTED:
                continue
            if isinstance(data, bytes):
                audio_bytes += len(data)
                chunks += 1

    try:
        await asyncio.wait_for(_drain(), timeout=timeout)
    except asyncio.TimeoutError:
        pass

    return {
        "audio_bytes": audio_bytes,
        "chunks": chunks,
        "turn_complete": turn_complete,
        "input_transcript": session.get_transcription(clear=False),
        "output_transcript": session.get_response_transcription(clear=False),
    }


async def test_31_flash_live(service: GeminiLiveService) -> bool:
    print("\n=== TEST 1: gemini-3.1-flash-live-preview (assistant) ===")
    session = await service.create_session(
        session_id="test_31_flash",
        language="en-IN",
        model="gemini-3.1-flash-live-preview",
    )
    assert session.model == "gemini-3.1-flash-live-preview"
    assert not session.is_translation

    await session.connect()
    await asyncio.sleep(1)
    await session.send_text(
        "Hello, please say a one-sentence greeting so I can hear your voice."
    )

    result = await collect_responses(session, RECEIVE_TIMEOUT)
    await service.close_session("test_31_flash")

    print(f"  audio: {result['audio_bytes']} bytes in {result['chunks']} chunks")
    print(f"  turn_complete: {result['turn_complete']}")
    print(f"  model said: {result['output_transcript'][:200]!r}")

    ok = result["audio_bytes"] > 0
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


async def test_35_live_translate(service: GeminiLiveService) -> bool:
    print("\n=== TEST 2: gemini-3.5-live-translate-preview (Hindi speech -> English) ===")
    if not HINDI_PCM.exists():
        print("  SKIP: /tmp/test_hi_16k.pcm not found (run the ffmpeg command)")
        return False

    session = await service.create_session(
        session_id="test_35_translate",
        language="en-IN",
        model="gemini-3.5-live-translate-preview",
    )
    assert session.model == "gemini-3.5-live-translate-preview"
    assert session.is_translation

    await session.connect()
    await asyncio.sleep(1)

    pcm = HINDI_PCM.read_bytes()[: 16000 * 2 * MAX_INPUT_SECONDS]
    for i in range(0, len(pcm), CHUNK_BYTES):
        await session.send_audio(pcm[i : i + CHUNK_BYTES])
        await asyncio.sleep(CHUNK_BYTES / (16000 * 2))  # real-time pacing

    # Trailing silence so VAD detects end of speech
    silence = b"\x00" * CHUNK_BYTES
    for _ in range(12):
        await session.send_audio(silence)
        await asyncio.sleep(CHUNK_BYTES / (16000 * 2))

    result = await collect_responses(session, RECEIVE_TIMEOUT)
    await service.close_session("test_35_translate")

    print(f"  audio: {result['audio_bytes']} bytes in {result['chunks']} chunks")
    print(f"  heard (input transcript): {result['input_transcript'][:200]!r}")
    print(f"  translation (output transcript): {result['output_transcript'][:200]!r}")

    ok = result["audio_bytes"] > 0 or bool(result["output_transcript"].strip())
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


async def main() -> int:
    config = GeminiLiveConfig.from_yaml()
    if not config.api_key and not config.project_id:
        print("FAIL: no GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT configured")
        return 1

    service = GeminiLiveService(config=config)
    if not service.client:
        print("FAIL: GenAI client not initialized")
        return 1

    results = [
        await test_31_flash_live(service),
        await test_35_live_translate(service),
    ]

    print(f"\n{'=' * 50}\n{sum(results)}/{len(results)} live model tests passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
