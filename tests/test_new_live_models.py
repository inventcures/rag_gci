"""
Live end-to-end test for the new Gemini Live models.

Requires GEMINI_API_KEY in .env and network access. Makes real Live API calls:
1. gemini-3.1-flash-live-preview  - text seed in, audio + transcript out

(The translate model is no longer offered: it has no system instruction and no
RAG, so nothing it says has the assistant's safety layer. Review 3, ticket 12.)

Usage:
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

RECEIVE_TIMEOUT = 30

# Not a pytest module. These are manual probes that open a real billed Live session
# and need credentials, so they
# cannot run in CI and must never be collected as if they could. They are run by
# hand with `python tests/test_new_live_models.py`.
#
# Collection previously failed on the missing `service` argument, which pytest read
# as a fixture request. That stayed invisible while google-genai was absent, because
# the module failed to import before collection reached it. Installing the
# dependency exposed it, which is the useful part of the failure.
__test__ = False


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
    ]

    print(f"\n{'=' * 50}\n{sum(results)}/{len(results)} live model tests passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
