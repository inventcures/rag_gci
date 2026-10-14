#!/usr/bin/env python3
"""
Gemini Live: time from the end of the caller's first utterance to first audio.

Each question is spoken by Gemini TTS (resampled to 16 kHz PCM) and streamed to a real Live
session in real time (20 ms chunks), followed by silence, as a phone caller
would. The session runs the real GeminiLiveService with the RAG tool, the
real answer path (voice prompt, Dose Boundary) and the live answer model;
retrieval returns the top chunks of the live index by word overlap (query
embedding needs model weights).

Two variants, alternated per question:

  before   the session connects when the first audio arrives, and the
           system prompt has no acknowledgement allowance
  after    the session connects on config (here 2 s before speech) and the
           prompt allows a short acknowledgement before the tool result

Reported: first audio after the end of speech, whether the RAG tool ran in
the turn, and whether that first audio came before the tool call (an
acknowledgement).

    GEMINI_API_KEY=... python scripts/benchmark_live_first_audio.py \\
        --out docs/latency/live-first-audio-14.md
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_retell_stream import pipeline  # noqa: E402
from benchmark_thinking import fmt, load_chunks, pct  # noqa: E402

QUESTIONS = [
    "My mother has cancer and is in pain at night. What can we do?",
    "My father is breathless when he lies down. How can I help him?",
    "She has not passed stool for four days after starting morphine. What should I do?",
    "He is very restless and confused in the evenings. Is this normal?",
    "Her mouth is very dry and painful. How do we care for it at home?",
    "He is vomiting after every meal. What can help?",
]
ACK_LINE = "- Before calling the tool you may say one short acknowledgement"
CHUNK_MS = 20
RATE = 16000
_TTS_CLIENT = None
ACK_OVERRIDE = None  # --ack-line: try another acknowledgement instruction


async def speech(text: str) -> bytes:
    """The question as 16 kHz PCM, spoken by Gemini TTS (resampled from 24 kHz)."""
    import audioop

    from google import genai
    from google.genai import types

    global _TTS_CLIENT
    if _TTS_CLIENT is None:
        _TTS_CLIENT = genai.Client()  # kept: a collected client closes its transport
    response = await asyncio.to_thread(
        _TTS_CLIENT.models.generate_content,
        model="gemini-2.5-flash-preview-tts", contents=f"Say: {text}",
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")))))
    pcm24 = response.candidates[0].content.parts[0].inline_data.data
    return audioop.ratecv(pcm24, 2, 1, 24000, RATE, None)[0]


async def trial(service, pcm: bytes, variant: str):
    import gemini_live.service as live

    original = live.GeminiLiveService._build_system_instruction
    replacement = None if variant == "before" else ACK_OVERRIDE
    if variant == "before" or ACK_OVERRIDE:
        def swap_ack(self, language, custom_instruction=None):
            text = original(self, language, custom_instruction)
            return "\n".join((replacement or "") if l.startswith(ACK_LINE) else l
                             for l in text.split("\n")
                             if replacement is not None or not l.startswith(ACK_LINE))
        live.GeminiLiveService._build_system_instruction = swap_ack
    try:
        session = await service.create_session(f"bench_{uuid.uuid4().hex[:8]}", language="en-IN")
    finally:
        live.GeminiLiveService._build_system_instruction = original

    tool_calls = []
    run_query = session._run_rag_query

    async def counted(query):
        tool_calls.append(time.perf_counter())
        return await run_query(query)

    session._run_rag_query = counted
    if variant == "after":
        await session.connect()
        await asyncio.sleep(2.0)

    first_audio = None
    speech_end = None
    done = asyncio.Event()

    async def listen():
        nonlocal first_audio
        async for item in session.receive_audio():
            if isinstance(item, (bytes, bytearray)) and speech_end is not None and first_audio is None:
                first_audio = time.perf_counter()
            # The turn is over when audio has played after the tool result
            # (an acknowledgement alone is not the answer).
            if item == "TURN_COMPLETE" and first_audio is not None and tool_calls:
                done.set()
                return

    step = RATE * 2 * CHUNK_MS // 1000
    listener = None
    try:
        for n, start in enumerate(range(0, len(pcm), step)):
            if n == 0 and variant == "before":
                await session.connect()
            if listener is None:
                listener = asyncio.create_task(listen())
            await session.send_audio(pcm[start:start + step])
            await asyncio.sleep(CHUNK_MS / 1000)
        speech_end = time.perf_counter()
        silence = b"\x00" * step
        deadline = speech_end + 25
        while first_audio is None and time.perf_counter() < deadline:
            await session.send_audio(silence)
            await asyncio.sleep(CHUNK_MS / 1000)
        # Long enough for the tool to run after an acknowledgement.
        turn_deadline = min(deadline, speech_end + 12)
        while not done.is_set() and time.perf_counter() < turn_deadline:
            await session.send_audio(silence)
            await asyncio.sleep(CHUNK_MS / 1000)
        print(f"   said: {' '.join(session.response_buffer)[:160]!r}", flush=True)
        if first_audio is None:
            return None, bool(tool_calls), False
        acked = bool(tool_calls) and first_audio < tool_calls[0]
        return first_audio - speech_end, bool(tool_calls), acked
    finally:
        if listener:
            listener.cancel()
        await session.disconnect()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--ack-line", help="replace the acknowledgement line in the 'after' prompt")
    args = ap.parse_args()
    global ACK_OVERRIDE
    ACK_OVERRIDE = args.ack_line

    from gemini_live.service import GeminiLiveService
    from safety_enhancements import get_safety_manager

    docs, metas = load_chunks()
    service = GeminiLiveService(rag_pipeline=pipeline(docs, metas), safety_manager=get_safety_manager())
    audio = [await speech(q) for q in QUESTIONS]

    results = {"before": [], "after": []}
    failures = {"before": 0, "after": 0}
    for _ in range(args.rounds):
        for q, pcm in zip(QUESTIONS, audio):
            for variant in ("before", "after"):
                try:
                    took, tool, acked = await trial(service, pcm, variant)
                except Exception as e:  # a failed connect: recorded, not timed
                    print(f"{variant:6s} failed: {type(e).__name__}", flush=True)
                    failures[variant] += 1
                    continue
                results[variant].append((took, tool, acked))
                print(f"{variant:6s} first_audio={fmt(took)} tool={tool} acked={acked} | {q[:50]}", flush=True)

    lines = ["# Gemini Live: end of speech to first audio", "",
             f"`scripts/benchmark_live_first_audio.py`, {len(QUESTIONS)} spoken questions x {args.rounds}, "
             f"model {service.model}, audio streamed in real time then silence. Seconds from the end of "
             "speech to the first audio chunk.", "",
             "| variant | n | p50 | p95 | max | RAG tool ran | spoke before the tool | failed to connect |",
             "|---|---|---|---|---|---|---|---|"]
    for variant, rows in results.items():
        times = [t for t, _, _ in rows if t is not None]
        lines.append(f"| {variant} | {len(rows)} | {fmt(pct(times, .5))} | {fmt(pct(times, .95))} | "
                     f"{fmt(max(times) if times else None)} | {sum(1 for _, t, _ in rows if t)} | "
                     f"{sum(1 for _, _, a in rows if a)} | {failures[variant]} |")
    print("\n".join(lines))
    if args.out:
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
