"""
WhatsApp acknowledges at once, overlaps its steps, and never speaks a dose.

Before this, the webhook answered Twilio only after RAG, translation and TTS
(so slow answers were retried as duplicates), every reply paused on fixed
one-second sleeps, Twilio sends blocked the event loop, and the Gemini Live
voice-note path waited for a marker it never recognised (so it ran until the
session died), injected a chunk count as "context", and sent Live's audio with
no Dose Boundary at all.

What these pin down:

  * the webhook returns within 200 ms while the message is still being answered
  * a replayed webhook (same MessageSid) is processed once
  * translation overlaps the English send in the trace; no fixed sleeps remain
  * a Live voice note returns audio within the deadline in a simulated session
  * a dose in what Live said: its audio is never sent; the redacted text is
    sent and spoken instead
  * Live is grounded in the guarded answer, sent as context (no reply asked)
"""

import asyncio
import inspect
import json
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import post_work  # noqa: E402
from tests.test_text_path_dose_guard import (  # noqa: E402
    DOSE_ANSWER,
    FakeSender,
    RealisticRag,
    assert_no_dose,
    make_bot,
)

NUMBER = "+919800000001"
SAFE_ANSWER = "Sit upright, use a fan near the face, and call your nurse if it worsens."


def bot_with_dedupe(rag):
    from collections import OrderedDict

    bot = make_bot(rag)
    bot._seen_message_ids = OrderedDict()
    return bot


class SlowSender(FakeSender):
    def __init__(self, delay):
        super().__init__()
        self.delay = delay

    async def send_text_message(self, to, text):
        await asyncio.sleep(self.delay)
        return await super().send_text_message(to, text)


class SlowTranslateRag(RealisticRag):
    async def translate_text(self, text, target_language):
        await asyncio.sleep(0.2)
        return await super().translate_text(text, target_language)


class Traced(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="traces-")
        env = mock.patch.dict(os.environ, {"TRACE_DIR": self.dir, "TRACING_ENABLED": "1",
                                           "RETRIEVAL_AUDIT_DIR": tempfile.mkdtemp()})
        env.start()
        self.addCleanup(env.stop)

    def spans(self):
        out = []
        for path in Path(self.dir).glob("*.jsonl"):
            out += [json.loads(line) for line in path.read_text().splitlines()]
        return out


def interval(span):
    end = datetime.fromisoformat(span["ts"]).timestamp()
    return end - span["ms"] / 1000, end


class Webhook(unittest.TestCase):
    def test_returns_at_once_while_the_answer_is_still_running(self):
        from fastapi.testclient import TestClient

        bot = bot_with_dedupe(RealisticRag(answer=SAFE_ANSWER))

        async def slow_process(*args):
            await asyncio.sleep(1.0)

        from tests.test_webhook_verification import twilio_signature

        params = {"From": f"whatsapp:{NUMBER}", "To": "whatsapp:+1",
                  "Body": "breathless at night", "MessageSid": "SM1"}
        env = {"TWILIO_AUTH_TOKEN": "token", "PUBLIC_BASE_URL": "https://palli.example.org"}
        signature = twilio_signature("token", "https://palli.example.org/webhook", params)
        with mock.patch.object(bot, "_process_twilio_message", slow_process), \
                mock.patch.dict(os.environ, env):
            client = TestClient(bot.create_webhook_app())
            t0 = time.perf_counter()
            response = client.post("/webhook", data=params, headers={"X-Twilio-Signature": signature})
            took = time.perf_counter() - t0
        self.assertEqual(response.status_code, 200)
        self.assertLess(took, 0.2)

    def test_a_replayed_message_is_processed_once(self):
        bot = bot_with_dedupe(RealisticRag(answer=SAFE_ANSWER))
        calls = []

        async def process(*args):
            calls.append(args)

        async def go():
            with mock.patch.object(bot, "_process_twilio_message", process):
                first = bot.accept_twilio_message("SM42", NUMBER, "+1", "hi", None, None, "0")
                second = bot.accept_twilio_message("SM42", NUMBER, "+1", "hi", None, None, "0")
                await post_work.drain()
            return first, second

        self.assertEqual(asyncio.run(go()), (True, False))
        self.assertEqual(len(calls), 1)

    def test_the_server_webhook_also_acknowledges_at_once(self):
        source = Path(__file__).resolve().parents[1].joinpath("simple_rag_server.py").read_text(encoding="utf-8")
        start = source.index("async def whatsapp_webhook(")
        body = source[start:source.index('@app.get("/media/{filename}")', start)]
        self.assertIn("accept_twilio_message", body)
        self.assertNotIn("await whatsapp_bot._process_twilio_message", body)


class Overlap(Traced):
    def test_translation_overlaps_the_english_send(self):
        bot = make_bot(SlowTranslateRag(answer=SAFE_ANSWER))
        bot.twilio_api = SlowSender(delay=0.2)
        bot.user_preferences[NUMBER] = {"language": "hi"}
        from observability import trace_request

        async def go():
            with trace_request("whatsapp", name="message"):
                await bot._handle_twilio_text_message(NUMBER, "breathless at night")

        t0 = time.perf_counter()
        asyncio.run(go())
        took = time.perf_counter() - t0
        self.assertEqual(len(bot.twilio_api.texts), 2)
        self.assertTrue(bot.twilio_api.texts[0].startswith("🇬🇧 English"))
        spans = self.spans()
        send = interval(next(s for s in spans if s["name"] == "send.text"))
        translate = interval(next(s for s in spans if s["name"] == "translate"))
        self.assertLess(max(send[0], translate[0]), min(send[1], translate[1]) - 0.1)
        self.assertLess(took, 0.2 + 0.2 + 0.2 + 0.15, "send, translate and send ran back to back")

    def test_no_fixed_sleeps_remain(self):
        import whatsapp_bot

        self.assertNotIn("asyncio.sleep(", inspect.getsource(whatsapp_bot))


# ---------------------------------------------------------------------------
# Gemini Live voice notes, in a simulated session
# ---------------------------------------------------------------------------

TURN_COMPLETE = b"__TURN_COMPLETE__"
INTERRUPTED = b"__INTERRUPTED__"


class FakeLiveSession:
    TURN_COMPLETE = TURN_COMPLETE
    INTERRUPTED = INTERRUPTED

    def __init__(self, spoken, audio=b"\x01\x02" * 400, delay=0.05):
        self.spoken = spoken
        self.audio = audio
        self.delay = delay
        self.is_active = True
        self.session_id = "live-test"
        self.texts = []
        self.audio_in = 0
        self.queue = asyncio.Queue()
        self.response_buffer = []

    async def send_text(self, text, turn_complete=True):
        self.texts.append((text, turn_complete))

    async def send_audio(self, chunk):
        self.audio_in += len(chunk)

    async def end_audio(self):
        async def reply():
            await asyncio.sleep(self.delay)
            self.response_buffer.append(self.spoken)
            await self.queue.put(self.audio)
            await self.queue.put(TURN_COMPLETE)
            # The session stays open after the turn, as a real one does.

        asyncio.get_running_loop().create_task(reply())

    async def receive_audio(self):
        while True:
            yield await self.queue.get()

    def get_response_transcription(self, clear=True):
        text = " ".join(self.response_buffer)
        if clear:
            self.response_buffer.clear()
        return text


class FakeAudio:
    def convert_to_pcm(self, path):
        return b"\x00" * 3200

    def chunk_audio(self, pcm):
        return [pcm[i:i + 1600] for i in range(0, len(pcm), 1600)]

    def convert_from_pcm(self, pcm, fmt):
        path = Path(tempfile.mkdtemp()) / "reply.mp3"
        path.write_bytes(pcm)
        return str(path)


class SpeakingTts:
    def __init__(self):
        self.spoken = []

    async def synthesize_speech(self, text, language):
        self.spoken.append(text)
        path = Path(tempfile.mkdtemp()) / "tts.mp3"
        path.write_bytes(b"x")
        return {"audio_available": True, "audio_file": str(path)}


def live_bot(rag_answer, spoken):
    from gemini_live.service import GeminiLiveService

    bot = make_bot(RealisticRag(answer=rag_answer))
    bot.gemini_live_enabled = True
    bot.twilio_api.client = object()
    bot.tts_service = SpeakingTts()
    bot.audio_handler = FakeAudio()
    session = FakeLiveSession(spoken)

    class Manager:
        async def get_or_create_session(self, user_id, language, voice):
            return session

    bot.gemini_session_manager = Manager()
    service = SimpleNamespace(config=SimpleNamespace(rag_context_enabled=True))
    bot.gemini_service = SimpleNamespace(
        inject_guarded_answer=lambda s, r: GeminiLiveService.inject_guarded_answer(service, s, r))

    async def download(url):
        path = Path(tempfile.mkdtemp()) / "note.ogg"
        path.write_bytes(b"ogg")
        return str(path)

    bot.twilio_api.download_media = download
    return bot, session


class LiveVoiceNotes(Traced):
    def test_returns_live_audio_within_the_deadline(self):
        bot, session = live_bot(SAFE_ANSWER, spoken="Sit upright and use a fan.")
        t0 = time.perf_counter()
        asyncio.run(asyncio.wait_for(
            bot._handle_twilio_media_message(NUMBER, "https://x/m", "audio/ogg"),
            timeout=bot.live_reply_deadline_s()))
        self.assertLess(time.perf_counter() - t0, 2.0)
        self.assertEqual(len(bot.twilio_api.audio), 1, "Live's audio was not sent")
        self.assertEqual(bot.tts_service.spoken, [], "fell back to TTS although Live answered")
        self.assertTrue(any("Sit upright and use a fan." in t for t in bot.twilio_api.texts))
        injected, turn_complete = session.texts[0]
        self.assertIn(SAFE_ANSWER, injected)
        self.assertFalse(turn_complete, "grounding must not ask Live for a reply of its own")
        self.assertNotIn("context_used", injected)

    def test_a_dose_in_what_live_said_is_never_sent(self):
        bot, session = live_bot(SAFE_ANSWER, spoken=DOSE_ANSWER)
        asyncio.run(bot._handle_twilio_media_message(NUMBER, "https://x/m", "audio/ogg"))
        sent_audio = [a[0] for a in bot.twilio_api.audio]
        self.assertTrue(all("reply.mp3" not in path for path in sent_audio), "Live audio with a dose was sent")
        self.assertTrue(bot.tts_service.spoken, "the guarded answer was not spoken")
        for text in bot.twilio_api.texts + bot.tts_service.spoken:
            assert_no_dose(self, text, "WhatsApp Live voice note")

    def test_grounding_is_the_guarded_answer(self):
        bot, session = live_bot(DOSE_ANSWER, spoken="Please speak to your doctor about the dose.")
        asyncio.run(bot._handle_twilio_media_message(NUMBER, "https://x/m", "audio/ogg"))
        assert_no_dose(self, session.texts[0][0], "Live grounding")

    def test_a_reply_past_the_deadline_sends_the_guarded_answer(self):
        bot, session = live_bot(SAFE_ANSWER, spoken="late")
        session.delay = 1.0
        with mock.patch.dict(os.environ, {"WHATSAPP_LIVE_REPLY_DEADLINE_S": "0.2"}):
            asyncio.run(bot._handle_twilio_media_message(NUMBER, "https://x/m", "audio/ogg"))
        self.assertIn(SAFE_ANSWER[:20], bot.tts_service.spoken[0])
        self.assertTrue(all("reply.mp3" not in a[0] for a in bot.twilio_api.audio))


if __name__ == "__main__":
    unittest.main()
