"""
Gemini Live: the caller hears something sooner, and nothing unguarded.

Before this, the web voice client's Live session connected when the first
audio arrived (adding the handshake to the first answer), every final
transcript was embedded by the RAG classifier even in tool mode where the
result was thrown away, and the non-tool injection path handed the model the
unguarded draft answer.

What these pin down:

  * the session opens on the config message, and start_audio reuses it
    (one connect); a failed pre-connect falls back to connecting on
    start_audio; an unused one is closed by the idle cleanup
  * in tool mode the transcript handler returns before the classifier embeds
  * the non-tool injection and the tool result both pass the Dose Boundary
  * the system prompt allows a short acknowledgement before the tool and
    nothing medical until its result
"""

import asyncio
import os
import sys
import unittest
from datetime import timedelta
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemini_live.config import GeminiLiveConfig  # noqa: E402
from gemini_live.service import GeminiLiveSession  # noqa: E402
from gemini_live.session_manager import PreConnect, SessionManager  # noqa: E402

DOSE_ANSWER = "Give morphine 10 mg every 4 hours for the pain at night."
MODEL = "gemini-3.1-flash-live-preview"


class FakeLiveSession:
    def __init__(self, fail=False):
        self.fail = fail
        self.is_active = False
        self.model = MODEL
        self.connects = 0
        self.participant_id = None

    async def connect(self):
        await asyncio.sleep(0.01)
        if self.fail:
            raise RuntimeError("handshake refused")
        self.connects += 1
        self.is_active = True

    async def disconnect(self):
        self.is_active = False


class FakeService:
    def __init__(self, fail_first=False):
        self.created = []
        self.fail_first = fail_first

    def resolve_model(self, model):
        return model or MODEL

    async def create_session(self, session_id, language, voice, model):
        s = FakeLiveSession(fail=self.fail_first and not self.created)
        self.created.append(s)
        return s

    async def close_session(self, session_id):
        pass


class PreConnecting(unittest.TestCase):
    def manager(self, **kw):
        m = SessionManager.__new__(SessionManager)
        m.service = FakeService(**kw)
        m.sessions, m.user_sessions = {}, {}
        m.timeout = timedelta(minutes=14)
        return m

    def test_config_connects_and_start_audio_reuses_it(self):
        async def go():
            m = self.manager()
            pre = PreConnect(m)
            await pre.open("u1", "en-IN", None)  # the config message
            await asyncio.sleep(0.05)  # the caller getting ready to speak
            self.assertEqual(m.service.created[0].connects, 1, "connected before any audio")
            await pre.settle()  # start_audio
            session = await m.get_or_create_session("u1", language="en-IN", model=None)
            return m, session

        m, session = asyncio.run(go())
        self.assertEqual(len(m.service.created), 1)
        self.assertIs(session, m.service.created[0])

    def test_start_audio_waits_for_a_connect_under_way(self):
        async def go():
            m = self.manager()
            pre = PreConnect(m)
            await pre.open("u1", "en-IN", None)
            await pre.settle()  # start_audio arrives mid-handshake
            await m.get_or_create_session("u1", language="en-IN", model=None)
            return m

        self.assertEqual(len(asyncio.run(go()).service.created), 1)

    def test_failed_pre_connect_falls_back_to_start_audio(self):
        async def go():
            m = self.manager(fail_first=True)
            pre = PreConnect(m)
            await pre.open("u1", "en-IN", None)
            await pre.settle()  # logged, not raised
            return m, await m.get_or_create_session("u1", language="en-IN", model=None)

        m, session = asyncio.run(go())
        self.assertEqual(len(m.service.created), 2)
        self.assertTrue(session.is_active)

    def test_unused_pre_connect_is_closed_by_idle_cleanup(self):
        async def go():
            m = self.manager()
            pre = PreConnect(m)
            await pre.open("u1", "en-IN", None)
            await pre.settle()
            info = next(iter(m.sessions.values()))
            info.last_activity -= timedelta(minutes=15)  # the caller never spoke
            await m._cleanup_expired_sessions()
            return m, info.session

        m, session = asyncio.run(go())
        self.assertEqual(m.sessions, {})
        self.assertFalse(session.is_active)

    def test_settle_without_open_is_a_no_op(self):
        asyncio.run(PreConnect(self.manager()).settle())


class FakeClassifier:
    def __init__(self):
        self.embedded = []

    def strip_filler_words(self, text):
        return text

    def is_out_of_scope(self, text):
        return False, ""

    def should_query_rag(self, text):
        self.embedded.append(text)
        return True, "medical"


class FakeRag:
    def __init__(self):
        self.calls = 0

    async def query(self, question, conversation_id=None, user_id=None, top_k=5):
        self.calls += 1
        return {"status": "success", "answer": DOSE_ANSWER, "context_used": 2,
                "sources": [{"filename": "handbook.pdf"}]}


class NoEscalation:
    async def check_voice_query(self, **kw):
        return type("R", (), {"should_escalate": False, "modified_transcript": None})()


def live_session(tool_mode):
    from safety_enhancements import get_safety_manager

    config = GeminiLiveConfig(rag_top_k=5)
    service = type("S", (), {
        "model": MODEL, "rag_pipeline": FakeRag(), "config": config,
        "query_classifier": FakeClassifier(), "safety_manager": get_safety_manager(),
    })()
    session = GeminiLiveSession(service, "s1", None, language="en-IN", voice="Aoede")
    session.participant_id = None
    session._rag_tool_enabled = tool_mode
    session._session, session._running = object(), True
    session.injected = []

    async def send(text, turn_complete=False):
        session.injected.append(text)

    session._send_text_now = send
    return session


def handle_transcript(session, text):
    session._pending_transcription.append(text)
    with mock.patch("voice_safety_wrapper.get_voice_safety_wrapper", return_value=NoEscalation()):
        asyncio.run(session._query_rag_and_inject())


class TranscriptHandling(unittest.TestCase):
    def test_tool_mode_skips_the_classifier_embedding(self):
        session = live_session(tool_mode=True)
        handle_transcript(session, "my mother has pain at night, what can we do")
        self.assertEqual(session.service.query_classifier.embedded, [])
        self.assertEqual(session.service.rag_pipeline.calls, 0)
        self.assertEqual(session.injected, [])

    def test_injection_path_passes_the_dose_boundary(self):
        session = live_session(tool_mode=False)
        handle_transcript(session, "my mother has pain at night, what can we do")
        self.assertEqual(len(session.injected), 1)
        self.assertNotIn("10 mg", session.injected[0])
        self.assertNotIn("every 4 hours", session.injected[0])

    def test_tool_result_passes_the_dose_boundary(self):
        session = live_session(tool_mode=True)
        out = asyncio.run(session._run_rag_query("how much morphine at night?"))
        self.assertNotIn("10 mg", out)


class Prompt(unittest.TestCase):
    def test_acknowledgement_allowed_but_nothing_medical_before_the_tool(self):
        from gemini_live.service import GeminiLiveService

        service = GeminiLiveService.__new__(GeminiLiveService)
        service.config = GeminiLiveConfig(rag_top_k=5)
        service.rag_pipeline = FakeRag()
        text = service._build_system_instruction("en-IN")
        self.assertIn("one short acknowledgement", text)
        self.assertIn("Say nothing medical until the tool result has arrived", text)


if __name__ == "__main__":
    unittest.main()
