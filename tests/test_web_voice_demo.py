#!/usr/bin/env python3
"""
The web voice page works as a booth demo (docs/deploy_voice_demo.md).

What these pin down:

  * an access code, when one is configured, keeps a socket without it out,
    and the page's settings route never reveals the code;
  * the made-up-people notice is on for any Railway deployment;
  * a conversation outlives one Live session: when Live ends a session while
    the caller keeps talking, the socket opens a new one and carries on;
  * an answer that drew on the knowledge base names its sources to the caller,
    and the Live tool records them.

The socket tests reuse the scripted Live harness of test_web_live_voice.
"""

import asyncio
import os
import sys
import unittest
from typing import Dict, List
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from starlette.websockets import WebSocketDisconnect  # noqa: E402

import web_demo  # noqa: E402
from tests.test_live_tool_safety import FakeRag, FakeSafety, build_service, build_session  # noqa: E402
from tests.test_web_live_voice import (  # noqa: E402
    SAFE_REPLY,
    ScriptedLiveService,
    WebVoiceSocket,
    setUpModule as _live_setup,
    tearDownModule as _live_teardown,
)

CODE = "booth-42"


def setUpModule():
    _live_setup()


def tearDownModule():
    _live_teardown()


class DemoSettings(unittest.TestCase):
    """web_demo on its own."""

    def test_no_code_configured_lets_everyone_in(self):
        self.assertTrue(web_demo.access_code_matches(None, {}))
        self.assertTrue(web_demo.access_code_matches("anything", {}))

    def test_a_configured_code_must_match(self):
        env = {"WEB_VOICE_ACCESS_CODE": CODE}
        self.assertTrue(web_demo.access_code_matches(CODE, env))
        self.assertTrue(web_demo.access_code_matches(f" {CODE} ", env))
        self.assertFalse(web_demo.access_code_matches("booth-41", env))
        self.assertFalse(web_demo.access_code_matches(None, env))
        self.assertFalse(web_demo.access_code_matches("", env))

    def test_the_notice_is_on_for_any_railway_deployment(self):
        self.assertTrue(web_demo.demo_notice({"RAILWAY_ENVIRONMENT_NAME": "production"}))
        self.assertFalse(web_demo.demo_notice({}))

    def test_the_notice_can_be_set_either_way(self):
        self.assertTrue(web_demo.demo_notice({"WEB_VOICE_DEMO_NOTICE": "true"}))
        self.assertFalse(web_demo.demo_notice({"WEB_VOICE_DEMO_NOTICE": "false",
                                               "RAILWAY_ENVIRONMENT_NAME": "production"}))

    def test_the_settings_never_contain_the_code(self):
        settings = web_demo.demo_settings({"WEB_VOICE_ACCESS_CODE": CODE})
        self.assertEqual(settings, {"notice": False, "code_required": True})
        self.assertNotIn(CODE, repr(settings))


class TheAccessCode(WebVoiceSocket):
    """With WEB_VOICE_ACCESS_CODE set, the socket needs ?code=."""

    def setUp(self):
        super().setUp()
        patcher = mock.patch.dict(os.environ, {"WEB_VOICE_ACCESS_CODE": CODE})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_socket_without_the_code_is_refused_before_any_live_session(self):
        with self.client.websocket_connect("/ws/voice") as ws:
            frame = ws.receive_json()
            with self.assertRaises(WebSocketDisconnect) as closed:
                ws.receive_json()
        self.assertEqual(frame["type"], "error")
        self.assertEqual(frame["code"], "access_code")
        self.assertEqual(closed.exception.code, 1008)
        self.assertEqual(self.service.sessions, [])

    def test_a_wrong_code_is_refused(self):
        with self.client.websocket_connect("/ws/voice?code=booth-41") as ws:
            frame = ws.receive_json()
        self.assertEqual(frame.get("code"), "access_code")

    def test_the_right_code_gets_a_conversation(self):
        ScriptedLiveService.script = [([b"SAFE-1"], SAFE_REPLY, "How do I keep him comfortable?")]
        with self.client.websocket_connect(f"/ws/voice?code={CODE}") as ws:
            self.start(ws, {})
            frames = self.turn(ws)
        self.assertIn(b"SAFE-1", [f["bytes"] for f in frames if f["type"] == "audio"])

    def test_the_page_learns_a_code_is_needed_but_not_the_code(self):
        body = self.client.get("/api/voice/demo").json()
        self.assertTrue(body["code_required"])
        self.assertNotIn(CODE, repr(body))


class AConversationOutlivesOneLiveSession(WebVoiceSocket):
    """Live ends sessions (about 15 minutes, go_away); the caller keeps talking."""

    def test_audio_after_live_ended_the_session_opens_a_new_one(self):
        ScriptedLiveService.script = [([b"FIRST"], SAFE_REPLY, "How do I keep him comfortable?")]
        with self.client.websocket_connect("/ws/voice") as ws:
            _, first = self.start(ws, {})
            self.turn(ws)
            # Each session copies the script when it opens: the next one's.
            ScriptedLiveService.script = [
                ([b"SECOND"], "Sit with him and keep the room calm.", "What else can I do?")]
            # What the session looks like after Live closed it.
            first._running = False
            first.is_connected = first.is_active = False
            ws.send_bytes(b"\x00\x01" * 160)
            frames = self.frames_until(ws, "turn_complete") + self.exchange(ws)
        renewed = [f for f in frames if f["type"] == "session_renewed"]
        self.assertEqual(len(renewed), 1, frames)
        self.assertEqual(len(self.service.sessions), 2)
        self.assertIsNot(self.service.sessions[-1], first)
        self.assertIn(b"SECOND", [f["bytes"] for f in frames if f["type"] == "audio"])
        self.assertEqual([f for f in frames if f["type"] == "error"], [])

    def test_a_failed_reopen_is_not_retried_on_every_chunk(self):
        ScriptedLiveService.script = [([b"FIRST"], SAFE_REPLY, "How do I keep him comfortable?")]
        with self.client.websocket_connect("/ws/voice") as ws:
            _, first = self.start(ws, {})
            self.turn(ws)
            first._running = False
            first.is_connected = first.is_active = False
            failing = mock.AsyncMock(side_effect=RuntimeError("Live is down"))
            with mock.patch.object(type(self.service), "create_session", failing):
                for _ in range(5):
                    ws.send_bytes(b"\x00\x01" * 160)
                frames = self.exchange(ws)
        errors = [f for f in frames if f["type"] == "error"]
        self.assertEqual(len(errors), 1, frames)
        self.assertEqual(failing.await_count, 1)


class GroundedAnswersNameTheirSources(WebVoiceSocket):
    """The assistant's transcription carries the sources its lookup returned."""

    def test_a_turn_after_a_lookup_names_its_sources(self):
        ScriptedLiveService.script = [([b"SAFE"], SAFE_REPLY, "How do I keep him comfortable?")]
        with self.client.websocket_connect("/ws/voice") as ws:
            _, session = self.start(ws, {})
            session._turn_looked_up = True
            session._turn_sources = ["Palliative care handbook", "Palliative care handbook", "WHO guide"]
            frames = self.turn(ws)
        spoken = [f for f in frames if f["type"] == "transcription" and f.get("role") == "assistant"]
        self.assertEqual(spoken[0]["sources"], ["Palliative care handbook", "WHO guide"])
        # Taken: the next turn does not repeat them.
        self.assertEqual(session.take_turn_sources(), (False, []))

    def test_a_turn_without_a_lookup_names_none(self):
        ScriptedLiveService.script = [([b"HELLO"], "Hello, how can I help?", "Hello")]
        with self.client.websocket_connect("/ws/voice") as ws:
            self.start(ws, {})
            frames = self.turn(ws)
        spoken = [f for f in frames if f["type"] == "transcription" and f.get("role") == "assistant"]
        self.assertNotIn("sources", spoken[0])


class TheLiveToolRecordsItsSources(unittest.TestCase):
    """_run_rag_query notes the lookup and the source names for the turn."""

    def test_the_lookup_and_its_sources_are_recorded(self):
        session = build_session(build_service(FakeRag(answer="Keep him comfortable."), FakeSafety(
            response="Keep him comfortable.")))
        asyncio.run(session._run_rag_query("how do I keep him comfortable?"))
        self.assertEqual(session.take_turn_sources(), (True, ["handbook.pdf"]))

    def test_a_failed_lookup_still_counts_as_a_lookup_with_no_sources(self):
        session = build_session(build_service(None, FakeSafety()))
        asyncio.run(session._run_rag_query("anything"))
        self.assertEqual(session.take_turn_sources(), (True, []))


if __name__ == "__main__":
    unittest.main()
