#!/usr/bin/env python3
"""
Web Live voice is identified server-side, dose-checked and pinned to the
release (review 3, ticket 12).

Before this, `/ws/voice` on the main server:

  * took the caller's id from the client's config frame, so a browser could
    name another person and load their patient context or join their Live
    session, and every anonymous `web_<ts>` id got a longitudinal record;
  * let the client choose the Live model (`config.model`, `set_model`),
    including the translate model, which has no system instruction and no RAG;
  * streamed Live's audio as it arrived, so a spoken dose was never checked.

What these pin down, over the socket of the app create_app() builds. Only the
Gemini network is scripted: a real GeminiLiveSession whose replies come from
a script, managed by the real SessionManager.
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from gemini_live.config import SUPPORTED_MODELS, GeminiLiveConfig  # noqa: E402
from gemini_live.service import GeminiLiveError, GeminiLiveService, GeminiLiveSession  # noqa: E402
from participants import is_person  # noqa: E402
from tests.test_route_policy import ADMIN_SECRET, MOBILE_SECRET, build_server_app  # noqa: E402

TRANSLATE_MODEL = "gemini-3.5-live-translate-preview"
OTHER_USER = "patient-of-someone-else"
DOSE_REPLY = "Give 10 mg of morphine every 4 hours."
SAFE_REPLY = "Keep the patient comfortable and call the palliative care nurse."
QUESTION = "How much morphine should I give for the pain?"


def release_model() -> str:
    """The Live model the study release record names."""
    import study_release

    return study_release.get_study_release().live_model


class ScriptedLiveSession(GeminiLiveSession):
    """A real GeminiLiveSession with the Gemini network replaced by a script.

    The first audio a caller sends plays the next scripted turn into the
    session's output queue, exactly as `_receive_loop` would: the audio
    chunks, the transcripts into their buffers, then TURN_COMPLETE.
    """

    def __init__(self, *args, script: List[Tuple[List[bytes], str, str]], **kwargs):
        super().__init__(*args, **kwargs)
        self.script = list(script)

    async def connect(self) -> None:
        self._running = True
        self.is_connected = self.is_active = True

    async def disconnect(self) -> None:
        self._running = False
        self.is_connected = self.is_active = False
        await self._audio_out_queue.put(None)

    async def send_audio(self, audio_chunk: bytes) -> None:
        if not self.script:
            return
        chunks, spoken, heard = self.script.pop(0)
        self.transcription_buffer.append(heard)
        for chunk in chunks:
            await self._audio_out_queue.put(chunk)
        self.response_buffer.append(spoken)
        await self._audio_out_queue.put(self.TURN_COMPLETE)


class ScriptedLiveService:
    """Stands in for GeminiLiveService: no Google client, scripted sessions.

    Model resolution is the real service's, so a model it refuses is refused.
    """

    script: List[Tuple[List[bytes], str, str]] = []
    instances: List["ScriptedLiveService"] = []

    def __init__(self, rag_pipeline=None, **_kwargs):
        self.config = GeminiLiveConfig()
        self.model = self.config.model
        self.rag_pipeline = rag_pipeline
        self.sessions: List[ScriptedLiveSession] = []
        ScriptedLiveService.instances.append(self)

    def is_available(self) -> bool:
        return True

    def resolve_model(self, model: Optional[str]) -> str:
        return GeminiLiveService.resolve_model(self, model)

    async def create_session(self, session_id: str, language: str = "en-IN", voice: str = "Aoede",
                             system_instruction: Optional[str] = None,
                             model: Optional[str] = None) -> ScriptedLiveSession:
        session = ScriptedLiveSession(self, session_id, None, language=language, voice=voice,
                                      model=self.resolve_model(model), script=self.script)
        self.sessions.append(session)
        return session

    async def close_session(self, session_id: str) -> None:
        return None


_module_patches: List = []


def setUpModule():
    """The admin key, and no release probe (it opens Chroma and writes under data/)."""
    import study_release

    _module_patches.extend([
        mock.patch.dict(os.environ, {"ADMIN_JWT_SECRET": ADMIN_SECRET}),
        mock.patch.object(study_release.StudyRelease, "verify_on_startup", return_value=[]),
        mock.patch.object(study_release.StudyRelease, "_live_embedding_model",
                          staticmethod(lambda: "unavailable")),
    ])
    for patcher in _module_patches:
        patcher.start()


def tearDownModule():
    while _module_patches:
        _module_patches.pop().stop()


class WebVoiceSocket(unittest.TestCase):
    """The real server app with its startup run, and Live scripted."""

    app = None

    @classmethod
    def setUpClass(cls):
        if WebVoiceSocket.app is None:
            WebVoiceSocket.app = build_server_app()

    def setUp(self):
        import simple_rag_server as srv
        from auth.middleware import get_jwt_handler, init_auth

        ScriptedLiveService.script = []
        ScriptedLiveService.instances = []
        patches = [
            mock.patch.object(srv, "GeminiLiveService", ScriptedLiveService),
            mock.patch.object(srv, "GRAPHRAG_AVAILABLE", False),
            mock.patch.object(srv, "KNOWLEDGE_GRAPH_AVAILABLE", False),
            mock.patch("llm.client.warm_up", mock.AsyncMock()),
            mock.patch.dict(os.environ, {"GEMINI_LIVE_ENABLED": "true", "RETELL_ENABLED": ""}),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = TestClient(self.app, raise_server_exceptions=False)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

        init_auth(jwt_secret=MOBILE_SECRET)
        pair = get_jwt_handler().create_token_pair("asha-1", "asha", "site-1")
        self.signed_in = {"Authorization": f"Bearer {pair.access_token}"}

    @property
    def service(self) -> ScriptedLiveService:
        return ScriptedLiveService.instances[-1]

    def frames_until(self, ws, wanted: str) -> List[Dict]:
        """Every frame up to and including the first JSON frame of type `wanted`.
        Binary frames come back as {"type": "audio", "bytes": ...}."""
        frames = []
        while True:
            message = ws.receive()
            if message.get("bytes") is not None:
                frames.append({"type": "audio", "bytes": message["bytes"]})
                continue
            frame = json.loads(message["text"])
            frames.append(frame)
            if frame.get("type") == wanted:
                return frames

    def start(self, ws, config: Dict) -> Tuple[Dict, ScriptedLiveSession]:
        """Send the config and start_audio; return the ack and the Live session."""
        ws.send_json({"type": "config", "language": "en-IN", **config})
        ack = self.frames_until(ws, "config_ack")[-1]
        ws.send_json({"type": "start_audio"})
        self.frames_until(ws, "session_created")
        return ack, self.service.sessions[-1]

    def turn(self, ws) -> List[Dict]:
        """Send one chunk of caller audio; return every frame of Live's turn,
        up to its assistant transcription."""
        ws.send_bytes(b"\x00\x01" * 160)
        frames = self.frames_until(ws, "turn_complete")
        return frames + self.exchange(ws)

    def exchange(self, ws, *messages: Dict) -> List[Dict]:
        """Send `messages`, then a ping; every frame up to the pong. The socket
        handles frames in order, so nothing the messages caused is missed."""
        for message in messages:
            ws.send_json(message)
        ws.send_json({"type": "ping"})
        return self.frames_until(ws, "pong")


class TheParticipantComesFromTheSession(WebVoiceSocket):
    """The caller is who their token says, never who the frame says."""

    def test_a_frame_naming_another_user_neither_loads_their_context_nor_joins_their_session(self):
        manager_sessions = []
        with self.client.websocket_connect("/ws/voice") as ws:
            ack, session = self.start(ws, {"user_id": OTHER_USER})
            manager_sessions = [s.session_id for s in self.service.sessions]
        self.assertNotEqual(session.rag_user_id, OTHER_USER)
        self.assertNotIn(OTHER_USER, session.rag_user_id)
        self.assertNotIn(OTHER_USER, session.session_id)
        self.assertNotEqual(ack.get("user_id"), OTHER_USER)
        self.assertEqual(len(manager_sessions), 1)

    def test_a_signed_in_participant_is_keyed_on_their_token_not_the_frame(self):
        with self.client.websocket_connect("/ws/voice", headers=self.signed_in) as ws:
            _, session = self.start(ws, {"user_id": OTHER_USER})
        self.assertEqual(session.rag_user_id, "asha-1")
        self.assertNotIn(OTHER_USER, session.session_id)

    def test_two_anonymous_callers_naming_the_same_id_get_separate_sessions(self):
        with self.client.websocket_connect("/ws/voice") as first:
            _, one = self.start(first, {"user_id": OTHER_USER})
            with self.client.websocket_connect("/ws/voice") as second:
                _, two = self.start(second, {"user_id": OTHER_USER})
        self.assertIsNot(one, two)
        self.assertNotEqual(one.rag_user_id, two.rag_user_id)

    def test_an_anonymous_web_session_leaves_no_longitudinal_record(self):
        with self.client.websocket_connect("/ws/voice") as ws:
            _, session = self.start(ws, {})
        # The answer path keeps a longitudinal record only for a person.
        self.assertFalse(is_person(session.rag_user_id), session.rag_user_id)

    def test_an_admin_token_is_not_a_participant_on_web_voice(self):
        from auth.policy import issue_admin_token

        admin = {"Authorization": f"Bearer {issue_admin_token('dr-ashish')}"}
        with self.client.websocket_connect("/ws/voice", headers=admin) as ws:
            _, session = self.start(ws, {})
        self.assertFalse(is_person(session.rag_user_id), session.rag_user_id)


class TheLiveModelIsTheReleases(WebVoiceSocket):
    """The study release fixes the Live model; the client cannot choose it."""

    def test_a_client_chosen_model_is_rejected(self):
        with self.client.websocket_connect("/ws/voice") as ws:
            ws.send_json({"type": "config", "language": "en-IN", "model": TRANSLATE_MODEL})
            frames = self.frames_until(ws, "config_ack")
            ws.send_json({"type": "start_audio"})
            self.frames_until(ws, "session_created")
        self.assertIn("error", [f["type"] for f in frames])
        self.assertEqual(frames[-1]["model"], release_model())
        self.assertEqual({s.model for s in self.service.sessions}, {release_model()})

    def test_set_model_is_rejected(self):
        with self.client.websocket_connect("/ws/voice") as ws:
            self.start(ws, {})
            frames = self.exchange(ws, {"type": "set_model", "model": TRANSLATE_MODEL},
                                   {"type": "start_audio"})
        errors = [f for f in frames if f["type"] == "error"]
        self.assertEqual(len(errors), 1, frames)
        self.assertIn("release", errors[0]["error"].lower())
        self.assertEqual({s.model for s in self.service.sessions}, {release_model()})

    def test_the_models_route_lists_only_the_release_model(self):
        body = self.client.get("/api/voice/models").json()
        self.assertEqual([m["id"] for m in body["models"]], [release_model()])
        self.assertEqual(body["default"], release_model())


class TheReleaseRecordPinsTheLiveModel(unittest.TestCase):
    """One source of truth: the release record names the model sessions open with."""

    def test_config_yaml_does_not_choose_the_live_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("gemini_live:\n  enabled: true\n  model: \"gemini-live-2.5-flash-preview-native-audio-09-2025\"\n")
            with mock.patch.dict(os.environ, {"GEMINI_LIVE_MODEL": ""}):
                self.assertEqual(GeminiLiveConfig.from_yaml(str(path)).model, release_model())

    def test_the_release_record_names_a_model_the_service_runs(self):
        self.assertIn(release_model(), SUPPORTED_MODELS)

    def test_the_translate_model_is_not_offered(self):
        self.assertNotIn(TRANSLATE_MODEL, SUPPORTED_MODELS)
        with self.assertRaises(GeminiLiveError):
            GeminiLiveService.resolve_model(mock.Mock(model=release_model()), TRANSLATE_MODEL)

    def test_the_release_running_its_own_model_shows_no_live_model_drift(self):
        import study_release

        release = study_release.StudyRelease(record=study_release.get_study_release().record)
        live = release.live_fingerprint()["voice.live.model"]
        self.assertEqual(live, release.live_model)


class WhatLiveSaysPassesTheDoseBoundary(WebVoiceSocket):
    """Live's audio is released only after its transcript passes the Dose Boundary."""

    def test_a_turn_that_speaks_a_dose_releases_the_deferral_instead_of_the_audio(self):
        from dosage_guard import deferral_for

        ScriptedLiveService.script = [([b"DOSE-AUDIO-1", b"DOSE-AUDIO-2"], DOSE_REPLY, QUESTION)]
        with self.client.websocket_connect("/ws/voice") as ws:
            self.start(ws, {})
            frames = self.turn(ws)
        self.assertEqual([f for f in frames if f["type"] == "audio"], [])
        spoken = [f for f in frames if f["type"] == "transcription" and f.get("role") == "assistant"]
        self.assertEqual(len(spoken), 1)
        self.assertEqual(spoken[0]["text"], deferral_for("en-IN"))
        self.assertNotIn("10 mg", json.dumps(frames))

    def test_a_turn_without_a_dose_releases_its_audio_in_order(self):
        ScriptedLiveService.script = [([b"SAFE-1", b"SAFE-2"], SAFE_REPLY, "How do I keep him comfortable?")]
        with self.client.websocket_connect("/ws/voice") as ws:
            self.start(ws, {})
            frames = self.turn(ws)
        self.assertEqual([f["bytes"] for f in frames if f["type"] == "audio"], [b"SAFE-1", b"SAFE-2"])
        types_seen = [f["type"] for f in frames]
        self.assertLess(types_seen.index("audio"), types_seen.index("turn_complete"))
        spoken = [f for f in frames if f["type"] == "transcription" and f.get("role") == "assistant"]
        self.assertIn("comfortable", spoken[0]["text"])

    def test_a_turn_with_no_transcript_is_withheld(self):
        from dosage_guard import deferral_for

        ScriptedLiveService.script = [([b"UNCHECKED-AUDIO"], "", QUESTION)]
        with self.client.websocket_connect("/ws/voice") as ws:
            self.start(ws, {})
            frames = self.turn(ws)
        self.assertEqual([f for f in frames if f["type"] == "audio"], [])
        spoken = [f for f in frames if f["type"] == "transcription" and f.get("role") == "assistant"]
        self.assertEqual(spoken[0]["text"], deferral_for("en-IN"))

    def test_under_the_emergency_override_the_guidance_is_spoken_instead_of_lives_audio(self):
        """The Emergency Override keeps the text, guidance first; Live's own
        audio never carried that guidance, so it is withheld and the guarded
        text is spoken instead."""
        from dosage_guard import emergency_guidance_for

        said: List[str] = []

        async def speak(text: str, language: str) -> bytes:
            said.append(text)
            return b"SPOKEN-GUIDED-TEXT"

        ScriptedLiveService.script = [([b"LIVE-DOSE-AUDIO"], "Give morphine 10 mg now.",
                                       "My father is not breathing and unconscious")]
        with mock.patch("web_voice_fallback.speak_for_web", speak):
            with self.client.websocket_connect("/ws/voice") as ws:
                self.start(ws, {})
                frames = self.turn(ws)
        audio = [f["bytes"] for f in frames if f["type"] == "audio"]
        self.assertNotIn(b"LIVE-DOSE-AUDIO", audio)
        self.assertEqual(audio, [b"SPOKEN-GUIDED-TEXT"])
        self.assertEqual(len(said), 1)
        # Heard first, with the number to call (not cut as a citation would be).
        self.assertTrue(said[0].startswith("Call emergency services 108/102 immediately!"), said[0])
        spoken = [f for f in frames if f["type"] == "transcription" and f.get("role") == "assistant"]
        self.assertTrue(spoken[0]["text"].startswith(emergency_guidance_for("en-IN")))
        self.assertTrue(spoken[0].get("withheld"))

    def test_the_deferral_is_in_the_sessions_language(self):
        from dosage_guard import deferral_for

        ScriptedLiveService.script = [([b"DOSE-AUDIO"], "मॉर्फिन 10 mg हर 4 घंटे में दें।", "दर्द के लिए कितनी दवा दूं?")]
        with self.client.websocket_connect("/ws/voice") as ws:
            ws.send_json({"type": "config", "language": "hi-IN"})
            self.frames_until(ws, "config_ack")
            ws.send_json({"type": "start_audio"})
            self.frames_until(ws, "session_created")
            frames = self.turn(ws)
        self.assertEqual([f for f in frames if f["type"] == "audio"], [])
        spoken = [f for f in frames if f["type"] == "transcription" and f.get("role") == "assistant"]
        self.assertEqual(spoken[0]["text"], deferral_for("hi-IN"))
        self.assertNotEqual(deferral_for("hi-IN"), deferral_for("en-IN"))


if __name__ == "__main__":
    unittest.main()
