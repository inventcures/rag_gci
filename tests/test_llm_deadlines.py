"""
Every model call states its thinking level and how long its caller can wait.

Before this, every call ran at the model's default thinking level with a 45 s
timeout, and the DeepSeek fallback (when enabled) got a fresh 45 s after that.
Thinking tokens count against max_output_tokens, so at the default level a
2048-token answer was cut off mid-sentence. A failed generation reached the
caller as "Error generating response: ...".

What these pin down:

  * the thinking level per purpose, its config override, and the token
    headroom added on top of the visible-answer limit
  * translation calls get their own short timeout
  * the fallback gets what is left of the same budget, not a fresh one
  * a hung model on a voice channel (Retell, mobile voice) produces the spoken
    fallback within the voice deadline, never the error text
  * the thinking levels and deadlines are part of the release fingerprint
"""

import asyncio
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import client as llm_client  # noqa: E402
from llm.client import (  # noqa: E402
    GENERATION_FAILED_ANSWER,
    LLMClient,
    LLMSettings,
    deadline,
    load_llm_settings,
    remaining,
    set_llm_client,
)


class HangingClient(LLMClient):
    """Gemini never answers; DeepSeek answers after `fallback_delay`."""

    def __init__(self, settings, fallback_delay=0.0):
        super().__init__(settings)
        self.fallback_delay = fallback_delay
        self.calls = []

    async def _gemini(self, prompt, system, max_tokens, temperature, json_mode, thinking=None):
        self.calls.append(("gemini", thinking))
        await asyncio.sleep(3600)

    async def _deepseek(self, prompt, system, max_tokens, temperature, json_mode):
        self.calls.append(("deepseek", None))
        await asyncio.sleep(self.fallback_delay)
        return "fallback text"


def settings(**kw):
    base = dict(api_key="test-key")
    base.update(kw)
    return LLMSettings(**base)


class Levels(unittest.TestCase):
    def test_defaults_per_purpose(self):
        s = settings()
        self.assertEqual(s.thinking_for("query_translation"), "minimal")
        self.assertEqual(s.thinking_for("answer_translation"), "minimal")
        self.assertEqual(s.thinking_for("clarification"), "minimal")
        self.assertEqual(s.thinking_for("answer"), llm_client.DEFAULT_THINKING["answer"])
        self.assertEqual(s.thinking_for("kg_entity_extraction"), "high")

    def test_config_overrides_and_rejects_unknown_levels(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "config.yaml"
            cfg.write_text("llm:\n  thinking:\n    answer: medium\n    clarification: loads\n"
                           "  deadlines:\n    voice_s: 6\n    translation_s: 2.5\n")
            s = load_llm_settings(cfg, env={})
            self.assertEqual(s.thinking_for("answer"), "medium")
            self.assertEqual(s.thinking_for("clarification"), "minimal")
            self.assertEqual(s.voice_deadline_s, 6.0)
            self.assertEqual(s.timeout_for("query_translation"), 2.5)
            s = load_llm_settings(cfg, env={"LLM_VOICE_DEADLINE_S": "8"})
            self.assertEqual(s.voice_deadline_s, 8.0)

    def test_thinking_config_and_headroom(self):
        c = LLMClient(settings())
        cfg = c._gemini_config(None, 2048, 0.3, False, "low")
        self.assertEqual(str(cfg.thinking_config.thinking_level).split(".")[-1].upper(), "LOW")
        self.assertEqual(cfg.max_output_tokens, 2048 + llm_client.THINKING_HEADROOM["low"])
        cfg = c._gemini_config(None, 2048, 0.3, False, "high")
        self.assertGreaterEqual(cfg.max_output_tokens - 2048, 8192)
        old = LLMClient(settings(model="gemini-2.5-flash"))._gemini_config(None, 512, 0.1, False, "low")
        self.assertIsNone(old.thinking_config)
        self.assertEqual(old.max_output_tokens, 512)

    def test_each_call_passes_its_level(self):
        c = HangingClient(settings(timeout_s=0.05))
        asyncio.run(c.generate("x", purpose="query_translation"))
        asyncio.run(c.generate("x", purpose="answer", thinking="medium"))
        self.assertEqual(c.calls, [("gemini", "minimal"), ("gemini", "medium")])

    def test_answers_think_less_inside_a_voice_turn(self):
        from llm.client import in_voice_turn, within_voice_deadline

        c = HangingClient(settings(timeout_s=0.05))
        set_llm_client(c)
        try:
            async def go():
                self.assertFalse(in_voice_turn())
                await c.generate("x", purpose="answer")
                await within_voice_deadline(lambda: c.generate("x", purpose="answer"), 1.0)
                await within_voice_deadline(lambda: c.generate("x", purpose="query_translation"), 1.0)
                self.assertFalse(in_voice_turn())

            asyncio.run(go())
        finally:
            set_llm_client(None)
        self.assertEqual(c.calls, [("gemini", "low"), ("gemini", "minimal"), ("gemini", "minimal")])

    def test_answer_translation_has_its_own_deadline(self):
        s = settings()
        self.assertEqual(s.timeout_for("query_translation"), 3.0)
        self.assertEqual(s.timeout_for("answer_translation"), 8.0)
        self.assertEqual(s.timeout_for("answer"), 45.0)

    def test_describe_carries_levels_and_deadlines(self):
        d = settings().describe()
        self.assertEqual(d["thinking"]["query_translation"], "minimal")
        self.assertEqual(d["deadlines"]["voice_s"], llm_client.DEFAULT_VOICE_DEADLINE_S)
        self.assertNotIn("test-key", str(d))


class Budgets(unittest.TestCase):
    def test_translation_has_a_short_timeout(self):
        c = HangingClient(settings(translation_deadline_s=0.2))
        started = time.perf_counter()
        r = asyncio.run(c.generate("x", purpose="query_translation"))
        self.assertFalse(r.ok)
        self.assertLess(time.perf_counter() - started, 0.6)

    def test_fallback_shares_the_budget(self):
        c = HangingClient(settings(timeout_s=0.4, fallback_enabled=True, fallback_api_key="k"),
                          fallback_delay=5.0)
        started = time.perf_counter()
        r = asyncio.run(c.generate("x", purpose="answer"))
        self.assertFalse(r.ok)
        # Old behaviour: 0.4 s for Gemini, then a fresh 0.4 s for DeepSeek.
        self.assertLess(time.perf_counter() - started, 0.6)

    def test_fallback_answers_within_what_is_left(self):
        c = HangingClient(settings(timeout_s=0.5, fallback_enabled=True, fallback_api_key="k"))

        async def go():
            with deadline(0.3):
                return await c.generate("x", purpose="answer")

        r = asyncio.run(go())
        self.assertFalse(r.ok)  # Gemini used all 0.3 s; nothing left for the fallback
        self.assertIn("no time left", r.error)

    def test_enclosing_deadline_wins_and_nests(self):
        async def go():
            self.assertIsNone(remaining())
            with deadline(5):
                with deadline(0.1):
                    self.assertLess(remaining(), 0.11)
                with deadline(60):
                    self.assertLess(remaining(), 5.01)
            self.assertIsNone(remaining())

        asyncio.run(go())


class VoiceFallback(unittest.TestCase):
    """A hung model on a voice channel: the caller hears the fallback in time."""

    def setUp(self):
        self.hung = HangingClient(settings(voice_deadline_s=0.4))
        self.previous = llm_client._client
        set_llm_client(self.hung)

    def tearDown(self):
        set_llm_client(self.previous)

    def pipeline(self):
        from tests.test_answer_path_retrieval import make_pipeline

        p = make_pipeline()
        del p._generate_answer_with_citations  # the real one, calling the hung model
        return p

    def test_failed_generation_never_reaches_the_caller(self):
        p = self.pipeline()

        async def go():
            with deadline(0.3):
                return await p.query("How do I manage breathlessness at night?")

        result = asyncio.run(go())
        self.assertEqual(result["answer"].split("\n")[0][:40], GENERATION_FAILED_ANSWER[:40])
        self.assertNotIn("Error", result["answer"])
        self.assertEqual(result["model_used"], "error")

    def test_retell_speaks_the_fallback_within_the_deadline(self):
        from retell_integration.custom_llm_server import RetellCustomLLMHandler
        from tests.test_retell_turns import FakeSocket, turn, wait_for

        rag = self.pipeline()

        async def go():
            handler = RetellCustomLLMHandler(rag_pipeline=rag)
            ws = FakeSocket()
            call = asyncio.create_task(handler.handle_websocket(ws, "call-dl"))
            asked = asyncio.get_running_loop().time()
            await ws.inbox.put(turn(1, "My father is breathless at night"))
            await wait_for(lambda: ws.responses(), timeout=5)
            took = asyncio.get_running_loop().time() - asked
            await ws.inbox.put(None)
            await call
            return ws.responses()[0]["content"], took

        content, took = asyncio.run(go())
        self.assertLess(took, 0.4 + llm_client.VOICE_GRACE_S)
        self.assertNotIn("Error", content)
        self.assertTrue(content.strip())

    def test_mobile_voice_turn_answers_within_the_deadline(self):
        import voice_session as vs
        from tests.test_voice_session import FakeSafetyManager

        class Stuck:
            async def query(self, question, conversation_id=None, user_id=None, top_k=5, source_language="en"):
                await asyncio.sleep(3600)

        handler = vs.VoiceTurnHandler(rag_pipeline=Stuck(), safety_manager=FakeSafetyManager())
        started = time.perf_counter()
        turn = asyncio.run(handler.handle(vs.VoiceSession(), "breathless at night"))
        self.assertLess(time.perf_counter() - started, 0.4 + llm_client.VOICE_GRACE_S + 0.3)
        self.assertEqual(turn.answer, GENERATION_FAILED_ANSWER)


class Fingerprint(unittest.TestCase):
    def test_levels_and_deadlines_drift(self):
        import study_release

        with tempfile.TemporaryDirectory() as d:
            record = Path(d) / "release.yaml"
            record.write_text(
                "release_name: t\nstatus: draft\nmodels:\n  llm:\n"
                "    thinking: {answer: high}\n"
                "    deadlines: {voice_s: 7, translation_s: 3, default_s: 45}\n")
            release = study_release.StudyRelease(record=study_release.load_yaml(record), path=record)
            found = {f.component for f in release.check_drift()}
            self.assertIn("llm.thinking", found)
            live = release.live_fingerprint()
            self.assertIn("answer=", live["llm.thinking"])
            self.assertEqual(study_release.compact_mapping({"voice_s": 7, "a": "x"}), "a=x,voice_s=7")
            self.assertEqual(study_release.compact_mapping({"v": 7.0}), "v=7")

    def test_record_matches_the_live_defaults(self):
        import study_release

        path = Path(study_release.__file__).parent / "data/study/release.yaml"
        release = study_release.StudyRelease(record=study_release.load_yaml(path), path=path)
        found = {f.component: f for f in release.check_drift()}
        self.assertNotIn("llm.thinking", found)
        self.assertNotIn("llm.deadlines", found)


if __name__ == "__main__":
    unittest.main()
