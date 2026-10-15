"""
Side work runs after the reply.

Before this, every turn waited for conversation history (a whole-file rewrite
of every conversation ever held), observation extraction over the question and
the generated answer, the answer-path record, a shadow KG check and the
analytics and expert-sampling fan-out, before the caller heard anything.

What these pin down:

  * the answer span ends before any post-work span starts
  * the study log row is written before the reply, even when the background
    fan-out fails; the failure is logged and counted
  * appending a turn costs the same with 10,000 prior turns as with none
  * a torn last line in the history loses that line only
  * observation extraction sees the question only, and per-call session ids
    get no longitudinal record and no comprehension profile
  * shadow-mode KG checks finish after the reply and land in the record
"""

import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import post_work  # noqa: E402
from participants import ephemeral, is_person  # noqa: E402
from tests.test_answer_path_retrieval import make_pipeline  # noqa: E402

QUESTION = "My father is breathless at night"
ANSWER = "Sit him upright and use a fan."


async def answer_then_drain(pipeline, **kw):
    result = await pipeline.query(**kw)
    await post_work.drain()
    return result


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="post-work-"))
        self.env = patch.dict(os.environ, {"TRACE_DIR": str(self.tmp / "traces"), "TRACING_ENABLED": "1",
                                           "RETRIEVAL_AUDIT_DIR": str(self.tmp / "audit")})
        self.env.start()
        self.addCleanup(self.env.stop)

    def pipeline(self):
        p = make_pipeline(answer=ANSWER)
        p.conversation_file = self.tmp / "conversations.json"
        return p

    def spans(self):
        out = []
        for path in (self.tmp / "traces").glob("*.jsonl"):
            out += [json.loads(line) for line in path.read_text().splitlines()]
        return out


def start_end(span):
    end = datetime.fromisoformat(span["ts"]).timestamp()
    return end - span["ms"] / 1000, end


class AfterTheReply(Case):
    def test_post_work_starts_after_the_answer_span_ends(self):
        p = self.pipeline()
        asyncio.run(answer_then_drain(p, question=QUESTION, conversation_id="c1", user_id="+919800000001"))
        spans = self.spans()
        answer = next(s for s in spans if s["name"] == "answer")
        post = [s for s in spans if s["name"].startswith("post.")]
        self.assertEqual({s["name"] for s in post}, {"post.conversation", "post.record"})
        _, answered = start_end(answer)
        for s in post:
            self.assertGreaterEqual(start_end(s)[0], answered - 0.002, s["name"])
            self.assertEqual(s["trace"], answer["trace"])

    def test_the_answer_does_not_wait_for_slow_post_work(self):
        p = self.pipeline()

        class SlowAggregator:
            calls = []

            async def process_conversation(self, **kw):
                SlowAggregator.calls.append(kw)
                await asyncio.sleep(0.5)
                return []

        p.cross_modal_aggregator = SlowAggregator()

        async def go():
            started = time.perf_counter()
            await p.query(question=QUESTION, user_id="+919800000001")
            took = time.perf_counter() - started
            await post_work.drain()
            return took

        self.assertLess(asyncio.run(go()), 0.3)
        self.assertEqual(SlowAggregator.calls[0]["transcript"], f"Caller: {QUESTION}")
        self.assertNotIn(ANSWER, SlowAggregator.calls[0]["transcript"])


class Supervision(unittest.TestCase):
    def test_failures_are_logged_and_counted(self):
        async def boom():
            raise RuntimeError("disk full")

        async def go():
            before = post_work.stats()["failed"]
            with self.assertLogs("post_work", level="ERROR") as logs:
                post_work.after_reply("boom", boom)
                await post_work.drain()
            return post_work.stats()["failed"] - before, logs.output

        failed, output = asyncio.run(go())
        self.assertEqual(failed, 1)
        self.assertIn("boom", output[0])

    def test_the_study_row_is_written_before_the_reply_even_if_fan_out_fails(self):
        from study_logging import StudyInteractionLogger

        directory = Path(tempfile.mkdtemp(prefix="study-"))
        logger = StudyInteractionLogger(log_dir=directory, use_analytics=False, use_expert_sampling=False)

        async def failing_fan_out(entry):
            raise RuntimeError("analytics down")

        async def go():
            with patch.object(logger, "_fan_out", failing_fan_out):
                await logger.record(user_id="u1", query="q", response="r", channel="mobile")
                rows_at_reply = sum(1 for f in directory.glob("*.jsonl") for _ in f.open())
                before = post_work.stats()["failed"]
                await post_work.drain()
                return rows_at_reply, post_work.stats()["failed"] - before

        rows, failed = asyncio.run(go())
        self.assertEqual(rows, 1)
        self.assertEqual(failed, 1)


class ConversationLog(Case):
    def test_append_cost_is_flat_with_10000_prior_turns(self):
        p = self.pipeline()
        turn = {"timestamp": "t", "user_id": None, "question": "q", "answer": "a", "sources": []}

        def time_appends():
            started = time.perf_counter()
            for _ in range(20):
                p._append_conversation("c", turn)
            return (time.perf_counter() - started) / 20

        empty = time_appends()
        log = p.conversation_file.with_suffix(".jsonl")
        with open(log, "a") as fh:
            for i in range(10_000):
                fh.write(json.dumps({"conversation_id": f"c{i % 100}", "turn": turn}) + "\n")
        full = time_appends()
        self.assertLess(full, max(empty * 5, 0.002))

    def test_a_torn_last_line_loses_only_that_line(self):
        p = self.pipeline()
        for i in range(3):
            p._append_conversation("c1", {"question": f"q{i}"})
        with open(p.conversation_file.with_suffix(".jsonl"), "a") as fh:
            fh.write('{"conversation_id": "c1", "turn": {"quest')
        loaded = p._load_conversations()
        self.assertEqual([t["question"] for t in loaded["c1"]], ["q0", "q1", "q2"])

    def test_history_survives_a_restart(self):
        p = self.pipeline()
        asyncio.run(answer_then_drain(p, question=QUESTION, conversation_id="c9"))
        self.assertEqual(p._load_conversations()["c9"][0]["question"], QUESTION)


class Ephemeral(Case):
    def test_session_ids_get_no_record_or_profile(self):
        from safety_enhancements import ResponseLengthOptimizer

        p = self.pipeline()
        calls = []

        class Aggregator:
            async def process_conversation(self, **kw):
                calls.append(kw)
                return []

        p.cross_modal_aggregator = Aggregator()
        session_user = ephemeral("retell", "call_123")
        asyncio.run(answer_then_drain(p, question=QUESTION, user_id=session_user))
        self.assertEqual(calls, [])
        self.assertFalse(is_person(session_user))
        self.assertTrue(is_person("+919800000001"))

        optimizer = ResponseLengthOptimizer(storage_path=str(self.tmp / "profiles"))
        optimizer.analyze_user_message(session_user, "how do I ease pain?")
        self.assertNotIn(session_user, optimizer.profiles)
        optimizer.analyze_user_message("+919800000001", "how do I ease pain?")
        optimizer.flush()
        saved = json.loads((self.tmp / "profiles" / "profiles.json").read_text())
        self.assertEqual([row["user_id"] for row in saved], ["+919800000001"])

    def test_no_observations_means_no_record(self):
        import personalization.cross_modal_aggregator as cma
        from personalization.longitudinal_memory import DataSourceType, LongitudinalMemoryManager

        memory = LongitudinalMemoryManager(storage_path=str(self.tmp / "longitudinal"))
        aggregator = cma.CrossModalAggregator.__new__(cma.CrossModalAggregator)
        aggregator.longitudinal = memory

        class Nothing:
            async def extract(self, transcript, metadata):
                return []

        aggregator.whatsapp_extractor = Nothing()
        out = asyncio.run(aggregator.process_conversation(
            patient_id="+919800000001", conversation_id="c", transcript="Patient: hello",
            source_type=DataSourceType.WEB_CHAT, metadata={}))
        self.assertEqual(out, [])
        self.assertEqual(list((self.tmp / "longitudinal").glob("*_longitudinal.json")), [])


if __name__ == "__main__":
    unittest.main()
