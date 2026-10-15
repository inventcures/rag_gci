"""
Per-stage timing traces (observability/tracing.py, scripts/trace_summary.py).

What these pin down:

  * a request writes one root span and child spans, linked by trace and parent
  * a trace opened inside another (query() under a channel handler) nests
  * spans opened in asyncio.to_thread or a task attach to the request's trace
  * the HTTP middleware names the channel from the path and records the route
    template, never the concrete path
  * query() writes the answer-path stages, and the stage timings the release
    record carries are the same numbers as the spans
  * no span ever contains question text, answer text, a phone number or a name
  * the summary command reports p50/p95 per stage per channel
  * tracing failures never reach the caller
"""

import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from observability import record_span, span, trace_request  # noqa: E402
from observability.middleware import TraceMiddleware, channel_for_path  # noqa: E402

PHONE = "+919812345678"
QUESTION = "Meri maa ko raat mein saans phoolti hai, kya karein?"
ANSWER = "Sit her upright and use a fan near her face."


def read_spans(directory):
    spans = []
    for path in Path(directory).glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            spans.append(json.loads(line))
    return spans


class TraceCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="traces-")
        self.env = patch.dict(os.environ, {"TRACE_DIR": self.dir, "TRACING_ENABLED": "1"})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def spans(self):
        return read_spans(self.dir)


class SpanTree(TraceCase):
    def test_root_and_children_share_a_trace(self):
        with trace_request("bolna", user_id=PHONE) as root:
            with span("search", n_results=5):
                pass
            with span("generate"):
                with span("llm", purpose="answer"):
                    pass

        spans = {s["name"]: s for s in self.spans()}
        self.assertEqual(set(spans), {"request", "search", "generate", "llm"})
        self.assertEqual({s["trace"] for s in spans.values()}, {root.trace_id})
        self.assertIsNone(spans["request"]["parent"])
        self.assertEqual(spans["search"]["parent"], spans["request"]["span"])
        self.assertEqual(spans["llm"]["parent"], spans["generate"]["span"])
        self.assertTrue(all(s["channel"] == "bolna" for s in spans.values()))
        self.assertEqual(spans["search"]["attrs"], {"n_results": 5})

    def test_participant_is_the_study_pseudonym(self):
        from study_logging import pseudonymise

        with trace_request("whatsapp", user_id=PHONE):
            with span("search"):
                pass
        for s in self.spans():
            self.assertEqual(s["participant"], pseudonymise(PHONE))

    def test_nested_trace_request_becomes_a_child(self):
        with trace_request("retell", name="voice_turn"):
            with trace_request("direct", name="answer"):
                pass
        spans = {s["name"]: s for s in self.spans()}
        self.assertEqual(spans["answer"]["parent"], spans["voice_turn"]["span"])
        self.assertEqual(spans["answer"]["channel"], "retell")

    def test_threads_and_tasks_attach_to_the_request(self):
        async def turn():
            with trace_request("mobile_voice"):
                def blocking():
                    with span("embed"):
                        time.sleep(0.01)

                await asyncio.to_thread(blocking)

                async def side_work():
                    with span("post.observations"):
                        await asyncio.sleep(0)

                await asyncio.create_task(side_work())

        asyncio.run(turn())
        spans = {s["name"]: s for s in self.spans()}
        root = spans["request"]["span"]
        self.assertEqual(spans["embed"]["parent"], root)
        self.assertEqual(spans["post.observations"]["parent"], root)
        self.assertGreaterEqual(spans["embed"]["ms"], 10)

    def test_errors_are_recorded_and_reraised(self):
        with self.assertRaises(ValueError):
            with trace_request("web"):
                with span("generate"):
                    raise ValueError("boom")
        spans = {s["name"]: s for s in self.spans()}
        self.assertEqual(spans["generate"]["status"], "error")
        self.assertEqual(spans["generate"]["attrs"]["error"], "ValueError")

    def test_record_span_writes_an_already_timed_stage(self):
        with trace_request("web"):
            ms = record_span("retrieve", 12.345, method="vector")
        self.assertEqual(ms, 12.35)
        spans = {s["name"]: s for s in self.spans()}
        self.assertEqual(spans["retrieve"]["ms"], 12.35)
        self.assertEqual(spans["retrieve"]["parent"], spans["request"]["span"])

    def test_disabled_tracing_writes_nothing(self):
        with patch.dict(os.environ, {"TRACING_ENABLED": "0"}):
            with trace_request("web"):
                with span("search"):
                    pass
        self.assertEqual(self.spans(), [])

    def test_a_failing_writer_never_reaches_the_caller(self):
        with patch.dict(os.environ, {"TRACE_DIR": "/proc/definitely/not/writable"}):
            with trace_request("web"):
                with span("search"):
                    value = 42
        self.assertEqual(value, 42)

    def test_overhead_is_small(self):
        started = time.perf_counter()
        for _ in range(200):
            with trace_request("web"):
                with span("search"):
                    pass
        per_request_ms = (time.perf_counter() - started) * 1000 / 200
        self.assertLess(per_request_ms, 1.0)


class NoPHI(TraceCase):
    def test_text_bearing_attributes_are_dropped(self):
        with trace_request("whatsapp", user_id=PHONE, question=QUESTION):
            with span("generate", answer=ANSWER, prompt=QUESTION, transcript=QUESTION,
                      phone=PHONE, caller_name="Sunita", purpose="answer") as s:
                s.set(text=ANSWER, model="gemini-3.5-flash")
        raw = "".join(p.read_text() for p in Path(self.dir).glob("*.jsonl"))
        for forbidden in (QUESTION, ANSWER, PHONE, "Sunita"):
            self.assertNotIn(forbidden, raw)
        spans = {s["name"]: s for s in self.spans()}
        self.assertEqual(spans["generate"]["attrs"], {"purpose": "answer", "model": "gemini-3.5-flash"})

    def test_long_strings_are_truncated(self):
        with trace_request("web"):
            with span("search", method="x" * 500):
                pass
        spans = {s["name"]: s for s in self.spans()}
        self.assertEqual(len(spans["search"]["attrs"]["method"]), 64)


class Middleware(TraceCase):
    def test_channel_comes_from_the_path(self):
        self.assertEqual(channel_for_path("/api/bolna/query"), "bolna")
        self.assertEqual(channel_for_path("/api/mobile/v1/query"), "mobile")
        self.assertEqual(channel_for_path("/webhook"), "whatsapp")
        self.assertEqual(channel_for_path("/api/query"), "web")
        self.assertEqual(channel_for_path("/api/kg/stats"), "http")

    def test_route_template_not_concrete_path(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.add_middleware(TraceMiddleware)

        @app.get("/api/mobile/v1/patients/{patient_id}")
        async def patient(patient_id: str):
            with span("load"):
                return {"ok": True}

        @app.get("/health")
        async def health():
            return {"ok": True}

        client = TestClient(app)
        client.get(f"/api/mobile/v1/patients/{PHONE}")
        client.get("/health")

        spans = {s["name"]: s for s in self.spans()}
        self.assertEqual(set(spans), {"http", "load"})
        self.assertEqual(spans["http"]["channel"], "mobile")
        self.assertEqual(spans["http"]["attrs"]["route"], "/api/mobile/v1/patients/{patient_id}")
        self.assertEqual(spans["http"]["attrs"]["status_code"], 200)
        self.assertEqual(spans["load"]["parent"], spans["http"]["span"])
        self.assertNotIn(PHONE, json.dumps(self.spans()))


class AnswerPath(TraceCase):
    """query() end to end with the fake collection the answer-path tests use."""

    def setUp(self):
        super().setUp()
        self.audit_env = patch.dict(os.environ, {"RETRIEVAL_AUDIT_DIR": tempfile.mkdtemp()})
        self.audit_env.start()

    def tearDown(self):
        self.audit_env.stop()
        super().tearDown()

    def test_query_writes_each_stage_and_the_record_uses_the_same_numbers(self):
        from tests.test_answer_path_retrieval import make_pipeline

        pipeline = make_pipeline(answer=ANSWER)
        with trace_request("bolna", user_id=PHONE):
            result = asyncio.run(pipeline.query(question=QUESTION, user_id=None, source_language="en"))

        spans = self.spans()
        names = {s["name"] for s in spans}
        self.assertTrue({"request", "answer", "search", "retrieve", "generate", "record"} <= names, names)
        self.assertEqual({s["trace"] for s in spans}.__len__(), 1)

        by_name = {s["name"]: s for s in spans}
        stage = result["retrieval"]["stage_latency_ms"]
        self.assertEqual(stage["generation"], round(by_name["generate"]["ms"], 1))
        self.assertEqual(stage["retrieval"], round(by_name["retrieve"]["ms"], 1))

        raw = "".join(p.read_text() for p in Path(self.dir).glob("*.jsonl"))
        for forbidden in (QUESTION, ANSWER, PHONE):
            self.assertNotIn(forbidden, raw)

    def test_non_english_query_traces_the_probe(self):
        from tests.test_answer_path_retrieval import make_pipeline

        pipeline = make_pipeline(answer=ANSWER)
        asyncio.run(pipeline.query(question=QUESTION, source_language="hi"))
        names = [s["name"] for s in self.spans()]
        self.assertIn("probe", names)
        self.assertIn("search", names)


class Summary(TraceCase):
    def test_p50_p95_and_share_per_channel(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import trace_summary

        for ms in (10, 20, 30, 40):
            with trace_request("retell", name="voice_turn"):
                record_span("generate", ms)
        with trace_request("web"):
            record_span("search", 5)

        summary = trace_summary.summarise(trace_summary.load(next(Path(self.dir).glob("*.jsonl"))))
        self.assertEqual(set(summary), {"retell", "web"})
        generate = summary["retell"]["stages"]["generate"]
        self.assertEqual(generate["count"], 4)
        self.assertEqual(generate["p50_ms"], 20.0)
        self.assertEqual(generate["p95_ms"], 40.0)
        self.assertEqual(summary["retell"]["roots"], 4)
        self.assertIsNotNone(generate["share"])
        markdown = trace_summary.to_markdown(summary)
        self.assertIn("### retell (4 requests)", markdown)
        self.assertIn("| generate | 4 | 20.0 | 40.0 |", markdown)


if __name__ == "__main__":
    unittest.main()


class TraceFileHandles(TraceCase):
    """A turn's spans share one open file instead of opening it per span."""

    def test_one_open_per_file(self):
        from observability import tracing

        tracing.close_trace_files()
        opens = []
        real_open = os.open

        def counting_open(path, *args, **kwargs):
            opens.append(path)
            return real_open(path, *args, **kwargs)

        with patch.object(tracing.os, "open", counting_open):
            with trace_request("web"):
                for _ in range(10):
                    with span("stage"):
                        pass
        self.assertEqual(len(opens), 1)
        self.assertEqual(len(self.spans()), 11)

    def test_removed_file_is_reopened(self):
        with span("first"):
            pass
        for path in Path(self.dir).glob("*.jsonl"):
            path.unlink()
        with span("second"):
            pass
        self.assertEqual([s["name"] for s in self.spans()], ["second"])

    def test_new_trace_dir_is_used(self):
        with span("here"):
            pass
        other = tempfile.mkdtemp(prefix="traces-")
        with patch.dict(os.environ, {"TRACE_DIR": other}):
            with span("there"):
                pass
        self.assertEqual([s["name"] for s in read_spans(other)], ["there"])
        self.assertEqual([s["name"] for s in self.spans()], ["here"])
