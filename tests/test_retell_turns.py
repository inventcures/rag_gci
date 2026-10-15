"""
Retell answers every turn, and every answer passes the Dose Boundary.

Before this, every Retell turn failed: the endpoint and the handler both
accepted the socket, the cleanup called a method that did not exist, and the RAG
call passed `conversation_context=`, which query() does not accept. Nothing in
the path applied the dose guard, so the obvious one-line fix would have spoken
doses.

What these pin down:

  * a response_required gets a response, built from query() called with its
    real signature (the fake has no **kw, as in the text-path dose tests)
  * a dose-bearing draft is redacted before Retell can speak it
  * ping_pong is answered while a slow answer is still being generated
  * a newer response_required cancels the older turn; only the newer answer
    is sent
  * session cleanup is safe and idempotent; the endpoint never accepts twice
"""

import asyncio
import inspect
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retell_integration.custom_llm_server import RetellCustomLLMHandler  # noqa: E402
from tests.test_text_path_dose_guard import (  # noqa: E402
    DOSE_ANSWER,
    RealisticRag,
    assert_no_dose,
)


class SlowRag(RealisticRag):
    def __init__(self, answer, delay):
        super().__init__(answer=answer)
        self.delay = delay

    async def query(self, question: str, conversation_id=None, user_id=None,
                    top_k: int = 5, source_language: str = "en"):
        await asyncio.sleep(self.delay)
        return await super().query(question, conversation_id, user_id, top_k, source_language)


class FakeSocket:
    """Enough of Starlette's WebSocket for the handler. accept() twice raises, as Starlette does."""

    def __init__(self):
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.sent = []
        self.sent_at = []
        self.accepts = 0

    async def accept(self):
        self.accepts += 1
        if self.accepts > 1:
            raise RuntimeError('Expected ASGI message "websocket.send" or "websocket.close"')

    async def receive_text(self):
        item = await self.inbox.get()
        if item is None:
            from fastapi import WebSocketDisconnect

            raise WebSocketDisconnect()
        return json.dumps(item)

    async def send_json(self, data):
        self.sent.append(data)
        self.sent_at.append(asyncio.get_running_loop().time())

    def responses(self):
        return [m for m in self.sent if m.get("response_type") == "response"]


def turn(response_id, text):
    return {"interaction_type": "response_required", "response_id": response_id,
            "transcript": [{"role": "user", "content": text}]}


async def wait_for(predicate, timeout=3.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


class RetellTurns(unittest.TestCase):
    def run_call(self, rag, script):
        async def go():
            handler = RetellCustomLLMHandler(rag_pipeline=rag)
            ws = FakeSocket()
            call = asyncio.create_task(handler.handle_websocket(ws, "call-1"))
            await script(ws)
            await ws.inbox.put(None)
            await call
            return handler, ws

        return asyncio.run(go())

    def test_a_turn_gets_a_guarded_answer(self):
        rag = RealisticRag(answer=DOSE_ANSWER)

        async def script(ws):
            await ws.inbox.put({"interaction_type": "call_details", "call": {"metadata": {"language": "en-IN"}}})
            await ws.inbox.put(turn(1, "How should morphine be given for cancer pain?"))
            await wait_for(lambda: ws.responses())

        handler, ws = self.run_call(rag, script)
        self.assertEqual(ws.accepts, 1)
        responses = ws.responses()
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0]["response_id"], 1)
        self.assertTrue(responses[0]["content_complete"])
        self.assertTrue(responses[0]["content"].strip())
        assert_no_dose(self, responses[0]["content"], "Retell")
        self.assertEqual(len(rag.calls), 1)
        self.assertEqual(rag.calls[0]["source_language"], "en")

    def test_query_is_called_with_its_real_signature(self):
        from simple_rag_server import SimpleRAGPipeline

        real = inspect.signature(SimpleRAGPipeline.query)
        fake = inspect.signature(RealisticRag.query)
        self.assertEqual(list(real.parameters), list(fake.parameters))

    def test_ping_pong_flows_while_an_answer_is_slow(self):
        rag = SlowRag(answer="Sit upright and use a fan.", delay=0.5)

        async def script(ws):
            await ws.inbox.put(turn(1, "My father is breathless at night"))
            await asyncio.sleep(0.05)
            sent_ping = asyncio.get_running_loop().time()
            await ws.inbox.put({"interaction_type": "ping_pong", "timestamp": 123})
            await wait_for(lambda: any(m.get("response_type") == "ping_pong" for m in ws.sent))
            pong_at = next(t for m, t in zip(ws.sent, ws.sent_at) if m.get("response_type") == "ping_pong")
            self.assertLess(pong_at - sent_ping, 0.1)
            self.assertEqual(ws.responses(), [], "pong must not wait for the answer")
            await wait_for(lambda: ws.responses())

        self.run_call(rag, script)

    def test_a_newer_question_cancels_the_older_answer(self):
        rag = SlowRag(answer="Sit upright and use a fan.", delay=0.3)

        async def script(ws):
            await ws.inbox.put(turn(1, "My father is breathless"))
            await asyncio.sleep(0.05)
            await ws.inbox.put(turn(2, "Actually he is also in pain"))
            await wait_for(lambda: ws.responses())
            await asyncio.sleep(0.4)

        _, ws = self.run_call(rag, script)
        self.assertEqual([r["response_id"] for r in ws.responses()], [2])

    def test_end_session_is_idempotent_and_cancels_the_turn(self):
        async def go():
            handler = RetellCustomLLMHandler(rag_pipeline=SlowRag("x", delay=5))
            ws = FakeSocket()
            call = asyncio.create_task(handler.handle_websocket(ws, "call-2"))
            await ws.inbox.put(turn(1, "breathless at night"))
            await wait_for(lambda: "call-2" in handler.active_sessions
                           and handler.active_sessions["call-2"].current_task is not None)
            task = handler.active_sessions["call-2"].current_task
            handler.end_session("call-2")
            handler.end_session("call-2")
            await asyncio.sleep(0)
            self.assertTrue(task.cancelled() or task.done())
            await ws.inbox.put(None)
            await call
            self.assertNotIn("call-2", handler.active_sessions)

        asyncio.run(go())

    def test_a_failing_pipeline_still_answers(self):
        class Broken(RealisticRag):
            async def query(self, question: str, conversation_id=None, user_id=None,
                            top_k: int = 5, source_language: str = "en"):
                raise RuntimeError("chroma down")

        async def script(ws):
            await ws.inbox.put(turn(1, "breathless at night"))
            await wait_for(lambda: ws.responses())

        _, ws = self.run_call(Broken(), script)
        self.assertTrue(ws.responses()[0]["content"])


class RetellEndpoint(unittest.TestCase):
    def test_endpoint_does_not_accept_before_the_handler(self):
        source = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "simple_rag_server.py"), encoding="utf-8").read()
        start = source.index('async def retell_llm_websocket')
        body = source[start:source.index("@app.post(\"/api/retell/webhook\"", start)]
        enabled_path = body[body.index("logger.info(f\"Retell LLM WebSocket connected"):]
        self.assertNotIn("websocket.accept()", enabled_path)
        self.assertIn("end_session", body)
        self.assertTrue(hasattr(RetellCustomLLMHandler, "end_session"))


if __name__ == "__main__":
    unittest.main()
