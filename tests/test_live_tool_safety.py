#!/usr/bin/env python3
"""
The Live tool-call path must not be a way around the dose boundary.

gemini_live's tool handler returned raw retrieval text to Gemini, which then spoke
it. That path looks like the safe one, because it is grounded in the verified
knowledge base, and it was the one route that skipped the boundary entirely.

ADR 0007 also requires grounding to stay in India with only audio leaving, which is
why the tool calls back to this server rather than running retrieval at Google. So
the filtering has to happen here, on the way out to the model.
"""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DOSE_ANSWER = (
    "Start morphine at 10 mg every 4 hours and titrate by 5 mg. Also give "
    "paracetamol 1 g twice daily."
)


class FakeSafety:
    """Stands in for SafetyEnhancementsManager, which is what actually filters."""

    def __init__(self, response=None):
        self.response = response if response is not None else "Use opioids as guided by your clinician."
        self.calls = []

    def process_response(self, query, response, sources=None, language="en", **kw):
        self.calls.append({"query": query, "response": response})
        return type("R", (), {"response": self.response, "dosage_blocked": True})()


class FakeRag:
    def __init__(self, answer=DOSE_ANSWER):
        self.answer = answer

    async def query(self, question, conversation_id=None, user_id=None, top_k=5):
        return {"status": "success", "answer": self.answer,
                "sources": [{"filename": "handbook.pdf"}]}


def build_service(rag, safety):
    """A service shaped like the real one, including config, which the tool reads."""
    from gemini_live.config import GeminiLiveConfig

    return type("S", (), {
        "rag_pipeline": rag,
        "safety_manager": safety,
        "config": GeminiLiveConfig(rag_top_k=5),
    })()


def build_session(service):
    from gemini_live.config import GeminiLiveConfig
    from gemini_live.session_manager import GeminiLiveSession

    session = GeminiLiveSession.__new__(GeminiLiveSession)
    session.service = service
    session.session_id = "s1"
    session.config = GeminiLiveConfig(rag_top_k=5)
    return session


class LiveToolAppliesDoseBoundary(unittest.TestCase):
    def test_grounding_text_is_filtered_before_the_model_sees_it(self):
        safety = FakeSafety()
        session = build_session(build_service(FakeRag(), safety))

        out = asyncio.run(session._run_rag_query("how much morphine?"))

        self.assertEqual(len(safety.calls), 1, "the tool skipped the dose boundary")
        self.assertNotIn("10 mg", out)
        self.assertNotIn("5 mg", out)
        self.assertIn("opioids", out.lower())

    def test_the_boundary_receives_the_users_own_question(self):
        safety = FakeSafety()
        session = build_session(build_service(FakeRag(), safety))

        asyncio.run(session._run_rag_query("how much morphine?"))
        self.assertEqual(safety.calls[0]["query"], "how much morphine?")

    def test_no_knowledge_base_result_still_warns_rather_than_answers(self):
        """An empty retrieval must not read as permission to answer from memory."""
        safety = FakeSafety()
        session = build_session(build_service(None, safety))

        out = asyncio.run(session._run_rag_query("anything"))
        self.assertIn("No relevant information", out)
        self.assertEqual(safety.calls, [], "nothing was retrieved, so nothing to filter")


if __name__ == "__main__":
    unittest.main(verbosity=2)
