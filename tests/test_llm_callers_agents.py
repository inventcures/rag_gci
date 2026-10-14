#!/usr/bin/env python3
"""
Memory agents, meta agent, Context-1 and the clarification system call the
shared LLM client (llm/client.py), not Groq.

Each caller is checked twice with a fake client: a good response is used, and
an `ok=False` response (no key, or Gemini and DeepSeek both failed) sends the
caller to its existing rule-based path instead of raising.
"""

import asyncio
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import LLMResponse, set_llm_client  # noqa: E402


class FakeLLM:
    def __init__(self, text: str = "", ok: bool = True):
        self.text = text
        self.ok = ok
        self.calls = []

    async def generate(self, prompt, **kw):
        self.calls.append({"prompt": prompt, **kw})
        if not self.ok:
            return LLMResponse("", False, "none", "gemini-test", error="no key")
        return LLMResponse(self.text, True, "gemini", "gemini-test")


def run(coro):
    return asyncio.run(coro)


class _Base(unittest.TestCase):
    def tearDown(self):
        set_llm_client(None)

    def use(self, text: str = "", ok: bool = True) -> FakeLLM:
        fake = FakeLLM(text, ok)
        set_llm_client(fake)
        return fake


class TestMemoryAgents(_Base):
    def _store(self):
        from memory_agents.memory_store import MemoryStore

        tmp = tempfile.mkdtemp()
        return MemoryStore(db_path=os.path.join(tmp, "mem.db"))

    def test_ingest_uses_shared_client(self):
        from memory_agents.ingest_agent import PatientIngestAgent

        fake = self.use(
            '{"summary": "Pain 7/10 after morphine", "entities": ["pain"], '
            '"topics": ["symptom_management"], "importance_score": 0.8}'
        )
        agent = PatientIngestAgent(self._store())
        out = run(agent._llm_extract("Pain 7/10"))
        self.assertEqual(out["summary"], "Pain 7/10 after morphine")
        self.assertEqual(out["importance_score"], 0.8)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.calls[0]["purpose"], "memory_ingest")
        self.assertFalse(hasattr(agent, "_groq_api_key"))

    def test_ingest_degrades_when_llm_unavailable(self):
        from memory_agents.ingest_agent import PatientIngestAgent

        self.use(ok=False)
        self.assertIsNone(run(PatientIngestAgent(self._store())._llm_extract("Pain 7/10")))

    def test_query_agent_degrades_to_rule_based(self):
        from memory_agents.query_agent import PatientQueryAgent

        mem = SimpleNamespace(timestamp=1_700_000_000, summary="Pain worse at night")
        agent = PatientQueryAgent(self._store())

        fake = self.use("Pain is worse at night (Observation #1).")
        self.assertEqual(
            run(agent._synthesize("How is the pain?", [mem], [])),
            "Pain is worse at night (Observation #1).",
        )
        self.assertEqual(fake.calls[0]["purpose"], "memory_query")

        self.use(ok=False)
        self.assertIsNone(run(agent._synthesize("How is the pain?", [mem], [])))


class TestMetaAgent(_Base):
    def test_evaluator_uses_shared_client_and_degrades(self):
        from meta_agent.evaluator import ResponseEvaluator

        evaluator = ResponseEvaluator(evaluations_dir=tempfile.mkdtemp())
        fake = self.use(
            '{"relevance": 0.9, "safety": 1.0, "completeness": 0.7, "citation": 0.5, "overall": 0.8}'
        )
        scores = run(evaluator._llm_evaluate("q", "r", None))
        self.assertEqual(scores["overall"], 0.8)
        self.assertEqual(fake.calls[0]["purpose"], "meta_evaluate")

        self.use(ok=False)
        self.assertIsNone(run(evaluator._llm_evaluate("q", "r", None)))

    def test_clinical_meta_agent_falls_back_to_rules(self):
        from meta_agent.clinical_meta_agent import ClinicalMetaAgent
        from meta_agent.evaluator import ResponseEvaluator

        agent = ClinicalMetaAgent(evaluator=ResponseEvaluator(evaluations_dir=tempfile.mkdtemp()))
        patterns = {"low_relevance": ["q1"]}

        fake = self.use(
            '[{"domain": "retrieval", "change_type": "prompt_modification", "description": "x"}]'
        )
        improvements = run(agent._generate_improvements(patterns, []))
        self.assertEqual([i.description for i in improvements], ["x"])
        self.assertEqual(fake.calls[0]["purpose"], "meta_improve")

        self.use(ok=False)
        expected = agent._rule_based_improvements(patterns)
        got = run(agent._generate_improvements(patterns, []))
        self.assertEqual(
            [(i.domain, i.change_type) for i in got],
            [(i.domain, i.change_type) for i in expected],
        )


class TestContext1(_Base):
    def test_decompose_uses_shared_client_and_degrades(self):
        from context1_integration.agent import Context1RetrievalAgent
        from context1_integration.config import Context1Config

        agent = Context1RetrievalAgent(Context1Config())
        query = "morphine dosing for renal failure and constipation management"

        fake = self.use('```json\n["morphine renal dosing", "opioid constipation"]\n```')
        self.assertEqual(
            run(agent._decompose_query(query)),
            ["morphine renal dosing", "opioid constipation"],
        )
        self.assertEqual(fake.calls[0]["purpose"], "context1_decompose")

        self.use(ok=False)
        self.assertEqual(run(agent._decompose_query(query)), agent._rule_based_decompose(query))

    def test_old_groq_model_key_is_ignored(self):
        from context1_integration.config import Context1Config

        cfg = Context1Config.from_dict({"groq_model": "llama-3.1-8b-instant", "max_hops": 2})
        self.assertEqual(cfg.max_hops, 2)
        self.assertFalse(hasattr(cfg, "groq_model"))


class TestClarification(_Base):
    def test_clarity_analysis_uses_shared_client_and_degrades(self):
        from smart_clarification_system import ClarityLevel, SmartClarificationSystem

        # Old callers still pass a key positionally; it is ignored.
        system = SmartClarificationSystem("ignored-key")

        fake = self.use(
            '{"needs_clarification": true, "clarity_level": "ambiguous", "confidence": 0.9, '
            '"reasoning": "no location", "suggested_questions": ["Where is the pain?"]}'
        )
        out = run(system._llm_clarity_analysis("I have pain"))
        self.assertEqual(out["clarity_level"], ClarityLevel.AMBIGUOUS)
        self.assertEqual(out["suggested_questions"], ["Where is the pain?"])
        self.assertEqual(fake.calls[0]["purpose"], "clarification")

        self.use(ok=False)
        out = run(system._llm_clarity_analysis("I have pain"))
        self.assertFalse(out["needs_clarification"])
        self.assertEqual(out["clarity_level"], ClarityLevel.CLEAR)

    def test_no_key_constructor(self):
        from smart_clarification_system import SmartClarificationSystem

        self.assertIsNotNone(SmartClarificationSystem())


if __name__ == "__main__":
    unittest.main()
