"""
The V2 study collectors record what they are given, in temporary storage.

Moved from the root `test_v2_modules.py` (round-3 ticket 20), which printed each
module's output, asserted nothing, and wrote into the real data/ directory. The
study logger (study_logging.py) wraps RealtimeMetrics, UsageAnalytics and the
ExpertSampler, so these numbers reach the study dashboard.

What these pin down, each against a fresh temporary store:

  * the clinical validator flags an "as much morphine as you need" answer as a
    safety issue
  * expert review samples every answer at rate 1.0 and none at rate 0.0
  * the RAG success rate and latency percentiles are computed from what was recorded
  * usage is broken out by language
  * feedback is counted with its rating, channel and language, and the Hindi
    feedback prompt is Hindi
  * a follow-up question is recognised as one; a new topic is not
  * a recorded symptom and allergy appear in the patient context summary
"""

import asyncio
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics import MetricType, RealtimeMetrics, UsageAnalytics  # noqa: E402
from clinical_validation import ClinicalValidator, ExpertSampler, FeedbackCollector  # noqa: E402
from clinical_validation.feedback import FeedbackChannel, FeedbackType  # noqa: E402
from personalization import ContextMemory, InteractionHistory  # noqa: E402

SAFE_QUERY = "How do I prevent bedsores?"
SAFE_ANSWER = "Change the patient's position often and keep the skin clean and dry. Ask your nurse."
SOURCES = [{"title": "Home care guide", "content": "Change position often and keep skin dry."}]


def run(coro):
    return asyncio.run(coro)


class TempStore(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="v2-modules-")
        self.addCleanup(shutil.rmtree, self.dir, True)

    def path(self, name: str) -> str:
        return os.path.join(self.dir, name)


class ClinicalValidation(TempStore):
    def test_an_unlimited_morphine_answer_is_flagged_as_a_safety_issue(self):
        result = ClinicalValidator().validate(
            query="Can I take as much morphine as I want?",
            response="You can take as much morphine as you need for pain relief.",
            sources=[], context=None,
        )
        self.assertIn("safety", [issue.category.value for issue in result.issues])

    def test_a_grounded_care_answer_raises_no_safety_issue(self):
        result = ClinicalValidator().validate(
            query=SAFE_QUERY, response=SAFE_ANSWER, sources=SOURCES, context=None,
        )
        self.assertNotIn("safety", [issue.category.value for issue in result.issues])

    def sample(self, rate: float):
        sampler = ExpertSampler(storage_path=self.path(f"samples-{rate}"),
                                sample_rate=rate, max_samples_per_day=10)
        validation = ClinicalValidator().validate(
            query=SAFE_QUERY, response=SAFE_ANSWER, sources=SOURCES, context=None,
        ).to_dict()
        sample = run(sampler.maybe_sample(
            query=SAFE_QUERY, response=SAFE_ANSWER, language="en-IN", sources=SOURCES,
            validation_result=validation, session_id="session-1", user_id="user-1",
        ))
        return sampler, sample

    def test_expert_review_samples_every_answer_at_rate_one(self):
        sampler, sample = self.sample(1.0)
        self.assertIsNotNone(sample)
        self.assertEqual(sample.status.value, "pending")
        self.assertEqual(len(run(sampler.get_pending_samples(limit=5))), 1)
        self.assertEqual(run(sampler.get_statistics())["total_samples"], 1)

    def test_expert_review_samples_nothing_at_rate_zero(self):
        sampler, sample = self.sample(0.0)
        self.assertIsNone(sample)
        self.assertEqual(run(sampler.get_pending_samples(limit=5)), [])

    def test_feedback_is_counted_with_its_rating_channel_and_language(self):
        collector = FeedbackCollector(storage_path=self.path("feedback"), auto_prompt_rate=0.5)
        feedback = run(collector.collect_feedback(
            query=SAFE_QUERY, response=SAFE_ANSWER, rating=4,
            feedback_type=FeedbackType.HELPFUL, channel=FeedbackChannel.VOICE_PROMPT,
            session_id="session-1", language="hi-IN",
        ))
        self.assertEqual(feedback.rating, 4)
        stats = run(collector.get_statistics(days=7))
        self.assertEqual(stats["total_feedback"], 1)
        self.assertEqual(stats["average_rating"], 4.0)
        self.assertEqual(stats["by_channel"]["voice_prompt"], 1)
        self.assertEqual(stats["by_language"]["hi-IN"]["count"], 1)

    def test_the_hindi_feedback_prompt_is_in_hindi(self):
        collector = FeedbackCollector(storage_path=self.path("feedback"), auto_prompt_rate=0.5)
        ask = collector.get_feedback_prompt("hi-IN")["ask"]
        self.assertTrue(any("ऀ" <= ch <= "ॿ" for ch in ask), ask)


class Analytics(TempStore):
    def test_rag_success_rate_is_the_share_of_successful_retrievals(self):
        metrics = RealtimeMetrics(window_seconds=300, enable_alerts=True)
        for success in (True, True, False):
            run(metrics.record_rag_query(success=success, latency_ms=100, sources_count=1))
        rate = run(metrics.get_success_rate(MetricType.RAG_SUCCESS_RATE))
        self.assertEqual((rate["count"], rate["successful"], rate["failed"]), (3, 2, 1))
        self.assertAlmostEqual(rate["rate"], 2 / 3, places=3)

    def test_latency_percentiles_come_from_the_recorded_latencies(self):
        metrics = RealtimeMetrics(window_seconds=300, enable_alerts=True)

        async def record():
            for i in range(10):
                await metrics.record_latency(MetricType.RESPONSE_LATENCY_MS, 200 + i * 50)

        run(record())
        stats = run(metrics.get_latency_stats(MetricType.RESPONSE_LATENCY_MS))
        self.assertEqual((stats["count"], stats["min"], stats["max"]), (10, 200, 650))
        self.assertEqual(stats["avg"], 425.0)
        self.assertGreater(stats["p95"], stats["p50"])

    def test_usage_is_broken_out_by_language(self):
        usage = UsageAnalytics(storage_path=self.path("usage"), retention_days=90)

        async def record():
            for i in range(5):
                await usage.record_query(
                    user_id=f"user_{i}", language="hi-IN" if i % 2 == 0 else "en-IN",
                    query_type="symptom_inquiry", used_rag=True, rag_success=True,
                    validation_passed=True,
                )

        run(record())
        stats = run(usage.get_language_stats(days=1))
        self.assertEqual(stats["total_queries"], 5)
        self.assertEqual(stats["by_language"]["hi-IN"]["count"], 3)
        self.assertEqual(stats["by_language"]["en-IN"]["count"], 2)
        self.assertEqual(run(usage.get_summary())["today"]["unique_users"], 5)


class Personalization(TempStore):
    def test_a_follow_up_question_is_recognised_and_a_new_topic_is_not(self):
        history = InteractionHistory(storage_path=self.path("history"), max_history_days=30)

        async def converse():
            await history.get_or_create_session(user_id="user-1", language="hi-IN", channel="voice")
            await history.add_turn(
                user_id="user-1", query="How to manage pain at night?",
                response="Keep a regular routine.", language="hi-IN", used_rag=True,
                response_time_ms=300,
            )
            return await history.get_recent_context("user-1", max_turns=5)

        recent = run(converse())
        self.assertEqual(len(recent), 1)
        self.assertTrue(history.is_followup_query("What about at night?", recent))
        self.assertFalse(history.is_followup_query("How do I prevent bedsores?", recent))

    def test_a_recorded_symptom_and_allergy_reach_the_context_summary(self):
        memory = ContextMemory(storage_path=self.path("context"), context_expiry_days=90)

        async def record():
            await memory.get_or_create_context("user-1")
            await memory.add_symptom("user-1", "breathlessness", "moderate")
            await memory.add_allergy("user-1", "ibuprofen")
            return await memory.get_or_create_context("user-1")

        summary = memory.get_context_summary(run(record()))
        self.assertIn("breathlessness", summary)
        self.assertIn("ibuprofen", summary)


if __name__ == "__main__":
    unittest.main()
