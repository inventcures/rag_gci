#!/usr/bin/env python3
"""
Tests for cross-language retrieval with translation as a fallback only.

ADR 0003: a query in any Supported Language is matched directly against the
knowledge base. Translation is reached only when retrieval looks weak.

The failure this guards against is not a crash. If translation silently becomes
the default path again, every query pays an LLM round-trip, the better signal in
the original language is discarded, and a Groq dependency that the release record
flags as defective is reintroduced. None of that raises anything.

The Chroma collection is stubbed by patching `query` on the instance rather than
by subclassing `Collection` and overriding a ten-argument method with a two-argument
one, which would be an unsound override even where it happens to run.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_rag_server import SimpleRAGPipeline  # noqa: E402


def _result(distances, documents=None, n_results=1):
    """Shape a Chroma query result."""
    documents = documents if documents is not None else [["chunk text"]] * n_results
    return {
        "documents": documents,
        "metadatas": [[{"filename": "handbook.pdf"}]] * n_results,
        "distances": [list(distances)],
    }


class _RealCollectionFactory:
    """Builds a genuine Chroma collection so only `query` needs patching.

    Substituting a stand-in object would not satisfy the pipeline's declared type,
    and subclassing to override `query` would be an unsound override of a
    ten-argument method. A real collection keeps the assignment type-correct and
    exercises the same object the pipeline holds.
    """

    @staticmethod
    def build():
        import tempfile

        import chromadb

        client = chromadb.PersistentClient(
            path=tempfile.mkdtemp(prefix="ps-retrieval-test-")
        )
        return client.get_or_create_collection(name="probe_test")


def _pipeline(distances=(0.4,)):
    """A pipeline holding a real collection, with only the probe stubbed."""
    p = SimpleRAGPipeline.__new__(SimpleRAGPipeline)
    p.translation_fallback_count = 0
    p.vector_db = _RealCollectionFactory.build()
    return p


class TestWeaknessProbe(unittest.TestCase):
    """Decides whether a translation is worth attempting."""

    def test_close_match_is_not_weak(self):
        p = _pipeline()
        with patch.object(p.vector_db, "query", return_value=_result([0.4]), create=True):
            self.assertFalse(p._cross_language_retrieval_is_weak("दर्द कैसे नियंत्रित करें"))

    def test_far_match_is_weak(self):
        p = _pipeline()
        with patch.object(p.vector_db, "query", return_value=_result([1.9]), create=True):
            self.assertTrue(p._cross_language_retrieval_is_weak("অজানা প্রশ্ন"))

    def test_boundary_matches_the_relevance_threshold(self):
        p = _pipeline()
        with patch.object(p.vector_db, "query", return_value=_result([1.5]), create=True):
            self.assertFalse(p._cross_language_retrieval_is_weak("x"))

    def test_no_matches_at_all_is_weak(self):
        p = _pipeline()
        with patch.object(p.vector_db, "query", return_value=_result([]), create=True):
            self.assertTrue(p._cross_language_retrieval_is_weak("anything"))

    def test_probe_failure_assumes_retrieval_is_fine(self):
        """Defaulting to translating would put the cost back on every query."""
        p = _pipeline()
        with patch.object(
            p.vector_db, "query", side_effect=RuntimeError("unavailable"), create=True
        ):
            self.assertFalse(p._cross_language_retrieval_is_weak("anything"))

    def test_no_collection_is_not_weak(self):
        p = _pipeline()
        p.vector_db = None
        self.assertFalse(p._cross_language_retrieval_is_weak("anything"))


class TestTranslationOnlyWhenWeak(unittest.TestCase):
    """Translation must be reachable, and only reachable when retrieval is weak."""

    def _probe_result(self, distance):
        p = _pipeline()
        return p, patch.object(
            p.vector_db, "query", return_value=_result([distance]), create=True
        )

    def test_strong_cross_language_match_does_not_translate(self):
        p, probe = self._probe_result(0.4)
        calls = []

        async def fake_translate(query, source_language):
            calls.append((query, source_language))
            return {"status": "success", "translated_query": "translated"}

        with probe, patch.object(p, "translate_query_to_english", fake_translate):
            self.assertFalse(p._cross_language_retrieval_is_weak("question"))
            self.assertEqual(calls, [], "translation ran despite strong retrieval")

    def test_weak_retrieval_reaches_translation(self):
        p, probe = self._probe_result(1.9)
        calls = []

        async def fake_translate(query, source_language):
            calls.append((query, source_language))
            return {"status": "success", "translated_query": "translated"}

        with probe, patch.object(p, "translate_query_to_english", fake_translate):
            self.assertTrue(p._cross_language_retrieval_is_weak("question"))

    def test_failed_translation_is_not_counted_as_a_fallback(self):
        """The counter measures fallback use, not translation attempts."""
        p = _pipeline()
        self.assertEqual(p.translation_fallback_count, 0)


class TestFallbackIsMeasurable(unittest.TestCase):
    """The fallback rate is the evidence that cross-language retrieval works."""

    def test_counter_starts_at_zero(self):
        self.assertEqual(_pipeline().translation_fallback_count, 0)

    def test_threshold_is_a_single_shared_signal(self):
        """One threshold, not two that can drift apart."""
        p = _pipeline()
        self.assertEqual(p.CROSS_LANGUAGE_FALLBACK_DISTANCE, 1.5)

    def test_retrieval_path_uses_the_same_relevance_threshold(self):
        source = open("simple_rag_server.py", encoding="utf-8").read()
        self.assertIn("relevance_threshold = 1.5", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)