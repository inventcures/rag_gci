#!/usr/bin/env python3
"""
Tests for retrieval index identity and drift detection.

The failure this guards against is silent and was measured: an index built with an
English-only embedding model scores 1.00 on English and 0.00 on all ten Indic
languages. Nothing errors. It just returns confident, unfounded guidance.

A dimension or model mismatch between the running configuration and the live index
must therefore be detected and reported rather than searched through.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval_index import (  # noqa: E402
    EmbeddingIdentity,
    IndexDrift,
    detect_drift,
    load_identity_from_config,
)


class TestEmbeddingIdentity(unittest.TestCase):
    def test_identity_carries_model_and_dimension(self):
        identity = EmbeddingIdentity(model="BAAI/bge-m3", dimension=1024)
        self.assertEqual(identity.model, "BAAI/bge-m3")
        self.assertEqual(identity.dimension, 1024)

    def test_identity_is_hashable_and_comparable(self):
        a = EmbeddingIdentity(model="BAAI/bge-m3", dimension=1024)
        b = EmbeddingIdentity(model="BAAI/bge-m3", dimension=1024)
        c = EmbeddingIdentity(model="all-MiniLM-L6-v2", dimension=384)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len({a, b, c}), 2)

    def test_zero_dimension_is_rejected(self):
        with self.assertRaises(ValueError):
            EmbeddingIdentity(model="x", dimension=0)

    def test_missing_model_is_rejected(self):
        with self.assertRaises(ValueError):
            EmbeddingIdentity(model="", dimension=384)


class TestDriftDetection(unittest.TestCase):
    """The measured failure: 384-dim English-only index, 1024-dim multilingual config."""

    def test_matching_identity_is_no_drift(self):
        expected = EmbeddingIdentity(model="BAAI/bge-m3", dimension=1024)
        observed = EmbeddingIdentity(model="BAAI/bge-m3", dimension=1024)
        result = detect_drift(expected, observed)
        self.assertFalse(result.has_drift)
        self.assertEqual(result.reasons, [])

    def test_dimension_mismatch_is_drift(self):
        expected = EmbeddingIdentity(model="BAAI/bge-m3", dimension=1024)
        observed = EmbeddingIdentity(model="bge-small-en-v1.5", dimension=384)
        result = detect_drift(expected, observed)
        self.assertTrue(result.has_drift)
        self.assertIn("dimension_mismatch", result.reasons)
        self.assertIn("model_mismatch", result.reasons)

    def test_dimension_mismatch_alone_is_drift(self):
        # Same model name, different dimensionality: still refuse to search.
        expected = EmbeddingIdentity(model="BAAI/bge-m3", dimension=1024)
        observed = EmbeddingIdentity(model="BAAI/bge-m3", dimension=768)
        result = detect_drift(expected, observed)
        self.assertTrue(result.has_drift)
        self.assertIn("dimension_mismatch", result.reasons)
        self.assertNotIn("model_mismatch", result.reasons)

    def test_model_mismatch_alone_is_drift(self):
        expected = EmbeddingIdentity(model="BAAI/bge-m3", dimension=384)
        observed = EmbeddingIdentity(model="all-MiniLM-L6-v2", dimension=384)
        result = detect_drift(expected, observed)
        self.assertTrue(result.has_drift)
        self.assertIn("model_mismatch", result.reasons)

    def test_unknown_observed_dimension_is_drift_not_silence(self):
        # An unobservable index must not be treated as healthy.
        expected = EmbeddingIdentity(model="BAAI/bge-m3", dimension=1024)
        result = detect_drift(expected, None)
        self.assertTrue(result.has_drift)
        self.assertIn("index_unobservable", result.reasons)

    def test_drift_is_not_a_warning(self):
        # Drift must be actionable: it has to carry what to do about it.
        expected = EmbeddingIdentity(model="BAAI/bge-m3", dimension=1024)
        observed = EmbeddingIdentity(model="bge-small-en-v1.5", dimension=384)
        result = detect_drift(expected, observed)
        self.assertIn("reindex", result.remediation.lower())

    def test_report_is_serialisable_for_the_release_record(self):
        expected = EmbeddingIdentity(model="BAAI/bge-m3", dimension=1024)
        observed = EmbeddingIdentity(model="bge-small-en-v1.5", dimension=384)
        payload = detect_drift(expected, observed).to_dict()
        for key in ("expected", "observed", "reasons", "remediation"):
            self.assertIn(key, payload)
        self.assertEqual(payload["expected"]["dimension"], 1024)
        self.assertEqual(payload["observed"]["dimension"], 384)


class TestObserveIndex(unittest.TestCase):
    """
    observe_index reads a live ChromaDB collection.

    These use fakes rather than a real collection so the suite stays fast and so
    the shapes ChromaDB actually returns are pinned down deliberately.
    """

    def test_reads_dimension_from_a_list_of_vectors(self):
        import retrieval_index as ri

        class FakeCollection:
            metadata = None

            def get(self, limit=1, include=None):
                return {"embeddings": [[0.0] * 1024]}

        self.assertEqual(ri.observe_index(FakeCollection()).dimension, 1024)

    def test_handles_numpy_backed_embeddings(self):
        """ChromaDB returns a numpy array; truthiness on it is ambiguous."""
        import numpy as np

        import retrieval_index as ri

        class FakeCollection:
            metadata = None

            def get(self, limit=1, include=None):
                return {"embeddings": np.zeros((1, 384), dtype="float32")}

        observed = ri.observe_index(FakeCollection())
        self.assertIsNotNone(observed)
        self.assertEqual(observed.dimension, 384)

    def test_empty_collection_is_unobservable_not_drifted_by_itself(self):
        import retrieval_index as ri

        class EmptyCollection:
            metadata = None

            def get(self, limit=1, include=None):
                return {"embeddings": []}

        self.assertIsNone(ri.observe_index(EmptyCollection()))

    def test_collection_that_raises_is_unobservable(self):
        import retrieval_index as ri

        class BrokenCollection:
            metadata = None

            def get(self, limit=1, include=None):
                raise RuntimeError("collection unavailable")

        self.assertIsNone(ri.observe_index(BrokenCollection()))

    def test_metadata_is_preferred_over_inference(self):
        """Recorded identity is evidence; dimensionality alone is a guess."""
        import retrieval_index as ri

        class FakeCollection:
            metadata = {"embedding_model": "BAAI/bge-m3", "embedding_dimension": 1024}

            def get(self, limit=1, include=None):
                return {"embeddings": [[0.0] * 1024]}

        drift = ri.check_live_index(
            FakeCollection(), ri.EmbeddingIdentity(model="BAAI/bge-m3", dimension=1024)
        )
        self.assertFalse(drift.has_drift)


class TestConfigIdentity(unittest.TestCase):
    def test_bge_m3_is_the_default_model(self):
        # ADR 0003. Changing this default back is a regression.
        identity = load_identity_from_config({})
        self.assertEqual(identity.model, "BAAI/bge-m3")
        self.assertEqual(identity.dimension, 1024)

    def test_config_override_is_honoured(self):
        identity = load_identity_from_config(
            {"embedding": {"model": "all-MiniLM-L6-v2", "dimension": 384}}
        )
        self.assertEqual(identity.model, "all-MiniLM-L6-v2")
        self.assertEqual(identity.dimension, 384)

    def test_local_path_override_is_supported(self):
        # The weights do not live in the repository; the server needs a way to
        # point at a local copy rather than re-downloading 2.3 GB.
        identity = load_identity_from_config(
            {"embedding": {"model": "/srv/models/bge-m3"}}
        )
        self.assertEqual(identity.model, "/srv/models/bge-m3")

    def test_dimension_defaults_from_the_model_when_unset(self):
        identity = load_identity_from_config({"embedding": {"model": "all-MiniLM-L6-v2"}})
        self.assertEqual(identity.dimension, 384)


if __name__ == "__main__":
    unittest.main(verbosity=2)
