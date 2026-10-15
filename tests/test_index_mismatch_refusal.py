#!/usr/bin/env python3
"""
A mismatched embedding index refuses to answer (review-3 ticket 11).

The pipeline compared the configured embedding identity against the live index
at start-up, logged "the pipeline will refuse to answer until the corpus is
reindexed", set `retrieval_available = False`, and then never read it. Every
query searched the mismatched index and answered from it: confident text built
on vectors from the wrong geometry (ADR 0003 measured 0.00 hit@5 on every Indic
language for the old English-only index).

What these pin down, against a real ChromaDB index on disk and the real
SimpleRAGPipeline start-up (only the embedding model's weights are faked):

  * with an index built by another model, query() (every channel's entry
    point) returns the service-unavailable message in the participant's
    language, and no retrieval or answer generation runs
  * GET /health reports the mismatch
  * after reindexing with the release's model and restarting, queries answer
  * a fresh deployment (empty index) is not refused: its first upload is
    embedded with the release's model
"""

import asyncio
import hashlib
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, List
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import simple_rag_server as srs  # noqa: E402
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction  # noqa: E402
from retrieval_index import EmbeddingIdentity, write_index_metadata  # noqa: E402

RELEASE_MODEL = "BAAI/bge-m3"
RELEASE_DIM = 1024
OLD_MODEL = "BAAI/bge-small-en-v1.5"
OLD_DIM = 384

PASSAGES = [
    "Morphine is the usual strong opioid for cancer pain in palliative care.",
    "Breathlessness can be eased by sitting upright and using a fan on the face.",
    "Mouth care with soft swabs keeps a bedbound patient comfortable.",
]
GENERATED = "Morphine is commonly used for cancer pain; your doctor decides how."


class FakeSentenceTransformer:
    """Stands in for the model weights: deterministic vectors of one width."""

    def __init__(self, dimension: int):
        self.dimension = dimension

    def encode(self, sentences: Any, **_: Any) -> np.ndarray:
        texts = [sentences] if isinstance(sentences, str) else list(sentences)
        rows = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            row = np.random.default_rng(seed).random(self.dimension)
            rows.append(row / np.linalg.norm(row))
        return np.array(rows, dtype=np.float32)

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension


def seed_model(name: str, dimension: int) -> FakeSentenceTransformer:
    """Make Chroma's embedding function for `name` use fake weights."""
    model = FakeSentenceTransformer(dimension)
    SentenceTransformerEmbeddingFunction.models[name] = model
    return model


def build_index(data_dir: Path, model: str, dimension: int, record_identity: bool = True) -> None:
    """Write the 'documents' collection the way a build with `model` leaves it."""
    import chromadb

    seed_model(model, dimension)
    client = chromadb.PersistentClient(path=str(data_dir / "chroma_db"))
    collection = client.get_or_create_collection(
        name="documents",
        embedding_function=SentenceTransformerEmbeddingFunction(model_name=model),
    )
    collection.add(
        ids=[f"doc_chunk_{i}" for i in range(len(PASSAGES))],
        documents=PASSAGES,
        metadatas=[{"filename": "handbook.pdf", "chunk_index": i, "total_chunks": len(PASSAGES)}
                   for i in range(len(PASSAGES))],
    )
    if record_identity:
        write_index_metadata(collection, EmbeddingIdentity(model=model, dimension=dimension))


def start_pipeline(data_dir: Path) -> "srs.SimpleRAGPipeline":
    """Start the real pipeline over `data_dir`, as the server does at boot."""
    release_model = seed_model(RELEASE_MODEL, RELEASE_DIM)
    with mock.patch.object(srs, "load_embedding_model", return_value=release_model):
        pipeline = srs.SimpleRAGPipeline(data_dir=str(data_dir))
    pipeline.generated: List[str] = []

    async def generate(question, context, metadatas, should_fuse=False):
        pipeline.generated.append(context)
        return GENERATED, "fake-llm"

    pipeline._generate_answer_with_citations = generate
    return pipeline


class IndexMismatch(unittest.TestCase):
    def setUp(self):
        self.data_dir = Path(tempfile.mkdtemp(prefix="index-mismatch-"))
        self.addCleanup(shutil.rmtree, self.data_dir, ignore_errors=True)
        audit = tempfile.mkdtemp(prefix="index-mismatch-audit-")
        self.addCleanup(shutil.rmtree, audit, ignore_errors=True)
        env = mock.patch.dict(os.environ, {"RETRIEVAL_AUDIT_DIR": audit, "EMBEDDING_MODEL_PATH": ""})
        env.start()
        self.addCleanup(env.stop)
        # The fake weights are seeded into Chroma's class-level model cache.
        cached_models = dict(SentenceTransformerEmbeddingFunction.models)

        def restore_models():
            SentenceTransformerEmbeddingFunction.models.clear()
            SentenceTransformerEmbeddingFunction.models.update(cached_models)

        self.addCleanup(restore_models)
        self.searches: List[str] = []
        real_search = srs.vector_search

        async def counting_search(collection, embedder, text, n_results):
            self.searches.append(text)
            return await real_search(collection, embedder, text, n_results)

        patcher = mock.patch.object(srs, "vector_search", counting_search)
        patcher.start()
        self.addCleanup(patcher.stop)

    def ask(self, pipeline, question="Is morphine used for cancer pain?", language="en"):
        return asyncio.run(pipeline.query(question, source_language=language))

    def assert_refused(self, pipeline, result):
        self.assertEqual(self.searches, [], "a mismatched index must not be searched")
        self.assertEqual(pipeline.generated, [], "no answer may be drafted from a mismatched index")
        self.assertNotIn(GENERATED, result["answer"])
        self.assertEqual(result["sources"], [])
        self.assertEqual(result.get("refusal_reason"), "index_mismatch")

    # -- refusal --------------------------------------------------------------

    def test_index_built_by_another_model_refuses_every_query(self):
        build_index(self.data_dir, OLD_MODEL, OLD_DIM)
        pipeline = start_pipeline(self.data_dir)

        for question in ("Is morphine used for cancer pain?", "How do I ease breathlessness?"):
            result = self.ask(pipeline, question)
            self.assert_refused(pipeline, result)
            self.assertIn("temporarily unavailable", result["answer"])
            self.assertIn("108", result["answer"], "the refusal still points to emergency help")
        self.assertFalse(pipeline.retrieval_available)

    def test_index_built_by_a_same_width_model_is_refused(self):
        """The silent case: an English-only 1024d build answers without any error."""
        build_index(self.data_dir, "BAAI/bge-large-en-v1.5", RELEASE_DIM)
        pipeline = start_pipeline(self.data_dir)

        self.assert_refused(pipeline, self.ask(pipeline))

    def test_index_without_identity_record_is_refused_by_its_dimension(self):
        """An old English-only build recorded nothing; its 384d vectors give it away."""
        build_index(self.data_dir, OLD_MODEL, OLD_DIM, record_identity=False)
        pipeline = start_pipeline(self.data_dir)

        self.assert_refused(pipeline, self.ask(pipeline))

    def test_refusal_is_in_the_participants_supported_language(self):
        build_index(self.data_dir, OLD_MODEL, OLD_DIM)
        pipeline = start_pipeline(self.data_dir)

        result = self.ask(pipeline, "क्या कैंसर के दर्द में मॉर्फिन दी जाती है?", language="hi")
        self.assert_refused(pipeline, result)
        self.assertIn("अस्थायी रूप से उपलब्ध नहीं", result["answer"])
        self.assertNotIn("temporarily unavailable", result["answer"])

    def test_refusal_is_recorded_in_the_retrieval_record(self):
        build_index(self.data_dir, OLD_MODEL, OLD_DIM)
        pipeline = start_pipeline(self.data_dir)

        result = self.ask(pipeline)
        self.assertEqual(result["retrieval"]["retrievers_active"], [])
        self.assertEqual(result["retrieval"]["refusal_reason"], "index_mismatch")

    # -- health ---------------------------------------------------------------

    def health(self, pipeline):
        import gradio as gr
        from fastapi.testclient import TestClient

        from retrieval.settings import RetrievalSettings, set_active_settings

        # create_app sets provider env vars and module-level retrieval state.
        self.addCleanup(set_active_settings, RetrievalSettings())
        self.addCleanup(setattr, srs, "_rag_method", srs._rag_method)
        with mock.patch.object(srs, "SimpleRAGPipeline", return_value=pipeline), \
                mock.patch.object(srs.SimpleAdminUI, "create_gradio_interface", return_value=gr.Blocks()), \
                mock.patch.dict(os.environ):
            app = srs.create_app(srs._parse_args([]))
        response = TestClient(app).get("/health")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_health_reports_the_mismatch(self):
        build_index(self.data_dir, OLD_MODEL, OLD_DIM)
        body = self.health(start_pipeline(self.data_dir))

        self.assertEqual(body["status"], "degraded")
        index = body["embedding_index"]
        self.assertFalse(index["retrieval_available"])
        self.assertTrue(index["has_drift"])
        self.assertIn("dimension_mismatch", index["reasons"])
        self.assertEqual(index["expected"], {"model": RELEASE_MODEL, "dimension": RELEASE_DIM})
        self.assertEqual(index["observed"]["dimension"], OLD_DIM)
        self.assertIn("Reindex", index["remediation"])

    def test_health_reports_a_matching_index_as_healthy(self):
        build_index(self.data_dir, RELEASE_MODEL, RELEASE_DIM)
        body = self.health(start_pipeline(self.data_dir))

        self.assertEqual(body["status"], "healthy")
        self.assertTrue(body["embedding_index"]["retrieval_available"])
        self.assertFalse(body["embedding_index"]["has_drift"])

    # -- recovery -------------------------------------------------------------

    def test_after_reindexing_with_the_release_model_queries_answer_again(self):
        from reindex_service import reindex_with_identity

        build_index(self.data_dir, OLD_MODEL, OLD_DIM)
        before = start_pipeline(self.data_dir)
        self.assert_refused(before, self.ask(before))
        del before

        release_model = seed_model(RELEASE_MODEL, RELEASE_DIM)
        with mock.patch("reindex_service.load_model", return_value=release_model):
            outcome = reindex_with_identity(
                str(self.data_dir / "chroma_db"), "documents",
                EmbeddingIdentity(model=RELEASE_MODEL, dimension=RELEASE_DIM),
            )
        self.assertTrue(outcome.replaced, outcome.error)

        after = start_pipeline(self.data_dir)  # the operator restarts the server
        result = self.ask(after)
        self.assertTrue(after.retrieval_available)
        self.assertEqual(result["status"], "success")
        self.assertIn(GENERATED, result["answer"])
        self.assertNotIn("refusal_reason", result)
        self.assertTrue(self.searches, "the reindexed index is searched")
        self.assertIn("Morphine", after.generated[0])

    def test_fresh_deployment_with_an_empty_index_is_not_refused(self):
        """No index yet: what is uploaded next is embedded with the release's model."""
        pipeline = start_pipeline(self.data_dir)
        self.assertTrue(pipeline.retrieval_available)
        pipeline.vector_db.add(ids=["c0"], documents=[PASSAGES[0]],
                               metadatas=[{"filename": "handbook.pdf", "chunk_index": 0, "total_chunks": 1}])
        del pipeline

        restarted = start_pipeline(self.data_dir)
        result = self.ask(restarted)
        self.assertTrue(restarted.retrieval_available)
        self.assertNotIn("refusal_reason", result)
        self.assertIn(GENERATED, result["answer"])


if __name__ == "__main__":
    unittest.main()
