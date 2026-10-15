"""
The index is rebuilt only through the reindex service, and only with the release's model.

Round-3 review, finding A.3: `clear_and_rebuild.py` rebuilt the index with
EmbeddingGemma-300m or MiniLM, both English-only, against ADR 0003's bge-m3,
which scored 0.96 mean hit@5 across the eleven languages where the English-only
index scored 0.00 on all ten Indic ones. Seven root scripts each re-created the
index their own way. They are gone; this pins down the one that is left.

What these pin down, driving the reindex service against a real Chroma store:

  * a model other than the study release's is refused, and the index is untouched
  * a release that records no embedding model refuses every rebuild
  * the release's model rebuilds the index and records its identity on it
  * orphaned chunks (documents no longer in the metadata) are pruned, others kept
  * the reindex script refuses when config.yaml and the release disagree
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb  # noqa: E402

import reindex_service  # noqa: E402
from retrieval_index import EmbeddingIdentity, check_live_index, observe_index  # noqa: E402
from study_release import StudyRelease  # noqa: E402

COLLECTION = "documents"
RELEASE_MODEL = EmbeddingIdentity(model="BAAI/bge-m3", dimension=1024)
ENGLISH_ONLY = EmbeddingIdentity(model="all-MiniLM-L6-v2", dimension=384)


def release_with(model=None, dimensions=None) -> StudyRelease:
    embedding = {}
    if model:
        embedding["model"] = model
    if dimensions:
        embedding["dimensions"] = dimensions
    return StudyRelease(record={"release_name": "test", "models": {"embedding": embedding}})


RELEASE = release_with("BAAI/bge-m3", 1024)


class FakeSentenceTransformer:
    """Stands in for the hub download: deterministic vectors of the release's width."""

    def __init__(self, *args, **kwargs):
        self.width = 1024

    def encode(self, texts, **kwargs):
        return [[float(i % 7) / 7.0] * self.width for i, _ in enumerate(texts)]


class ChromaStore(unittest.TestCase):
    def setUp(self):
        self.path = tempfile.mkdtemp(prefix="reindex-")
        client = chromadb.PersistentClient(path=self.path)
        collection = client.create_collection(COLLECTION)
        collection.add(
            ids=["docA_chunk_0", "docA_chunk_1", "gone_chunk_0"],
            embeddings=[[0.1] * 384, [0.2] * 384, [0.3] * 384],
            documents=["Pressure sores: turn every two hours.", "Keep skin dry.", "Old text."],
            metadatas=[{"doc_id": "docA"}, {"doc_id": "docA"}, {"doc_id": "gone"}],
        )

    def collection(self):
        return chromadb.PersistentClient(path=self.path).get_collection(COLLECTION)


class AnotherModelIsRefused(ChromaStore):
    def test_a_model_other_than_the_releases_is_refused_and_the_index_is_untouched(self):
        with self.assertRaises(reindex_service.ReleaseModelMismatch) as caught:
            reindex_service.reindex_with_identity(
                chroma_path=self.path, collection_name=COLLECTION,
                identity=ENGLISH_ONLY, release=RELEASE,
            )
        self.assertIn("BAAI/bge-m3", str(caught.exception))
        self.assertEqual(self.collection().count(), 3)
        self.assertEqual(observe_index(self.collection()).dimension, 384)

    def test_the_releases_model_at_another_width_is_refused(self):
        with self.assertRaises(reindex_service.ReleaseModelMismatch):
            reindex_service.reindex_with_identity(
                chroma_path=self.path, collection_name=COLLECTION,
                identity=EmbeddingIdentity(model="BAAI/bge-m3", dimension=768),
                release=RELEASE,
            )

    def test_a_release_that_records_no_embedding_model_refuses_every_rebuild(self):
        with self.assertRaises(reindex_service.ReleaseModelMismatch):
            reindex_service.reindex_with_identity(
                chroma_path=self.path, collection_name=COLLECTION,
                identity=RELEASE_MODEL, release=release_with(),
            )
        self.assertEqual(self.collection().count(), 3)


class TheReleasesModelRebuilds(ChromaStore):
    def setUp(self):
        super().setUp()
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        cache = SentenceTransformerEmbeddingFunction.models
        saved = cache.pop(RELEASE_MODEL.model, None)

        def restore():
            cache.pop(RELEASE_MODEL.model, None)
            if saved is not None:
                cache[RELEASE_MODEL.model] = saved

        self.addCleanup(restore)
        patcher = mock.patch("sentence_transformers.SentenceTransformer", FakeSentenceTransformer)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_index_is_rebuilt_at_the_releases_width_and_records_its_model(self):
        result = reindex_service.reindex_with_identity(
            chroma_path=self.path, collection_name=COLLECTION,
            identity=RELEASE_MODEL, release=RELEASE,
        )
        self.assertIsNone(result.error)
        self.assertTrue(result.replaced)
        self.assertEqual(result.chunks_written, 3)
        drift = check_live_index(self.collection(), RELEASE_MODEL)
        self.assertEqual(drift.observed, RELEASE_MODEL)
        self.assertFalse(drift.has_drift, drift.reasons)

    def test_reindex_for_release_takes_the_model_from_the_release(self):
        result = reindex_service.reindex_for_release(
            chroma_path=self.path, collection_name=COLLECTION, release=RELEASE,
        )
        self.assertEqual(result.identity, RELEASE_MODEL)
        self.assertEqual(observe_index(self.collection()).dimension, 1024)


class OrphanedChunksArePruned(ChromaStore):
    def test_chunks_of_documents_no_longer_in_the_metadata_are_removed(self):
        removed = reindex_service.prune_orphaned_chunks(
            chroma_path=self.path, collection_name=COLLECTION, known_doc_ids={"docA"},
        )
        self.assertEqual(removed, ["gone_chunk_0"])
        self.assertEqual(sorted(self.collection().get()["ids"]), ["docA_chunk_0", "docA_chunk_1"])

    def test_nothing_is_removed_when_every_chunk_belongs_to_a_known_document(self):
        removed = reindex_service.prune_orphaned_chunks(
            chroma_path=self.path, collection_name=COLLECTION, known_doc_ids={"docA", "gone"},
        )
        self.assertEqual(removed, [])
        self.assertEqual(self.collection().count(), 3)


class TheReindexScriptRefusesADisagreement(ChromaStore):
    def run_script(self, config_model, *extra):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        import reindex_with_configured_model as script

        folder = Path(tempfile.mkdtemp(prefix="reindex-cfg-"))
        config = folder / "config.yaml"
        config.write_text(f"embedding:\n  model: \"{config_model}\"\n", encoding="utf-8")
        release = folder / "release.yaml"
        release.write_text(json.dumps({"models": {"embedding": {
            "model": "BAAI/bge-m3", "dimensions": 1024}}}), encoding="utf-8")
        argv = ["reindex", "--yes", "--chroma-path", self.path,
                "--config", str(config), "--release", str(release), *extra]
        with mock.patch.object(sys, "argv", argv):
            return script.main()

    def test_an_english_only_config_is_refused_before_the_index_is_touched(self):
        self.assertEqual(self.run_script("all-MiniLM-L6-v2"), 1)
        self.assertEqual(self.collection().count(), 3)
        self.assertEqual(observe_index(self.collection()).dimension, 384)

    def test_prune_orphans_removes_chunks_missing_from_the_metadata(self):
        metadata = Path(tempfile.mkdtemp(prefix="reindex-meta-")) / "document_metadata.json"
        metadata.write_text(json.dumps({"docA": {"filename": "a.pdf"}}), encoding="utf-8")
        code = self.run_script("BAAI/bge-m3", "--prune-orphans", "--metadata", str(metadata))
        self.assertEqual(code, 0)
        self.assertEqual(sorted(self.collection().get()["ids"]), ["docA_chunk_0", "docA_chunk_1"])


if __name__ == "__main__":
    unittest.main()
