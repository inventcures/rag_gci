"""
A question is embedded once per turn, off the event loop, by the one model.

Before this, Chroma embedded `query_texts` synchronously on the event loop, a
non-English question was embedded twice (probe, then search), the server
loaded bge-m3 twice (once by name from the hub even when a local path was
configured), and hybrid mode's "parallel" vector search blocked the loop so it
ran before the tree search, not alongside it.

What these pin down:

  * a non-English turn writes one embed span, and one vector search
  * a repeated question is served from the exact-text cache
  * the event loop stays responsive (< 50 ms) while a slow embedding runs
  * translation still triggers on exactly the probe's threshold; when it does,
    the translated text is embedded and searched
  * Chroma's embedding function reuses the loaded model: no second load
  * hybrid mode's vector and tree searches overlap
"""

import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.embedder import QueryEmbedder, share_loaded_model  # noqa: E402
from tests.test_answer_path_retrieval import VECTOR_DOCS, make_pipeline  # noqa: E402


class EmbeddingCollection:
    """A collection that only accepts query_embeddings, as the answer path now uses."""

    def __init__(self, docs, delay=0.0, top_distance=None):
        self.docs = docs
        self.delay = delay
        self.top_distance = top_distance
        self.calls = []

    def query(self, query_embeddings, n_results):
        self.calls.append({"vector": query_embeddings[0], "n_results": n_results})
        if self.delay:
            time.sleep(self.delay)
        docs = self.docs[:n_results]
        distances = [d[3] for d in docs]
        if self.top_distance is not None and distances:
            distances[0] = self.top_distance
        return {
            "documents": [[d[0] for d in docs]],
            "metadatas": [[{"filename": d[1], "chunk_index": d[2], "total_chunks": 10} for d in docs]],
            "distances": [distances],
        }


class CountingModel:
    def __init__(self, delay=0.0):
        self.delay = delay
        self.texts = []
        self.lock = threading.Lock()

    def __call__(self, texts):
        with self.lock:
            self.texts.extend(texts)
        if self.delay:
            time.sleep(self.delay)
        return [[float(len(t)), 1.0, 0.5] for t in texts]


class TraceCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="traces-")
        self.env = patch.dict(os.environ, {"TRACE_DIR": self.dir, "TRACING_ENABLED": "1",
                                           "RETRIEVAL_AUDIT_DIR": tempfile.mkdtemp()})
        self.env.start()
        self.addCleanup(self.env.stop)

    def spans(self):
        out = []
        for path in Path(self.dir).glob("*.jsonl"):
            out += [json.loads(line) for line in path.read_text().splitlines()]
        return out

    def pipeline(self, model, collection):
        p = make_pipeline(answer="Sit upright and use a fan.")
        p.vector_db = collection
        p.query_embedder = QueryEmbedder(model)
        return p


QUESTION_HI = "मेरे पिता को रात में सांस फूलती है"


class EmbedOnce(TraceCase):
    def test_non_english_turn_embeds_and_searches_once(self):
        model, collection = CountingModel(), EmbeddingCollection(VECTOR_DOCS)
        p = self.pipeline(model, collection)
        result = asyncio.run(p.query(question=QUESTION_HI, source_language="hi"))
        self.assertEqual(result["status"], "success")
        self.assertEqual(model.texts, [QUESTION_HI])
        self.assertEqual(len(collection.calls), 1)
        names = [s["name"] for s in self.spans()]
        self.assertEqual(names.count("embed"), 1)
        search = next(s for s in self.spans() if s["name"] == "search")
        self.assertTrue(search["attrs"].get("reused_probe"))

    def test_a_repeated_question_is_not_re_embedded(self):
        model, collection = CountingModel(), EmbeddingCollection(VECTOR_DOCS)
        p = self.pipeline(model, collection)
        for _ in range(2):
            asyncio.run(p.query(question="How do I ease breathlessness?", source_language="en"))
        self.assertEqual(model.texts, ["How do I ease breathlessness?"])
        self.assertEqual(p.query_embedder.hits, 1)

    def test_weak_probe_translates_and_searches_the_translation(self):
        model = CountingModel()
        collection = EmbeddingCollection(VECTOR_DOCS, top_distance=1.9)
        p = self.pipeline(model, collection)

        async def translate(question, source_language):
            return {"status": "success", "translated_query": "breathless at night"}

        p.translate_query_to_english = translate
        asyncio.run(p.query(question=QUESTION_HI, source_language="hi"))
        self.assertEqual(model.texts, [QUESTION_HI, "breathless at night"])
        self.assertEqual(len(collection.calls), 2)
        self.assertEqual(p.translation_fallback_count, 1)

    def test_probe_threshold_is_unchanged(self):
        for distance, translates in ((1.5, False), (1.51, True), (0.4, False)):
            collection = EmbeddingCollection(VECTOR_DOCS, top_distance=distance)
            p = self.pipeline(CountingModel(), collection)
            weak, _ = asyncio.run(p._cross_language_probe(QUESTION_HI, 5))
            self.assertEqual(weak, translates, distance)
        self.assertEqual(type(p).CROSS_LANGUAGE_FALLBACK_DISTANCE, 1.5)


class EventLoopStaysFree(TraceCase):
    def test_keepalive_answers_while_a_slow_embedding_runs(self):
        model, collection = CountingModel(delay=0.4), EmbeddingCollection(VECTOR_DOCS)
        p = self.pipeline(model, collection)

        async def go():
            answer = asyncio.create_task(p.query(question="pain at night", source_language="en"))
            await asyncio.sleep(0.05)
            worst = 0.0
            while not answer.done():
                sent = time.perf_counter()
                await asyncio.sleep(0)  # a keepalive frame being handled
                worst = max(worst, time.perf_counter() - sent)
                await asyncio.sleep(0.01)
            await answer
            return worst

        self.assertLess(asyncio.run(go()), 0.05)


class OneModel(unittest.TestCase):
    def test_chroma_embedding_function_reuses_the_loaded_model(self):
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        loaded = object()
        name = "test/shared-model-for-embed-once"
        self.addCleanup(SentenceTransformerEmbeddingFunction.models.pop, name, None)
        share_loaded_model(name, loaded)
        with patch("sentence_transformers.SentenceTransformer",
                   side_effect=AssertionError("second model load")):
            ef = SentenceTransformerEmbeddingFunction(model_name=name)
        self.assertIs(ef._model, loaded)
        self.assertEqual(ef.get_config()["model_name"], name)

    def test_the_pipeline_seeds_the_cache_before_building_the_function(self):
        source = Path(__file__).resolve().parents[1].joinpath("simple_rag_server.py").read_text(encoding="utf-8")
        init = source[source.index("def _initialize_components"):source.index("def _initialize_longitudinal_memory")]
        self.assertLess(init.index("share_loaded_model("), init.index("SentenceTransformerEmbeddingFunction("))
        self.assertIn("QueryEmbedder(embedding_func)", init)


class SameResults(unittest.TestCase):
    """Searching by the embedder's vector returns exactly what query_texts did."""

    def test_query_embeddings_match_query_texts_on_a_real_collection(self):
        import chromadb
        from chromadb.api.types import EmbeddingFunction

        class Hashing(EmbeddingFunction):
            def __init__(self):
                pass

            def __call__(self, input):
                import numpy as np

                out = []
                for text in input:
                    v = np.zeros(32, dtype=np.float32)
                    for i, ch in enumerate(text):
                        v[(ord(ch) * 31 + i) % 32] += 1.0
                    out.append(v)
                return out

            @staticmethod
            def name():
                return "hashing-test"

            def get_config(self):
                return {}

            @staticmethod
            def build_from_config(config):
                return Hashing()

        ef = Hashing()
        client = chromadb.PersistentClient(path=tempfile.mkdtemp(prefix="embed-once-"))
        collection = client.get_or_create_collection("docs", embedding_function=ef)
        texts = [d[0] for d in VECTOR_DOCS] + ["सांस फूलना", "ব্যথা", "pain relief at home"]
        collection.add(ids=[str(i) for i in range(len(texts))], documents=texts)
        embedder = QueryEmbedder(ef)
        for question in ("breathless at night", QUESTION_HI, "ব্যথা কমানো"):
            by_text = collection.query(query_texts=[question], n_results=5)
            by_vector = asyncio.run(embedder.search(collection, question, 5))
            self.assertEqual(by_text["ids"], by_vector["ids"])
            self.assertEqual(by_text["distances"], by_vector["distances"])


class HybridOverlaps(TraceCase):
    def test_vector_and_tree_searches_overlap(self):
        import simple_rag_server

        class SlowTree:
            async def search(self, query):
                await asyncio.sleep(0.2)
                return type("R", (), {"context": "Tree text.", "selected_nodes": [
                    {"filename": "t.pdf", "start_page": 1, "end_page": 2}]})()

        p = self.pipeline(CountingModel(), EmbeddingCollection(VECTOR_DOCS, delay=0.2))
        with patch.object(simple_rag_server, "_rag_method", "hybrid"), \
                patch.object(simple_rag_server, "_pageindex_enabled", True), \
                patch.object(simple_rag_server, "_pageindex_engine", SlowTree()):
            asyncio.run(p.query(question="pain at night", source_language="en"))

        def interval(name):
            s = next(x for x in self.spans() if x["name"] == name)
            end = datetime.fromisoformat(s["ts"]).timestamp()
            return end - s["ms"] / 1000, end

        v0, v1 = interval("search")
        t0, t1 = interval("pageindex.search")
        self.assertLess(max(v0, t0), min(v1, t1) - 0.1, "searches ran one after the other")


if __name__ == "__main__":
    unittest.main()
