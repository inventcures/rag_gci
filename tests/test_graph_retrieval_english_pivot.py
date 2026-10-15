#!/usr/bin/env python3
"""
Graph retrieval for questions in a Supported Language other than English.

The knowledge graph and the GraphRAG index match English names
("morphine", "breathlessness"). Before ticket 18 the graph retrievers were
handed the question exactly as the vector path searched it, so a Marathi
question found nothing in the kg, graphrag and fused modes, and nothing said
why.

What these tests pin down:

  * the graph retrievers receive the English Pivot (the Query Translation of
    the question), and a Marathi question whose pivot names a KG entity gets
    graph passages;
  * the vector path keeps ADR 0003's probe-first rule: it searches the
    question as written and Query Translation stays a weak-evidence fallback
    there;
  * an empty graph result records why it was empty.

The seams are the answer path's own: SimpleRAGPipeline.query() with a fake
collection, the in-memory graph store, and a stub Query Translation.
"""

import asyncio
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.graph_store import Fact, InMemoryGraphStore  # noqa: E402
from retrieval.router import (  # noqa: E402
    GraphRAGRetriever,
    KnowledgeGraphRetriever,
    RetrievalRouter,
    VectorRetriever,
)
from retrieval.settings import RetrievalSettings, set_active_settings  # noqa: E402

MARATHI_QUESTION = "श्वास घ्यायला त्रास होत असेल तर मॉर्फिन वापरता येईल का?"
ENGLISH_PIVOT = "Can morphine be used for breathlessness?"

FACTS = [
    Fact("Morphine", "Medication", "TREATS", "Dyspnea", "Symptom", {"evidence": "Cochrane Review"}),
    Fact("Constipation", "SideEffect", "SIDE_EFFECT_OF", "Morphine", "Medication", {"frequency": "common"}),
]

# Close enough that the probe does not ask for a translation (ADR 0003).
STRONG_DOCS = [("Breathlessness can be eased with a fan and positioning.", "homecare.pdf", 1, 0.42)]


def run(coro):
    return asyncio.run(coro)


class RecordingCollection:
    """A Chroma-shaped collection that remembers what text it was asked for."""

    def __init__(self, docs):
        self.docs = docs
        self.searched = []

    def query(self, query_texts, n_results):
        self.searched.extend(query_texts)
        docs = self.docs[:n_results]
        return {
            "documents": [[d[0] for d in docs]],
            "metadatas": [[{"filename": d[1], "chunk_index": d[2], "total_chunks": 10} for d in docs]],
            "distances": [[d[3] for d in docs]],
        }

    def count(self):
        return len(self.docs)


class RecordingGraphRAGLoader:
    """The GraphRAG index's English entity search."""

    def __init__(self):
        self.searched = []

    async def search_entities(self, query, top_k=10):
        self.searched.append(query)
        if "morphine" in query.lower():
            return [{"title": "MORPHINE", "description": "Strong opioid used for breathlessness.",
                     "type": "Medication"}]
        return []

    async def get_entity_relationships(self, name):
        return []


class FakeGraphRAGEngine:
    def __init__(self):
        self.data_loader = RecordingGraphRAGLoader()


def make_pipeline(docs=STRONG_DOCS):
    from simple_rag_server import SimpleRAGPipeline

    p = SimpleRAGPipeline.__new__(SimpleRAGPipeline)
    p.vector_db = RecordingCollection(docs)
    p.document_metadata = {}
    p.conversations = {}
    p.context_injector = None
    p.cross_modal_aggregator = None
    p.translation_fallback_count = 0
    p.retrieval_router = None
    p.kg_checker = None
    p.graph_store = None
    p.generated = []
    p.translated = []

    async def generate(question, context, metadatas, should_fuse=False):
        p.generated.append({"context": context, "metadatas": metadatas})
        return "Morphine can ease breathlessness; ask your doctor.", "fake-llm"

    async def translate(query, source_language):
        p.translated.append((query, source_language))
        return {"status": "success", "translated_query": ENGLISH_PIVOT, "original_query": query}

    p._generate_answer_with_citations = generate
    p.translate_query_to_english = translate
    return p


class GraphModesUseTheEnglishPivot(unittest.TestCase):
    def setUp(self):
        self.audit_dir = tempfile.mkdtemp(prefix="pivot-audit-")
        self.env = patch.dict(os.environ, {"RETRIEVAL_AUDIT_DIR": self.audit_dir})
        self.env.start()
        self.engine = FakeGraphRAGEngine()

    def tearDown(self):
        self.env.stop()
        set_active_settings(RetrievalSettings())
        import simple_rag_server

        simple_rag_server._rag_method = "vector"

    def configure(self, pipeline, method):
        import simple_rag_server

        s = set_active_settings(RetrievalSettings(rag_method=method))
        simple_rag_server._rag_method = method
        pipeline.configure_answer_path(
            s,
            graph_store=InMemoryGraphStore(FACTS, version="kg-test-1"),
            graphrag_engine=self.engine,
        )

    def test_a_marathi_question_whose_english_pivot_hits_a_kg_entity_returns_graph_passages(self):
        p = make_pipeline()
        self.configure(p, "kg")
        result = run(p.query(MARATHI_QUESTION, source_language="mr"))
        self.assertEqual(result["status"], "success")
        self.assertIn("kg", result["retrieval"]["contributing"])
        self.assertIn("Clinical knowledge graph", p.generated[0]["context"])
        self.assertIn("Morphine is used to treat Dyspnea", p.generated[0]["context"])
        self.assertEqual(result["retrieval"]["graph_query"],
                         {"source_language": "mr", "pivot": "english"})

    def test_a_marathi_question_reaches_the_graphrag_index_in_english(self):
        for method in ("graphrag", "fused"):
            with self.subTest(method=method):
                self.engine = FakeGraphRAGEngine()
                p = make_pipeline()
                self.configure(p, method)
                result = run(p.query(MARATHI_QUESTION, source_language="mr"))
                self.assertEqual(self.engine.data_loader.searched, [ENGLISH_PIVOT])
                self.assertIn("graphrag", result["retrieval"]["contributing"])

    def test_the_vector_path_still_searches_the_question_as_written(self):
        """ADR 0003: Cross-Language Retrieval first, for the vector path."""
        p = make_pipeline()
        self.configure(p, "kg")
        run(p.query(MARATHI_QUESTION, source_language="mr"))
        self.assertTrue(p.vector_db.searched)
        self.assertEqual(set(p.vector_db.searched), {MARATHI_QUESTION})

    def test_the_vector_path_translates_only_when_evidence_is_weak(self):
        """The default (vector) mode makes no Query Translation on strong evidence."""
        p = make_pipeline()
        self.configure(p, "vector")
        run(p.query(MARATHI_QUESTION, source_language="mr"))
        self.assertEqual(p.translated, [])

    def test_weak_evidence_reuses_one_query_translation_for_both_paths(self):
        p = make_pipeline(docs=[("Unrelated chunk.", "x.pdf", 0, 1.9)])
        self.configure(p, "kg")
        result = run(p.query(MARATHI_QUESTION, source_language="mr"))
        self.assertEqual(len(p.translated), 1, "one Query Translation, not two")
        self.assertIn(ENGLISH_PIVOT, p.vector_db.searched)
        self.assertIn("kg", result["retrieval"]["contributing"])

    def test_an_english_question_is_not_translated(self):
        p = make_pipeline()
        self.configure(p, "kg")
        result = run(p.query("Can morphine be used for breathlessness?", source_language="en"))
        self.assertEqual(p.translated, [])
        self.assertIn("kg", result["retrieval"]["contributing"])
        self.assertEqual(result["retrieval"]["graph_query"],
                         {"source_language": "en", "pivot": "not_needed"})

    def test_a_failed_pivot_records_why_the_graph_was_empty(self):
        p = make_pipeline()

        async def failing(query, source_language):
            p.translated.append((query, source_language))
            return {"status": "error", "error": "timeout"}

        p.translate_query_to_english = failing
        self.configure(p, "kg")
        result = run(p.query(MARATHI_QUESTION, source_language="mr"))
        retrieval = result["retrieval"]
        self.assertEqual(retrieval["graph_query"], {"source_language": "mr", "pivot": "failed"})
        self.assertIn("kg", retrieval["retrievers_empty"])
        self.assertIn("English pivot unavailable", retrieval["retrievers_empty"]["kg"])
        # The vector path still answered.
        self.assertEqual(result["status"], "success")
        self.assertNotIn("kg", retrieval["contributing"])


class EmptyGraphResultsSayWhy(unittest.TestCase):
    """RetrievalRouter records a reason for every empty retriever."""

    def router(self, facts=FACTS):
        r = RetrievalRouter(RetrievalSettings(rag_method="fused"))
        r.attach("vector", VectorRetriever(RecordingCollection(STRONG_DOCS)))
        r.attach("kg", KnowledgeGraphRetriever(InMemoryGraphStore(facts)))
        r.attach("graphrag", GraphRAGRetriever(FakeGraphRAGEngine()))
        return r

    def test_no_clinical_entity_in_the_question(self):
        routed = run(self.router().retrieve("How do I sit with my mother tonight?", 5, method="kg"))
        empty = routed.record()["retrievers_empty"]
        self.assertIn("no clinical entity", empty["kg"])

    def test_an_entity_with_no_facts_in_the_graph(self):
        routed = run(self.router(facts=[]).retrieve("Is senna safe?", 5, method="kg"))
        self.assertIn("no facts", routed.record()["retrievers_empty"]["kg"])
        self.assertIn("Senna", routed.record()["retrievers_empty"]["kg"])

    def test_no_graphrag_entity_matched(self):
        routed = run(self.router().retrieve("How do I sit with my mother tonight?", 5, method="graphrag"))
        self.assertIn("no GraphRAG entity", routed.record()["retrievers_empty"]["graphrag"])

    def test_graph_retrievers_get_the_graph_query_and_vector_the_question(self):
        r = self.router()
        routed = run(r.retrieve(MARATHI_QUESTION, 5, method="kg", graph_query=ENGLISH_PIVOT))
        self.assertIn("kg", routed.record()["contributing"])
        self.assertEqual(set(r.retrievers["vector"].vector_db.searched), {MARATHI_QUESTION})

    def test_a_non_empty_result_records_no_reason(self):
        routed = run(self.router().retrieve(ENGLISH_PIVOT, 5, method="kg"))
        self.assertNotIn("kg", routed.record()["retrievers_empty"])


if __name__ == "__main__":
    unittest.main()
