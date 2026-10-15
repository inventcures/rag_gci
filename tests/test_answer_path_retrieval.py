#!/usr/bin/env python3
"""
Knowledge graph and GraphRAG on the answer path.

What these tests pin down:

  * one shared Reciprocal Rank Fusion function, with the arithmetic checked by hand
  * every retrieval mode (vector, kg, graphrag, fused) through the same query()
  * the K26 knowledge-graph cross-reference check, including a contradicted claim
  * graceful fallback when Neo4j is down or slow: the user still gets an answer
  * voice (Gemini Live tool and the Sarvam turn handler) runs the configured path
  * the dose boundary still holds with graph context in the prompt
  * the per-interaction release record names the retrievers, fusion and KG version

The default (vector, check off) must behave exactly as before; several tests
assert that nothing new runs in that configuration.
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.fusion import reciprocal_rank_fusion  # noqa: E402
from retrieval.graph_store import Fact, InMemoryGraphStore  # noqa: E402
from retrieval.kg_check import (  # noqa: E402
    SAFE_FALLBACK,
    ClaimExtractor,
    KGCrossReferenceChecker,
)
from retrieval.router import (  # noqa: E402
    GraphRAGRetriever,
    KnowledgeGraphRetriever,
    PageIndexRetriever,
    RetrievalRouter,
    VectorRetriever,
)
from retrieval.settings import (  # noqa: E402
    RetrievalSettings,
    load_retrieval_settings,
    set_active_settings,
)

# A small slice of the base palliative-care graph (graph_builder.py).
BASE_FACTS = [
    Fact("Morphine", "Medication", "TREATS", "Pain", "Symptom", {"evidence": "WHO Pain Ladder Step 3"}),
    Fact("Morphine", "Medication", "TREATS", "Dyspnea", "Symptom", {"evidence": "Cochrane Review"}),
    Fact("Constipation", "SideEffect", "SIDE_EFFECT_OF", "Morphine", "Medication", {"frequency": "common"}),
    Fact("Drowsiness", "SideEffect", "SIDE_EFFECT_OF", "Morphine", "Medication", {"frequency": "common"}),
    Fact("Senna", "Medication", "TREATS", "Constipation", "Symptom", {"evidence": "Palliative Care Formulary"}),
    Fact("Lactulose", "Medication", "TREATS", "Constipation", "Symptom"),
    Fact("Kidney Failure", "Condition", "CONTRAINDICATES", "Ibuprofen", "Medication"),
]


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeCollection:
    """Shaped like a Chroma collection's query()."""

    def __init__(self, docs, fail=False):
        self.docs = docs  # list of (text, filename, chunk_index, distance)
        self.fail = fail
        self.calls = 0

    def query(self, query_texts, n_results):
        self.calls += 1
        if self.fail:
            raise RuntimeError("chroma unavailable")
        docs = self.docs[:n_results]
        return {
            "documents": [[d[0] for d in docs]],
            "metadatas": [[{"filename": d[1], "chunk_index": d[2], "total_chunks": 10} for d in docs]],
            "distances": [[d[3] for d in docs]],
        }

    def count(self):
        return len(self.docs)


VECTOR_DOCS = [
    ("Morphine is the first-line strong opioid for cancer pain.", "handbook.pdf", 3, 0.42),
    ("Constipation is common with opioids; start a laxative.", "handbook.pdf", 7, 0.55),
    ("Breathlessness can be eased with a fan and positioning.", "homecare.pdf", 1, 0.61),
]


class FakePageIndex:
    async def search(self, query):
        return type("R", (), {"selected_nodes": [
            {"filename": "opioids.pdf", "start_page": 4, "end_page": 5,
             "text": "Opioid titration is done by the treating doctor."},
        ], "context": ""})()


class FakeGraphRAGLoader:
    async def search_entities(self, query, top_k=10):
        return [{"title": "MORPHINE", "description": "Strong opioid used for pain and breathlessness.", "type": "Medication"}]

    async def get_entity_relationships(self, name):
        return [{"source": "MORPHINE", "target": "CONSTIPATION",
                 "description": "Morphine commonly causes constipation."}]


class FakeGraphRAGEngine:
    data_loader = FakeGraphRAGLoader()


def settings(**kw):
    return RetrievalSettings(**kw)


def router_for(method, graph=None, vector_fail=False, **kw):
    r = RetrievalRouter(settings(rag_method=method, **kw))
    r.attach("vector", VectorRetriever(FakeCollection(VECTOR_DOCS, fail=vector_fail)))
    r.attach("kg", KnowledgeGraphRetriever(graph if graph is not None else InMemoryGraphStore(BASE_FACTS)))
    r.attach("graphrag", GraphRAGRetriever(FakeGraphRAGEngine()))
    r.attach("pageindex", PageIndexRetriever(FakePageIndex()))
    return r


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


class ReciprocalRankFusion(unittest.TestCase):
    def test_worked_example_from_the_module_docstring(self):
        fused = reciprocal_rank_fusion([["A", "B", "C"], ["C", "D"]], key=lambda x: x, k=60)
        scores = {f.item: round(f.score, 5) for f in fused}
        self.assertEqual(scores["C"], round(1 / 63 + 1 / 61, 5))
        self.assertEqual(scores["A"], round(1 / 61, 5))
        self.assertEqual([f.item for f in fused][0], "C", "an item in both lists must win")

    def test_ties_keep_first_seen_order(self):
        fused = reciprocal_rank_fusion([["A", "B"], ["X", "Y"]], key=lambda x: x)
        self.assertEqual([f.item for f in fused], ["A", "X", "B", "Y"])

    def test_duplicate_within_one_list_counts_once(self):
        fused = reciprocal_rank_fusion([["A", "A", "B"]], key=lambda x: x, k=0)
        self.assertEqual(fused[0].score, 1.0)
        self.assertEqual(fused[1].ranks, {"0": 3})

    def test_weights_and_names(self):
        fused = reciprocal_rank_fusion(
            [["A"], ["B"]], key=lambda x: x, k=0, weights=[1.0, 2.0], names=["vector", "kg"]
        )
        self.assertEqual(fused[0].item, "B")
        self.assertEqual(fused[0].ranks, {"kg": 1})

    def test_rejects_mismatched_weights(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([["A"]], key=lambda x: x, weights=[1.0, 2.0])

    def test_context1_uses_the_shared_function(self):
        from context1_integration import tools

        a = tools.ChunkResult("c1", "t", {}, 0.9, "vector")
        b = tools.ChunkResult("c2", "t", {}, 0.8, "vector")
        c = tools.ChunkResult("c2", "t", {}, 0.1, "keyword")
        out = tools._reciprocal_rank_fusion([[a, b], [c]])
        self.assertEqual([x.chunk_id for x in out], ["c2", "c1"])
        self.assertAlmostEqual(out[0].score, 1 / 62 + 1 / 61)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class Settings(unittest.TestCase):
    def test_defaults_reproduce_the_study_release(self):
        s = load_retrieval_settings(env={})
        self.assertEqual(s.rag_method, "vector")
        self.assertEqual(s.kg_check_mode, "off")
        self.assertFalse(s.is_routed)

    def test_cli_beats_env_beats_config(self):
        s = load_retrieval_settings(env={"RAG_METHOD": "fused"}, cli_rag_method="kg")
        self.assertEqual((s.rag_method, s.source["rag_method"]), ("kg", "cli"))
        s = load_retrieval_settings(env={"RAG_METHOD": "graphrag", "KG_CHECK_MODE": "flag"})
        self.assertEqual((s.rag_method, s.kg_check_mode), ("graphrag", "flag"))

    def test_unknown_values_are_refused(self):
        with self.assertRaises(ValueError):
            RetrievalSettings(rag_method="graph")
        with self.assertRaises(ValueError):
            RetrievalSettings(kg_check_mode="strict")

    def test_yaml_bare_off_reads_as_off(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            fh.write("retrieval:\n  kg_check:\n    mode: off\n")
        from pathlib import Path

        self.assertEqual(load_retrieval_settings(Path(fh.name), env={}).kg_check_mode, "off")

    def test_fusion_description_for_the_log(self):
        self.assertEqual(settings(rag_method="fused").fusion_description()["method"], "rrf")
        self.assertEqual(settings().fusion_description(), {"method": "none"})


# ---------------------------------------------------------------------------
# Retrieval modes through the router
# ---------------------------------------------------------------------------


class RoutedModes(unittest.TestCase):
    def test_kg_mode_fuses_vector_and_graph(self):
        out = run(router_for("kg").retrieve("Is morphine safe for breathlessness?", top_k=5))
        self.assertEqual(out.record()["retrievers_active"], ["vector", "kg"])
        self.assertIn("kg", {p.retriever for p in out.passages})
        kg_text = next(p.text for p in out.passages if p.retriever == "kg")
        # The edge linking the two asked-about entities leads the passage.
        self.assertTrue(kg_text.startswith("Morphine is used to treat Dyspnea"))
        self.assertIn("Clinical knowledge graph", out.context_text)
        self.assertEqual(out.record()["fusion"]["method"], "rrf")

    def test_graphrag_mode(self):
        out = run(router_for("graphrag").retrieve("morphine for pain", top_k=5))
        self.assertEqual(out.record()["retrievers_active"], ["vector", "graphrag"])
        self.assertIn("GraphRAG index", out.context_text)

    def test_fused_mode_runs_all_four(self):
        out = run(router_for("fused").retrieve("morphine for pain", top_k=10))
        self.assertEqual(out.record()["retrievers_active"], ["vector", "pageindex", "kg", "graphrag"])
        self.assertEqual(out.record()["contributing"], ["graphrag", "kg", "pageindex", "vector"])
        for p in out.passages:
            self.assertIn("rrf_score", p.metadata)

    def test_top_k_is_respected_after_fusion(self):
        out = run(router_for("fused").retrieve("morphine for pain", top_k=2))
        self.assertEqual(len(out.passages), 2)

    def test_only_vector_distances_feed_the_evidence_badge(self):
        out = run(router_for("kg").retrieve("morphine for pain", top_k=10))
        self.assertEqual(len(out.distances), len([p for p in out.passages if p.retriever == "vector"]))

    def test_vector_relevance_threshold_still_applies(self):
        r = RetrievalRouter(settings(rag_method="kg"))
        r.attach("vector", VectorRetriever(FakeCollection([("far", "x.pdf", 0, 1.9)])))
        out = run(r.retrieve("q", top_k=5))
        self.assertEqual(out.passages, [])

    def test_neo4j_down_degrades_to_vector(self):
        down = InMemoryGraphStore(BASE_FACTS, available=False)
        out = run(router_for("kg", graph=down).retrieve("morphine for pain", top_k=5))
        self.assertEqual(out.record()["retrievers_failed"], {"kg": "unavailable"})
        self.assertEqual({p.retriever for p in out.passages}, {"vector"})

    def test_slow_graph_is_cut_off_by_its_timeout(self):
        slow = InMemoryGraphStore(BASE_FACTS, delay_s=0.5)
        out = run(router_for("kg", graph=slow, kg_timeout_s=0.05).retrieve("morphine pain", top_k=5))
        self.assertEqual(out.record()["retrievers_failed"], {"kg": "timeout"})
        self.assertTrue(out.passages)

    def test_missing_retriever_is_recorded_not_raised(self):
        r = RetrievalRouter(settings(rag_method="graphrag"))
        r.attach("vector", VectorRetriever(FakeCollection(VECTOR_DOCS)))
        out = run(r.retrieve("q", top_k=5))
        self.assertEqual(out.record()["retrievers_failed"], {"graphrag": "not_configured"})

    def test_vector_failure_hands_back_to_the_default_path(self):
        self.assertIsNone(run(router_for("kg", vector_fail=True).retrieve("q", top_k=5)))

    def test_graph_dose_facts_never_reach_the_context(self):
        dosed = BASE_FACTS + [
            Fact("Morphine", "Medication", "TREATS", "Cancer Pain", "Symptom",
                 {"effectiveness": "give 10 mg every 4 hours"}),
        ]
        out = run(router_for("kg", graph=InMemoryGraphStore(dosed)).retrieve("morphine for cancer pain", top_k=5))
        self.assertNotIn("10 mg", out.context_text)
        self.assertIn("Morphine", out.context_text)


# ---------------------------------------------------------------------------
# K26 cross-reference check
# ---------------------------------------------------------------------------


CONTRADICTED_ANSWER = (
    "Morphine relieves cancer pain well. Morphine also relieves constipation. "
    "Senna is used to treat constipation."
)


class ClaimExtraction(unittest.TestCase):
    def setUp(self):
        self.ex = ClaimExtractor()

    def describe(self, text):
        return [c.describe() for c in self.ex.extract(text)]

    def test_relations_are_read_between_the_two_mentions(self):
        self.assertEqual(
            self.describe("Senna should be given to treat constipation caused by morphine."),
            ["Senna treats Constipation", "Morphine causes Constipation"],
        )

    def test_a_cue_elsewhere_in_the_sentence_does_not_make_a_claim(self):
        # "cause" belongs to morphine; senna must not inherit it.
        self.assertEqual(
            self.describe("Morphine can cause constipation, so a laxative such as senna should be given."),
            ["Morphine causes Constipation"],
        )

    def test_induced_modifier_belongs_to_the_target(self):
        self.assertEqual(self.describe("Lactulose prevents opioid-induced constipation."),
                         ["Lactulose treats Constipation"])

    def test_negation_and_contraindication(self):
        self.assertEqual(self.describe("Morphine does not cause constipation."),
                         ["Morphine NOT causes Constipation"])
        self.assertEqual(self.describe("Ibuprofen should be avoided in kidney failure."),
                         ["Ibuprofen is contraindicated in Kidney Failure"])

    def test_specific_mention_beats_generic(self):
        self.assertEqual(self.describe("Gabapentin helps neuropathic pain."),
                         ["Gabapentin treats Neuropathic Pain"])


class CrossReferenceCheck(unittest.TestCase):
    def check(self, answer, mode="block", graph=None, **kw):
        checker = KGCrossReferenceChecker(
            graph if graph is not None else InMemoryGraphStore(BASE_FACTS, version="kg-test-1"),
            mode=mode, **kw,
        )
        return run(checker.check(answer))

    def test_contradicted_claim_is_found_and_blocked(self):
        answer, result = self.check(CONTRADICTED_ANSWER)
        verdicts = {c.describe(): c.verdict for c in result.claims}
        self.assertEqual(verdicts["Morphine treats Constipation"], "contradicted")
        self.assertEqual(verdicts["Senna treats Constipation"], "supported")
        self.assertEqual(result.action, "blocked")
        self.assertNotIn("also relieves constipation", answer)
        self.assertIn("Senna is used to treat constipation.", answer)
        self.assertIn("conflicted with the clinical knowledge base", answer)
        self.assertEqual(result.kg_version, "kg-test-1")

    def test_negated_claim_against_a_recorded_side_effect_is_contradicted(self):
        _, result = self.check("Morphine does not cause constipation.", mode="shadow")
        self.assertEqual(result.claims[0].verdict, "contradicted")

    def test_contraindication_contradicts_a_treatment_claim(self):
        _, result = self.check("Ibuprofen is used for kidney failure.", mode="shadow")
        self.assertEqual(result.claims[0].verdict, "contradicted")

    def test_unknown_medication_is_unsupported(self):
        _, result = self.check("Tramadol helps with nausea.", mode="shadow")
        self.assertEqual((result.claims[0].verdict, result.claims[0].reason),
                         ("unsupported", "medication not in graph"))

    def test_shadow_mode_never_changes_the_answer(self):
        answer, result = self.check(CONTRADICTED_ANSWER, mode="shadow")
        self.assertEqual(answer, CONTRADICTED_ANSWER)
        self.assertEqual((result.action, result.contradicted), ("logged", 1))

    def test_flag_mode_appends_a_caution(self):
        answer, result = self.check(CONTRADICTED_ANSWER, mode="flag")
        self.assertTrue(answer.startswith(CONTRADICTED_ANSWER))
        self.assertIn("conflicts with the clinical knowledge base", answer)
        self.assertEqual(result.action, "flagged")

    def test_block_mode_flags_unsupported_unless_told_to_remove_it(self):
        answer, result = self.check("Morphine relieves cancer pain well. Tramadol helps with nausea.")
        self.assertEqual(result.action, "flagged")
        self.assertIn("Tramadol", answer)
        answer, result = self.check(
            "Morphine relieves cancer pain well and is widely used. Tramadol helps with nausea.",
            block_unsupported=True,
        )
        self.assertNotIn("Tramadol", answer)

    def test_removing_everything_gives_the_safe_fallback(self):
        answer, _ = self.check("Morphine relieves constipation.")
        self.assertEqual(answer, SAFE_FALLBACK)

    def test_supported_answer_is_untouched(self):
        text = "Senna is used to treat constipation."
        answer, result = self.check(text)
        self.assertEqual((answer, result.action), (text, "none"))

    def test_neo4j_down_sends_the_answer_unchanged(self):
        down = InMemoryGraphStore(BASE_FACTS, available=False)
        answer, result = self.check(CONTRADICTED_ANSWER, graph=down)
        self.assertEqual(answer, CONTRADICTED_ANSWER)
        self.assertEqual(result.status, "unavailable")

    def test_graph_that_raises_mid_lookup_is_unavailable_not_an_error(self):
        class Flaky(InMemoryGraphStore):
            async def relations_between(self, pairs):
                from retrieval.graph_store import GraphUnavailable
                raise GraphUnavailable("connection reset")

        answer, result = self.check(CONTRADICTED_ANSWER, graph=Flaky(BASE_FACTS))
        self.assertEqual((answer, result.status), (CONTRADICTED_ANSWER, "unavailable"))

    def test_slow_graph_times_out(self):
        slow = InMemoryGraphStore(BASE_FACTS, delay_s=0.5)
        answer, result = self.check(CONTRADICTED_ANSWER, graph=slow, timeout_s=0.05)
        self.assertEqual((answer, result.status), (CONTRADICTED_ANSWER, "timeout"))

    def test_off_mode_does_nothing(self):
        answer, result = self.check(CONTRADICTED_ANSWER, mode="off")
        self.assertEqual((answer, result.status, result.claims), (CONTRADICTED_ANSWER, "skipped", []))

    def test_record_carries_names_and_verdicts_not_answer_text(self):
        _, result = self.check(CONTRADICTED_ANSWER)
        record = result.to_dict()
        self.assertEqual(record["contradicted"], 1)
        self.assertNotIn("sentence", record["claims"][0])
        json.dumps(record)


# ---------------------------------------------------------------------------
# query() end to end, every channel's entry point
# ---------------------------------------------------------------------------


def make_pipeline(answer="Morphine relieves cancer pain well.", vector_fail=False):
    """A SimpleRAGPipeline with a fake collection and a captured generator."""
    from simple_rag_server import SimpleRAGPipeline

    p = SimpleRAGPipeline.__new__(SimpleRAGPipeline)
    p.vector_db = FakeCollection(VECTOR_DOCS, fail=vector_fail)
    p.document_metadata = {}
    p.conversations = {}
    p.context_injector = None
    p.cross_modal_aggregator = None
    p.translation_fallback_count = 0
    p.retrieval_router = None
    p.kg_checker = None
    p.graph_store = None
    p.generated = []

    async def generate(question, context, metadatas, should_fuse=False):
        p.generated.append({"context": context, "metadatas": metadatas})
        return answer, "fake-llm"

    p._generate_answer_with_citations = generate
    return p


class QueryEndToEnd(unittest.TestCase):
    def setUp(self):
        self.audit_dir = tempfile.mkdtemp(prefix="retrieval-audit-")
        self.env = patch.dict(os.environ, {"RETRIEVAL_AUDIT_DIR": self.audit_dir})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        set_active_settings(RetrievalSettings())
        import simple_rag_server

        simple_rag_server._rag_method = "vector"

    def configure(self, pipeline, method="vector", kg_check="off", graph=None, **kw):
        import simple_rag_server

        s = set_active_settings(RetrievalSettings(rag_method=method, kg_check_mode=kg_check, **kw))
        simple_rag_server._rag_method = method
        pipeline.configure_answer_path(
            s,
            graph_store=graph if graph is not None else InMemoryGraphStore(BASE_FACTS, version="kg-test-1"),
            graphrag_engine=FakeGraphRAGEngine(),
            pageindex_engine=FakePageIndex(),
        )
        return s

    def audit_rows(self):
        rows = []
        for name in os.listdir(self.audit_dir):
            with open(os.path.join(self.audit_dir, name)) as fh:
                rows += [json.loads(line) for line in fh]
        return rows

    def test_default_path_is_unchanged(self):
        p = make_pipeline()
        self.configure(p)  # vector, check off
        result = run(p.query("Is morphine used for cancer pain?"))
        self.assertEqual(result["status"], "success")
        self.assertIsNone(p.retrieval_router, "no router in the default configuration")
        self.assertIsNone(p.kg_checker, "no checker in the default configuration")
        self.assertIn("Morphine relieves cancer pain well.", result["answer"])
        self.assertEqual(result["retrieval"]["retrievers_active"], ["vector"])
        self.assertEqual(result["retrieval"]["kg_version"], "not_in_answer_path")
        self.assertEqual(result["kg_check"]["status"], "skipped")
        self.assertNotIn("Clinical knowledge graph", p.generated[0]["context"])

    def test_patient_context_is_fetched_while_retrieval_runs(self):
        p = make_pipeline()
        self.configure(p)
        import threading

        started = threading.Event()
        seen = {}
        search = p.vector_db.query

        def slow_search(*args, **kwargs):
            # Run sequentially, the injector cannot start until this returns.
            seen["overlapped"] = started.wait(timeout=2)
            return search(*args, **kwargs)

        p.vector_db.query = slow_search

        class Injector:
            async def inject_context(self, user_id, question, max_length):
                started.set()
                return "Known: lung cancer, on oral morphine."

        p.context_injector = Injector()
        result = run(p.query("Is morphine used for cancer pain?", user_id="web_test"))
        self.assertEqual(result["status"], "success")
        self.assertTrue(seen["overlapped"], "fetched while the search ran")
        self.assertIn("Known: lung cancer", p.generated[0]["context"])
        self.assertTrue(result["patient_context_used"])

    def test_unused_patient_context_fetch_is_cancelled(self):
        p = make_pipeline()
        p.vector_db = FakeCollection([])  # nothing found: early return
        self.configure(p)
        state = {}

        class Injector:
            async def inject_context(self, user_id, question, max_length):
                await asyncio.sleep(5)
                state["finished"] = True

        p.context_injector = Injector()

        async def go():
            result = await p.query("Is morphine used for cancer pain?", user_id="web_test")
            await asyncio.sleep(0)
            state["left_running"] = [
                t for t in asyncio.all_tasks()
                if t is not asyncio.current_task() and not t.done()
                and "_patient_context" in repr(t.get_coro())]
            return result

        result = run(go())
        self.assertEqual(result["sources"], [])
        self.assertEqual(state["left_running"], [])
        self.assertNotIn("finished", state)

    def test_each_routed_mode_reaches_the_prompt(self):
        expected = {"kg": "Clinical knowledge graph", "graphrag": "GraphRAG index", "fused": "opioids.pdf"}
        for method, marker in expected.items():
            with self.subTest(method=method):
                p = make_pipeline()
                self.configure(p, method=method)
                result = run(p.query("Is morphine used for cancer pain?"))
                self.assertEqual(result["status"], "success")
                self.assertEqual(result["retrieval"]["rag_method"], method)
                self.assertIn(marker, p.generated[0]["context"])
                self.assertEqual(result["retrieval"]["fusion"]["method"], "rrf")

    def test_neo4j_down_still_answers_and_logs_the_fallback(self):
        p = make_pipeline()
        down = InMemoryGraphStore(BASE_FACTS, available=False)
        self.configure(p, method="kg", kg_check="block", graph=down)
        result = run(p.query("Is morphine used for cancer pain?"))
        self.assertEqual(result["status"], "success")
        self.assertIn("Morphine relieves cancer pain well.", result["answer"])
        self.assertEqual(result["retrieval"]["retrievers_failed"], {"kg": "unavailable"})
        self.assertEqual(result["retrieval"]["kg_version"], "unavailable")
        self.assertEqual(result["kg_check"]["status"], "unavailable")

    def test_router_missing_falls_back_to_default_vector(self):
        import simple_rag_server

        p = make_pipeline()
        set_active_settings(RetrievalSettings(rag_method="kg"))
        simple_rag_server._rag_method = "kg"
        result = run(p.query("morphine for pain"))
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["retrieval"]["fallback_reason"], "router_not_configured")

    def test_contradicted_claim_is_blocked_and_logged(self):
        p = make_pipeline(answer=CONTRADICTED_ANSWER)
        self.configure(p, method="kg", kg_check="block")
        result = run(p.query("What helps constipation from morphine?", user_id="user-42"))
        self.assertNotIn("also relieves constipation", result["answer"])
        self.assertEqual(result["kg_check"]["contradicted"], 1)
        self.assertEqual(result["kg_check"]["action"], "blocked")

        rows = self.audit_rows()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["retrieval"]["kg_version"], "kg-test-1")
        self.assertEqual(row["retrieval"]["retrievers_active"], ["vector", "kg"])
        self.assertEqual(row["retrieval"]["fusion"], {"method": "rrf", "k": 60, "per_retriever_top_k": 5})
        self.assertEqual(row["kg_check"]["contradicted"], 1)
        self.assertNotEqual(row["participant_id"], "user-42", "participant id must be pseudonymised")
        self.assertNotIn("constipation", json.dumps(row["retrieval"]).lower())
        for stage in ("retrieval", "generation", "kg_check", "total"):
            self.assertIn(stage, row["retrieval"]["stage_latency_ms"])

    def test_shadow_check_runs_after_the_reply_and_lands_in_the_record(self):
        import post_work

        p = make_pipeline(answer=CONTRADICTED_ANSWER)
        self.configure(p, method="vector", kg_check="shadow")

        async def go():
            result = await p.query("constipation?")
            at_reply = dict(result["kg_check"])
            await post_work.drain()
            return at_reply

        at_reply = run(go())
        self.assertEqual(at_reply, {"mode": "shadow", "status": "pending"})
        row = self.audit_rows()[0]
        self.assertEqual(row["kg_check"]["action"], "logged")
        self.assertIn("kg_check", row["retrieval"]["stage_latency_ms"])

    def test_kg_check_runs_on_the_vector_path_too(self):
        p = make_pipeline(answer=CONTRADICTED_ANSWER)
        self.configure(p, method="vector", kg_check="shadow")
        result = run(p.query("constipation?"))
        self.assertEqual(result["answer"].split("\n")[0], CONTRADICTED_ANSWER.split("\n")[0])
        self.assertEqual(result["kg_check"]["action"], "logged")
        self.assertEqual(result["retrieval"]["kg_version"], "kg-test-1")


# ---------------------------------------------------------------------------
# The evidence badge is computed from real retrieval distances only
# ---------------------------------------------------------------------------


class TreeOnlyPageIndex:
    """A PageIndex engine that finds one section; tree search has no distance."""

    async def search(self, query, doc_ids=None):
        node = {"filename": "opioids.pdf", "start_page": 4, "end_page": 5,
                "text": "Opioid titration is done by the treating doctor."}
        return type("R", (), {
            "selected_nodes": [node],
            "context": "Source: opioids.pdf (pg 4-5)\n" + node["text"],
        })()


class EvidenceBadgeUsesRealDistances(unittest.TestCase):
    """PageIndex and hybrid answers never feed an invented 0.5 to the badge."""

    def setUp(self):
        import simple_rag_server

        self.server = simple_rag_server
        self.saved = (simple_rag_server._rag_method, simple_rag_server._pageindex_enabled,
                      simple_rag_server._pageindex_engine)
        self.audit_dir = tempfile.mkdtemp(prefix="retrieval-audit-")
        self.env = patch.dict(os.environ, {"RETRIEVAL_AUDIT_DIR": self.audit_dir})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        set_active_settings(RetrievalSettings())
        (self.server._rag_method, self.server._pageindex_enabled,
         self.server._pageindex_engine) = self.saved

    def badge_for(self, method):
        if not self.server.SAFETY_ENHANCEMENTS_AVAILABLE:
            self.skipTest("safety_enhancements not importable")
        p = make_pipeline()
        set_active_settings(RetrievalSettings())
        self.server._rag_method = method
        self.server._pageindex_enabled = True
        self.server._pageindex_engine = TreeOnlyPageIndex()
        result = run(p.query("Is morphine used for cancer pain?"))
        self.assertEqual(result["status"], "success")
        return result["safety_enhancements"]["evidence_badge"]

    def test_pageindex_answer_badge_says_no_distance(self):
        badge = self.badge_for("pageindex")
        self.assertIsNone(badge["mean_distance"])
        self.assertEqual(badge["distance_basis"], "no distance")

    def test_hybrid_answer_badge_uses_only_the_vector_distances(self):
        badge = self.badge_for("hybrid")
        real = [d[3] for d in VECTOR_DOCS]
        self.assertAlmostEqual(badge["mean_distance"], round(sum(real) / len(real), 3))
        self.assertEqual(badge["distance_basis"], "retrieval")

    def test_vector_answer_badge_uses_the_search_distances(self):
        badge = self.badge_for("vector")
        real = [d[3] for d in VECTOR_DOCS]
        self.assertAlmostEqual(badge["mean_distance"], round(sum(real) / len(real), 3))


# ---------------------------------------------------------------------------
# Voice runs the same path
# ---------------------------------------------------------------------------


class VoiceUsesTheConfiguredPath(unittest.TestCase):
    def setUp(self):
        self.audit_dir = tempfile.mkdtemp(prefix="retrieval-audit-")
        self.env = patch.dict(os.environ, {"RETRIEVAL_AUDIT_DIR": self.audit_dir})
        self.env.start()
        import simple_rag_server

        self.pipeline = make_pipeline(
            answer="Morphine relieves cancer pain. Start morphine at 10 mg every 4 hours."
        )
        s = set_active_settings(RetrievalSettings(rag_method="kg", kg_check_mode="flag"))
        simple_rag_server._rag_method = "kg"
        self.pipeline.configure_answer_path(s, graph_store=InMemoryGraphStore(BASE_FACTS))

    def tearDown(self):
        self.env.stop()
        set_active_settings(RetrievalSettings())
        import simple_rag_server

        simple_rag_server._rag_method = "vector"

    def test_gemini_live_tool_goes_through_query_and_the_dose_boundary(self):
        from gemini_live.config import GeminiLiveConfig
        from gemini_live.session_manager import GeminiLiveSession
        from safety_enhancements import get_safety_manager

        service = type("S", (), {
            "rag_pipeline": self.pipeline,
            "safety_manager": get_safety_manager(),
            "config": GeminiLiveConfig(rag_top_k=5),
        })()
        session = GeminiLiveSession.__new__(GeminiLiveSession)
        session.service = service
        session.session_id = "live-1"
        session.config = GeminiLiveConfig(rag_top_k=5)

        out = run(session._run_rag_query("morphine for cancer pain"))
        self.assertIn("Clinical knowledge graph", self.pipeline.generated[0]["context"])
        self.assertNotIn("10 mg", out, "the dose boundary must still hold on the Live tool path")

    def test_sarvam_turn_handler_records_the_real_method(self):
        import voice_session as vs
        from safety_enhancements import get_safety_manager

        rows = []

        class Logger:
            async def record(self, **kw):
                rows.append(kw)

        handler = vs.VoiceTurnHandler(
            rag_pipeline=self.pipeline,
            safety_manager=get_safety_manager(),
            study_logger=Logger(),
        )
        turn = run(handler.handle(vs.VoiceSession(language="en-IN"), "morphine for cancer pain"))
        self.assertNotIn("10 mg", turn.answer)
        self.assertEqual(rows[0]["rag_method"], "kg")
        self.assertEqual(rows[0]["retrieval"]["retrievers_active"], ["vector", "kg"])
        self.assertEqual(rows[0]["kg_check"]["mode"], "flag")


class StudyLogCarriesTheRecord(unittest.TestCase):
    def test_contradicted_claim_lands_in_the_flagged_tier(self):
        import study_logging

        entry = study_logging.InteractionLog(
            participant_id="p", site_id="s", care_role="ASHA Worker",
            source_count=2, kg_check={"contradicted": 1},
        )
        reasons = [i["reason"] for i in study_logging._issues_for(entry)]
        self.assertIn("kg_contradicted_claim", reasons)

    def test_release_fingerprint_includes_retrieval_settings(self):
        from study_release import StudyRelease

        set_active_settings(RetrievalSettings(rag_method="fused", kg_check_mode="shadow"))
        try:
            live = StudyRelease(record={}).live_fingerprint()
        finally:
            set_active_settings(RetrievalSettings())
        self.assertEqual((live["retrieval.rag_method"], live["retrieval.kg_check"]), ("fused", "shadow"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
