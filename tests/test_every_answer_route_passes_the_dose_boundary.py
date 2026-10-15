#!/usr/bin/env python3
"""
Every answer-returning route passes the Dose Boundary.

Before this, the main query, the voice paths and WhatsApp were guarded, but the
knowledge-graph query, the treatment and side-effect lookups, the PageIndex and
GraphRAG queries (and the admin tab that calls GraphRAG) and the patient-memory
query handed model- or graph-derived text straight to the caller. The GraphRAG
index even extracts a Dosage entity type.

What these pin down, over HTTP against the app create_app() builds:

  * every route of the app is classified in answer_routes.py, as an answer
    route or as one that returns no answer text; a new route that is neither
    fails the classification test
  * every answer route is driven here with a stub that emits a dose (or names
    the test that does so for its channel), and the dose never reaches the
    caller: the body carries a Redacted Answer or a Deferral instead
  * the admin GraphRAG tab shows the guarded text
"""

import asyncio
import json
import os
import re
import sys
import unittest
from pathlib import Path
from typing import Any, Callable, Dict, List
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

import test_route_policy as harness  # noqa: E402
from dose_detector import contains_dose  # noqa: E402
from dosage_guard import _redaction_note, _refusal_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# A dose-only line: alone it can only be deferred.
DOSE_LINE = "Give morphine 10 mg every 4 hours."
# A real answer with one dosing sentence: the gate keeps the rest (Redacted Answer).
DOSE_ANSWER = (
    "Morphine is an opioid used for moderate to severe pain in palliative care. "
    "It often causes constipation, so a laxative is usually started at the same time. "
    "Give morphine 10 mg every 4 hours. "
    "Tell the doctor at once if the patient becomes very drowsy or confused."
)


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def strings_in(value: Any) -> List[str]:
    """Every string anywhere in a decoded JSON body."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in strings_in(v)]
    if isinstance(value, (list, tuple)):
        return [s for v in value for s in strings_in(v)]
    return []


class _ToDict:
    """A result object with the to_dict() the routes call."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self._data))


# -- stubs that emit a dose -----------------------------------------------------


class DoseKnowledgeGraph:
    """Stands in for KnowledgeGraphRAG: every lookup carries dose text."""

    def __init__(self, *args, **kwargs):
        self.entity_extractor = mock.MagicMock()

        async def extract(text, use_llm=True):
            entity = _ToDict({"name": "Morphine", "properties": {"dose": DOSE_LINE}})
            return [entity], []

        self.entity_extractor.extract = extract

    async def initialize(self) -> bool:
        return True

    def is_graph_available(self) -> bool:
        return True

    async def query(self, question, include_vector_results=False, vector_results=None, top_k=10):
        return {"question": question,
                "graph_results": [{"medication": "Morphine", "dosage": DOSE_LINE,
                                   "note": DOSE_ANSWER}]}

    async def get_treatments_for_symptom(self, symptom):
        return [{"medication": "Morphine", "relationship": "TREATS", "evidence": DOSE_LINE,
                 "symptom": symptom}]

    async def get_side_effects(self, medication):
        return [{"side_effect": "Drowsiness", "frequency": "common above 30 mg a day"}]

    async def get_entity_graph(self, entity_name, depth=1):
        return {"entity": entity_name,
                "neighbors": [{"name": "Morphine", "properties": {"dosage": DOSE_LINE}}],
                "visualization": {"nodes": [{"id": "m", "label": "Morphine",
                                             "title": DOSE_LINE}], "edges": []}}

    async def search_entities(self, q, entity_type=None, limit=20):
        return [{"name": "Morphine", "description": DOSE_LINE}]

    def get_visualization_html(self, viz_data) -> str:
        titles = "".join(f"<p>{node['title']}</p>" for node in viz_data.to_dict()["nodes"])
        return f"<html>{titles}</html>"


class DoseGraphRAGEngine:
    """Stands in for GraphRAGQueryEngine: the answer and the index carry doses."""

    def __init__(self, *args, **kwargs):
        self.data_loader = mock.MagicMock()

        async def search_entities(query="", top_k=20, entity_type=None):
            return [{"title": "Morphine", "type": "Dosage", "description": DOSE_LINE, "score": 1.0}]

        async def get_entity_relationships(entity_name):
            return [{"source": entity_name, "target": "Pain", "description": DOSE_LINE}]

        async def get_community_reports_by_level(level):
            return [{"title": "Opioids", "summary": DOSE_ANSWER}]

        self.data_loader.search_entities = search_entities
        self.data_loader.get_entity_relationships = get_entity_relationships
        self.data_loader.get_community_reports_by_level = get_community_reports_by_level

    async def initialize(self) -> None:
        return None

    def _result(self, query):
        return _ToDict({"query": query, "response": DOSE_ANSWER, "method": "local",
                        "sources": [], "entities": [{"title": "Morphine", "description": DOSE_LINE}],
                        "communities": [], "confidence": 0.9, "metadata": {}})

    async def auto_search(self, query):
        return self._result(query)

    async def local_search(self, query, top_k_entities=10):
        return self._result(query)


class DosePageIndex:
    """Stands in for the PageIndex engine and storage."""

    async def search(self, query, doc_ids=None):
        return _ToDict({"query": query, "context": DOSE_ANSWER, "selected_nodes": [],
                        "reasoning": DOSE_LINE, "doc_sources": [], "confidence": 0.8,
                        "duration_ms": 1.0, "metadata": {}})

    def load_tree(self, doc_id):
        return {"doc_id": doc_id, "nodes": [{"title": "Dosing", "summary": DOSE_LINE}]}


class DoseMemoryAgent:
    """Stands in for the memory agents: a summary of past talk that repeats a dose."""

    async def query(self, patient_id, question):
        return _ToDict({"answer": DOSE_ANSWER, "cited_memory_ids": ["m1"],
                        "cited_insight_ids": [], "raw_memories_used": 1, "insights_used": 0})

    async def get_patient_insights(self, patient_id):
        return [_ToDict({"insight_id": "i1", "text": DOSE_LINE})]


class _OwnsEveryPatient:
    async def user_owns_patient(self, user_id, patient_id):
        return True


async def _pipeline_query(self, question, conversation_id=None, user_id=None, top_k=5,
                          source_language="en", **kwargs):
    return {"status": "success", "answer": DOSE_ANSWER, "sources": []}


# -- the routes, each driven with its dose stub ----------------------------------

Call = Callable[["AnswerRoutesAreGated"], Any]

CASES: Dict[str, Call] = {
    "POST /api/query": lambda t: t.client.post(
        "/api/query", data={"query": "How much morphine?"}, headers=t.admin),
    "POST /api/mobile/v1/query": lambda t: t.client.post(
        "/api/mobile/v1/query", json={"query": "How much morphine?", "language": "en"},
        headers=t.participant),
    "POST /api/kg/query": lambda t: t.client.post(
        "/api/kg/query", json={"question": "What treats pain?"}, headers=t.admin),
    "POST /api/kg/extract": lambda t: t.client.post(
        "/api/kg/extract", json={"text": "The patient takes morphine."}, headers=t.admin),
    "GET /api/kg/entity/{entity_name}": lambda t: t.client.get(
        "/api/kg/entity/Morphine", headers=t.admin),
    "GET /api/kg/treatments/{symptom}": lambda t: t.client.get(
        "/api/kg/treatments/pain", headers=t.admin),
    "GET /api/kg/side-effects/{medication}": lambda t: t.client.get(
        "/api/kg/side-effects/morphine", headers=t.admin),
    "GET /api/kg/search": lambda t: t.client.get(
        "/api/kg/search", params={"q": "morph"}, headers=t.admin),
    "GET /api/kg/visualization/{entity_name}": lambda t: t.client.get(
        "/api/kg/visualization/Morphine", headers=t.admin),
    "POST /api/graphrag/query": lambda t: t.client.post(
        "/api/graphrag/query", json={"query": "How is pain managed?"}, headers=t.admin),
    "GET /api/graphrag/entities": lambda t: t.client.get(
        "/api/graphrag/entities", params={"query": "morphine"}, headers=t.admin),
    "GET /api/graphrag/entity/{entity_name}/relationships": lambda t: t.client.get(
        "/api/graphrag/entity/Morphine/relationships", headers=t.admin),
    "GET /api/graphrag/communities": lambda t: t.client.get(
        "/api/graphrag/communities", headers=t.admin),
    "POST /api/pageindex/query": lambda t: t.client.post(
        "/api/pageindex/query", data={"query": "How is pain managed?"}, headers=t.admin),
    "GET /api/pageindex/tree/{doc_id}": lambda t: t.client.get(
        "/api/pageindex/tree/doc-1", headers=t.admin),
    "POST /api/mobile/v1/patient/{patient_id}/query-memory": lambda t: t.client.post(
        "/api/mobile/v1/patient/p1/query-memory", json={"question": "What did we discuss?"},
        headers=t.participant),
    "GET /api/mobile/v1/patient/{patient_id}/insights": lambda t: t.client.get(
        "/api/mobile/v1/patient/p1/insights", headers=t.participant),
}

# Answer routes whose channel needs a socket, a signed webhook or audio; each is
# driven with a dose by the named test.
ELSEWHERE: Dict[str, str] = {
    "POST /api/bolna/llm/{language}/v1/chat/completions":
        "tests/test_phone_providers_residency.py::test_a_streamed_turn_speaks_the_guarded_answer",
    "POST /webhook":
        "tests/test_text_path_dose_guard.py::test_twilio_text_message_and_its_translation",
    "WS /ws/retell/llm/{call_id}": "tests/test_retell_turns.py::test_a_turn_gets_a_guarded_answer",
    "WS /api/mobile/v1/ws/voice":
        "tests/test_voice_endpoint_integration.py::test_voice_answers_obey_the_same_dose_boundary_as_text",
    "POST /api/mobile/v1/query/voice":
        "tests/test_voice_session.py::test_a_dose_spoken_aloud_is_still_refused",
    "GET /api/mobile/v1/cache/bundle":
        "tests/test_offline_bundle_cache.py::test_every_cached_answer_is_guarded",
    "WS /ws/voice":
        "tests/test_web_live_voice.py::test_a_turn_that_speaks_a_dose_releases_the_deferral_instead_of_the_audio",
}


def setUpModule():
    harness.setUpModule()


def tearDownModule():
    harness.tearDownModule()


class AnswerRoutesAreGated(unittest.TestCase):
    """The real create_app(), with every answer source replaced by a dose stub."""

    app = None

    @classmethod
    def setUpClass(cls):
        import simple_rag_server as srv

        if AnswerRoutesAreGated.app is None:
            AnswerRoutesAreGated.app = harness.build_server_app()
        app = AnswerRoutesAreGated.app

        cls.patches = [
            mock.patch.object(srv, "KNOWLEDGE_GRAPH_AVAILABLE", True),
            mock.patch.object(srv, "KnowledgeGraphRAG", DoseKnowledgeGraph),
            mock.patch.object(srv, "GRAPHRAG_AVAILABLE", True),
            mock.patch.object(srv, "GraphRAGConfig", mock.MagicMock(name="GraphRAGConfig"), create=True),
            mock.patch.object(srv, "GraphRAGIndexer", mock.MagicMock(name="GraphRAGIndexer"), create=True),
            mock.patch.object(srv, "GraphRAGQueryEngine", DoseGraphRAGEngine, create=True),
            mock.patch.object(harness._FakePipeline, "query", _pipeline_query, create=True),
        ]
        for patch in cls.patches:
            patch.start()
        # The KG and GraphRAG services are built by their startup handlers.
        for handler in app.router.on_startup:
            if handler.__name__ in ("startup_knowledge_graph", "startup_graphrag"):
                run(handler())
        cls.client = TestClient(app, raise_server_exceptions=False)

    @classmethod
    def tearDownClass(cls):
        for patch in reversed(cls.patches):
            patch.stop()

    def setUp(self):
        import mobile_api.dependencies as deps
        import mobile_api.router as router_module
        import simple_rag_server as srv
        from auth.middleware import get_jwt_handler, init_auth
        from auth.policy import issue_admin_token

        engine = DosePageIndex()
        memory = DoseMemoryAgent()
        for patch in (
            mock.patch.object(srv, "_pageindex_enabled", True),
            mock.patch.object(srv, "_pageindex_engine", engine),
            mock.patch.object(srv, "_pageindex_storage", engine),
            mock.patch.object(deps, "_memory_agents_query", memory),
            mock.patch.object(deps, "_memory_agents_store", memory),
            mock.patch.object(router_module, "get_memory_manager", lambda: _OwnsEveryPatient()),
            mock.patch.object(router_module._study_logger, "record", mock.AsyncMock()),
        ):
            patch.start()
            self.addCleanup(patch.stop)

        init_auth(jwt_secret=harness.MOBILE_SECRET)
        token = get_jwt_handler().create_token_pair("asha-1", "asha", "site-1").access_token
        self.participant = {"Authorization": f"Bearer {token}"}
        self.admin = {"Authorization": f"Bearer {issue_admin_token('dr-gate')}"}

    def assert_gated(self, route: str, response) -> None:
        self.assertEqual(response.status_code, 200, f"{route}: {response.text}")
        if response.headers.get("content-type", "").startswith("text/html"):
            texts = [response.text]
        else:
            texts = strings_in(response.json())
        leaked = [s for s in texts if contains_dose(s)]
        self.assertEqual(leaked, [], f"{route} returned a dose")
        deferral, note = _refusal_for("en"), _redaction_note("en").strip()
        self.assertTrue(any(deferral in s or note in s for s in texts),
                        f"{route} returned neither a Redacted Answer nor a Deferral")

    def test_every_answer_route_withholds_a_stubbed_dose(self):
        for route, call in CASES.items():
            with self.subTest(route=route):
                self.assert_gated(route, call(self))

    def test_a_graph_answer_carries_the_safety_record(self):
        body = CASES["POST /api/graphrag/query"](self).json()
        self.assertTrue(body["safety"]["dosage_blocked"])
        self.assertTrue(body["result"]["response"].endswith(_redaction_note("en")))

    def test_the_memory_answer_is_a_redacted_answer_in_the_callers_language(self):
        response = self.client.post(
            "/api/mobile/v1/patient/p1/query-memory",
            json={"question": "हमने क्या बात की?", "language": "hi"}, headers=self.participant)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["answer"].endswith(_redaction_note("hi")))
        self.assertTrue(body["safety"]["dosage_blocked"])


class EveryRouteIsClassified(unittest.TestCase):
    """A new route must be declared as an answer route or as one with no answer text."""

    @classmethod
    def setUpClass(cls):
        if AnswerRoutesAreGated.app is None:
            AnswerRoutesAreGated.app = harness.build_server_app()
        cls.app = AnswerRoutesAreGated.app

    def test_every_route_of_the_app_is_classified(self):
        from answer_routes import unclassified_routes

        self.assertEqual(unclassified_routes(self.app), [],
                         "classify these routes in answer_routes.py")

    def test_a_new_route_without_a_classification_is_reported(self):
        from answer_routes import ANSWER_ROUTES, unclassified_routes

        answers = {k: v for k, v in ANSWER_ROUTES.items() if k != "POST /api/kg/query"}
        self.assertIn("POST /api/kg/query", unclassified_routes(self.app, answers=answers))

    def test_no_route_is_both_an_answer_route_and_not(self):
        from answer_routes import ANSWER_ROUTES, NO_ANSWER_TEXT

        self.assertEqual(set(ANSWER_ROUTES) & set(NO_ANSWER_TEXT), set())

    def test_every_answer_route_is_driven_with_a_dose(self):
        from answer_routes import ANSWER_ROUTES

        self.assertEqual(set(CASES) & set(ELSEWHERE), set())
        self.assertEqual(set(ANSWER_ROUTES) - set(CASES) - set(ELSEWHERE), set(),
                         "add a dose-stub case for these answer routes")
        self.assertEqual((set(CASES) | set(ELSEWHERE)) - set(ANSWER_ROUTES), set(),
                         "these cases name routes answer_routes.py does not list")

    def test_the_tests_named_for_other_channels_exist(self):
        for route, ref in ELSEWHERE.items():
            path, name = ref.split("::")
            with self.subTest(route=route):
                self.assertRegex((ROOT / path).read_text(encoding="utf-8"),
                                 rf"def {re.escape(name)}\(")


class StructuredPayloadsPassTheGate(unittest.TestCase):
    """guard_answer_payload, the gate for graph and index data."""

    def test_a_critical_question_gets_the_guidance_first_but_graph_data_keeps_no_dose(self):
        from dosage_guard import emergency_guidance_for
        from safety_enhancements import guard_answer_payload

        payload = {"response": "Lay the person on their side and keep them warm.",
                   "entities": [{"title": "Morphine", "description": DOSE_LINE}]}
        guarded, safety = guard_answer_payload(
            payload, query="He has stopped breathing and is unconscious", language="en",
            answer_key="response")
        self.assertTrue(guarded["response"].startswith(emergency_guidance_for("en")))
        self.assertEqual(safety["emergency_level"], "critical")
        self.assertFalse(contains_dose(guarded["entities"][0]["description"]))

    def test_clean_data_passes_unchanged(self):
        from safety_enhancements import guard_answer_payload

        payload = [{"side_effect": "Drowsiness", "frequency": "common"}, 3, None]
        guarded, safety = guard_answer_payload(payload, query="morphine")
        self.assertEqual(guarded, payload)
        self.assertFalse(safety["dosage_blocked"])

    def test_a_language_outside_the_eleven_records_the_fallback(self):
        from safety_enhancements import guard_answer_payload

        _, safety = guard_answer_payload({"answer": DOSE_LINE}, query="q", language="fr",
                                         answer_key="answer")
        self.assertTrue(safety["dosage_blocked"])
        self.assertEqual(safety["dosage_language_fallback"], "fr")


class AdminGraphRAGTabShowsTheGuardedText(unittest.TestCase):
    def test_the_tab_never_displays_a_dose(self):
        """Even if the route it calls handed back a dose, the tab shows none."""
        import simple_rag_server as srv

        async def unguarded_route(query, method):
            return {"status": "success",
                    "result": {"response": DOSE_ANSWER, "method": "local",
                               "entities": [{"title": "Morphine", "description": DOSE_LINE}],
                               "metadata": {"note": DOSE_LINE}}}

        ui = srv.SimpleAdminUI.__new__(srv.SimpleAdminUI)
        ui.admin_api = {"graphrag_query": unguarded_route}
        response, entities, metadata = run(ui._handle_graphrag_query("How much morphine?", "local"))
        shown = strings_in([response, entities, metadata])
        self.assertEqual([s for s in shown if contains_dose(s)], [])
        self.assertTrue(response.endswith(_redaction_note("en")))


if __name__ == "__main__":
    unittest.main()
