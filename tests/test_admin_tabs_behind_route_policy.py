#!/usr/bin/env python3
"""
The admin panel's Knowledge Graph and GraphRAG tabs work behind the route policy.

The tabs used to call the server over HTTP (``localhost:8000/api/kg/...``,
``/api/graphrag/...``) with no credential. Since every non-public route needs a
token (auth/policy.py), each of those calls got 401 and the tab showed an error.
The panel itself is gated as a mount, so whoever drives a tab is already a
signed-in admin; the tabs now call the server's own route functions in process,
so no token is minted, forwarded or exempted.

What these pin down, driving each tab handler of the real create_app() as the
signed-in admin would: every tab returns data, not an error.
"""

import asyncio
import inspect
import os
import sys
import unittest
from typing import Any, Dict, List
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import test_errors_and_logs_leak_nothing as errors_harness  # noqa: E402
from test_every_answer_route_passes_the_dose_boundary import (  # noqa: E402
    DoseGraphRAGEngine,
    DoseKnowledgeGraph,
)


class TabKnowledgeGraph(DoseKnowledgeGraph):
    """The KG stub, with the service calls the status tabs make."""

    async def health_check(self) -> Dict[str, Any]:
        return {"neo4j_healthy": True}

    async def get_statistics(self) -> Dict[str, Any]:
        return {"nodes": 12, "relationships": 30}

    async def import_base_knowledge(self) -> Dict[str, Any]:
        return {"entities_created": 5}


class TabGraphRAGEngine(DoseGraphRAGEngine):
    """The GraphRAG stub, with the statistics the stats tab reads."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        async def get_stats():
            return {"entities": 40, "communities": 4}

        self.data_loader.get_stats = get_stats


class TabIndexer:
    """Stands in for GraphRAGIndexer."""

    def __init__(self, *args, **kwargs):
        import simple_rag_server as srv

        self.method = srv.IndexingMethod.STANDARD
        self.status = srv.IndexingStatus.PENDING
        self.progress = 0.0
        self.indexed = asyncio.Event()

    async def index_documents(self, update_mode: bool = False) -> None:
        self.indexed.set()

    def get_stats(self) -> Dict[str, Any]:
        return {"documents": 2}

    def get_error(self):
        return None

    def get_duration(self):
        return None

    async def verify_index(self) -> Dict[str, Any]:
        return {"valid": True, "files": 6}


def run(value):
    """The handler's result, awaiting it on a fresh loop when it is a coroutine."""
    if inspect.isawaitable(value):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(value)
        finally:
            loop.close()
    return value


def errors_in(value: Any) -> List[str]:
    """Every error or auth-failure marker anywhere in a tab's output."""
    found: List[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "error" and item:
                found.append(str(item))
            if key == "status" and item == "error":
                found.append("status=error")
            found.extend(errors_in(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(errors_in(item))
    elif isinstance(value, str) and (value.startswith("Error") or "Not authenticated" in value):
        found.append(value)
    return found


class AdminTabsWorkForASignedInAdmin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import simple_rag_server as srv

        captured: Dict[str, Any] = {}

        def capture(self, rag_pipeline):
            self.rag_pipeline = rag_pipeline
            captured["ui"] = self

        cls.patches = [
            mock.patch.object(srv, "KNOWLEDGE_GRAPH_AVAILABLE", True),
            mock.patch.object(srv, "KnowledgeGraphRAG", TabKnowledgeGraph),
            mock.patch.object(srv, "GRAPHRAG_AVAILABLE", True),
            mock.patch.object(srv, "GraphRAGConfig", mock.MagicMock(name="GraphRAGConfig"), create=True),
            mock.patch.object(srv, "GraphRAGIndexer", TabIndexer, create=True),
            mock.patch.object(srv, "GraphRAGQueryEngine", TabGraphRAGEngine, create=True),
        ]
        for patch in cls.patches:
            patch.start()
        build = [
            mock.patch.object(srv, "SimpleRAGPipeline", errors_harness._FakePipeline),
            mock.patch.object(srv.SimpleAdminUI, "__init__", capture),
            mock.patch.object(srv.SimpleAdminUI, "create_gradio_interface", lambda self: None),
            mock.patch.object(srv.gr, "mount_gradio_app", lambda app, *a, **k: app),
            mock.patch.dict(os.environ, {"TWILIO_ACCOUNT_SID": "", "TWILIO_AUTH_TOKEN": ""}),
        ]
        for patch in build:
            patch.start()
        try:
            app = srv.create_app(srv._parse_args([]))
        finally:
            for patch in reversed(build):
                patch.stop()
        cls.ui = captured["ui"]
        for handler in app.router.on_startup:
            if handler.__name__ in ("startup_knowledge_graph", "startup_graphrag"):
                run(handler())

    @classmethod
    def tearDownClass(cls):
        for patch in reversed(cls.patches):
            patch.stop()

    def assert_data(self, tab: str, output: Any) -> Any:
        self.assertEqual(errors_in(output), [], f"the {tab} tab showed an error")
        return output

    def test_knowledge_graph_tabs(self):
        ui = self.ui
        result = self.assert_data("KG query", run(ui._handle_kg_query("What treats pain?")))
        self.assertIn("graph_results", result)
        result = self.assert_data("treatments", run(ui._handle_kg_treatments("pain")))
        self.assertTrue(result["treatments"])
        result = self.assert_data("side effects", run(ui._handle_kg_side_effects("morphine")))
        self.assertTrue(result["side_effects"])
        result = self.assert_data("KG health", run(ui._handle_kg_health()))
        self.assertEqual(result["status"], "healthy")
        result = self.assert_data("KG stats", run(ui._handle_kg_stats()))
        self.assertEqual(result["nodes"], 12)
        result = self.assert_data("KG import", run(ui._handle_kg_import()))
        self.assertEqual(result["status"], "success")

    def test_graphrag_tabs(self):
        ui = self.ui
        response, entities, metadata = self.assert_data(
            "GraphRAG query", run(ui._handle_graphrag_query("How is pain managed?", "auto")))
        self.assertIn("Morphine is an opioid", response)
        result = self.assert_data("start indexing", run(ui._handle_graphrag_start_indexing("fast", False)))
        self.assertEqual(result["status"], "started")
        result = self.assert_data("index status", run(ui._handle_graphrag_index_status()))
        self.assertEqual(result["stats"], {"documents": 2})
        table, entities = self.assert_data(
            "entity search", run(ui._handle_graphrag_entity_search("morphine", "All")))
        self.assertEqual(table[0][0], "Morphine")
        result = self.assert_data("GraphRAG stats", run(ui._handle_graphrag_stats()))
        self.assertEqual(result["entities"], 40)
        result = self.assert_data("verify", run(ui._handle_graphrag_verify()))
        self.assertTrue(result["valid"])

    def test_no_tab_calls_the_server_over_http(self):
        import simple_rag_server as srv

        source = inspect.getsource(srv.SimpleAdminUI)
        self.assertNotIn("localhost:8000", source)


if __name__ == "__main__":
    unittest.main()
