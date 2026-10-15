#!/usr/bin/env python3
"""
GraphRAG's own model layer is off unless it is region-pinned to India.

GraphRAG's native search and indexing call the model configured in
data/graphrag/settings.yaml through GraphRAG's own LiteLLM layer, not the one
shared Gemini client (llm/client.py). The shipped configuration uses a Gemini
API key, which is the global endpoint, so a question or a document sent that
way can leave India. Ticket 18: no GraphRAG call goes to an endpoint that is not
region-pinned.

The seam is the configured transport: GraphRAGConfig.model_residency() reads
it, and the query engine and indexer consult it before importing GraphRAG's
model layer. GraphRAG's API modules are replaced by recording fakes in
sys.modules, so a call that would leave the process is seen, not made.
"""

import asyncio
import os
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphrag_integration.config import GraphRAGConfig  # noqa: E402
from graphrag_integration.indexer import GraphRAGIndexer, IndexingStatus  # noqa: E402
from graphrag_integration.query_engine import GraphRAGQueryEngine  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SHIPPED_SETTINGS = REPO / "data" / "graphrag" / "settings.yaml"


@contextmanager
def modules_replaced(modules):
    """Put `modules` in sys.modules for the block, restoring only those names.

    patch.dict(sys.modules) would also drop every module first imported inside
    the block (numpy through pandas), and numpy cannot be imported twice."""
    saved = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def run(coro):
    return asyncio.run(coro)


def vertex_models(location="asia-south1", **extra):
    """A settings `models:` block on Vertex AI in `location`."""
    return {
        "default_chat_model": {"type": "chat", "model_provider": "vertex_ai",
                               "model": "gemini-3.5-flash", "vertex_location": location, **extra},
        "default_embedding_model": {"type": "embedding", "model_provider": "vertex_ai",
                                    "model": "gemini-embedding-001", "vertex_location": location},
    }


class RecordingGraphRAG:
    """Fake `graphrag.api` modules: every call is recorded, none leaves."""

    def __init__(self):
        self.calls = []
        query = types.ModuleType("graphrag.api.query")
        index = types.ModuleType("graphrag.api.index")
        api = types.ModuleType("graphrag.api")
        root = types.ModuleType("graphrag")

        def recorder(name):
            async def call(*args, **kwargs):
                self.calls.append(name)
                return types.SimpleNamespace(response="model text", context_data={})
            return call

        for name in ("global_search", "local_search", "drift_search", "basic_search"):
            setattr(query, name, recorder(name))
        index.build_index = recorder("build_index")
        api.build_index = index.build_index
        api.query, api.index = query, index
        root.api = api
        self.modules = {"graphrag": root, "graphrag.api": api,
                        "graphrag.api.query": query, "graphrag.api.index": index}

    def installed(self):
        return modules_replaced(self.modules)


class TheConfiguredTransport(unittest.TestCase):
    def test_the_shipped_settings_are_not_region_pinned(self):
        """The Gemini API key transport is the global endpoint."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            config = GraphRAGConfig.from_yaml(str(SHIPPED_SETTINGS))
        residency = config.model_residency()
        self.assertFalse(residency["pinned"])
        self.assertIn("not region-pinned", residency["reason"])

    def test_vertex_in_asia_south1_is_region_pinned(self):
        config = GraphRAGConfig(tempfile.mkdtemp(), settings={"models": vertex_models()})
        self.assertTrue(config.model_residency()["pinned"])

    def test_vertex_outside_india_is_not_region_pinned(self):
        config = GraphRAGConfig(tempfile.mkdtemp(), settings={"models": vertex_models("us-central1")})
        residency = config.model_residency()
        self.assertFalse(residency["pinned"])
        self.assertIn("us-central1", residency["reason"])

    def test_vertex_without_a_location_is_not_region_pinned(self):
        models = vertex_models()
        del models["default_chat_model"]["vertex_location"]
        config = GraphRAGConfig(tempfile.mkdtemp(), settings={"models": models})
        self.assertFalse(config.model_residency()["pinned"])

    def test_a_custom_api_base_is_not_region_pinned(self):
        """An endpoint override can point anywhere."""
        config = GraphRAGConfig(tempfile.mkdtemp(), settings={
            "models": vertex_models(api_base="https://proxy.example.com/v1")})
        self.assertFalse(config.model_residency()["pinned"])

    def test_one_unpinned_model_turns_the_layer_off(self):
        models = vertex_models()
        models["default_embedding_model"] = {"type": "embedding", "model_provider": "gemini",
                                             "model": "gemini-embedding-001", "api_key": "k"}
        config = GraphRAGConfig(tempfile.mkdtemp(), settings={"models": models})
        self.assertFalse(config.model_residency()["pinned"])

    def test_no_settings_at_all_is_not_region_pinned(self):
        self.assertFalse(GraphRAGConfig(tempfile.mkdtemp()).model_residency()["pinned"])


class NoGraphRAGCallLeavesIndia(unittest.TestCase):
    def shipped(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            return GraphRAGConfig.from_yaml(str(SHIPPED_SETTINGS))

    def test_no_search_with_the_shipped_settings_calls_the_model_layer(self):
        fake = RecordingGraphRAG()
        engine = GraphRAGQueryEngine(self.shipped(), enable_cache=False)
        with fake.installed():
            for search in (engine.global_search, engine.local_search,
                           engine.drift_search, engine.basic_search, engine.auto_search):
                with self.subTest(search=search.__name__):
                    result = run(search("What helps breathlessness?"))
                    self.assertEqual(result.metadata.get("model_layer"), "off")
        self.assertEqual(fake.calls, [])

    def test_indexing_with_the_shipped_settings_is_refused(self):
        fake = RecordingGraphRAG()
        indexer = GraphRAGIndexer(self.shipped())
        with fake.installed(), self.assertRaises(RuntimeError) as raised:
            run(indexer.index_documents())
        self.assertEqual(fake.calls, [])
        self.assertEqual(indexer.status, IndexingStatus.FAILED)
        self.assertIn("not region-pinned", str(raised.exception))

    def test_a_region_pinned_layer_is_used(self):
        fake = RecordingGraphRAG()
        config = GraphRAGConfig(tempfile.mkdtemp(), settings={"models": vertex_models()})
        engine = GraphRAGQueryEngine(config, enable_cache=False)
        with fake.installed():
            result = run(engine.local_search("What helps breathlessness?"))
        self.assertEqual(fake.calls, ["local_search"])
        self.assertNotEqual(result.metadata.get("model_layer"), "off")


if __name__ == "__main__":
    unittest.main()
