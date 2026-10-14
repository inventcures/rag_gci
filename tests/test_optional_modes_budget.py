"""
Optional retrieval modes stay inside their budgets.

Before this, turning on PageIndex, GraphRAG or the knowledge graph for an
experiment could stall the server: PageIndex navigated documents one after
another with a 2 s gap forced between LLM calls, and read every tree from
disk on every query; GraphRAG scanned its tables with DataFrame.iterrows on
the event loop; the KG matched unlabelled nodes on toLower(name), a scan of
the whole graph per name.

What these pin down:

  * PageIndex over 10 documents finishes inside its fused-mode timeout,
    reads each tree from disk once, and keeps results in document order
  * a GraphRAG lookup over a large entity table does not block a concurrent
    keepalive, and returns what the old scan returned
  * KG lookups match `KGEntity.name_lower` (indexed); entities are written
    with that label and property, and older graphs are backfilled. With a
    Neo4j to talk to (NEO4J_URI), EXPLAIN shows an index seek.
"""

import asyncio
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def tree(n):
    return {
        "node_id": "0001", "title": f"Guide {n}", "level": 0, "start_page": 1, "end_page": 10,
        "summary": "Overview", "text": "",
        "children": [{"node_id": "0002", "title": "Breathlessness", "level": 1, "start_page": 2,
                      "end_page": 4, "summary": "Breathlessness at home",
                      "text": f"Sit upright and use a fan (guide {n}).", "children": []}],
    }


class PageIndexBudget(unittest.TestCase):
    def engine(self, tmpdir, docs=10, delay=0.4):
        from pageindex_integration.config import PageIndexConfig
        from pageindex_integration.query_engine import PageIndexQueryEngine
        from pageindex_integration.storage import PageIndexStorage
        from retrieval.settings import RetrievalSettings

        config = PageIndexConfig(root_dir=tmpdir)
        config.search.cache_enabled = False
        storage = PageIndexStorage(config)
        for n in range(docs):
            storage.save_tree(f"doc{n}", tree(n), {"filename": f"guide{n}.pdf", "page_count": 10})
        engine = PageIndexQueryEngine(config, storage)

        async def navigate(messages, **kw):
            await asyncio.sleep(delay)
            return '{"thinking": "breathlessness section", "node_list": ["0002"]}'

        engine._llm.chat_async = navigate
        self.timeout = RetrievalSettings().pageindex_timeout_s
        return engine, storage

    def test_ten_documents_inside_the_fused_timeout(self):
        with tempfile.TemporaryDirectory() as tmpdir, \
             mock.patch("pageindex_integration.query_engine.count_tokens", lambda t: len(t) // 4):
            engine, _ = self.engine(tmpdir)
            started = time.perf_counter()
            result = asyncio.run(asyncio.wait_for(engine.search("breathless at night"), self.timeout))
            took = time.perf_counter() - started
        self.assertLess(took, 2.0, "documents were navigated one after another")
        self.assertEqual([n["filename"] for n in result.selected_nodes],
                         [f"guide{n}.pdf" for n in range(10)])

    def test_the_real_rate_limiter_does_not_serialise_a_query(self):
        """The default 30 per minute admits one query's ten calls at once."""
        from pageindex_integration.llm_adapter import LLMAdapter

        with tempfile.TemporaryDirectory() as tmpdir, \
             mock.patch("pageindex_integration.query_engine.count_tokens", lambda t: len(t) // 4):
            engine, _ = self.engine(tmpdir, delay=0.0)

            class Response:
                ok, text, error = True, '{"node_list": ["0002"]}', None

            class Client:
                async def generate(self, prompt, **kw):
                    await asyncio.sleep(0.2)
                    return Response()

            engine._llm = LLMAdapter(engine._config.llm)
            with mock.patch("pageindex_integration.llm_adapter.get_llm_client", return_value=Client()):
                started = time.perf_counter()
                asyncio.run(engine.search("breathless at night"))
                self.assertLess(time.perf_counter() - started, 1.5)

    def test_trees_are_read_from_disk_once(self):
        import json as real_json

        with tempfile.TemporaryDirectory() as tmpdir, \
             mock.patch("pageindex_integration.query_engine.count_tokens", lambda t: len(t) // 4):
            engine, storage = self.engine(tmpdir, docs=3, delay=0.0)
            with mock.patch("pageindex_integration.storage.json.load",
                            side_effect=real_json.load) as reads:
                asyncio.run(engine.search("breathless at night"))
                asyncio.run(engine.search("mouth care"))
            self.assertEqual(reads.call_count, 3)
            # A rebuilt tree is picked up.
            storage.save_tree("doc0", tree(99), {"filename": "guide0.pdf", "page_count": 10})
            self.assertEqual(storage.load_tree("doc0")["title"], "Guide 99")


def old_search_entities(df, query, top_k=10):
    """The iterrows scan this replaced, kept to check results are unchanged."""
    query_lower = query.lower()
    matches = []
    for _, row in df.iterrows():
        score = 0.0
        title = str(row.get("title", row.get("name", ""))).lower()
        description = str(row.get("description", "")).lower()
        if query_lower in title:
            score += 3.0
        if query_lower in description:
            score += 1.5
        for word in query_lower.split():
            if len(word) >= 3:
                if word in title:
                    score += 0.5
                if word in description:
                    score += 0.25
        if score > 0:
            matches.append((row.get("title"), score))
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[:top_k]


class GraphRAGOffTheLoop(unittest.TestCase):
    def loader(self, rows):
        import pandas as pd

        from graphrag_integration.config import GraphRAGConfig
        from graphrag_integration.data_loader import GraphRAGDataLoader

        words = ["pain", "breathlessness", "nausea", "morphine", "fan", "mouth", "constipation"]
        loader = GraphRAGDataLoader(GraphRAGConfig(tempfile.mkdtemp()))
        loader._entities = pd.DataFrame({
            "id": [str(i) for i in range(rows)],
            "title": [f"{words[i % 7]} {i}" for i in range(rows)],
            "type": ["Symptom"] * rows,
            "description": [f"about {words[(i * 3) % 7]} at home" for i in range(rows)],
        })
        loader._relationships = pd.DataFrame({
            "source": [f"{words[i % 7]} {i}" for i in range(rows)],
            "target": [f"{words[(i + 1) % 7]} {i}" for i in range(rows)],
            "description": ["relieves"] * rows, "weight": [1.0] * rows,
        })
        loader._loaded = True
        # What loading does after reading the tables (in its worker thread).
        loader._entity_rows()
        loader._relationship_rows()
        return loader

    def test_large_lookup_does_not_block_a_keepalive(self):
        loader = self.loader(300_000)

        async def go():
            gaps, stop = [], asyncio.Event()

            async def keepalive():
                last = time.perf_counter()
                while not stop.is_set():
                    await asyncio.sleep(0.01)
                    now = time.perf_counter()
                    gaps.append(now - last)
                    last = now

            ticker = asyncio.create_task(keepalive())
            await asyncio.sleep(0.05)
            await loader.search_entities("morphine at home", top_k=5)
            await loader.get_entity_relationships("morphine")
            stop.set()
            await ticker
            return max(gaps)

        self.assertLess(asyncio.run(go()), 0.25, "the lookup blocked the event loop")

    def test_results_unchanged(self):
        loader = self.loader(2_000)
        for query in ("morphine", "pain at home", "breathlessness 12", "zz"):
            new = asyncio.run(loader.search_entities(query, top_k=10))
            self.assertEqual([(m["title"], m["score"]) for m in new],
                             old_search_entities(loader._entities, query))
        rels = asyncio.run(loader.get_entity_relationships("Fan 4"))
        self.assertTrue(rels and all("fan 4" in (r["source"] + r["target"]).lower() for r in rels))


class KnowledgeGraphIndexedLookup(unittest.TestCase):
    def test_lookups_use_the_indexed_lowercase_name_with_a_label(self):
        from retrieval.graph_store import Neo4jGraphStore

        for cypher in (Neo4jGraphStore._FACTS_CYPHER, Neo4jGraphStore._PAIRS_CYPHER):
            self.assertNotIn("toLower(", cypher)
            self.assertIn("KGEntity {name_lower:", cypher)

    def test_entities_are_written_and_backfilled_for_the_index(self):
        from knowledge_graph.entity_extractor import Entity, EntityType
        from knowledge_graph.graph_builder import GraphBuilder

        class Client:
            def __init__(self):
                self.writes, self.nodes = [], []

            async def execute_write(self, query, params=None):
                self.writes.append(" ".join(query.split()))
                return {}

            async def create_node(self, label, properties, unique_key="name", extra_labels=()):
                self.nodes.append((label, properties, tuple(extra_labels)))
                return {"name": properties["name"]}

        client = Client()
        builder = GraphBuilder(client)
        asyncio.run(builder.initialize_schema())
        self.assertTrue(any("FOR (n:KGEntity) ON (n.name_lower)" in w for w in client.writes))
        self.assertTrue(any("SET n:KGEntity, n.name_lower = toLower(n.name)" in w for w in client.writes))

        asyncio.run(builder.add_entity(Entity(name="Morphine", type=EntityType.MEDICATION)))
        label, props, extra = client.nodes[0]
        self.assertEqual((label, props["name_lower"], extra), ("Medication", "morphine", ("KGEntity",)))

    def test_create_node_adds_the_label_without_matching_on_it(self):
        from knowledge_graph.neo4j_client import Neo4jClient

        seen = {}

        class Session:
            async def run(self, query, params):
                seen["query"] = " ".join(query.split())

                class Result:
                    async def single(self):
                        return None
                return Result()

        class Ctx:
            async def __aenter__(self):
                return Session()

            async def __aexit__(self, *a):
                return None

        client = Neo4jClient.__new__(Neo4jClient)
        client.session = lambda: Ctx()
        asyncio.run(client.create_node("Medication", {"name": "Morphine"}, extra_labels=("KGEntity",)))
        self.assertIn("MERGE (n:Medication {name: $unique_value})", seen["query"])
        self.assertIn("SET n:KGEntity", seen["query"])

    @unittest.skipUnless(os.getenv("NEO4J_URI"), "needs a Neo4j (NEO4J_URI) to EXPLAIN against")
    def test_plan_is_an_index_seek(self):
        from knowledge_graph.neo4j_client import Neo4jClient
        from retrieval.graph_store import Neo4jGraphStore

        async def go():
            client = Neo4jClient()
            await client.connect()
            await client.execute_write(
                "CREATE INDEX kg_entity_name_lower IF NOT EXISTS FOR (n:KGEntity) ON (n.name_lower)")
            async with client.session() as session:
                result = await session.run("EXPLAIN " + Neo4jGraphStore._FACTS_CYPHER,
                                           {"names": ["morphine"], "rel_types": ["TREATS"], "limit": 5})
                summary = await result.consume()
            await client.close()
            return str(summary.plan)

        self.assertIn("NodeIndexSeek", asyncio.run(go()))


if __name__ == "__main__":
    unittest.main()
