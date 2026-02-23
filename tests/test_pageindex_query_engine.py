"""Tests for PageIndex query engine."""
import pytest
import asyncio
import tempfile
import os
from unittest.mock import AsyncMock
from pageindex_integration.config import PageIndexConfig
from pageindex_integration.storage import PageIndexStorage
from pageindex_integration.query_engine import PageIndexQueryEngine, PageIndexSearchResult, _extract_json


def _run(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def test_search_result_to_dict():
    result = PageIndexSearchResult(
        query="test query",
        context="some context",
        selected_nodes=[{"node_id": "0001"}],
        reasoning="because section 1 is relevant",
        doc_sources=[{"doc_id": "abc", "filename": "test.pdf"}],
        confidence=0.8,
        duration_ms=100.0,
    )
    d = result.to_dict()
    assert d["query"] == "test query"
    assert d["confidence"] == 0.8
    assert len(d["selected_nodes"]) == 1
    assert d["duration_ms"] == 100.0


def test_extract_json_direct():
    text = '{"thinking": "relevant", "node_list": ["0001"]}'
    result = _extract_json(text)
    assert result is not None
    import json
    parsed = json.loads(result)
    assert "node_list" in parsed


def test_extract_json_code_fence():
    text = '```json\n{"thinking": "relevant", "node_list": ["0001"]}\n```'
    result = _extract_json(text)
    assert result is not None


def test_extract_json_with_surrounding_text():
    text = 'Here is my analysis:\n{"thinking": "Section about pain", "node_list": ["0001", "0002"]}'
    result = _extract_json(text)
    assert result is not None


def test_extract_json_no_json():
    text = "I cannot find any relevant sections."
    result = _extract_json(text)
    assert result is None


def test_search_no_trees():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PageIndexConfig(root_dir=tmpdir)
        storage = PageIndexStorage(config)
        engine = PageIndexQueryEngine(config, storage)

        result = _run(engine.search("test query"))
        assert result.context == ""
        assert result.confidence == 0.0
        assert "No tree indexes" in result.reasoning


def test_search_with_tree():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PageIndexConfig(root_dir=tmpdir)
        storage = PageIndexStorage(config)

        tree = {
            "node_id": "0001",
            "title": "Pain Management",
            "level": 0,
            "start_page": 1,
            "end_page": 10,
            "summary": "Overview of pain management",
            "text": "",
            "children": [
                {
                    "node_id": "0002",
                    "title": "Opioid Dosing",
                    "level": 1,
                    "start_page": 5,
                    "end_page": 8,
                    "summary": "Guidelines for opioid dosing in palliative care",
                    "text": "Start with low dose morphine 5mg PO q4h. Titrate to effect.",
                    "children": [],
                },
            ],
        }
        storage.save_tree("doc1", tree, {"filename": "guide.pdf", "page_count": 10})

        engine = PageIndexQueryEngine(config, storage)
        engine._llm.chat_async = AsyncMock(
            return_value='{"thinking": "Section about opioid dosing is relevant", "node_list": ["0002"]}'
        )

        result = _run(engine.search("What is the starting dose for morphine?"))
        assert len(result.selected_nodes) == 1
        assert "morphine" in result.context.lower()
        assert result.confidence > 0


def test_search_with_doc_id_filter():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PageIndexConfig(root_dir=tmpdir)
        storage = PageIndexStorage(config)

        tree = {"node_id": "0001", "title": "Doc 1", "level": 0, "text": "content", "children": []}
        storage.save_tree("doc1", tree, {"filename": "doc1.pdf"})
        storage.save_tree("doc2", tree, {"filename": "doc2.pdf"})

        engine = PageIndexQueryEngine(config, storage)
        engine._llm.chat_async = AsyncMock(
            return_value='{"thinking": "relevant", "node_list": ["0001"]}'
        )

        result = _run(engine.search("test", doc_ids=["doc1"]))
        assert all(s["doc_id"] == "doc1" for s in result.doc_sources)


def test_get_stats():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PageIndexConfig(root_dir=tmpdir)
        config.search.cache_enabled = False
        storage = PageIndexStorage(config)
        engine = PageIndexQueryEngine(config, storage)
        stats = engine.get_stats()
        assert stats["cache"] is None


def test_repr():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PageIndexConfig(root_dir=tmpdir)
        storage = PageIndexStorage(config)
        engine = PageIndexQueryEngine(config, storage)
        r = repr(engine)
        assert "PageIndexQueryEngine" in r
