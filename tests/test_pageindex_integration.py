"""End-to-end integration tests for PageIndex."""
import os
import pytest
import asyncio
import tempfile
from unittest.mock import AsyncMock
from pageindex_integration.config import PageIndexConfig
from pageindex_integration.storage import PageIndexStorage
from pageindex_integration.tree_builder import PageIndexTreeBuilder
from pageindex_integration.query_engine import PageIndexQueryEngine
from pageindex_integration.utils import flatten_tree, strip_text_from_tree, tree_stats, count_tokens


def _run(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@pytest.fixture
def integration_setup():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PageIndexConfig(root_dir=tmpdir)
        storage = PageIndexStorage(config)
        builder = PageIndexTreeBuilder(config, storage)
        engine = PageIndexQueryEngine(config, storage)

        builder._llm.chat_async = AsyncMock(return_value="Test summary of medical content")
        engine._llm.chat_async = AsyncMock(
            return_value='{"thinking": "Section about pain management is relevant", "node_list": ["0002"]}'
        )

        yield config, storage, builder, engine, tmpdir


def test_full_pipeline(integration_setup):
    config, storage, builder, engine, tmpdir = integration_setup

    md_path = os.path.join(tmpdir, "test_doc.md")
    with open(md_path, 'w') as f:
        f.write("""# Pain Management Guide

## Assessment
Use the Numeric Rating Scale (NRS) 0-10.

## WHO Analgesic Ladder
Step 1: Non-opioid. Step 2: Weak opioid. Step 3: Strong opioid.

## Opioid Dosing
Start low, titrate slowly. Morphine 5-10mg PO q4h.
""")

    # Build tree
    result = _run(builder.build_tree(md_path, "pain_guide", {"filename": "test_doc.md"}))
    assert result["status"] == "completed"
    assert result["node_count"] >= 4  # root + 3 sections

    # Verify storage
    assert storage.has_tree("pain_guide")
    tree = storage.load_tree("pain_guide")
    assert tree is not None
    assert len(tree["children"]) == 1  # single H1
    assert len(tree["children"][0]["children"]) == 3  # 3 H2 sections under H1

    # Query
    search_result = _run(engine.search("What are opioid dosing guidelines?"))
    assert search_result.query == "What are opioid dosing guidelines?"
    assert len(search_result.selected_nodes) > 0
    assert search_result.confidence > 0

    # Stats
    stats = storage.get_stats()
    assert stats["total_trees"] == 1
    assert stats["completed"] == 1


def test_multi_document_pipeline(integration_setup):
    config, storage, builder, engine, tmpdir = integration_setup

    for i in range(3):
        md_path = os.path.join(tmpdir, f"doc{i}.md")
        with open(md_path, 'w') as f:
            f.write(f"# Document {i}\n\n## Section A\nContent A for doc {i}.\n\n## Section B\nContent B for doc {i}.")
        _run(builder.build_tree(md_path, f"doc{i}", {"filename": f"doc{i}.md"}))

    assert storage.get_stats()["total_trees"] == 3

    search_result = _run(engine.search("Section A content"))
    assert len(search_result.doc_sources) >= 1


def test_delete_and_reindex(integration_setup):
    config, storage, builder, engine, tmpdir = integration_setup

    md_path = os.path.join(tmpdir, "temp.md")
    with open(md_path, 'w') as f:
        f.write("# Temp Doc\n\nSome content.")

    _run(builder.build_tree(md_path, "temp_doc", {"filename": "temp.md"}))
    assert storage.has_tree("temp_doc")

    storage.delete_tree("temp_doc")
    assert not storage.has_tree("temp_doc")

    _run(builder.build_tree(md_path, "temp_doc", {"filename": "temp.md"}))
    assert storage.has_tree("temp_doc")


# Utils tests

def test_flatten_tree():
    tree = {
        "node_id": "0001", "title": "Root",
        "children": [
            {"node_id": "0002", "title": "Child 1", "children": []},
            {"node_id": "0003", "title": "Child 2", "children": [
                {"node_id": "0004", "title": "Grandchild", "children": []},
            ]},
        ],
    }
    node_map = flatten_tree(tree)
    assert len(node_map) == 4
    assert "0001" in node_map
    assert "0004" in node_map


def test_strip_text_from_tree():
    tree = {
        "node_id": "0001", "title": "Root", "text": "root text",
        "children": [
            {"node_id": "0002", "title": "Child", "text": "child text", "children": []},
        ],
    }
    stripped = strip_text_from_tree(tree)
    assert "text" not in stripped
    assert "text" not in stripped["children"][0]
    assert stripped["title"] == "Root"
    # Original should be unchanged
    assert tree["text"] == "root text"


def test_tree_stats():
    tree = {
        "node_id": "0001", "title": "Root", "text": "abc",
        "children": [
            {"node_id": "0002", "title": "C1", "text": "de", "children": []},
            {"node_id": "0003", "title": "C2", "text": "fgh", "children": [
                {"node_id": "0004", "title": "GC", "text": "i", "children": []},
            ]},
        ],
    }
    stats = tree_stats(tree)
    assert stats["node_count"] == 4
    assert stats["max_depth"] == 2
    assert stats["total_text_chars"] == 9  # abc + de + fgh + i


def test_count_tokens():
    assert count_tokens("hello world") > 0
    assert count_tokens("") == 0


def test_import_all():
    """Verify all public imports work."""
    from pageindex_integration import (
        PageIndexConfig,
        PageIndexTreeBuilder,
        PageIndexQueryEngine,
        PageIndexStorage,
        LLMAdapter,
    )
    assert PageIndexConfig is not None
    assert PageIndexTreeBuilder is not None
    assert PageIndexQueryEngine is not None
    assert PageIndexStorage is not None
    assert LLMAdapter is not None
