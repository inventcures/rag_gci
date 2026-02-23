"""Tests for PageIndex tree builder."""
import os
import pytest
import asyncio
import tempfile
from unittest.mock import AsyncMock
from pageindex_integration.config import PageIndexConfig
from pageindex_integration.storage import PageIndexStorage
from pageindex_integration.tree_builder import PageIndexTreeBuilder, IndexingStatus


@pytest.fixture
def builder_setup():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PageIndexConfig(root_dir=tmpdir)
        storage = PageIndexStorage(config)
        builder = PageIndexTreeBuilder(config, storage)
        builder._llm.chat_async = AsyncMock(return_value="Test summary of medical content")
        yield builder, storage, tmpdir


def _run(coro):
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def test_build_tree_markdown(builder_setup):
    builder, storage, tmpdir = builder_setup
    md_content = """# Chapter 1
Introduction text here.

## Section 1.1
Details about section 1.1.

## Section 1.2
Details about section 1.2.

# Chapter 2
More content here.
"""
    md_path = os.path.join(tmpdir, "test.md")
    with open(md_path, 'w') as f:
        f.write(md_content)

    result = _run(builder.build_tree(md_path, "test_doc", {"filename": "test.md"}))
    assert result["status"] == "completed"
    assert result["node_count"] > 0
    assert storage.has_tree("test_doc")


def test_build_tree_text_short(builder_setup):
    builder, storage, tmpdir = builder_setup
    txt_path = os.path.join(tmpdir, "short.txt")
    with open(txt_path, 'w') as f:
        f.write("This is a short document about pain management.")

    result = _run(builder.build_tree(txt_path, "short_doc", {"filename": "short.txt"}))
    assert result["status"] == "completed"
    assert result["node_count"] >= 1


def test_build_tree_unsupported(builder_setup):
    builder, storage, tmpdir = builder_setup
    bad_path = os.path.join(tmpdir, "file.xyz")
    with open(bad_path, 'w') as f:
        f.write("data")

    result = _run(builder.build_tree(bad_path, "bad_doc"))
    assert result["status"] == "failed"
    assert "Unsupported" in result["error"]


def test_build_tree_missing_file(builder_setup):
    builder, storage, tmpdir = builder_setup
    result = _run(builder.build_tree("/nonexistent/file.pdf", "missing_doc"))
    assert result["status"] == "failed"


def test_indexing_status(builder_setup):
    builder, _, _ = builder_setup
    assert builder.get_status("unknown") == IndexingStatus.PENDING.value


def test_batch_index(builder_setup):
    builder, storage, tmpdir = builder_setup
    md1 = os.path.join(tmpdir, "doc1.md")
    md2 = os.path.join(tmpdir, "doc2.md")
    with open(md1, 'w') as f:
        f.write("# Doc 1\nContent of doc 1.")
    with open(md2, 'w') as f:
        f.write("# Doc 2\nContent of doc 2.")

    documents = [
        {"doc_id": "d1", "file_path": md1, "metadata": {"filename": "doc1.md"}},
        {"doc_id": "d2", "file_path": md2, "metadata": {"filename": "doc2.md"}},
    ]
    result = _run(builder.batch_index(documents))
    assert result["total"] == 2
    assert result["completed"] == 2
    assert result["failed"] == 0


def test_node_id_generation(builder_setup):
    builder, _, _ = builder_setup
    builder._reset_counter()
    assert builder._next_node_id() == "0001"
    assert builder._next_node_id() == "0002"
    builder._reset_counter()
    assert builder._next_node_id() == "0001"


def test_repr(builder_setup):
    builder, _, _ = builder_setup
    r = repr(builder)
    assert "PageIndexTreeBuilder" in r
