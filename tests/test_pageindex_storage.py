"""Tests for PageIndex storage layer."""
import pytest
import tempfile
from pathlib import Path
from pageindex_integration.config import PageIndexConfig
from pageindex_integration.storage import PageIndexStorage


@pytest.fixture
def tmp_storage():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PageIndexConfig(root_dir=tmpdir)
        yield PageIndexStorage(config), config


def test_save_and_load_tree(tmp_storage):
    storage, config = tmp_storage
    tree = {
        "node_id": "0001",
        "title": "Test Document",
        "level": 0,
        "text": "Hello world",
        "children": [],
    }
    storage.save_tree("doc1", tree, {"filename": "test.pdf", "page_count": 5})
    loaded = storage.load_tree("doc1")
    assert loaded is not None
    assert loaded["title"] == "Test Document"
    assert loaded["node_id"] == "0001"


def test_has_tree(tmp_storage):
    storage, _ = tmp_storage
    assert not storage.has_tree("doc1")
    tree = {"node_id": "0001", "title": "Test", "children": []}
    storage.save_tree("doc1", tree, {"filename": "test.pdf"})
    assert storage.has_tree("doc1")


def test_delete_tree(tmp_storage):
    storage, _ = tmp_storage
    tree = {"node_id": "0001", "title": "Test", "children": []}
    storage.save_tree("doc1", tree, {"filename": "test.pdf"})
    assert storage.has_tree("doc1")
    assert storage.delete_tree("doc1")
    assert not storage.has_tree("doc1")


def test_delete_nonexistent(tmp_storage):
    storage, _ = tmp_storage
    assert not storage.delete_tree("nonexistent")


def test_list_trees(tmp_storage):
    storage, _ = tmp_storage
    tree = {"node_id": "0001", "title": "Test", "children": []}
    storage.save_tree("doc1", tree, {"filename": "a.pdf"})
    storage.save_tree("doc2", tree, {"filename": "b.pdf"})
    entries = storage.list_trees()
    assert len(entries) == 2
    filenames = {e.filename for e in entries}
    assert "a.pdf" in filenames
    assert "b.pdf" in filenames


def test_get_stats(tmp_storage):
    storage, _ = tmp_storage
    tree = {
        "node_id": "0001", "title": "Root", "text": "x",
        "children": [
            {"node_id": "0002", "title": "Child", "text": "y", "children": []},
        ],
    }
    storage.save_tree("doc1", tree, {"filename": "test.pdf", "page_count": 10})
    stats = storage.get_stats()
    assert stats["total_trees"] == 1
    assert stats["completed"] == 1
    assert stats["total_nodes"] == 2
    assert stats["total_pages"] == 10


def test_set_status(tmp_storage):
    storage, _ = tmp_storage
    storage.set_status("doc1", "building")
    entries = storage.list_trees()
    assert len(entries) == 1
    assert entries[0].status == "building"


def test_load_nonexistent(tmp_storage):
    storage, _ = tmp_storage
    assert storage.load_tree("nonexistent") is None


def test_persistence(tmp_storage):
    storage, config = tmp_storage
    tree = {"node_id": "0001", "title": "Test", "children": []}
    storage.save_tree("doc1", tree, {"filename": "test.pdf"})

    # Create a new storage instance from the same config
    storage2 = PageIndexStorage(config)
    assert storage2.has_tree("doc1")
    loaded = storage2.load_tree("doc1")
    assert loaded["title"] == "Test"


def test_entry_to_dict(tmp_storage):
    storage, _ = tmp_storage
    tree = {"node_id": "0001", "title": "Test", "children": []}
    storage.save_tree("doc1", tree, {"filename": "test.pdf", "page_count": 5})
    entries = storage.list_trees()
    d = entries[0].to_dict()
    assert d["doc_id"] == "doc1"
    assert d["filename"] == "test.pdf"
    assert d["status"] == "completed"
    assert d["node_count"] == 1


def test_repr(tmp_storage):
    storage, _ = tmp_storage
    r = repr(storage)
    assert "PageIndexStorage" in r
