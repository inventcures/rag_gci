"""
Phase 3 Tests: Indexing Pipeline

Run with: python3 tests/test_graphrag_indexer.py
"""

import os
import sys
import asyncio
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import types  # noqa: E402
from contextlib import contextmanager  # noqa: E402

# Indexing calls GraphRAG's own model layer, so since ticket 18 it runs only
# when that layer is region-pinned (Vertex AI in asia-south1). These settings
# are, and `no_network_build_index` stands in for GraphRAG's build_index so
# that no test indexes for real whether or not graphrag is installed.
PINNED_SETTINGS = {
    "models": {
        "default_chat_model": {"type": "chat", "model_provider": "vertex_ai",
                               "model": "gemini-3.5-flash", "vertex_location": "asia-south1"},
        "default_embedding_model": {"type": "embedding", "model_provider": "vertex_ai",
                                    "model": "gemini-embedding-001", "vertex_location": "asia-south1"},
    }
}


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


def pinned_config():
    """A GraphRAGConfig over ./data/graphrag whose model layer is region-pinned."""
    from graphrag_integration.config import GraphRAGConfig

    return GraphRAGConfig(root_dir="./data/graphrag", settings=PINNED_SETTINGS)


@contextmanager
def no_network_build_index():
    """GraphRAG's index API replaced by a no-op, so indexing never leaves the process."""
    async def build_index(*args, **kwargs):
        return []

    index = types.ModuleType("graphrag.api.index")
    index.build_index = build_index
    api = types.ModuleType("graphrag.api")
    api.build_index, api.index = build_index, index
    root = types.ModuleType("graphrag")
    root.api = api
    with modules_replaced({"graphrag": root, "graphrag.api": api, "graphrag.api.index": index}):
        yield


def test_indexer_creation():
    """Test basic indexer creation."""
    from graphrag_integration.config import GraphRAGConfig
    from graphrag_integration.indexer import GraphRAGIndexer, IndexingMethod, IndexingStatus

    config = GraphRAGConfig(root_dir="./data/graphrag")
    indexer = GraphRAGIndexer(config)

    assert indexer.config == config
    assert indexer.method == IndexingMethod.STANDARD
    assert indexer.status == IndexingStatus.PENDING
    assert indexer.progress == 0

    print("PASS: Indexer creation")
    return True


def test_indexer_with_fast_method():
    """Test indexer with fast indexing method."""
    from graphrag_integration.config import GraphRAGConfig
    from graphrag_integration.indexer import GraphRAGIndexer, IndexingMethod

    config = GraphRAGConfig(root_dir="./data/graphrag")
    indexer = GraphRAGIndexer(config, method=IndexingMethod.FAST)

    assert indexer.method == IndexingMethod.FAST

    print("PASS: Indexer with fast method")
    return True


def test_indexer_callback_registration():
    """Test callback registration and removal."""
    from graphrag_integration.config import GraphRAGConfig
    from graphrag_integration.indexer import GraphRAGIndexer

    config = GraphRAGConfig(root_dir="./data/graphrag")
    indexer = GraphRAGIndexer(config)

    callback_called = []

    def test_callback(status: str, progress: int):
        callback_called.append((status, progress))

    indexer.add_callback(test_callback)
    assert len(indexer._callbacks) == 1

    # Manually notify to test callback
    indexer._notify_callbacks("Test status", 50)
    assert len(callback_called) == 1
    assert callback_called[0] == ("Test status", 50)
    assert indexer.progress == 50

    # Remove callback
    indexer.remove_callback(test_callback)
    assert len(indexer._callbacks) == 0

    print("PASS: Callback registration")
    return True


def test_indexer_mock_indexing():
    """Test mock indexing (without GraphRAG installed)."""
    from graphrag_integration.config import GraphRAGConfig
    from graphrag_integration.indexer import GraphRAGIndexer, IndexingStatus

    config = GraphRAGConfig(root_dir="./data/graphrag")
    indexer = GraphRAGIndexer(config)

    # Run mock indexing
    result = asyncio.run(indexer._mock_index())

    assert result["status"] == "mock"
    assert result["mock"] is True
    assert "documents_processed" in result
    assert "entities_extracted" in result
    assert "relationships_extracted" in result
    assert "communities_created" in result

    print("PASS: Mock indexing")
    return True


def test_indexer_full_index():
    """Test the full indexing pipeline with a region-pinned model layer."""
    from graphrag_integration.indexer import GraphRAGIndexer, IndexingStatus

    indexer = GraphRAGIndexer(pinned_config())

    progress_updates = []

    def track_progress(status: str, progress: int):
        progress_updates.append((status, progress))

    indexer.add_callback(track_progress)

    # Run indexing
    with no_network_build_index():
        result = asyncio.run(indexer.index_documents())

    assert result["status"] in ["success", "mock"]
    assert indexer.status == IndexingStatus.COMPLETED
    assert indexer.progress == 100
    assert "duration_seconds" in result
    assert len(progress_updates) > 0

    print("PASS: Full indexing pipeline")
    return True


def test_indexer_stats():
    """Test statistics collection."""
    from graphrag_integration.indexer import GraphRAGIndexer

    indexer = GraphRAGIndexer(pinned_config())

    # Run indexing first
    with no_network_build_index():
        asyncio.run(indexer.index_documents())

    stats = indexer.get_stats()
    assert isinstance(stats, dict)
    assert "status" in stats

    duration = indexer.get_duration()
    assert duration is not None
    assert duration >= 0

    print("PASS: Statistics collection")
    return True


def test_indexer_error_handling():
    """Test error retrieval."""
    from graphrag_integration.indexer import GraphRAGIndexer

    indexer = GraphRAGIndexer(pinned_config())

    # Initially no error
    assert indexer.get_error() is None

    # After successful indexing, still no error
    with no_network_build_index():
        asyncio.run(indexer.index_documents())
    assert indexer.get_error() is None

    print("PASS: Error handling")
    return True


def test_indexer_repr():
    """Test string representation."""
    from graphrag_integration.config import GraphRAGConfig
    from graphrag_integration.indexer import GraphRAGIndexer

    config = GraphRAGConfig(root_dir="./data/graphrag")
    indexer = GraphRAGIndexer(config)

    repr_str = repr(indexer)
    assert "GraphRAGIndexer" in repr_str
    assert "standard" in repr_str
    assert "pending" in repr_str

    str_str = str(indexer)
    assert str_str == repr_str

    print("PASS: String representation")
    return True


def test_indexing_method_enum():
    """Test IndexingMethod enum values."""
    from graphrag_integration.indexer import IndexingMethod

    assert IndexingMethod.STANDARD.value == "standard"
    assert IndexingMethod.FAST.value == "fast"

    print("PASS: IndexingMethod enum")
    return True


def test_indexing_status_enum():
    """Test IndexingStatus enum values."""
    from graphrag_integration.indexer import IndexingStatus

    assert IndexingStatus.PENDING.value == "pending"
    assert IndexingStatus.RUNNING.value == "running"
    assert IndexingStatus.COMPLETED.value == "completed"
    assert IndexingStatus.FAILED.value == "failed"

    print("PASS: IndexingStatus enum")
    return True


def test_prompts_exist():
    """Test that all required prompt files exist."""
    prompts_dir = Path("./graphrag_integration/prompts")

    required_prompts = [
        "entity_extraction.txt",
        "community_report.txt",
        "global_search_map.txt",
        "global_search_reduce.txt",
        "local_search.txt",
    ]

    for prompt_name in required_prompts:
        prompt_path = prompts_dir / prompt_name
        assert prompt_path.exists(), f"Missing prompt: {prompt_name}"

        # Check file is not empty
        content = prompt_path.read_text()
        assert len(content) > 100, f"Prompt too short: {prompt_name}"

    print("PASS: Prompts exist")
    return True


def test_entity_extraction_prompt_content():
    """Test entity extraction prompt has palliative care specifics."""
    prompt_path = Path("./graphrag_integration/prompts/entity_extraction.txt")
    content = prompt_path.read_text()

    # Check for palliative care specific content
    assert "palliative" in content.lower()
    assert "Symptom" in content
    assert "Medication" in content
    assert "morphine" in content.lower()
    assert "opioid" in content.lower()
    assert "WHO" in content
    assert "TREATS" in content
    assert "SIDE_EFFECT_OF" in content

    print("PASS: Entity extraction prompt content")
    return True


def test_community_report_prompt_content():
    """Test community report prompt has medical focus."""
    prompt_path = Path("./graphrag_integration/prompts/community_report.txt")
    content = prompt_path.read_text()

    # Check for medical report structure
    assert "community" in content.lower()
    assert "TITLE" in content
    assert "SUMMARY" in content
    assert "SAFETY" in content
    assert "medication" in content.lower()

    print("PASS: Community report prompt content")
    return True


def test_verify_index_structure():
    """Test verify_index method returns correct structure."""
    from graphrag_integration.config import GraphRAGConfig
    from graphrag_integration.indexer import GraphRAGIndexer

    config = GraphRAGConfig(root_dir="./data/graphrag")
    indexer = GraphRAGIndexer(config)

    # Run verification (will have errors since no real index exists)
    result = asyncio.run(indexer.verify_index())

    assert "valid" in result
    assert "errors" in result
    assert "warnings" in result
    assert "files_checked" in result
    assert "file_stats" in result
    assert isinstance(result["errors"], list)
    assert isinstance(result["warnings"], list)

    print("PASS: Verify index structure")
    return True


def run_all_tests():
    """Run all Phase 3 tests."""
    print("=" * 60)
    print("Phase 3 Tests: Indexing Pipeline")
    print("=" * 60)
    print()

    tests = [
        test_indexer_creation,
        test_indexer_with_fast_method,
        test_indexer_callback_registration,
        test_indexer_mock_indexing,
        test_indexer_full_index,
        test_indexer_stats,
        test_indexer_error_handling,
        test_indexer_repr,
        test_indexing_method_enum,
        test_indexing_status_enum,
        test_prompts_exist,
        test_entity_extraction_prompt_content,
        test_community_report_prompt_content,
        test_verify_index_structure,
    ]

    results = []
    for test in tests:
        print(f"Running: {test.__name__}")
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"FAIL: {e}")
            results.append(False)
        print()

    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    if all(results):
        print("Phase 3 COMPLETE - Ready for Phase 4")
        return 0
    else:
        print("Phase 3 INCOMPLETE - Fix failing tests")
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
