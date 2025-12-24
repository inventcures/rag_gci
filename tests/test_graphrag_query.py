"""
Phase 4 Tests: Query Engine

Run with: python3 tests/test_graphrag_query.py
"""

import sys
import asyncio
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_data_loader_creation():
    """Test basic data loader creation."""
    from graphrag_integration.config import GraphRAGConfig
    from graphrag_integration.data_loader import GraphRAGDataLoader

    config = GraphRAGConfig(root_dir="./data/graphrag")
    loader = GraphRAGDataLoader(config)

    assert loader.config == config
    assert loader._loaded is False
    assert loader.is_loaded() is False

    print("PASS: Data loader creation")
    return True


def test_data_loader_load_all():
    """Test loading all data (graceful with no data)."""
    from graphrag_integration.config import GraphRAGConfig
    from graphrag_integration.data_loader import GraphRAGDataLoader

    config = GraphRAGConfig(root_dir="./data/graphrag")
    loader = GraphRAGDataLoader(config)

    # Should not raise even if no parquet files exist
    result = asyncio.run(loader.load_all())

    # Result depends on whether parquet files exist
    assert isinstance(result, bool)

    print("PASS: Data loader load_all")
    return True


def test_data_loader_search_entities():
    """Test entity search functionality."""
    from graphrag_integration.config import GraphRAGConfig
    from graphrag_integration.data_loader import GraphRAGDataLoader

    config = GraphRAGConfig(root_dir="./data/graphrag")
    loader = GraphRAGDataLoader(config)

    # Search should return empty list if no data
    entities = asyncio.run(loader.search_entities("morphine", top_k=5))

    assert isinstance(entities, list)

    print("PASS: Data loader search entities")
    return True


def test_data_loader_get_stats():
    """Test getting data statistics."""
    from graphrag_integration.config import GraphRAGConfig
    from graphrag_integration.data_loader import GraphRAGDataLoader

    config = GraphRAGConfig(root_dir="./data/graphrag")
    loader = GraphRAGDataLoader(config)

    stats = asyncio.run(loader.get_stats())

    assert isinstance(stats, dict)
    assert "entities" in stats
    assert "relationships" in stats
    assert "communities" in stats
    assert "text_units" in stats
    assert "loaded" in stats

    print("PASS: Data loader get_stats")
    return True


def test_data_loader_repr():
    """Test data loader string representation."""
    from graphrag_integration.config import GraphRAGConfig
    from graphrag_integration.data_loader import GraphRAGDataLoader

    config = GraphRAGConfig(root_dir="./data/graphrag")
    loader = GraphRAGDataLoader(config)

    repr_str = repr(loader)
    assert "GraphRAGDataLoader" in repr_str
    assert "loaded=" in repr_str

    print("PASS: Data loader repr")
    return True


def test_query_engine_creation():
    """Test basic query engine creation."""
    from graphrag_integration.config import GraphRAGConfig
    from graphrag_integration.query_engine import GraphRAGQueryEngine

    config = GraphRAGConfig(root_dir="./data/graphrag")
    engine = GraphRAGQueryEngine(config)

    assert engine.config == config
    assert engine._initialized is False
    assert engine.data_loader is not None

    print("PASS: Query engine creation")
    return True


def test_query_engine_initialize():
    """Test query engine initialization."""
    from graphrag_integration.config import GraphRAGConfig
    from graphrag_integration.query_engine import GraphRAGQueryEngine

    config = GraphRAGConfig(root_dir="./data/graphrag")
    engine = GraphRAGQueryEngine(config)

    asyncio.run(engine.initialize())

    assert engine._initialized is True

    print("PASS: Query engine initialization")
    return True


def test_search_method_enum():
    """Test SearchMethod enum values."""
    from graphrag_integration.query_engine import SearchMethod

    assert SearchMethod.GLOBAL.value == "global"
    assert SearchMethod.LOCAL.value == "local"
    assert SearchMethod.DRIFT.value == "drift"
    assert SearchMethod.BASIC.value == "basic"

    print("PASS: SearchMethod enum")
    return True


def test_search_result_creation():
    """Test SearchResult dataclass."""
    from graphrag_integration.query_engine import SearchResult, SearchMethod

    result = SearchResult(
        query="test query",
        response="test response",
        method=SearchMethod.LOCAL,
        sources=[],
        entities=[{"name": "morphine", "type": "Medication"}],
        communities=[],
        confidence=0.9,
        metadata={"test": True},
    )

    assert result.query == "test query"
    assert result.response == "test response"
    assert result.method == SearchMethod.LOCAL
    assert result.confidence == 0.9
    assert len(result.entities) == 1

    print("PASS: SearchResult creation")
    return True


def test_search_result_to_dict():
    """Test SearchResult conversion to dict."""
    from graphrag_integration.query_engine import SearchResult, SearchMethod

    result = SearchResult(
        query="test",
        response="test response",
        method=SearchMethod.GLOBAL,
        sources=[],
        entities=[],
        communities=[],
        confidence=0.8,
        metadata={},
    )

    result_dict = result.to_dict()

    assert result_dict["query"] == "test"
    assert result_dict["response"] == "test response"
    assert result_dict["method"] == "global"
    assert result_dict["confidence"] == 0.8

    print("PASS: SearchResult to_dict")
    return True


def test_query_analysis_global():
    """Test query analysis for global search."""
    from graphrag_integration.config import GraphRAGConfig
    from graphrag_integration.query_engine import GraphRAGQueryEngine, SearchMethod

    config = GraphRAGConfig(root_dir="./data/graphrag")
    engine = GraphRAGQueryEngine(config)

    method = engine._analyze_query("What are the main approaches to pain management?")
    assert method in [SearchMethod.GLOBAL, SearchMethod.LOCAL]

    method = engine._analyze_query("Give me an overview of all palliative care themes")
    assert method == SearchMethod.GLOBAL

    print("PASS: Query analysis global")
    return True


def test_query_analysis_local():
    """Test query analysis for local search."""
    from graphrag_integration.config import GraphRAGConfig
    from graphrag_integration.query_engine import GraphRAGQueryEngine, SearchMethod

    config = GraphRAGConfig(root_dir="./data/graphrag")
    engine = GraphRAGQueryEngine(config)

    method = engine._analyze_query("What are the side effects of morphine?")
    assert method == SearchMethod.LOCAL

    method = engine._analyze_query("Tell me about oxycodone dosage")
    assert method == SearchMethod.LOCAL

    print("PASS: Query analysis local")
    return True


def test_query_analysis_drift():
    """Test query analysis for DRIFT search."""
    from graphrag_integration.config import GraphRAGConfig
    from graphrag_integration.query_engine import GraphRAGQueryEngine, SearchMethod

    config = GraphRAGConfig(root_dir="./data/graphrag")
    engine = GraphRAGQueryEngine(config)

    method = engine._analyze_query("How should pain be managed in a patient with renal failure?")
    assert method == SearchMethod.DRIFT

    method = engine._analyze_query("For a patient with liver disease, considering multiple symptoms, what is the impact of treatment?")
    assert method == SearchMethod.DRIFT

    print("PASS: Query analysis DRIFT")
    return True


def test_fallback_search():
    """Test fallback search mechanism."""
    from graphrag_integration.config import GraphRAGConfig
    from graphrag_integration.query_engine import GraphRAGQueryEngine, SearchMethod

    config = GraphRAGConfig(root_dir="./data/graphrag")
    engine = GraphRAGQueryEngine(config)

    result = asyncio.run(engine._fallback_search("test query", SearchMethod.LOCAL))

    assert result.query == "test query"
    assert result.method == SearchMethod.LOCAL
    assert result.metadata.get("fallback") is True

    print("PASS: Fallback search")
    return True


def test_global_search():
    """Test global search (uses fallback without GraphRAG)."""
    from graphrag_integration.config import GraphRAGConfig
    from graphrag_integration.query_engine import GraphRAGQueryEngine, SearchMethod

    config = GraphRAGConfig(root_dir="./data/graphrag")
    engine = GraphRAGQueryEngine(config)

    result = asyncio.run(engine.global_search("What are the main pain management approaches?"))

    assert result.query == "What are the main pain management approaches?"
    assert result.method == SearchMethod.GLOBAL
    assert isinstance(result.response, str)

    print("PASS: Global search")
    return True


def test_local_search():
    """Test local search (uses fallback without GraphRAG)."""
    from graphrag_integration.config import GraphRAGConfig
    from graphrag_integration.query_engine import GraphRAGQueryEngine, SearchMethod

    config = GraphRAGConfig(root_dir="./data/graphrag")
    engine = GraphRAGQueryEngine(config)

    result = asyncio.run(engine.local_search("What is morphine?"))

    assert result.query == "What is morphine?"
    assert result.method == SearchMethod.LOCAL
    assert isinstance(result.response, str)

    print("PASS: Local search")
    return True


def test_drift_search():
    """Test DRIFT search (uses fallback without GraphRAG)."""
    from graphrag_integration.config import GraphRAGConfig
    from graphrag_integration.query_engine import GraphRAGQueryEngine, SearchMethod

    config = GraphRAGConfig(root_dir="./data/graphrag")
    engine = GraphRAGQueryEngine(config)

    result = asyncio.run(engine.drift_search("How should pain be managed in renal failure?"))

    assert result.query == "How should pain be managed in renal failure?"
    assert result.method == SearchMethod.DRIFT
    assert isinstance(result.response, str)

    print("PASS: DRIFT search")
    return True


def test_basic_search():
    """Test basic search (uses fallback without GraphRAG)."""
    from graphrag_integration.config import GraphRAGConfig
    from graphrag_integration.query_engine import GraphRAGQueryEngine, SearchMethod

    config = GraphRAGConfig(root_dir="./data/graphrag")
    engine = GraphRAGQueryEngine(config)

    result = asyncio.run(engine.basic_search("morphine"))

    assert result.query == "morphine"
    assert result.method == SearchMethod.BASIC
    assert isinstance(result.response, str)

    print("PASS: Basic search")
    return True


def test_auto_search():
    """Test auto search method selection."""
    from graphrag_integration.config import GraphRAGConfig
    from graphrag_integration.query_engine import GraphRAGQueryEngine

    config = GraphRAGConfig(root_dir="./data/graphrag")
    engine = GraphRAGQueryEngine(config)

    # Test various query types
    result = asyncio.run(engine.auto_search("What is morphine?"))
    assert result is not None
    assert isinstance(result.response, str)

    print("PASS: Auto search")
    return True


def test_query_engine_repr():
    """Test query engine string representation."""
    from graphrag_integration.config import GraphRAGConfig
    from graphrag_integration.query_engine import GraphRAGQueryEngine

    config = GraphRAGConfig(root_dir="./data/graphrag")
    engine = GraphRAGQueryEngine(config)

    repr_str = repr(engine)
    assert "GraphRAGQueryEngine" in repr_str
    assert "initialized=" in repr_str

    str_str = str(engine)
    assert str_str == repr_str

    print("PASS: Query engine repr")
    return True


def run_all_tests():
    """Run all Phase 4 tests."""
    print("=" * 60)
    print("Phase 4 Tests: Query Engine")
    print("=" * 60)
    print()

    tests = [
        # Data Loader tests
        test_data_loader_creation,
        test_data_loader_load_all,
        test_data_loader_search_entities,
        test_data_loader_get_stats,
        test_data_loader_repr,
        # Query Engine tests
        test_query_engine_creation,
        test_query_engine_initialize,
        test_search_method_enum,
        test_search_result_creation,
        test_search_result_to_dict,
        # Query Analysis tests
        test_query_analysis_global,
        test_query_analysis_local,
        test_query_analysis_drift,
        # Search tests
        test_fallback_search,
        test_global_search,
        test_local_search,
        test_drift_search,
        test_basic_search,
        test_auto_search,
        test_query_engine_repr,
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
        print("Phase 4 COMPLETE - Ready for Phase 5")
        return 0
    else:
        print("Phase 4 INCOMPLETE - Fix failing tests")
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
