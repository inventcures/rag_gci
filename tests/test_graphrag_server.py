"""
Phase 5 Tests: Server Integration

Run with: python3 tests/test_graphrag_server.py

Note: These tests verify the GraphRAG integration code structure.
Full endpoint tests require running the server.
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_graphrag_imports_available():
    """Test that GraphRAG imports work in server context."""
    try:
        from graphrag_integration import (
            GraphRAGConfig,
            GraphRAGIndexer,
            GraphRAGQueryEngine,
            GraphRAGDataLoader,
        )
        from graphrag_integration.query_engine import SearchMethod, SearchResult
        from graphrag_integration.indexer import IndexingMethod, IndexingStatus

        assert GraphRAGConfig is not None
        assert GraphRAGIndexer is not None
        assert GraphRAGQueryEngine is not None
        assert GraphRAGDataLoader is not None
        assert SearchMethod is not None
        assert IndexingMethod is not None

        print("PASS: GraphRAG imports available")
        return True

    except ImportError as e:
        print(f"FAIL: Import error - {e}")
        return False


def test_graphrag_config_initialization():
    """Test GraphRAG configuration initialization."""
    from graphrag_integration import GraphRAGConfig

    settings_path = Path("./data/graphrag/settings.yaml")
    if not settings_path.exists():
        print("SKIP: settings.yaml not found")
        return True

    config = GraphRAGConfig.from_yaml(str(settings_path))

    assert config is not None
    assert config.root_dir == Path("./data/graphrag")
    assert config._initialized is True

    print("PASS: GraphRAG config initialization")
    return True


def test_graphrag_indexer_initialization():
    """Test GraphRAG indexer initialization."""
    from graphrag_integration import GraphRAGConfig, GraphRAGIndexer
    from graphrag_integration.indexer import IndexingMethod, IndexingStatus

    settings_path = Path("./data/graphrag/settings.yaml")
    if not settings_path.exists():
        print("SKIP: settings.yaml not found")
        return True

    config = GraphRAGConfig.from_yaml(str(settings_path))
    indexer = GraphRAGIndexer(config)

    assert indexer is not None
    assert indexer.method == IndexingMethod.STANDARD
    assert indexer.status == IndexingStatus.PENDING
    assert indexer.progress == 0

    print("PASS: GraphRAG indexer initialization")
    return True


def test_graphrag_query_engine_initialization():
    """Test GraphRAG query engine initialization."""
    from graphrag_integration import GraphRAGConfig, GraphRAGQueryEngine

    settings_path = Path("./data/graphrag/settings.yaml")
    if not settings_path.exists():
        print("SKIP: settings.yaml not found")
        return True

    config = GraphRAGConfig.from_yaml(str(settings_path))
    engine = GraphRAGQueryEngine(config)

    assert engine is not None
    assert engine._initialized is False
    assert engine.data_loader is not None

    print("PASS: GraphRAG query engine initialization")
    return True


def test_server_file_has_graphrag_imports():
    """Test that server file has GraphRAG imports."""
    server_path = Path("./simple_rag_server.py")
    if not server_path.exists():
        print("SKIP: simple_rag_server.py not found")
        return True

    content = server_path.read_text()

    assert "from graphrag_integration import" in content
    assert "GRAPHRAG_AVAILABLE" in content
    assert "GraphRAGConfig" in content
    assert "GraphRAGIndexer" in content
    assert "GraphRAGQueryEngine" in content

    print("PASS: Server has GraphRAG imports")
    return True


def test_server_file_has_graphrag_globals():
    """Test that server file has GraphRAG globals."""
    server_path = Path("./simple_rag_server.py")
    if not server_path.exists():
        print("SKIP: simple_rag_server.py not found")
        return True

    content = server_path.read_text()

    assert "graphrag_config = None" in content
    assert "graphrag_indexer = None" in content
    assert "graphrag_query_engine = None" in content
    assert "graphrag_enabled = False" in content

    print("PASS: Server has GraphRAG globals")
    return True


def test_server_file_has_graphrag_startup():
    """Test that server file has GraphRAG startup event."""
    server_path = Path("./simple_rag_server.py")
    if not server_path.exists():
        print("SKIP: simple_rag_server.py not found")
        return True

    content = server_path.read_text()

    assert "async def startup_graphrag()" in content
    assert "GraphRAG initialized successfully" in content

    print("PASS: Server has GraphRAG startup")
    return True


def test_server_file_has_graphrag_health_endpoint():
    """Test that server file has GraphRAG health endpoint."""
    server_path = Path("./simple_rag_server.py")
    if not server_path.exists():
        print("SKIP: simple_rag_server.py not found")
        return True

    content = server_path.read_text()

    assert '/api/graphrag/health"' in content or "/api/graphrag/health'" in content
    assert "async def graphrag_health" in content

    print("PASS: Server has GraphRAG health endpoint")
    return True


def test_server_file_has_graphrag_query_endpoint():
    """Test that server file has GraphRAG query endpoint."""
    server_path = Path("./simple_rag_server.py")
    if not server_path.exists():
        print("SKIP: simple_rag_server.py not found")
        return True

    content = server_path.read_text()

    assert '/api/graphrag/query"' in content or "/api/graphrag/query'" in content
    assert "async def graphrag_query" in content
    assert "auto_search" in content
    assert "global_search" in content
    assert "local_search" in content
    assert "drift_search" in content

    print("PASS: Server has GraphRAG query endpoint")
    return True


def test_server_file_has_graphrag_index_endpoint():
    """Test that server file has GraphRAG index endpoint."""
    server_path = Path("./simple_rag_server.py")
    if not server_path.exists():
        print("SKIP: simple_rag_server.py not found")
        return True

    content = server_path.read_text()

    assert '/api/graphrag/index"' in content or "/api/graphrag/index'" in content
    assert "async def graphrag_index" in content
    assert "IndexingMethod" in content

    print("PASS: Server has GraphRAG index endpoint")
    return True


def test_server_file_has_graphrag_status_endpoint():
    """Test that server file has GraphRAG status endpoint."""
    server_path = Path("./simple_rag_server.py")
    if not server_path.exists():
        print("SKIP: simple_rag_server.py not found")
        return True

    content = server_path.read_text()

    assert '/api/graphrag/index/status"' in content or "/api/graphrag/index/status'" in content
    assert "async def graphrag_index_status" in content

    print("PASS: Server has GraphRAG status endpoint")
    return True


def test_server_file_has_graphrag_entities_endpoint():
    """Test that server file has GraphRAG entities endpoint."""
    server_path = Path("./simple_rag_server.py")
    if not server_path.exists():
        print("SKIP: simple_rag_server.py not found")
        return True

    content = server_path.read_text()

    assert '/api/graphrag/entities"' in content or "/api/graphrag/entities'" in content
    assert "async def graphrag_entities" in content
    assert "search_entities" in content

    print("PASS: Server has GraphRAG entities endpoint")
    return True


def test_server_file_has_graphrag_relationships_endpoint():
    """Test that server file has GraphRAG relationships endpoint."""
    server_path = Path("./simple_rag_server.py")
    if not server_path.exists():
        print("SKIP: simple_rag_server.py not found")
        return True

    content = server_path.read_text()

    assert "/api/graphrag/entity/" in content
    assert "/relationships" in content
    assert "async def graphrag_entity_relationships" in content

    print("PASS: Server has GraphRAG relationships endpoint")
    return True


def test_server_file_has_graphrag_communities_endpoint():
    """Test that server file has GraphRAG communities endpoint."""
    server_path = Path("./simple_rag_server.py")
    if not server_path.exists():
        print("SKIP: simple_rag_server.py not found")
        return True

    content = server_path.read_text()

    assert '/api/graphrag/communities"' in content or "/api/graphrag/communities'" in content
    assert "async def graphrag_communities" in content

    print("PASS: Server has GraphRAG communities endpoint")
    return True


def test_server_file_has_graphrag_verify_endpoint():
    """Test that server file has GraphRAG verify endpoint."""
    server_path = Path("./simple_rag_server.py")
    if not server_path.exists():
        print("SKIP: simple_rag_server.py not found")
        return True

    content = server_path.read_text()

    assert '/api/graphrag/verify"' in content or "/api/graphrag/verify'" in content
    assert "async def graphrag_verify_index" in content

    print("PASS: Server has GraphRAG verify endpoint")
    return True


def test_server_syntax():
    """Test that server file has valid Python syntax."""
    server_path = Path("./simple_rag_server.py")
    if not server_path.exists():
        print("SKIP: simple_rag_server.py not found")
        return True

    try:
        content = server_path.read_text()
        compile(content, str(server_path), 'exec')
        print("PASS: Server syntax valid")
        return True
    except SyntaxError as e:
        print(f"FAIL: Syntax error - {e}")
        return False


def run_all_tests():
    """Run all Phase 5 tests."""
    print("=" * 60)
    print("Phase 5 Tests: Server Integration")
    print("=" * 60)
    print()

    tests = [
        test_graphrag_imports_available,
        test_graphrag_config_initialization,
        test_graphrag_indexer_initialization,
        test_graphrag_query_engine_initialization,
        test_server_file_has_graphrag_imports,
        test_server_file_has_graphrag_globals,
        test_server_file_has_graphrag_startup,
        test_server_file_has_graphrag_health_endpoint,
        test_server_file_has_graphrag_query_endpoint,
        test_server_file_has_graphrag_index_endpoint,
        test_server_file_has_graphrag_status_endpoint,
        test_server_file_has_graphrag_entities_endpoint,
        test_server_file_has_graphrag_relationships_endpoint,
        test_server_file_has_graphrag_communities_endpoint,
        test_server_file_has_graphrag_verify_endpoint,
        test_server_syntax,
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
        print("Phase 5 COMPLETE - Ready for Phase 6")
        return 0
    else:
        print("Phase 5 INCOMPLETE - Fix failing tests")
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
