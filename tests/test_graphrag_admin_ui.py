"""
Phase 6 Tests: Admin UI Integration

Run with: python3 tests/test_graphrag_admin_ui.py

Note: These tests verify the GraphRAG admin UI code structure.
Full UI tests require running the server with Gradio.
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_server_has_graphrag_tab():
    """Test that server file has GraphRAG tab."""
    server_path = Path("./simple_rag_server.py")
    if not server_path.exists():
        print("SKIP: simple_rag_server.py not found")
        return True

    content = server_path.read_text()

    assert 'gr.TabItem("🔗 GraphRAG")' in content
    assert "Microsoft GraphRAG Integration" in content

    print("PASS: Server has GraphRAG tab")
    return True


def test_server_has_graphrag_query_tab():
    """Test that server has GraphRAG query subtab."""
    server_path = Path("./simple_rag_server.py")
    if not server_path.exists():
        print("SKIP: simple_rag_server.py not found")
        return True

    content = server_path.read_text()

    assert 'gr.TabItem("🔍 Query")' in content
    assert "graphrag_query_input" in content
    assert "graphrag_method" in content
    assert "graphrag_query_btn" in content

    print("PASS: Server has GraphRAG query tab")
    return True


def test_server_has_graphrag_indexing_tab():
    """Test that server has GraphRAG indexing subtab."""
    server_path = Path("./simple_rag_server.py")
    if not server_path.exists():
        print("SKIP: simple_rag_server.py not found")
        return True

    content = server_path.read_text()

    assert 'gr.TabItem("📥 Indexing")' in content
    assert "graphrag_index_method" in content
    assert "graphrag_update_mode" in content
    assert "graphrag_start_index_btn" in content

    print("PASS: Server has GraphRAG indexing tab")
    return True


def test_server_has_graphrag_entities_tab():
    """Test that server has GraphRAG entities subtab."""
    server_path = Path("./simple_rag_server.py")
    if not server_path.exists():
        print("SKIP: simple_rag_server.py not found")
        return True

    content = server_path.read_text()

    assert 'gr.TabItem("🏷️ Entities")' in content
    assert "graphrag_entity_search" in content
    assert "graphrag_entity_type_filter" in content
    assert "graphrag_entities_table" in content

    print("PASS: Server has GraphRAG entities tab")
    return True


def test_server_has_graphrag_statistics_tab():
    """Test that server has GraphRAG statistics subtab."""
    server_path = Path("./simple_rag_server.py")
    if not server_path.exists():
        print("SKIP: simple_rag_server.py not found")
        return True

    content = server_path.read_text()

    assert 'gr.TabItem("📊 Statistics")' in content
    assert "graphrag_stats_btn" in content
    assert "graphrag_verify_btn" in content
    assert "graphrag_stats_output" in content

    print("PASS: Server has GraphRAG statistics tab")
    return True


def test_server_has_graphrag_query_handler():
    """Test that server has GraphRAG query handler."""
    server_path = Path("./simple_rag_server.py")
    if not server_path.exists():
        print("SKIP: simple_rag_server.py not found")
        return True

    content = server_path.read_text()

    assert "def _handle_graphrag_query" in content
    assert "/api/graphrag/query" in content

    print("PASS: Server has GraphRAG query handler")
    return True


def test_server_has_graphrag_indexing_handler():
    """Test that server has GraphRAG indexing handler."""
    server_path = Path("./simple_rag_server.py")
    if not server_path.exists():
        print("SKIP: simple_rag_server.py not found")
        return True

    content = server_path.read_text()

    assert "def _handle_graphrag_start_indexing" in content
    assert "def _handle_graphrag_index_status" in content

    print("PASS: Server has GraphRAG indexing handlers")
    return True


def test_server_has_graphrag_entity_handler():
    """Test that server has GraphRAG entity search handler."""
    server_path = Path("./simple_rag_server.py")
    if not server_path.exists():
        print("SKIP: simple_rag_server.py not found")
        return True

    content = server_path.read_text()

    assert "def _handle_graphrag_entity_search" in content
    assert "/api/graphrag/entities" in content

    print("PASS: Server has GraphRAG entity handler")
    return True


def test_server_has_graphrag_stats_handler():
    """Test that server has GraphRAG stats handler."""
    server_path = Path("./simple_rag_server.py")
    if not server_path.exists():
        print("SKIP: simple_rag_server.py not found")
        return True

    content = server_path.read_text()

    assert "def _handle_graphrag_stats" in content
    assert "/api/graphrag/stats" in content

    print("PASS: Server has GraphRAG stats handler")
    return True


def test_server_has_graphrag_verify_handler():
    """Test that server has GraphRAG verify handler."""
    server_path = Path("./simple_rag_server.py")
    if not server_path.exists():
        print("SKIP: simple_rag_server.py not found")
        return True

    content = server_path.read_text()

    assert "def _handle_graphrag_verify" in content
    assert "/api/graphrag/verify" in content

    print("PASS: Server has GraphRAG verify handler")
    return True


def test_server_has_search_method_options():
    """Test that server has all search method options in UI."""
    server_path = Path("./simple_rag_server.py")
    if not server_path.exists():
        print("SKIP: simple_rag_server.py not found")
        return True

    content = server_path.read_text()

    # Check for search method radio options
    assert '"auto"' in content
    assert '"global"' in content
    assert '"local"' in content
    assert '"drift"' in content
    assert '"basic"' in content

    print("PASS: Server has all search method options")
    return True


def test_server_has_entity_type_options():
    """Test that server has entity type filter options."""
    server_path = Path("./simple_rag_server.py")
    if not server_path.exists():
        print("SKIP: simple_rag_server.py not found")
        return True

    content = server_path.read_text()

    # Check for entity type dropdown options
    assert '"Symptom"' in content
    assert '"Medication"' in content
    assert '"Condition"' in content
    assert '"Treatment"' in content
    assert '"SideEffect"' in content

    print("PASS: Server has entity type options")
    return True


def test_server_has_indexing_method_options():
    """Test that server has indexing method options."""
    server_path = Path("./simple_rag_server.py")
    if not server_path.exists():
        print("SKIP: simple_rag_server.py not found")
        return True

    content = server_path.read_text()

    # Check for indexing method radio options
    assert '"standard"' in content
    assert '"fast"' in content
    assert "Update Mode" in content

    print("PASS: Server has indexing method options")
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


def test_graphrag_ui_handlers_section():
    """Test that server has GraphRAG UI handlers section."""
    server_path = Path("./simple_rag_server.py")
    if not server_path.exists():
        print("SKIP: simple_rag_server.py not found")
        return True

    content = server_path.read_text()

    assert "GRAPHRAG UI HANDLERS" in content

    print("PASS: Server has GraphRAG UI handlers section")
    return True


def run_all_tests():
    """Run all Phase 6 tests."""
    print("=" * 60)
    print("Phase 6 Tests: Admin UI Integration")
    print("=" * 60)
    print()

    tests = [
        test_server_has_graphrag_tab,
        test_server_has_graphrag_query_tab,
        test_server_has_graphrag_indexing_tab,
        test_server_has_graphrag_entities_tab,
        test_server_has_graphrag_statistics_tab,
        test_server_has_graphrag_query_handler,
        test_server_has_graphrag_indexing_handler,
        test_server_has_graphrag_entity_handler,
        test_server_has_graphrag_stats_handler,
        test_server_has_graphrag_verify_handler,
        test_server_has_search_method_options,
        test_server_has_entity_type_options,
        test_server_has_indexing_method_options,
        test_server_syntax,
        test_graphrag_ui_handlers_section,
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
        print("Phase 6 COMPLETE - Ready for Phase 7")
        return 0
    else:
        print("Phase 6 INCOMPLETE - Fix failing tests")
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
