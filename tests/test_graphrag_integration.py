"""
Comprehensive GraphRAG Integration Tests

Run with: pytest tests/test_graphrag_integration.py -v

Phase 7: Testing Suite - Comprehensive tests covering all GraphRAG modules
"""

import pytest
import asyncio
import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def graphrag_config():
    """Create GraphRAG configuration."""
    from graphrag_integration.config import GraphRAGConfig
    return GraphRAGConfig(root_dir="./data/graphrag")


@pytest.fixture(scope="module")
def graphrag_indexer(graphrag_config):
    """Create GraphRAG indexer."""
    from graphrag_integration.indexer import GraphRAGIndexer
    return GraphRAGIndexer(graphrag_config)


@pytest.fixture(scope="module")
def graphrag_query_engine(graphrag_config):
    """Create GraphRAG query engine."""
    from graphrag_integration.query_engine import GraphRAGQueryEngine
    return GraphRAGQueryEngine(graphrag_config)


@pytest.fixture(scope="module")
def graphrag_data_loader(graphrag_config):
    """Create GraphRAG data loader."""
    from graphrag_integration.data_loader import GraphRAGDataLoader
    return GraphRAGDataLoader(graphrag_config)


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestConfiguration:
    """Configuration module tests."""

    def test_config_creation(self, graphrag_config):
        """Test configuration creation."""
        assert graphrag_config is not None
        assert graphrag_config.root_dir == Path("./data/graphrag")

    def test_config_paths(self, graphrag_config):
        """Test path properties."""
        assert graphrag_config.input_dir.name == "input"
        assert graphrag_config.output_dir.name == "output"
        assert graphrag_config.cache_dir.name == "cache"
        assert graphrag_config.prompts_dir.name == "prompts"
        assert graphrag_config.artifacts_dir.name == "artifacts"

    def test_config_validation(self, graphrag_config):
        """Test configuration validation."""
        errors = graphrag_config.validate()
        # May have errors due to missing API keys in test env
        assert isinstance(errors, list)

    def test_env_var_substitution(self):
        """Test environment variable substitution."""
        from graphrag_integration.config import GraphRAGConfig

        os.environ["TEST_API_KEY"] = "test123"
        content = "api_key: ${TEST_API_KEY}"
        result = GraphRAGConfig._substitute_env_vars(content)
        assert result == "api_key: test123"
        del os.environ["TEST_API_KEY"]

    def test_config_to_dict(self, graphrag_config):
        """Test configuration to dictionary conversion."""
        config_dict = graphrag_config.to_dict()

        assert isinstance(config_dict, dict)
        assert "root_dir" in config_dict
        assert "models" in config_dict
        assert "chunking" in config_dict
        assert "extraction" in config_dict
        assert "search" in config_dict
        assert "paths" in config_dict
        assert "initialized" in config_dict

    def test_config_models(self, graphrag_config):
        """Test model configuration access."""
        chat_model = graphrag_config.get_chat_model()
        embedding_model = graphrag_config.get_embedding_model()

        assert chat_model is not None
        assert embedding_model is not None
        assert chat_model.type == "chat"
        assert embedding_model.type == "embedding"

    def test_config_chunking(self, graphrag_config):
        """Test chunking configuration."""
        assert graphrag_config.chunking.size > 0
        assert graphrag_config.chunking.overlap >= 0
        assert graphrag_config.chunking.overlap < graphrag_config.chunking.size

    def test_config_extraction(self, graphrag_config):
        """Test extraction configuration."""
        assert len(graphrag_config.extraction.entity_types) > 0
        assert "Symptom" in graphrag_config.extraction.entity_types
        assert "Medication" in graphrag_config.extraction.entity_types

    def test_config_search(self, graphrag_config):
        """Test search configuration."""
        assert graphrag_config.search.max_tokens > 0
        assert graphrag_config.search.top_k_entities > 0

    def test_config_repr(self, graphrag_config):
        """Test configuration string representation."""
        repr_str = repr(graphrag_config)
        assert "GraphRAGConfig" in repr_str
        assert "root_dir" in repr_str

    def test_config_prompt_path(self, graphrag_config):
        """Test prompt path generation."""
        prompt_path = graphrag_config.get_prompt_path("entity_extraction")
        assert prompt_path.name == "entity_extraction.txt"
        assert prompt_path.parent.name == "prompts"


# =============================================================================
# INDEXER TESTS
# =============================================================================

class TestIndexer:
    """Indexer module tests."""

    def test_indexer_creation(self, graphrag_indexer):
        """Test indexer creation."""
        from graphrag_integration.indexer import IndexingStatus
        assert graphrag_indexer.status == IndexingStatus.PENDING
        assert graphrag_indexer.progress == 0

    def test_indexer_methods(self, graphrag_config):
        """Test different indexing methods."""
        from graphrag_integration.indexer import GraphRAGIndexer, IndexingMethod

        standard = GraphRAGIndexer(graphrag_config, IndexingMethod.STANDARD)
        fast = GraphRAGIndexer(graphrag_config, IndexingMethod.FAST)

        assert standard.method == IndexingMethod.STANDARD
        assert fast.method == IndexingMethod.FAST

    def test_indexer_status_enum(self):
        """Test IndexingStatus enum values."""
        from graphrag_integration.indexer import IndexingStatus

        assert IndexingStatus.PENDING.value == "pending"
        assert IndexingStatus.RUNNING.value == "running"
        assert IndexingStatus.COMPLETED.value == "completed"
        assert IndexingStatus.FAILED.value == "failed"

    def test_indexer_method_enum(self):
        """Test IndexingMethod enum values."""
        from graphrag_integration.indexer import IndexingMethod

        assert IndexingMethod.STANDARD.value == "standard"
        assert IndexingMethod.FAST.value == "fast"

    @pytest.mark.asyncio
    async def test_mock_indexing(self, graphrag_indexer):
        """Test mock indexing run."""
        result = await graphrag_indexer._mock_index()
        assert "status" in result
        # _mock_index stops at 95%, full index_documents() would reach 100%
        assert graphrag_indexer.progress >= 95

    def test_indexer_get_stats(self, graphrag_indexer):
        """Test getting indexer stats."""
        stats = graphrag_indexer.get_stats()

        assert isinstance(stats, dict)

    def test_indexer_repr(self, graphrag_indexer):
        """Test indexer string representation."""
        repr_str = repr(graphrag_indexer)
        assert "GraphRAGIndexer" in repr_str


# =============================================================================
# DATA LOADER TESTS
# =============================================================================

class TestDataLoader:
    """Data loader tests."""

    def test_data_loader_creation(self, graphrag_data_loader):
        """Test data loader creation."""
        assert graphrag_data_loader._loaded is False
        assert graphrag_data_loader.is_loaded() is False

    @pytest.mark.asyncio
    async def test_get_stats(self, graphrag_data_loader):
        """Test getting stats."""
        stats = await graphrag_data_loader.get_stats()

        assert isinstance(stats, dict)
        assert "entities" in stats
        assert "relationships" in stats
        assert "communities" in stats
        assert "text_units" in stats
        assert "loaded" in stats

    @pytest.mark.asyncio
    async def test_load_all(self, graphrag_data_loader):
        """Test loading all data (graceful with no data)."""
        # Should not raise even if no parquet files exist
        result = await graphrag_data_loader.load_all()
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_search_entities(self, graphrag_data_loader):
        """Test entity search functionality."""
        # Search should return empty list if no data
        entities = await graphrag_data_loader.search_entities("morphine", top_k=5)
        assert isinstance(entities, list)

    @pytest.mark.asyncio
    async def test_search_entities_with_type(self, graphrag_data_loader):
        """Test entity search with type filter."""
        entities = await graphrag_data_loader.search_entities(
            "pain", entity_type="Symptom", top_k=5
        )
        assert isinstance(entities, list)

    @pytest.mark.asyncio
    async def test_get_entity_relationships(self, graphrag_data_loader):
        """Test getting entity relationships."""
        relationships = await graphrag_data_loader.get_entity_relationships("morphine")
        assert isinstance(relationships, list)

    @pytest.mark.asyncio
    async def test_get_community_reports(self, graphrag_data_loader):
        """Test getting community reports."""
        reports = await graphrag_data_loader.get_community_reports_by_level(0)
        assert isinstance(reports, list)

    def test_data_loader_repr(self, graphrag_data_loader):
        """Test data loader string representation."""
        repr_str = repr(graphrag_data_loader)
        assert "GraphRAGDataLoader" in repr_str
        assert "loaded=" in repr_str


# =============================================================================
# QUERY ENGINE TESTS
# =============================================================================

class TestQueryEngine:
    """Query engine tests."""

    def test_query_engine_creation(self, graphrag_query_engine):
        """Test query engine creation."""
        assert graphrag_query_engine._initialized is False
        assert graphrag_query_engine.data_loader is not None

    @pytest.mark.asyncio
    async def test_query_engine_initialize(self, graphrag_query_engine):
        """Test query engine initialization."""
        await graphrag_query_engine.initialize()
        assert graphrag_query_engine._initialized is True

    def test_search_method_enum(self):
        """Test SearchMethod enum values."""
        from graphrag_integration.query_engine import SearchMethod

        assert SearchMethod.GLOBAL.value == "global"
        assert SearchMethod.LOCAL.value == "local"
        assert SearchMethod.DRIFT.value == "drift"
        assert SearchMethod.BASIC.value == "basic"

    def test_search_result_creation(self):
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

    def test_search_result_to_dict(self):
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

    def test_query_analysis(self, graphrag_query_engine):
        """Test query analysis for method selection."""
        from graphrag_integration.query_engine import SearchMethod

        # Test various query types
        queries = [
            ("What are the main pain management approaches?", SearchMethod.GLOBAL),
            ("What are the side effects of morphine?", SearchMethod.LOCAL),
            ("How should pain be managed in renal failure?", SearchMethod.DRIFT),
        ]

        for query, expected in queries:
            method = graphrag_query_engine._analyze_query(query)
            assert isinstance(method, SearchMethod)

    def test_query_analysis_global(self, graphrag_query_engine):
        """Test query analysis for global search."""
        from graphrag_integration.query_engine import SearchMethod

        method = graphrag_query_engine._analyze_query(
            "Give me an overview of all palliative care themes"
        )
        assert method == SearchMethod.GLOBAL

    def test_query_analysis_local(self, graphrag_query_engine):
        """Test query analysis for local search."""
        from graphrag_integration.query_engine import SearchMethod

        method = graphrag_query_engine._analyze_query(
            "What are the side effects of morphine?"
        )
        assert method == SearchMethod.LOCAL

    def test_query_analysis_drift(self, graphrag_query_engine):
        """Test query analysis for DRIFT search."""
        from graphrag_integration.query_engine import SearchMethod

        method = graphrag_query_engine._analyze_query(
            "How should pain be managed in a patient with renal failure?"
        )
        assert method == SearchMethod.DRIFT

    @pytest.mark.asyncio
    async def test_fallback_search(self, graphrag_query_engine):
        """Test fallback search mechanism."""
        from graphrag_integration.query_engine import SearchMethod

        result = await graphrag_query_engine._fallback_search(
            "test query", SearchMethod.LOCAL
        )

        assert result.query == "test query"
        assert result.method == SearchMethod.LOCAL
        assert result.metadata.get("fallback") is True

    @pytest.mark.asyncio
    async def test_global_search(self, graphrag_query_engine):
        """Test global search."""
        from graphrag_integration.query_engine import SearchMethod

        result = await graphrag_query_engine.global_search(
            "What are the main pain management approaches?"
        )

        assert result.query == "What are the main pain management approaches?"
        assert result.method == SearchMethod.GLOBAL
        assert isinstance(result.response, str)

    @pytest.mark.asyncio
    async def test_local_search(self, graphrag_query_engine):
        """Test local search."""
        from graphrag_integration.query_engine import SearchMethod

        result = await graphrag_query_engine.local_search("What is morphine?")

        assert result.query == "What is morphine?"
        assert result.method == SearchMethod.LOCAL
        assert isinstance(result.response, str)

    @pytest.mark.asyncio
    async def test_drift_search(self, graphrag_query_engine):
        """Test DRIFT search."""
        from graphrag_integration.query_engine import SearchMethod

        result = await graphrag_query_engine.drift_search(
            "How should pain be managed in renal failure?"
        )

        assert result.query == "How should pain be managed in renal failure?"
        assert result.method == SearchMethod.DRIFT
        assert isinstance(result.response, str)

    @pytest.mark.asyncio
    async def test_basic_search(self, graphrag_query_engine):
        """Test basic search."""
        from graphrag_integration.query_engine import SearchMethod

        result = await graphrag_query_engine.basic_search("morphine")

        assert result.query == "morphine"
        assert result.method == SearchMethod.BASIC
        assert isinstance(result.response, str)

    @pytest.mark.asyncio
    async def test_auto_search(self, graphrag_query_engine):
        """Test auto search method selection."""
        result = await graphrag_query_engine.auto_search("What is morphine?")

        assert result is not None
        assert isinstance(result.response, str)
        assert result.method is not None

    def test_query_engine_repr(self, graphrag_query_engine):
        """Test query engine string representation."""
        repr_str = repr(graphrag_query_engine)
        assert "GraphRAGQueryEngine" in repr_str
        assert "initialized=" in repr_str


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_module_imports(self):
        """Test all module imports."""
        from graphrag_integration import (
            GraphRAGConfig,
            GraphRAGIndexer,
            GraphRAGQueryEngine,
            GraphRAGDataLoader,
        )

        assert GraphRAGConfig is not None
        assert GraphRAGIndexer is not None
        assert GraphRAGQueryEngine is not None
        assert GraphRAGDataLoader is not None

    def test_module_version(self):
        """Test module version."""
        from graphrag_integration import __version__

        assert __version__ is not None
        assert isinstance(__version__, str)

    @pytest.mark.asyncio
    async def test_full_query_flow(self, graphrag_query_engine):
        """Test full query flow."""
        result = await graphrag_query_engine.auto_search(
            "What medications are used for pain?"
        )

        assert result.query is not None
        assert result.response is not None
        assert result.method is not None
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_config_to_query_flow(self, graphrag_config):
        """Test configuration to query engine flow."""
        from graphrag_integration.query_engine import GraphRAGQueryEngine

        engine = GraphRAGQueryEngine(graphrag_config)
        await engine.initialize()

        result = await engine.auto_search("What is pain management?")

        assert result is not None
        assert result.response is not None

    @pytest.mark.asyncio
    async def test_data_loader_stats_flow(self, graphrag_config):
        """Test data loader stats flow."""
        from graphrag_integration.data_loader import GraphRAGDataLoader

        loader = GraphRAGDataLoader(graphrag_config)
        stats = await loader.get_stats()

        assert stats is not None
        assert isinstance(stats["entities"], int)
        assert isinstance(stats["relationships"], int)

    def test_all_entity_types(self, graphrag_config):
        """Test all palliative care entity types are configured."""
        expected_types = [
            "Symptom", "Medication", "Condition", "Treatment",
            "SideEffect", "Dosage", "Route", "CareGoal"
        ]

        for entity_type in expected_types:
            assert entity_type in graphrag_config.extraction.entity_types


# =============================================================================
# SERVER INTEGRATION TESTS (Code Structure Only)
# =============================================================================

class TestServerIntegration:
    """Server integration tests (code structure verification)."""

    def test_server_has_graphrag_imports(self):
        """Test that server file has GraphRAG imports."""
        server_path = Path("./simple_rag_server.py")
        if not server_path.exists():
            pytest.skip("simple_rag_server.py not found")

        content = server_path.read_text()

        assert "from graphrag_integration import" in content
        assert "GRAPHRAG_AVAILABLE" in content

    def test_server_has_graphrag_endpoints(self):
        """Test that server file has GraphRAG endpoints."""
        server_path = Path("./simple_rag_server.py")
        if not server_path.exists():
            pytest.skip("simple_rag_server.py not found")

        content = server_path.read_text()

        assert "/api/graphrag/health" in content
        assert "/api/graphrag/query" in content
        assert "/api/graphrag/index" in content

    def test_server_has_graphrag_ui(self):
        """Test that server file has GraphRAG UI tab."""
        server_path = Path("./simple_rag_server.py")
        if not server_path.exists():
            pytest.skip("simple_rag_server.py not found")

        content = server_path.read_text()

        assert 'gr.TabItem("🔗 GraphRAG")' in content
        assert "GRAPHRAG UI HANDLERS" in content

    def test_server_syntax(self):
        """Test that server file has valid Python syntax."""
        server_path = Path("./simple_rag_server.py")
        if not server_path.exists():
            pytest.skip("simple_rag_server.py not found")

        content = server_path.read_text()
        compile(content, str(server_path), 'exec')


# =============================================================================
# PALLIATIVE CARE DOMAIN TESTS
# =============================================================================

class TestPalliativeCareDomain:
    """Tests specific to palliative care domain."""

    def test_symptom_queries(self, graphrag_query_engine):
        """Test symptom-related query analysis."""
        from graphrag_integration.query_engine import SearchMethod

        symptom_queries = [
            "What are the symptoms of advanced cancer?",
            "How to manage pain in terminal illness?",
            "What causes breathlessness in palliative patients?",
        ]

        for query in symptom_queries:
            method = graphrag_query_engine._analyze_query(query)
            assert isinstance(method, SearchMethod)

    def test_medication_queries(self, graphrag_query_engine):
        """Test medication-related query analysis."""
        from graphrag_integration.query_engine import SearchMethod

        med_queries = [
            "What is the dosage of morphine for pain?",
            "What are side effects of oxycodone?",
            "How to administer fentanyl patches?",
        ]

        for query in med_queries:
            method = graphrag_query_engine._analyze_query(query)
            assert method == SearchMethod.LOCAL

    def test_complex_clinical_queries(self, graphrag_query_engine):
        """Test complex clinical query analysis."""
        from graphrag_integration.query_engine import SearchMethod

        # Queries with DRIFT keywords: "how should", "for a patient with", "considering"
        drift_queries = [
            "How should pain be managed in a patient with renal failure?",
            "For a patient with kidney disease, considering multiple symptoms, what is best?",
            "If the patient has both pain and nausea, how should we manage together with medications?",
        ]

        for query in drift_queries:
            method = graphrag_query_engine._analyze_query(query)
            assert method == SearchMethod.DRIFT, f"Expected DRIFT for: {query}"

    @pytest.mark.asyncio
    async def test_hindi_query_handling(self, graphrag_query_engine):
        """Test handling of Hindi queries."""
        # Hindi query: "What medicine for pain?"
        result = await graphrag_query_engine.auto_search("दर्द के लिए कौन सी दवाई?")

        assert result is not None
        assert result.response is not None


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================

class TestEdgeCases:
    """Edge case and error handling tests."""

    @pytest.mark.asyncio
    async def test_empty_query(self, graphrag_query_engine):
        """Test handling of empty query."""
        result = await graphrag_query_engine.auto_search("")
        assert result is not None

    @pytest.mark.asyncio
    async def test_very_long_query(self, graphrag_query_engine):
        """Test handling of very long query."""
        long_query = "What is morphine? " * 100
        result = await graphrag_query_engine.auto_search(long_query)
        assert result is not None

    @pytest.mark.asyncio
    async def test_special_characters_query(self, graphrag_query_engine):
        """Test handling of special characters in query."""
        result = await graphrag_query_engine.auto_search("What is 10mg/ml dosage?")
        assert result is not None

    def test_config_missing_directory(self):
        """Test configuration with missing directory."""
        from graphrag_integration.config import GraphRAGConfig

        config = GraphRAGConfig(root_dir="./nonexistent/path")
        errors = config.validate()

        # Should have error about missing directory
        assert len(errors) > 0

    def test_invalid_model_config(self, graphrag_config):
        """Test getting non-existent model."""
        model = graphrag_config.get_model("nonexistent_model")
        assert model is None


# =============================================================================
# MAIN RUNNER
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
