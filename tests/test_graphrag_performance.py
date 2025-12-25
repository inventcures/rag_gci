"""
Phase 8 Tests: Performance Optimization

Run with: pytest tests/test_graphrag_performance.py -v

Tests for caching, batch processing, async throttling, and memory management.
"""

import pytest
import asyncio
import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def query_cache():
    """Create QueryCache instance."""
    from graphrag_integration.utils import QueryCache
    return QueryCache(maxsize=10, ttl_seconds=60)


@pytest.fixture
def batch_processor():
    """Create BatchProcessor instance."""
    from graphrag_integration.utils import BatchProcessor
    return BatchProcessor(batch_size=3, max_concurrent=2)


@pytest.fixture
def async_throttler():
    """Create AsyncThrottler instance."""
    from graphrag_integration.utils import AsyncThrottler
    return AsyncThrottler(max_concurrent=3)


@pytest.fixture
def memory_manager():
    """Create MemoryManager instance."""
    from graphrag_integration.utils import MemoryManager
    return MemoryManager(threshold_mb=500)


@pytest.fixture
def graphrag_config():
    """Create GraphRAG configuration."""
    from graphrag_integration.config import GraphRAGConfig
    return GraphRAGConfig(root_dir="./data/graphrag")


# =============================================================================
# QUERY CACHE TESTS
# =============================================================================

class TestQueryCache:
    """Query cache tests."""

    def test_cache_creation(self, query_cache):
        """Test cache creation."""
        assert query_cache is not None
        assert len(query_cache) == 0

    def test_cache_set_get(self, query_cache):
        """Test setting and getting cache values."""
        query_cache.set("test query", "local", {"result": "test"})

        result = query_cache.get("test query", "local")
        assert result is not None
        assert result["result"] == "test"

    def test_cache_miss(self, query_cache):
        """Test cache miss."""
        result = query_cache.get("nonexistent", "local")
        assert result is None

    def test_cache_case_insensitivity(self, query_cache):
        """Test that queries are case-insensitive."""
        query_cache.set("Test Query", "local", {"result": "test"})

        result = query_cache.get("test query", "local")
        assert result is not None

    def test_cache_method_differentiation(self, query_cache):
        """Test that different methods have different cache entries."""
        query_cache.set("query", "local", {"method": "local"})
        query_cache.set("query", "global", {"method": "global"})

        local_result = query_cache.get("query", "local")
        global_result = query_cache.get("query", "global")

        assert local_result["method"] == "local"
        assert global_result["method"] == "global"

    def test_cache_lru_eviction(self):
        """Test LRU eviction when maxsize exceeded."""
        from graphrag_integration.utils import QueryCache

        cache = QueryCache(maxsize=3)

        cache.set("query1", "local", {"id": 1})
        cache.set("query2", "local", {"id": 2})
        cache.set("query3", "local", {"id": 3})

        # Access query1 to make it recently used
        cache.get("query1", "local")

        # Add query4, should evict query2 (least recently used)
        cache.set("query4", "local", {"id": 4})

        assert cache.get("query1", "local") is not None
        assert cache.get("query2", "local") is None  # Evicted
        assert cache.get("query3", "local") is not None
        assert cache.get("query4", "local") is not None

    def test_cache_invalidate(self, query_cache):
        """Test cache invalidation."""
        query_cache.set("query", "local", {"result": "test"})
        assert query_cache.get("query", "local") is not None

        result = query_cache.invalidate("query", "local")
        assert result is True
        assert query_cache.get("query", "local") is None

    def test_cache_clear(self, query_cache):
        """Test clearing cache."""
        query_cache.set("query1", "local", {"id": 1})
        query_cache.set("query2", "local", {"id": 2})

        query_cache.clear()

        assert len(query_cache) == 0
        assert query_cache.get("query1", "local") is None

    def test_cache_stats(self, query_cache):
        """Test cache statistics."""
        query_cache.set("query", "local", {"result": "test"})
        query_cache.get("query", "local")  # Hit
        query_cache.get("nonexistent", "local")  # Miss

        stats = query_cache.get_stats()

        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_cache_repr(self, query_cache):
        """Test cache string representation."""
        repr_str = repr(query_cache)
        assert "QueryCache" in repr_str
        assert "hit_rate" in repr_str

    @pytest.mark.asyncio
    async def test_cache_async_operations(self, query_cache):
        """Test async cache operations."""
        await query_cache.set_async("async query", "local", {"async": True})
        result = await query_cache.get_async("async query", "local")

        assert result is not None
        assert result["async"] is True


# =============================================================================
# BATCH PROCESSOR TESTS
# =============================================================================

class TestBatchProcessor:
    """Batch processor tests."""

    def test_processor_creation(self, batch_processor):
        """Test batch processor creation."""
        assert batch_processor is not None
        assert batch_processor.batch_size == 3

    @pytest.mark.asyncio
    async def test_batch_processing(self, batch_processor):
        """Test batch processing of items."""
        items = [1, 2, 3, 4, 5, 6, 7]

        async def process_item(item):
            return item * 2

        result = await batch_processor.process(items, process_item)

        assert result.success_count == 7
        assert result.error_count == 0
        assert sorted(result.results) == [2, 4, 6, 8, 10, 12, 14]

    @pytest.mark.asyncio
    async def test_batch_processing_with_errors(self, batch_processor):
        """Test batch processing with some failures."""
        items = [1, 2, 3, 4, 5]

        async def process_item(item):
            if item == 3:
                raise ValueError("Error on item 3")
            return item * 2

        result = await batch_processor.process(items, process_item)

        assert result.success_count == 4
        assert result.error_count == 1

    @pytest.mark.asyncio
    async def test_batch_processing_progress(self, batch_processor):
        """Test progress callback during batch processing."""
        items = [1, 2, 3, 4, 5, 6]
        progress_values = []

        async def process_item(item):
            return item

        def on_progress(progress):
            progress_values.append(progress)

        await batch_processor.process(items, process_item, on_progress=on_progress)

        # Should have progress updates (2 batches for 6 items with batch_size=3)
        assert len(progress_values) == 2
        assert progress_values[-1] == 100

    @pytest.mark.asyncio
    async def test_batch_processing_duration(self, batch_processor):
        """Test that duration is tracked."""
        items = [1, 2, 3]

        async def process_item(item):
            await asyncio.sleep(0.01)
            return item

        result = await batch_processor.process(items, process_item)

        assert result.duration_seconds > 0

    def test_batch_size_setter(self, batch_processor):
        """Test setting batch size."""
        batch_processor.batch_size = 5
        assert batch_processor.batch_size == 5

    def test_batch_size_validation(self, batch_processor):
        """Test batch size validation."""
        with pytest.raises(ValueError):
            batch_processor.batch_size = 0

    def test_processor_repr(self, batch_processor):
        """Test processor string representation."""
        repr_str = repr(batch_processor)
        assert "BatchProcessor" in repr_str
        assert "batch_size" in repr_str


# =============================================================================
# ASYNC THROTTLER TESTS
# =============================================================================

class TestAsyncThrottler:
    """Async throttler tests."""

    def test_throttler_creation(self, async_throttler):
        """Test throttler creation."""
        assert async_throttler is not None

    @pytest.mark.asyncio
    async def test_throttler_context_manager(self, async_throttler):
        """Test throttler as context manager."""
        async with async_throttler:
            # Should not raise
            pass

    @pytest.mark.asyncio
    async def test_throttler_concurrency_limit(self):
        """Test that throttler limits concurrency."""
        from graphrag_integration.utils import AsyncThrottler

        throttler = AsyncThrottler(max_concurrent=2)
        concurrent_count = 0
        max_concurrent = 0

        async def task():
            nonlocal concurrent_count, max_concurrent
            async with throttler:
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)
                await asyncio.sleep(0.05)
                concurrent_count -= 1

        # Run 5 tasks, should never exceed 2 concurrent
        await asyncio.gather(*[task() for _ in range(5)])

        assert max_concurrent <= 2

    def test_throttler_stats(self, async_throttler):
        """Test throttler statistics."""
        stats = async_throttler.get_stats()

        assert "max_concurrent" in stats
        assert "request_count" in stats
        assert stats["max_concurrent"] == 3

    def test_throttler_repr(self, async_throttler):
        """Test throttler string representation."""
        repr_str = repr(async_throttler)
        assert "AsyncThrottler" in repr_str
        assert "max_concurrent" in repr_str


# =============================================================================
# MEMORY MANAGER TESTS
# =============================================================================

class TestMemoryManager:
    """Memory manager tests."""

    def test_manager_creation(self, memory_manager):
        """Test memory manager creation."""
        assert memory_manager is not None

    def test_get_memory_usage(self, memory_manager):
        """Test getting memory usage."""
        usage_mb = memory_manager.get_memory_usage_mb()
        assert usage_mb > 0

    def test_should_cleanup(self, memory_manager):
        """Test cleanup check."""
        result = memory_manager.should_cleanup()
        assert isinstance(result, bool)

    def test_cleanup(self, memory_manager):
        """Test garbage collection cleanup."""
        collected = memory_manager.cleanup()
        assert collected >= 0

    def test_get_memory_report(self, memory_manager):
        """Test getting memory report."""
        report = memory_manager.get_memory_report()

        assert "current_mb" in report
        assert "threshold_mb" in report
        assert "above_threshold" in report
        assert "cleanup_count" in report

    def test_manager_repr(self, memory_manager):
        """Test manager string representation."""
        repr_str = repr(memory_manager)
        assert "MemoryManager" in repr_str
        assert "threshold_mb" in repr_str


# =============================================================================
# TIMER TESTS
# =============================================================================

class TestTimer:
    """Timer utility tests."""

    def test_timer_context_manager(self):
        """Test timer as context manager."""
        from graphrag_integration.utils import Timer

        with Timer() as t:
            time.sleep(0.01)

        assert t.elapsed > 0
        assert t.elapsed_ms > 0

    def test_timer_elapsed(self):
        """Test elapsed time measurement."""
        from graphrag_integration.utils import Timer

        with Timer() as t:
            time.sleep(0.05)

        # Should be around 50ms
        assert 0.04 < t.elapsed < 0.2


# =============================================================================
# QUERY ENGINE CACHING INTEGRATION TESTS
# =============================================================================

class TestQueryEngineCaching:
    """Tests for query engine caching integration."""

    def test_query_engine_cache_enabled(self, graphrag_config):
        """Test that query engine has cache enabled by default."""
        from graphrag_integration.query_engine import GraphRAGQueryEngine

        engine = GraphRAGQueryEngine(graphrag_config)
        assert engine._cache_enabled is True
        assert engine._cache is not None

    def test_query_engine_cache_disabled(self, graphrag_config):
        """Test creating query engine with cache disabled."""
        from graphrag_integration.query_engine import GraphRAGQueryEngine

        engine = GraphRAGQueryEngine(graphrag_config, enable_cache=False)
        assert engine._cache_enabled is False

    def test_query_engine_enable_cache(self, graphrag_config):
        """Test enabling cache on query engine."""
        from graphrag_integration.query_engine import GraphRAGQueryEngine

        engine = GraphRAGQueryEngine(graphrag_config, enable_cache=False)
        engine.enable_cache(maxsize=50, ttl_seconds=1800)

        assert engine._cache_enabled is True
        assert engine._cache is not None

    def test_query_engine_disable_cache(self, graphrag_config):
        """Test disabling cache on query engine."""
        from graphrag_integration.query_engine import GraphRAGQueryEngine

        engine = GraphRAGQueryEngine(graphrag_config)
        engine.disable_cache()

        assert engine._cache_enabled is False

    def test_query_engine_clear_cache(self, graphrag_config):
        """Test clearing query engine cache."""
        from graphrag_integration.query_engine import GraphRAGQueryEngine

        engine = GraphRAGQueryEngine(graphrag_config)
        engine._cache.set("test", "local", {"result": "test"})
        engine.clear_cache()

        assert len(engine._cache) == 0

    def test_query_engine_cache_stats(self, graphrag_config):
        """Test query engine cache statistics."""
        from graphrag_integration.query_engine import GraphRAGQueryEngine

        engine = GraphRAGQueryEngine(graphrag_config)
        stats = engine.get_cache_stats()

        assert "enabled" in stats
        assert "size" in stats
        assert stats["enabled"] is True

    def test_query_engine_performance_stats(self, graphrag_config):
        """Test query engine performance statistics."""
        from graphrag_integration.query_engine import GraphRAGQueryEngine

        engine = GraphRAGQueryEngine(graphrag_config)
        stats = engine.get_performance_stats()

        assert "query_count" in stats
        assert "cache_hits" in stats
        assert "cache_hit_rate" in stats
        assert "total_query_time" in stats
        assert "avg_query_time" in stats

    @pytest.mark.asyncio
    async def test_query_engine_cache_hit(self, graphrag_config):
        """Test cache hit on repeated query."""
        from graphrag_integration.query_engine import GraphRAGQueryEngine

        engine = GraphRAGQueryEngine(graphrag_config)

        # First query - cache miss
        result1 = await engine.local_search("What is morphine?")

        # Second query - should be cache hit
        result2 = await engine.local_search("What is morphine?")

        stats = engine.get_performance_stats()
        assert stats["cache_hits"] >= 1


# =============================================================================
# INDEXER BATCH PROCESSING INTEGRATION TESTS
# =============================================================================

class TestIndexerBatchProcessing:
    """Tests for indexer batch processing integration."""

    def test_indexer_has_batch_processor(self, graphrag_config):
        """Test that indexer has batch processor."""
        from graphrag_integration.indexer import GraphRAGIndexer

        indexer = GraphRAGIndexer(graphrag_config)
        assert indexer._batch_processor is not None

    def test_indexer_has_memory_manager(self, graphrag_config):
        """Test that indexer has memory manager."""
        from graphrag_integration.indexer import GraphRAGIndexer

        indexer = GraphRAGIndexer(graphrag_config)
        assert indexer._memory_manager is not None

    def test_indexer_set_batch_size(self, graphrag_config):
        """Test setting batch size on indexer."""
        from graphrag_integration.indexer import GraphRAGIndexer

        indexer = GraphRAGIndexer(graphrag_config, batch_size=5)
        assert indexer._batch_processor.batch_size == 5

        indexer.set_batch_size(10)
        assert indexer._batch_processor.batch_size == 10

    def test_indexer_get_performance_stats(self, graphrag_config):
        """Test getting performance stats from indexer."""
        from graphrag_integration.indexer import GraphRAGIndexer

        indexer = GraphRAGIndexer(graphrag_config)
        stats = indexer.get_performance_stats()

        assert "documents_processed" in stats
        assert "batch_size" in stats
        assert "memory_report" in stats

    def test_indexer_trigger_memory_cleanup(self, graphrag_config):
        """Test triggering memory cleanup on indexer."""
        from graphrag_integration.indexer import GraphRAGIndexer

        indexer = GraphRAGIndexer(graphrag_config)
        collected = indexer.trigger_memory_cleanup()

        assert collected >= 0

    @pytest.mark.asyncio
    async def test_indexer_batch_process_documents(self, graphrag_config):
        """Test batch processing documents."""
        from graphrag_integration.indexer import GraphRAGIndexer

        indexer = GraphRAGIndexer(graphrag_config, batch_size=2)

        documents = ["doc1.txt", "doc2.txt", "doc3.txt"]

        async def mock_processor(doc):
            return {"doc": doc, "processed": True}

        result = await indexer.batch_process_documents(
            documents=documents,
            processor=mock_processor
        )

        assert result["success_count"] == 3
        assert result["error_count"] == 0


# =============================================================================
# MODULE EXPORTS TESTS
# =============================================================================

class TestModuleExports:
    """Test module exports for Phase 8."""

    def test_query_cache_export(self):
        """Test QueryCache is exported."""
        from graphrag_integration import QueryCache
        assert QueryCache is not None

    def test_batch_processor_export(self):
        """Test BatchProcessor is exported."""
        from graphrag_integration import BatchProcessor
        assert BatchProcessor is not None

    def test_async_throttler_export(self):
        """Test AsyncThrottler is exported."""
        from graphrag_integration import AsyncThrottler
        assert AsyncThrottler is not None

    def test_memory_manager_export(self):
        """Test MemoryManager is exported."""
        from graphrag_integration import MemoryManager
        assert MemoryManager is not None


# =============================================================================
# MAIN RUNNER
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
