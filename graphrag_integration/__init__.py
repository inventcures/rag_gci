"""
GraphRAG Integration Module for Palli Sahayak

This module provides integration with Microsoft GraphRAG for enhanced
retrieval-augmented generation capabilities in palliative care knowledge access.

Components:
    - GraphRAGConfig: Configuration management
    - GraphRAGIndexer: Document indexing pipeline
    - GraphRAGQueryEngine: Query execution (Global, Local, DRIFT)
    - GraphRAGDataLoader: Parquet file management
    - QueryCache: LRU cache for query results
    - BatchProcessor: Batch processing utilities
    - AsyncThrottler: Async rate limiting
    - MemoryManager: Memory optimization utilities

Usage:
    from graphrag_integration import (
        GraphRAGConfig,
        GraphRAGIndexer,
        GraphRAGQueryEngine,
        QueryCache,
    )

    # Initialize
    config = GraphRAGConfig.from_yaml("./data/graphrag/settings.yaml")
    indexer = GraphRAGIndexer(config)
    query_engine = GraphRAGQueryEngine(config)

    # Index documents
    await indexer.index_documents()

    # Query with caching
    cache = QueryCache(maxsize=100)
    result = await query_engine.global_search("What are pain management options?")
    cache.set("What are pain management options?", "global", result)
"""

# Lazy imports to avoid circular dependencies and missing module errors during setup
def __getattr__(name):
    """Lazy import of module components."""
    if name == "GraphRAGConfig":
        from graphrag_integration.config import GraphRAGConfig
        return GraphRAGConfig
    elif name == "GraphRAGIndexer":
        from graphrag_integration.indexer import GraphRAGIndexer
        return GraphRAGIndexer
    elif name == "GraphRAGQueryEngine":
        from graphrag_integration.query_engine import GraphRAGQueryEngine
        return GraphRAGQueryEngine
    elif name == "GraphRAGDataLoader":
        from graphrag_integration.data_loader import GraphRAGDataLoader
        return GraphRAGDataLoader
    elif name == "QueryCache":
        from graphrag_integration.utils import QueryCache
        return QueryCache
    elif name == "BatchProcessor":
        from graphrag_integration.utils import BatchProcessor
        return BatchProcessor
    elif name == "AsyncThrottler":
        from graphrag_integration.utils import AsyncThrottler
        return AsyncThrottler
    elif name == "MemoryManager":
        from graphrag_integration.utils import MemoryManager
        return MemoryManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GraphRAGConfig",
    "GraphRAGIndexer",
    "GraphRAGQueryEngine",
    "GraphRAGDataLoader",
    "QueryCache",
    "BatchProcessor",
    "AsyncThrottler",
    "MemoryManager",
]

__version__ = "1.0.0"
