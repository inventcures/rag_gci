"""
PageIndex-Style RAG Integration Module for Palli Sahayak

Vectorless, reasoning-based retrieval using hierarchical tree indexes.
Documents are parsed into tree structures, and LLMs reason over node
summaries to find relevant sections — no vector database required.

Components:
    - PageIndexConfig: Configuration management
    - PageIndexTreeBuilder: Document -> tree indexing
    - PageIndexQueryEngine: Tree-based search and retrieval
    - PageIndexStorage: Tree JSON persistence
    - LLMAdapter: Groq/OpenAI LLM interface

Usage:
    from pageindex_integration import (
        PageIndexConfig,
        PageIndexTreeBuilder,
        PageIndexQueryEngine,
        PageIndexStorage,
    )

    config = PageIndexConfig()
    storage = PageIndexStorage(config)
    builder = PageIndexTreeBuilder(config, storage)
    engine = PageIndexQueryEngine(config, storage)

    await builder.build_tree("/path/to/document.pdf", "doc_id_123")
    result = await engine.search("What are opioid dosing guidelines?")
"""


def __getattr__(name):
    """Lazy import of module components."""
    if name == "PageIndexConfig":
        from pageindex_integration.config import PageIndexConfig
        return PageIndexConfig
    elif name == "PageIndexTreeBuilder":
        from pageindex_integration.tree_builder import PageIndexTreeBuilder
        return PageIndexTreeBuilder
    elif name == "PageIndexQueryEngine":
        from pageindex_integration.query_engine import PageIndexQueryEngine
        return PageIndexQueryEngine
    elif name == "PageIndexStorage":
        from pageindex_integration.storage import PageIndexStorage
        return PageIndexStorage
    elif name == "LLMAdapter":
        from pageindex_integration.llm_adapter import LLMAdapter
        return LLMAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PageIndexConfig",
    "PageIndexTreeBuilder",
    "PageIndexQueryEngine",
    "PageIndexStorage",
    "LLMAdapter",
]

__version__ = "1.0.0"
