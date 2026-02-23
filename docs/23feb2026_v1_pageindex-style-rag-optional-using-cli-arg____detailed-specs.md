# PageIndex-Style RAG Integration Specifications

## Palli Sahayak Voice AI Helpline - Vectorless Tree-Based Retrieval

**Version**: 1.0.0
**Document Status**: Implementation Specification
**Last Updated**: February 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Prerequisites and Dependencies](#3-prerequisites-and-dependencies)
4. [Directory Structure](#4-directory-structure)
5. [Phase 1: Foundation and LLM Adapter](#5-phase-1-foundation-and-llm-adapter)
6. [Phase 2: Storage Layer](#6-phase-2-storage-layer)
7. [Phase 3: Tree Builder (Indexing Pipeline)](#7-phase-3-tree-builder-indexing-pipeline)
8. [Phase 4: Query Engine](#8-phase-4-query-engine)
9. [Phase 5: Server Integration](#9-phase-5-server-integration)
10. [Phase 6: Admin UI Integration](#10-phase-6-admin-ui-integration)
11. [Phase 7: Testing Suite](#11-phase-7-testing-suite)
12. [API Reference](#12-api-reference)
13. [Configuration Reference](#13-configuration-reference)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Executive Summary

### 1.1 Purpose

Integrate a PageIndex-style vectorless RAG system into Palli Sahayak to enable:
- **Reasoning-based retrieval**: LLM navigates document hierarchy instead of vector similarity
- **Structure preservation**: Full sections retrieved instead of fragmented chunks
- **Explainable results**: Complete reasoning trace showing why sections were selected
- **Page-level citations**: Exact page references in source attribution

### 1.2 Key Benefits

| Benefit | Description |
|---------|-------------|
| **Higher accuracy on structured docs** | 2-3x improvement on complex medical documents vs vector-only RAG |
| **No vector DB needed** | Tree indexes are plain JSON files — no ChromaDB dependency for this path |
| **Better coherence** | Retrieves complete sections, not fragmented 1000-char chunks |
| **Explainability** | Full reasoning trace: which nodes were considered and why |
| **Page-level citations** | Exact page ranges in source attribution |

### 1.3 Integration Strategy

PageIndex operates as an **optional alternative** to the existing ChromaDB vector search, selectable via `--rag-method`:

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAG METHOD ROUTING                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  --rag-method vector (default)                                   │
│   └──► ChromaDB Vector Search ──► LLM Response                  │
│                                                                  │
│  --rag-method pageindex                                          │
│   └──► Tree Navigation (LLM) ──► Node Extraction ──► Response   │
│                                                                  │
│  --rag-method hybrid                                             │
│   ├──► ChromaDB Vector Search ──┐                                │
│   └──► Tree Navigation (LLM) ──┼──► Context Merge ──► Response  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.4 CLI Usage

```bash
# Default: vector search (existing behavior, unchanged)
python simple_rag_server.py

# PageIndex tree-based search
python simple_rag_server.py --rag-method pageindex

# Hybrid: both vector + tree in parallel
python simple_rag_server.py --rag-method hybrid

# Combined with other flags
python simple_rag_server.py --rag-method pageindex --provider bolna --port 8001
```

---

## 2. Architecture Overview

### 2.1 PageIndex Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  PAGEINDEX RAG ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 INDEXING PIPELINE                         │    │
│  │                                                          │    │
│  │  Document ──► Structure Detection ──► Tree Construction  │    │
│  │            (TOC / Headings / LLM)   (Nested JSON nodes)  │    │
│  │                      │                                    │    │
│  │                      ▼                                    │    │
│  │              LLM Summary Generation                       │    │
│  │           (per-node summaries via Groq)                   │    │
│  │                      │                                    │    │
│  │                      ▼                                    │    │
│  │              Tree JSON Persistence                        │    │
│  │           (data/pageindex/trees/)                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   QUERY PIPELINE                         │    │
│  │                                                          │    │
│  │  Query ──► Load Trees (summaries only, no full text)     │    │
│  │                      │                                    │    │
│  │                      ▼                                    │    │
│  │         LLM Tree Navigation (Step 1)                      │    │
│  │    "Given this tree structure and query,                  │    │
│  │     which nodes contain relevant information?"            │    │
│  │              │                                            │    │
│  │              ▼                                            │    │
│  │     Extract Full Text from Selected Nodes (Step 2)        │    │
│  │              │                                            │    │
│  │              ▼                                            │    │
│  │     Feed Context to Answer Generation                     │    │
│  │    (existing _generate_answer_with_citations pipeline)    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Tree Node Structure

Each node in the tree represents a section of the document:

```json
{
    "node_id": "0001",
    "title": "Chapter 3: Pain Management in Palliative Care",
    "level": 1,
    "start_page": 45,
    "end_page": 78,
    "summary": "Comprehensive overview of pain assessment scales, WHO analgesic ladder, opioid dosing guidelines, adjuvant therapies, and breakthrough pain management in palliative care settings.",
    "text": "Full text of the section... (only loaded during context extraction, not during tree navigation)",
    "children": [
        {
            "node_id": "0002",
            "title": "3.1 Pain Assessment",
            "level": 2,
            "start_page": 45,
            "end_page": 52,
            "summary": "Pain assessment tools including NRS, VAS, FLACC for non-verbal patients...",
            "text": "...",
            "children": []
        },
        {
            "node_id": "0003",
            "title": "3.2 WHO Analgesic Ladder",
            "level": 2,
            "start_page": 53,
            "end_page": 61,
            "summary": "Three-step WHO approach: non-opioid, weak opioid, strong opioid...",
            "text": "...",
            "children": [...]
        }
    ]
}
```

### 2.3 Document Flow Comparison

| Step | Vector RAG (Current) | PageIndex RAG (New) |
|------|---------------------|-------------------|
| **Parse** | Extract text | Extract text + detect structure (TOC/headings) |
| **Index** | Chunk (1000 chars) → Embed → ChromaDB | Build tree → LLM summaries → JSON file |
| **Retrieve** | Vector similarity search | LLM reasons over tree summaries → selects nodes |
| **Context** | Top-k chunks (fragmented) | Full sections from selected nodes (coherent) |
| **Answer** | Same LLM generation pipeline | Same LLM generation pipeline |

---

## 3. Prerequisites and Dependencies

### 3.1 New Dependencies

Add to `requirements.txt`:

```
# PageIndex RAG (tree-based retrieval)
pymupdf>=1.26.0
tiktoken>=0.5.0
```

**Already present**: `PyPDF2`, `python-dotenv`, `pyyaml`, `aiohttp`, `requests`

### 3.2 Environment Variables

Add to `.env.example`:

```bash
# PageIndex RAG (Optional - reasoning-based retrieval)
# LLM provider for tree building and navigation: "groq" or "openai"
PAGEINDEX_LLM_PROVIDER=groq
# Only needed if PAGEINDEX_LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
# Model override (optional, defaults to provider's best reasoning model)
PAGEINDEX_MODEL=
```

No new API keys required when using Groq (default) — reuses existing `GROQ_API_KEY`.

### 3.3 System Requirements

- Python 3.10+
- Groq API key (existing) OR OpenAI API key (optional)
- ~50MB disk per 100-page document tree index

---

## 4. Directory Structure

### 4.1 Module Structure

```
pageindex_integration/
├── __init__.py           # Lazy imports, __all__, __version__
├── config.py             # PageIndexConfig dataclass
├── llm_adapter.py        # LLMAdapter (Groq + optional OpenAI)
├── storage.py            # PageIndexStorage (tree JSON persistence)
├── tree_builder.py       # PageIndexTreeBuilder (document → tree)
├── query_engine.py       # PageIndexQueryEngine (tree search)
└── utils.py              # Shared utilities (token counting, tree manipulation)
```

### 4.2 Data Structure

```
data/pageindex/
├── index.json            # Master index: doc_id → tree metadata
├── trees/                # Per-document tree JSON files
│   ├── {doc_id}_tree.json
│   └── ...
└── cache/                # Query result cache
```

### 4.3 Test Structure

```
tests/
├── test_pageindex_config.py
├── test_pageindex_storage.py
├── test_pageindex_tree_builder.py
├── test_pageindex_query_engine.py
├── test_pageindex_llm_adapter.py
└── test_pageindex_integration.py
```

---

## 5. Phase 1: Foundation and LLM Adapter

### 5.1 Module Init

**File**: `pageindex_integration/__init__.py`

```python
"""
PageIndex-Style RAG Integration Module for Palli Sahayak

Vectorless, reasoning-based retrieval using hierarchical tree indexes.
Documents are parsed into tree structures, and LLMs reason over node
summaries to find relevant sections — no vector database required.

Components:
    - PageIndexConfig: Configuration management
    - PageIndexTreeBuilder: Document → tree indexing
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

    # Index a document
    await builder.build_tree("/path/to/document.pdf", "doc_id_123")

    # Query
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
```

### 5.2 Configuration Module

**File**: `pageindex_integration/config.py`

```python
import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TreeBuildConfig:
    """Configuration for tree building."""
    toc_check_pages: int = 20
    max_pages_per_node: int = 10
    max_tokens_per_node: int = 20000
    add_node_summary: bool = True
    add_node_id: bool = True
    summary_max_tokens: int = 200
    min_section_chars: int = 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "toc_check_pages": self.toc_check_pages,
            "max_pages_per_node": self.max_pages_per_node,
            "max_tokens_per_node": self.max_tokens_per_node,
            "add_node_summary": self.add_node_summary,
            "add_node_id": self.add_node_id,
            "summary_max_tokens": self.summary_max_tokens,
            "min_section_chars": self.min_section_chars,
        }


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""
    provider: str = "groq"           # "groq" or "openai"
    model: str = ""                  # empty = use provider default
    temperature: float = 0.0
    max_tokens: int = 4096
    max_retries: int = 3
    request_timeout: int = 120
    rate_limit_rpm: int = 30         # requests per minute

    @property
    def effective_model(self) -> str:
        if self.model:
            return self.model
        if self.provider == "openai":
            return "gpt-4o"
        return "qwen/qwen3-32b"

    @property
    def api_key(self) -> str:
        if self.provider == "openai":
            return os.getenv("OPENAI_API_KEY", "")
        return os.getenv("GROQ_API_KEY", "")

    @property
    def base_url(self) -> str:
        if self.provider == "openai":
            return "https://api.openai.com/v1"
        return "https://api.groq.com/openai/v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.effective_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_retries": self.max_retries,
            "request_timeout": self.request_timeout,
            "rate_limit_rpm": self.rate_limit_rpm,
            "api_key": "***" if self.api_key else "",
        }


@dataclass
class SearchConfig:
    """Configuration for tree search."""
    max_nodes_per_query: int = 5
    max_context_tokens: int = 8000
    cache_enabled: bool = True
    cache_maxsize: int = 100
    cache_ttl_seconds: int = 3600

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_nodes_per_query": self.max_nodes_per_query,
            "max_context_tokens": self.max_context_tokens,
            "cache_enabled": self.cache_enabled,
            "cache_maxsize": self.cache_maxsize,
            "cache_ttl_seconds": self.cache_ttl_seconds,
        }


class PageIndexConfig:
    """
    Configuration manager for PageIndex integration.

    Loads from environment variables and optional YAML.
    Follows the same pattern as graphrag_integration/config.py.

    Example:
        config = PageIndexConfig()
        config = PageIndexConfig(root_dir="./data/pageindex")
        config = PageIndexConfig.from_yaml("./config.yaml")
    """

    def __init__(self, root_dir: str = "./data/pageindex"):
        self.root_dir = Path(root_dir)
        self.tree_build = TreeBuildConfig()
        self.llm = LLMConfig(
            provider=os.getenv("PAGEINDEX_LLM_PROVIDER", "groq"),
            model=os.getenv("PAGEINDEX_MODEL", ""),
        )
        self.search = SearchConfig()

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PageIndexConfig":
        """Load pageindex section from a YAML config file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required: pip install pyyaml")

        with open(yaml_path, 'r') as f:
            settings = yaml.safe_load(f)

        pi_settings = settings.get("pageindex", {})
        config = cls(root_dir=pi_settings.get("root_dir", "./data/pageindex"))

        # Parse tree_build
        tb = pi_settings.get("tree_build", {})
        for key, val in tb.items():
            if hasattr(config.tree_build, key):
                setattr(config.tree_build, key, val)

        # Parse llm
        llm = pi_settings.get("llm", {})
        for key, val in llm.items():
            if hasattr(config.llm, key):
                setattr(config.llm, key, val)

        # Parse search
        search = pi_settings.get("search", {})
        for key, val in search.items():
            if hasattr(config.search, key):
                setattr(config.search, key, val)

        return config

    @property
    def trees_dir(self) -> Path:
        return self.root_dir / "trees"

    @property
    def cache_dir(self) -> Path:
        return self.root_dir / "cache"

    @property
    def index_file(self) -> Path:
        return self.root_dir / "index.json"

    def validate(self) -> List[str]:
        errors = []
        if not self.llm.api_key:
            key_name = "OPENAI_API_KEY" if self.llm.provider == "openai" else "GROQ_API_KEY"
            errors.append(f"LLM API key not configured (set {key_name})")
        if self.tree_build.max_pages_per_node < 1:
            errors.append(f"Invalid max_pages_per_node: {self.tree_build.max_pages_per_node}")
        if self.search.max_context_tokens < 100:
            errors.append(f"Invalid max_context_tokens: {self.search.max_context_tokens}")
        return errors

    def ensure_directories(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.trees_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_dir": str(self.root_dir),
            "tree_build": self.tree_build.to_dict(),
            "llm": self.llm.to_dict(),
            "search": self.search.to_dict(),
        }

    def __repr__(self) -> str:
        return (
            f"PageIndexConfig(root_dir={self.root_dir}, "
            f"provider={self.llm.provider}, "
            f"model={self.llm.effective_model})"
        )
```

### 5.3 LLM Adapter

**File**: `pageindex_integration/llm_adapter.py`

```python
import os
import logging
import asyncio
import time
from typing import List, Dict, Any, Optional

import aiohttp
import requests

from pageindex_integration.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMAdapter:
    """
    Unified LLM interface supporting Groq and OpenAI backends.

    Both providers expose OpenAI-compatible chat completions APIs.
    Groq: https://api.groq.com/openai/v1/chat/completions
    OpenAI: https://api.openai.com/v1/chat/completions

    Features:
    - Sync and async chat completions
    - Automatic retry with exponential backoff
    - Rate limiting
    - Token counting via tiktoken

    Example:
        adapter = LLMAdapter(config.llm)
        response = await adapter.chat_async([
            {"role": "system", "content": "You are a document analyst."},
            {"role": "user", "content": "Analyze this tree structure..."},
        ])
    """

    def __init__(self, config: LLMConfig):
        self._config = config
        self._last_request_time = 0.0
        self._min_interval = 60.0 / max(config.rate_limit_rpm, 1)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        return {
            "model": self._config.effective_model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self._config.temperature),
            "max_tokens": kwargs.get("max_tokens", self._config.max_tokens),
        }

    async def _rate_limit(self) -> None:
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    async def chat_async(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> str:
        """
        Async chat completion with retry.

        Args:
            messages: Chat messages in OpenAI format
            **kwargs: Override temperature, max_tokens

        Returns:
            Response content string

        Raises:
            RuntimeError: After exhausting retries
        """
        url = f"{self._config.base_url}/chat/completions"
        payload = self._payload(messages, **kwargs)

        for attempt in range(self._config.max_retries):
            await self._rate_limit()
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        headers=self._headers(),
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self._config.request_timeout),
                    ) as resp:
                        if resp.status == 429:
                            retry_after = float(resp.headers.get("Retry-After", 2 ** attempt))
                            logger.warning(f"Rate limited, waiting {retry_after}s")
                            await asyncio.sleep(retry_after)
                            continue
                        resp.raise_for_status()
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"]
            except (aiohttp.ClientError, asyncio.TimeoutError, KeyError) as e:
                if attempt == self._config.max_retries - 1:
                    raise RuntimeError(
                        f"LLM request failed after {self._config.max_retries} attempts: {e}"
                    )
                wait = 2 ** attempt
                logger.warning(f"LLM request failed (attempt {attempt + 1}), retrying in {wait}s: {e}")
                await asyncio.sleep(wait)

        raise RuntimeError("LLM request failed: exhausted retries")

    def chat_sync(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> str:
        """Synchronous chat completion (for non-async contexts)."""
        url = f"{self._config.base_url}/chat/completions"
        payload = self._payload(messages, **kwargs)

        for attempt in range(self._config.max_retries):
            try:
                resp = requests.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self._config.request_timeout,
                )
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", 2 ** attempt))
                    time.sleep(retry_after)
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except (requests.RequestException, KeyError) as e:
                if attempt == self._config.max_retries - 1:
                    raise RuntimeError(
                        f"LLM request failed after {self._config.max_retries} attempts: {e}"
                    )
                time.sleep(2 ** attempt)

        raise RuntimeError("LLM request failed: exhausted retries")

    def __repr__(self) -> str:
        return (
            f"LLMAdapter(provider={self._config.provider}, "
            f"model={self._config.effective_model})"
        )
```

### 5.4 Utilities

**File**: `pageindex_integration/utils.py`

```python
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    except ImportError:
        return len(text) // 4  # rough approximation


def flatten_tree(tree: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Flatten a tree into a node_id -> node mapping for O(1) lookup."""
    node_map = {}

    def _walk(node: Dict[str, Any]):
        node_id = node.get("node_id")
        if node_id:
            node_map[node_id] = node
        for child in node.get("children", []):
            _walk(child)

    if isinstance(tree, list):
        for node in tree:
            _walk(node)
    else:
        _walk(tree)

    return node_map


def strip_text_from_tree(tree: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-copy tree structure removing 'text' fields.
    Used during tree navigation to reduce token count —
    LLM only needs titles and summaries to reason about relevance.
    """
    import copy
    stripped = copy.deepcopy(tree)

    def _strip(node: Dict[str, Any]):
        node.pop("text", None)
        for child in node.get("children", []):
            _strip(child)

    if isinstance(stripped, list):
        for node in stripped:
            _strip(node)
    else:
        _strip(stripped)

    return stripped


def extract_node_texts(
    node_ids: List[str],
    node_map: Dict[str, Dict[str, Any]],
    max_tokens: int = 8000,
) -> List[Dict[str, Any]]:
    """
    Extract text content from selected nodes, respecting token budget.

    Returns list of dicts with keys: node_id, title, text, start_page, end_page
    """
    results = []
    total_tokens = 0

    for nid in node_ids:
        node = node_map.get(nid)
        if not node:
            logger.warning(f"Node {nid} not found in tree")
            continue

        text = node.get("text", "")
        tokens = count_tokens(text)

        if total_tokens + tokens > max_tokens:
            remaining = max_tokens - total_tokens
            if remaining > 100:
                # Truncate to fit
                ratio = remaining / max(tokens, 1)
                truncated = text[:int(len(text) * ratio)]
                results.append({
                    "node_id": nid,
                    "title": node.get("title", ""),
                    "text": truncated + "\n[... truncated ...]",
                    "start_page": node.get("start_page", 0),
                    "end_page": node.get("end_page", 0),
                })
            break

        results.append({
            "node_id": nid,
            "title": node.get("title", ""),
            "text": text,
            "start_page": node.get("start_page", 0),
            "end_page": node.get("end_page", 0),
        })
        total_tokens += tokens

    return results


def tree_stats(tree: Dict[str, Any]) -> Dict[str, Any]:
    """Compute statistics about a tree structure."""
    node_count = 0
    max_depth = 0
    total_text_len = 0

    def _walk(node: Dict[str, Any], depth: int):
        nonlocal node_count, max_depth, total_text_len
        node_count += 1
        max_depth = max(max_depth, depth)
        total_text_len += len(node.get("text", ""))
        for child in node.get("children", []):
            _walk(child, depth + 1)

    if isinstance(tree, list):
        for node in tree:
            _walk(node, 0)
    else:
        _walk(tree, 0)

    return {
        "node_count": node_count,
        "max_depth": max_depth,
        "total_text_chars": total_text_len,
    }
```

### 5.5 Tests

```bash
python -c "from pageindex_integration import PageIndexConfig; print(PageIndexConfig())"
python -c "from pageindex_integration import LLMAdapter; print('OK')"
```

### 5.6 Commit Message

```
feat(pageindex): Phase 1 - foundation, config, and LLM adapter
```

---

## 6. Phase 2: Storage Layer

### 6.1 Storage Implementation

**File**: `pageindex_integration/storage.py`

```python
import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

from pageindex_integration.config import PageIndexConfig

logger = logging.getLogger(__name__)


@dataclass
class TreeIndexEntry:
    """Metadata for a single tree index."""
    doc_id: str
    filename: str
    tree_path: str
    indexed_at: str
    node_count: int
    page_count: int
    status: str  # "pending", "building", "completed", "failed"
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PageIndexStorage:
    """
    Manages tree index JSON files on disk.

    Each document gets a {doc_id}_tree.json in data/pageindex/trees/.
    A master index at data/pageindex/index.json maps doc_ids to metadata.

    Example:
        storage = PageIndexStorage(config)
        storage.save_tree("abc123", tree_dict, {"filename": "guide.pdf", "page_count": 100})
        tree = storage.load_tree("abc123")
        entries = storage.list_trees()
    """

    def __init__(self, config: PageIndexConfig):
        self._config = config
        config.ensure_directories()
        self._index: Dict[str, TreeIndexEntry] = {}
        self._load_index()

    def _load_index(self) -> None:
        if self._config.index_file.exists():
            try:
                with open(self._config.index_file, 'r') as f:
                    raw = json.load(f)
                self._index = {
                    k: TreeIndexEntry(**v) for k, v in raw.items()
                }
                logger.info(f"Loaded {len(self._index)} tree index entries")
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Failed to load tree index: {e}")
                self._index = {}
        else:
            self._index = {}

    def _save_index(self) -> None:
        with open(self._config.index_file, 'w') as f:
            json.dump(
                {k: v.to_dict() for k, v in self._index.items()},
                f, indent=2
            )

    def save_tree(
        self,
        doc_id: str,
        tree: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Path:
        """Persist tree JSON and update index."""
        tree_filename = f"{doc_id}_tree.json"
        tree_path = self._config.trees_dir / tree_filename

        with open(tree_path, 'w') as f:
            json.dump(tree, f, indent=2)

        from pageindex_integration.utils import tree_stats
        stats = tree_stats(tree)

        self._index[doc_id] = TreeIndexEntry(
            doc_id=doc_id,
            filename=metadata.get("filename", ""),
            tree_path=str(tree_path),
            indexed_at=datetime.now().isoformat(),
            node_count=stats["node_count"],
            page_count=metadata.get("page_count", 0),
            status="completed",
        )
        self._save_index()
        logger.info(f"Saved tree for {doc_id}: {stats['node_count']} nodes")
        return tree_path

    def load_tree(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Load tree JSON by doc_id. Returns None if not found."""
        entry = self._index.get(doc_id)
        if not entry or entry.status != "completed":
            return None

        tree_path = Path(entry.tree_path)
        if not tree_path.exists():
            logger.warning(f"Tree file missing for {doc_id}: {tree_path}")
            return None

        with open(tree_path, 'r') as f:
            return json.load(f)

    def has_tree(self, doc_id: str) -> bool:
        entry = self._index.get(doc_id)
        return entry is not None and entry.status == "completed"

    def delete_tree(self, doc_id: str) -> bool:
        entry = self._index.get(doc_id)
        if not entry:
            return False

        tree_path = Path(entry.tree_path)
        if tree_path.exists():
            tree_path.unlink()

        del self._index[doc_id]
        self._save_index()
        return True

    def set_status(self, doc_id: str, status: str, error: str = "") -> None:
        """Update indexing status for a document."""
        if doc_id in self._index:
            self._index[doc_id].status = status
            self._index[doc_id].error = error
            self._save_index()

    def list_trees(self) -> List[TreeIndexEntry]:
        return list(self._index.values())

    def get_stats(self) -> Dict[str, Any]:
        completed = [e for e in self._index.values() if e.status == "completed"]
        return {
            "total_trees": len(self._index),
            "completed": len(completed),
            "total_nodes": sum(e.node_count for e in completed),
            "total_pages": sum(e.page_count for e in completed),
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"PageIndexStorage(trees={stats['total_trees']}, nodes={stats['total_nodes']})"
```

### 6.2 Index File Format

`data/pageindex/index.json`:

```json
{
    "abc123def456": {
        "doc_id": "abc123def456",
        "filename": "palliative_care_essentials.pdf",
        "tree_path": "data/pageindex/trees/abc123def456_tree.json",
        "indexed_at": "2026-02-23T14:30:00",
        "node_count": 47,
        "page_count": 120,
        "status": "completed",
        "error": ""
    }
}
```

### 6.3 Commit Message

```
feat(pageindex): Phase 2 - tree storage layer
```

---

## 7. Phase 3: Tree Builder (Indexing Pipeline)

### 7.1 Tree Builder Implementation

**File**: `pageindex_integration/tree_builder.py`

```python
import os
import re
import json
import logging
import asyncio
from pathlib import Path
from enum import Enum
from typing import Dict, List, Any, Optional, Callable

from pageindex_integration.config import PageIndexConfig
from pageindex_integration.storage import PageIndexStorage
from pageindex_integration.llm_adapter import LLMAdapter
from pageindex_integration.utils import count_tokens

logger = logging.getLogger(__name__)


class IndexingStatus(str, Enum):
    PENDING = "pending"
    BUILDING = "building"
    COMPLETED = "completed"
    FAILED = "failed"


class PageIndexTreeBuilder:
    """
    Builds hierarchical tree indexes from documents.

    Supports PDF (via pymupdf TOC + text extraction), Markdown (heading hierarchy),
    DOCX (heading styles), and plain text (LLM-generated structure).

    The tree building process:
    1. Extract document structure (TOC, headings, or LLM-detected sections)
    2. Extract text per section with page boundaries
    3. Generate LLM summaries for each node
    4. Persist tree JSON via PageIndexStorage

    Example:
        builder = PageIndexTreeBuilder(config, storage)
        result = await builder.build_tree("/path/to/doc.pdf", "doc_id_123")
        print(f"Built tree with {result['node_count']} nodes")
    """

    def __init__(
        self,
        config: PageIndexConfig,
        storage: PageIndexStorage,
        on_progress: Optional[Callable[[str, int], None]] = None,
    ):
        self._config = config
        self._storage = storage
        self._llm = LLMAdapter(config.llm)
        self._on_progress = on_progress
        self._status: Dict[str, IndexingStatus] = {}
        self._node_counter = 0

    def _next_node_id(self) -> str:
        self._node_counter += 1
        return f"{self._node_counter:04d}"

    def _reset_counter(self) -> None:
        self._node_counter = 0

    async def build_tree(
        self,
        file_path: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build tree index for a single document.

        Args:
            file_path: Path to document file
            doc_id: Document identifier (from SimpleRAGPipeline)
            metadata: Optional document metadata (filename, page_count, etc.)

        Returns:
            Dict with keys: status, node_count, tree_path, error
        """
        metadata = metadata or {}
        self._reset_counter()
        self._status[doc_id] = IndexingStatus.BUILDING
        self._storage.set_status(doc_id, "building")

        try:
            ext = Path(file_path).suffix.lower()

            if ext == ".pdf":
                tree = await self._build_tree_pdf(file_path)
            elif ext == ".md":
                tree = await self._build_tree_markdown(file_path)
            elif ext == ".docx":
                tree = await self._build_tree_docx(file_path)
            elif ext == ".txt":
                tree = await self._build_tree_text(file_path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")

            # Generate summaries for nodes that don't have them
            if self._config.tree_build.add_node_summary:
                await self._generate_summaries(tree)

            # Persist
            tree_path = self._storage.save_tree(doc_id, tree, {
                "filename": metadata.get("filename", Path(file_path).name),
                "page_count": metadata.get("page_count", 0),
            })

            self._status[doc_id] = IndexingStatus.COMPLETED
            from pageindex_integration.utils import tree_stats
            stats = tree_stats(tree)

            logger.info(f"Tree built for {doc_id}: {stats['node_count']} nodes")
            return {
                "status": "completed",
                "node_count": stats["node_count"],
                "tree_path": str(tree_path),
                "error": "",
            }

        except Exception as e:
            self._status[doc_id] = IndexingStatus.FAILED
            self._storage.set_status(doc_id, "failed", str(e))
            logger.error(f"Tree build failed for {doc_id}: {e}")
            return {
                "status": "failed",
                "node_count": 0,
                "tree_path": "",
                "error": str(e),
            }

    async def _build_tree_pdf(self, file_path: str) -> Dict[str, Any]:
        """Build tree from PDF using TOC extraction or heading detection."""
        import fitz  # pymupdf

        doc = fitz.open(file_path)
        toc = doc.get_toc()  # [[level, title, page_number], ...]

        if toc:
            tree = self._toc_to_tree(toc, doc)
        else:
            tree = self._pages_to_tree(doc)

        doc.close()
        return tree

    def _toc_to_tree(self, toc: List, doc) -> Dict[str, Any]:
        """Convert PDF TOC to tree structure with text from pages."""
        root = {
            "node_id": self._next_node_id(),
            "title": doc.metadata.get("title", Path(doc.name).stem),
            "level": 0,
            "start_page": 1,
            "end_page": len(doc),
            "summary": "",
            "text": "",
            "children": [],
        }

        # Build flat list with page ranges
        entries = []
        for i, (level, title, page_num) in enumerate(toc):
            next_page = toc[i + 1][2] if i + 1 < len(toc) else len(doc) + 1
            entries.append({
                "level": level,
                "title": title.strip(),
                "start_page": page_num,
                "end_page": next_page - 1,
            })

        # Build nested tree from flat list
        stack = [root]
        for entry in entries:
            node = {
                "node_id": self._next_node_id(),
                "title": entry["title"],
                "level": entry["level"],
                "start_page": entry["start_page"],
                "end_page": entry["end_page"],
                "summary": "",
                "text": self._extract_pages_text(
                    doc, entry["start_page"], entry["end_page"]
                ),
                "children": [],
            }

            # Find parent: walk stack back to find level < current
            while len(stack) > 1 and stack[-1]["level"] >= entry["level"]:
                stack.pop()

            stack[-1]["children"].append(node)
            stack.append(node)

        return root

    def _pages_to_tree(self, doc) -> Dict[str, Any]:
        """Fallback: group pages into chunks when no TOC is available."""
        max_pages = self._config.tree_build.max_pages_per_node
        root = {
            "node_id": self._next_node_id(),
            "title": doc.metadata.get("title", Path(doc.name).stem),
            "level": 0,
            "start_page": 1,
            "end_page": len(doc),
            "summary": "",
            "text": "",
            "children": [],
        }

        for start in range(0, len(doc), max_pages):
            end = min(start + max_pages, len(doc))
            node = {
                "node_id": self._next_node_id(),
                "title": f"Pages {start + 1}-{end}",
                "level": 1,
                "start_page": start + 1,
                "end_page": end,
                "summary": "",
                "text": self._extract_pages_text(doc, start + 1, end),
                "children": [],
            }
            root["children"].append(node)

        return root

    def _extract_pages_text(self, doc, start_page: int, end_page: int) -> str:
        """Extract text from a page range (1-indexed)."""
        texts = []
        for page_num in range(max(0, start_page - 1), min(end_page, len(doc))):
            page = doc[page_num]
            texts.append(page.get_text())
        return "\n".join(texts)

    async def _build_tree_markdown(self, file_path: str) -> Dict[str, Any]:
        """Build tree from Markdown using heading hierarchy."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        headings = [(m.start(), len(m.group(1)), m.group(2).strip()) for m in heading_pattern.finditer(content)]

        root = {
            "node_id": self._next_node_id(),
            "title": Path(file_path).stem,
            "level": 0,
            "start_page": 1,
            "end_page": 1,
            "summary": "",
            "text": "",
            "children": [],
        }

        if not headings:
            root["text"] = content
            return root

        stack = [root]
        for i, (pos, level, title) in enumerate(headings):
            next_pos = headings[i + 1][0] if i + 1 < len(headings) else len(content)
            text = content[pos:next_pos].strip()
            # Remove the heading line itself from text body
            text = re.sub(r'^#{1,6}\s+.+\n?', '', text, count=1).strip()

            node = {
                "node_id": self._next_node_id(),
                "title": title,
                "level": level,
                "start_page": 1,
                "end_page": 1,
                "summary": "",
                "text": text,
                "children": [],
            }

            while len(stack) > 1 and stack[-1]["level"] >= level:
                stack.pop()

            stack[-1]["children"].append(node)
            stack.append(node)

        return root

    async def _build_tree_docx(self, file_path: str) -> Dict[str, Any]:
        """Build tree from DOCX using heading styles."""
        import docx as python_docx

        doc = python_docx.Document(file_path)
        md_lines = []
        for para in doc.paragraphs:
            style = para.style.name.lower()
            if style.startswith("heading"):
                try:
                    level = int(style.replace("heading", "").strip())
                    md_lines.append(f"{'#' * level} {para.text}")
                except ValueError:
                    md_lines.append(para.text)
            else:
                md_lines.append(para.text)

        # Write temp markdown and parse
        md_content = "\n".join(md_lines)
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp:
            tmp.write(md_content)
            tmp_path = tmp.name

        try:
            tree = await self._build_tree_markdown(tmp_path)
        finally:
            os.unlink(tmp_path)

        tree["title"] = Path(file_path).stem
        return tree

    async def _build_tree_text(self, file_path: str) -> Dict[str, Any]:
        """Build tree from plain text using LLM-detected section boundaries."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # For short texts, single node
        if count_tokens(content) < 2000:
            return {
                "node_id": self._next_node_id(),
                "title": Path(file_path).stem,
                "level": 0,
                "start_page": 1,
                "end_page": 1,
                "summary": "",
                "text": content,
                "children": [],
            }

        # Use LLM to detect section boundaries
        detect_prompt = f"""Analyze the following text and identify distinct sections or topics.
Return a JSON array of objects with "title" and "start_char" (character offset where section begins).

Text (first 3000 chars):
{content[:3000]}

Return ONLY valid JSON array, no other text:"""

        response = await self._llm.chat_async([
            {"role": "user", "content": detect_prompt},
        ], max_tokens=1024)

        try:
            sections = json.loads(response.strip())
        except json.JSONDecodeError:
            # Fallback: split into fixed-size sections
            sections = []
            chunk_size = 5000
            for i in range(0, len(content), chunk_size):
                sections.append({
                    "title": f"Section {i // chunk_size + 1}",
                    "start_char": i,
                })

        root = {
            "node_id": self._next_node_id(),
            "title": Path(file_path).stem,
            "level": 0,
            "start_page": 1,
            "end_page": 1,
            "summary": "",
            "text": "",
            "children": [],
        }

        for i, section in enumerate(sections):
            start = section.get("start_char", 0)
            end = sections[i + 1]["start_char"] if i + 1 < len(sections) else len(content)
            text = content[start:end].strip()

            if text:
                root["children"].append({
                    "node_id": self._next_node_id(),
                    "title": section.get("title", f"Section {i + 1}"),
                    "level": 1,
                    "start_page": 1,
                    "end_page": 1,
                    "summary": "",
                    "text": text,
                    "children": [],
                })

        return root

    async def _generate_summaries(self, tree: Dict[str, Any]) -> None:
        """Generate LLM summaries for nodes that have text but no summary."""
        nodes_to_summarize = []

        def _collect(node: Dict[str, Any]):
            if node.get("text") and not node.get("summary"):
                nodes_to_summarize.append(node)
            for child in node.get("children", []):
                _collect(child)

        _collect(tree)

        if not nodes_to_summarize:
            return

        logger.info(f"Generating summaries for {len(nodes_to_summarize)} nodes")

        for i, node in enumerate(nodes_to_summarize):
            text_preview = node["text"][:2000]
            prompt = f"""Summarize the following section in 1-2 sentences. Focus on key medical concepts, treatments, or recommendations mentioned.

Section title: {node.get('title', 'Untitled')}
Text:
{text_preview}

Summary:"""

            try:
                summary = await self._llm.chat_async(
                    [{"role": "user", "content": prompt}],
                    max_tokens=self._config.tree_build.summary_max_tokens,
                )
                node["summary"] = summary.strip()
            except Exception as e:
                logger.warning(f"Failed to generate summary for node {node['node_id']}: {e}")
                node["summary"] = node.get("title", "")

            if self._on_progress:
                self._on_progress(f"Summarizing nodes", int((i + 1) / len(nodes_to_summarize) * 100))

    async def batch_index(
        self,
        documents: List[Dict[str, Any]],
        on_progress: Optional[Callable[[int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Batch index multiple documents.

        Args:
            documents: List of dicts with keys: doc_id, file_path, metadata
            on_progress: Progress callback (0-100)

        Returns:
            Dict with keys: total, completed, failed, results
        """
        results = []
        for i, doc in enumerate(documents):
            result = await self.build_tree(
                file_path=doc["file_path"],
                doc_id=doc["doc_id"],
                metadata=doc.get("metadata", {}),
            )
            results.append({"doc_id": doc["doc_id"], **result})

            if on_progress:
                on_progress(int((i + 1) / len(documents) * 100))

        completed = [r for r in results if r["status"] == "completed"]
        failed = [r for r in results if r["status"] == "failed"]

        return {
            "total": len(documents),
            "completed": len(completed),
            "failed": len(failed),
            "results": results,
        }

    def get_status(self, doc_id: str) -> Optional[str]:
        return self._status.get(doc_id, IndexingStatus.PENDING).value

    def __repr__(self) -> str:
        active = sum(1 for s in self._status.values() if s == IndexingStatus.BUILDING)
        return f"PageIndexTreeBuilder(active={active})"
```

### 7.2 LLM Prompts for Tree Building

**Summary generation prompt** (embedded in tree_builder.py):

```
Summarize the following section in 1-2 sentences. Focus on key medical
concepts, treatments, or recommendations mentioned.

Section title: {title}
Text:
{text_preview}

Summary:
```

**Plain text section detection prompt** (for .txt files without headings):

```
Analyze the following text and identify distinct sections or topics.
Return a JSON array of objects with "title" and "start_char" (character
offset where section begins).

Text (first 3000 chars):
{content}

Return ONLY valid JSON array, no other text:
```

### 7.3 Commit Message

```
feat(pageindex): Phase 3 - tree builder (indexing pipeline)
```

---

## 8. Phase 4: Query Engine

### 8.1 Query Engine Implementation

**File**: `pageindex_integration/query_engine.py`

```python
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from pageindex_integration.config import PageIndexConfig
from pageindex_integration.storage import PageIndexStorage
from pageindex_integration.llm_adapter import LLMAdapter
from pageindex_integration.utils import (
    flatten_tree,
    strip_text_from_tree,
    extract_node_texts,
    count_tokens,
)

logger = logging.getLogger(__name__)


@dataclass
class PageIndexSearchResult:
    """Result of a PageIndex tree search."""
    query: str
    context: str
    selected_nodes: List[Dict[str, Any]]
    reasoning: str
    doc_sources: List[Dict[str, Any]]
    confidence: float
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "context": self.context,
            "selected_nodes": self.selected_nodes,
            "reasoning": self.reasoning,
            "doc_sources": self.doc_sources,
            "confidence": self.confidence,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


class PageIndexQueryEngine:
    """
    Reasoning-based retrieval over document tree indexes.

    Two-step process:
    1. LLM reasons over tree summaries to select relevant nodes
    2. Full text extracted from selected nodes as context

    The context then feeds into the existing answer generation pipeline
    in simple_rag_server.py (_generate_answer_with_citations).

    Example:
        engine = PageIndexQueryEngine(config, storage)
        result = await engine.search("What are morphine dosing guidelines?")
        print(result.context)       # Retrieved text
        print(result.reasoning)     # Why these sections were selected
        print(result.doc_sources)   # Source documents + page ranges
    """

    def __init__(self, config: PageIndexConfig, storage: PageIndexStorage):
        self._config = config
        self._storage = storage
        self._llm = LLMAdapter(config.llm)
        self._cache: Optional[Any] = None

        if config.search.cache_enabled:
            try:
                from graphrag_integration.utils import QueryCache
                self._cache = QueryCache(
                    maxsize=config.search.cache_maxsize,
                    ttl_seconds=config.search.cache_ttl_seconds,
                )
            except ImportError:
                logger.debug("QueryCache not available, caching disabled")

    async def search(
        self,
        query: str,
        doc_ids: Optional[List[str]] = None,
    ) -> PageIndexSearchResult:
        """
        Search across indexed trees.

        Args:
            query: User query
            doc_ids: Optional list of doc_ids to search (None = all)

        Returns:
            PageIndexSearchResult with context, sources, reasoning
        """
        start_time = time.time()

        # Check cache
        if self._cache:
            cached = self._cache.get(query, "pageindex")
            if cached is not None:
                logger.debug(f"Cache hit for: {query[:50]}...")
                return cached

        # Load trees
        entries = self._storage.list_trees()
        if doc_ids:
            entries = [e for e in entries if e.doc_id in doc_ids]

        completed_entries = [e for e in entries if e.status == "completed"]
        if not completed_entries:
            return PageIndexSearchResult(
                query=query,
                context="",
                selected_nodes=[],
                reasoning="No tree indexes available.",
                doc_sources=[],
                confidence=0.0,
                duration_ms=0.0,
            )

        all_selected_nodes = []
        all_sources = []
        all_reasoning = []

        for entry in completed_entries:
            tree = self._storage.load_tree(entry.doc_id)
            if not tree:
                continue

            # Step 1: LLM tree navigation
            node_ids, reasoning = await self._navigate_tree(query, tree, entry.filename)
            all_reasoning.append(f"[{entry.filename}]: {reasoning}")

            if not node_ids:
                continue

            # Step 2: Extract text from selected nodes
            node_map = flatten_tree(tree)
            extracted = extract_node_texts(
                node_ids,
                node_map,
                max_tokens=self._config.search.max_context_tokens // max(len(completed_entries), 1),
            )

            for ext in extracted:
                ext["filename"] = entry.filename
                ext["doc_id"] = entry.doc_id
                all_selected_nodes.append(ext)

            all_sources.append({
                "doc_id": entry.doc_id,
                "filename": entry.filename,
                "selected_nodes": len(extracted),
                "page_ranges": [
                    f"pg {e['start_page']}-{e['end_page']}" for e in extracted
                ],
            })

        # Build context text
        context_parts = []
        for node in all_selected_nodes:
            source_label = f"Source: {node['filename']}"
            if node.get("start_page"):
                source_label += f" (pg {node['start_page']}-{node['end_page']})"
            context_parts.append(f"{source_label}\n{node['text']}")

        context = "\n\n".join(context_parts)

        duration_ms = (time.time() - start_time) * 1000
        confidence = min(1.0, len(all_selected_nodes) / max(self._config.search.max_nodes_per_query, 1))

        result = PageIndexSearchResult(
            query=query,
            context=context,
            selected_nodes=all_selected_nodes,
            reasoning="\n".join(all_reasoning),
            doc_sources=all_sources,
            confidence=confidence,
            duration_ms=duration_ms,
        )

        # Cache result
        if self._cache:
            self._cache.set(query, "pageindex", result)

        return result

    async def _navigate_tree(
        self,
        query: str,
        tree: Dict[str, Any],
        filename: str,
    ) -> tuple:
        """
        LLM reasons over tree to select relevant node_ids.

        Returns: (list of node_ids, reasoning string)
        """
        stripped = strip_text_from_tree(tree)
        tree_json = json.dumps(stripped, indent=2)

        # Check token budget
        tree_tokens = count_tokens(tree_json)
        if tree_tokens > 50000:
            logger.warning(f"Tree for {filename} is {tree_tokens} tokens, truncating")
            tree_json = tree_json[:200000]  # rough char limit

        prompt = TREE_NAVIGATION_PROMPT.format(
            filename=filename,
            tree_structure=tree_json,
            query=query,
            max_nodes=self._config.search.max_nodes_per_query,
        )

        try:
            response = await self._llm.chat_async(
                [{"role": "user", "content": prompt}],
                max_tokens=1024,
            )

            # Parse JSON response
            json_match = _extract_json(response)
            if json_match:
                parsed = json.loads(json_match)
                node_ids = parsed.get("node_list", parsed.get("nodes", []))
                reasoning = parsed.get("thinking", parsed.get("reasoning", ""))
                return node_ids, reasoning

            logger.warning(f"Failed to parse tree navigation response for {filename}")
            return [], f"Parse error: {response[:200]}"

        except Exception as e:
            logger.error(f"Tree navigation failed for {filename}: {e}")
            return [], str(e)

    def get_stats(self) -> Dict[str, Any]:
        stats = {"cache": None}
        if self._cache:
            stats["cache"] = self._cache.get_stats()
        return stats

    def __repr__(self) -> str:
        return f"PageIndexQueryEngine(provider={self._config.llm.provider})"


def _extract_json(text: str) -> Optional[str]:
    """Extract JSON object from LLM response (may contain markdown fences)."""
    # Try direct parse
    text = text.strip()
    if text.startswith("{"):
        return text

    # Try extracting from code fence
    import re
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try finding JSON object
    match = re.search(r'\{[^{}]*"node_list"[^{}]*\}', text, re.DOTALL)
    if match:
        return match.group(0)

    return None


TREE_NAVIGATION_PROMPT = """You are a document analyst. Given a hierarchical tree structure of a document and a user query, identify which sections (nodes) contain information relevant to answering the query.

DOCUMENT: {filename}

TREE STRUCTURE (titles and summaries only, no full text):
{tree_structure}

USER QUERY: {query}

INSTRUCTIONS:
1. Analyze each node's title and summary to determine relevance to the query
2. Select up to {max_nodes} most relevant nodes
3. Prefer leaf nodes (more specific) over parent nodes (more general) when both are relevant
4. If a parent node's children don't individually cover the query but the parent does, select the parent

Return a JSON object with:
- "thinking": Your reasoning about which sections are relevant and why (1-2 sentences)
- "node_list": Array of node_id strings for the most relevant sections

Return ONLY the JSON object, no other text:"""
```

### 8.2 Commit Message

```
feat(pageindex): Phase 4 - query engine
```

---

## 9. Phase 5: Server Integration

### 9.1 Import Block

Add after line 116 in `simple_rag_server.py` (after GraphRAG imports):

```python
# PageIndex integration imports (vectorless reasoning-based RAG)
try:
    from pageindex_integration import (
        PageIndexConfig,
        PageIndexTreeBuilder,
        PageIndexQueryEngine,
        PageIndexStorage,
    )
    PAGEINDEX_AVAILABLE = True
except ImportError:
    PAGEINDEX_AVAILABLE = False
    PageIndexConfig = None
    PageIndexTreeBuilder = None
    PageIndexQueryEngine = None
    PageIndexStorage = None
```

### 9.2 CLI Argument

Add after line 4399 (after `--provider` arg):

```python
parser.add_argument(
    "--rag-method",
    choices=["vector", "pageindex", "hybrid"],
    default="vector",
    help="RAG method: vector=ChromaDB (default), pageindex=tree-based, hybrid=both"
)
```

### 9.3 Global State and Startup

Add as global variables and a startup event:

```python
# PageIndex service globals
_pageindex_config = None
_pageindex_storage = None
_pageindex_builder = None
_pageindex_engine = None
_pageindex_enabled = False
_rag_method = "vector"
```

Startup initialization (inside the function that creates the FastAPI app):

```python
@app.on_event("startup")
async def startup_pageindex():
    global _pageindex_config, _pageindex_storage, _pageindex_builder
    global _pageindex_engine, _pageindex_enabled, _rag_method

    _rag_method = args.rag_method

    if not PAGEINDEX_AVAILABLE:
        if _rag_method in ("pageindex", "hybrid"):
            logger.warning("PageIndex requested but module not available, falling back to vector")
            _rag_method = "vector"
        return

    if _rag_method not in ("pageindex", "hybrid"):
        logger.info("PageIndex not selected as RAG method")
        return

    try:
        _pageindex_config = PageIndexConfig()
        _pageindex_storage = PageIndexStorage(_pageindex_config)
        _pageindex_builder = PageIndexTreeBuilder(_pageindex_config, _pageindex_storage)
        _pageindex_engine = PageIndexQueryEngine(_pageindex_config, _pageindex_storage)
        _pageindex_enabled = True

        stats = _pageindex_storage.get_stats()
        logger.info(f"PageIndex initialized: {stats['total_trees']} trees, {stats['total_nodes']} nodes")
        logger.info(f"RAG method: {_rag_method}")
    except Exception as e:
        logger.error(f"Failed to initialize PageIndex: {e}")
        _pageindex_enabled = False
        if _rag_method == "pageindex":
            logger.warning("Falling back to vector search")
            _rag_method = "vector"
```

### 9.4 Hook into Document Upload

In `SimpleRAGPipeline.add_documents()`, after the metadata is saved (around line 1103), add:

```python
# Trigger PageIndex tree building if enabled
if _pageindex_enabled and _pageindex_builder:
    asyncio.create_task(
        _pageindex_builder.build_tree(
            permanent_path,
            doc_id,
            metadata={
                "filename": Path(file_path).name,
                "page_count": doc_result.get("page_count", 1),
            },
        )
    )
    logger.info(f"Queued PageIndex tree build for {doc_id}")
```

### 9.5 Modify Query Routing

In `SimpleRAGPipeline.query()`, replace the vector search block (around line 1207) with routing logic:

```python
# Route retrieval based on --rag-method
if _rag_method == "pageindex" and _pageindex_enabled:
    # PageIndex tree-based retrieval
    pi_result = await _pageindex_engine.search(query_for_search)
    context_text = pi_result.context

    if not context_text:
        return {
            "status": "success",
            "answer": "No relevant sections found in the document tree indexes. Please try a different query or ensure documents are indexed.",
            "sources": [],
            "conversation_id": conversation_id,
        }

    # Build metadata list for citation pipeline
    relevant_metadatas = []
    for node in pi_result.selected_nodes:
        relevant_metadatas.append({
            "filename": node.get("filename", ""),
            "chunk_index": 0,
            "total_chunks": 1,
            "page_range": f"{node.get('start_page', 0)}-{node.get('end_page', 0)}",
        })

    # Continue to existing answer generation
    answer, model_used = await self._generate_answer_with_citations(
        question, context_text, relevant_metadatas, should_fuse=len(pi_result.selected_nodes) > 1
    )

elif _rag_method == "hybrid" and _pageindex_enabled:
    # Parallel: vector + tree search
    import asyncio as aio

    async def _vector_search():
        return self.vector_db.query(query_texts=[query_for_search], n_results=top_k)

    async def _tree_search():
        return await _pageindex_engine.search(query_for_search)

    vector_results, pi_result = await aio.gather(_vector_search(), _tree_search())

    # Merge contexts: vector chunks + tree sections
    # Vector context
    vector_context_parts = []
    if vector_results['documents'] and vector_results['documents'][0]:
        for doc, meta in zip(vector_results['documents'][0], vector_results['metadatas'][0]):
            vector_context_parts.append(f"Source: {meta['filename']} (chunk {meta['chunk_index']+1})\n{doc}")

    # Tree context
    tree_context = pi_result.context if pi_result.context else ""

    # Combine
    context_text = ""
    if vector_context_parts:
        context_text += "--- Vector Search Results ---\n" + "\n\n".join(vector_context_parts)
    if tree_context:
        if context_text:
            context_text += "\n\n"
        context_text += "--- Tree Search Results ---\n" + tree_context

    # Build merged metadata
    relevant_metadatas = vector_results.get('metadatas', [[]])[0]
    for node in pi_result.selected_nodes:
        relevant_metadatas.append({
            "filename": node.get("filename", ""),
            "chunk_index": 0,
            "total_chunks": 1,
            "page_range": f"{node.get('start_page', 0)}-{node.get('end_page', 0)}",
        })

    answer, model_used = await self._generate_answer_with_citations(
        question, context_text, relevant_metadatas, should_fuse=True
    )

else:
    # Default: existing vector search (unchanged)
    search_results = self.vector_db.query(
        query_texts=[query_for_search],
        n_results=top_k
    )
    # ... (existing code continues unchanged) ...
```

### 9.6 API Endpoints

Add after the GraphRAG endpoints section:

```python
# ==========================================================================
# PAGEINDEX RAG ENDPOINTS
# ==========================================================================

@app.get("/api/pageindex/health")
async def pageindex_health():
    return {
        "available": PAGEINDEX_AVAILABLE,
        "enabled": _pageindex_enabled,
        "rag_method": _rag_method,
        "provider": _pageindex_config.llm.provider if _pageindex_config else None,
    }

@app.get("/api/pageindex/stats")
async def pageindex_stats():
    if not _pageindex_enabled:
        return {"error": "PageIndex not enabled"}
    return _pageindex_storage.get_stats()

@app.post("/api/pageindex/query")
async def pageindex_query(query: str = Form(...), doc_ids: str = Form("")):
    if not _pageindex_enabled:
        return {"error": "PageIndex not enabled"}
    doc_id_list = [d.strip() for d in doc_ids.split(",") if d.strip()] or None
    result = await _pageindex_engine.search(query, doc_ids=doc_id_list)
    return result.to_dict()

@app.post("/api/pageindex/index")
async def pageindex_index_doc(doc_id: str = Form(...)):
    if not _pageindex_enabled:
        return {"error": "PageIndex not enabled"}
    meta = rag_pipeline.document_metadata.get(doc_id)
    if not meta:
        return {"error": f"Document {doc_id} not found"}
    result = await _pageindex_builder.build_tree(
        meta["file_path"], doc_id,
        metadata={"filename": meta["filename"], "page_count": meta.get("page_count", 0)},
    )
    return result

@app.post("/api/pageindex/index/batch")
async def pageindex_index_batch():
    if not _pageindex_enabled:
        return {"error": "PageIndex not enabled"}
    documents = []
    for doc_id, meta in rag_pipeline.document_metadata.items():
        if not _pageindex_storage.has_tree(doc_id):
            documents.append({
                "doc_id": doc_id,
                "file_path": meta["file_path"],
                "metadata": {"filename": meta["filename"], "page_count": meta.get("page_count", 0)},
            })
    if not documents:
        return {"message": "All documents already indexed", "total": 0}
    result = await _pageindex_builder.batch_index(documents)
    return result

@app.get("/api/pageindex/index/status")
async def pageindex_index_status():
    if not _pageindex_enabled:
        return {"error": "PageIndex not enabled"}
    return {
        "trees": [e.to_dict() for e in _pageindex_storage.list_trees()],
        "stats": _pageindex_storage.get_stats(),
    }

@app.get("/api/pageindex/trees")
async def pageindex_list_trees():
    if not _pageindex_enabled:
        return {"error": "PageIndex not enabled"}
    return [e.to_dict() for e in _pageindex_storage.list_trees()]

@app.get("/api/pageindex/tree/{doc_id}")
async def pageindex_get_tree(doc_id: str):
    if not _pageindex_enabled:
        return {"error": "PageIndex not enabled"}
    tree = _pageindex_storage.load_tree(doc_id)
    if not tree:
        return {"error": f"Tree not found for {doc_id}"}
    return tree
```

### 9.7 Commit Message

```
feat(pageindex): Phase 5 - server integration with CLI routing
```

---

## 10. Phase 6: Admin UI Integration

### 10.1 Gradio Tab

Add inside `create_gradio_interface()` after the GraphRAG tab:

```python
# PAGEINDEX TAB
with gr.TabItem("📄 PageIndex RAG"):
    gr.Markdown("## PageIndex - Reasoning-Based RAG")
    gr.Markdown("*Tree-based document retrieval using LLM reasoning over document structure*")

    with gr.Tabs():
        # Query tab
        with gr.TabItem("🔍 Query"):
            with gr.Row():
                with gr.Column():
                    pi_query_input = gr.Textbox(
                        label="Query",
                        placeholder="e.g., What are opioid dosing guidelines for cancer pain?",
                        lines=2,
                    )
                    pi_query_btn = gr.Button("🔍 Search", variant="primary")
                with gr.Column():
                    pi_response = gr.Markdown(label="Response")
                    with gr.Accordion("Reasoning Trace", open=False):
                        pi_reasoning = gr.Markdown(label="Tree Navigation Reasoning")
                    pi_sources = gr.JSON(label="Source Documents")

            pi_query_btn.click(
                fn=self._handle_pageindex_query,
                inputs=[pi_query_input],
                outputs=[pi_response, pi_reasoning, pi_sources],
            )

        # Indexing tab
        with gr.TabItem("📥 Indexing"):
            with gr.Row():
                pi_index_all_btn = gr.Button("🌳 Index All Documents", variant="primary")
                pi_index_refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary")
            pi_index_status = gr.JSON(label="Indexing Status")

            pi_index_all_btn.click(
                fn=self._handle_pageindex_index_all,
                outputs=[pi_index_status],
            )
            pi_index_refresh_btn.click(
                fn=self._handle_pageindex_status,
                outputs=[pi_index_status],
            )

        # Statistics tab
        with gr.TabItem("📊 Statistics"):
            pi_stats_btn = gr.Button("🔄 Refresh Statistics", variant="primary")
            pi_stats_output = gr.JSON(label="PageIndex Statistics")

            pi_stats_btn.click(
                fn=self._handle_pageindex_stats,
                outputs=[pi_stats_output],
            )
```

### 10.2 Handler Methods

Add to the `SimpleAdminUI` class:

```python
async def _handle_pageindex_query(self, query: str):
    if not _pageindex_enabled or not _pageindex_engine:
        return "PageIndex not enabled", "", {}
    result = await _pageindex_engine.search(query)
    return result.context[:3000], result.reasoning, result.doc_sources

async def _handle_pageindex_index_all(self):
    if not _pageindex_enabled or not _pageindex_builder:
        return {"error": "PageIndex not enabled"}
    documents = []
    for doc_id, meta in self.rag_pipeline.document_metadata.items():
        if not _pageindex_storage.has_tree(doc_id):
            documents.append({
                "doc_id": doc_id,
                "file_path": meta["file_path"],
                "metadata": {"filename": meta["filename"], "page_count": meta.get("page_count", 0)},
            })
    if not documents:
        return {"message": "All documents already indexed"}
    return await _pageindex_builder.batch_index(documents)

def _handle_pageindex_status(self):
    if not _pageindex_enabled:
        return {"error": "PageIndex not enabled"}
    return {
        "trees": [e.to_dict() for e in _pageindex_storage.list_trees()],
        "stats": _pageindex_storage.get_stats(),
    }

def _handle_pageindex_stats(self):
    if not _pageindex_enabled:
        return {"error": "PageIndex not enabled"}
    stats = _pageindex_storage.get_stats()
    if _pageindex_engine:
        stats["query_cache"] = _pageindex_engine.get_stats().get("cache")
    return stats
```

### 10.3 Commit Message

```
feat(pageindex): Phase 6 - admin UI integration
```

---

## 11. Phase 7: Testing Suite

### 11.1 Config Tests

**File**: `tests/test_pageindex_config.py`

```python
"""Tests for PageIndex configuration module."""
import os
import pytest
from pageindex_integration.config import PageIndexConfig, TreeBuildConfig, LLMConfig, SearchConfig


def test_default_config():
    config = PageIndexConfig()
    assert config.root_dir.name == "pageindex"
    assert config.llm.provider == "groq"
    assert config.tree_build.toc_check_pages == 20
    assert config.search.cache_enabled is True


def test_config_paths():
    config = PageIndexConfig(root_dir="/tmp/test_pageindex")
    assert str(config.trees_dir).endswith("trees")
    assert str(config.cache_dir).endswith("cache")
    assert str(config.index_file).endswith("index.json")


def test_config_validation_missing_key():
    config = PageIndexConfig()
    # Temporarily unset API key
    old = os.environ.pop("GROQ_API_KEY", None)
    try:
        errors = config.validate()
        assert any("API key" in e for e in errors)
    finally:
        if old:
            os.environ["GROQ_API_KEY"] = old


def test_config_validation_invalid_params():
    config = PageIndexConfig()
    config.tree_build.max_pages_per_node = 0
    errors = config.validate()
    assert any("max_pages_per_node" in e for e in errors)


def test_llm_config_groq():
    llm = LLMConfig(provider="groq")
    assert llm.effective_model == "qwen/qwen3-32b"
    assert "groq" in llm.base_url


def test_llm_config_openai():
    llm = LLMConfig(provider="openai")
    assert llm.effective_model == "gpt-4o"
    assert "openai" in llm.base_url


def test_config_to_dict():
    config = PageIndexConfig()
    d = config.to_dict()
    assert "root_dir" in d
    assert "tree_build" in d
    assert "llm" in d
    assert "search" in d
```

### 11.2 Storage Tests

**File**: `tests/test_pageindex_storage.py`

```python
"""Tests for PageIndex storage layer."""
import json
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
    assert storage.delete_tree("doc1")
    assert not storage.has_tree("doc1")


def test_list_trees(tmp_storage):
    storage, _ = tmp_storage
    tree = {"node_id": "0001", "title": "Test", "children": []}
    storage.save_tree("doc1", tree, {"filename": "a.pdf"})
    storage.save_tree("doc2", tree, {"filename": "b.pdf"})
    entries = storage.list_trees()
    assert len(entries) == 2


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
```

### 11.3 LLM Adapter Tests

**File**: `tests/test_pageindex_llm_adapter.py`

```python
"""Tests for PageIndex LLM adapter."""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from pageindex_integration.config import LLMConfig
from pageindex_integration.llm_adapter import LLMAdapter


@pytest.fixture
def groq_adapter():
    config = LLMConfig(provider="groq", model="test-model")
    return LLMAdapter(config)


def test_adapter_headers(groq_adapter):
    headers = groq_adapter._headers()
    assert "Authorization" in headers
    assert "Content-Type" in headers


def test_adapter_payload(groq_adapter):
    messages = [{"role": "user", "content": "hello"}]
    payload = groq_adapter._payload(messages)
    assert payload["model"] == "test-model"
    assert payload["messages"] == messages
    assert payload["temperature"] == 0.0


def test_adapter_repr(groq_adapter):
    r = repr(groq_adapter)
    assert "groq" in r
    assert "test-model" in r
```

### 11.4 Query Engine Tests

**File**: `tests/test_pageindex_query_engine.py`

```python
"""Tests for PageIndex query engine."""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from pageindex_integration.query_engine import PageIndexSearchResult, _extract_json


def test_search_result_to_dict():
    result = PageIndexSearchResult(
        query="test",
        context="some context",
        selected_nodes=[],
        reasoning="because",
        doc_sources=[],
        confidence=0.8,
        duration_ms=100.0,
    )
    d = result.to_dict()
    assert d["query"] == "test"
    assert d["confidence"] == 0.8


def test_extract_json_direct():
    assert _extract_json('{"node_list": ["0001"]}') is not None


def test_extract_json_code_fence():
    text = '```json\n{"node_list": ["0001"]}\n```'
    assert _extract_json(text) is not None


def test_extract_json_with_text():
    text = 'Here is my analysis:\n{"node_list": ["0001", "0002"]}'
    result = _extract_json(text)
    assert result is not None
```

### 11.5 Tree Builder Tests

**File**: `tests/test_pageindex_tree_builder.py`

```python
"""Tests for PageIndex tree builder."""
import os
import pytest
import tempfile
from pathlib import Path
from pageindex_integration.config import PageIndexConfig
from pageindex_integration.storage import PageIndexStorage
from pageindex_integration.tree_builder import PageIndexTreeBuilder, IndexingStatus


@pytest.fixture
def builder_setup():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PageIndexConfig(root_dir=tmpdir)
        storage = PageIndexStorage(config)
        builder = PageIndexTreeBuilder(config, storage)
        yield builder, storage, tmpdir


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

    import asyncio
    # Mock LLM for summary generation
    from unittest.mock import AsyncMock
    builder._llm.chat_async = AsyncMock(return_value="Test summary")

    result = asyncio.get_event_loop().run_until_complete(
        builder.build_tree(md_path, "test_doc", {"filename": "test.md"})
    )
    assert result["status"] == "completed"
    assert result["node_count"] > 0
    assert storage.has_tree("test_doc")


def test_indexing_status(builder_setup):
    builder, _, _ = builder_setup
    assert builder.get_status("unknown") == IndexingStatus.PENDING.value
```

### 11.6 Integration Tests

**File**: `tests/test_pageindex_integration.py`

```python
"""End-to-end integration tests for PageIndex."""
import os
import pytest
import tempfile
import asyncio
from unittest.mock import AsyncMock
from pageindex_integration.config import PageIndexConfig
from pageindex_integration.storage import PageIndexStorage
from pageindex_integration.tree_builder import PageIndexTreeBuilder
from pageindex_integration.query_engine import PageIndexQueryEngine


@pytest.fixture
def integration_setup():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PageIndexConfig(root_dir=tmpdir)
        storage = PageIndexStorage(config)
        builder = PageIndexTreeBuilder(config, storage)
        engine = PageIndexQueryEngine(config, storage)

        # Mock LLM calls
        builder._llm.chat_async = AsyncMock(return_value="Test summary of medical content")
        engine._llm.chat_async = AsyncMock(return_value='{"thinking": "Section about pain management is relevant", "node_list": ["0002"]}')

        yield config, storage, builder, engine, tmpdir


def test_full_pipeline(integration_setup):
    config, storage, builder, engine, tmpdir = integration_setup

    # Create test markdown
    md_path = os.path.join(tmpdir, "test_doc.md")
    with open(md_path, 'w') as f:
        f.write("""# Pain Management Guide

## Assessment
Use the Numeric Rating Scale (NRS) 0-10.

## WHO Analgesic Ladder
Step 1: Non-opioid. Step 2: Weak opioid. Step 3: Strong opioid.

## Opioid Dosing
Start low, titrate slowly. Morphine 5-10mg PO q4h.
""")

    loop = asyncio.get_event_loop()

    # Build tree
    result = loop.run_until_complete(
        builder.build_tree(md_path, "pain_guide", {"filename": "test_doc.md"})
    )
    assert result["status"] == "completed"

    # Query
    search_result = loop.run_until_complete(
        engine.search("What are opioid dosing guidelines?")
    )
    assert search_result.query == "What are opioid dosing guidelines?"
    assert len(search_result.selected_nodes) > 0

    # Stats
    stats = storage.get_stats()
    assert stats["total_trees"] == 1
```

### 11.7 Running Tests

```bash
# All PageIndex tests
pytest tests/test_pageindex_*.py -v

# Specific module
pytest tests/test_pageindex_config.py -v
pytest tests/test_pageindex_storage.py -v

# With coverage
pytest tests/test_pageindex_*.py -v --cov=pageindex_integration
```

### 11.8 Commit Message

```
test(pageindex): Phase 7 - comprehensive test suite
```

---

## 12. API Reference

### 12.1 Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/pageindex/health` | Health check and status |
| `GET` | `/api/pageindex/stats` | Tree index statistics |
| `POST` | `/api/pageindex/query` | Query via tree search |
| `POST` | `/api/pageindex/index` | Index a single document |
| `POST` | `/api/pageindex/index/batch` | Batch index all documents |
| `GET` | `/api/pageindex/index/status` | Get indexing status |
| `GET` | `/api/pageindex/trees` | List all tree indexes |
| `GET` | `/api/pageindex/tree/{doc_id}` | Get tree for a document |

### 12.2 Request/Response Examples

**Health Check**:
```
GET /api/pageindex/health

Response:
{
    "available": true,
    "enabled": true,
    "rag_method": "pageindex",
    "provider": "groq"
}
```

**Query**:
```
POST /api/pageindex/query
Content-Type: application/x-www-form-urlencoded

query=What+are+morphine+dosing+guidelines

Response:
{
    "query": "What are morphine dosing guidelines",
    "context": "Source: palliative_care.pdf (pg 53-61)\nThe WHO analgesic ladder...",
    "selected_nodes": [
        {"node_id": "0003", "title": "3.2 WHO Analgesic Ladder", "start_page": 53, "end_page": 61}
    ],
    "reasoning": "Section 3.2 covers the WHO analgesic ladder including morphine dosing...",
    "doc_sources": [
        {"doc_id": "abc123", "filename": "palliative_care.pdf", "selected_nodes": 1, "page_ranges": ["pg 53-61"]}
    ],
    "confidence": 0.8,
    "duration_ms": 1250.5
}
```

**Batch Index**:
```
POST /api/pageindex/index/batch

Response:
{
    "total": 7,
    "completed": 6,
    "failed": 1,
    "results": [
        {"doc_id": "abc123", "status": "completed", "node_count": 47},
        {"doc_id": "def456", "status": "failed", "error": "PDF has no extractable text"}
    ]
}
```

---

## 13. Configuration Reference

### 13.1 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PAGEINDEX_LLM_PROVIDER` | `groq` | LLM backend: `groq` or `openai` |
| `PAGEINDEX_MODEL` | (auto) | Model override. Empty = provider default |
| `GROQ_API_KEY` | (required) | Groq API key (reused from existing config) |
| `OPENAI_API_KEY` | (optional) | Required only if provider=openai |

### 13.2 YAML Configuration

Add to `config.yaml`:

```yaml
pageindex:
  root_dir: "./data/pageindex"
  tree_build:
    toc_check_pages: 20
    max_pages_per_node: 10
    max_tokens_per_node: 20000
    add_node_summary: true
    summary_max_tokens: 200
    min_section_chars: 100
  llm:
    provider: "groq"
    model: ""  # empty = use provider default
    temperature: 0.0
    max_tokens: 4096
    max_retries: 3
    rate_limit_rpm: 30
  search:
    max_nodes_per_query: 5
    max_context_tokens: 8000
    cache_enabled: true
    cache_maxsize: 100
    cache_ttl_seconds: 3600
```

### 13.3 CLI Arguments

| Argument | Values | Default | Description |
|----------|--------|---------|-------------|
| `--rag-method` | `vector`, `pageindex`, `hybrid` | `vector` | Retrieval method |
| `--host` | IP address | `0.0.0.0` | Host binding |
| `--port` | Integer | `8001` | Port binding |
| `--provider` | `g/b/r` | `b` | Voice AI provider |
| `--no-ngrok` | Flag | False | Disable ngrok |

---

## 14. Troubleshooting

### 14.1 Common Errors

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: pageindex_integration` | Ensure `pageindex_integration/` dir exists with `__init__.py` |
| `ImportError: fitz` | Run `pip install pymupdf>=1.26.0` |
| `ImportError: tiktoken` | Run `pip install tiktoken>=0.5.0` |
| `RuntimeError: LLM request failed` | Check API key and rate limits |
| `No tree indexes available` | Run batch indexing first via admin UI or API |
| `Tree file missing` | Re-index the document |
| `Rate limited (429)` | Reduce `rate_limit_rpm` in config or wait |

### 14.2 Debugging Commands

```bash
# Check PageIndex module
python -c "from pageindex_integration import PageIndexConfig; print(PageIndexConfig())"

# Check LLM adapter
python -c "from pageindex_integration import LLMAdapter; from pageindex_integration.config import LLMConfig; a = LLMAdapter(LLMConfig()); print(a)"

# Check tree storage
python -c "from pageindex_integration import PageIndexConfig, PageIndexStorage; c = PageIndexConfig(); s = PageIndexStorage(c); print(s.get_stats())"

# Test tree building on a markdown file
python -c "
import asyncio
from pageindex_integration import PageIndexConfig, PageIndexStorage, PageIndexTreeBuilder
config = PageIndexConfig()
storage = PageIndexStorage(config)
builder = PageIndexTreeBuilder(config, storage)
result = asyncio.run(builder.build_tree('test.md', 'test_id'))
print(result)
"

# Verify server with PageIndex
python simple_rag_server.py --rag-method pageindex --no-ngrok
curl http://localhost:8001/api/pageindex/health
```

### 14.3 Performance Tuning

| Setting | Impact | Recommendation |
|---------|--------|----------------|
| `rate_limit_rpm` | LLM call frequency during indexing | 30 RPM for Groq free tier, 100+ for paid |
| `max_nodes_per_query` | Nodes retrieved per query | 3-5 for focused answers, 5-10 for comprehensive |
| `max_context_tokens` | Context size sent to answer LLM | 8000 for Groq (leaves room for response), 16000 for OpenAI |
| `max_pages_per_node` | Granularity of page-based fallback trees | 5-10 for detailed, 15-20 for coarse |
| `cache_ttl_seconds` | How long query results are cached | 3600 (1hr) for stable docs, 300 for frequently updated |

---

**Document Version**: 1.0.0
**Last Updated**: February 2026
