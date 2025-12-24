# Claude Code Implementation Guide

## Palli Sahayak Voice AI Helpline - AI Agent Instructions

This document provides structured instructions for Claude Code (or any AI coding assistant) to implement features progressively and accurately.

---

## How to Use This Document

When implementing a feature:

1. **Read the relevant section** in this document
2. **Follow phases sequentially** - do not skip phases
3. **Run tests after each phase** - do not proceed if tests fail
4. **Commit after each successful phase**

---

## Current Implementation Tasks

### GraphRAG Integration (Priority: HIGH)

**Specification Document**: `docs/graphrag_specs.md`
**Reference**: AGENTS.md → "Microsoft GraphRAG Integration" section

#### Phase-by-Phase Implementation Instructions

---

### PHASE 1: Foundation Setup

**Objective**: Create directory structure and install dependencies

**Steps**:

1. Create the setup script:
```bash
# Create directory structure
mkdir -p graphrag_integration/prompts
mkdir -p data/graphrag/{input,output/artifacts,cache}
mkdir -p tests
```

2. Create `graphrag_integration/__init__.py`:
```python
"""
GraphRAG Integration Module for Palli Sahayak
"""

from graphrag_integration.config import GraphRAGConfig
from graphrag_integration.indexer import GraphRAGIndexer
from graphrag_integration.query_engine import GraphRAGQueryEngine
from graphrag_integration.data_loader import GraphRAGDataLoader

__all__ = [
    "GraphRAGConfig",
    "GraphRAGIndexer",
    "GraphRAGQueryEngine",
    "GraphRAGDataLoader",
]

__version__ = "1.0.0"
```

3. Create symlink from `data/graphrag/input` to `uploads/`

4. Add dependencies to `requirements.txt`:
```
graphrag>=2.7.0
graspologic>=3.3.0
pandas>=2.0.0
pyarrow>=14.0.0
lancedb>=0.4.0
```

**Tests to Run**:
```bash
python -c "from pathlib import Path; assert Path('graphrag_integration').exists()"
python -c "from pathlib import Path; assert Path('data/graphrag/output').exists()"
```

**Commit Message**: `feat(graphrag): Phase 1 - foundation setup`

---

### PHASE 2: Configuration Module

**Objective**: Implement configuration management

**Files to Create**:

1. `data/graphrag/settings.yaml` - Copy from `docs/graphrag_specs.md` Section 6.2

2. `graphrag_integration/config.py` - Copy full implementation from `docs/graphrag_specs.md` Section 6.2.2

**Key Classes**:
- `ModelConfig` - dataclass for LLM configuration
- `ChunkingConfig` - dataclass for chunking settings
- `EntityExtractionConfig` - dataclass for extraction settings
- `SearchConfig` - dataclass for search settings
- `GraphRAGConfig` - main configuration class

**Tests to Run**:
```bash
python -c "from graphrag_integration.config import GraphRAGConfig; c = GraphRAGConfig('./data/graphrag'); print(c)"
python tests/test_graphrag_config.py
```

**Commit Message**: `feat(graphrag): Phase 2 - configuration module`

---

### PHASE 3: Indexing Pipeline

**Objective**: Implement document indexing

**Files to Create**:

1. `graphrag_integration/indexer.py` - Copy from `docs/graphrag_specs.md` Section 7.2.1

2. `graphrag_integration/prompts/entity_extraction.txt` - Copy from Section 7.2.2

3. `graphrag_integration/prompts/community_report.txt` - Copy from Section 7.2.2

4. `graphrag_integration/prompts/__init__.py`:
```python
"""Custom prompts for palliative care entity extraction."""
```

**Key Classes**:
- `IndexingMethod` - enum (STANDARD, FAST)
- `IndexingStatus` - enum (PENDING, RUNNING, COMPLETED, FAILED)
- `GraphRAGIndexer` - main indexer class

**Tests to Run**:
```bash
python -c "from graphrag_integration.indexer import GraphRAGIndexer, IndexingMethod; print(IndexingMethod.STANDARD)"
python tests/test_graphrag_indexer.py
```

**Commit Message**: `feat(graphrag): Phase 3 - indexing pipeline`

---

### PHASE 4: Query Engine

**Objective**: Implement all search methods

**Files to Create**:

1. `graphrag_integration/query_engine.py` - Copy from `docs/graphrag_specs.md` Section 8.2.1

2. `graphrag_integration/data_loader.py` - Copy from Section 8.2.2

**Key Classes**:
- `SearchMethod` - enum (GLOBAL, LOCAL, DRIFT, BASIC)
- `SearchResult` - dataclass for results
- `GraphRAGQueryEngine` - main query class
- `GraphRAGDataLoader` - parquet file loader

**Key Methods**:
- `global_search(query)` - corpus-wide search
- `local_search(query)` - entity-focused search
- `drift_search(query)` - multi-hop reasoning
- `basic_search(query)` - vector similarity
- `auto_search(query)` - automatic method selection

**Tests to Run**:
```bash
python -c "from graphrag_integration.query_engine import GraphRAGQueryEngine, SearchMethod; print(SearchMethod.GLOBAL)"
python tests/test_graphrag_query.py
```

**Commit Message**: `feat(graphrag): Phase 4 - query engine`

---

### PHASE 5: Server Integration

**Objective**: Add FastAPI endpoints

**Files to Modify**:

1. `simple_rag_server.py` - Add GraphRAG endpoints from `docs/graphrag_specs.md` Section 9.2.1

**Endpoints to Add**:
- `GET /api/graphrag/health`
- `GET /api/graphrag/stats`
- `POST /api/graphrag/query`
- `POST /api/graphrag/index`
- `GET /api/graphrag/index/status`
- `GET /api/graphrag/entities`
- `GET /api/graphrag/entity/{name}/relationships`

**Request Models to Add**:
- `GraphRAGQueryRequest`
- `GraphRAGIndexRequest`

**Tests to Run**:
```bash
curl http://localhost:8000/api/graphrag/health
python tests/test_graphrag_server.py
```

**Commit Message**: `feat(graphrag): Phase 5 - server integration`

---

### PHASE 6: Admin UI Integration

**Objective**: Add Gradio admin tabs

**Files to Modify**:

1. `simple_rag_server.py` - Add GraphRAG tab in `create_admin_interface()` from `docs/graphrag_specs.md` Section 10.2

**UI Tabs to Add**:
- Query tab (search with method selection)
- Indexing tab (start/monitor indexing)
- Entity Explorer tab (browse entities)
- Statistics tab (view stats)

**Tests to Run**:
```bash
# Manual test: Navigate to http://localhost:8000/admin and verify GraphRAG tab
```

**Commit Message**: `feat(graphrag): Phase 6 - admin UI integration`

---

### PHASE 7: Testing Suite

**Objective**: Implement comprehensive tests

**Files to Create**:

1. `tests/test_graphrag_config.py` - From Section 6.3
2. `tests/test_graphrag_indexer.py` - From Section 7.3
3. `tests/test_graphrag_query.py` - From Section 8.3
4. `tests/test_graphrag_server.py` - From Section 9.3
5. `tests/test_graphrag_integration.py` - From Section 11.1

**Tests to Run**:
```bash
pytest tests/test_graphrag_*.py -v
pytest tests/test_graphrag_integration.py -v --tb=short
```

**Commit Message**: `test(graphrag): Phase 7 - comprehensive test suite`

---

### PHASE 8: Performance Optimization

**Objective**: Add caching and optimization

**Files to Create/Modify**:

1. `graphrag_integration/utils.py` - Add QueryCache class from Section 12.1

**Optimizations**:
- Query result caching (LRU)
- Async batch processing
- Memory management

**Tests to Run**:
```bash
python -c "from graphrag_integration.utils import QueryCache; c = QueryCache(); print('Cache OK')"
pytest tests/test_graphrag_integration.py -v
```

**Commit Message**: `perf(graphrag): Phase 8 - caching and optimization`

---

## Implementation Checklist

```
[ ] Phase 1: Foundation Setup
    [ ] Directories created
    [ ] __init__.py files created
    [ ] Dependencies added
    [ ] Tests pass

[ ] Phase 2: Configuration Module
    [ ] settings.yaml created
    [ ] config.py implemented
    [ ] Environment variable substitution works
    [ ] Tests pass

[ ] Phase 3: Indexing Pipeline
    [ ] indexer.py implemented
    [ ] Prompts created
    [ ] Progress callbacks work
    [ ] Tests pass

[ ] Phase 4: Query Engine
    [ ] query_engine.py implemented
    [ ] data_loader.py implemented
    [ ] All search methods work
    [ ] Tests pass

[ ] Phase 5: Server Integration
    [ ] Endpoints added
    [ ] Startup initialization works
    [ ] API tests pass

[ ] Phase 6: Admin UI
    [ ] GraphRAG tab added
    [ ] All sub-tabs functional
    [ ] UI tests pass

[ ] Phase 7: Testing Suite
    [ ] All test files created
    [ ] All tests passing
    [ ] Coverage > 80%

[ ] Phase 8: Optimization
    [ ] Caching implemented
    [ ] Performance verified
    [ ] Final tests pass
```

---

## Error Handling Guidelines

### Common Errors and Solutions

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: graphrag` | Run `pip install graphrag>=2.7.0` |
| `FileNotFoundError: settings.yaml` | Create `data/graphrag/settings.yaml` |
| `KeyError: GRAPHRAG_API_KEY` | Set environment variable in `.env` |
| `ImportError: GraphRAGConfig` | Check `__init__.py` exports |
| `Empty parquet files` | Run indexing first |

### Debugging Commands

```bash
# Check GraphRAG installation
python -c "import graphrag; print(graphrag.__version__)"

# Check configuration
python -c "from graphrag_integration.config import GraphRAGConfig; c = GraphRAGConfig.from_yaml('./data/graphrag/settings.yaml'); print(c.validate())"

# Check data loader
python -c "import asyncio; from graphrag_integration.data_loader import GraphRAGDataLoader; from graphrag_integration.config import GraphRAGConfig; c = GraphRAGConfig('./data/graphrag'); d = GraphRAGDataLoader(c); print(asyncio.run(d.get_stats()))"
```

---

## Code Quality Standards

### Before Committing Each Phase:

1. **Syntax Check**:
```bash
python -m py_compile graphrag_integration/*.py
```

2. **Import Check**:
```bash
python -c "from graphrag_integration import *"
```

3. **Type Hints**: All functions must have type hints

4. **Docstrings**: All classes and public methods must have docstrings

5. **Error Handling**: Use try/except with specific exceptions

6. **Logging**: Use `logger.info/error/warning` not `print()`

---

## Git Commit Convention

```
feat(graphrag): Phase N - description

- Bullet point changes
- Another change

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

---

## Quick Reference

### File Locations
- Specs: `docs/graphrag_specs.md`
- Config: `data/graphrag/settings.yaml`
- Module: `graphrag_integration/`
- Tests: `tests/test_graphrag_*.py`

### Key Imports
```python
from graphrag_integration import (
    GraphRAGConfig,
    GraphRAGIndexer,
    GraphRAGQueryEngine,
    GraphRAGDataLoader,
)
from graphrag_integration.query_engine import SearchMethod, SearchResult
from graphrag_integration.indexer import IndexingMethod, IndexingStatus
```

### API Endpoint Patterns
```
GET  /api/graphrag/health
GET  /api/graphrag/stats
POST /api/graphrag/query    {"query": "...", "method": "auto"}
POST /api/graphrag/index    {"method": "standard"}
GET  /api/graphrag/index/status
GET  /api/graphrag/entities?query=...
```

---

**Document Version**: 1.0.0
**Last Updated**: December 2025
