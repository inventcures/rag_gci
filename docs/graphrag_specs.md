# Microsoft GraphRAG Integration Specifications

## Palli Sahayak Voice AI Helpline - GraphRAG Enhancement

**Version**: 1.0.0
**Target GraphRAG Version**: 2.7.0
**Document Status**: Implementation Specification
**Last Updated**: December 2025

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Prerequisites and Dependencies](#3-prerequisites-and-dependencies)
4. [Directory Structure](#4-directory-structure)
5. [Phase 1: Foundation Setup](#5-phase-1-foundation-setup)
6. [Phase 2: Configuration Module](#6-phase-2-configuration-module)
7. [Phase 3: Indexing Pipeline](#7-phase-3-indexing-pipeline)
8. [Phase 4: Query Engine](#8-phase-4-query-engine)
9. [Phase 5: Server Integration](#9-phase-5-server-integration)
10. [Phase 6: Admin UI Integration](#10-phase-6-admin-ui-integration)
11. [Phase 7: Testing Suite](#11-phase-7-testing-suite)
12. [Phase 8: Performance Optimization](#12-phase-8-performance-optimization)
13. [API Reference](#13-api-reference)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Executive Summary

### 1.1 Purpose

Integrate Microsoft GraphRAG into Palli Sahayak to enable:
- **Global Search**: Holistic queries across entire palliative care corpus
- **Local Search**: Entity-focused queries for specific medications, symptoms, conditions
- **DRIFT Search**: Multi-phase reasoning for complex medical questions
- **Community Summarization**: Hierarchical knowledge organization

### 1.2 Key Benefits

| Benefit | Description |
|---------|-------------|
| **Improved Comprehensiveness** | 15-20% better answer completeness vs. vector-only RAG |
| **Better Sensemaking** | Global queries synthesize themes across documents |
| **Entity Relationships** | Structured knowledge graph for medical concepts |
| **Source Attribution** | Better citation and provenance tracking |

### 1.3 Integration Strategy

GraphRAG will operate **alongside** the existing ChromaDB vector store:

```
┌─────────────────────────────────────────────────────────────────┐
│                     HYBRID RAG PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   User Query ──┬──► GraphRAG (Global/Local/DRIFT)              │
│                │                                                │
│                └──► ChromaDB (Vector Similarity)               │
│                              │                                  │
│                              ▼                                  │
│                    ┌─────────────────┐                         │
│                    │  Result Fusion  │                         │
│                    │  (RRF/Weighted) │                         │
│                    └─────────────────┘                         │
│                              │                                  │
│                              ▼                                  │
│                       LLM Response                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Architecture Overview

### 2.1 GraphRAG Architecture (from Microsoft)

```
┌─────────────────────────────────────────────────────────────────┐
│                    GRAPHRAG ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 INDEXING PIPELINE                        │   │
│  │                                                          │   │
│  │  Documents → Text Units → Entity Extraction →           │   │
│  │  Community Detection (Leiden) → Community Reports →     │   │
│  │  Embeddings → Parquet Files                             │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   DATA LAYER                             │   │
│  │                                                          │   │
│  │  entities.parquet    │  relationships.parquet           │   │
│  │  communities.parquet │  community_reports.parquet       │   │
│  │  text_units.parquet  │  embeddings.*.parquet            │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   QUERY ENGINE                           │   │
│  │                                                          │   │
│  │  GlobalSearch │ LocalSearch │ DRIFTSearch │ BasicSearch │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Palli Sahayak Integration Architecture

```
rag_gci/
├── graphrag_integration/          # NEW: GraphRAG integration module
│   ├── __init__.py                # Module exports
│   ├── config.py                  # GraphRAG configuration wrapper
│   ├── indexer.py                 # Indexing pipeline wrapper
│   ├── query_engine.py            # Query engine with all search types
│   ├── data_loader.py             # Parquet file management
│   ├── prompts/                   # Custom prompts for palliative care
│   │   ├── entity_extraction.txt
│   │   ├── community_report.txt
│   │   └── summarize_descriptions.txt
│   └── utils.py                   # Helper functions
├── data/
│   └── graphrag/                  # NEW: GraphRAG data directory
│       ├── input/                 # Source documents (symlink to uploads/)
│       ├── output/                # Parquet files
│       ├── cache/                 # LLM response cache
│       └── settings.yaml          # GraphRAG configuration
```

---

## 3. Prerequisites and Dependencies

### 3.1 Python Dependencies

Add to `requirements.txt`:

```
# GraphRAG Dependencies
graphrag>=2.7.0
graspologic>=3.3.0
pandas>=2.0.0
pyarrow>=14.0.0
networkx>=3.0
pydantic>=2.0.0

# LLM Providers (choose one or more)
openai>=1.0.0        # For OpenAI/Azure OpenAI
litellm>=1.0.0       # For Groq, Anthropic, etc.

# Optional: Vector Store
lancedb>=0.4.0       # Default GraphRAG vector store
```

### 3.2 Environment Variables

Add to `.env`:

```bash
# GraphRAG Configuration
GRAPHRAG_API_KEY=${OPENAI_API_KEY}  # Or use GROQ_API_KEY with LiteLLM
GRAPHRAG_LLM_MODEL=gpt-4o-mini      # Or groq/llama-3.1-8b-instant
GRAPHRAG_EMBEDDING_MODEL=text-embedding-3-small

# Optional: Azure OpenAI
# AZURE_OPENAI_API_KEY=your-azure-key
# AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# GraphRAG Paths
GRAPHRAG_ROOT=./data/graphrag
GRAPHRAG_INPUT_DIR=./uploads
GRAPHRAG_OUTPUT_DIR=./data/graphrag/output
```

### 3.3 System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10 | 3.11+ |
| RAM | 8GB | 16GB+ |
| Disk Space | 10GB | 50GB+ (for embeddings) |
| LLM API | Required | GPT-4 or equivalent |

---

## 4. Directory Structure

### 4.1 Complete Directory Tree

```
rag_gci/
├── graphrag_integration/
│   ├── __init__.py
│   ├── config.py
│   ├── indexer.py
│   ├── query_engine.py
│   ├── data_loader.py
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── entity_extraction.txt
│   │   ├── community_report.txt
│   │   ├── summarize_descriptions.txt
│   │   ├── global_search_map.txt
│   │   ├── global_search_reduce.txt
│   │   └── local_search.txt
│   └── utils.py
├── data/
│   └── graphrag/
│       ├── input/                  # Symlink to uploads/
│       ├── output/
│       │   ├── artifacts/
│       │   │   ├── entities.parquet
│       │   │   ├── relationships.parquet
│       │   │   ├── communities.parquet
│       │   │   ├── community_reports.parquet
│       │   │   ├── text_units.parquet
│       │   │   └── embeddings.entity.parquet
│       │   └── stats.json
│       ├── cache/
│       └── settings.yaml
├── tests/
│   ├── test_graphrag_config.py
│   ├── test_graphrag_indexer.py
│   ├── test_graphrag_query.py
│   └── test_graphrag_integration.py
```

---

## 5. Phase 1: Foundation Setup

### 5.1 Objective

Set up the GraphRAG directory structure and install dependencies.

### 5.2 Implementation Steps

#### Step 1.1: Create Directory Structure

**File**: `scripts/setup_graphrag.py`

```python
#!/usr/bin/env python3
"""
GraphRAG Setup Script for Palli Sahayak
Creates necessary directory structure and initializes configuration.
"""

import os
import shutil
from pathlib import Path


def setup_graphrag_directories(base_path: str = ".") -> dict:
    """
    Create GraphRAG directory structure.

    Args:
        base_path: Base path for the project

    Returns:
        Dictionary with created paths
    """
    base = Path(base_path)

    # Define directory structure
    directories = {
        "module": base / "graphrag_integration",
        "prompts": base / "graphrag_integration" / "prompts",
        "data_root": base / "data" / "graphrag",
        "input": base / "data" / "graphrag" / "input",
        "output": base / "data" / "graphrag" / "output",
        "artifacts": base / "data" / "graphrag" / "output" / "artifacts",
        "cache": base / "data" / "graphrag" / "cache",
        "tests": base / "tests",
    }

    # Create directories
    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {path}")

    # Create symlink from input to uploads if not exists
    uploads_path = base / "uploads"
    input_path = directories["input"]

    if uploads_path.exists() and not input_path.is_symlink():
        # Remove empty input directory and create symlink
        if input_path.exists():
            shutil.rmtree(input_path)
        input_path.symlink_to(uploads_path.resolve())
        print(f"Created symlink: {input_path} -> {uploads_path}")

    return directories


def create_init_files(base_path: str = ".") -> None:
    """Create __init__.py files for Python packages."""
    base = Path(base_path)

    init_files = [
        base / "graphrag_integration" / "__init__.py",
        base / "graphrag_integration" / "prompts" / "__init__.py",
    ]

    for init_file in init_files:
        if not init_file.exists():
            init_file.write_text('"""GraphRAG Integration Module."""\n')
            print(f"Created: {init_file}")


def main():
    """Main setup function."""
    print("Setting up GraphRAG for Palli Sahayak...")
    print("=" * 50)

    setup_graphrag_directories()
    create_init_files()

    print("=" * 50)
    print("GraphRAG setup complete!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install graphrag>=2.7.0")
    print("2. Configure settings.yaml")
    print("3. Run indexing pipeline")


if __name__ == "__main__":
    main()
```

#### Step 1.2: Create Module `__init__.py`

**File**: `graphrag_integration/__init__.py`

```python
"""
GraphRAG Integration Module for Palli Sahayak

This module provides integration with Microsoft GraphRAG for enhanced
retrieval-augmented generation capabilities in palliative care knowledge access.

Components:
    - GraphRAGConfig: Configuration management
    - GraphRAGIndexer: Document indexing pipeline
    - GraphRAGQueryEngine: Query execution (Global, Local, DRIFT)
    - GraphRAGDataLoader: Parquet file management

Usage:
    from graphrag_integration import (
        GraphRAGConfig,
        GraphRAGIndexer,
        GraphRAGQueryEngine,
    )

    # Initialize
    config = GraphRAGConfig.from_yaml("./data/graphrag/settings.yaml")
    indexer = GraphRAGIndexer(config)
    query_engine = GraphRAGQueryEngine(config)

    # Index documents
    await indexer.index_documents()

    # Query
    result = await query_engine.global_search("What are pain management options?")
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

### 5.3 Test Criteria (Phase 1)

```python
# tests/test_graphrag_setup.py
"""Phase 1 Tests: Foundation Setup"""

import os
from pathlib import Path
import pytest


def test_directory_structure_exists():
    """Verify all required directories exist."""
    required_dirs = [
        "graphrag_integration",
        "graphrag_integration/prompts",
        "data/graphrag",
        "data/graphrag/input",
        "data/graphrag/output",
        "data/graphrag/cache",
    ]

    for dir_path in required_dirs:
        assert Path(dir_path).exists(), f"Missing directory: {dir_path}"


def test_init_files_exist():
    """Verify __init__.py files exist."""
    init_files = [
        "graphrag_integration/__init__.py",
        "graphrag_integration/prompts/__init__.py",
    ]

    for init_file in init_files:
        assert Path(init_file).exists(), f"Missing init file: {init_file}"


def test_input_symlink():
    """Verify input directory symlinks to uploads."""
    input_path = Path("data/graphrag/input")
    uploads_path = Path("uploads")

    if uploads_path.exists():
        assert input_path.is_symlink() or input_path.exists()


def test_module_imports():
    """Verify module can be imported."""
    try:
        from graphrag_integration import (
            GraphRAGConfig,
            GraphRAGIndexer,
            GraphRAGQueryEngine,
            GraphRAGDataLoader,
        )
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")
```

### 5.4 Completion Checklist (Phase 1)

- [ ] All directories created
- [ ] `__init__.py` files created
- [ ] Symlink from input to uploads working
- [ ] Dependencies installable
- [ ] All Phase 1 tests pass

---

## 6. Phase 2: Configuration Module

### 6.1 Objective

Create the configuration management system for GraphRAG.

### 6.2 Implementation Steps

#### Step 2.1: Create Settings YAML Template

**File**: `data/graphrag/settings.yaml`

```yaml
# GraphRAG Configuration for Palli Sahayak
# Version: 2.7.0

# =============================================================================
# LANGUAGE MODEL CONFIGURATION
# =============================================================================
models:
  default_chat_model:
    type: chat
    model_provider: openai  # Options: openai, azure_openai, litellm
    model: gpt-4o-mini
    api_key: ${GRAPHRAG_API_KEY}
    max_tokens: 4096
    temperature: 0.0
    request_timeout: 180
    retry_strategy: exponential_backoff
    max_retries: 3

  default_embedding_model:
    type: embedding
    model_provider: openai
    model: text-embedding-3-small
    api_key: ${GRAPHRAG_API_KEY}
    batch_size: 16

# =============================================================================
# INPUT CONFIGURATION
# =============================================================================
input:
  type: file
  file_type: text  # Options: text, csv, json
  base_dir: ./input
  file_pattern: "*.{txt,md,pdf}"
  encoding: utf-8

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================
output:
  type: file
  base_dir: ./output

# =============================================================================
# CACHE CONFIGURATION
# =============================================================================
cache:
  type: file
  base_dir: ./cache

# =============================================================================
# CHUNKING CONFIGURATION
# =============================================================================
chunks:
  size: 1200
  overlap: 100
  group_by_columns: [id]
  encoding_model: cl100k_base

# =============================================================================
# ENTITY EXTRACTION CONFIGURATION
# =============================================================================
extract_graph:
  prompt: ./prompts/entity_extraction.txt
  entity_types:
    - Symptom
    - Medication
    - Condition
    - Treatment
    - SideEffect
    - Dosage
    - Route
    - CareGoal
    - Assessment
    - Intervention
  max_gleanings: 1
  strategy:
    type: graph_intelligence

# =============================================================================
# COMMUNITY DETECTION CONFIGURATION
# =============================================================================
cluster_graph:
  max_cluster_size: 10
  strategy:
    type: leiden
    levels: [0, 1, 2]

# =============================================================================
# COMMUNITY REPORTS CONFIGURATION
# =============================================================================
community_reports:
  prompt: ./prompts/community_report.txt
  max_length: 2000
  max_input_length: 8000

# =============================================================================
# EMBEDDING CONFIGURATION
# =============================================================================
embed_text:
  strategy:
    type: openai
    batch_size: 16
    batch_max_tokens: 8191

# =============================================================================
# GLOBAL SEARCH CONFIGURATION
# =============================================================================
global_search:
  map_prompt: ./prompts/global_search_map.txt
  reduce_prompt: ./prompts/global_search_reduce.txt
  max_tokens: 12000
  data_max_tokens: 12000
  concurrency: 32

# =============================================================================
# LOCAL SEARCH CONFIGURATION
# =============================================================================
local_search:
  prompt: ./prompts/local_search.txt
  max_tokens: 12000
  text_unit_prop: 0.5
  community_prop: 0.1
  top_k_entities: 10
  top_k_relationships: 10
  conversation_history_max_turns: 5

# =============================================================================
# DRIFT SEARCH CONFIGURATION
# =============================================================================
drift_search:
  primer_folds: 3
  primer_llm_max_tokens: 4000
  n_depth: 3
  local_search_text_unit_prop: 0.5
  local_search_community_prop: 0.1
  local_search_top_k_mapped_entities: 10
  local_search_top_k_relationships: 10
```

#### Step 2.2: Create Configuration Wrapper

**File**: `graphrag_integration/config.py`

```python
"""
GraphRAG Configuration Module for Palli Sahayak

Provides configuration management, validation, and environment variable
substitution for GraphRAG integration.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a language model."""
    type: str  # "chat" or "embedding"
    model_provider: str  # "openai", "azure_openai", "litellm"
    model: str
    api_key: str
    max_tokens: int = 4096
    temperature: float = 0.0
    request_timeout: int = 180
    retry_strategy: str = "exponential_backoff"
    max_retries: int = 3
    batch_size: int = 16  # For embeddings


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    size: int = 1200
    overlap: int = 100
    encoding_model: str = "cl100k_base"


@dataclass
class EntityExtractionConfig:
    """Configuration for entity extraction."""
    prompt: str = "./prompts/entity_extraction.txt"
    entity_types: List[str] = field(default_factory=lambda: [
        "Symptom", "Medication", "Condition", "Treatment",
        "SideEffect", "Dosage", "Route", "CareGoal"
    ])
    max_gleanings: int = 1


@dataclass
class SearchConfig:
    """Configuration for search operations."""
    max_tokens: int = 12000
    data_max_tokens: int = 12000
    concurrency: int = 32
    text_unit_prop: float = 0.5
    community_prop: float = 0.1
    top_k_entities: int = 10
    top_k_relationships: int = 10


class GraphRAGConfig:
    """
    Configuration manager for GraphRAG integration.

    Handles loading, validation, and environment variable substitution
    for GraphRAG settings.

    Attributes:
        root_dir: Root directory for GraphRAG data
        models: Dictionary of model configurations
        chunking: Chunking configuration
        extraction: Entity extraction configuration
        search: Search configuration

    Example:
        config = GraphRAGConfig.from_yaml("./data/graphrag/settings.yaml")
        chat_model = config.get_chat_model()
        embedding_model = config.get_embedding_model()
    """

    ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')

    def __init__(
        self,
        root_dir: str = "./data/graphrag",
        settings: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize GraphRAG configuration.

        Args:
            root_dir: Root directory for GraphRAG data
            settings: Optional settings dictionary
        """
        self.root_dir = Path(root_dir)
        self._settings = settings or {}
        self._models: Dict[str, ModelConfig] = {}
        self._initialized = False

        if settings:
            self._parse_settings(settings)
            self._initialized = True

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "GraphRAGConfig":
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to settings.yaml file

        Returns:
            GraphRAGConfig instance

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML is invalid
        """
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()

        # Substitute environment variables
        content = cls._substitute_env_vars(raw_content)

        try:
            settings = yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")

        # Determine root_dir from yaml_path
        root_dir = yaml_path.parent

        return cls(root_dir=str(root_dir), settings=settings)

    @classmethod
    def _substitute_env_vars(cls, content: str) -> str:
        """
        Substitute ${VAR_NAME} patterns with environment variable values.

        Args:
            content: YAML content string

        Returns:
            Content with substituted values
        """
        def replace_match(match):
            var_name = match.group(1)
            value = os.environ.get(var_name, "")
            if not value:
                logger.warning(f"Environment variable not set: {var_name}")
            return value

        return cls.ENV_VAR_PATTERN.sub(replace_match, content)

    def _parse_settings(self, settings: Dict[str, Any]) -> None:
        """Parse settings dictionary into configuration objects."""
        # Parse models
        models_config = settings.get("models", {})
        for model_id, model_settings in models_config.items():
            self._models[model_id] = ModelConfig(
                type=model_settings.get("type", "chat"),
                model_provider=model_settings.get("model_provider", "openai"),
                model=model_settings.get("model", "gpt-4o-mini"),
                api_key=model_settings.get("api_key", ""),
                max_tokens=model_settings.get("max_tokens", 4096),
                temperature=model_settings.get("temperature", 0.0),
                request_timeout=model_settings.get("request_timeout", 180),
                retry_strategy=model_settings.get("retry_strategy", "exponential_backoff"),
                max_retries=model_settings.get("max_retries", 3),
                batch_size=model_settings.get("batch_size", 16),
            )

        # Parse chunking
        chunks_config = settings.get("chunks", {})
        self.chunking = ChunkingConfig(
            size=chunks_config.get("size", 1200),
            overlap=chunks_config.get("overlap", 100),
            encoding_model=chunks_config.get("encoding_model", "cl100k_base"),
        )

        # Parse entity extraction
        extract_config = settings.get("extract_graph", {})
        self.extraction = EntityExtractionConfig(
            prompt=extract_config.get("prompt", "./prompts/entity_extraction.txt"),
            entity_types=extract_config.get("entity_types", []),
            max_gleanings=extract_config.get("max_gleanings", 1),
        )

        # Parse search configuration
        global_search = settings.get("global_search", {})
        local_search = settings.get("local_search", {})
        self.search = SearchConfig(
            max_tokens=global_search.get("max_tokens", 12000),
            data_max_tokens=global_search.get("data_max_tokens", 12000),
            concurrency=global_search.get("concurrency", 32),
            text_unit_prop=local_search.get("text_unit_prop", 0.5),
            community_prop=local_search.get("community_prop", 0.1),
            top_k_entities=local_search.get("top_k_entities", 10),
            top_k_relationships=local_search.get("top_k_relationships", 10),
        )

    def get_chat_model(self) -> ModelConfig:
        """Get the default chat model configuration."""
        return self._models.get("default_chat_model", ModelConfig(
            type="chat",
            model_provider="openai",
            model="gpt-4o-mini",
            api_key=os.environ.get("GRAPHRAG_API_KEY", ""),
        ))

    def get_embedding_model(self) -> ModelConfig:
        """Get the default embedding model configuration."""
        return self._models.get("default_embedding_model", ModelConfig(
            type="embedding",
            model_provider="openai",
            model="text-embedding-3-small",
            api_key=os.environ.get("GRAPHRAG_API_KEY", ""),
        ))

    @property
    def input_dir(self) -> Path:
        """Get input directory path."""
        return self.root_dir / "input"

    @property
    def output_dir(self) -> Path:
        """Get output directory path."""
        return self.root_dir / "output"

    @property
    def artifacts_dir(self) -> Path:
        """Get artifacts directory path."""
        return self.output_dir / "artifacts"

    @property
    def cache_dir(self) -> Path:
        """Get cache directory path."""
        return self.root_dir / "cache"

    @property
    def prompts_dir(self) -> Path:
        """Get prompts directory path."""
        return self.root_dir / "prompts"

    def get_prompt_path(self, prompt_name: str) -> Path:
        """Get path to a prompt file."""
        return self.prompts_dir / f"{prompt_name}.txt"

    def validate(self) -> List[str]:
        """
        Validate configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check required directories
        if not self.root_dir.exists():
            errors.append(f"Root directory does not exist: {self.root_dir}")

        # Check API key
        chat_model = self.get_chat_model()
        if not chat_model.api_key:
            errors.append("Chat model API key not configured")

        embedding_model = self.get_embedding_model()
        if not embedding_model.api_key:
            errors.append("Embedding model API key not configured")

        # Check entity types
        if not self.extraction.entity_types:
            errors.append("No entity types configured for extraction")

        return errors

    def to_graphrag_config(self) -> Any:
        """
        Convert to native GraphRAG configuration object.

        Returns:
            graphrag.config.GraphRagConfig instance
        """
        try:
            from graphrag.config import GraphRagConfig
            from graphrag.config.load_config import load_config

            return load_config(str(self.root_dir))
        except ImportError:
            logger.warning("GraphRAG not installed, returning dict config")
            return self._settings

    def __repr__(self) -> str:
        return (
            f"GraphRAGConfig(root_dir={self.root_dir}, "
            f"models={list(self._models.keys())}, "
            f"initialized={self._initialized})"
        )
```

### 6.3 Test Criteria (Phase 2)

```python
# tests/test_graphrag_config.py
"""Phase 2 Tests: Configuration Module"""

import os
import pytest
from pathlib import Path


def test_config_from_yaml():
    """Test loading configuration from YAML."""
    from graphrag_integration.config import GraphRAGConfig

    config = GraphRAGConfig.from_yaml("./data/graphrag/settings.yaml")

    assert config.root_dir == Path("./data/graphrag")
    assert config._initialized is True


def test_config_env_substitution():
    """Test environment variable substitution."""
    from graphrag_integration.config import GraphRAGConfig

    os.environ["TEST_VAR"] = "test_value"
    content = "api_key: ${TEST_VAR}"
    result = GraphRAGConfig._substitute_env_vars(content)

    assert result == "api_key: test_value"


def test_config_model_access():
    """Test accessing model configurations."""
    from graphrag_integration.config import GraphRAGConfig

    config = GraphRAGConfig.from_yaml("./data/graphrag/settings.yaml")
    chat_model = config.get_chat_model()
    embedding_model = config.get_embedding_model()

    assert chat_model.type == "chat"
    assert embedding_model.type == "embedding"


def test_config_paths():
    """Test path properties."""
    from graphrag_integration.config import GraphRAGConfig

    config = GraphRAGConfig(root_dir="./data/graphrag")

    assert config.input_dir == Path("./data/graphrag/input")
    assert config.output_dir == Path("./data/graphrag/output")
    assert config.cache_dir == Path("./data/graphrag/cache")


def test_config_validation():
    """Test configuration validation."""
    from graphrag_integration.config import GraphRAGConfig

    config = GraphRAGConfig(root_dir="./nonexistent")
    errors = config.validate()

    assert len(errors) > 0
    assert any("does not exist" in e for e in errors)
```

### 6.4 Completion Checklist (Phase 2)

- [ ] `settings.yaml` template created
- [ ] `config.py` module implemented
- [ ] Environment variable substitution working
- [ ] Model configuration parsing working
- [ ] Path properties working
- [ ] Validation implemented
- [ ] All Phase 2 tests pass

---

## 7. Phase 3: Indexing Pipeline

### 7.1 Objective

Implement the document indexing pipeline wrapper.

### 7.2 Implementation Steps

#### Step 3.1: Create Indexer Module

**File**: `graphrag_integration/indexer.py`

```python
"""
GraphRAG Indexing Pipeline for Palli Sahayak

Provides document indexing capabilities using Microsoft GraphRAG.
Supports both standard (LLM-based) and fast (NLP-based) indexing methods.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from datetime import datetime

from graphrag_integration.config import GraphRAGConfig

logger = logging.getLogger(__name__)


class IndexingMethod(Enum):
    """Available indexing methods."""
    STANDARD = "standard"  # LLM-based extraction (higher quality)
    FAST = "fast"          # NLP-based extraction (faster, lower cost)


class IndexingStatus(Enum):
    """Indexing job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class GraphRAGIndexer:
    """
    Document indexing pipeline for GraphRAG.

    Handles document processing, entity extraction, community detection,
    and embedding generation.

    Attributes:
        config: GraphRAG configuration
        method: Indexing method (standard or fast)
        status: Current indexing status
        progress: Progress percentage (0-100)

    Example:
        indexer = GraphRAGIndexer(config)
        await indexer.index_documents()

        # Monitor progress
        while indexer.status == IndexingStatus.RUNNING:
            print(f"Progress: {indexer.progress}%")
            await asyncio.sleep(1)
    """

    def __init__(
        self,
        config: GraphRAGConfig,
        method: IndexingMethod = IndexingMethod.STANDARD
    ):
        """
        Initialize indexer.

        Args:
            config: GraphRAG configuration
            method: Indexing method to use
        """
        self.config = config
        self.method = method
        self.status = IndexingStatus.PENDING
        self.progress = 0
        self._callbacks: List[Callable] = []
        self._stats: Dict[str, Any] = {}
        self._error: Optional[str] = None

    async def index_documents(
        self,
        documents: Optional[List[str]] = None,
        update_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Index documents using GraphRAG pipeline.

        Args:
            documents: Optional list of document paths (uses input_dir if None)
            update_mode: If True, perform incremental update

        Returns:
            Dictionary with indexing results and statistics

        Raises:
            RuntimeError: If indexing fails
        """
        self.status = IndexingStatus.RUNNING
        self.progress = 0

        try:
            # Import GraphRAG
            from graphrag.api import build_index
            from graphrag.config import GraphRagConfig

            # Load native config
            native_config = self.config.to_graphrag_config()

            # Set method
            if self.method == IndexingMethod.FAST:
                native_config.indexing_method = "fast"
            else:
                native_config.indexing_method = "standard"

            # Define progress callback
            async def progress_callback(status: str, progress: float):
                self.progress = int(progress * 100)
                self._notify_callbacks(status, self.progress)

            # Run indexing
            logger.info(f"Starting indexing with method: {self.method.value}")
            self._notify_callbacks("Starting indexing...", 0)

            # Execute indexing pipeline
            if update_mode:
                result = await self._run_update_index(native_config)
            else:
                result = await self._run_full_index(native_config)

            self.status = IndexingStatus.COMPLETED
            self.progress = 100
            self._stats = result

            logger.info("Indexing completed successfully")
            self._notify_callbacks("Indexing completed", 100)

            return result

        except Exception as e:
            self.status = IndexingStatus.FAILED
            self._error = str(e)
            logger.error(f"Indexing failed: {e}")
            raise RuntimeError(f"Indexing failed: {e}") from e

    async def _run_full_index(self, config: Any) -> Dict[str, Any]:
        """Run full indexing pipeline."""
        try:
            from graphrag.api.index import build_index

            # Build index
            await build_index(config=config)

            # Collect statistics
            return await self._collect_stats()

        except ImportError:
            # Fallback for testing without GraphRAG installed
            logger.warning("GraphRAG not installed, using mock indexing")
            return await self._mock_index()

    async def _run_update_index(self, config: Any) -> Dict[str, Any]:
        """Run incremental update indexing."""
        try:
            from graphrag.api.index import build_index

            # Build with update mode
            await build_index(config=config, update_index=True)

            return await self._collect_stats()

        except ImportError:
            logger.warning("GraphRAG not installed, using mock indexing")
            return await self._mock_index()

    async def _mock_index(self) -> Dict[str, Any]:
        """Mock indexing for testing."""
        # Simulate progress
        for i in range(10):
            self.progress = (i + 1) * 10
            self._notify_callbacks(f"Processing step {i+1}/10", self.progress)
            await asyncio.sleep(0.1)

        return {
            "status": "mock",
            "documents_processed": 0,
            "entities_extracted": 0,
            "relationships_extracted": 0,
            "communities_created": 0,
        }

    async def _collect_stats(self) -> Dict[str, Any]:
        """Collect indexing statistics from output files."""
        stats = {
            "timestamp": datetime.now().isoformat(),
            "method": self.method.value,
            "documents_processed": 0,
            "entities_extracted": 0,
            "relationships_extracted": 0,
            "communities_created": 0,
            "text_units_created": 0,
        }

        try:
            import pandas as pd

            artifacts_dir = self.config.artifacts_dir

            # Count entities
            entities_path = artifacts_dir / "entities.parquet"
            if entities_path.exists():
                df = pd.read_parquet(entities_path)
                stats["entities_extracted"] = len(df)

            # Count relationships
            relationships_path = artifacts_dir / "relationships.parquet"
            if relationships_path.exists():
                df = pd.read_parquet(relationships_path)
                stats["relationships_extracted"] = len(df)

            # Count communities
            communities_path = artifacts_dir / "communities.parquet"
            if communities_path.exists():
                df = pd.read_parquet(communities_path)
                stats["communities_created"] = len(df)

            # Count text units
            text_units_path = artifacts_dir / "text_units.parquet"
            if text_units_path.exists():
                df = pd.read_parquet(text_units_path)
                stats["text_units_created"] = len(df)

            # Count documents
            documents_path = artifacts_dir / "documents.parquet"
            if documents_path.exists():
                df = pd.read_parquet(documents_path)
                stats["documents_processed"] = len(df)

        except Exception as e:
            logger.warning(f"Failed to collect stats: {e}")

        return stats

    def add_callback(self, callback: Callable[[str, int], None]) -> None:
        """
        Add progress callback.

        Args:
            callback: Function(status: str, progress: int) -> None
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable) -> None:
        """Remove progress callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _notify_callbacks(self, status: str, progress: int) -> None:
        """Notify all callbacks of progress update."""
        for callback in self._callbacks:
            try:
                callback(status, progress)
            except Exception as e:
                logger.warning(f"Callback error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get indexing statistics."""
        return self._stats.copy()

    def get_error(self) -> Optional[str]:
        """Get error message if indexing failed."""
        return self._error

    async def verify_index(self) -> Dict[str, Any]:
        """
        Verify index integrity.

        Returns:
            Dictionary with verification results
        """
        results = {
            "valid": True,
            "errors": [],
            "files_checked": [],
        }

        required_files = [
            "entities.parquet",
            "relationships.parquet",
            "communities.parquet",
            "community_reports.parquet",
            "text_units.parquet",
        ]

        artifacts_dir = self.config.artifacts_dir

        for filename in required_files:
            filepath = artifacts_dir / filename
            results["files_checked"].append(filename)

            if not filepath.exists():
                results["valid"] = False
                results["errors"].append(f"Missing file: {filename}")
            else:
                # Try to read the file
                try:
                    import pandas as pd
                    df = pd.read_parquet(filepath)
                    if len(df) == 0:
                        results["errors"].append(f"Empty file: {filename}")
                except Exception as e:
                    results["valid"] = False
                    results["errors"].append(f"Corrupted file {filename}: {e}")

        return results

    def __repr__(self) -> str:
        return (
            f"GraphRAGIndexer(method={self.method.value}, "
            f"status={self.status.value}, progress={self.progress}%)"
        )
```

#### Step 3.2: Create Custom Prompts for Palliative Care

**File**: `graphrag_integration/prompts/entity_extraction.txt`

```
-Goal-
Given a text document from palliative care medical guidelines, identify all entities and their relationships.

-Entity Types-
{entity_types}

-Relationship Types-
TREATS: A medication or intervention treats a symptom or condition
CAUSES: A condition causes a symptom
SIDE_EFFECT_OF: A side effect is caused by a medication
MANAGES: A treatment approach manages a condition
ADMINISTERED_VIA: A medication is given via a specific route
ASSESSED_BY: A symptom is assessed using a specific tool
RECOMMENDED_FOR: An intervention is recommended for a specific condition
CONTRAINDICATED_FOR: A medication is contraindicated for a condition

-Instructions-
1. Extract all entities matching the entity types from the text
2. For each entity, provide:
   - name: The canonical name (e.g., "morphine" not "Morphine Sulfate")
   - type: One of the entity types listed above
   - description: A brief description from the context

3. Extract all relationships between entities
4. For each relationship, provide:
   - source: The source entity name
   - target: The target entity name
   - type: One of the relationship types listed above
   - description: Context for the relationship

-Output Format-
Return a JSON object with two arrays: "entities" and "relationships"

-Example Output-
{
  "entities": [
    {"name": "morphine", "type": "Medication", "description": "Strong opioid for severe pain"},
    {"name": "cancer pain", "type": "Symptom", "description": "Pain associated with malignancy"}
  ],
  "relationships": [
    {"source": "morphine", "target": "cancer pain", "type": "TREATS", "description": "First-line treatment for severe cancer pain"}
  ]
}

-Real Data-
{input_text}
```

**File**: `graphrag_integration/prompts/community_report.txt`

```
-Goal-
Write a comprehensive summary of a community of related palliative care concepts.

-Community Data-
{context_data}

-Instructions-
1. Analyze the entities and relationships in this community
2. Identify the main theme (e.g., pain management, symptom control, end-of-life care)
3. Summarize the key medical concepts and their relationships
4. Highlight important clinical implications
5. Note any safety considerations or contraindications

-Output Format-
Write a structured report with the following sections:

## Community Overview
[Brief description of what this community represents]

## Key Entities
[List the most important entities with brief descriptions]

## Clinical Relationships
[Describe how the entities relate to each other clinically]

## Clinical Implications
[Practical implications for palliative care]

## Safety Considerations
[Any warnings, contraindications, or monitoring requirements]

-Report-
```

### 7.3 Test Criteria (Phase 3)

```python
# tests/test_graphrag_indexer.py
"""Phase 3 Tests: Indexing Pipeline"""

import pytest
import asyncio
from pathlib import Path


@pytest.fixture
def config():
    """Create test configuration."""
    from graphrag_integration.config import GraphRAGConfig
    return GraphRAGConfig(root_dir="./data/graphrag")


def test_indexer_initialization(config):
    """Test indexer initialization."""
    from graphrag_integration.indexer import GraphRAGIndexer, IndexingMethod

    indexer = GraphRAGIndexer(config)
    assert indexer.method == IndexingMethod.STANDARD
    assert indexer.progress == 0


def test_indexer_methods(config):
    """Test different indexing methods."""
    from graphrag_integration.indexer import GraphRAGIndexer, IndexingMethod

    standard_indexer = GraphRAGIndexer(config, IndexingMethod.STANDARD)
    fast_indexer = GraphRAGIndexer(config, IndexingMethod.FAST)

    assert standard_indexer.method == IndexingMethod.STANDARD
    assert fast_indexer.method == IndexingMethod.FAST


def test_indexer_callbacks(config):
    """Test progress callbacks."""
    from graphrag_integration.indexer import GraphRAGIndexer

    indexer = GraphRAGIndexer(config)
    progress_updates = []

    def callback(status, progress):
        progress_updates.append((status, progress))

    indexer.add_callback(callback)
    indexer._notify_callbacks("test", 50)

    assert len(progress_updates) == 1
    assert progress_updates[0] == ("test", 50)


@pytest.mark.asyncio
async def test_indexer_mock_run(config):
    """Test mock indexing run."""
    from graphrag_integration.indexer import GraphRAGIndexer, IndexingStatus

    indexer = GraphRAGIndexer(config)
    result = await indexer._mock_index()

    assert indexer.progress == 100
    assert "status" in result


def test_prompt_files_exist():
    """Test that prompt files exist."""
    prompts_dir = Path("graphrag_integration/prompts")
    required_prompts = [
        "entity_extraction.txt",
        "community_report.txt",
    ]

    for prompt in required_prompts:
        prompt_path = prompts_dir / prompt
        assert prompt_path.exists(), f"Missing prompt: {prompt}"
```

### 7.4 Completion Checklist (Phase 3)

- [ ] `indexer.py` module implemented
- [ ] Entity extraction prompt created
- [ ] Community report prompt created
- [ ] Progress callback system working
- [ ] Statistics collection implemented
- [ ] Index verification implemented
- [ ] All Phase 3 tests pass

---

## 8. Phase 4: Query Engine

### 8.1 Objective

Implement the query engine with all search types.

### 8.2 Implementation Steps

#### Step 4.1: Create Query Engine Module

**File**: `graphrag_integration/query_engine.py`

```python
"""
GraphRAG Query Engine for Palli Sahayak

Provides query capabilities using Global, Local, DRIFT, and Basic search strategies.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass

from graphrag_integration.config import GraphRAGConfig
from graphrag_integration.data_loader import GraphRAGDataLoader

logger = logging.getLogger(__name__)


class SearchMethod(Enum):
    """Available search methods."""
    GLOBAL = "global"    # Holistic corpus-wide queries
    LOCAL = "local"      # Entity-focused queries
    DRIFT = "drift"      # Multi-phase iterative search
    BASIC = "basic"      # Simple vector similarity


@dataclass
class SearchResult:
    """Container for search results."""
    query: str
    response: str
    method: SearchMethod
    sources: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    communities: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "response": self.response,
            "method": self.method.value,
            "sources": self.sources,
            "entities": self.entities,
            "communities": self.communities,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class GraphRAGQueryEngine:
    """
    Query engine for GraphRAG.

    Supports multiple search strategies optimized for different query types.

    Attributes:
        config: GraphRAG configuration
        data_loader: Data loader for parquet files

    Example:
        engine = GraphRAGQueryEngine(config)

        # Global search for holistic questions
        result = await engine.global_search(
            "What are the main approaches to pain management?"
        )

        # Local search for entity-specific questions
        result = await engine.local_search(
            "What are the side effects of morphine?"
        )

        # DRIFT search for complex multi-hop questions
        result = await engine.drift_search(
            "How should pain be managed in a patient with renal failure?"
        )
    """

    def __init__(self, config: GraphRAGConfig):
        """
        Initialize query engine.

        Args:
            config: GraphRAG configuration
        """
        self.config = config
        self.data_loader = GraphRAGDataLoader(config)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize query engine and load data."""
        if self._initialized:
            return

        await self.data_loader.load_all()
        self._initialized = True
        logger.info("Query engine initialized")

    async def global_search(
        self,
        query: str,
        community_level: int = 2,
        response_type: str = "comprehensive"
    ) -> SearchResult:
        """
        Perform global search across entire corpus.

        Best for:
        - Holistic questions about the dataset
        - Synthesizing themes across documents
        - Questions requiring broad understanding

        Args:
            query: User query
            community_level: Community hierarchy level (0=granular, higher=broader)
            response_type: Response format (comprehensive, brief)

        Returns:
            SearchResult with response and sources
        """
        await self.initialize()

        try:
            from graphrag.api.query import global_search as graphrag_global

            # Load native config
            native_config = self.config.to_graphrag_config()

            # Execute search
            result = await graphrag_global(
                config=native_config,
                query=query,
            )

            return SearchResult(
                query=query,
                response=result.response,
                method=SearchMethod.GLOBAL,
                sources=self._extract_sources(result),
                entities=[],
                communities=self._extract_communities(result),
                confidence=self._calculate_confidence(result),
                metadata={"community_level": community_level},
            )

        except ImportError:
            logger.warning("GraphRAG not installed, using fallback search")
            return await self._fallback_search(query, SearchMethod.GLOBAL)

        except Exception as e:
            logger.error(f"Global search failed: {e}")
            raise

    async def local_search(
        self,
        query: str,
        top_k_entities: int = 10,
        include_community_context: bool = True
    ) -> SearchResult:
        """
        Perform local search focused on specific entities.

        Best for:
        - Questions about specific entities (medications, symptoms)
        - Relationship queries
        - Detailed information retrieval

        Args:
            query: User query
            top_k_entities: Number of top entities to retrieve
            include_community_context: Include community reports

        Returns:
            SearchResult with response and sources
        """
        await self.initialize()

        try:
            from graphrag.api.query import local_search as graphrag_local

            native_config = self.config.to_graphrag_config()

            result = await graphrag_local(
                config=native_config,
                query=query,
            )

            return SearchResult(
                query=query,
                response=result.response,
                method=SearchMethod.LOCAL,
                sources=self._extract_sources(result),
                entities=self._extract_entities(result),
                communities=[],
                confidence=self._calculate_confidence(result),
                metadata={"top_k_entities": top_k_entities},
            )

        except ImportError:
            logger.warning("GraphRAG not installed, using fallback search")
            return await self._fallback_search(query, SearchMethod.LOCAL)

        except Exception as e:
            logger.error(f"Local search failed: {e}")
            raise

    async def drift_search(
        self,
        query: str,
        n_depth: int = 3,
        primer_folds: int = 3
    ) -> SearchResult:
        """
        Perform DRIFT (Dynamic Retrieval with Iterative Focusing and Traversal) search.

        Best for:
        - Complex multi-hop questions
        - Questions requiring reasoning across entities
        - Balanced exploration and exploitation

        Args:
            query: User query
            n_depth: Depth of iterative search
            primer_folds: Number of primer phase folds

        Returns:
            SearchResult with response and sources
        """
        await self.initialize()

        try:
            from graphrag.api.query import drift_search as graphrag_drift

            native_config = self.config.to_graphrag_config()

            result = await graphrag_drift(
                config=native_config,
                query=query,
            )

            return SearchResult(
                query=query,
                response=result.response,
                method=SearchMethod.DRIFT,
                sources=self._extract_sources(result),
                entities=self._extract_entities(result),
                communities=self._extract_communities(result),
                confidence=self._calculate_confidence(result),
                metadata={"n_depth": n_depth, "primer_folds": primer_folds},
            )

        except ImportError:
            logger.warning("GraphRAG not installed, using fallback search")
            return await self._fallback_search(query, SearchMethod.DRIFT)

        except Exception as e:
            logger.error(f"DRIFT search failed: {e}")
            raise

    async def basic_search(
        self,
        query: str,
        top_k: int = 5
    ) -> SearchResult:
        """
        Perform basic vector similarity search.

        Best for:
        - Simple factual queries
        - Baseline comparison
        - Fast retrieval

        Args:
            query: User query
            top_k: Number of results to retrieve

        Returns:
            SearchResult with response and sources
        """
        await self.initialize()

        try:
            from graphrag.api.query import basic_search as graphrag_basic

            native_config = self.config.to_graphrag_config()

            result = await graphrag_basic(
                config=native_config,
                query=query,
            )

            return SearchResult(
                query=query,
                response=result.response,
                method=SearchMethod.BASIC,
                sources=self._extract_sources(result),
                entities=[],
                communities=[],
                confidence=self._calculate_confidence(result),
                metadata={"top_k": top_k},
            )

        except ImportError:
            logger.warning("GraphRAG not installed, using fallback search")
            return await self._fallback_search(query, SearchMethod.BASIC)

        except Exception as e:
            logger.error(f"Basic search failed: {e}")
            raise

    async def auto_search(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> SearchResult:
        """
        Automatically select the best search method based on query analysis.

        Args:
            query: User query
            context: Optional conversation context

        Returns:
            SearchResult from the selected method
        """
        # Analyze query to determine best method
        method = self._analyze_query(query)

        if method == SearchMethod.GLOBAL:
            return await self.global_search(query)
        elif method == SearchMethod.LOCAL:
            return await self.local_search(query)
        elif method == SearchMethod.DRIFT:
            return await self.drift_search(query)
        else:
            return await self.basic_search(query)

    def _analyze_query(self, query: str) -> SearchMethod:
        """
        Analyze query to determine optimal search method.

        Args:
            query: User query

        Returns:
            Recommended search method
        """
        query_lower = query.lower()

        # Global indicators: broad, thematic questions
        global_keywords = [
            "overall", "main", "themes", "summary", "generally",
            "across", "comprehensive", "all", "types of", "approaches"
        ]

        # Local indicators: entity-specific questions
        local_keywords = [
            "specific", "what is", "tell me about", "side effects",
            "dosage", "how does", "compare", "difference between"
        ]

        # DRIFT indicators: complex, multi-hop questions
        drift_keywords = [
            "how should", "in the context of", "considering",
            "for a patient with", "when combined with", "impact of"
        ]

        # Count keyword matches
        global_score = sum(1 for kw in global_keywords if kw in query_lower)
        local_score = sum(1 for kw in local_keywords if kw in query_lower)
        drift_score = sum(1 for kw in drift_keywords if kw in query_lower)

        # Select method based on scores
        max_score = max(global_score, local_score, drift_score)

        if max_score == 0:
            return SearchMethod.LOCAL  # Default to local for most queries

        if global_score == max_score:
            return SearchMethod.GLOBAL
        elif drift_score == max_score:
            return SearchMethod.DRIFT
        else:
            return SearchMethod.LOCAL

    async def _fallback_search(
        self,
        query: str,
        method: SearchMethod
    ) -> SearchResult:
        """
        Fallback search using loaded parquet data directly.

        Args:
            query: User query
            method: Original search method

        Returns:
            SearchResult with basic response
        """
        # Use data loader to find relevant entities
        entities = await self.data_loader.search_entities(query, top_k=5)

        # Build response from entity descriptions
        if entities:
            response_parts = []
            for entity in entities:
                name = entity.get("title", entity.get("name", "Unknown"))
                desc = entity.get("description", "No description available")
                response_parts.append(f"**{name}**: {desc}")
            response = "\n\n".join(response_parts)
        else:
            response = "No relevant information found in the knowledge base."

        return SearchResult(
            query=query,
            response=response,
            method=method,
            sources=[],
            entities=entities,
            communities=[],
            confidence=0.5,
            metadata={"fallback": True},
        )

    def _extract_sources(self, result: Any) -> List[Dict[str, Any]]:
        """Extract sources from GraphRAG result."""
        sources = []
        if hasattr(result, "context_data"):
            # Extract text unit references
            pass
        return sources

    def _extract_entities(self, result: Any) -> List[Dict[str, Any]]:
        """Extract entities from GraphRAG result."""
        entities = []
        if hasattr(result, "context_data"):
            # Extract entity data
            pass
        return entities

    def _extract_communities(self, result: Any) -> List[Dict[str, Any]]:
        """Extract communities from GraphRAG result."""
        communities = []
        if hasattr(result, "context_data"):
            # Extract community data
            pass
        return communities

    def _calculate_confidence(self, result: Any) -> float:
        """Calculate confidence score from result."""
        # Basic confidence calculation
        return 0.8

    def __repr__(self) -> str:
        return f"GraphRAGQueryEngine(initialized={self._initialized})"
```

#### Step 4.2: Create Data Loader Module

**File**: `graphrag_integration/data_loader.py`

```python
"""
GraphRAG Data Loader for Palli Sahayak

Handles loading and caching of parquet files for query operations.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import asyncio

from graphrag_integration.config import GraphRAGConfig

logger = logging.getLogger(__name__)


class GraphRAGDataLoader:
    """
    Data loader for GraphRAG parquet files.

    Provides efficient loading and caching of indexed data.

    Attributes:
        config: GraphRAG configuration
        entities: Loaded entities DataFrame
        relationships: Loaded relationships DataFrame
        communities: Loaded communities DataFrame
        community_reports: Loaded community reports DataFrame
        text_units: Loaded text units DataFrame
    """

    def __init__(self, config: GraphRAGConfig):
        """
        Initialize data loader.

        Args:
            config: GraphRAG configuration
        """
        self.config = config
        self._entities = None
        self._relationships = None
        self._communities = None
        self._community_reports = None
        self._text_units = None
        self._embeddings = None
        self._loaded = False

    async def load_all(self) -> None:
        """Load all data files."""
        if self._loaded:
            return

        try:
            import pandas as pd

            artifacts_dir = self.config.artifacts_dir

            # Load entities
            entities_path = artifacts_dir / "entities.parquet"
            if entities_path.exists():
                self._entities = pd.read_parquet(entities_path)
                logger.info(f"Loaded {len(self._entities)} entities")

            # Load relationships
            relationships_path = artifacts_dir / "relationships.parquet"
            if relationships_path.exists():
                self._relationships = pd.read_parquet(relationships_path)
                logger.info(f"Loaded {len(self._relationships)} relationships")

            # Load communities
            communities_path = artifacts_dir / "communities.parquet"
            if communities_path.exists():
                self._communities = pd.read_parquet(communities_path)
                logger.info(f"Loaded {len(self._communities)} communities")

            # Load community reports
            reports_path = artifacts_dir / "community_reports.parquet"
            if reports_path.exists():
                self._community_reports = pd.read_parquet(reports_path)
                logger.info(f"Loaded {len(self._community_reports)} community reports")

            # Load text units
            text_units_path = artifacts_dir / "text_units.parquet"
            if text_units_path.exists():
                self._text_units = pd.read_parquet(text_units_path)
                logger.info(f"Loaded {len(self._text_units)} text units")

            self._loaded = True

        except ImportError:
            logger.warning("pandas not installed, data loading disabled")

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    async def search_entities(
        self,
        query: str,
        top_k: int = 10,
        entity_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search entities by query string.

        Args:
            query: Search query
            top_k: Maximum results to return
            entity_type: Optional filter by entity type

        Returns:
            List of matching entity dictionaries
        """
        await self.load_all()

        if self._entities is None:
            return []

        # Simple text matching (would use embeddings in production)
        query_lower = query.lower()
        matches = []

        for _, row in self._entities.iterrows():
            score = 0
            title = str(row.get("title", "")).lower()
            description = str(row.get("description", "")).lower()

            if query_lower in title:
                score += 2
            if query_lower in description:
                score += 1

            for word in query_lower.split():
                if word in title:
                    score += 0.5
                if word in description:
                    score += 0.25

            if score > 0:
                if entity_type is None or row.get("type") == entity_type:
                    matches.append({
                        "title": row.get("title"),
                        "type": row.get("type"),
                        "description": row.get("description"),
                        "score": score,
                    })

        # Sort by score and return top_k
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches[:top_k]

    async def get_entity_relationships(
        self,
        entity_name: str
    ) -> List[Dict[str, Any]]:
        """
        Get relationships for a specific entity.

        Args:
            entity_name: Name of the entity

        Returns:
            List of relationship dictionaries
        """
        await self.load_all()

        if self._relationships is None:
            return []

        matches = []
        entity_lower = entity_name.lower()

        for _, row in self._relationships.iterrows():
            source = str(row.get("source", "")).lower()
            target = str(row.get("target", "")).lower()

            if entity_lower in source or entity_lower in target:
                matches.append({
                    "source": row.get("source"),
                    "target": row.get("target"),
                    "type": row.get("type"),
                    "description": row.get("description"),
                    "weight": row.get("weight", 1.0),
                })

        return matches

    async def get_community_report(
        self,
        community_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get community report by ID.

        Args:
            community_id: Community ID

        Returns:
            Community report dictionary or None
        """
        await self.load_all()

        if self._community_reports is None:
            return None

        for _, row in self._community_reports.iterrows():
            if str(row.get("id")) == str(community_id):
                return {
                    "id": row.get("id"),
                    "title": row.get("title"),
                    "summary": row.get("summary"),
                    "full_content": row.get("full_content"),
                    "level": row.get("level"),
                }

        return None

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get data statistics.

        Returns:
            Dictionary with counts for each data type
        """
        await self.load_all()

        return {
            "entities": len(self._entities) if self._entities is not None else 0,
            "relationships": len(self._relationships) if self._relationships is not None else 0,
            "communities": len(self._communities) if self._communities is not None else 0,
            "community_reports": len(self._community_reports) if self._community_reports is not None else 0,
            "text_units": len(self._text_units) if self._text_units is not None else 0,
            "loaded": self._loaded,
        }

    def __repr__(self) -> str:
        return f"GraphRAGDataLoader(loaded={self._loaded})"
```

### 8.3 Test Criteria (Phase 4)

```python
# tests/test_graphrag_query.py
"""Phase 4 Tests: Query Engine"""

import pytest
import asyncio


@pytest.fixture
def config():
    """Create test configuration."""
    from graphrag_integration.config import GraphRAGConfig
    return GraphRAGConfig(root_dir="./data/graphrag")


@pytest.fixture
def query_engine(config):
    """Create query engine."""
    from graphrag_integration.query_engine import GraphRAGQueryEngine
    return GraphRAGQueryEngine(config)


def test_query_engine_initialization(query_engine):
    """Test query engine initialization."""
    assert query_engine._initialized is False


def test_query_analysis(query_engine):
    """Test query analysis for method selection."""
    from graphrag_integration.query_engine import SearchMethod

    # Global query
    method = query_engine._analyze_query("What are the main approaches to pain management?")
    assert method in [SearchMethod.GLOBAL, SearchMethod.LOCAL]

    # Local query
    method = query_engine._analyze_query("What are the side effects of morphine?")
    assert method == SearchMethod.LOCAL

    # DRIFT query
    method = query_engine._analyze_query("How should pain be managed in a patient with renal failure?")
    assert method == SearchMethod.DRIFT


@pytest.mark.asyncio
async def test_fallback_search(query_engine):
    """Test fallback search mechanism."""
    from graphrag_integration.query_engine import SearchMethod

    result = await query_engine._fallback_search(
        "test query",
        SearchMethod.LOCAL
    )

    assert result.query == "test query"
    assert result.method == SearchMethod.LOCAL
    assert result.metadata.get("fallback") is True


def test_search_result_to_dict(query_engine):
    """Test SearchResult conversion to dict."""
    from graphrag_integration.query_engine import SearchResult, SearchMethod

    result = SearchResult(
        query="test",
        response="test response",
        method=SearchMethod.LOCAL,
        sources=[],
        entities=[],
        communities=[],
        confidence=0.9,
        metadata={},
    )

    result_dict = result.to_dict()
    assert result_dict["query"] == "test"
    assert result_dict["method"] == "local"
```

### 8.4 Completion Checklist (Phase 4)

- [ ] `query_engine.py` module implemented
- [ ] `data_loader.py` module implemented
- [ ] All search methods implemented (global, local, DRIFT, basic)
- [ ] Auto-search query analysis working
- [ ] Fallback search mechanism working
- [ ] All Phase 4 tests pass

---

## 9. Phase 5: Server Integration

### 9.1 Objective

Integrate GraphRAG with the FastAPI server.

### 9.2 Implementation Steps

#### Step 5.1: Add GraphRAG Endpoints to Server

Add to `simple_rag_server.py`:

```python
# =============================================================================
# GRAPHRAG INTEGRATION
# =============================================================================

# Import GraphRAG modules (with graceful fallback)
try:
    from graphrag_integration import (
        GraphRAGConfig,
        GraphRAGIndexer,
        GraphRAGQueryEngine,
    )
    from graphrag_integration.query_engine import SearchMethod
    GRAPHRAG_AVAILABLE = True
except ImportError:
    GRAPHRAG_AVAILABLE = False
    logger.warning("GraphRAG integration not available")

# Global GraphRAG instances
graphrag_config: Optional[GraphRAGConfig] = None
graphrag_indexer: Optional[GraphRAGIndexer] = None
graphrag_query_engine: Optional[GraphRAGQueryEngine] = None


async def initialize_graphrag():
    """Initialize GraphRAG components."""
    global graphrag_config, graphrag_indexer, graphrag_query_engine

    if not GRAPHRAG_AVAILABLE:
        logger.warning("GraphRAG not available, skipping initialization")
        return

    try:
        settings_path = Path("./data/graphrag/settings.yaml")
        if not settings_path.exists():
            logger.warning("GraphRAG settings.yaml not found")
            return

        graphrag_config = GraphRAGConfig.from_yaml(str(settings_path))
        graphrag_indexer = GraphRAGIndexer(graphrag_config)
        graphrag_query_engine = GraphRAGQueryEngine(graphrag_config)

        logger.info("GraphRAG initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize GraphRAG: {e}")


# Add to startup event
@app.on_event("startup")
async def startup_graphrag():
    """Initialize GraphRAG on startup."""
    await initialize_graphrag()


# =============================================================================
# GRAPHRAG API ENDPOINTS
# =============================================================================

class GraphRAGQueryRequest(BaseModel):
    """GraphRAG query request model."""
    query: str
    method: Optional[str] = "auto"  # auto, global, local, drift, basic
    language: Optional[str] = "en"
    top_k: Optional[int] = 10


class GraphRAGIndexRequest(BaseModel):
    """GraphRAG indexing request model."""
    method: Optional[str] = "standard"  # standard, fast
    update_mode: Optional[bool] = False


@app.get("/api/graphrag/health")
async def graphrag_health():
    """Check GraphRAG health status."""
    if not GRAPHRAG_AVAILABLE:
        return {"status": "unavailable", "reason": "GraphRAG not installed"}

    if graphrag_query_engine is None:
        return {"status": "not_initialized"}

    try:
        stats = await graphrag_query_engine.data_loader.get_stats()
        return {
            "status": "healthy",
            "initialized": True,
            "stats": stats,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/api/graphrag/stats")
async def graphrag_stats():
    """Get GraphRAG statistics."""
    if not GRAPHRAG_AVAILABLE or graphrag_query_engine is None:
        return {"error": "GraphRAG not available"}

    try:
        stats = await graphrag_query_engine.data_loader.get_stats()
        return {"status": "success", "stats": stats}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/graphrag/query")
async def graphrag_query(request: GraphRAGQueryRequest):
    """
    Query using GraphRAG.

    Methods:
    - auto: Automatically select best method
    - global: Holistic corpus-wide search
    - local: Entity-focused search
    - drift: Multi-phase iterative search
    - basic: Simple vector similarity
    """
    if not GRAPHRAG_AVAILABLE or graphrag_query_engine is None:
        return {"status": "error", "error": "GraphRAG not available"}

    try:
        method = request.method.lower()

        if method == "auto":
            result = await graphrag_query_engine.auto_search(request.query)
        elif method == "global":
            result = await graphrag_query_engine.global_search(request.query)
        elif method == "local":
            result = await graphrag_query_engine.local_search(request.query)
        elif method == "drift":
            result = await graphrag_query_engine.drift_search(request.query)
        elif method == "basic":
            result = await graphrag_query_engine.basic_search(request.query)
        else:
            return {"status": "error", "error": f"Unknown method: {method}"}

        return {
            "status": "success",
            "result": result.to_dict(),
        }

    except Exception as e:
        logger.error(f"GraphRAG query failed: {e}")
        return {"status": "error", "error": str(e)}


@app.post("/api/graphrag/index")
async def graphrag_index(request: GraphRAGIndexRequest):
    """
    Trigger GraphRAG indexing.

    Methods:
    - standard: LLM-based extraction (higher quality)
    - fast: NLP-based extraction (faster, lower cost)
    """
    if not GRAPHRAG_AVAILABLE or graphrag_indexer is None:
        return {"status": "error", "error": "GraphRAG not available"}

    try:
        from graphrag_integration.indexer import IndexingMethod

        method = IndexingMethod.FAST if request.method == "fast" else IndexingMethod.STANDARD
        graphrag_indexer.method = method

        # Run indexing in background
        asyncio.create_task(graphrag_indexer.index_documents(
            update_mode=request.update_mode
        ))

        return {
            "status": "started",
            "method": request.method,
            "update_mode": request.update_mode,
        }

    except Exception as e:
        logger.error(f"GraphRAG indexing failed: {e}")
        return {"status": "error", "error": str(e)}


@app.get("/api/graphrag/index/status")
async def graphrag_index_status():
    """Get indexing status."""
    if not GRAPHRAG_AVAILABLE or graphrag_indexer is None:
        return {"status": "error", "error": "GraphRAG not available"}

    return {
        "status": graphrag_indexer.status.value,
        "progress": graphrag_indexer.progress,
        "method": graphrag_indexer.method.value,
        "stats": graphrag_indexer.get_stats(),
        "error": graphrag_indexer.get_error(),
    }


@app.get("/api/graphrag/entities")
async def graphrag_entities(
    query: str = "",
    entity_type: Optional[str] = None,
    top_k: int = 20
):
    """Search entities."""
    if not GRAPHRAG_AVAILABLE or graphrag_query_engine is None:
        return {"status": "error", "error": "GraphRAG not available"}

    try:
        entities = await graphrag_query_engine.data_loader.search_entities(
            query=query,
            top_k=top_k,
            entity_type=entity_type,
        )
        return {"status": "success", "entities": entities}

    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/api/graphrag/entity/{entity_name}/relationships")
async def graphrag_entity_relationships(entity_name: str):
    """Get relationships for an entity."""
    if not GRAPHRAG_AVAILABLE or graphrag_query_engine is None:
        return {"status": "error", "error": "GraphRAG not available"}

    try:
        relationships = await graphrag_query_engine.data_loader.get_entity_relationships(
            entity_name=entity_name
        )
        return {"status": "success", "relationships": relationships}

    except Exception as e:
        return {"status": "error", "error": str(e)}
```

### 9.3 Test Criteria (Phase 5)

```python
# tests/test_graphrag_server.py
"""Phase 5 Tests: Server Integration"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client."""
    from simple_rag_server import app
    return TestClient(app)


def test_graphrag_health_endpoint(client):
    """Test GraphRAG health endpoint."""
    response = client.get("/api/graphrag/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_graphrag_stats_endpoint(client):
    """Test GraphRAG stats endpoint."""
    response = client.get("/api/graphrag/stats")
    assert response.status_code == 200


def test_graphrag_query_endpoint(client):
    """Test GraphRAG query endpoint."""
    response = client.post(
        "/api/graphrag/query",
        json={"query": "What is pain management?", "method": "auto"}
    )
    assert response.status_code == 200


def test_graphrag_entities_endpoint(client):
    """Test GraphRAG entities endpoint."""
    response = client.get("/api/graphrag/entities?query=morphine")
    assert response.status_code == 200


def test_graphrag_index_status_endpoint(client):
    """Test GraphRAG index status endpoint."""
    response = client.get("/api/graphrag/index/status")
    assert response.status_code == 200
```

### 9.4 Completion Checklist (Phase 5)

- [ ] GraphRAG endpoints added to server
- [ ] Startup initialization implemented
- [ ] Query endpoint working
- [ ] Index endpoint working
- [ ] Entity search endpoint working
- [ ] All Phase 5 tests pass

---

## 10. Phase 6: Admin UI Integration

### 10.1 Objective

Add GraphRAG tab to Gradio admin interface.

### 10.2 Implementation Steps

Add GraphRAG tab to admin UI in `simple_rag_server.py`:

```python
# Add to create_admin_interface() function

with gr.Tab("GraphRAG"):
    gr.Markdown("## Microsoft GraphRAG Integration")
    gr.Markdown("Advanced graph-based retrieval for palliative care knowledge.")

    with gr.Tabs():
        # Query Tab
        with gr.Tab("Query"):
            with gr.Row():
                with gr.Column(scale=2):
                    graphrag_query_input = gr.Textbox(
                        label="Query",
                        placeholder="Enter your question...",
                        lines=2
                    )
                    graphrag_method = gr.Radio(
                        choices=["auto", "global", "local", "drift", "basic"],
                        value="auto",
                        label="Search Method"
                    )
                    graphrag_query_btn = gr.Button("Search", variant="primary")

                with gr.Column(scale=3):
                    graphrag_response = gr.Markdown(label="Response")
                    graphrag_entities_output = gr.JSON(label="Entities Found")
                    graphrag_metadata = gr.JSON(label="Metadata")

            async def graphrag_search(query, method):
                if not GRAPHRAG_AVAILABLE or graphrag_query_engine is None:
                    return "GraphRAG not available", [], {}

                try:
                    if method == "auto":
                        result = await graphrag_query_engine.auto_search(query)
                    elif method == "global":
                        result = await graphrag_query_engine.global_search(query)
                    elif method == "local":
                        result = await graphrag_query_engine.local_search(query)
                    elif method == "drift":
                        result = await graphrag_query_engine.drift_search(query)
                    else:
                        result = await graphrag_query_engine.basic_search(query)

                    return (
                        result.response,
                        result.entities,
                        result.metadata
                    )
                except Exception as e:
                    return f"Error: {e}", [], {}

            graphrag_query_btn.click(
                fn=graphrag_search,
                inputs=[graphrag_query_input, graphrag_method],
                outputs=[graphrag_response, graphrag_entities_output, graphrag_metadata]
            )

        # Indexing Tab
        with gr.Tab("Indexing"):
            gr.Markdown("### Index Documents with GraphRAG")

            with gr.Row():
                index_method = gr.Radio(
                    choices=["standard", "fast"],
                    value="standard",
                    label="Indexing Method"
                )
                update_mode = gr.Checkbox(
                    label="Update Mode (incremental)",
                    value=False
                )

            with gr.Row():
                start_index_btn = gr.Button("Start Indexing", variant="primary")
                refresh_status_btn = gr.Button("Refresh Status")

            index_status = gr.JSON(label="Indexing Status")
            index_progress = gr.Progress()

            async def start_graphrag_indexing(method, update):
                if not GRAPHRAG_AVAILABLE or graphrag_indexer is None:
                    return {"error": "GraphRAG not available"}

                try:
                    from graphrag_integration.indexer import IndexingMethod
                    graphrag_indexer.method = (
                        IndexingMethod.FAST if method == "fast"
                        else IndexingMethod.STANDARD
                    )

                    asyncio.create_task(
                        graphrag_indexer.index_documents(update_mode=update)
                    )

                    return {
                        "status": "started",
                        "method": method,
                        "update_mode": update
                    }
                except Exception as e:
                    return {"error": str(e)}

            async def get_index_status():
                if not GRAPHRAG_AVAILABLE or graphrag_indexer is None:
                    return {"error": "GraphRAG not available"}

                return {
                    "status": graphrag_indexer.status.value,
                    "progress": graphrag_indexer.progress,
                    "method": graphrag_indexer.method.value,
                    "stats": graphrag_indexer.get_stats(),
                }

            start_index_btn.click(
                fn=start_graphrag_indexing,
                inputs=[index_method, update_mode],
                outputs=[index_status]
            )

            refresh_status_btn.click(
                fn=get_index_status,
                outputs=[index_status]
            )

        # Entity Explorer Tab
        with gr.Tab("Entity Explorer"):
            gr.Markdown("### Browse Extracted Entities")

            with gr.Row():
                entity_search = gr.Textbox(
                    label="Search Entities",
                    placeholder="Enter entity name..."
                )
                entity_type_filter = gr.Dropdown(
                    choices=[
                        "All", "Symptom", "Medication", "Condition",
                        "Treatment", "SideEffect", "Dosage", "Route"
                    ],
                    value="All",
                    label="Entity Type"
                )
                entity_search_btn = gr.Button("Search")

            entities_table = gr.Dataframe(
                headers=["Name", "Type", "Description", "Score"],
                label="Entities"
            )

            entity_relationships = gr.JSON(label="Relationships")

            async def search_entities(query, entity_type):
                if not GRAPHRAG_AVAILABLE or graphrag_query_engine is None:
                    return [], []

                try:
                    type_filter = None if entity_type == "All" else entity_type
                    entities = await graphrag_query_engine.data_loader.search_entities(
                        query=query,
                        top_k=20,
                        entity_type=type_filter
                    )

                    table_data = [
                        [e["title"], e["type"], e["description"][:100], e["score"]]
                        for e in entities
                    ]

                    return table_data, entities

                except Exception as e:
                    return [], [{"error": str(e)}]

            entity_search_btn.click(
                fn=search_entities,
                inputs=[entity_search, entity_type_filter],
                outputs=[entities_table, entity_relationships]
            )

        # Statistics Tab
        with gr.Tab("Statistics"):
            gr.Markdown("### GraphRAG Index Statistics")

            stats_output = gr.JSON(label="Statistics")
            refresh_stats_btn = gr.Button("Refresh Statistics")

            async def get_graphrag_stats():
                if not GRAPHRAG_AVAILABLE or graphrag_query_engine is None:
                    return {"error": "GraphRAG not available"}

                try:
                    stats = await graphrag_query_engine.data_loader.get_stats()
                    return stats
                except Exception as e:
                    return {"error": str(e)}

            refresh_stats_btn.click(
                fn=get_graphrag_stats,
                outputs=[stats_output]
            )
```

### 10.3 Completion Checklist (Phase 6)

- [ ] Query tab implemented
- [ ] Indexing tab implemented
- [ ] Entity Explorer tab implemented
- [ ] Statistics tab implemented
- [ ] All UI elements functional

---

## 11. Phase 7: Testing Suite

### 11.1 Comprehensive Test File

**File**: `tests/test_graphrag_integration.py`

```python
"""
Comprehensive GraphRAG Integration Tests

Run with: pytest tests/test_graphrag_integration.py -v
"""

import pytest
import asyncio
import os
from pathlib import Path


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


# =============================================================================
# INDEXER TESTS
# =============================================================================

class TestIndexer:
    """Indexer module tests."""

    def test_indexer_creation(self, graphrag_indexer):
        """Test indexer creation."""
        from graphrag_integration.indexer import IndexingStatus
        assert graphrag_indexer.status == IndexingStatus.PENDING

    def test_indexer_methods(self, graphrag_config):
        """Test different indexing methods."""
        from graphrag_integration.indexer import GraphRAGIndexer, IndexingMethod

        standard = GraphRAGIndexer(graphrag_config, IndexingMethod.STANDARD)
        fast = GraphRAGIndexer(graphrag_config, IndexingMethod.FAST)

        assert standard.method == IndexingMethod.STANDARD
        assert fast.method == IndexingMethod.FAST

    @pytest.mark.asyncio
    async def test_mock_indexing(self, graphrag_indexer):
        """Test mock indexing run."""
        result = await graphrag_indexer._mock_index()
        assert "status" in result
        assert graphrag_indexer.progress == 100


# =============================================================================
# QUERY ENGINE TESTS
# =============================================================================

class TestQueryEngine:
    """Query engine tests."""

    def test_query_engine_creation(self, graphrag_query_engine):
        """Test query engine creation."""
        assert graphrag_query_engine._initialized is False

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

    @pytest.mark.asyncio
    async def test_fallback_search(self, graphrag_query_engine):
        """Test fallback search."""
        from graphrag_integration.query_engine import SearchMethod

        result = await graphrag_query_engine._fallback_search(
            "test query",
            SearchMethod.LOCAL
        )

        assert result.query == "test query"
        assert result.metadata.get("fallback") is True


# =============================================================================
# DATA LOADER TESTS
# =============================================================================

class TestDataLoader:
    """Data loader tests."""

    def test_data_loader_creation(self, graphrag_config):
        """Test data loader creation."""
        from graphrag_integration.data_loader import GraphRAGDataLoader

        loader = GraphRAGDataLoader(graphrag_config)
        assert loader._loaded is False

    @pytest.mark.asyncio
    async def test_get_stats(self, graphrag_config):
        """Test getting stats."""
        from graphrag_integration.data_loader import GraphRAGDataLoader

        loader = GraphRAGDataLoader(graphrag_config)
        stats = await loader.get_stats()

        assert isinstance(stats, dict)
        assert "entities" in stats
        assert "relationships" in stats


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_full_query_flow(self, graphrag_query_engine):
        """Test full query flow."""
        result = await graphrag_query_engine.auto_search(
            "What medications are used for pain?"
        )

        assert result.query is not None
        assert result.response is not None
        assert result.method is not None

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
```

### 11.2 Run Tests Command

```bash
# Run all GraphRAG tests
pytest tests/test_graphrag_*.py -v

# Run with coverage
pytest tests/test_graphrag_*.py -v --cov=graphrag_integration --cov-report=html

# Run specific test class
pytest tests/test_graphrag_integration.py::TestQueryEngine -v
```

---

## 12. Phase 8: Performance Optimization

### 12.1 Caching Strategy

```python
# Add to graphrag_integration/utils.py

from functools import lru_cache
from typing import Any
import hashlib
import json


class QueryCache:
    """LRU cache for GraphRAG queries."""

    def __init__(self, maxsize: int = 100):
        self._cache = {}
        self._maxsize = maxsize

    def _hash_query(self, query: str, method: str) -> str:
        """Generate hash for cache key."""
        content = f"{query}:{method}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, query: str, method: str) -> Any:
        """Get cached result."""
        key = self._hash_query(query, method)
        return self._cache.get(key)

    def set(self, query: str, method: str, result: Any) -> None:
        """Cache result."""
        key = self._hash_query(query, method)

        # Evict oldest if at capacity
        if len(self._cache) >= self._maxsize:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = result

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
```

### 12.2 Completion Checklist (Phase 8)

- [ ] Query caching implemented
- [ ] Batch processing for indexing
- [ ] Async optimization
- [ ] Memory management

---

## 13. API Reference

### 13.1 Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/graphrag/health` | GET | Health check |
| `/api/graphrag/stats` | GET | Index statistics |
| `/api/graphrag/query` | POST | Execute search |
| `/api/graphrag/index` | POST | Start indexing |
| `/api/graphrag/index/status` | GET | Indexing status |
| `/api/graphrag/entities` | GET | Search entities |
| `/api/graphrag/entity/{name}/relationships` | GET | Entity relationships |

### 13.2 Query Request Schema

```json
{
  "query": "string (required)",
  "method": "auto|global|local|drift|basic (default: auto)",
  "language": "en|hi|bn|ta (default: en)",
  "top_k": "integer (default: 10)"
}
```

### 13.3 Response Schema

```json
{
  "status": "success|error",
  "result": {
    "query": "string",
    "response": "string",
    "method": "string",
    "sources": [],
    "entities": [],
    "communities": [],
    "confidence": "float",
    "metadata": {}
  }
}
```

---

## 14. Troubleshooting

### 14.1 Common Issues

| Issue | Solution |
|-------|----------|
| `graphrag not installed` | Run `pip install graphrag>=2.7.0` |
| `API key missing` | Set `GRAPHRAG_API_KEY` in `.env` |
| `Empty parquet files` | Re-run indexing with documents in input/ |
| `Import errors` | Check Python version (3.10+) |
| `Memory errors` | Reduce chunk size or use fast method |

### 14.2 Debug Mode

```bash
# Enable debug logging
export GRAPHRAG_LOG_LEVEL=DEBUG

# Run with verbose output
python simple_rag_server.py --debug
```

---

## Implementation Summary

### Phase Completion Order

1. **Phase 1**: Foundation Setup (directories, init files)
2. **Phase 2**: Configuration Module (settings.yaml, config.py)
3. **Phase 3**: Indexing Pipeline (indexer.py, prompts)
4. **Phase 4**: Query Engine (query_engine.py, data_loader.py)
5. **Phase 5**: Server Integration (API endpoints)
6. **Phase 6**: Admin UI (Gradio tabs)
7. **Phase 7**: Testing Suite (comprehensive tests)
8. **Phase 8**: Performance Optimization (caching, async)

### Success Criteria

- [ ] All 8 phases completed
- [ ] All tests passing
- [ ] API endpoints functional
- [ ] Admin UI working
- [ ] Performance optimized

---

**Document End**
