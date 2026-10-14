"""
PageIndex Configuration Module for Palli Sahayak

Provides configuration management for the PageIndex-style RAG integration.
Follows the same dataclass pattern as graphrag_integration/config.py.

Usage:
    from pageindex_integration.config import PageIndexConfig

    config = PageIndexConfig()
    config = PageIndexConfig(root_dir="./data/pageindex")
    config = PageIndexConfig.from_yaml("./config.yaml")

    errors = config.validate()
"""

import logging
from pathlib import Path
from dataclasses import dataclass, fields
from typing import Dict, Any, List

from llm import get_llm_client

logger = logging.getLogger(__name__)


@dataclass
class TreeBuildConfig:
    """Configuration for tree building (indexing)."""
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
    """
    Generation settings for PageIndex LLM calls.

    Which model answers is not set here: every call goes through the shared
    client (llm/client.py: Gemini, optional DeepSeek fallback), and
    `provider` / `effective_model` report that client's settings.
    """
    temperature: float = 0.0
    max_tokens: int = 4096
    rate_limit_rpm: int = 30

    @property
    def provider(self) -> str:
        return get_llm_client().settings.describe()["provider"]

    @property
    def effective_model(self) -> str:
        return get_llm_client().settings.model

    @property
    def is_configured(self) -> bool:
        s = get_llm_client().settings
        return s.transport != "unconfigured" or (s.fallback_enabled and bool(s.fallback_api_key))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.effective_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "rate_limit_rpm": self.rate_limit_rpm,
            "shared_client": get_llm_client().settings.describe(),
        }


@dataclass
class SearchConfig:
    """Configuration for tree search queries."""
    max_nodes_per_query: int = 5
    max_context_tokens: int = 8000
    # Documents navigated at once per query (one LLM call each).
    max_concurrent_docs: int = 8
    cache_enabled: bool = True
    cache_maxsize: int = 100
    cache_ttl_seconds: int = 3600

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_nodes_per_query": self.max_nodes_per_query,
            "max_context_tokens": self.max_context_tokens,
            "max_concurrent_docs": self.max_concurrent_docs,
            "cache_enabled": self.cache_enabled,
            "cache_maxsize": self.cache_maxsize,
            "cache_ttl_seconds": self.cache_ttl_seconds,
        }


class PageIndexConfig:
    """
    Configuration manager for PageIndex integration.

    Loads from environment variables and optional YAML.
    Follows the same pattern as graphrag_integration/config.py.
    """

    def __init__(self, root_dir: str = "./data/pageindex"):
        self.root_dir = Path(root_dir)
        self.tree_build = TreeBuildConfig()
        self.llm = LLMConfig()
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

        tb = pi_settings.get("tree_build", {})
        for key, val in tb.items():
            if hasattr(config.tree_build, key):
                setattr(config.tree_build, key, val)

        # Only generation settings are read; provider/model keys from older
        # config files are ignored (the shared LLM client decides those).
        llm_fields = {f.name for f in fields(LLMConfig)}
        llm = pi_settings.get("llm", {})
        for key, val in llm.items():
            if key in llm_fields:
                setattr(config.llm, key, val)

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
        if not self.llm.is_configured:
            errors.append(
                "LLM API key not configured (set GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT)"
            )
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
