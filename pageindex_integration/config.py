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

import os
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List

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
    """Configuration for LLM provider (Groq or OpenAI)."""
    provider: str = "groq"
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096
    max_retries: int = 3
    request_timeout: int = 120
    rate_limit_rpm: int = 30

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
    """Configuration for tree search queries."""
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

        tb = pi_settings.get("tree_build", {})
        for key, val in tb.items():
            if hasattr(config.tree_build, key):
                setattr(config.tree_build, key, val)

        llm = pi_settings.get("llm", {})
        for key, val in llm.items():
            if hasattr(config.llm, key):
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
