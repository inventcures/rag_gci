"""Tests for PageIndex configuration module."""
import os
import pytest
import tempfile
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
    old = os.environ.pop("GROQ_API_KEY", None)
    old_openai = os.environ.pop("OPENAI_API_KEY", None)
    try:
        config = PageIndexConfig()
        config.llm.provider = "groq"
        config.llm.model = ""
        errors = config.validate()
        assert any("API key" in e for e in errors)
    finally:
        if old:
            os.environ["GROQ_API_KEY"] = old
        if old_openai:
            os.environ["OPENAI_API_KEY"] = old_openai


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


def test_llm_config_custom_model():
    llm = LLMConfig(provider="groq", model="llama-3.1-8b-instant")
    assert llm.effective_model == "llama-3.1-8b-instant"


def test_config_to_dict():
    config = PageIndexConfig()
    d = config.to_dict()
    assert "root_dir" in d
    assert "tree_build" in d
    assert "llm" in d
    assert "search" in d
    assert d["llm"]["provider"] == "groq"


def test_config_repr():
    config = PageIndexConfig()
    r = repr(config)
    assert "PageIndexConfig" in r
    assert "groq" in r


def test_tree_build_config_to_dict():
    tb = TreeBuildConfig()
    d = tb.to_dict()
    assert d["toc_check_pages"] == 20
    assert d["max_pages_per_node"] == 10


def test_search_config_to_dict():
    sc = SearchConfig()
    d = sc.to_dict()
    assert d["max_nodes_per_query"] == 5
    assert d["cache_enabled"] is True


def test_ensure_directories():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PageIndexConfig(root_dir=os.path.join(tmpdir, "pi"))
        config.ensure_directories()
        assert config.root_dir.exists()
        assert config.trees_dir.exists()
        assert config.cache_dir.exists()


def test_from_yaml():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
pageindex:
  root_dir: /tmp/test_pi
  tree_build:
    toc_check_pages: 30
    max_pages_per_node: 15
  llm:
    provider: openai
    temperature: 0.1
  search:
    max_nodes_per_query: 3
""")
        f.flush()
        config = PageIndexConfig.from_yaml(f.name)

    assert config.tree_build.toc_check_pages == 30
    assert config.tree_build.max_pages_per_node == 15
    assert config.llm.provider == "openai"
    assert config.llm.temperature == 0.1
    assert config.search.max_nodes_per_query == 3
    os.unlink(f.name)
