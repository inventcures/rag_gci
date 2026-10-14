"""Tests for PageIndex configuration module."""
import os
import pytest
import tempfile
from llm import LLMClient, LLMSettings, get_llm_client, set_llm_client
from pageindex_integration.config import PageIndexConfig, TreeBuildConfig, LLMConfig, SearchConfig


@pytest.fixture(autouse=True)
def shared_client():
    """Install a shared LLM client with known settings; restore afterwards."""
    previous = get_llm_client()
    client = set_llm_client(LLMClient(LLMSettings(model="gemini-3.5-flash", api_key="test-key")))
    yield client
    set_llm_client(previous)


def test_default_config():
    config = PageIndexConfig()
    assert config.root_dir.name == "pageindex"
    assert config.llm.provider == "gemini"
    assert config.llm.effective_model == "gemini-3.5-flash"
    assert config.tree_build.toc_check_pages == 20
    assert config.search.cache_enabled is True


def test_config_paths():
    config = PageIndexConfig(root_dir="/tmp/test_pageindex")
    assert str(config.trees_dir).endswith("trees")
    assert str(config.cache_dir).endswith("cache")
    assert str(config.index_file).endswith("index.json")


def test_config_validation_missing_key():
    set_llm_client(LLMClient(LLMSettings()))
    errors = PageIndexConfig().validate()
    assert any("API key" in e for e in errors)
    assert any("GEMINI_API_KEY" in e for e in errors)


def test_config_validation_configured():
    assert PageIndexConfig().validate() == []


def test_config_validation_fallback_only_counts_as_configured():
    set_llm_client(LLMClient(LLMSettings(fallback_enabled=True, fallback_api_key="ds-key")))
    assert not any("API key" in e for e in PageIndexConfig().validate())


def test_config_validation_invalid_params():
    config = PageIndexConfig()
    config.tree_build.max_pages_per_node = 0
    errors = config.validate()
    assert any("max_pages_per_node" in e for e in errors)


def test_llm_config_reports_shared_client():
    llm = LLMConfig()
    assert llm.provider == "gemini"
    assert llm.effective_model == "gemini-3.5-flash"
    assert llm.is_configured is True


def test_llm_config_follows_client_model():
    set_llm_client(LLMClient(LLMSettings(model="gemini-3.8-flash", api_key="k")))
    assert LLMConfig().effective_model == "gemini-3.8-flash"


def test_llm_config_has_no_legacy_provider_settings():
    llm = LLMConfig()
    for attr in ("api_key", "base_url"):
        assert not hasattr(llm, attr)
    with pytest.raises(TypeError):
        LLMConfig(provider="groq")


def test_config_to_dict():
    config = PageIndexConfig()
    d = config.to_dict()
    assert "root_dir" in d
    assert "tree_build" in d
    assert "llm" in d
    assert "search" in d
    assert d["llm"]["provider"] == "gemini"
    assert d["llm"]["model"] == "gemini-3.5-flash"
    assert d["llm"]["shared_client"]["transport"] == "gemini_api"
    assert "test-key" not in str(d)


def test_config_repr():
    config = PageIndexConfig()
    r = repr(config)
    assert "PageIndexConfig" in r
    assert "gemini" in r


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
    model: gpt-4o
    temperature: 0.1
  search:
    max_nodes_per_query: 3
""")
        f.flush()
        config = PageIndexConfig.from_yaml(f.name)

    assert config.tree_build.toc_check_pages == 30
    assert config.tree_build.max_pages_per_node == 15
    # Legacy provider/model keys are ignored; the shared client decides.
    assert config.llm.provider == "gemini"
    assert config.llm.effective_model == "gemini-3.5-flash"
    assert config.llm.temperature == 0.1
    assert config.search.max_nodes_per_query == 3
    os.unlink(f.name)
