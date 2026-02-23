"""Tests for PageIndex LLM adapter."""
import pytest
from pageindex_integration.config import LLMConfig
from pageindex_integration.llm_adapter import LLMAdapter


@pytest.fixture
def groq_adapter():
    config = LLMConfig(provider="groq", model="test-model")
    return LLMAdapter(config)


@pytest.fixture
def openai_adapter():
    config = LLMConfig(provider="openai", model="gpt-4o-mini")
    return LLMAdapter(config)


def test_groq_headers(groq_adapter):
    headers = groq_adapter._headers()
    assert "Authorization" in headers
    assert "Content-Type" in headers
    assert headers["Content-Type"] == "application/json"


def test_groq_payload(groq_adapter):
    messages = [{"role": "user", "content": "hello"}]
    payload = groq_adapter._payload(messages)
    assert payload["model"] == "test-model"
    assert payload["messages"] == messages
    assert payload["temperature"] == 0.0


def test_payload_overrides(groq_adapter):
    messages = [{"role": "user", "content": "hello"}]
    payload = groq_adapter._payload(messages, temperature=0.5, max_tokens=512)
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 512


def test_openai_payload(openai_adapter):
    messages = [{"role": "system", "content": "You are helpful."}]
    payload = openai_adapter._payload(messages)
    assert payload["model"] == "gpt-4o-mini"


def test_groq_repr(groq_adapter):
    r = repr(groq_adapter)
    assert "groq" in r
    assert "test-model" in r


def test_openai_repr(openai_adapter):
    r = repr(openai_adapter)
    assert "openai" in r
    assert "gpt-4o-mini" in r


def test_rate_limit_interval():
    config = LLMConfig(provider="groq", rate_limit_rpm=60)
    adapter = LLMAdapter(config)
    assert adapter._min_interval == pytest.approx(1.0, abs=0.01)

    config2 = LLMConfig(provider="groq", rate_limit_rpm=30)
    adapter2 = LLMAdapter(config2)
    assert adapter2._min_interval == pytest.approx(2.0, abs=0.01)
