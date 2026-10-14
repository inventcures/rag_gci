"""Tests for PageIndex LLM adapter."""
import asyncio

import pytest

from llm import LLMResponse, get_llm_client, set_llm_client
from pageindex_integration.config import LLMConfig
from pageindex_integration.llm_adapter import LLMAdapter


class FakeClient:
    """Stands in for the shared LLM client; records each generate call."""

    def __init__(self, text="ok", ok=True):
        from llm import LLMSettings

        self.settings = LLMSettings(model="gemini-3.5-flash", api_key="test-key")
        self.calls = []
        self._text = text
        self._ok = ok

    def _response(self):
        if self._ok:
            return LLMResponse(self._text, True, "gemini", self.settings.model)
        return LLMResponse("", False, "none", self.settings.model, error="gemini: down")

    async def generate(self, prompt, **kw):
        self.calls.append({"prompt": prompt, **kw})
        return self._response()

    def generate_sync(self, prompt, **kw):
        self.calls.append({"prompt": prompt, **kw})
        return self._response()


@pytest.fixture
def fake_client():
    previous = get_llm_client()
    client = FakeClient(text="answer")
    set_llm_client(client)
    yield client
    set_llm_client(previous)


@pytest.fixture
def adapter():
    # High rpm so the rate limiter never sleeps in tests.
    return LLMAdapter(LLMConfig(rate_limit_rpm=6000))


def test_split_single_user_message():
    system, prompt = LLMAdapter._split_messages([{"role": "user", "content": "hello"}])
    assert system is None
    assert prompt == "hello"


def test_split_system_message():
    system, prompt = LLMAdapter._split_messages([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hello"},
    ])
    assert system == "You are helpful."
    assert prompt == "hello"


def test_split_multi_turn():
    _, prompt = LLMAdapter._split_messages([
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ])
    assert "User: q1" in prompt and "Assistant: a1" in prompt and "User: q2" in prompt


def test_request_defaults(adapter):
    req = adapter._request([{"role": "user", "content": "hello"}])
    assert req["prompt"] == "hello"
    assert req["temperature"] == 0.0
    assert req["max_tokens"] == 4096
    assert req["json_mode"] is False
    assert req["purpose"] == "pageindex"


def test_request_overrides(adapter):
    req = adapter._request([{"role": "user", "content": "hello"}], temperature=0.5, max_tokens=512)
    assert req["temperature"] == 0.5
    assert req["max_tokens"] == 512


def test_chat_async_uses_shared_client(adapter, fake_client):
    text = asyncio.run(adapter.chat_async(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}],
        max_tokens=100,
    ))
    assert text == "answer"
    assert fake_client.calls == [{
        "prompt": "hello", "system": "sys", "temperature": 0.0,
        "max_tokens": 100, "json_mode": False, "purpose": "pageindex",
    }]


def test_chat_async_raises_when_llm_fails(adapter):
    previous = get_llm_client()
    set_llm_client(FakeClient(ok=False))
    try:
        with pytest.raises(RuntimeError, match="gemini: down"):
            asyncio.run(adapter.chat_async([{"role": "user", "content": "hello"}]))
    finally:
        set_llm_client(previous)


def test_chat_sync_uses_shared_client(adapter, fake_client):
    assert adapter.chat_sync([{"role": "user", "content": "hello"}]) == "answer"
    assert fake_client.calls[0]["prompt"] == "hello"


def test_repr_reports_shared_client(adapter, fake_client):
    r = repr(adapter)
    assert "gemini" in r
    assert "gemini-3.5-flash" in r


def test_rate_limit_is_a_window_not_a_gap():
    """Calls up to the per-minute limit go at once; the next waits for the window."""
    import time

    adapter = LLMAdapter(LLMConfig(rate_limit_rpm=2))
    adapter._window_s = 0.3

    async def three():
        started = time.monotonic()
        stamps = []
        for _ in range(3):
            await adapter._rate_limit()
            stamps.append(time.monotonic() - started)
        return stamps

    stamps = asyncio.run(three())
    assert stamps[1] < 0.05
    assert stamps[2] >= 0.25
