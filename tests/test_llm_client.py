"""
The shared LLM client (llm/client.py): settings precedence, Gemini first, DeepSeek
only when enabled, and never raising.
"""

import asyncio
from pathlib import Path

import pytest

from llm.client import LLMClient, LLMSettings, load_llm_settings


def _config(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(body, encoding="utf-8")
    return p


class FakeClient(LLMClient):
    """Records which providers were called; each returns or raises as told."""

    def __init__(self, settings, gemini=None, deepseek=None):
        super().__init__(settings)
        self.calls = []
        self._g = gemini
        self._d = deepseek

    async def _gemini(self, prompt, system, max_tokens, temperature, json_mode, thinking=None):
        self.calls.append("gemini")
        if isinstance(self._g, BaseException):
            raise self._g
        if self._g == "hang":
            await asyncio.sleep(10)
        return self._g

    async def _deepseek(self, prompt, system, max_tokens, temperature, json_mode):
        self.calls.append("deepseek")
        if isinstance(self._d, BaseException):
            raise self._d
        return self._d


def run(coro):
    return asyncio.run(coro)


# -- settings ---------------------------------------------------------------

def test_defaults_are_india_resident_and_fallback_off(tmp_path):
    s = load_llm_settings(tmp_path / "missing.yaml", env={})
    assert s.model == "gemini-3.5-flash"
    assert s.location == "asia-south1"
    assert s.fallback_enabled is False
    assert s.describe()["fallback"] is None


def test_repo_config_matches_defaults():
    s = load_llm_settings(Path("config.yaml"), env={"GOOGLE_CLOUD_PROJECT": "p"})
    d = s.describe()
    assert (d["model"], d["location"], d["transport"], d["fallback"]) == (
        "gemini-3.5-flash", "asia-south1", "vertex", None,
    )


def test_env_beats_config_beats_default(tmp_path):
    cfg = _config(tmp_path, "llm:\n  model: gemini-3.8-flash\n  location: us-central1\n")
    s = load_llm_settings(cfg, env={"LLM_MODEL": "gemini-x"})
    assert s.model == "gemini-x" and s.source["model"] == "env"
    assert s.location == "us-central1" and s.source["location"] == "config"


def test_unexpanded_placeholder_counts_as_unset(tmp_path):
    cfg = _config(tmp_path, "llm:\n  model: ${LLM_MODEL}\n")
    assert load_llm_settings(cfg, env={}).model == "gemini-3.5-flash"


def test_fallback_enabled_from_config_and_env(tmp_path):
    cfg = _config(tmp_path, "llm:\n  fallback:\n    enabled: true\n    model: deepseek-v4-flash\n")
    s = load_llm_settings(cfg, env={"DEEPSEEK_API_KEY": "k"})
    assert s.fallback_enabled and s.fallback_model == "deepseek-v4-flash"
    assert s.describe()["fallback"] == {"provider": "deepseek", "model": "deepseek-v4-flash"}
    assert load_llm_settings(cfg, env={"LLM_FALLBACK_ENABLED": "false"}).fallback_enabled is False


def test_describe_never_includes_keys():
    s = LLMSettings(api_key="secret-gemini", fallback_enabled=True, fallback_api_key="secret-ds")
    assert "secret" not in repr(s.describe())


def test_transport_selection():
    assert LLMSettings(project="p").transport == "vertex"
    assert LLMSettings(api_key="k").transport == "gemini_api"
    assert LLMSettings().transport == "unconfigured"
    # The Gemini API is not region-pinned, so the location reported is global.
    assert LLMSettings(api_key="k").describe()["location"] == "global"


# -- generate ---------------------------------------------------------------

def test_primary_success_strips_think_blocks():
    c = FakeClient(LLMSettings(project="p"), gemini="<think>plan</think> Answer.")
    r = run(c.generate("q"))
    assert r.ok and r.text == "Answer." and r.provider == "gemini" and not r.used_fallback
    assert c.calls == ["gemini"]


def test_no_deepseek_call_when_fallback_disabled():
    c = FakeClient(LLMSettings(project="p", fallback_api_key="k"), gemini=RuntimeError("down"), deepseek="x")
    r = run(c.generate("q"))
    assert not r.ok and r.text == ""
    assert c.calls == ["gemini"]
    assert "down" in r.error


def test_fallback_used_only_when_enabled_and_primary_fails():
    s = LLMSettings(project="p", fallback_enabled=True, fallback_api_key="k")
    ok = FakeClient(s, gemini="fine", deepseek="never")
    assert run(ok.generate("q")).provider == "gemini" and ok.calls == ["gemini"]

    failing = FakeClient(s, gemini=RuntimeError("503"), deepseek="from deepseek")
    r = run(failing.generate("q"))
    assert r.ok and r.used_fallback and r.provider == "deepseek" and r.text == "from deepseek"
    assert failing.calls == ["gemini", "deepseek"]


def test_fallback_enabled_without_key_is_not_called():
    c = FakeClient(LLMSettings(project="p", fallback_enabled=True), gemini=RuntimeError("x"), deepseek="y")
    r = run(c.generate("q"))
    assert not r.ok and c.calls == ["gemini"] and "DEEPSEEK_API_KEY" in r.error


def test_unconfigured_primary_is_skipped_not_raised():
    c = FakeClient(LLMSettings(), gemini="unused")
    r = run(c.generate("q"))
    assert not r.ok and c.calls == [] and "GEMINI_API_KEY" in r.error


def test_timeout_is_reported_not_raised():
    c = FakeClient(LLMSettings(project="p", timeout_s=0.05), gemini="hang")
    r = run(c.generate("q"))
    assert not r.ok and "TimeoutError" in r.error


def test_both_fail_returns_both_reasons():
    s = LLMSettings(project="p", fallback_enabled=True, fallback_api_key="k")
    c = FakeClient(s, gemini=RuntimeError("g-err"), deepseek=RuntimeError("d-err"))
    r = run(c.generate("q"))
    assert not r.ok and "g-err" in r.error and "d-err" in r.error


def test_release_fingerprint_reads_shared_client():
    from study_release import StudyRelease

    fp = StudyRelease._llm_settings()
    assert fp["model"] == "gemini-3.5-flash" and fp["fallback"] == "none"


def test_warm_up_builds_the_client_off_the_event_loop():
    import threading

    from llm.client import warm_up

    client = LLMClient(LLMSettings(api_key="k"))
    built_on = []

    def build():
        built_on.append(threading.current_thread())
        return object()

    client._genai_client = build
    assert run(warm_up(client)) is True
    assert built_on and built_on[0] is not threading.main_thread()


def test_warm_up_without_credentials_does_nothing():
    from llm.client import warm_up

    client = LLMClient(LLMSettings())
    client._genai_client = lambda: (_ for _ in ()).throw(AssertionError("built"))
    assert run(warm_up(client)) is False


def test_warm_up_failure_is_not_raised():
    from llm.client import warm_up

    client = LLMClient(LLMSettings(api_key="k"))

    def broken():
        raise RuntimeError("no credentials file")

    client._genai_client = broken
    assert run(warm_up(client)) is False
