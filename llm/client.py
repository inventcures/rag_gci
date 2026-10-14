"""
The one way this codebase calls a text LLM.

Every text-generation call (answer drafting, query translation, knowledge-graph
extraction, Cypher generation, PageIndex tree search, memory and meta agents,
clarification) goes through `LLMClient.generate`. Before this module those calls
were spread across Gemini, Groq (Qwen3, Llama 3.1), OpenAI (gpt-4o, gpt-4o-mini)
and a MedGemma endpoint, each with its own key, model string and error handling.

    primary   Gemini, via google-genai
              Vertex AI when GOOGLE_CLOUD_PROJECT is set (region-pinned with
              llm.location), else the Gemini API with GEMINI_API_KEY
    fallback  DeepSeek, OpenAI-compatible chat completions with DEEPSEEK_API_KEY;
              used only when llm.fallback.enabled is true and the primary failed

Configuration, highest precedence first: environment, the `llm:` block in
config.yaml, built-in default.

    llm:
      model: gemini-3.5-flash         # LLM_MODEL
      location: asia-south1           # LLM_LOCATION (Vertex only)
      timeout_s: 45
      fallback:
        enabled: false                # LLM_FALLBACK_ENABLED
        provider: deepseek
        model: deepseek-flash         # LLM_FALLBACK_MODEL
        base_url: https://api.deepseek.com

The residency position decides the defaults, not cost: see docs/adr/0007. Gemini
3.8 Flash and DeepSeek are each one setting away, and turning either on changes
the release fingerprint (study_release.py), so it shows as drift until the study
release record is updated.

`generate` never raises. A failure comes back as `LLMResponse(ok=False)` with
the reason, so a caller can degrade (template Cypher, untranslated query, the
"consult your doctor" answer) rather than fail a user's turn.
"""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_LOCATION = "asia-south1"
DEFAULT_FALLBACK_MODEL = "deepseek-flash"
DEFAULT_FALLBACK_BASE_URL = "https://api.deepseek.com"

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class LLMSettings:
    model: str = DEFAULT_MODEL
    location: str = DEFAULT_LOCATION
    project: Optional[str] = None
    api_key: Optional[str] = None
    timeout_s: float = 45.0
    fallback_enabled: bool = False
    fallback_provider: str = "deepseek"
    fallback_model: str = DEFAULT_FALLBACK_MODEL
    fallback_base_url: str = DEFAULT_FALLBACK_BASE_URL
    fallback_api_key: Optional[str] = None
    source: Dict[str, str] = field(default_factory=dict)

    @property
    def transport(self) -> str:
        if self.project:
            return "vertex"
        if self.api_key:
            return "gemini_api"
        return "unconfigured"

    def describe(self) -> Dict[str, Any]:
        """What the release record and logs show. Never includes keys."""
        return {
            "provider": "gemini",
            "model": self.model,
            "transport": self.transport,
            "location": self.location if self.transport == "vertex" else "global",
            "fallback": (
                {"provider": self.fallback_provider, "model": self.fallback_model}
                if self.fallback_enabled
                else None
            ),
        }


@dataclass
class LLMResponse:
    text: str
    ok: bool
    provider: str
    model: str
    latency_ms: float = 0.0
    error: Optional[str] = None
    used_fallback: bool = False


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _yaml_llm_block(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {}
    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        block = data.get("llm") or {}
        return block if isinstance(block, dict) else {}
    except Exception:
        logger.exception("Could not read llm settings from %s", config_path)
        return {}


def load_llm_settings(
    config_path: Path = Path("config.yaml"), env: Optional[Dict[str, str]] = None
) -> LLMSettings:
    env = dict(os.environ) if env is None else env
    block = _yaml_llm_block(config_path)
    fb = block.get("fallback") or {}
    source: Dict[str, str] = {}

    def pick(name: str, env_key: str, yaml_value: Any, default: Any) -> Any:
        if env.get(env_key):
            source[name] = "env"
            return env[env_key]
        # Unexpanded ${VAR} placeholders in config.yaml count as unset.
        if yaml_value is not None and not str(yaml_value).startswith("${"):
            source[name] = "config"
            return yaml_value
        source[name] = "default"
        return default

    return LLMSettings(
        model=str(pick("model", "LLM_MODEL", block.get("model"), DEFAULT_MODEL)),
        location=str(pick("location", "LLM_LOCATION", block.get("location"), DEFAULT_LOCATION)),
        project=env.get("GOOGLE_CLOUD_PROJECT") or None,
        api_key=env.get("GEMINI_API_KEY") or None,
        timeout_s=float(block.get("timeout_s", 45)),
        fallback_enabled=_as_bool(pick("fallback_enabled", "LLM_FALLBACK_ENABLED", fb.get("enabled"), False)),
        fallback_provider=str(fb.get("provider", "deepseek")),
        fallback_model=str(pick("fallback_model", "LLM_FALLBACK_MODEL", fb.get("model"), DEFAULT_FALLBACK_MODEL)),
        fallback_base_url=str(fb.get("base_url", DEFAULT_FALLBACK_BASE_URL)).rstrip("/"),
        fallback_api_key=env.get("DEEPSEEK_API_KEY") or None,
        source=source,
    )


class LLMClient:
    """Gemini first, DeepSeek only if enabled. Async; never raises."""

    def __init__(self, settings: Optional[LLMSettings] = None):
        self.settings = settings or load_llm_settings()
        self._genai = None

    # -- primary ----------------------------------------------------------
    def _genai_client(self):
        if self._genai is None:
            from google import genai

            s = self.settings
            if s.transport == "vertex":
                self._genai = genai.Client(vertexai=True, project=s.project, location=s.location)
            else:
                self._genai = genai.Client(api_key=s.api_key)
        return self._genai

    async def _gemini(
        self, prompt: str, system: Optional[str], max_tokens: int, temperature: float, json_mode: bool
    ) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system or None,
            response_mime_type="application/json" if json_mode else None,
        )
        response = await self._genai_client().aio.models.generate_content(
            model=self.settings.model, contents=prompt, config=config
        )
        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise RuntimeError("Gemini returned no text")
        return text

    # -- fallback ---------------------------------------------------------
    async def _deepseek(
        self, prompt: str, system: Optional[str], max_tokens: int, temperature: float, json_mode: bool
    ) -> str:
        import aiohttp

        s = self.settings
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: Dict[str, Any] = {
            "model": s.fallback_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        timeout = aiohttp.ClientTimeout(total=s.timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{s.fallback_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {s.fallback_api_key}", "Content-Type": "application/json"},
                json=payload,
            ) as resp:
                body = await resp.text()
                if resp.status != 200:
                    raise RuntimeError(f"DeepSeek HTTP {resp.status}: {body[:300]}")
                data = json.loads(body)
        text = (data["choices"][0]["message"].get("content") or "").strip()
        if not text:
            raise RuntimeError("DeepSeek returned no text")
        return text

    # -- public -----------------------------------------------------------
    async def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        json_mode: bool = False,
        purpose: str = "",
    ) -> LLMResponse:
        """Generate text. `purpose` labels the call in logs (e.g. "answer")."""
        s = self.settings
        started = time.perf_counter()
        errors: List[str] = []

        if s.transport != "unconfigured":
            try:
                text = await asyncio.wait_for(
                    self._gemini(prompt, system, max_tokens, temperature, json_mode), s.timeout_s
                )
                return LLMResponse(
                    _THINK.sub("", text).strip(), True, "gemini", s.model,
                    (time.perf_counter() - started) * 1000,
                )
            except Exception as e:
                errors.append(f"gemini: {type(e).__name__}: {e}")
                logger.warning("LLM primary failed (%s): %s", purpose or "call", errors[-1])
        else:
            errors.append("gemini: no GOOGLE_CLOUD_PROJECT or GEMINI_API_KEY")

        if s.fallback_enabled:
            if not s.fallback_api_key:
                errors.append("deepseek: DEEPSEEK_API_KEY not set")
            else:
                try:
                    text = await asyncio.wait_for(
                        self._deepseek(prompt, system, max_tokens, temperature, json_mode), s.timeout_s
                    )
                    logger.warning("LLM fallback used (%s): %s", purpose or "call", s.fallback_model)
                    return LLMResponse(
                        _THINK.sub("", text).strip(), True, s.fallback_provider, s.fallback_model,
                        (time.perf_counter() - started) * 1000, used_fallback=True,
                    )
                except Exception as e:
                    errors.append(f"deepseek: {type(e).__name__}: {e}")
                    logger.error("LLM fallback failed (%s): %s", purpose or "call", errors[-1])

        return LLMResponse(
            "", False, "none", s.model, (time.perf_counter() - started) * 1000,
            error="; ".join(errors),
        )

    def generate_sync(self, prompt: str, **kw: Any) -> LLMResponse:
        """For the few synchronous callers. Must not be called inside a running loop."""
        return asyncio.run(self.generate(prompt, **kw))


_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def set_llm_client(client: Optional[LLMClient]) -> Optional[LLMClient]:
    """Replace the shared client (tests, or reconfiguration at startup)."""
    global _client
    _client = client
    return client
