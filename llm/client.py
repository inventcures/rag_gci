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
      thinking:                       # per purpose; "default" for the rest
        answer: low                   # text channels
        voice_answer: minimal         # an answer inside a voice turn
        query_translation: minimal
      deadlines:
        translation_s: 3              # query translation
        answer_translation_s: 8       # answer translation (WhatsApp)
        voice_s: 7                    # a whole voice turn (within_voice_deadline)
      fallback:
        enabled: false                # LLM_FALLBACK_ENABLED
        provider: deepseek
        model: deepseek-flash         # LLM_FALLBACK_MODEL
        base_url: https://api.deepseek.com

The residency position decides the defaults, not cost: see docs/adr/0007. Gemini
3.8 Flash and DeepSeek are each one setting away, and turning either on changes
the release fingerprint (study_release.py), so it shows as drift until the study
release record is updated.

Every call states how much the model thinks (`thinking_for(purpose)`) and how
long its caller can wait. A call's budget is the smaller of its purpose's
timeout and what is left of an enclosing `deadline(...)` (a voice turn); the
fallback provider gets what is left of that same budget, not a fresh timeout.

`generate` never raises. A failure comes back as `LLMResponse(ok=False)` with
the reason, so a caller can degrade (template Cypher, untranslated query, the
"consult your doctor" answer) rather than fail a user's turn.
"""

import asyncio
import contextvars
import json
import logging
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Iterator, List, Optional, TypeVar

from observability import current_span, record_span, span

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_LOCATION = "asia-south1"
DEFAULT_FALLBACK_MODEL = "deepseek-flash"
DEFAULT_FALLBACK_BASE_URL = "https://api.deepseek.com"

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

THINKING_LEVELS = ("minimal", "low", "medium", "high")

# How much the model thinks, per purpose. Chosen by measurement against the
# live model (scripts/benchmark_thinking.py, docs/latency/thinking-11.md,
# docs/decisions/latency.md#11):
#   answer          "low": judged level with "high" (21 wins, 19 losses of 40)
#                   at 7.5 s to first token instead of 14.2 s
#   voice_answer    "minimal": the only level inside a voice deadline (first
#                   token 1.0 s, whole answer 4.7 s), at a quality cost against
#                   "high" (12 wins, 28 losses) that needs clinical review
#   translation, clarification: "minimal"; translations were judged identical
# Background work (graph extraction, memory, meta agents) is not on a caller's
# clock and keeps "high". A "voice_<purpose>" entry applies inside
# within_voice_deadline().
DEFAULT_THINKING: Dict[str, str] = {
    "answer": "low",
    "voice_answer": "minimal",
    "query_translation": "minimal",
    "answer_translation": "minimal",
    "clarification": "minimal",
    "default": "high",
}

# Gemini counts thinking tokens against max_output_tokens. Without headroom a
# thinking model spends the whole cap thinking: at the model's default level,
# 2048-token answers came back cut off mid-sentence after ~400 characters.
# `max_tokens` stays the visible-answer limit; the headroom is added on top.
THINKING_HEADROOM: Dict[str, int] = {"minimal": 512, "low": 2048, "medium": 4096, "high": 8192}

DEFAULT_TRANSLATION_DEADLINE_S = 3.0   # query translation: on every non-English turn
DEFAULT_ANSWER_TRANSLATION_DEADLINE_S = 8.0  # minimal-thinking p95 was 4.5 s
DEFAULT_VOICE_DEADLINE_S = 7.0

# Absolute time.monotonic() by which the enclosing turn must have its answer.
_deadline: "contextvars.ContextVar[Optional[float]]" = contextvars.ContextVar("llm_deadline", default=None)
_voice_turn: "contextvars.ContextVar[bool]" = contextvars.ContextVar("llm_voice_turn", default=False)


def in_voice_turn() -> bool:
    """True inside within_voice_deadline(): a caller is listening."""
    return _voice_turn.get()


@contextmanager
def deadline(seconds: float) -> Iterator[float]:
    """Every LLM call inside this block finishes within `seconds` from now
    (or the enclosing deadline, if sooner). Tasks created inside inherit it."""
    end = time.monotonic() + seconds
    outer = _deadline.get()
    if outer is not None:
        end = min(end, outer)
    token = _deadline.set(end)
    try:
        yield end
    finally:
        _deadline.reset(token)


def remaining() -> Optional[float]:
    """Seconds left before the enclosing deadline, or None outside one."""
    end = _deadline.get()
    return None if end is None else end - time.monotonic()


T = TypeVar("T")

# What the caller hears when the model fails or a voice turn's deadline runs
# out, in place of the error text that used to reach them.
GENERATION_FAILED_ANSWER = (
    "I could not prepare an answer just now. Please ask again in a moment, or "
    "speak to your doctor, nurse or palliative care team."
)

# How long past the voice deadline a turn may run before it is abandoned. The
# LLM call itself ends at the deadline (and the answer path then sends its
# fallback answer); this only bounds work outside the model call.
VOICE_GRACE_S = 1.0


def voice_deadline_s() -> float:
    """How long a voice turn may take to answer (llm.deadlines.voice_s)."""
    return get_llm_client().settings.voice_deadline_s


async def within_voice_deadline(make: Callable[[], Awaitable[T]], seconds: Optional[float] = None) -> T:
    """Run one voice turn's answer under the voice deadline.

    Model calls inside get what is left of `seconds` (llm.deadlines.voice_s by
    default); the turn as a whole is abandoned `VOICE_GRACE_S` later with
    asyncio.TimeoutError, which each voice channel turns into its spoken
    fallback."""
    seconds = voice_deadline_s() if seconds is None else seconds
    token = _voice_turn.set(True)
    try:
        with deadline(seconds):
            return await asyncio.wait_for(make(), seconds + VOICE_GRACE_S)
    finally:
        _voice_turn.reset(token)


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
    thinking: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_THINKING))
    translation_deadline_s: float = DEFAULT_TRANSLATION_DEADLINE_S
    answer_translation_deadline_s: float = DEFAULT_ANSWER_TRANSLATION_DEADLINE_S
    voice_deadline_s: float = DEFAULT_VOICE_DEADLINE_S
    source: Dict[str, str] = field(default_factory=dict)

    def thinking_for(self, purpose: str, voice: bool = False) -> str:
        if voice and self.thinking.get(f"voice_{purpose}"):
            return self.thinking[f"voice_{purpose}"]
        return self.thinking.get(purpose) or self.thinking.get("default") or "high"

    def timeout_for(self, purpose: str) -> float:
        if purpose == "query_translation":
            return min(self.timeout_s, self.translation_deadline_s)
        if purpose == "answer_translation":
            return min(self.timeout_s, self.answer_translation_deadline_s)
        return self.timeout_s

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
            "thinking": dict(sorted(self.thinking.items())),
            "deadlines": self.deadlines(),
        }

    def deadlines(self) -> Dict[str, float]:
        return {"default_s": self.timeout_s, "translation_s": self.translation_deadline_s,
                "answer_translation_s": self.answer_translation_deadline_s,
                "voice_s": self.voice_deadline_s}


class LLMStreamError(RuntimeError):
    """A streamed generation failed; `partial` says whether text was yielded."""

    def __init__(self, message: str, partial: bool):
        super().__init__(message)
        self.partial = partial


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
    deadlines = block.get("deadlines") or {}
    source: Dict[str, str] = {}

    thinking = dict(DEFAULT_THINKING)
    for purpose, level in (block.get("thinking") or {}).items():
        level = str(level).strip().lower()
        if level in THINKING_LEVELS:
            thinking[str(purpose)] = level
        else:
            logger.warning("Ignoring llm.thinking.%s=%r: not one of %s", purpose, level, THINKING_LEVELS)

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
        thinking=thinking,
        translation_deadline_s=float(pick("translation_deadline_s", "LLM_TRANSLATION_DEADLINE_S",
                                          deadlines.get("translation_s"), DEFAULT_TRANSLATION_DEADLINE_S)),
        answer_translation_deadline_s=float(pick(
            "answer_translation_deadline_s", "LLM_ANSWER_TRANSLATION_DEADLINE_S",
            deadlines.get("answer_translation_s"), DEFAULT_ANSWER_TRANSLATION_DEADLINE_S)),
        voice_deadline_s=float(pick("voice_deadline_s", "LLM_VOICE_DEADLINE_S",
                                    deadlines.get("voice_s"), DEFAULT_VOICE_DEADLINE_S)),
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

    def _gemini_config(self, system: Optional[str], max_tokens: int, temperature: float,
                       json_mode: bool, thinking: Optional[str]):
        from google.genai import types

        # Thinking levels are a Gemini 3 setting; older models reject them.
        think = (types.ThinkingConfig(thinking_level=thinking.upper())
                 if thinking and self.settings.model.startswith("gemini-3") else None)
        return types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens + (THINKING_HEADROOM.get(thinking or "", 8192) if think else 0),
            system_instruction=system or None,
            response_mime_type="application/json" if json_mode else None,
            thinking_config=think,
        )

    async def _gemini(
        self, prompt: str, system: Optional[str], max_tokens: int, temperature: float, json_mode: bool,
        thinking: Optional[str] = None,
    ) -> str:
        config = self._gemini_config(system, max_tokens, temperature, json_mode, thinking)
        response = await self._genai_client().aio.models.generate_content(
            model=self.settings.model, contents=prompt, config=config
        )
        usage = getattr(response, "usage_metadata", None)
        active = current_span()
        if usage is not None and active is not None:
            active.set(output_tokens=getattr(usage, "candidates_token_count", None),
                       thought_tokens=getattr(usage, "thoughts_token_count", None))
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
        thinking: Optional[str] = None,
    ) -> LLMResponse:
        """Generate text. `purpose` labels the call and picks its thinking level
        and timeout; `thinking` overrides the level for this call."""
        level = thinking or self.settings.thinking_for(purpose, voice=in_voice_turn())
        budget = self.settings.timeout_for(purpose)
        left = remaining()
        if left is not None:
            budget = min(budget, left)
        with span("llm", purpose=purpose or "call", model=self.settings.model,
                  thinking=level, budget_s=round(budget, 2)) as llm_span:
            response = await self._generate(
                prompt, system=system, max_tokens=max_tokens, temperature=temperature,
                json_mode=json_mode, purpose=purpose, thinking=level, budget=budget,
            )
            llm_span.set(ok=response.ok, provider=response.provider,
                         used_fallback=response.used_fallback)
            return response

    async def _generate(
        self,
        prompt: str,
        *,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        json_mode: bool,
        purpose: str,
        thinking: Optional[str] = None,
        budget: Optional[float] = None,
    ) -> LLMResponse:
        s = self.settings
        started = time.perf_counter()
        end = started + (s.timeout_s if budget is None else budget)
        errors: List[str] = []

        if end - time.perf_counter() <= 0:
            errors.append("no time left in the caller's deadline")
        elif s.transport != "unconfigured":
            try:
                text = await asyncio.wait_for(
                    self._gemini(prompt, system, max_tokens, temperature, json_mode, thinking),
                    end - time.perf_counter(),
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
            left = end - time.perf_counter()
            if not s.fallback_api_key:
                errors.append("deepseek: DEEPSEEK_API_KEY not set")
            elif left <= 0.25:
                # The fallback shares the caller's budget; a fresh timeout here
                # is how a voice turn used to wait 90 s.
                errors.append("deepseek: no time left in the budget")
            else:
                try:
                    text = await asyncio.wait_for(
                        self._deepseek(prompt, system, max_tokens, temperature, json_mode), left
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

    async def _gemini_stream(self, prompt: str, system: Optional[str], max_tokens: int,
                             temperature: float, thinking: Optional[str]) -> AsyncIterator[Any]:
        config = self._gemini_config(system, max_tokens, temperature, False, thinking)
        return await self._genai_client().aio.models.generate_content_stream(
            model=self.settings.model, contents=prompt, config=config)

    async def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        purpose: str = "",
        thinking: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Yield the answer as the model writes it, within the same budget as
        `generate`. If Gemini fails before its first text, the fallback (when
        enabled) answers in one piece. Raises LLMStreamError on failure.

        Each wait for the next piece is bounded by what is left of the budget;
        the span records time to first text."""
        s = self.settings
        level = thinking or s.thinking_for(purpose, voice=in_voice_turn())
        budget = s.timeout_for(purpose)
        left = remaining()
        if left is not None:
            budget = min(budget, left)
        started = time.perf_counter()
        end = started + budget
        attrs: Dict[str, Any] = {"purpose": purpose or "call", "model": s.model, "thinking": level,
                                 "budget_s": round(budget, 2), "streamed": True}
        errors: List[str] = []
        yielded = False
        try:
            if budget <= 0:
                errors.append("no time left in the caller's deadline")
            elif s.transport != "unconfigured":
                try:
                    chunks = await asyncio.wait_for(
                        self._gemini_stream(prompt, system, max_tokens, temperature, level),
                        end - time.perf_counter())
                    iterator = chunks.__aiter__()
                    while True:
                        try:
                            chunk = await asyncio.wait_for(iterator.__anext__(), end - time.perf_counter())
                        except StopAsyncIteration:
                            break
                        usage = getattr(chunk, "usage_metadata", None)
                        if usage is not None:
                            attrs.update(output_tokens=getattr(usage, "candidates_token_count", None),
                                         thought_tokens=getattr(usage, "thoughts_token_count", None))
                        text = getattr(chunk, "text", None)
                        if text:
                            if not yielded:
                                attrs["first_text_ms"] = round((time.perf_counter() - started) * 1000, 1)
                                yielded = True
                            yield text
                    if yielded:
                        attrs.update(ok=True, provider="gemini")
                        return
                    errors.append("gemini: returned no text")
                except (Exception, asyncio.TimeoutError) as e:
                    errors.append(f"gemini: {type(e).__name__}: {e}")
                    logger.warning("LLM stream failed (%s): %s", purpose or "call", errors[-1])
                    if yielded:
                        raise LLMStreamError("; ".join(errors), partial=True) from e
            else:
                errors.append("gemini: no GOOGLE_CLOUD_PROJECT or GEMINI_API_KEY")

            left = end - time.perf_counter()
            if s.fallback_enabled and s.fallback_api_key and left > 0.25:
                try:
                    text = await asyncio.wait_for(
                        self._deepseek(prompt, system, max_tokens, temperature, False), left)
                    attrs.update(ok=True, provider=s.fallback_provider, used_fallback=True,
                                 first_text_ms=round((time.perf_counter() - started) * 1000, 1))
                    yielded = True
                    yield _THINK.sub("", text).strip()
                    return
                except Exception as e:
                    errors.append(f"deepseek: {type(e).__name__}: {e}")
            raise LLMStreamError("; ".join(errors), partial=False)
        finally:
            attrs.setdefault("ok", False)
            record_span("llm", (time.perf_counter() - started) * 1000, **attrs)

    def generate_sync(self, prompt: str, **kw: Any) -> LLMResponse:
        """For the few synchronous callers. Must not be called inside a running loop."""
        return asyncio.run(self.generate(prompt, **kw))


_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


async def warm_up(client: Optional[LLMClient] = None) -> bool:
    """Build the shared client's Gemini connection object off the event loop
    (startup). Returns whether it is ready; never raises."""
    client = client or get_llm_client()
    if client.settings.transport == "unconfigured":
        return False
    started = time.perf_counter()
    try:
        genai_client = await asyncio.to_thread(client._genai_client)
        if client.settings.transport == "vertex":
            # The first OAuth token exchange would otherwise sit inside the
            # first caller's answer. Private in google-genai, so best-effort:
            # a renamed method only loses the warm-up.
            fetch = getattr(getattr(genai_client, "_api_client", None), "_access_token", None)
            if callable(fetch):
                await asyncio.to_thread(fetch)
    except Exception:
        logger.warning("LLM client warm-up failed; it will be built on first use", exc_info=True)
        return False
    record_span("llm.warm_up", (time.perf_counter() - started) * 1000,
                transport=client.settings.transport)
    return True


def set_llm_client(client: Optional[LLMClient]) -> Optional[LLMClient]:
    """Replace the shared client (tests, or reconfiguration at startup)."""
    global _client
    _client = client
    return client
