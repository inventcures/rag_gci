"""
LLM Adapter for PageIndex Integration

Unified interface supporting Groq and OpenAI backends via their
OpenAI-compatible chat completions APIs.

Groq:   https://api.groq.com/openai/v1/chat/completions
OpenAI: https://api.openai.com/v1/chat/completions

Usage:
    from pageindex_integration.config import LLMConfig
    from pageindex_integration.llm_adapter import LLMAdapter

    adapter = LLMAdapter(LLMConfig(provider="groq"))
    response = await adapter.chat_async([
        {"role": "user", "content": "Analyze this tree..."},
    ])
"""

import logging
import asyncio
import time
from typing import List, Dict, Any

import aiohttp
import requests

from pageindex_integration.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMAdapter:
    """
    Unified LLM interface for Groq and OpenAI backends.

    Features:
    - Sync and async chat completions
    - Automatic retry with exponential backoff
    - Rate limiting (requests per minute)
    """

    def __init__(self, config: LLMConfig):
        self._config = config
        self._last_request_time = 0.0
        self._min_interval = 60.0 / max(config.rate_limit_rpm, 1)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        return {
            "model": self._config.effective_model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self._config.temperature),
            "max_tokens": kwargs.get("max_tokens", self._config.max_tokens),
        }

    async def _rate_limit(self) -> None:
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    async def chat_async(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> str:
        """
        Async chat completion with retry and rate limiting.

        Args:
            messages: Chat messages in OpenAI format
            **kwargs: Override temperature, max_tokens

        Returns:
            Response content string

        Raises:
            RuntimeError: After exhausting retries
        """
        url = f"{self._config.base_url}/chat/completions"
        payload = self._payload(messages, **kwargs)

        for attempt in range(self._config.max_retries):
            await self._rate_limit()
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        headers=self._headers(),
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self._config.request_timeout),
                    ) as resp:
                        if resp.status == 429:
                            retry_after = float(resp.headers.get("Retry-After", 2 ** attempt))
                            logger.warning(f"Rate limited, waiting {retry_after}s")
                            await asyncio.sleep(retry_after)
                            continue
                        resp.raise_for_status()
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"]
            except (aiohttp.ClientError, asyncio.TimeoutError, KeyError) as e:
                if attempt == self._config.max_retries - 1:
                    raise RuntimeError(
                        f"LLM request failed after {self._config.max_retries} attempts: {e}"
                    )
                wait = 2 ** attempt
                logger.warning(f"LLM request failed (attempt {attempt + 1}), retrying in {wait}s: {e}")
                await asyncio.sleep(wait)

        raise RuntimeError("LLM request failed: exhausted retries")

    def chat_sync(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> str:
        """Synchronous chat completion for non-async contexts."""
        url = f"{self._config.base_url}/chat/completions"
        payload = self._payload(messages, **kwargs)

        for attempt in range(self._config.max_retries):
            try:
                resp = requests.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self._config.request_timeout,
                )
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", 2 ** attempt))
                    time.sleep(retry_after)
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except (requests.RequestException, KeyError) as e:
                if attempt == self._config.max_retries - 1:
                    raise RuntimeError(
                        f"LLM request failed after {self._config.max_retries} attempts: {e}"
                    )
                time.sleep(2 ** attempt)

        raise RuntimeError("LLM request failed: exhausted retries")

    def __repr__(self) -> str:
        return (
            f"LLMAdapter(provider={self._config.provider}, "
            f"model={self._config.effective_model})"
        )
