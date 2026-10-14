"""
LLM Adapter for PageIndex Integration

Chat-message interface over the shared LLM client (llm/client.py: Gemini,
optional DeepSeek fallback). The tree builder and query engine pass chat
messages; this adapter turns them into one prompt plus a system instruction.

Usage:
    from pageindex_integration.config import LLMConfig
    from pageindex_integration.llm_adapter import LLMAdapter

    adapter = LLMAdapter(LLMConfig())
    response = await adapter.chat_async([
        {"role": "user", "content": "Analyze this tree..."},
    ])
"""

import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple

from llm import get_llm_client
from pageindex_integration.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMAdapter:
    """
    Chat interface for PageIndex over the shared LLM client.

    Features:
    - Sync and async chat completions
    - Rate limiting (requests per minute)
    - Raises RuntimeError when the LLM does not answer, so callers keep
      their non-LLM fallbacks (fixed-size sections, title as summary)
    """

    def __init__(self, config: LLMConfig):
        self._config = config
        self._last_request_time = 0.0
        self._min_interval = 60.0 / max(config.rate_limit_rpm, 1)

    @staticmethod
    def _split_messages(messages: List[Dict[str, str]]) -> Tuple[Optional[str], str]:
        """Return (system instruction, prompt) from chat messages."""
        system_parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
        turns = [m for m in messages if m.get("role") != "system"]
        if len(turns) == 1:
            prompt = turns[0].get("content", "")
        else:
            prompt = "\n\n".join(
                f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}" for m in turns
            )
        return ("\n\n".join(system_parts) or None), prompt

    def _request(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Keyword arguments for LLMClient.generate."""
        system, prompt = self._split_messages(messages)
        return {
            "prompt": prompt,
            "system": system,
            "temperature": kwargs.get("temperature", self._config.temperature),
            "max_tokens": kwargs.get("max_tokens", self._config.max_tokens),
            "json_mode": kwargs.get("json_mode", False),
            "purpose": kwargs.get("purpose", "pageindex"),
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
        Async chat completion with rate limiting.

        Args:
            messages: Chat messages ({"role": ..., "content": ...})
            **kwargs: Override temperature, max_tokens; set json_mode, purpose

        Returns:
            Response content string

        Raises:
            RuntimeError: If the shared client returned no answer
        """
        await self._rate_limit()
        request = self._request(messages, **kwargs)
        prompt = request.pop("prompt")
        response = await get_llm_client().generate(prompt, **request)
        if not response.ok:
            raise RuntimeError(f"LLM request failed: {response.error}")
        return response.text

    def chat_sync(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> str:
        """Synchronous chat completion for non-async contexts (not inside a running loop)."""
        request = self._request(messages, **kwargs)
        prompt = request.pop("prompt")
        response = get_llm_client().generate_sync(prompt, **request)
        if not response.ok:
            raise RuntimeError(f"LLM request failed: {response.error}")
        return response.text

    def __repr__(self) -> str:
        return (
            f"LLMAdapter(provider={self._config.provider}, "
            f"model={self._config.effective_model})"
        )
