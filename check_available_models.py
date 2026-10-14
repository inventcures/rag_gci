#!/usr/bin/env python3
"""
Show the configured text LLM (Gemini primary, optional DeepSeek fallback) and
make one test call through the shared client in llm/client.py.
"""

import asyncio
import json

from dotenv import load_dotenv

from llm import LLMClient, load_llm_settings

load_dotenv()


def main() -> None:
    settings = load_llm_settings()
    print(json.dumps(settings.describe(), indent=2))
    print(f"settings source: {settings.source}")

    response = asyncio.run(
        LLMClient(settings).generate("Reply with the single word: ok", max_tokens=16, purpose="check")
    )
    if response.ok:
        via = " (fallback)" if response.used_fallback else ""
        print(f"✅ {response.provider}/{response.model}{via} answered in {response.latency_ms:.0f} ms: {response.text!r}")
    else:
        print(f"❌ No model answered: {response.error}")


if __name__ == "__main__":
    main()
