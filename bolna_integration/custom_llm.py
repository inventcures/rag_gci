"""
The OpenAI chat-completions protocol, as Bolna's custom LLM speaks it.

A Bolna agent built by bolna_integration.config names this server as its LLM
(`provider: custom`, `base_url: <server>/api/bolna/llm/<language>/v1`). Bolna
then POSTs every turn to `<base_url>/chat/completions` with the OpenAI SDK and
speaks the reply. These helpers read the turn out of that request and write the
guarded answer back in OpenAI's shape, whole or streamed (server-sent events
ending in `data: [DONE]`). See https://www.bolna.ai/docs/customizations/using-custom-llm.
"""

import json
import re
import time
import uuid
from typing import Any, Dict, Iterator, List

from bolna_integration.config import CALLER_LINE

# CALLER_LINE with each {variable} as a named group: what Bolna substituted.
_CALLER_PATTERN = re.compile(
    re.escape(CALLER_LINE)
    .replace(re.escape("{from_number}"), r"(?P<from_number>[^;\n]*)")
    .replace(re.escape("{execution_id}"), r"(?P<execution_id>\S*)")
)


def _text(content: Any) -> str:
    """A message's text: a string, or the text parts of a content list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(part.get("text", "")) for part in content
                        if isinstance(part, dict) and part.get("type") == "text")
    return ""


def last_user_text(messages: List[Dict[str, Any]]) -> str:
    """The caller's latest words: the last user message's text."""
    for message in reversed(messages or []):
        if isinstance(message, dict) and message.get("role") == "user":
            return _text(message.get("content")).strip()
    return ""


def caller_fields(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    """from_number and execution_id from the filled CALLER_LINE in the system prompt.

    Empty when the prompt has no such line. An unfilled variable comes back as
    written ("{from_number}"), which bolna_caller_id does not take as an identity.
    """
    for message in messages or []:
        if isinstance(message, dict) and message.get("role") == "system":
            found = _CALLER_PATTERN.search(_text(message.get("content")))
            if found:
                return {key: value.strip() for key, value in found.groupdict().items()}
    return {}


def completion_body(text: str, model: str) -> Dict[str, Any]:
    """A whole chat completion whose one choice says `text`."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": text}}],
    }


def completion_events(text: str, model: str) -> Iterator[str]:
    """The same reply streamed: one content delta, a stop chunk, then [DONE]."""
    base = {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion.chunk",
            "created": int(time.time()), "model": model}
    chunks = [
        {"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None},
        {"index": 0, "delta": {}, "finish_reason": "stop"},
    ]
    for choice in chunks:
        yield f"data: {json.dumps({**base, 'choices': [choice]}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
