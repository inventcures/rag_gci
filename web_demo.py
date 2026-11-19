"""
Web voice demo settings
=======================
Two things the public web voice page (``/voice/``) needs on a demo
deployment, such as the Railway test service (docs/deploy_voice_demo.md):

* **An access code.** When ``WEB_VOICE_ACCESS_CODE`` is set, ``/ws/voice``
  refuses a socket whose ``?code=`` does not match it. The page passes on the
  ``code`` in its own address, so the demo link is
  ``https://<host>/voice/?code=<code>``. Anyone who only saw the bare address
  on a slide cannot start a conversation and run up Gemini charges. Unset,
  the socket is open, as before.

* **The made-up-people notice.** A demo deployment is not India-hosted, so
  the page tells visitors not to share real names or health details. It is
  shown when ``WEB_VOICE_DEMO_NOTICE`` is true, and by default on any Railway
  deployment (Railway sets ``RAILWAY_ENVIRONMENT_NAME``), because Railway has
  no India region and is never the study host.
"""

from __future__ import annotations

import hmac
import os
from typing import Dict, Mapping, Optional

# How long the voice socket waits before trying again to reopen a Live
# session that ended mid-conversation, after a failed attempt.
LIVE_REOPEN_BACKOFF_S = 5.0

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def access_code(env: Optional[Mapping[str, str]] = None) -> str:
    """The configured access code; "" when the demo is open."""
    env = os.environ if env is None else env
    return env.get("WEB_VOICE_ACCESS_CODE", "").strip()


def access_code_matches(given: Optional[str], env: Optional[Mapping[str, str]] = None) -> bool:
    """True when no code is configured, or `given` is the configured code."""
    expected = access_code(env)
    if not expected:
        return True
    return hmac.compare_digest((given or "").strip().encode(), expected.encode())


def demo_notice(env: Optional[Mapping[str, str]] = None) -> bool:
    """Whether the page shows the made-up-people notice."""
    env = os.environ if env is None else env
    value = env.get("WEB_VOICE_DEMO_NOTICE", "").strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    return bool(env.get("RAILWAY_ENVIRONMENT_NAME", "").strip())


def demo_settings(env: Optional[Mapping[str, str]] = None) -> Dict[str, bool]:
    """What /api/voice/demo returns. Never the code itself."""
    return {"notice": demo_notice(env), "code_required": bool(access_code(env))}
