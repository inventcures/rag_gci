"""
Phone providers that send call audio or text outside India (decision D11).

Identifiable data stays India-hosted, so in the study release a phone provider
serves only when the release record lists every outside service it uses under
`providers.outside_india` (data/study/release.yaml) *and* the operator has
turned it on. Anything else is refused with a message that says which services
the record leaves out.

The services:

  * Bolna: the Bolna platform, plus the providers its agent configuration
    names (telephony, transcriber, synthesizer). They are read from the agent
    this server builds (bolna_integration.config), so the list cannot drift
    from what Bolna is given. The agent's LLM is this server, so no outside
    LLM is listed.
  * Retell: the Retell platform (call audio and transcription) and its
    Cartesia voice, plus `retell.post_call_analysis` (gpt-4o-mini) when
    RETELL_POST_CALL_ANALYSIS=true.

study_release registers each service as an outside provider, so the release
fingerprint lists the ones the server has turned on, and one turned on but not
listed shows as RELEASE DRIFT.
"""

import functools
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import yaml
from fastapi import HTTPException

logger = logging.getLogger(__name__)

BOLNA = "bolna"
RETELL = "retell"
RETELL_SERVICES = ("retell", "retell.cartesia")
RETELL_ANALYSIS = "retell.post_call_analysis"

# The operator's switch for Bolna (config.yaml bolna.enabled, or BOLNA_ENABLED).
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

_LABELS = {BOLNA: "Bolna", RETELL: "Retell"}


def _flag(name: str) -> bool:
    """An environment switch: on only when set to "true"."""
    return os.getenv(name, "").strip().lower() == "true"


def bolna_enabled() -> bool:
    """Bolna is turned on: config.yaml `bolna.enabled: true` or BOLNA_ENABLED=true."""
    if _flag("BOLNA_ENABLED"):
        return True
    if not CONFIG_PATH.exists():
        return False
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    section = config.get("bolna") if isinstance(config, dict) else None
    return isinstance(section, dict) and section.get("enabled") is True


def retell_enabled() -> bool:
    """Retell is turned on: RETELL_ENABLED=true (set by `-p r`)."""
    return _flag("RETELL_ENABLED")


def retell_analysis_enabled() -> bool:
    """Retell's post-call analysis (gpt-4o-mini) is asked for: RETELL_POST_CALL_ANALYSIS=true."""
    return _flag("RETELL_POST_CALL_ANALYSIS")


@functools.lru_cache(maxsize=1)
def bolna_services() -> List[str]:
    """The outside services of the Bolna agent this server builds."""
    from bolna_integration.config import bolna_outside_services, get_palli_sahayak_agent_config

    return bolna_outside_services(get_palli_sahayak_agent_config(server_url="https://server.invalid",
                                                                 language="en"))


def services(provider: str) -> List[str]:
    """Every outside service `provider` uses as configured now; each must be listed."""
    if provider == BOLNA:
        return list(bolna_services())
    if provider == RETELL:
        return [*RETELL_SERVICES, *([RETELL_ANALYSIS] if retell_analysis_enabled() else [])]
    raise ValueError(f"{provider!r} is not a phone provider")


def all_services() -> List[str]:
    """Every service name a phone provider can use (for the fingerprint registry)."""
    return [*bolna_services(), *RETELL_SERVICES, RETELL_ANALYSIS]


def service_enabled(name: str) -> bool:
    """Whether the server has turned on the provider that uses service `name`."""
    if name == BOLNA or name.startswith(BOLNA + "."):
        return bolna_enabled()
    if name == RETELL_ANALYSIS:
        return retell_enabled() and retell_analysis_enabled()
    if name in RETELL_SERVICES:
        return retell_enabled()
    raise ValueError(f"{name!r} is not a phone service")


def _enabled(provider: str) -> bool:
    return bolna_enabled() if provider == BOLNA else retell_enabled()


def released_services() -> Set[str]:
    """The outside providers the release record lists."""
    from study_release import get_study_release

    return set(get_study_release().approved_outside_providers())


def refusal(provider: str) -> Optional[str]:
    """Why `provider` may not serve now, or None when it may.

    The record is checked first: a service it does not list is refused even
    when the operator turned the provider on.
    """
    label = _LABELS.get(provider)
    if label is None:
        raise ValueError(f"{provider!r} is not a phone provider")
    missing = [name for name in services(provider) if name not in released_services()]
    if missing:
        return (
            f"{label} is off in this study release: it sends call audio or text outside India "
            f"({', '.join(missing)}), and the release record does not list "
            f"{'it' if len(missing) == 1 else 'them'} under providers.outside_india."
        )
    if not _enabled(provider):
        switch = "bolna.enabled or BOLNA_ENABLED" if provider == BOLNA else "RETELL_ENABLED (-p r)"
        return f"{label} is not enabled on this server ({switch})."
    return None


def require_released(provider: str) -> Callable[[], None]:
    """A FastAPI dependency: 503 with the refusal when `provider` may not serve.

    List it after the route's webhook verifier, so an unsigned request is still
    rejected as unverified.
    """
    async def dependency() -> None:
        message = refusal(provider)
        if message:
            logger.warning("Phone route refused: %s", message)
            raise HTTPException(status_code=503, detail=message)

    dependency.__name__ = f"require_{provider}_released"
    return dependency


def registry() -> Dict[str, Callable[[], bool]]:
    """Service name to its enabled check, for study_release.register_outside_provider."""
    return {name: functools.partial(service_enabled, name) for name in all_services()}
