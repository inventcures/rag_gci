#!/usr/bin/env python3
"""
Voice providers and arbitration
===============================
The real-time path is Gemini Live (ADR 0007). The fallback is the India-resident
Sarvam path. This module decides which one answers.

Why arbitration is separate
---------------------------
Because the decision is the risky part and it is pure. A Live session that
connects and then goes quiet mid-answer is worse than one that never connects:
the user is left mid-sentence with the app apparently still thinking. So the
decision is made on measured latency, not on whether a socket opened, and it lives
here where it can be tested without a provider, a network, or a clock.

Provider availability
---------------------
`GeminiLiveProvider` imports `gemini_live` lazily. `google-genai` is not
installed in every environment, and the study build must still work when it is
not, so the import failing resolves to "Live unavailable" and the Sarvam path
answers. That is the same shape as the fallback decision itself: a provider that
cannot serve must not be able to strand a user mid-conversation.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from voice_provider import PROVIDER_LIVE, VoiceProviderControl, get_voice_provider_control
from voice_session import (
    FIRST_TOKEN_TIMEOUT_S,
    PER_TURN_STALL_TIMEOUT_S,
    VOICE_PATH_FALLBACK,
    VOICE_PATH_LIVE,
    VoiceSession,
    VoiceTurn,
    classify_fallback,
)

logger = logging.getLogger(__name__)


class SupportsAudioTurns(Protocol):
    """
    The slice of VoiceTurnHandler the router actually uses.

    Declared structurally so the router can be driven by a test double without the
    type checker rejecting it, and so adding a method to VoiceTurnHandler does not
    silently widen what the router depends on.
    """

    async def handle_audio(
        self, session: VoiceSession, audio: bytes, voice_path: str = ...
    ) -> Optional[VoiceTurn]: ...


@dataclass
class LiveAvailability:
    """Whether the Live path can be used at all, and why not when it cannot."""
    available: bool
    reason: Optional[str] = None

    @classmethod
    def probe(cls) -> "LiveAvailability":
        try:
            import google.genai  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return cls(False, f"live_sdk_unavailable: {type(exc).__name__}")
        try:
            from gemini_live import GeminiLiveService  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return cls(False, f"live_service_unavailable: {type(exc).__name__}")
        return cls(True)


@dataclass
class ProviderOutcome:
    """What a provider returned, including how long it took to first respond."""
    turn: Optional[VoiceTurn]
    first_token_latency_s: Optional[float] = None
    turn_latency_s: Optional[float] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.turn is not None and self.error is None


class LiveVoiceProvider:
    """
    Gemini Live, proxied through the server's own session service.

    The client never talks to Google. Audio goes to our endpoint, we hold the Live
    session, which is what keeps the release identifier and the study log in one
    place.
    """

    def __init__(self, service: Any, session: VoiceSession):
        self.service = service
        self.session = session

    async def serve(self, audio: bytes) -> ProviderOutcome:
        """
        Stream one Live turn, cut short when Live goes quiet.

        The first event must arrive within FIRST_TOKEN_TIMEOUT_S and the final
        turn within PER_TURN_STALL_TIMEOUT_S of the start (ADR 0007's
        "first-token timeout or per-turn stall", the release's
        voice_fallback trigger). Past either, the outcome carries the reason
        and no turn, so the router answers on Sarvam instead of leaving the
        caller in silence.
        """
        started = time.time()
        first_token_at: Optional[float] = None
        stream: Any = None
        try:
            stream = self.service.stream_audio(audio, self.session.language)
            while True:
                if first_token_at is None:
                    budget, reason = FIRST_TOKEN_TIMEOUT_S, "first_token_timeout"
                else:
                    budget, reason = PER_TURN_STALL_TIMEOUT_S - (time.time() - started), "turn_stalled"
                try:
                    event = await asyncio.wait_for(stream.__anext__(), timeout=max(budget, 0.0))
                except asyncio.TimeoutError:
                    logger.warning("Live went quiet (%s); falling back to Sarvam", reason)
                    return ProviderOutcome(
                        turn=None,
                        first_token_latency_s=(first_token_at - started) if first_token_at else None,
                        turn_latency_s=time.time() - started,
                        error=reason,
                    )
                except StopAsyncIteration:
                    break
                if first_token_at is None:
                    first_token_at = time.time()
                if event.get("final"):
                    return ProviderOutcome(
                        turn=event.get("turn"),
                        first_token_latency_s=first_token_at - started,
                        turn_latency_s=time.time() - started,
                    )
            return ProviderOutcome(
                turn=None,
                first_token_latency_s=(first_token_at - started) if first_token_at else None,
                turn_latency_s=time.time() - started,
                error="live_produced_no_final_turn",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Live provider failed: %s", exc)
            return ProviderOutcome(
                turn=None,
                turn_latency_s=time.time() - started,
                error=f"live_error: {type(exc).__name__}",
            )
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:  # noqa: BLE001
                    logger.debug("Live stream did not close cleanly", exc_info=True)


class SarvamVoiceProvider:
    """The India-resident turn-based path. Always available in the study build."""

    def __init__(self, handler: SupportsAudioTurns, session: VoiceSession):
        self.handler = handler
        self.session = session

    async def serve(self, audio: bytes) -> ProviderOutcome:
        started = time.time()
        try:
            # Labelled as the path it is, so the turn's trace says fallback.
            turn = await self.handler.handle_audio(
                self.session, audio, voice_path=VOICE_PATH_FALLBACK)
        except Exception as exc:  # noqa: BLE001
            return ProviderOutcome(
                turn=None, turn_latency_s=time.time() - started,
                error=f"sarvam_error: {type(exc).__name__}",
            )
        elapsed = time.time() - started
        return ProviderOutcome(
            turn=turn,
            # The whole turn arrives at once on this path, so first-token and
            # turn latency are the same number.
            first_token_latency_s=elapsed,
            turn_latency_s=elapsed,
        )


@dataclass
class Arbitration:
    """What the router decided, recorded with the interaction so it is measurable."""
    voice_path: str
    fallback_reason: Optional[str] = None
    live_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "voice_path": self.voice_path,
            "fallback_reason": self.fallback_reason,
            "live_reason": self.live_reason,
        }


class VoiceRouter:
    """
    Chooses the provider for each turn and reports which one answered.

    The router never decides policy. Policy is in `classify_fallback`, which is
    pure and tested on its own.
    """

    def __init__(
        self,
        handler: SupportsAudioTurns,
        live_service_factory: Optional[Any] = None,
        control: Optional[VoiceProviderControl] = None,
        channel: str = "mobile_voice",
    ):
        self.handler = handler
        self.live_service_factory = live_service_factory
        # The deployment's control, so an administrator's override reaches
        # every router (ADR 0007: "deployment-wide").
        self.control = control or get_voice_provider_control()
        self.channel = channel
        self.stats: Dict[str, int] = {}

    def _count(self, key: str) -> None:
        self.stats[key] = self.stats.get(key, 0) + 1
        if key in _TURN_OUTCOMES:
            record_voice_path(self.channel, key)
        elif key.startswith("live_to_fallback:"):
            record_voice_path(self.channel, key.split(":", 1)[1])

    async def route(
        self,
        session: VoiceSession,
        audio: bytes,
        live_enabled: Optional[bool] = None,
    ) -> ProviderOutcome:
        """
        Answer one turn.

        `live_enabled` defaults to whatever the administrator has selected, so an
        override applies to every session without a restart. Passing it explicitly
        exists for tests only; the admin dashboard goes through VoiceProviderControl.
        """
        if live_enabled is None:
            live_enabled = self.control.current().live_enabled

        availability = LiveAvailability.probe()
        use_live = live_enabled and availability.available and self.live_service_factory

        if not live_enabled:
            self._count("operator_forced_fallback")
            return await self._fallback(session, audio, "operator_forced_fallback")

        if not self.live_service_factory:
            # The channel does not offer Live at all (mobile, ticket 15): not
            # a degradation, and recorded as such.
            self._count("live_not_offered")
            return await self._fallback(session, audio, "live_not_offered")

        if not use_live:
            self._count("live_unavailable")
            self._count(f"live_to_fallback:{availability.reason or 'live_unavailable'}")
            return await self._fallback(session, audio, availability.reason or "live_unavailable")

        live = LiveVoiceProvider(self.live_service_factory(session), session)
        outcome = await live.serve(audio)

        # A stall the provider cut short names itself; otherwise the timings decide.
        reason = outcome.error or classify_fallback(
            outcome.first_token_latency_s, outcome.turn_latency_s)
        if outcome.ok and reason is None:
            self._count("live")
            outcome.turn.voice_path = VOICE_PATH_LIVE
            return outcome

        # Unhealthy rather than broken: connected, then went quiet, or produced
        # nothing. Fall back so the user gets an answer instead of silence.
        self._count(f"live_to_fallback:{reason or 'live_unhealthy'}")
        return await self._fallback(session, audio, reason or "live_unhealthy")

    async def _fallback(
        self, session: VoiceSession, audio: bytes, reason: str
    ) -> ProviderOutcome:
        outcome = await SarvamVoiceProvider(self.handler, session).serve(audio)
        if outcome.turn is not None:
            outcome.turn.voice_path = VOICE_PATH_FALLBACK
            outcome.turn.fallback_reason = reason
        self._count("fallback")
        return outcome

    def report(self) -> Dict[str, Any]:
        """
        Fallback rate, per ADR 0007, for the turns this router answered.

        This number is the evidence that the primary path works. A high fallback
        rate says it does not, and is worth more than any single successful call.
        """
        reasons = {
            k.split(":", 1)[1]: v
            for k, v in self.stats.items()
            if k.startswith("live_to_fallback:")
        }
        return _report(
            live=self.stats.get("live", 0),
            reasons=reasons,
            forced=self.stats.get("operator_forced_fallback", 0),
            not_offered=self.stats.get("live_not_offered", 0),
            control=self.control,
        )


# Turn outcomes that are not Live failing: Live answered, the administrator
# forced Sarvam, or the channel never offers Live. Any other outcome recorded
# is the reason Live fell back on its own.
_TURN_OUTCOMES = (VOICE_PATH_LIVE, "operator_forced_fallback", "live_not_offered")

# Every voice channel's turns, process-wide: {channel: {outcome: count}}.
_DEPLOYMENT_STATS: Dict[str, Dict[str, int]] = {}


def record_voice_path(channel: str, outcome: str) -> None:
    """
    Count one voice turn for the deployment's report.

    `outcome` is "live", "operator_forced_fallback", "live_not_offered", or
    the reason Live fell back on its own ("first_token_timeout",
    "turn_stalled", "live_error: ..."). Web voice, WhatsApp voice notes and
    the mobile routes all record here, so the dashboard's rate covers the
    whole deployment and not one connection.
    """
    by_channel = _DEPLOYMENT_STATS.setdefault(channel, {})
    by_channel[outcome] = by_channel.get(outcome, 0) + 1


def voice_path_report(control: Optional[VoiceProviderControl] = None) -> Dict[str, Any]:
    """The deployment-wide report, across every voice channel (see VoiceRouter.report)."""
    live = forced = not_offered = 0
    reasons: Dict[str, int] = {}
    for counts in _DEPLOYMENT_STATS.values():
        for outcome, n in counts.items():
            if outcome == VOICE_PATH_LIVE:
                live += n
            elif outcome == "operator_forced_fallback":
                forced += n
            elif outcome == "live_not_offered":
                not_offered += n
            else:
                reasons[outcome] = reasons.get(outcome, 0) + n
    report = _report(live=live, reasons=reasons, forced=forced, not_offered=not_offered,
                     control=control or get_voice_provider_control())
    report["by_channel"] = {channel: dict(counts) for channel, counts in _DEPLOYMENT_STATS.items()}
    return report


def _report(
    live: int,
    reasons: Dict[str, int],
    forced: int,
    not_offered: int,
    control: VoiceProviderControl,
) -> Dict[str, Any]:
    """
    The fallback rate counts only Live failing on its own.

    Turns an administrator forced onto Sarvam, and turns on a channel that
    never offers Live, say nothing about whether Live works, so both stay out
    of the rate and are reported separately.
    """
    fallback = sum(reasons.values())
    total = live + fallback
    setting = control.current()
    return {
        "live": live,
        "fallback": fallback,
        "fallback_rate": round(fallback / total, 4) if total else None,
        "by_reason": dict(reasons),
        "operator_forced": forced,
        "live_not_offered": not_offered,
        "sarvam_turns": fallback + forced + not_offered,
        "selected_provider": setting.provider,
        "live_selected": setting.live_enabled,
        "live_available": LiveAvailability.probe().available,
    }


_router_instance: Optional[VoiceRouter] = None


def set_voice_router(router: VoiceRouter) -> None:
    """Register the process-wide router so the admin dashboard can read its counters."""
    global _router_instance
    _router_instance = router


def get_voice_router() -> VoiceRouter:
    """
    The router the WebSocket endpoint is actually using.

    A separate function rather than returning an optional, because the dashboard
    asking for usage figures and being handed a broken provider would be worse
    than asking for nothing. Returning None here made the panel report zero
    fallback turns forever, which is the silent-failure shape this whole area has
    been fighting.
    """
    if _router_instance is None:
        raise RuntimeError(
            "No voice router has been registered. The WebSocket endpoint builds "
            "one at startup; if this is raised before any voice traffic, usage "
            "figures are simply not available yet."
        )
    return _router_instance
