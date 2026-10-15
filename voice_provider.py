#!/usr/bin/env python3
"""
Voice provider selection
========================
Which provider answers voice: Gemini Live (default) or Sarvam (forced).

ADR 0007 records that the fallback is automatic, and separately that the backend
administrator can force the Sarvam path for the whole deployment during a
provider incident. This module is that switch: persisted, auditable, and readable
from the admin dashboard.

Two rules keep it honest.

**The default is Live.** Forcing Sarvam is an incident response, not a preference,
so every change is recorded with who, when, from, to, and a required reason. An
unaudited switch is how an operational override quietly becomes the permanent
residency position.

**Setting it to Sarvam is not the same as Live being broken.** The two are
reported separately. The fallback counter says how often Live degraded on its own;
the override says how long someone deliberately kept it off. Conflating them would
make a deliberate choice look like a fault, or a fault look like a choice.
"""

import json
import os
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PROVIDER_LIVE = "gemini_live"
PROVIDER_SARVAM = "sarvam"

VALID_PROVIDERS = (PROVIDER_LIVE, PROVIDER_SARVAM)


class AuditWriteError(RuntimeError):
    """The audit record for a provider change could not be written.

    The change has been rolled back: ADR 0007 allows no provider change without
    its attributed record."""

# Repo-relative, so it behaves the same on a laptop and on the deployed instance.
SETTINGS_PATH = Path("data/study/voice_provider.json")
AUDIT_PATH = Path("data/study/voice_provider_audit.jsonl")


@dataclass
class VoiceProviderSetting:
    provider: str = PROVIDER_LIVE
    updated_at: float = 0.0
    updated_by: str = ""
    reason: str = ""
    previous_provider: Optional[str] = None

    @property
    def live_enabled(self) -> bool:
        """True unless an administrator has deliberately forced Sarvam."""
        return self.provider == PROVIDER_LIVE

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["live_enabled"] = self.live_enabled
        return data


class VoiceProviderControl:
    """
    Reads and writes the selection.

    Kept deliberately small and synchronous. It is read on every voice turn and
    written rarely, so a file with an in-memory cache is the right amount of
    machinery; a database would be more moving parts than the decision warrants.
    """

    def __init__(self, path: Path = SETTINGS_PATH, audit_path: Path = AUDIT_PATH):
        self.path = path
        self.audit_path = audit_path
        self._lock = threading.Lock()
        self._cache: Optional[VoiceProviderSetting] = None
        self._cache_stamp: Optional[tuple] = None

    def current(self) -> VoiceProviderSetting:
        """The selection, re-read when the file changes: another server
        process may have switched it (one stat call per voice turn)."""
        with self._lock:
            stamp = self._stamp()
            if self._cache is not None and stamp == self._cache_stamp:
                return self._cache
            self._cache = self._read()
            self._cache_stamp = stamp
            return self._cache

    def _stamp(self) -> Optional[tuple]:
        try:
            st = self.path.stat()
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return None

    def _read(self) -> VoiceProviderSetting:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            # No file means nobody has overridden anything, so Live is on. That is
            # the ADR 0007 default and the safe reading.
            return VoiceProviderSetting(provider=PROVIDER_LIVE)
        except Exception:
            logger.exception("Voice provider setting unreadable; defaulting to Live")
            # An unreadable setting must not strand the service in fallback.
            return VoiceProviderSetting(provider=PROVIDER_LIVE)

        provider = str(raw.get("provider", PROVIDER_LIVE))
        if provider not in VALID_PROVIDERS:
            logger.warning("Unknown voice provider %r; defaulting to Live", provider)
            provider = PROVIDER_LIVE
        return VoiceProviderSetting(
            provider=provider,
            updated_at=float(raw.get("updated_at") or 0.0),
            updated_by=str(raw.get("updated_by") or ""),
            reason=str(raw.get("reason") or ""),
            previous_provider=raw.get("previous_provider"),
        )

    def set_provider(
        self,
        provider: str,
        actor: str,
        reason: str,
    ) -> VoiceProviderSetting:
        """
        Change the provider for the whole deployment.

        `reason` is required. Forcing the India-resident path is a decision
        someone has to be able to explain later, which means at write time rather
        than from memory.

        Raises AuditWriteError, with the previous setting restored, when the
        audit record cannot be written.
        """
        if provider not in VALID_PROVIDERS:
            raise ValueError(f"provider must be one of {VALID_PROVIDERS}, got {provider!r}")
        if not reason or not reason.strip():
            raise ValueError("a reason is required when changing the voice provider")

        with self._lock:
            before = self._read()  # the file, not this process's cache
            if before.provider == provider:
                # A no-op change would put a meaningless entry in the audit trail.
                return before

            setting = VoiceProviderSetting(
                provider=provider,
                updated_at=time.time(),
                updated_by=actor,
                reason=reason.strip(),
                previous_provider=before.provider,
            )
            try:
                previous_text: Optional[str] = self.path.read_text(encoding="utf-8")
            except FileNotFoundError:
                previous_text = None
            self._write_setting_file(json.dumps(setting.to_dict(), indent=2))
            try:
                self._audit(setting)
            except OSError as exc:
                # No change may stand without its record: put the file back.
                self._restore_setting_file(previous_text)
                logger.error(
                    "Voice provider change %s -> %s by %s rolled back: audit write failed (%s)",
                    before.provider, provider, actor, exc,
                )
                raise AuditWriteError("the audit record could not be written; nothing was changed") from exc
            self._cache = setting
            self._cache_stamp = self._stamp()

            logger.warning(
                "Voice provider changed %s -> %s by %s: %s",
                before.provider, provider, actor, reason,
            )
            return setting

    def _write_setting_file(self, text: str) -> None:
        """Write the setting whole then rename, so another process never reads half."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, self.path)

    def _restore_setting_file(self, previous_text: Optional[str]) -> None:
        """Put back the setting file as it was before a change (absent if it was absent)."""
        if previous_text is None:
            self.path.unlink(missing_ok=True)
        else:
            self._write_setting_file(previous_text)

    def _audit(self, setting: VoiceProviderSetting) -> None:
        """Append to the audit trail. Raises OSError when the record cannot be written."""
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(setting.to_dict(), ensure_ascii=False) + "\n")

    def history(self, limit: int = 20) -> List[Dict[str, Any]]:
        try:
            lines = self.audit_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        out: List[Dict[str, Any]] = []
        for line in lines[-limit:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return list(reversed(out))


_control: Optional[VoiceProviderControl] = None


def get_voice_provider_control() -> VoiceProviderControl:
    global _control
    if _control is None:
        _control = VoiceProviderControl()
    return _control
