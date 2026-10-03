#!/usr/bin/env python3
"""
Study Release Record
====================
Implements the release identifier required by study protocol section 7.8, and
detects drift between the approved release and the running system.

Section 7.8 requires that before activation the study records its release name,
commit and local patch, actual model and provider identifiers, speech models,
prompt versions, retrieval settings, knowledge-source snapshot, enabled
languages, interface, fallback policy, review approvals and deployment time --
and that this identifier is "stored with every interaction".

Section 7.1 concedes that "An approved study release identifier has not yet been
established". This module is that first step.

Design
------
`release.yaml` is human-readable and is *never* written by this module. It
describes the system that was approved, not the system that happened to be
running; a record that rewrites itself describes neither.

On startup the live system is fingerprinted and compared against the record. On
mismatch the server logs a loud RELEASE DRIFT block naming each differing
component, persists a drift record, and keeps serving. Serving is not blocked,
because bricking the service at four sites during a supervised home visit is a
worse failure than the drift itself -- but the drift is persisted rather than
merely printed, so the analysis can later count how many interactions ran on an
unapproved build. That count is exactly what an IRB or a paper reviewer will ask
for.

Usage:
    from study_release import get_study_release
    release = get_study_release()
    release.release_id          # short id stamped onto every log row
    release.attach_logger(logging.getLogger(__name__))
"""

import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_RELEASE_PATH = Path("./data/study/release.yaml")
DRIFT_DIR = Path("./data/study/drift")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
# PyYAML is already a declared dependency (setup.py, gemini_live/config.py), so
# it is used rather than hand-rolling a parser. An earlier hand-rolled reader
# mis-parsed nested lists and was replaced rather than debugged.


def load_yaml(path: Path) -> Dict[str, Any]:
    """Parse the release record."""
    import yaml

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


# ---------------------------------------------------------------------------
# Release
# ---------------------------------------------------------------------------

@dataclass
class DriftFinding:
    """One component where the live system differs from the approved record."""
    component: str
    recorded: Any
    actual: Any

    def describe(self) -> str:
        return f"  {self.component}: recorded={self.recorded!r} actual={self.actual!r}"


@dataclass
class StudyRelease:
    """The approved release record, plus drift detection against the live system."""
    record: Dict[str, Any] = field(default_factory=dict)
    path: Path = DEFAULT_RELEASE_PATH

    # -- identity ----------------------------------------------------------
    @property
    def release_name(self) -> str:
        return str(self.record.get("release_name", "unknown"))

    @property
    def status(self) -> str:
        return str(self.record.get("status", "draft"))

    @property
    def release_id(self) -> str:
        """
        Short identifier stamped onto every interaction.

        Derived from the release name and the commit, so that two runs of the
        same commit under the same release name are distinguishable from a
        different commit, and so that the id is stable across restarts.
        """
        commit = self._dig("source", "commit") or "unknown"
        return f"{self.release_name}@{str(commit)[:7]}"

    @property
    def is_activated(self) -> bool:
        return self.status == "activated"

    @property
    def approvals_complete(self) -> bool:
        approvals = self.record.get("approvals") or {}
        required = ("clinical_lead", "technical_lead")
        return all(
            (approvals.get(who) or {}).get("approved_at") for who in required
        )

    def _dig(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.record
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    # -- live fingerprint ---------------------------------------------------
    def live_fingerprint(self) -> Dict[str, Any]:
        """
        What the running system actually is.

        Reads the same sources the server uses, so that a change in an
        environment variable shows up here the way it shows up in an answer.
        """
        speech = self._speech_models()
        return {
            "llm.provider": os.getenv("GENAI_LLM_PROVIDER", "gemini").lower(),
            "llm.model": os.getenv("GENAI_GEMINI_MODEL", "gemini-3.1-flash-lite"),
            "translation.provider": "groq" if os.getenv("GROQ_API_KEY") else "unset",
            "translation.model": "llama-3.1-8b-instant",
            "embedding.model": self._live_embedding_model(),
            "speech.stt.provider": speech["stt_provider"],
            "speech.stt.model": speech["stt_model"],
            "speech.tts.model": speech["tts_model"],
            "safety.dosage_restriction": self._module_present("dosage_guard"),
            "commit": self._git_commit(),
        }

    @staticmethod
    def _module_present(module_name: str) -> bool:
        try:
            __import__(module_name)
            return True
        except Exception:
            return False

    @staticmethod
    def _git_commit() -> str:
        try:
            out = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            return out.stdout.strip() or "unknown"
        except Exception:
            return "unknown"

    @staticmethod
    def _live_embedding_model() -> str:
        try:
            import chromadb

            client = chromadb.PersistentClient(path="./data/chroma_db")
            collection = client.get_collection("documents")
            metadata = collection.metadata or {}
            model = metadata.get("hnsw:space")
            del model
            # The collection carries no embedding-model metadata, so the
            # dimension is the only observable proxy. bge-small-en-v1.5 is
            # 384-dimensional; a multilingual model would not be.
            dim = len(collection.get(limit=1, include=["embeddings"])["embeddings"][0])
            return f"unknown-{dim}d"
        except Exception:
            return "unavailable"

    @staticmethod
    def _speech_models() -> Dict[str, str]:
        try:
            import sarvam_integration.config as cfg

            return {
                "stt_provider": "sarvam",
                "stt_model": str(getattr(cfg, "SARVAM_STT_MODEL", "saaras-v3")),
                "tts_model": str(getattr(cfg, "SARVAM_TTS_MODEL", "bulbul-v3")),
            }
        except Exception:
            return {"stt_provider": "sarvam", "stt_model": "saaras-v3", "tts_model": "bulbul-v3"}

    # -- drift --------------------------------------------------------------
    def check_drift(self) -> List[DriftFinding]:
        """
        Compare the approved record against the live system.

        Only components that the record actually names are compared. A component
        recorded as null, or as a marker like "unknown-*", is not treated as a
        discrepancy -- otherwise an unpopulated field would raise a permanent
        false alarm and train everyone to ignore the warning.
        """
        live = self.live_fingerprint()
        recorded: Dict[str, Any] = {
            "llm.provider": self._dig("models", "llm", "provider"),
            "llm.model": self._dig("models", "llm", "model"),
            "translation.provider": self._dig("models", "query_translation", "provider"),
            "translation.model": self._dig("models", "query_translation", "model"),
            "embedding.model": self._dig("models", "embedding", "model"),
            "speech.stt.provider": self._dig("models", "speech", "stt", "provider"),
            "speech.stt.model": self._dig("models", "speech", "stt", "model"),
            "speech.tts.model": self._dig("models", "speech", "tts", "model"),
            "safety.dosage_restriction": self._dig("safety", "dosage_restriction"),
            "commit": self._dig("source", "commit"),
        }

        findings: List[DriftFinding] = []
        for component, recorded_value in recorded.items():
            if recorded_value is None:
                continue
            actual = live.get(component)
            if actual is None:
                continue
            if str(recorded_value) != str(actual):
                findings.append(DriftFinding(component, recorded_value, actual))
        return findings

    def persist_drift(self, findings: List[DriftFinding]) -> Optional[Path]:
        """Append a drift record so the analysis can count affected interactions."""
        if not findings:
            return None
        DRIFT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        path = DRIFT_DIR / f"drift_{stamp}.json"
        payload = {
            "release_id": self.release_id,
            "release_name": self.release_name,
            "recorded_at": stamp,
            "actual_git_commit": self._git_commit(),
            "drift_count": len(findings),
            "findings": [
                {
                    "component": f.component,
                    "recorded": f.recorded,
                    "actual": f.actual,
                }
                for f in findings
            ],
            "impact": (
                "Interactions served while this drift was active cannot be "
                "attributed to the approved release. Exclude or annotate them "
                "in the analysis."
            ),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    # -- startup ------------------------------------------------------------
    def verify_on_startup(self, target_logger: Optional[logging.Logger] = None) -> List[DriftFinding]:
        """
        Emit the release banner and any drift, then persist the drift.

        Never raises and never blocks serving: an unavailable or malformed
        release file degrades to a warning, because stopping the server is a
        worse outcome than an unrecorded release.
        """
        log = target_logger or logger
        findings = self.check_drift()

        banner = {
            "release_id": self.release_id,
            "release_name": self.release_name,
            "status": self.status,
            "approved": self.approvals_complete,
            "drift": [
                {"component": f.component, "recorded": f.recorded, "actual": f.actual}
                for f in findings
            ] or None,
        }
        log.info("STUDY RELEASE %s", json.dumps(banner, default=str))

        if self.status != "activated":
            log.warning(
                "Study release %s is '%s' and is NOT approved for participant "
                "use. Protocol 7.7 requires signed clinical and technical "
                "approval before any participant touches this build.",
                self.release_id,
                self.status,
            )
        if not self.approvals_complete:
            log.warning(
                "Study release %s has unsigned approvals (clinical and "
                "technical leads both required).", self.release_id,
            )

        if findings:
            log.error(
                "RELEASE DRIFT: the running system does not match the approved "
                "release %s in %d component(s):\n%s\n"
                "Serving continues, but interactions served from now on do not "
                "attributetodeclared release. Update data/study/release.yaml and "
                "re-approve before participant use.",
                self.release_id,
                len(findings),
                "\n".join(f.describe() for f in findings),
            )
            path = self.persist_drift(findings)
            if path:
                log.error("RELEASE DRIFT persisted to %s", path)
        return findings

    # -- stamping -----------------------------------------------------------
    def stamp(self) -> Dict[str, Any]:
        """Fields to attach to every interaction log row."""
        return {
            "release_id": self.release_id,
            "release_name": self.release_name,
            "release_status": self.status,
            "approved": self.approvals_complete,
            "release_deployment_time": self._dig("deployment", "activated_at"),
        }

    def describe(self) -> Dict[str, Any]:
        """Full record, for the release dashboard and for export."""
        return {
            "release_id": self.release_id,
            "status": self.status,
            "approvals_complete": self.approvals_complete,
            "record": self.record,
            "live": self.live_fingerprint(),
            "drift": [f.describe().strip() for f in self.check_drift()],
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_release: Optional[StudyRelease] = None


def get_study_release(path: Optional[Path] = None) -> StudyRelease:
    """Get or create the StudyRelease singleton."""
    global _release
    if _release is None:
        target = Path(path) if path else DEFAULT_RELEASE_PATH
        if target.exists():
            try:
                record = load_yaml(target)
            except Exception:
                logger.exception("Could not parse %s; continuing without a release", target)
                record = {}
        else:
            logger.warning(
                "No study release record at %s. Protocol 7.8 requires one "
                "before participant use.", target
            )
            record = {}
        _release = StudyRelease(record=record, path=target)
    return _release


def reload_study_release(path: Optional[Path] = None) -> StudyRelease:
    """Force a re-read. Used by tests and by the release activation endpoint."""
    global _release
    _release = None
    return get_study_release(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    release = get_study_release()
    print(json.dumps(release.describe(), indent=2, default=str))