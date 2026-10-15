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

import functools
import hashlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_RELEASE_PATH = Path("./data/study/release.yaml")
DRIFT_DIR = Path("./data/study/drift")
# The server's own record of what is in the vector index
# (SimpleRAGPipeline.metadata_file). Every upload and removal rewrites it.
DEFAULT_CORPUS_MANIFEST = Path("./data/document_metadata.json")

# Components that exist so that a change cannot be silent. For these a missing
# entry in the record is not "not compared": it compares as UNRECORDED, so a
# release that never named its corpus, Live model or outside providers drifts
# until the record names them.
REQUIRED_COMPONENTS = ("corpus.snapshot", "voice.live.model", "providers.outside_india")
UNRECORDED = "unrecorded"


def component_error(component: str, exc: BaseException) -> str:
    """Log a fingerprint component that could not be read, and return the value
    that stands in for it. The value names the failure, so it never matches a
    recorded value and shows as drift instead of passing as "unknown"."""
    logger.error("Release fingerprint: could not read %s: %s", component, exc, exc_info=exc)
    return f"error:{type(exc).__name__}"


# ---------------------------------------------------------------------------
# Running commit
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def running_commit() -> str:
    """
    The commit this process is running, read once per process.

    Read once on purpose: a `git pull` without a restart moves HEAD but not the
    code in memory, so the first reading is the true one. Falls back to the
    platform's build variable (Railway sets RAILWAY_GIT_COMMIT_SHA) where the
    deployment has no .git directory.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            cwd=Path(__file__).resolve().parent,
        )
        commit = out.stdout.strip()
        if commit:
            return commit
        reason = out.stderr.strip() or f"git exited {out.returncode}"
    except (OSError, subprocess.SubprocessError) as exc:
        reason = f"{type(exc).__name__}: {exc}"
    platform = os.getenv("RAILWAY_GIT_COMMIT_SHA", "").strip()
    if platform:
        return platform
    logger.error("Release fingerprint: running commit unreadable (%s); release_id will say so", reason)
    return "unknown"


# ---------------------------------------------------------------------------
# Outside providers
# ---------------------------------------------------------------------------
# A provider that receives participant audio or text outside India registers a
# check here. The fingerprint lists every one whose check says enabled, and the
# release record lists the approved ones under providers.outside_india.

_OUTSIDE_PROVIDERS: Dict[str, Callable[[], bool]] = {}


def register_outside_provider(name: str, is_enabled: Callable[[], bool]) -> None:
    """Register a provider that sends participant audio or text outside India.

    `is_enabled` must read the same setting the server reads when it decides
    to use the provider, so the fingerprint cannot disagree with what runs.
    Registering the same name again replaces the check.
    """
    _OUTSIDE_PROVIDERS[name] = is_enabled


def unregister_outside_provider(name: str) -> None:
    """Remove a registered provider check (tests)."""
    _OUTSIDE_PROVIDERS.pop(name, None)


def enabled_outside_providers() -> List[str]:
    """Names of the outside providers enabled now, sorted.

    A check that raises is logged and listed as `name=error:<Type>`: unknown
    is not the same as off.
    """
    enabled: List[str] = []
    for name, is_enabled in sorted(_OUTSIDE_PROVIDERS.items()):
        try:
            if is_enabled():
                enabled.append(name)
        except Exception as exc:  # a provider check is third-party config code
            enabled.append(f"{name}={component_error('providers.outside_india.' + name, exc)}")
    return enabled


def _gemini_live_enabled() -> bool:
    """Gemini Live (no Indian region; raw voice leaves India, ADR 0007)."""
    from gemini_live.config import GeminiLiveConfig, live_enabled

    return live_enabled(GeminiLiveConfig.from_yaml())


register_outside_provider("gemini_live", _gemini_live_enabled)


def _register_phone_providers() -> None:
    """Bolna's and Retell's outside services (phone_residency, decision D11)."""
    from phone_residency import registry

    for name, is_enabled in registry().items():
        register_outside_provider(name, is_enabled)


_register_phone_providers()


def provider_list(value: Any) -> Optional[str]:
    """A recorded or live provider list as one comparable value."""
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    return ",".join(sorted(str(v) for v in value)) or "none"


# ---------------------------------------------------------------------------
# Corpus snapshot
# ---------------------------------------------------------------------------

def corpus_snapshot(manifest: Path = DEFAULT_CORPUS_MANIFEST) -> str:
    """
    sha256 over what the index holds, read from the server's document manifest.

    Each document contributes its id, filename and chunk count. Upload time is
    left out, so copying the same index to another machine gives the same
    hash; a new upload has a new id, and a re-chunked document a new count. A
    missing manifest is an empty corpus, as it is to the server.
    """
    entries: Dict[str, Any] = {}
    if manifest.exists():
        loaded = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"{manifest} is not a mapping")
        entries = loaded
    canonical = [
        [doc_id, str((meta or {}).get("filename", "")), int((meta or {}).get("chunk_count", 0))]
        for doc_id, meta in sorted(entries.items())
    ]
    digest = hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
# PyYAML is already a declared dependency (requirements.txt, gemini_live/config.py), so
# it is used rather than hand-rolling a parser. An earlier hand-rolled reader
# mis-parsed nested lists and was replaced rather than debugged.


def compact_mapping(mapping: Any) -> Optional[str]:
    """`{"b": 2, "a": "x"}` -> "a=x,b=2", so a recorded mapping and the live
    one compare as one value. Numbers compare by value (7 == 7.0)."""
    if not isinstance(mapping, dict):
        return None
    def value(v: Any) -> str:
        return f"{float(v):g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)
    return ",".join(f"{k}={value(v)}" for k, v in sorted(mapping.items()))


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
    corpus_manifest: Path = DEFAULT_CORPUS_MANIFEST
    # Drift needs git and the vector store, so it is computed once (at startup,
    # or on first use) and reused until refresh_drift() or a reload.
    _drift: Optional[List["DriftFinding"]] = field(default=None, repr=False, compare=False)

    def drift(self) -> List["DriftFinding"]:
        """Drift findings, computed once and cached."""
        if self._drift is None:
            self._drift = self.check_drift()
        return self._drift

    def refresh_drift(self) -> List["DriftFinding"]:
        """Recompute drift (after a deploy or a vector-store rebuild)."""
        self._drift = self.check_drift()
        return self._drift

    def corpus_changed(
        self,
        target_logger: Optional[logging.Logger] = None,
        manifest: Optional[Path] = None,
    ) -> List["DriftFinding"]:
        """
        Recheck drift after the corpus changed (an upload or a removal).

        Without this the cached drift from start-up would keep reporting the
        old corpus, and a new document would change answers with nothing
        noticing. A drift that the change introduces is logged and persisted
        the same way start-up drift is. Runs git and reads the manifest, so
        call it off the event loop.

        `manifest` is the file the pipeline just wrote; it becomes the corpus
        manifest the fingerprint hashes, so the two cannot name different files.
        """
        log = target_logger or logger
        if manifest is not None:
            self.corpus_manifest = Path(manifest)
        findings = self.refresh_drift()
        corpus = [f for f in findings if f.component == "corpus.snapshot"]
        if corpus:
            log.error(
                "RELEASE DRIFT: the corpus no longer matches release %s:\n%s",
                self.release_id, "\n".join(f.describe() for f in corpus),
            )
            self.persist_drift(findings)
        return findings

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

        The release name and the commit that is actually running, so a study
        row names the code that answered it. The record's `source.commit` is
        only compared against the running commit (the `commit` drift
        component); it never stands in for it, or every row of a drifted
        build would claim the approved one.
        """
        return f"{self.release_name}@{running_commit()[:7]}"

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

    @property
    def live_model(self) -> str:
        """The Live model this release approves (`models.voice_primary.model`):
        the one every Live session opens with (gemini_live.config), and the
        recorded side of the `voice.live.model` drift component. "" when the
        record names none, which leaves Live unable to start."""
        return str(self._dig("models", "voice_primary", "model", default="") or "")

    def approved_outside_providers(self) -> List[str]:
        """The outside providers the record lists under providers.outside_india."""
        listed = self._dig("providers", "outside_india")
        if listed is None:
            return []
        if isinstance(listed, str):
            listed = [listed]
        return [str(name) for name in listed]

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
        retrieval = self._retrieval_settings()
        llm = self._llm_settings()
        return {
            "llm.provider": llm["provider"],
            "llm.model": llm["model"],
            "llm.location": llm["location"],
            "llm.fallback": llm["fallback"],
            "llm.thinking": llm["thinking"],
            "llm.deadlines": llm["deadlines"],
            # Translation uses the same shared client as answers.
            "translation.provider": llm["provider"],
            "translation.model": llm["model"],
            "embedding.model": self._live_embedding_model(),
            "embedding.runtime": self._embedding_runtime(),
            "speech.stt.provider": speech["stt_provider"],
            "speech.stt.model": speech["stt_model"],
            "speech.tts.model": speech["tts_model"],
            "safety.dosage_restriction": self._module_present("dosage_guard"),
            "retrieval.rag_method": retrieval["rag_method"],
            "retrieval.kg_check": retrieval["kg_check"],
            "commit": self._git_commit(),
            "corpus.snapshot": self._corpus_snapshot(),
            "voice.live.model": self._live_model(),
            "providers.outside_india": provider_list(enabled_outside_providers()),
        }

    def _corpus_snapshot(self) -> str:
        """The live corpus hash (see corpus_snapshot)."""
        try:
            return corpus_snapshot(self.corpus_manifest)
        except (OSError, ValueError, TypeError) as exc:
            return component_error("corpus.snapshot", exc)

    @staticmethod
    def _live_model() -> str:
        """The Live model sessions open with: the release record's, else GEMINI_LIVE_MODEL."""
        try:
            from gemini_live.config import GeminiLiveConfig

            return GeminiLiveConfig.from_yaml().model
        except Exception as exc:  # import of the Live stack can fail many ways
            return component_error("voice.live.model", exc)

    @staticmethod
    def _llm_settings() -> Dict[str, Any]:
        """The shared LLM client's settings (llm/client.py), as the server loads them."""
        try:
            from llm.client import load_llm_settings

            d = load_llm_settings().describe()
            fb = d.get("fallback")
            return {
                "provider": d["provider"],
                "model": d["model"],
                "location": d["location"],
                "fallback": f"{fb['provider']}:{fb['model']}" if fb else "none",
                "thinking": compact_mapping(d.get("thinking")),
                "deadlines": compact_mapping(d.get("deadlines")),
            }
        except Exception as exc:  # settings loading reads env, yaml and the SDK
            value = component_error("llm", exc)
            return {key: value for key in ("provider", "model", "location", "fallback", "thinking", "deadlines")}

    @staticmethod
    def _retrieval_settings() -> Dict[str, str]:
        """The answer path's live retrieval settings (retrieval/settings.py)."""
        try:
            from retrieval.settings import get_active_settings

            active = get_active_settings()
            return {"rag_method": active.rag_method, "kg_check": active.kg_check_mode}
        except Exception as exc:
            value = component_error("retrieval", exc)
            return {"rag_method": value, "kg_check": value}

    @staticmethod
    def _embedding_runtime() -> str:
        """The embedding runtime the server would load (config.yaml, env)."""
        try:
            import yaml

            from retrieval.embedding_runtime import configured_runtime

            config_path = Path("config.yaml")
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
            return configured_runtime(config or {})
        except Exception as exc:
            return component_error("embedding.runtime", exc)

    @staticmethod
    def _module_present(module_name: str) -> Any:
        """True when the module imports; a failed import is logged and named."""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False
        except Exception as exc:  # the module exists but raised on import
            return component_error(f"module {module_name}", exc)

    @staticmethod
    def _git_commit() -> str:
        """The running commit (see running_commit)."""
        return running_commit()

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
        except Exception as exc:  # chromadb raises its own and sqlite errors
            return component_error("embedding.model", exc)

    @staticmethod
    def _speech_models() -> Dict[str, str]:
        try:
            import sarvam_integration.config as cfg

            return {
                "stt_provider": "sarvam",
                "stt_model": str(getattr(cfg, "SARVAM_STT_MODEL", "saaras-v3")),
                "tts_model": str(getattr(cfg, "SARVAM_TTS_MODEL", "bulbul-v3")),
            }
        except Exception as exc:
            # Not the defaults: claiming the approved models when the
            # configuration could not be read is the silent "unknown" again.
            value = component_error("speech", exc)
            return {"stt_provider": value, "stt_model": value, "tts_model": value}

    # -- drift --------------------------------------------------------------
    def check_drift(self) -> List[DriftFinding]:
        """
        Compare the approved record against the live system.

        Only components that the record actually names are compared. A component
        recorded as null, or as a marker like "unknown-*", is not treated as a
        discrepancy -- otherwise an unpopulated field would raise a permanent
        false alarm and train everyone to ignore the warning. The exception is
        REQUIRED_COMPONENTS (corpus, Live model, outside providers): those
        compare an absent record as "unrecorded", because the point of them is
        that the corpus or a residency choice cannot change unnoticed.
        """
        live = self.live_fingerprint()
        recorded: Dict[str, Any] = {
            "llm.provider": self._dig("models", "llm", "provider"),
            "llm.model": self._dig("models", "llm", "model"),
            "llm.location": self._dig("models", "llm", "region"),
            "llm.fallback": self._dig("models", "llm", "fallback"),
            "llm.thinking": compact_mapping(self._dig("models", "llm", "thinking")),
            "llm.deadlines": compact_mapping(self._dig("models", "llm", "deadlines")),
            "translation.provider": self._dig("models", "query_translation", "provider"),
            "translation.model": self._dig("models", "query_translation", "model"),
            "embedding.model": self._dig("models", "embedding", "model"),
            "embedding.runtime": self._dig("models", "embedding", "runtime"),
            "speech.stt.provider": self._dig("models", "speech", "stt", "provider"),
            "speech.stt.model": self._dig("models", "speech", "stt", "model"),
            "speech.tts.model": self._dig("models", "speech", "tts", "model"),
            "safety.dosage_restriction": self._dig("safety", "dosage_restriction"),
            "retrieval.rag_method": self._dig("retrieval", "rag_method"),
            "retrieval.kg_check": self._dig("retrieval", "kg_check"),
            "commit": self._dig("source", "commit"),
            "corpus.snapshot": self._dig("knowledge", "snapshot_sha256"),
            "voice.live.model": self.live_model or None,
            "providers.outside_india": provider_list(self._dig("providers", "outside_india")),
        }
        for component in REQUIRED_COMPONENTS:
            if recorded[component] is None:
                recorded[component] = UNRECORDED

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
    def client_view(self) -> Dict[str, Any]:
        """
        What the app is allowed to know about this release.

        Exists because the app records ``release_approved`` on every interaction but
        nothing ever told it whether to. It stamped a value it had no way to
        establish, which made the field decorative: a study could look at the
        recorded rows and see approvals either way without being able to tell which
        build produced them.

        Approval and activation are reported separately on purpose. A release can be
        approved by the leads and not yet activated for participants, and the app
        needs both: the first says the record is signed, the second says the study
        has actually opened it.

        Uses the cached drift: this is served per request, and computing drift
        runs git and opens the vector store.
        """
        findings = self.drift()
        return {
            "release_id": self.release_id,
            "release_name": self.release_name,
            "status": self.status,
            "activated": self.is_activated,
            "approved": self.approvals_complete,
            "usable_with_participants": self.is_activated and self.approvals_complete,
            "drifting": bool(findings),
            "drift": [
                {"component": f.component, "recorded": f.recorded, "actual": f.actual}
                for f in findings
            ],
        }

    def verify_on_startup(self, target_logger: Optional[logging.Logger] = None) -> List[DriftFinding]:
        """
        Emit the release banner and any drift, then persist the drift.

        Never raises and never blocks serving: an unavailable or malformed
        release file degrades to a warning, because stopping the server is a
        worse outcome than an unrecorded release.
        """
        log = target_logger or logger
        findings = self.refresh_drift()

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