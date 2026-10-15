"""
The release record tells the truth (review round 3, ticket 10).

Before this:

  * every study row carried the release record's commit (83f7c96), not the
    commit that answered, so a drifted build's rows claimed the approved one;
  * the live fingerprint had no corpus, Live-model or outside-provider
    component, so an upload, a GEMINI_LIVE_MODEL change or turning a provider
    on changed answers with no drift reported;
  * a hybrid start-up whose PageIndex failed left "hybrid" in the release
    record while the vector path answered;
  * a fingerprint component that raised was turned into "unknown" (or, for
    speech, into the approved defaults) without a log line.

These drive the seams a researcher sees: the release-status route, the study
log, the drift findings, and the server's start-up.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import study_release  # noqa: E402
from mobile_api.router import mobile_router  # noqa: E402

RECORDED_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def git_head() -> str:
    """The commit the test process is running, read independently of study_release."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True,
    ).stdout.strip()


def write_manifest(path: Path, docs: dict) -> None:
    path.write_text(json.dumps(docs), encoding="utf-8")


HANDBOOK = {"a1": {"filename": "NPPC handbook.pdf", "chunk_count": 106, "indexed_at": "2025-09-19"}}


class ReleaseFixture(unittest.TestCase):
    """A release record that matches the live system on the three new components."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="release-truth-"))
        self.manifest = self.dir / "document_metadata.json"
        write_manifest(self.manifest, HANDBOOK)
        drift_dir = mock.patch.object(study_release, "DRIFT_DIR", self.dir / "drift")
        drift_dir.start()
        self.addCleanup(drift_dir.stop)
        self.addCleanup(study_release.reload_study_release)

        probe = study_release.StudyRelease(record={}, corpus_manifest=self.manifest)
        live = probe.live_fingerprint()
        self.live = live
        self.record = {
            "release_name": "evah-pilot-1",
            "status": "activated",
            "source": {"commit": RECORDED_COMMIT},
            "knowledge": {"snapshot_sha256": live["corpus.snapshot"]},
            "models": {"voice_primary": {"model": live["voice.live.model"]}},
            "providers": {"outside_india": [p for p in live["providers.outside_india"].split(",") if p != "none"]},
            "approvals": {
                "clinical_lead": {"name": "x", "approved_at": "2026-10-01"},
                "technical_lead": {"name": "y", "approved_at": "2026-10-01"},
            },
        }

    def load(self, record=None) -> study_release.StudyRelease:
        path = self.dir / "release.yaml"
        path.write_text(yaml.safe_dump(record or self.record), encoding="utf-8")
        release = study_release.reload_study_release(path)
        release.corpus_manifest = self.manifest
        return release

    def status(self) -> dict:
        app = FastAPI()
        app.include_router(mobile_router)
        response = TestClient(app).get("/api/mobile/v1/release/status")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def drifting(self, release: study_release.StudyRelease) -> set:
        return {f.component for f in release.refresh_drift()}


class ReleaseIdIsTheRunningCommit(ReleaseFixture):
    def test_release_status_names_the_running_commit_not_the_record(self):
        self.load()
        body = self.status()
        self.assertEqual(body["release_id"], f"evah-pilot-1@{git_head()[:7]}")
        self.assertNotIn(RECORDED_COMMIT[:7], body["release_id"])

    def test_a_record_commit_that_differs_is_reported_as_drift(self):
        self.load()
        drift = {d["component"]: d for d in self.status()["drift"]}
        self.assertIn("commit", drift)
        self.assertEqual(drift["commit"]["recorded"], RECORDED_COMMIT)
        self.assertEqual(drift["commit"]["actual"], git_head())

    def test_a_study_row_carries_the_running_commit(self):
        import study_logging

        self.load()
        logs = self.dir / "logs"
        study_logger = study_logging.StudyInteractionLogger(
            log_dir=logs, use_analytics=False, use_expert_sampling=False,
        )
        asyncio.run(study_logger.record(user_id="p1", query="pain at night", response="ok", channel="web"))
        rows = study_logging.read_entries(log_dir=logs)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["release_id"], f"evah-pilot-1@{git_head()[:7]}")


class DriftCoversCorpusLiveModelAndProviders(ReleaseFixture):
    def test_matching_record_has_no_drift_on_the_new_components(self):
        release = self.load()
        self.assertFalse(
            self.drifting(release) & set(study_release.REQUIRED_COMPONENTS),
            release.refresh_drift(),
        )

    def test_adding_a_document_flips_the_drift_status(self):
        release = self.load()
        self.assertNotIn("corpus.snapshot", self.drifting(release))
        write_manifest(self.manifest, {**HANDBOOK, "b2": {"filename": "new.pdf", "chunk_count": 4}})
        self.assertIn("corpus.snapshot", self.drifting(release))

    def test_an_upload_through_the_pipeline_flips_the_release_status(self):
        """The real add_documents, with the vector store and parser faked."""
        from simple_rag_server import SimpleRAGPipeline

        self.load()
        before = self.status()
        self.assertNotIn("corpus.snapshot", {d["component"] for d in before["drift"]})

        pipeline = SimpleRAGPipeline.__new__(SimpleRAGPipeline)
        pipeline.vector_db = FakeCollection()
        pipeline.document_processor = FakeProcessor()
        pipeline.document_metadata = json.loads(self.manifest.read_text(encoding="utf-8"))
        pipeline.documents_dir = self.dir / "documents"
        pipeline.documents_dir.mkdir()
        pipeline.metadata_file = self.manifest
        source = self.dir / "guide.pdf"
        source.write_bytes(b"%PDF fake")
        result = asyncio.run(pipeline.add_documents([str(source)]))
        self.assertEqual(result["results"][0]["status"], "success", result)

        after = self.status()
        self.assertTrue(after["drifting"])
        self.assertIn("corpus.snapshot", {d["component"] for d in after["drift"]})

    def test_changing_the_live_model_flips_the_drift_status(self):
        release = self.load()
        self.assertNotIn("voice.live.model", self.drifting(release))
        with mock.patch.dict(os.environ, {"GEMINI_LIVE_MODEL": "gemini-3.5-live-translate-preview"}):
            drift = {f.component: f for f in release.refresh_drift()}
        self.assertIn("voice.live.model", drift)
        self.assertEqual(drift["voice.live.model"].actual, "gemini-3.5-live-translate-preview")

    def test_enabling_an_outside_provider_flips_the_drift_status(self):
        release = self.load()
        self.assertNotIn("providers.outside_india", self.drifting(release))
        study_release.register_outside_provider("deepgram", lambda: True)
        self.addCleanup(study_release.unregister_outside_provider, "deepgram")
        drift = {f.component: f for f in release.refresh_drift()}
        self.assertIn("providers.outside_india", drift)
        self.assertIn("deepgram", drift["providers.outside_india"].actual)

    def test_turning_gemini_live_on_by_flag_shows_in_the_provider_list(self):
        with mock.patch.dict(os.environ, {"GEMINI_LIVE_ENABLED": "true"}):
            live = study_release.StudyRelease(record={}, corpus_manifest=self.manifest).live_fingerprint()
        self.assertIn("gemini_live", live["providers.outside_india"].split(","))

    def test_a_record_that_names_none_of_them_drifts_as_unrecorded(self):
        record = {k: v for k, v in self.record.items() if k not in ("knowledge", "models", "providers")}
        drift = {f.component: f for f in self.load(record).refresh_drift()}
        for component in study_release.REQUIRED_COMPONENTS:
            self.assertIn(component, drift)
            self.assertEqual(drift[component].recorded, study_release.UNRECORDED)


class FailedPageIndexStartRecordsVector(unittest.TestCase):
    def start(self, method: str) -> dict:
        import simple_rag_server as server
        from retrieval.settings import RetrievalSettings, get_active_settings, set_active_settings

        settings = RetrievalSettings(rag_method=method)
        set_active_settings(settings)
        self.addCleanup(set_active_settings, RetrievalSettings())
        for name in ("_rag_method", "_pageindex_enabled", "_pageindex_engine"):
            patcher = mock.patch.object(server, name, getattr(server, name))
            patcher.start()
            self.addCleanup(patcher.stop)
        with mock.patch.object(server, "PAGEINDEX_AVAILABLE", True), \
                mock.patch.object(server, "PageIndexConfig", side_effect=RuntimeError("no tree store"), create=True):
            server.start_pageindex(settings)
        self.assertIs(get_active_settings(), settings)
        return study_release.StudyRelease(record={}).live_fingerprint()

    def test_hybrid_with_pageindex_failing_records_vector(self):
        self.assertEqual(self.start("hybrid")["retrieval.rag_method"], "vector")

    def test_pageindex_with_pageindex_failing_records_vector(self):
        self.assertEqual(self.start("pageindex")["retrieval.rag_method"], "vector")


class NoComponentSilentlyBecomesUnknown(ReleaseFixture):
    def test_every_component_failure_is_logged_and_named(self):
        """Break every source the fingerprint reads; each must log and say error."""
        broken = RuntimeError("source unreadable")
        self.manifest.write_text("{not json", encoding="utf-8")
        study_release.register_outside_provider("broken_provider", mock.Mock(side_effect=broken))
        self.addCleanup(study_release.unregister_outside_provider, "broken_provider")
        with mock.patch("llm.client.load_llm_settings", side_effect=broken), \
                mock.patch("retrieval.settings.get_active_settings", side_effect=broken), \
                mock.patch("retrieval.embedding_runtime.configured_runtime", side_effect=broken), \
                mock.patch("chromadb.PersistentClient", side_effect=broken), \
                mock.patch("gemini_live.config.GeminiLiveConfig.from_yaml", side_effect=broken), \
                mock.patch.dict(sys.modules, {"sarvam_integration.config": None}), \
                self.assertLogs(study_release.logger, level=logging.ERROR) as logs:
            live = study_release.StudyRelease(record={}, corpus_manifest=self.manifest).live_fingerprint()

        for component, value in live.items():
            self.assertNotEqual(value, "unknown", component)
            self.assertNotEqual(value, "unavailable", component)
        self.assertTrue(str(live["llm.model"]).startswith("error:"))
        self.assertTrue(str(live["retrieval.rag_method"]).startswith("error:"))
        self.assertTrue(str(live["embedding.runtime"]).startswith("error:"))
        self.assertTrue(str(live["embedding.model"]).startswith("error:"))
        self.assertTrue(str(live["speech.stt.model"]).startswith("error:"))
        self.assertTrue(str(live["corpus.snapshot"]).startswith("error:"))
        self.assertTrue(str(live["voice.live.model"]).startswith("error:"))
        self.assertIn("broken_provider=error:RuntimeError", live["providers.outside_india"])
        self.assertGreaterEqual(len(logs.records), 8)

    def test_an_unreadable_component_is_drift_not_a_match(self):
        release = self.load()
        with mock.patch("gemini_live.config.GeminiLiveConfig.from_yaml", side_effect=RuntimeError("x")), \
                self.assertLogs(study_release.logger, level=logging.ERROR):
            self.assertIn("voice.live.model", self.drifting(release))


class FakeCollection:
    """The slice of a Chroma collection add_documents uses."""

    def __init__(self):
        self.store = {}

    def add(self, documents, metadatas, ids):
        for i, d, m in zip(ids, documents, metadatas):
            self.store[i] = (d, m)

    def get(self, ids=None, **kw):
        return {"ids": [i for i in ids if i in self.store]}


class FakeProcessor:
    def process_file(self, path):
        return {"status": "success", "chunks": ["Keep the patient comfortable."], "page_count": 1}


if __name__ == "__main__":
    unittest.main()
