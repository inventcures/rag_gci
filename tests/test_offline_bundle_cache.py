"""
The offline bundle is built once and served with an ETag.

Before this, every download ran twenty full answers (each an LLM call) and
wrote a new file that was never cleaned up, and those build-time answers were
logged in the retrieval audit as if participants had asked them.

What these pin down:

  * a second request for the same language makes no answer calls and a client
    holding the ETag gets 304
  * a new release or a changed corpus produces a fresh bundle
  * build answers are "build" rows in the retrieval audit
  * older bundle files for the language are pruned
  * concurrent first requests share one build
  * every cached answer still passes the Dose Boundary
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from auth.middleware import get_jwt_handler, init_auth  # noqa: E402
from mobile_api.dependencies import init_dependencies  # noqa: E402
from mobile_api.router import mobile_router  # noqa: E402
from offline import cache_builder  # noqa: E402
from safety_enhancements import SafetyEnhancementsManager  # noqa: E402
from tests.test_text_path_dose_guard import RealisticRag, assert_no_dose  # noqa: E402


class CountingRag(RealisticRag):
    def __init__(self):
        super().__init__()
        self.document_metadata = {"doc1": {"chunk_count": 10}}


class BundleCache(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="bundles-"))
        for target, value in ((cache_builder, "CACHE_DIR"),):
            p = mock.patch.object(target, value, self.dir)
            p.start()
            self.addCleanup(p.stop)
        cache_builder._BUNDLES.clear()
        self.addCleanup(cache_builder._BUNDLES.clear)
        self.rag = CountingRag()
        init_auth(jwt_secret="test-secret-for-offline-bundle-cache-checks")
        init_dependencies(self.rag, SafetyEnhancementsManager(), None, None)
        app = FastAPI()
        app.include_router(mobile_router)
        self.client = TestClient(app)
        token = get_jwt_handler().create_token_pair("asha-a", "asha", "site-1").access_token
        self.auth = {"Authorization": f"Bearer {token}"}

    def get(self, language="en-IN", etag=None):
        headers = dict(self.auth)
        if etag:
            headers["If-None-Match"] = etag
        return self.client.get(f"/api/mobile/v1/cache/bundle?language={language}", headers=headers)

    def test_second_request_is_cached_and_304s_with_the_etag(self):
        first = self.get()
        self.assertEqual(first.status_code, 200, first.text)
        calls = len(self.rag.calls)
        self.assertGreater(calls, 0)
        etag = first.headers["etag"]

        second = self.get()
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.headers["etag"], etag)
        self.assertEqual(len(self.rag.calls), calls, "the second request re-ran answers")

        not_modified = self.get(etag=etag)
        self.assertEqual(not_modified.status_code, 304)
        self.assertEqual(len(self.rag.calls), calls)

    def test_a_new_release_or_corpus_rebuilds(self):
        etag = self.get().headers["etag"]
        calls = len(self.rag.calls)

        self.rag.document_metadata["doc2"] = {"chunk_count": 3}
        corpus_etag = self.get().headers["etag"]
        self.assertNotEqual(corpus_etag, etag)
        self.assertGreater(len(self.rag.calls), calls)

        calls = len(self.rag.calls)
        release = mock.MagicMock(release_id="evah-pilot-2@abcdef0")
        with mock.patch("study_release.get_study_release", return_value=release):
            release_etag = self.get().headers["etag"]
        self.assertNotIn(release_etag, (etag, corpus_etag))
        self.assertGreater(len(self.rag.calls), calls)

    def test_old_files_are_pruned_and_a_restart_reuses_the_file(self):
        self.get()
        self.rag.document_metadata["doc2"] = {"chunk_count": 3}
        self.get()
        files = list(self.dir.glob("bundle_en-IN_*.json"))
        self.assertEqual(len(files), 1)

        cache_builder._BUNDLES.clear()  # a restart
        calls = len(self.rag.calls)
        self.assertEqual(self.get().status_code, 200)
        self.assertEqual(len(self.rag.calls), calls, "a restart rebuilt a bundle that was on disk")

    def test_build_answers_are_marked_in_the_audit(self):
        self.get()
        self.assertTrue(all(call["user_id"] == "system:offline_bundle" for call in self.rag.calls))
        from participants import is_person

        self.assertFalse(is_person("system:offline_bundle"))
        from retrieval.audit import record_retrieval

        with mock.patch.dict(os.environ, {"RETRIEVAL_AUDIT_DIR": str(self.dir / "audit")}):
            record_retrieval(user_id="system:offline_bundle", retrieval={})
            record_retrieval(user_id="+919800000001", retrieval={})
            rows = [json.loads(line) for f in (self.dir / "audit").glob("*.jsonl") for line in f.open()]
        self.assertEqual([r["kind"] for r in rows], ["build", "interaction"])

    def test_concurrent_first_requests_share_one_build(self):
        builder = cache_builder.CacheBundleBuilder(rag_pipeline=self.rag, safety_manager=SafetyEnhancementsManager())

        async def go():
            return await asyncio.gather(*(builder.get_bundle("hi-IN") for _ in range(5)))

        results = asyncio.run(go())
        self.assertEqual(len({key for _, key in results}), 1)
        questions = len(cache_builder.questions_for("hi-IN"))
        self.assertEqual(len(self.rag.calls), questions)

    def test_every_cached_answer_is_guarded(self):
        body = self.get().json()
        self.assertTrue(body["queries"])
        for entry in body["queries"]:
            assert_no_dose(self, entry["response_text"], "the offline bundle")


if __name__ == "__main__":
    unittest.main()
