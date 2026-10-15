"""
The app's participant gate shows the release record on disk.

Before this, GET /release/status built an empty StudyRelease(), so it always
answered "draft" and "not approved" whatever release.yaml said, and it ran git
and opened a Chroma client on every call, unauthenticated.

What these pin down:

  * the endpoint reports the status, name and approvals of the loaded record,
    and the release id of the commit that is running
  * drift is computed once, not per request: no subprocess and no vector-store
    client on a status call after the first
  * refresh_drift() recomputes on demand
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import study_release  # noqa: E402
from mobile_api.router import mobile_router  # noqa: E402

ACTIVATED = """
release_name: "evah-pilot-1"
status: activated
source:
  commit: "0123456789abcdef"
approvals:
  clinical_lead: {name: "x", approved_at: "2026-10-01"}
  technical_lead: {name: "y", approved_at: "2026-10-01"}
"""


class ReleaseStatus(unittest.TestCase):
    def setUp(self):
        directory = Path(tempfile.mkdtemp(prefix="release-"))
        self.path = directory / "release.yaml"
        self.path.write_text(ACTIVATED, encoding="utf-8")
        study_release.reload_study_release(self.path)
        self.addCleanup(study_release.reload_study_release)
        app = FastAPI()
        app.include_router(mobile_router)
        self.client = TestClient(app)

    def status(self):
        response = self.client.get("/api/mobile/v1/release/status")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_reports_the_record_on_disk(self):
        body = self.status()
        self.assertEqual(body["release_name"], "evah-pilot-1")
        self.assertEqual(body["status"], "activated")
        self.assertTrue(body["approved"])
        self.assertTrue(body["activated"])
        # The running commit, not the record's: the record's commit is only
        # compared against it (ticket 10, review round 3).
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        self.assertEqual(body["release_id"], f"evah-pilot-1@{head[:7]}")

    def test_no_subprocess_or_vector_store_per_request(self):
        self.status()  # first use computes drift
        with mock.patch.object(subprocess, "run", side_effect=AssertionError("git per request")), \
                mock.patch("chromadb.PersistentClient", side_effect=AssertionError("chroma per request")):
            for _ in range(3):
                self.status()

    def test_refresh_recomputes(self):
        release = study_release.get_study_release()
        with mock.patch.object(study_release.StudyRelease, "check_drift", return_value=[]) as check:
            release.refresh_drift()
            release.client_view()
            release.client_view()
        self.assertEqual(check.call_count, 1)


if __name__ == "__main__":
    unittest.main()
