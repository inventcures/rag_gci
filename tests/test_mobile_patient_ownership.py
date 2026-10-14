"""
Mobile shows only the caller's patients.

Before this, the patient list and sync pull returned every patient file on the
server to every app user, read with three full scans per pull, from a store the
answer path never wrote to. GET /patient/{id} called a method that did not
exist, and POST /observation called add_observation with the wrong arguments.

What these pin down, over HTTP with two app users:

  * sync pull and the patient list for user A never contain user B's patients
  * per-patient endpoints answer 404 for another care team's patient
  * an observation for a new patient makes it the writer's; one for another
    care team's patient is rejected
  * sync pull skips files untouched since the last sync without parsing them
  * the server hands the app the answer path's longitudinal store
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from auth.middleware import get_jwt_handler, init_auth  # noqa: E402
from mobile_api.dependencies import init_dependencies  # noqa: E402
from mobile_api.router import mobile_router  # noqa: E402
from personalization.longitudinal_memory import LongitudinalMemoryManager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = "/api/mobile/v1"


def observation(patient_id, entity="breathlessness", obs_id=None):
    return {"observation_id": obs_id or f"{patient_id}-{entity}", "patient_id": patient_id,
            "category": "symptom", "entity_name": entity, "severity": 3,
            "timestamp": time.time()}


class Ownership(unittest.TestCase):
    def setUp(self):
        self.store = tempfile.mkdtemp(prefix="longitudinal-")
        self.memory = LongitudinalMemoryManager(storage_path=self.store)
        init_auth(jwt_secret="test-secret-for-patient-ownership-checks")
        init_dependencies(None, None, self.memory, None)
        import mobile_api.router as router_module

        # Sync pull also assigns study vignettes, which writes under data/.
        vignettes = mock.patch.object(router_module._vignette_manager, "assign_vignettes", return_value=[])
        vignettes.start()
        self.addCleanup(vignettes.stop)
        app = FastAPI()
        app.include_router(mobile_router)
        self.client = TestClient(app)
        handler = get_jwt_handler()
        self.headers = {
            user: {"Authorization": f"Bearer {handler.create_token_pair(user, 'asha', 'site-1').access_token}"}
            for user in ("asha-a", "asha-b")
        }

    def push(self, user, *observations):
        response = self.client.post(f"{BASE}/sync/push", headers=self.headers[user],
                                    json={"observations": list(observations), "device_timestamp": time.time()})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def pull(self, user, since=0.0):
        response = self.client.post(f"{BASE}/sync/pull?last_sync_at={since}", headers=self.headers[user])
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def patients(self, user):
        return {p["patient_id"] for p in self.client.get(f"{BASE}/patients", headers=self.headers[user]).json()["patients"]}

    def test_each_user_sees_only_their_patients(self):
        self.push("asha-a", observation("pt-a1"), observation("pt-a2"))
        self.push("asha-b", observation("pt-b1"))

        self.assertEqual(self.patients("asha-a"), {"pt-a1", "pt-a2"})
        self.assertEqual(self.patients("asha-b"), {"pt-b1"})

        pull_a = self.pull("asha-a")
        self.assertEqual({p["patient_id"] for p in pull_a["patients"]}, {"pt-a1", "pt-a2"})
        self.assertEqual({o["observation_id"] for o in pull_a["observations"]},
                         {"pt-a1-breathlessness", "pt-a2-breathlessness"})
        self.assertTrue(all(m["provider_id"] == "asha-a" for m in pull_a["care_team_members"]))
        pull_b = self.pull("asha-b")
        self.assertNotIn("pt-a1", str(pull_b))

    def test_a_push_to_another_care_teams_patient_is_rejected(self):
        self.push("asha-a", observation("pt-a1"))
        result = self.push("asha-b", observation("pt-a1", entity="pain", obs_id="sneaky"))
        self.assertEqual((result["accepted"], result["rejected"]), (0, 1))
        record = self.client.get(f"{BASE}/patient/pt-a1", headers=self.headers["asha-a"]).json()
        self.assertNotIn("sneaky", str(record["observations"]))

    def test_per_patient_endpoints_hide_other_care_teams(self):
        self.push("asha-a", observation("pt-a1"))
        for path in ("/patient/pt-a1", "/careteam/pt-a1", "/medication/adherence/pt-a1",
                     "/fhir/export/pt-a1", "/patient/pt-a1/insights"):
            self.assertEqual(self.client.get(BASE + path, headers=self.headers["asha-b"]).status_code, 404, path)
        mine = self.client.get(f"{BASE}/patient/pt-a1", headers=self.headers["asha-a"])
        self.assertEqual(mine.status_code, 200, mine.text)
        self.assertEqual(mine.json()["patient"]["patient_id"], "pt-a1")
        self.assertEqual(self.client.get(f"{BASE}/patient/nobody", headers=self.headers["asha-a"]).status_code, 404)

    def test_create_observation_claims_a_new_patient(self):
        body = {"patient_id": "pt-new", "category": "symptom", "entity_name": "pain", "severity": 2}
        response = self.client.post(f"{BASE}/observation", headers=self.headers["asha-a"], json=body)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.patients("asha-a"), {"pt-new"})
        other = self.client.post(f"{BASE}/observation", headers=self.headers["asha-b"], json=body)
        self.assertEqual(other.status_code, 404)

    def test_unowned_records_are_visible_to_nobody(self):
        import asyncio

        async def admin_record():
            record = await self.memory.get_or_create_record("pt-admin")
            await self.memory.save_record(record)

        asyncio.run(admin_record())
        self.assertEqual(self.patients("asha-a"), set())
        self.assertEqual(self.pull("asha-a")["patients"], [])

    def test_sync_pull_does_not_parse_unchanged_files(self):
        self.push("asha-a", observation("pt-a1"), observation("pt-a2"))
        old = time.time() - 3600
        for path in Path(self.store).glob("*_longitudinal.json"):
            os.utime(path, (old, old))
        since = time.time() - 60
        with mock.patch.object(LongitudinalMemoryManager, "_read_json", wraps=LongitudinalMemoryManager._read_json) as reads:
            pull = self.pull("asha-a", since=since)
        self.assertEqual(reads.call_count, 0)
        self.assertEqual(pull["patients"], [])

    def test_patient_list_reparses_only_changed_files(self):
        self.push("asha-a", observation("pt-a1"), observation("pt-a2"))
        self.patients("asha-a")
        with mock.patch.object(LongitudinalMemoryManager, "_read_json", wraps=LongitudinalMemoryManager._read_json) as reads:
            self.patients("asha-a")
        self.assertEqual(reads.call_count, 0)


class OneStore(unittest.TestCase):
    def test_the_app_gets_the_answer_paths_store(self):
        source = (ROOT / "simple_rag_server.py").read_text(encoding="utf-8")
        start = source.index("init_mobile_deps(")
        call = source[start:source.index(")\n", source.index("medication_manager=", start))]
        self.assertIn('getattr(rag_pipeline, "longitudinal_manager", None)', call)
        self.assertNotIn("LongitudinalMemoryManager()", call)


if __name__ == "__main__":
    unittest.main()
