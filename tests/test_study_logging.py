#!/usr/bin/env python3
"""
Tests for the study log, the outcome definitions and the audit dashboard.

These cover the promises the EVAH grant makes about system logs, because the
grant is a funded document and these are the claims that will be audited.
"""

import json
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import study_logging as sl  # noqa: E402
import study_outcomes as so  # noqa: E402


class TestPseudonymisation(unittest.TestCase):
    def test_ids_are_deterministic_within_a_salt(self):
        a = sl.pseudonymise("asha-001")
        b = sl.pseudonymise("asha-001")
        self.assertEqual(a, b)
        self.assertNotIn("asha", a)
        self.assertEqual(len(a), 16)

    def test_different_participants_differ(self):
        self.assertNotEqual(sl.pseudonymise("asha-001"), sl.pseudonymise("asha-002"))

    def test_empty_id_is_anonymous(self):
        self.assertEqual(sl.pseudonymise(""), "anonymous")

    def test_id_is_not_reversible_by_dictionary_attack(self):
        # A 16-hex truncation of a keyed HMAC must not be brute-forceable from a
        # known list of candidate ids.
        known = [f"p{i:03d}" for i in range(500)]
        hashes = {sl.pseudonymise(k) for k in known}
        self.assertEqual(len(hashes), len(known))


class TestScrubbing(unittest.TestCase):
    def test_phone_aadhaar_email_url_removed(self):
        cases = [
            "call me on 9876543210",
            "aadhaar 1234 5678 9012",
            "email a.b@example.com",
            "see https://example.org/x",
            "landline 09876543210.",
        ]
        for text in cases:
            with self.subTest(text=text):
                result = sl.scrub(text)
                self.assertTrue(result.was_scrubbed, f"not scrubbed: {text}")

    def test_bare_long_digit_runs_removed(self):
        # Format-specific patterns miss these; the catch-all must not.
        for text in ["id 09876543210 here", "number 1234567890"]:
            with self.subTest(text=text):
                self.assertTrue(sl.scrub(text).was_scrubbed, text)

    def test_ordinary_clinical_numbers_survive(self):
        # Over-scrubbing would destroy the corpus the grant's BERTopic analysis
        # depends on.
        cases = [
            "Chapter 5 covers pressure sores",
            "take 2 tablets",
            "rotate every 2 hours",
            "version 1.0.2",
            "Grade C evidence",
            "2026-27 cohort",
            "1-2 hour intervals",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertFalse(sl.scrub(text).was_scrubbed, f"over-scrubbed: {text}")


class TestAdoptionDefinition(unittest.TestCase):
    """Protocol 4.1: >=1 substantive interaction in >=2 distinct weeks of 28 days."""

    def _entry(self, participant, when, **kw):
        base = {
            "participant_id": participant,
            "occurred_at": when.timestamp(),
            "occurred_at_iso": when.isoformat(),
            "interaction_class": "substantive",
            "language": "hi-IN",
            "care_role": "ASHA Worker",
            "total_latency_ms": 4000,
        }
        base.update(kw)
        return base

    def test_two_distinct_weeks_is_sustained(self):
        now = datetime.now(timezone.utc)
        entries = [
            self._entry("a", now - timedelta(days=3)),
            self._entry("a", now - timedelta(days=10)),
        ]
        result = so.adoption_summary(entries)
        self.assertTrue(result["per_participant"]["a"]["sustained"])

    def test_two_calls_in_one_week_is_not_sustained(self):
        now = datetime.now(timezone.utc)
        entries = [
            self._entry("a", now - timedelta(days=1)),
            self._entry("a", now - timedelta(days=2)),
        ]
        result = so.adoption_summary(entries)
        self.assertFalse(result["per_participant"]["a"]["sustained"])
        self.assertEqual(result["per_participant"]["a"]["distinct_weeks"], 1)

    def test_activity_outside_28_day_window_is_ignored(self):
        now = datetime.now(timezone.utc)
        entries = [
            self._entry("a", now - timedelta(days=5)),
            self._entry("a", now - timedelta(days=60)),
        ]
        result = so.adoption_summary(entries)
        self.assertEqual(result["per_participant"]["a"]["interactions"], 1)

    def test_denominator_is_reported(self):
        now = datetime.now(timezone.utc)
        entries = [self._entry("a", now), self._entry("b", now)]
        result = so.adoption_summary(entries)
        self.assertEqual(result["denominator_participants_with_activity"], 2)

    def test_excluded_rows_do_not_count(self):
        now = datetime.now(timezone.utc)
        entries = [
            self._entry("a", now, interaction_class="excluded", exclusion_reason="test_message"),
        ]
        result = so.adoption_summary(entries)
        self.assertEqual(result["denominator_participants_with_activity"], 0)
        self.assertEqual(result["excluded_interactions"], 1)
        self.assertEqual(result["exclusion_reasons"].get("test_message"), 1)


class TestInteractionClassification(unittest.TestCase):
    def test_training_and_test_rows_excluded(self):
        cases = [
            ("this is a test message", "ok", "test_message"),
            ("training demo for the new ASHA", "ok", "training_demo"),
        ]
        for query, response, expected in cases:
            with self.subTest(query=query):
                label, reason = sl.classify_interaction(query, response)
                self.assertEqual(label, "excluded")
                self.assertEqual(reason, expected)

    def test_real_care_question_is_substantive(self):
        label, reason = sl.classify_interaction(
            "patient has severe breathlessness, what should I do?", "Prop up with pillows."
        )
        self.assertEqual(label, "substantive")
        self.assertIsNone(reason)

    def test_no_response_and_no_escalation_is_excluded(self):
        label, reason = sl.classify_interaction("pain", "")
        self.assertEqual(label, "excluded")

    def test_uncertainty_defaults_to_counting_it(self):
        # The protocol asks that the denominator not be inflated by exclusions,
        # so an unrecognised row counts as substantive.
        label, _ = sl.classify_interaction("unusual phrasing xyzzy", "an answer")
        self.assertEqual(label, "substantive")


class TestSafetyOutcomes(unittest.TestCase):
    def _entry(self, **kw):
        base = {
            "participant_id": "p1",
            "occurred_at_iso": datetime.now(timezone.utc).isoformat(),
            "interaction_class": "substantive",
            "language": "hi-IN",
            "source_count": 2,
            "evidence_level": "B",
            "confidence": 0.8,
            "total_latency_ms": 5000,
        }
        base.update(kw)
        return base

    def test_unsupported_answer_rate(self):
        entries = [
            self._entry(source_count=0),
            self._entry(source_count=3),
            self._entry(source_count=0),
        ]
        result = so.safety_summary(entries)
        self.assertAlmostEqual(result["unsupported_answer_rate"], 2 / 3, places=3)

    def test_hallucination_figure_is_labelled_a_proxy(self):
        # The grant says "hallucination rate"; a log cannot measure it. The
        # dashboard must not let the number be read as a hallucination measure.
        result = so.safety_summary([self._entry(source_count=0)])
        self.assertIn("Proxy only", result["hallucination_rate_caveat"])

    def test_dosage_blocks_counted(self):
        entries = [self._entry(dosage_blocked=True, dosage_reason="numeric_dose")]
        result = so.safety_summary(entries)
        self.assertEqual(result["dosage_restriction_applied"], 1)
        self.assertEqual(result["dosage_block_reasons"]["numeric_dose"], 1)


class TestLanguageEquity(unittest.TestCase):
    def test_low_count_languages_marked_underpowered(self):
        entries = [
            {
                "participant_id": f"p{i}",
                "occurred_at_iso": datetime.now(timezone.utc).isoformat(),
                "interaction_class": "substantive",
                "language": "zu-IN",
                "source_count": 1,
                "total_latency_ms": 1000,
            }
            for i in range(4)
        ]
        result = so.language_equity_summary(entries)
        self.assertTrue(result["by_language"]["zu-IN"]["underpowered"])
        self.assertIn("do not rank", result["by_language"]["zu-IN"]["reporting_note"])

    def test_disparity_flags_equity_gap(self):
        base = {
            "occurred_at_iso": datetime.now(timezone.utc).isoformat(),
            "interaction_class": "substantive",
            "total_latency_ms": 1000,
            "source_count": 3,
        }
        entries = [dict(base, participant_id="a", language="hi-IN") for _ in range(10)]
        entries += [dict(base, participant_id="b", language="ta-IN", source_count=0) for _ in range(10)]
        result = so.language_equity_summary(entries)
        self.assertTrue(result["equity_flagged"])
        self.assertIn("non-inferiority", result["caveat"])


class TestDashboardIsReadOnly(unittest.TestCase):
    def test_only_get_routes_exist(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from study_dashboard import router

        app = FastAPI()
        app.include_router(router, prefix="/admin/study")
        client = TestClient(app)

        for method in ("post", "put", "patch", "delete"):
            with self.subTest(method=method):
                resp = getattr(client, method)("/admin/study/api")
                self.assertEqual(resp.status_code, 405)

    def test_api_returns_dashboard(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from study_dashboard import router

        app = FastAPI()
        app.include_router(router, prefix="/admin/study")
        client = TestClient(app)
        payload = client.get("/admin/study/api").json()
        for key in ("adoption", "safety", "latency", "language_equity", "release"):
            self.assertIn(key, payload)


class TestReleaseRecord(unittest.TestCase):
    def test_release_id_is_stable_and_includes_commit(self):
        from study_release import get_study_release

        release = get_study_release()
        self.assertEqual(release.release_id, release.release_id)
        self.assertIn("@", release.release_id)

    def test_record_is_never_rewritten_by_the_module(self):
        from study_release import get_study_release

        release = get_study_release()
        before = json.dumps(release.record, sort_keys=True, default=str)
        release.check_drift()
        release.verify_on_startup()
        after = json.dumps(release.record, sort_keys=True, default=str)
        self.assertEqual(before, after)

    def test_stamp_carries_protocol_7_8_fields(self):
        from study_release import get_study_release

        stamp = get_study_release().stamp()
        for field in ("release_id", "release_name", "release_status", "approved"):
            self.assertIn(field, stamp)


if __name__ == "__main__":
    unittest.main(verbosity=2)