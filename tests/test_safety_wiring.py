#!/usr/bin/env python3
"""
Tests that the DosageGuard is actually wired into the serving path.

`test_dosage_guard.py` proves the guard works. It does not prove the server calls
it. Those are different failures, and the second one is silent: a pipeline that no
longer applies the guard returns confident, dosed answers without raising anything.

These tests drive the real `SafetyEnhancementsManager.process_response` rather
than the guard directly, so a disconnect between the two is caught here.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safety_enhancements import SafetyEnhancementsManager  # noqa: E402


class TestGuardIsWiredIntoProcessResponse(unittest.TestCase):
    def setUp(self):
        self.manager = SafetyEnhancementsManager()

    def test_pipeline_exposes_a_dosage_guard(self):
        self.assertIsNotNone(self.manager.dosage_guard)

    def test_dosing_response_is_restricted_by_the_pipeline(self):
        result = self.manager.process_response(
            query="what dose of morphine?",
            response="Give morphine 10 mg orally every 4 hours and titrate upward.",
            sources=[{"filename": "handbook.pdf"}],
            language="en-IN",
        )
        self.assertTrue(result.dosage_blocked)
        self.assertNotIn("10 mg", result.response)

    def test_general_drug_question_passes_through(self):
        result = self.manager.process_response(
            query="what is morphine used for?",
            response="Morphine is a strong opioid used for severe cancer pain.",
            sources=[{"filename": "handbook.pdf"}],
            language="en-IN",
        )
        self.assertFalse(result.dosage_blocked)
        self.assertIn("strong opioid", result.response)

    def test_a_refusal_is_never_graded_as_clinical_guidance(self):
        result = self.manager.process_response(
            query="what dose of morphine?",
            response="Give morphine 10 mg every 4 hours.",
            sources=[{"filename": "handbook.pdf"}],
            language="en-IN",
        )
        self.assertEqual(result.evidence_level, "E")
        self.assertEqual(result.validation_status, "dosage_restricted")
        self.assertEqual(result.confidence, 0.0)

    def test_critical_emergency_still_overrides_the_dosage_guard(self):
        result = self.manager.process_response(
            query="patient is choking and cannot breathe, call an ambulance",
            response=(
                "Call 108 immediately. This is a medical emergency and the patient "
                "must go to hospital now."
            ),
            sources=[{"filename": "handbook.pdf"}],
            language="en-IN",
        )
        self.assertEqual(result.emergency_level, "critical")
        self.assertFalse(result.dosage_blocked)
        self.assertTrue(result.emergency_overrode_dosage)
        self.assertIn("108", result.response)

    def test_high_severity_does_not_override_the_dose_guard(self):
        """"Severe pain" is HIGH, not CRITICAL, and must not switch the guard off."""
        result = self.manager.process_response(
            query="patient has severe pain, what dose of morphine?",
            response="Give morphine 10 mg orally every 4 hours.",
            sources=[{"filename": "handbook.pdf"}],
            language="en-IN",
        )
        self.assertEqual(result.emergency_level, "high")
        self.assertTrue(result.dosage_blocked)

    def test_result_exposes_the_fields_the_api_returns(self):
        result = self.manager.process_response(
            query="what is haloperidol?",
            response="Haloperidol is an antipsychotic used for agitation.",
            sources=[],
            language="en-IN",
        )
        for field in (
            "response", "evidence_level", "emergency_level", "confidence",
            "validation_status", "disclaimer", "dosage_blocked",
        ):
            self.assertTrue(hasattr(result, field), f"missing {field}")


class TestDefenceInDepth(unittest.TestCase):
    def test_manager_wiring_survives_a_reconfigured_guard(self):
        """A guard replaced after construction is still used, not a captured one.

        Subclasses DosageGuard rather than standing in as an unrelated object, so
        the manager's declared type still holds. A test-only duck type would let the
        attribute be assigned something the manager could not actually call.
        """
        from dosage_guard import DosageGuard, DosageGuardResult

        class StubGuard(DosageGuard):
            def apply(self, response, language="en", query=""):
                return DosageGuardResult(
                    blocked=True,
                    response="stubbed",
                    reason="stub",
                )

        manager = SafetyEnhancementsManager()
        manager.dosage_guard = StubGuard()
        result = manager.process_response(
            query="q", response="Give 10 mg.", sources=[], language="en-IN"
        )
        self.assertEqual(result.response, "stubbed")
        self.assertTrue(result.dosage_blocked)


if __name__ == "__main__":
    unittest.main(verbosity=2)
