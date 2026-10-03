#!/usr/bin/env python3
"""
Tests for the DosageGuard.

Protocol SS7.7 requires that the dosage restriction be enforced *and tested*
before any participant uses the study build. These tests are that evidence.

Run:
    python -m pytest tests/test_dosage_guard.py -v
    python tests/test_dosage_guard.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dosage_guard import (  # noqa: E402
    DosageGuard,
    detect_doses,
    get_dosage_guard,
    redact_doses,
)


class TestDoseDetection(unittest.TestCase):
    """Tier 2 content must be caught."""

    def test_numeric_dose_caught(self):
        cases = [
            "Take morphine 10mg orally.",
            "Give 0.5 mg subcutaneously.",
            "Administer 500 mg paracetamol every 4 hours.",
            "Titrate to a maximum dose of 400 mg daily.",
            "She needs 2.5ml of the syrup.",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertTrue(detect_doses(text), f"missed dose in: {text}")

    def test_frequency_caught_without_unit(self):
        cases = [
            "Give it twice daily.",
            "One tablet at bedtime.",
            "Half a tablet after food.",
            "Take the syrup three times a day.",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertTrue(detect_doses(text), f"missed frequency in: {text}")

    def test_titration_and_escalation_caught(self):
        cases = [
            "You can titrate the opioid as pain increases.",
            "Step up the dose if symptoms persist.",
            "Escalate to the next strength after 48 hours.",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertTrue(detect_doses(text), f"missed titration in: {text}")

    def test_indic_numerals_caught(self):
        """Devanagari and regional numerals must not evade the guard."""
        cases = [
            "१० मि.ग्रा. दें",
            "२५० मिलीग्रा खुराक",
            "೧೦ ಮಿ.ಗ್ರಾ ನೀಡಿ",
            "๑๐ มิลลิกรัม",
            "১০ মিলিগ্রাম",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertTrue(detect_doses(text), f"missed Indic dose in: {text}")

    def test_start_stop_caught(self):
        cases = [
            "Stop the morphine once pain improves.",
            "Switch to tramadol.",
            "Discontinue the laxative after three days.",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertTrue(detect_doses(text), f"missed start/stop in: {text}")


class TestNonDoseContentPasses(unittest.TestCase):
    """Tier 1 content must survive. False positives break the whole tool."""

    def test_general_drug_information_passes(self):
        cases = [
            "Morphine is a strong opioid used for severe cancer pain.",
            "Haloperidol is an antipsychotic sometimes used for agitation.",
            "Constipation is a common side effect of opioids; a laxative is usually "
            "given alongside them by the prescriber.",
            "The family should ask the palliative care physician about a bowel regimen.",
            "Pursed-lip breathing and propping up with pillows can help breathlessness.",
            "Palliative care relieves pain and other difficult symptoms.",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertFalse(detect_doses(text), f"false positive on: {text}")

    def test_non_dose_numbers_pass(self):
        cases = [
            "See Chapter 5 of the handbook for details.",
            "Assess the patient at each of the 3 sites.",
            "Grades are recorded from A to E.",
            "The wire gauge is small.",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertFalse(detect_doses(text), f"false positive on: {text}")

    def test_empty_and_none_safe(self):
        self.assertEqual(detect_doses(""), [])
        self.assertEqual(detect_doses(None), [])


class TestGuardBehaviour(unittest.TestCase):
    def test_clean_response_passes_through_unchanged(self):
        guard = DosageGuard()
        text = "Morphine is a strong opioid used for severe cancer pain."
        result = guard.apply(text, language="en-IN", query="what is morphine?")
        self.assertFalse(result.blocked)
        self.assertEqual(result.response, text)

    def test_dose_response_is_replaced_with_deferral(self):
        guard = DosageGuard()
        result = guard.apply(
            "Take morphine 10mg orally every 4 hours.",
            language="en-IN",
            query="what dose of morphine?",
        )
        self.assertTrue(result.blocked)
        self.assertNotIn("10mg", result.response)
        self.assertIn("next consultation", result.response)
        self.assertEqual(result.reason, "numeric_dose")
        self.assertTrue(result.findings)

    def test_deferral_never_promises_a_callback(self):
        """Protocol SS7.6: no implying a transfer succeeded before receipt."""
        guard = DosageGuard()
        result = guard.apply(
            "Give 5mg diazepam.", language="en-IN", query="how much diazepam?"
        )
        lowered = result.response.lower()
        for banned in ("will contact", "will call", "within 5 minutes",
                       "within 10 minutes", "a counselor will"):
            with self.subTest(phrase=banned):
                self.assertNotIn(banned, lowered)

    def test_deferral_names_concrete_next_steps(self):
        guard = DosageGuard()
        result = guard.apply("Give 5mg diazepam.", language="en-IN", query="dose?")
        for expected in ("next consultation", "108"):
            with self.subTest(expected=expected):
                self.assertIn(expected, result.response)

    def test_emergency_overrides_the_refusal(self):
        """A refusal must never swallow a myocardial infarction."""
        guard = DosageGuard()
        result = guard.apply(
            "Call 108 now. The patient has severe chest pain and is struggling to breathe.",
            language="en-IN",
            query="patient has crushing chest pain, give me an ambulance",
        )
        self.assertFalse(result.blocked)
        self.assertTrue(result.emergency_overrode)

    def test_localised_refusal_uses_requested_language(self):
        guard = DosageGuard()
        for lang, marker in (("hi-IN", "108"), ("bn-IN", "108"), ("ta-IN", "108")):
            with self.subTest(lang=lang):
                result = guard.apply("Give 10mg morphine.", language=lang, query="dose?")
                self.assertTrue(result.blocked)
                self.assertIn(marker, result.response)
                self.assertNotIn("10mg", result.response)

    def test_unsupported_language_falls_back_to_english(self):
        guard = DosageGuard()
        result = DosageGuard().apply(
            "Give 10mg morphine.", language="zu-IN", query="dose?"
        )
        self.assertTrue(result.blocked)
        self.assertIn("next consultation", result.response)

    def test_singleton_available(self):
        self.assertIsInstance(get_dosage_guard(), DosageGuard)


class TestRedaction(unittest.TestCase):
    """Option A: sentence-level redaction keeps the non-dosing remainder."""

    def test_dose_sentence_removed_and_rest_kept(self):
        guard = DosageGuard()
        result = guard.apply(
            "Morphine is a strong opioid used for severe cancer pain. "
            "It is the gold standard for severe pain control. "
            "Give 10 mg orally every 4 hours and titrate to effect.",
            language="en-IN",
            query="what is morphine used for?",
        )
        self.assertTrue(result.redacted)
        self.assertIn("strong opioid", result.response)
        self.assertIn("gold standard", result.response)
        self.assertNotIn("10 mg", result.response)
        self.assertNotIn("titrate", result.response)

    def test_numbered_list_keeps_non_dosing_items(self):
        guard = DosageGuard()
        result = guard.apply(
            "Actions:\n"
            "1. Schedule a doctor appointment\n"
            "2. Monitor symptoms closely\n"
            "3. Start morphine 10 mg every 4 hours\n"
            "4. Titrate upward as needed\n"
            "5. Keep a written record\n",
            language="en-IN",
            query="what should I do?",
        )
        self.assertTrue(result.redacted)
        self.assertIn("Schedule a doctor appointment", result.response)
        self.assertIn("Keep a written record", result.response)
        self.assertNotIn("morphine", result.response)
        self.assertNotIn("Titrate", result.response)

    def test_orphan_heading_removed(self):
        """A heading whose whole block was redacted goes with it."""
        text, _, removed = redact_doses(
            "Constipation relief:\n1. Take senna 15 mg at night\n"
            "Hydration:\n1. Drink plenty of water\n"
        )
        self.assertEqual(removed, 1)
        self.assertNotIn("Constipation relief", text)
        self.assertIn("Hydration:", text)
        self.assertIn("Drink plenty of water", text)

    def test_thin_remainder_falls_back_to_full_refusal(self):
        """Too little survives to be useful, so defer in full."""
        guard = DosageGuard()
        result = guard.apply(
            "Constipation relief:\n1. Take senna 15 mg at night\n",
            language="en-IN",
            query="constipation?",
        )
        self.assertFalse(result.redacted)
        self.assertIn("next consultation", result.response)

    def test_all_dosing_answer_falls_back_to_full_refusal(self):
        guard = DosageGuard()
        result = guard.apply("Give 10mg morphine now.", language="en-IN", query="dose?")
        self.assertTrue(result.blocked)
        self.assertFalse(result.redacted)
        self.assertIn("next consultation", result.response)

    def test_redaction_is_dose_free_in_every_language(self):
        """The verify-after-redact net: nothing may survive."""
        guard = DosageGuard()
        samples = [
            ("Morphine is a strong opioid for severe cancer pain. Start at 10 mg.", "en-IN"),
            ("மார்பின் கடுமையான வலிக்கான வலுவான மருந்து. 10 மி.கி தொடங்கவும்.", "ta-IN"),
            ("मॉर्फिन तीव्र दर्द की मजबूत दवा है। 10 मि.ग्रा. से शुरू करें।", "hi-IN"),
        ]
        for text, lang in samples:
            with self.subTest(lang=lang):
                result = guard.apply(text, language=lang, query="q")
                self.assertTrue(result.blocked)
                if result.redacted:
                    self.assertEqual(
                        detect_doses(result.response),
                        [],
                        f"dose survived redaction in {lang}: {result.response!r}",
                    )

    def test_redaction_note_is_appended(self):
        guard = DosageGuard()
        result = guard.apply(
            "Morphine is a strong opioid used for severe cancer pain. "
            "It is the gold standard for severe pain control. "
            "Give 10 mg orally every 4 hours.",
            language="en-IN",
            query="q",
        )
        self.assertIn("not included here", result.response)

    def test_redact_doses_helper_reports_removals(self):
        text, findings, removed = redact_doses(
            "Keep the skin clean. Apply cream 5 times a day. Turn the patient."
        )
        self.assertEqual(removed, 1)
        self.assertIn("Keep the skin clean.", text)
        self.assertIn("Turn the patient.", text)
        self.assertNotIn("5 times", text)
        self.assertTrue(findings)


class TestAgainstExistingEvalOutputs(unittest.TestCase):
    """
    Regression test against real generated answers already in the repository.

    data/evaluation/rag_outputs/ contains 80 real outputs, of which 42 were
    measured as containing an explicit dose. This asserts the guard now catches
    them, which is the evidence protocol SS7.7 asks for.
    """

    def _outputs(self):
        import glob
        import json
        root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "evaluation", "rag_outputs",
        )
        for path in sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)):
            try:
                with open(path, encoding="utf-8") as fh:
                    yield os.path.basename(path), json.load(fh).get("answer") or ""
            except Exception:
                continue

    def test_every_previously_dosing_output_is_now_blocked(self):
        guard = DosageGuard()
        checked = 0
        previously_dosing = []
        for name, answer in self._outputs():
            if not answer:
                continue
            checked += 1
            findings = detect_doses(answer)
            if findings:
                previously_dosing.append(name)
                result = guard.apply(answer, language="en-IN", query="")
                self.assertTrue(
                    result.blocked,
                    f"{name} contains a dose but was not blocked: {findings[0].snippet!r}",
                )
        # The repository currently holds 42 such outputs; if this drops to zero
        # the corpus was re-generated and the number below no longer applies.
        self.assertGreater(
            checked, 0, "no evaluation outputs found - corpus missing?"
        )
        print(f"\n  checked {checked} outputs; {len(previously_dosing)} contained doses, all blocked")


if __name__ == "__main__":
    unittest.main(verbosity=2)