#!/usr/bin/env python3
"""
Tests for the offline question set.

Two defects this guards against, both of which shipped before:

  * The question set was English-only while the bundle was built per language, so
    a Marathi user received a bundle whose questions were in English and matched
    nothing.
  * One of the twenty questions was "Morphine dosage and side effects", which the
    Dose Boundary refuses to answer, so the bundle would have shipped an entry that
    the guard stripped on the way out.

What is asserted here is structure, not fluency. Whether "இறக்கும் நோயாளியின் வாயை
ஆரமகரமாக வைப்பது" reads naturally is a native-speaker question and no test can
settle it. What a test can settle is that every language has twenty questions, in
the same order, written in its own script, and that none of them asks for a dose.
"""

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offline.questions import (  # noqa: E402
    ENGLISH,
    SUPPORTED_LANGUAGES,
    TRANSLATIONS,
    has_translation,
    questions_for,
)

# Unicode block each language is expected to be written in.
EXPECTED_BLOCKS = {
    "hi-IN": (0x0900, 0x097F),
    "bn-IN": (0x0980, 0x09FF),
    "kn-IN": (0x0C80, 0x0CFF),
    "ml-IN": (0x0D00, 0x0D7F),
    "mr-IN": (0x0900, 0x097F),
    "od-IN": (0x0B00, 0x0B7F),
    "pa-IN": (0x0A00, 0x0A7F),
    "ta-IN": (0x0B80, 0x0BFF),
    "te-IN": (0x0C00, 0x0C7F),
    "gu-IN": (0x0A80, 0x0AFF),
}


class TestCoverage(unittest.TestCase):
    def test_eleven_supported_languages(self):
        self.assertEqual(len(SUPPORTED_LANGUAGES), 11)

    def test_every_language_has_a_translation(self):
        for lang in SUPPORTED_LANGUAGES:
            with self.subTest(lang=lang):
                self.assertTrue(has_translation(lang), f"no translation for {lang}")

    def test_every_language_has_twenty_questions(self):
        for lang in SUPPORTED_LANGUAGES:
            with self.subTest(lang=lang):
                self.assertEqual(
                    len(questions_for(lang)), len(ENGLISH),
                    f"{lang} has the wrong number of questions",
                )

    def test_unknown_language_falls_back_to_english_and_says_so(self):
        self.assertEqual(questions_for("zu-IN"), ENGLISH)
        self.assertFalse(has_translation("zu-IN"))


class TestPositionalAlignment(unittest.TestCase):
    """The bundle zips the lists, so a shift would pair a question with the wrong answer."""

    def test_each_language_has_the_same_count(self):
        for lang, questions in TRANSLATIONS.items():
            with self.subTest(lang=lang):
                self.assertEqual(len(questions), len(ENGLISH))

    def test_english_entry_order_is_unchanged_at_index_16(self):
        # A regression guard on the entry that was hand-corrected.
        self.assertEqual(
            ENGLISH[16], "How to keep the mouth comfortable in a dying patient"
        )

    def test_no_duplicates_within_a_language(self):
        for lang in SUPPORTED_LANGUAGES:
            with self.subTest(lang=lang):
                questions = questions_for(lang)
                self.assertEqual(len(questions), len(set(questions)))


class TestScriptIntegrity(unittest.TestCase):
    """A question in the wrong script is as broken as a missing one."""

    def test_each_language_is_written_in_its_own_script(self):
        for lang, (lo, hi) in EXPECTED_BLOCKS.items():
            with self.subTest(lang=lang):
                for question in questions_for(lang):
                    self.assertTrue(
                        any(lo <= ord(c) <= hi for c in question),
                        f"{lang} question not in its own script: {question!r}",
                    )

    def test_no_transliteration_left_in_indic_questions(self):
        """A Latin-script transliteration would match nothing at lookup time."""
        for lang in EXPECTED_BLOCKS:
            with self.subTest(lang=lang):
                for question in questions_for(lang):
                    self.assertIsNone(
                        re.search(r"[A-Za-z]{3,}", question),
                        f"{lang} contains latin text: {question!r}",
                    )

    def test_no_replacement_characters(self):
        for lang in SUPPORTED_LANGUAGES:
            with self.subTest(lang=lang):
                for question in questions_for(lang):
                    self.assertNotIn("\ufffd", question)


class TestNoDosageQuestion(unittest.TestCase):
    """
    ADR 0004: the bundle must not pre-answer a question the Dose Boundary refuses.

    A cached entry that the guard strips on the way out is worse than a cache miss,
    because the user is told an answer exists and then does not receive it.
    """

    DOSAGE_TERMS = re.compile(
        r"\b(dosage|dose|doses|titrate|titrating|escalate|milligram|strength)\b"
        r"|\b(ডোজ|মাত্রা|剂量|मात्रा|ডোজ)"
        r"|\d+\s*(mg|ml|mcg)",
        re.I,
    )

    def test_no_question_asks_for_a_dose(self):
        for lang in SUPPORTED_LANGUAGES:
            for question in questions_for(lang):
                with self.subTest(lang=lang, question=question[:40]):
                    self.assertIsNone(
                        self.DOSAGE_TERMS.search(question),
                        f"{lang} bundle contains a dosage question: {question!r}",
                    )

    def test_the_removed_morphine_dosage_question_is_gone(self):
        for lang in SUPPORTED_LANGUAGES:
            with self.subTest(lang=lang):
                self.assertFalse(
                    any("morphine" in q.lower() and "dosage" in q.lower()
                        for q in questions_for(lang))
                )


class TestClinicalCoverage(unittest.TestCase):
    """
    The set should cover what an ASHA worker actually asks on a visit.

    Coverage is asserted on the canonical English list only. Translations are
    positionally aligned to it, so requiring an English or Hindi marker to appear
    inside a Gujarati sentence would be testing translation wording, which is the
    one thing a test cannot settle here.
    """

    REQUIRED_TOPICS = {
        "pain": ("pain",),
        "breathlessness": ("breathless",),
        "nausea": ("nausea", "vomiting"),
        "anxiety": ("anxiety", "fear"),
        "mouth": ("mouth sore",),
        "pressure sores": ("bedsore",),
        "fever": ("fever",),
        "end of life": ("end of life", "dying", "died"),
        "constipation": ("constipation",),
        "sleep": ("sleep",),
    }

    def test_english_set_covers_each_required_topic(self):
        text = " ".join(ENGLISH).lower()
        for topic, markers in self.REQUIRED_TOPICS.items():
            with self.subTest(topic=topic):
                self.assertTrue(
                    any(m in text for m in markers),
                    f"question set does not cover {topic}",
                )

    def test_every_translated_position_is_non_empty(self):
        for lang in SUPPORTED_LANGUAGES:
            for i, question in enumerate(questions_for(lang)):
                with self.subTest(lang=lang, index=i):
                    self.assertTrue(question.strip())



if __name__ == "__main__":
    unittest.main(verbosity=2)
