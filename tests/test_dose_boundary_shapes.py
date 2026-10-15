#!/usr/bin/env python3
"""
Every dose shape the round-3 review found passing the Dose Boundary.

The review (Spec B3) ran `DosageGuard.apply` and listed doses it let through:
spelled-out units, doses written in words, counts of tablets and puffs,
"4-hourly", Hindi abbreviations, and any sentence that also mentioned an age.
Each one is a row below, driven through the answer gate in its Supported
Language. A row passes only when the answer is blocked (Redacted Answer or
Deferral) and the dose text itself is gone from what the participant reads.

Run:
    python -m pytest tests/test_dose_boundary_shapes.py -v
"""

import ast
import os
import re
import sys
import unicodedata
import unittest
import warnings
from pathlib import Path
from typing import List, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dosage_guard import DosageGuard  # noqa: E402

# A question that is not an emergency, so the Emergency Override cannot be what
# decides the outcome.
QUESTION = "What can we do about the pain?"

# (answer, Supported Language, the dose text that must not reach the participant)
REVIEW_B3_MISSES = [
    ("Morphine 5 milligrams is what the hospice used.", "en-IN", "5 milligrams"),
    ("The patch gives 25 micrograms per hour of fentanyl.", "en-IN", "25 micrograms"),
    ("Paracetamol 500 milligrammes TDS is usual.", "en-IN", "500 milligrammes"),
    ("Give five milligrams every four hours.", "en-IN", "five milligrams"),
    ("Take two tablets at night.", "en-IN", "two tablets"),
    ("Use 2 puffs of the inhaler when breathless.", "en-IN", "2 puffs"),
    ("Oral morphine is usually given 4-hourly.", "en-IN", "4-hourly"),
    ("मॉर्फिन 5 एमजी दें।", "hi-IN", "5 एमजी"),
    ("मॉर्फिन 10 एम.जी. दें।", "hi-IN", "10 एम.जी."),
    ("For patients aged 70, give morphine 10 mg every 4 hours.", "en-IN", "10 mg"),
]

# A dose written with the unit word of each Supported Language, in its own script.
# The unit is the only dose signal in each sentence, so a row can pass only if
# that language's unit word is recognised.
UNIT_WORD_IN_EACH_LANGUAGE = [
    ("Give the patient 10 milligrams of morphine.", "en-IN", "10 milligrams"),
    ("फेंटानिल पैच से 25 माइक्रोग्राम मिलता है।", "hi-IN", "25 माइक्रोग्राम"),
    ("रुग्णाला 10 मिलिग्रॅम मॉर्फिन द्या.", "mr-IN", "10 मिलिग्रॅम"),
    ("রোগীকে ৫ গ্রাম প্যারাসিটামল দিন।", "bn-IN", "৫ গ্রাম"),
    ("ਮਰੀਜ਼ ਨੂੰ 5 ਗ੍ਰਾਮ ਦਵਾਈ ਦਿਓ।", "pa-IN", "5 ਗ੍ਰਾਮ"),
    ("દર્દીને 10 મિલિગ્રામ મોર્ફિન આપો.", "gu-IN", "10 મિલિગ્રામ"),
    ("ରୋଗୀଙ୍କୁ 10 ମିଲିଗ୍ରାମ ମର୍ଫିନ ଦିଅନ୍ତୁ।", "od-IN", "10 ମିଲିଗ୍ରାମ"),
    ("രോഗിക്ക് 10 മില്ലിഗ്രാം മോർഫിൻ നൽകുക.", "ml-IN", "10 മില്ലിഗ്രാം"),
    ("நோயாளிக்கு 10 மில்லிகிராம் மார்பின் கொடுங்கள்.", "ta-IN", "10 மில்லிகிராம்"),
    ("రోగికి 10 మిల్లీగ్రాములు మార్ఫిన్ ఇవ్వండి.", "te-IN", "10 మిల్లీగ్రాములు"),
    ("ರೋಗಿಗೆ 10 ಮಿಲಿಗ್ರಾಂ ಮಾರ್ಫಿನ್ ನೀಡಿ.", "kn-IN", "10 ಮಿಲಿಗ್ರಾಂ"),
]


# Ordinary words that contain a dose vocabulary word inside them. Each must reach
# the participant untouched: the frequency word (बार, रात, ਵਾਰ, ...) is only a
# dose signal as a whole word, not as part of "बारे" (about) or "घरात" (at home).
NOT_A_DOSE_IN_EACH_SCRIPT = [
    ("परिवार को बीमारी के बारे में बताएँ।", "hi-IN"),        # बार in बारे
    ("दिनेश घर पर आराम कर रहे हैं।", "hi-IN"),              # दिन in दिनेश
    ("घरात आराम करा.", "mr-IN"),                             # रात in घरात
    ("ਪਰਿਵਾਰ ਨਾਲ ਗੱਲ ਕਰੋ।", "pa-IN"),                         # ਵਾਰ in ਪਰਿਵਾਰ
    ("તહેવાર પર પરિવાર સાથે રહો.", "gu-IN"),                  # વાર in તહેવાર, પરિવાર
    ("ಅವರು ಒಬ್ಬರೇ ಇರಬಾರದು.", "kn-IN"),                       # ಬಾರ in ಇರಬಾರದು
    ("முறையாக ஓய்வு எடுங்கள்.", "ta-IN"),                      # முறை in முறையாக
    ("రోజా పువ్వు అందంగా ఉంది.", "te-IN"),                     # రోజ in రోజా
    ("ସରୋଜ ଘରେ ଅଛନ୍ତି।", "od-IN"),                             # ରୋଜ in ସରୋଜ
]

# The same vocabulary used as whole words still marks a schedule.
SCHEDULE_AS_WHOLE_WORDS = [
    ("मॉर्फिन दिन में दो बार दें।", "hi-IN"),
    ("यह दवा रोज़ सुबह दें।", "hi-IN"),
    ("ਇਹ ਦਵਾਈ ਦੋ ਵਾਰ ਦਿਓ।", "pa-IN"),
    ("ಈ ಔಷಧಿಯನ್ನು ಪ್ರತಿ ದಿನ ನೀಡಿ.", "kn-IN"),
]


# Inflected schedule words. Before whole-word matching, these were caught only
# because a listed stem sat inside them (ಬಾರ in ಬಾರಿ, రోజ in రోజుకు, रात in
# रात्री). Each is now listed in its own right, so the boundary is no weaker.
INFLECTED_SCHEDULE_WORDS = [
    ("यह दवा रोज़ाना दें।", "hi-IN"),
    ("यह दवा रोजाना दें।", "hi-IN"),
    ("यह दवा प्रतिदिन दें।", "hi-IN"),
    ("यह दवा रात्रि में दें।", "hi-IN"),
    ("रातों को यह दवा दें।", "hi-IN"),
    ("मरीज़ को दो गोलियों से आराम मिला।", "hi-IN"),
    ("आँख में पाँच बूंदों से आराम मिला।", "hi-IN"),
    ("मरीज़ को दो खुराकें दें।", "hi-IN"),
    ("रात्री ही दवा द्या.", "mr-IN"),
    ("ಈ ಔಷಧಿಯನ್ನು ಎರಡು ಬಾರಿ ನೀಡಿ.", "kn-IN"),
    ("ಈ ಔಷಧಿಯನ್ನು ದಿನಕ್ಕೆ ನೀಡಿ.", "kn-IN"),
    ("ಈ ಔಷಧಿಯನ್ನು ದಿನವೂ ನೀಡಿ.", "kn-IN"),
    ("ಈ ಔಷಧಿಯನ್ನು ಪ್ರತಿದಿನ ನೀಡಿ.", "kn-IN"),
    ("ఈ మందు రోజుకు ఇవ్వండి.", "te-IN"),
    ("ఈ మందు రోజూ ఇవ్వండి.", "te-IN"),
    ("ఈ మందు మూడు రోజులు ఇవ్వండి.", "te-IN"),
    ("இந்த மருந்தை தினமும் கொடுங்கள்.", "ta-IN"),
    ("இந்த மருந்தை தினசரி கொடுங்கள்.", "ta-IN"),
    ("இந்த மருந்தை நாளுக்கு கொடுங்கள்.", "ta-IN"),
    ("இந்த மருந்தை வாரத்திற்கு கொடுங்கள்.", "ta-IN"),
    ("ਇਹ ਦਵਾਈ ਰੋਜ਼ਾਨਾ ਦਿਓ।", "pa-IN"),
    ("ਇਹ ਦਵਾਈ ਦੋ ਵਾਰੀ ਦਿਓ।", "pa-IN"),
    ("આ દવા દરરોજ આપો.", "gu-IN"),
    ("આ દવા દિવસમાં આપો.", "gu-IN"),
    ("ଏହି ଔଷଧ ପ୍ରତିଦିନ ଦିଅନ୍ତୁ।", "od-IN"),
    ("ଏହି ଔଷଧ ଦିନରେ ଦିଅନ୍ତୁ।", "od-IN"),
    ("ଏହି ଔଷଧ ଦିନକୁ ଦିଅନ୍ତୁ।", "od-IN"),
]


# Counts of a dose form and schedules in each Supported Language that Live
# speaks (final review, round 3, finding 1). Before this, the count nouns were
# English and Hindi only, so every row here passed the Dose Boundary and Live
# spoke it. The first rows are the review's own strings.
# (answer, Supported Language, the dose text that must not reach the participant)
COUNTS_AND_SCHEDULES_IN_EACH_LANGUAGE = [
    ("दर चार तासांनी दोन गोळ्या द्या", "mr-IN", "दोन गोळ्या"),
    ("இரண்டு மாத்திரைகளை ஒவ்வொரு நான்கு மணி நேரத்திற்கும் கொடுக்கவும்", "ta-IN", "இரண்டு மாத்திரைகளை"),
    ("દર ચાર કલાકે બે ગોળી આપો", "gu-IN", "બે ગોળી"),
    ("প্রতি চার ঘণ্টা অন্তর দুটি ট্যাবলেট দিন", "bn-IN", "দুটি ট্যাবলেট"),
    ("காலையில் 2 மாத்திரை சாப்பிடுங்கள்.", "ta-IN", "2 மாத்திரை"),
    ("দিনে দুবার খান", "bn-IN", "দুবার"),
    ("दिवसातून दोनदा घ्या", "mr-IN", "दोनदा"),
    ("ਦੋ ਗੋਲੀਆਂ ਲਓ", "pa-IN", "ਦੋ ਗੋਲੀਆਂ"),
    ("Take ½ tablet", "en-IN", "½ tablet"),
    ("Take 1½ tablets at night.", "en-IN", "1½ tablets"),
    ("Give ¼ tablet when the pain starts.", "en-IN", "¼ tablet"),
    ("Use ¾ teaspoon of the syrup.", "en-IN", "¾ teaspoon"),
    ("½ गोली दें।", "hi-IN", "½ गोली"),
    ("हर चार घंटे में दवा दें।", "hi-IN", "हर चार घंटे"),
    # Bengali
    ("রাতে ২টি ট্যাবলেট খান।", "bn-IN", "২টি ট্যাবলেট"),
    ("দিনে তিন বার এই ওষুধ দিন।", "bn-IN", "তিন বার"),
    ("প্রতি ৬ ঘণ্টায় ওষুধ দিন।", "bn-IN", "প্রতি ৬ ঘণ্টায়"),
    ("চোখে দুই ফোঁটা দিন।", "bn-IN", "দুই ফোঁটা"),
    ("ওষুধটি চার ঘণ্টা পর পর দিন।", "bn-IN", "চার ঘণ্টা পর পর"),
    # Marathi
    ("दिवसातून तीन वेळा औषध द्या.", "mr-IN", "तीन वेळा"),
    ("जेवणानंतर 2 गोळ्या द्या.", "mr-IN", "2 गोळ्या"),
    ("डोळ्यात दोन थेंब टाका.", "mr-IN", "दोन थेंब"),
    ("दर ६ तासांनी औषध द्या.", "mr-IN", "दर ६ तासांनी"),
    # Gujarati
    ("જમ્યા પછી બે વખત દવા આપો.", "gu-IN", "બે વખત"),
    ("દર 6 કલાકે દવા આપો.", "gu-IN", "દર 6 કલાકે"),
    ("એક ચમચી સીરપ આપો.", "gu-IN", "એક ચમચી"),
    ("અડધી ગોળી આપો.", "gu-IN", "અડધી ગોળી"),
    # Punjabi
    ("ਹਰ ਚਾਰ ਘੰਟੇ ਬਾਅਦ ਦਵਾਈ ਦਿਓ।", "pa-IN", "ਹਰ ਚਾਰ ਘੰਟੇ"),
    ("ਅੱਧੀ ਗੋਲੀ ਦਿਓ।", "pa-IN", "ਅੱਧੀ ਗੋਲੀ"),
    ("ਅੱਖ ਵਿੱਚ 2 ਬੂੰਦਾਂ ਪਾਓ।", "pa-IN", "2 ਬੂੰਦਾਂ"),
    # Tamil
    ("ஒரு நாளைக்கு மூன்று தடவை கொடுக்கவும்.", "ta-IN", "மூன்று தடவை"),
    ("இரண்டு சொட்டு கண்ணில் விடவும்.", "ta-IN", "இரண்டு சொட்டு"),
    ("அரை மாத்திரை கொடுக்கவும்.", "ta-IN", "அரை மாத்திரை"),
    ("இரவில் இரண்டு வேளை கொடுக்கவும்.", "ta-IN", "இரண்டு வேளை"),
    # Telugu
    ("రెండు మాత్రలు వేసుకోండి.", "te-IN", "రెండు మాత్రలు"),
    ("ప్రతి నాలుగు గంటలకు మందు ఇవ్వండి.", "te-IN", "ప్రతి నాలుగు గంటలకు"),
    ("నాలుగు గంటలకోసారి మందు ఇవ్వండి.", "te-IN", "నాలుగు గంటలకోసారి"),
    ("అర మాత్ర ఇవ్వండి.", "te-IN", "అర మాత్ర"),
    # Kannada
    ("ಎರಡು ಮಾತ್ರೆಗಳನ್ನು ನೀಡಿ.", "kn-IN", "ಎರಡು ಮಾತ್ರೆಗಳನ್ನು"),
    ("ಪ್ರತಿ ನಾಲ್ಕು ಗಂಟೆಗೆ ಔಷಧಿ ನೀಡಿ.", "kn-IN", "ಪ್ರತಿ ನಾಲ್ಕು ಗಂಟೆಗೆ"),
    ("ನಾಲ್ಕು ಗಂಟೆಗೊಮ್ಮೆ ಔಷಧಿ ನೀಡಿ.", "kn-IN", "ನಾಲ್ಕು ಗಂಟೆಗೊಮ್ಮೆ"),
    ("ಮೂರು ಸಲ ಔಷಧಿ ನೀಡಿ.", "kn-IN", "ಮೂರು ಸಲ"),
    # Malayalam
    ("രണ്ട് ഗുളിക കഴിക്കുക.", "ml-IN", "രണ്ട് ഗുളിക"),
    ("ഓരോ നാല് മണിക്കൂറിലും മരുന്ന് നൽകുക.", "ml-IN", "ഓരോ നാല് മണിക്കൂറിലും"),
    ("ദിവസം മൂന്നു നേരം മരുന്ന് നൽകുക.", "ml-IN", "മൂന്നു നേരം"),
    ("ദിവസം രണ്ട് തവണ മരുന്ന് നൽകുക.", "ml-IN", "രണ്ട് തവണ"),
    ("കണ്ണിൽ രണ്ടു തുള്ളി ഒഴിക്കുക.", "ml-IN", "രണ്ടു തുള്ളി"),
    # Odia
    ("ଦୁଇଟି ଟାବଲେଟ ଦିଅନ୍ତୁ।", "od-IN", "ଦୁଇଟି ଟାବଲେଟ"),
    ("ପ୍ରତି ଚାରି ଘଣ୍ଟାରେ ଔଷଧ ଦିଅନ୍ତୁ।", "od-IN", "ପ୍ରତି ଚାରି ଘଣ୍ଟାରେ"),
    ("ତିନି ଥର ଔଷଧ ଦିଅନ୍ତୁ।", "od-IN", "ତିନି ଥର"),
]

# Ordinary sentences built from the same words: a number word with no dose
# form after it, a time span with no "every", "give" (Bengali "দিন" is also
# "day"), "perhaps" (Tamil "ஒரு வேளை"), "only" (Telugu "మాత్రమే" starts with
# "మాత్ర"), "advice" (Kannada "ಸಲಹೆ" starts with "ಸಲ"), "o'clock".
# Each must reach the participant untouched.
ORDINARY_SENTENCES_WITH_COUNT_WORDS = [
    ("রোগীকে জল দিন।", "bn-IN"),
    ("দুই ঘণ্টা বিশ্রাম নিন।", "bn-IN"),
    ("আমি আপনাকে দুটি কথা বলি।", "bn-IN"),
    ("चार तासांनी परत या.", "mr-IN"),
    ("दोन गोष्टी लक्षात ठेवा.", "mr-IN"),
    ("બે દીકરા ઘરે છે.", "gu-IN"),
    ("ਦੋ ਗੱਲਾਂ ਯਾਦ ਰੱਖੋ।", "pa-IN"),
    ("ஒரு வேளை வலி அதிகமானால் மருத்துவரை அழைக்கவும்.", "ta-IN"),
    ("நான்கு மணிக்கு மருத்துவர் வருவார்.", "ta-IN"),
    ("రెండు మాత్రమే మిగిలాయి.", "te-IN"),
    ("ಎರಡು ಸಲಹೆಗಳನ್ನು ನೆನಪಿಡಿ.", "kn-IN"),
    ("രണ്ട് കുട്ടികൾ വീട്ടിലുണ്ട്.", "ml-IN"),
    ("കുറച്ച് നേരം വിശ്രമിക്കുക.", "ml-IN"),
    ("ଦୁଇ ଜଣ ଘରେ ଅଛନ୍ତି।", "od-IN"),
    ("About ½ of patients feel tired.", "en-IN"),
]


class TestCountsAndSchedulesInEveryLanguage(unittest.TestCase):
    """A tablet count or a schedule is withheld in every Supported Language."""

    def test_count_or_schedule_in_each_language_is_withheld(self):
        for answer, language, dose in COUNTS_AND_SCHEDULES_IN_EACH_LANGUAGE:
            with self.subTest(language=language, answer=answer):
                result = DosageGuard().apply(answer, language=language, query=QUESTION)
                self.assertTrue(result.blocked, f"dose passed the gate: {answer!r}")
                self.assertNotIn(dose, result.response)

    def test_the_whole_count_or_schedule_is_found(self):
        """The detector reports the dose span itself, not a fragment beside it."""
        import dose_detector

        for answer, _language, dose in COUNTS_AND_SCHEDULES_IN_EACH_LANGUAGE:
            with self.subTest(answer=answer):
                found = [f.pattern for f in dose_detector.detect_doses(answer)]
                self.assertTrue(
                    any(dose in p or p in dose for p in found), f"{dose!r} not in {found}"
                )

    def test_ordinary_sentence_with_a_count_word_passes_untouched(self):
        for answer, language in ORDINARY_SENTENCES_WITH_COUNT_WORDS:
            with self.subTest(language=language, answer=answer):
                result = DosageGuard().apply(answer, language=language, query=QUESTION)
                self.assertFalse(result.blocked, f"redacted a non-dose: {result.findings}")
                self.assertEqual(result.response, answer)

    def test_a_nukta_letter_matches_in_either_encoding(self):
        """"দেড়" (one and a half) is one code point or two, depending on the keyboard."""
        import dose_detector

        for rra in ("ড়", "ড়"):
            answer = f"দে{rra} ট্যাবলেট দিন।"
            with self.subTest(encoding=[hex(ord(c)) for c in rra]):
                self.assertTrue(dose_detector.contains_dose(answer))

    def test_live_withholds_the_review_turns(self):
        """Live's audio is withheld for each turn the review heard released."""
        from gemini_live.turn_release import check_live_turn

        for answer, language, _dose in COUNTS_AND_SCHEDULES_IN_EACH_LANGUAGE[:4]:
            with self.subTest(language=language):
                release = check_live_turn([b"audio"], answer, QUESTION, language)
                self.assertTrue(release.dosage_blocked)
                self.assertTrue(release.withheld)
                self.assertEqual(release.audio, [])


class TestIndicWordsMatchWholeWordsOnly(unittest.TestCase):
    """An Indic dose word inside a longer word is not a dose."""

    def test_word_containing_a_frequency_word_passes_untouched(self):
        for answer, language in NOT_A_DOSE_IN_EACH_SCRIPT:
            with self.subTest(language=language, answer=answer):
                result = DosageGuard().apply(answer, language=language, query=QUESTION)
                self.assertFalse(result.blocked, f"redacted a non-dose: {result.findings}")
                self.assertEqual(result.response, answer)

    def test_frequency_word_on_its_own_is_still_withheld(self):
        for answer, language in SCHEDULE_AS_WHOLE_WORDS:
            with self.subTest(language=language, answer=answer):
                result = DosageGuard().apply(answer, language=language, query=QUESTION)
                self.assertTrue(result.blocked, f"schedule passed the gate: {answer!r}")
                # The Deferral itself may use the same word ("अगली बार"), so
                # check that the schedule sentence is gone, not the word.
                self.assertNotIn(answer, result.response)

    def test_inflected_schedule_word_is_still_withheld(self):
        for answer, language in INFLECTED_SCHEDULE_WORDS:
            with self.subTest(language=language, answer=answer):
                result = DosageGuard().apply(answer, language=language, query=QUESTION)
                self.assertTrue(result.blocked, f"schedule passed the gate: {answer!r}")
                self.assertNotIn(answer, result.response)


class TestDoseBoundaryCatchesEveryShape(unittest.TestCase):
    """Each dose the review saw pass is now blocked at the answer gate."""

    def _assert_dose_withheld(self, answer: str, language: str, dose: str) -> None:
        result = DosageGuard().apply(answer, language=language, query=QUESTION)
        self.assertTrue(result.blocked, f"dose passed the gate: {answer!r}")
        self.assertNotIn(dose, result.response, f"dose reached the participant: {dose!r}")

    def test_every_review_b3_miss_is_withheld(self):
        for answer, language, dose in REVIEW_B3_MISSES:
            with self.subTest(answer=answer):
                self._assert_dose_withheld(answer, language, dose)

    def test_unit_word_of_every_supported_language_is_withheld(self):
        for answer, language, dose in UNIT_WORD_IN_EACH_LANGUAGE:
            with self.subTest(language=language):
                self._assert_dose_withheld(answer, language, dose)

    def test_age_exception_spares_only_the_age_not_the_dose_beside_it(self):
        answer = (
            "Pain is common in people aged 70 and over, and it can be eased. "
            "Keep a diary of when the pain comes and what helps. "
            "For patients aged 70, give morphine 10 mg every 4 hours."
        )
        result = DosageGuard().apply(answer, language="en-IN", query=QUESTION)
        self.assertTrue(result.redacted, result.response)
        self.assertIn("aged 70 and over", result.response)
        self.assertNotIn("10 mg", result.response)


ROOT = Path(__file__).resolve().parent.parent

# The modules that hold, or used to hold, the Dose Boundary's own vocabulary.
VOCABULARY_MODULES = ("dose_detector.py", "dosage_guard.py")


def _string_literals(path: Path) -> List[Tuple[int, str]]:
    """Every string constant in `path`, with regex \\uXXXX escapes decoded."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            decoded = re.sub(
                r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), node.value
            )
            out.append((node.lineno, decoded))
    return out


def _scripts(word: str) -> Set[str]:
    """The Unicode scripts of the non-ASCII letters and marks in `word`."""
    scripts = set()
    for ch in word:
        if ord(ch) < 128 or not unicodedata.category(ch).startswith(("L", "M")):
            continue
        scripts.add(unicodedata.name(ch, "UNKNOWN").split()[0])
    return scripts


class TestIndicVocabularyIsClean(unittest.TestCase):
    """
    The unit and frequency words are each written in one script.

    Round 3 found Punjabi, Gujarati and Bengali unit words that mixed in letters
    of another script, so they could never match real text, and a U+FFFD
    replacement character in the Odia frequency words.
    """

    def _modules(self) -> List[Path]:
        return [ROOT / name for name in VOCABULARY_MODULES if (ROOT / name).exists()]

    def test_no_replacement_character_in_the_dose_vocabulary(self):
        for path in self._modules():
            with self.subTest(module=path.name):
                self.assertNotIn("�", path.read_text(encoding="utf-8"))

    def test_every_vocabulary_word_is_in_a_single_script(self):
        for path in self._modules():
            for lineno, literal in _string_literals(path):
                for word in re.split(r"[|\s]+", literal):
                    scripts = _scripts(word)
                    with self.subTest(module=path.name, line=lineno, word=word):
                        self.assertLessEqual(
                            len(scripts), 1,
                            f"{path.name}:{lineno} mixes scripts {sorted(scripts)}: {word!r}",
                        )


# What a dose regex is, observed from outside: it finds the dose in the first
# sentence of a pair and not the near-identical non-dose in the second. A
# generic pattern such as r"\d+\s*\w+" finds both and is not a dose regex.
DOSE_PROBES = [
    ("Give 10 mg of morphine.", "Give 10 qz of morphine."),
    ("Give 5 milligrams of it.", "Give 5 kilometres of it."),
    ("Give 25 mcg of it.", "Give 25 xyz of it."),
    ("Give 5 ml of it.", "Give 5 km of it."),
    ("Take 2 tablets now.", "Take 2 tickets now."),
    ("Take two tablets now.", "Take two tickets now."),
    ("Use 2 puffs now.", "Use 2 pages now."),
    ("Give it 4-hourly.", "Give it 4-yearly."),
    ("10 मि.ग्रा. दें।", "10 कि.मी. दें।"),
]

# The one module allowed to define dose patterns.
DOSE_DETECTOR = Path("dose_detector.py")

# Not the study server's code: the vendored Kotaemon tree, generated docs and
# tool folders.
NOT_SCANNED = {".git", ".claude", "kotaemon-main", "site", "node_modules", "venv", ".venv"}

_REGEX_METACHARACTERS = set("\\()[]|?*+{}")


def _is_dose_regex(literal: str) -> bool:
    """True when `literal`, read as a regex, finds a dose probe but not its twin."""
    if not set(literal) & _REGEX_METACHARACTERS:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pattern = re.compile(literal, re.I)
    except (re.error, RecursionError, OverflowError, ValueError):
        return False
    return any(
        pattern.search(dose) and not pattern.search(twin) for dose, twin in DOSE_PROBES
    )


class TestOneDoseDetector(unittest.TestCase):
    """
    Only dose_detector defines what a dose looks like.

    Round 3 found four copies (a dead pattern table in the guard, the clinical
    validator's own regex, two in the observation extractors and one in a test),
    each catching a different subset. This fails when any other module holds a
    regex that recognises a dose, so a fifth copy cannot appear unnoticed.
    """

    def test_no_module_but_the_dose_detector_defines_a_dose_pattern(self):
        offenders = []
        for path in sorted(ROOT.rglob("*.py")):
            rel = path.relative_to(ROOT)
            if rel == DOSE_DETECTOR or set(rel.parts) & NOT_SCANNED:
                continue
            try:
                literals = _string_literals(path)
            except (SyntaxError, UnicodeDecodeError, ValueError):
                continue
            offenders.extend(
                f"{rel}:{lineno}: {literal[:80]!r}"
                for lineno, literal in literals
                if _is_dose_regex(literal)
            )
        self.assertEqual(
            offenders, [],
            "dose patterns outside dose_detector.py; call dose_detector.detect_doses instead",
        )

    def test_the_probes_recognise_the_detectors_own_patterns(self):
        """Guards the test above against probes that could never fire."""
        import dose_detector

        for dose, twin in DOSE_PROBES:
            with self.subTest(dose=dose):
                self.assertTrue(dose_detector.contains_dose(dose))
                self.assertFalse(dose_detector.contains_dose(twin))


if __name__ == "__main__":
    unittest.main(verbosity=2)
