#!/usr/bin/env python3
"""
Palli Sahayak Dose Detector
===========================
The one place in the server that knows what a medication dose looks like.

ADR 0004 draws the Dose Boundary at the dose. Every check that asks "does this
text state a dose?" calls `detect_doses` here: the Dose Boundary's gate
(`dosage_guard.DosageGuard.apply`), the streaming gate, graph-text redaction,
the clinical validator and the observation extractors. No other module may
define a dose pattern; `tests/test_dose_boundary_shapes.py` fails if one does.
Round 3 found four drifted copies, and every copy missed something the others
caught, so a second copy is treated as a defect rather than a convenience.

What counts as a dose (one reason per family, reported in the safety record):

    numeric_dose   a number, in digits or in English or Hindi words, followed by
                   a unit: abbreviated ("mg", "mcg"), spelled out ("milligrams",
                   "microgrammes") or the unit word of any Supported Language in
                   its own script ("मिलीग्राम", "এমজি"-style abbreviations, ...)
    fraction_dose  a count of a dose form: "half a tablet", "two tablets",
                   "2 puffs", "३ गोली"
    frequency      a schedule: "twice daily", "every four hours", "4-hourly",
                   "q4h", "BD", and the Indic day/time words
    titration      dose changes: "titrate", "step up", "maximum dose"
    start_stop     start, stop and switch decisions

A few phrases contain a number without being a dose ("aged 70", "chapter 5").
Each exception covers only its own span: a dose elsewhere in the same sentence
is still a dose. Round 3 found that a sentence-wide exception let
"For patients aged 70, give morphine 10 mg every 4 hours." through untouched.

Python's `\\d` already matches every Indic digit (Devanagari, Bengali, Gurmukhi,
Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam), so no digit normalisation
is needed and match offsets refer to the caller's own text.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "DOSE_REASONS",
    "FRACTION_DOSE",
    "FREQUENCY",
    "NUMERIC_DOSE",
    "START_STOP",
    "TITRATION",
    "UNIT_WORDS",
    "DoseFinding",
    "contains_dose",
    "dedupe_findings",
    "detect_doses",
]

# ---------------------------------------------------------------------------
# Reasons, strongest first. The order is the order findings are reported in, so
# the first finding's reason is the strongest signal available. The names are
# stored in study rows; do not rename them.
# ---------------------------------------------------------------------------

NUMERIC_DOSE = "numeric_dose"
FRACTION_DOSE = "fraction_dose"
FREQUENCY = "frequency"
TITRATION = "titration"
START_STOP = "start_stop"

DOSE_REASONS: Tuple[str, ...] = (
    NUMERIC_DOSE,
    FRACTION_DOSE,
    FREQUENCY,
    TITRATION,
    START_STOP,
)

# ---------------------------------------------------------------------------
# Vocabulary. Plain words, one script per literal, so that a mixed-script or
# corrupted entry is visible in review and caught by the test suite.
# ---------------------------------------------------------------------------

_EN_NUMBER_WORDS: Tuple[str, ...] = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty",
    "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand",
    "half", "quarter", "dozen",
)

_HI_NUMBER_WORDS: Tuple[str, ...] = (
    "एक", "दो", "तीन", "चार", "पाँच", "पांच", "छह", "छः", "सात", "आठ", "नौ",
    "दस", "ग्यारह", "बारह", "पंद्रह", "बीस", "पच्चीस", "तीस", "चालीस", "पचास",
    "सौ", "आधा", "आधी", "डेढ़", "ढाई",
)

# Latin-script units, longest first so a spelled-out unit is reported whole.
_LATIN_UNITS: Tuple[str, ...] = (
    r"microgrammes?", r"micrograms?", r"milligrammes?", r"milligrams?",
    r"millilitres?", r"milliliters?", r"grammes?", r"grams?",
    r"international\s+units?", r"units?",
    r"mcg", r"µg", r"μg", r"ug", r"mg", r"gm", r"g", r"ml", r"iu", r"cc", r"gy",
)

# The unit words of every Supported Language, in that language's own script.
# Abbreviations with full stops are written as they appear in text; the regex
# builder lets a space follow each full stop.
UNIT_WORDS: Dict[str, Tuple[str, ...]] = {
    # Hindi and Marathi share Devanagari. "एमजी" and "एम.जी." are the
    # transliterated "mg" that Hindi answers use as often as "मि.ग्रा.".
    "hi": (
        "माइक्रोग्राम", "मिलीग्राम", "मिलिग्राम", "मिग्रा", "मि.ग्रा.", "ग्राम",
        "मिलीलीटर", "मिलिलीटर", "मि.ली.", "मिली", "एमजी", "एम.जी.", "एम जी",
        "एमएल", "एम.एल.", "एमसीजी",
    ),
    "mr": (
        "मायक्रोग्रॅम", "मिलिग्रॅम", "मि.ग्रॅ.", "ग्रॅम", "मिलिलिटर", "मि.लि.",
    ),
    "bn": (
        "মাইক্রোগ্রাম", "মিলিগ্রাম", "মি.গ্রা.", "গ্রাম", "মিলিলিটার", "মি.লি.",
        "মিলি", "এমজি", "এম.জি.",
    ),
    "pa": (
        "ਮਾਈਕ੍ਰੋਗ੍ਰਾਮ", "ਮਿਲੀਗ੍ਰਾਮ", "ਮਿ.ਗ੍ਰਾ.", "ਮਿਗ੍ਰਾ", "ਗ੍ਰਾਮ", "ਮਿਲੀਲੀਟਰ",
        "ਮਿ.ਲੀ.", "ਮਿਲੀ", "ਐਮਜੀ", "ਐਮ.ਜੀ.",
    ),
    "gu": (
        "માઇક્રોગ્રામ", "મિલિગ્રામ", "મિલીગ્રામ", "મિ.ગ્રા.", "ગ્રામ", "મિલિલિટર",
        "મિ.લી.", "મિલી", "એમજી", "એમ.જી.",
    ),
    "or": (
        "ମାଇକ୍ରୋଗ୍ରାମ", "ମିଲିଗ୍ରାମ", "ମି.ଗ୍ରା.", "ମିଗ୍ରା", "ଗ୍ରାମ", "ମିଲିଲିଟର",
        "ମି.ଲି.", "ମିଲି",
    ),
    "ta": (
        "மைக்ரோகிராம்", "மில்லிகிராம்", "மி.கி.", "கிராம்", "மில்லிலிட்டர்",
        "மி.லி.", "மில்லி",
    ),
    "te": (
        "మైక్రోగ్రాము", "మిల్లీగ్రాము", "మిల్లీగ్రామ్", "మి.గ్రా.", "గ్రాము",
        "గ్రామ్", "మిల్లీలీటర్", "మి.లీ.", "మిల్లీ",
    ),
    "kn": (
        "ಮೈಕ್ರೋಗ್ರಾಂ", "ಮಿಲಿಗ್ರಾಂ", "ಮಿಲಿಗ್ರಾಮ್", "ಮಿ.ಗ್ರಾ.", "ಗ್ರಾಂ", "ಗ್ರಾಮ್",
        "ಮಿಲಿಲೀಟರ್", "ಮಿ.ಲೀ.", "ಮಿಲಿ",
    ),
    "ml": (
        "മൈക്രോഗ്രാം", "മില്ലിഗ്രാം", "മി.ഗ്രാ.", "ഗ്രാം", "മില്ലിലിറ്റർ",
        "മി.ലി.", "മില്ലി",
    ),
    # Thai is not a Supported Language. Its units stay because the Android
    # detector mirrors them and dropping a pattern never makes the boundary safer.
    "th": ("มิลลิกรัม", "มิลลิลิตร"),
}

# Counts of a dose form. Legacy forms may follow "a"/"an"/"half"; the others
# need a number, so that "it can be given as an injection" stays tier 1.
_COUNT_NOUNS_AFTER_ARTICLE: Tuple[str, ...] = (
    r"tablets?", r"tabs?", r"capsules?", r"caps?", r"pills?", r"doses?",
    r"drops?", r"spoonfuls?", r"teaspoons?(?:ful)?s?", r"tsp",
    r"tablespoons?(?:ful)?s?", r"tbsp", r"syrup", r"sachets?", r"satchets?",
)
_COUNT_NOUNS_AFTER_NUMBER: Tuple[str, ...] = _COUNT_NOUNS_AFTER_ARTICLE + (
    r"puffs?", r"sprays?", r"inhalations?", r"patch(?:es)?", r"ampoules?",
    r"ampules?", r"vials?", r"injections?", r"suppositor(?:y|ies)",
    r"lozenges?",
)
_HI_COUNT_NOUNS: Tuple[str, ...] = (
    "गोलियाँ", "गोलियां", "गोली", "टैबलेट", "टेबलेट", "कैप्सूल", "चम्मच",
    "बूँदें", "बूंदें", "बूँद", "बूंद", "पफ", "खुराक",
    # Plural and oblique forms (whole-word matching would otherwise miss them).
    "गोलियों", "बूँदों", "बूंदों", "खुराकें", "खुराकों",
)

# Day and time-of-day words that mark a schedule in Indic answers.
#
# The words are matched whole, so each inflected form a schedule uses is listed
# in its own right: before whole-word matching, "ಬಾರಿ" or "రోజుకు" were caught
# only because the stem "ಬಾರ" or "రోజ" sat inside them. The second line of each
# entry holds those forms (plural, oblique, locative and "every day" compounds).
_INDIC_FREQUENCY_WORDS: Dict[str, Tuple[str, ...]] = {
    "hi": ("दिनों", "दिन", "रोज़", "रोज", "सुबह", "शाम", "रात", "बार",
           "रोज़ाना", "रोजाना", "प्रतिदिन", "दिनभर", "रात्रि", "रातों"),
    "mr": ("रात्री", "रात्रीच्या", "दररोज"),
    "kn": ("ದಿನ", "ದೀನ", "ಬಾರ",
           "ಬಾರಿ", "ದಿನಕ್ಕೆ", "ದಿನವೂ", "ದಿನಗಳು", "ಪ್ರತಿದಿನ", "ದಿನಾ"),
    "ta": ("தினம்", "தின", "நாள்", "வாரம்", "முறை",
           "தினமும்", "தினசரி", "நாளுக்கு", "நாட்கள்", "முறைகள்",
           "வாரத்திற்கு", "வாரத்துக்கு"),
    "te": ("రోజు", "రోజ", "తరువాత", "సార్లు",
           "రోజుకు", "రోజూ", "రోజులు", "ప్రతిరోజు", "ప్రతిరోజూ", "సారి"),
    "pa": ("ਰੋਜ਼", "ਰੋਜ", "ਵਾਰ",
           "ਰੋਜ਼ਾਨਾ", "ਰੋਜਾਨਾ", "ਵਾਰੀ", "ਦਿਨ"),
    "gu": ("રોજ", "દિવસ", "વાર",
           "દરરોજ", "રોજના", "દિવસમાં", "દિવસે", "દિવસો"),
    "or": ("ଦିନ", "ରୋଜ",
           "ପ୍ରତିଦିନ", "ଦିନରେ", "ଦିନକୁ", "ଦିନେ"),
}

# ---------------------------------------------------------------------------
# Pattern assembly
# ---------------------------------------------------------------------------



def _letter_or_mark_class() -> str:
    """
    A regex character class of every letter and combining mark (Unicode L*, Mn,
    Mc) in the Latin and Indic blocks, plus the zero-width joiners that sit
    inside Indic words.

    Python's \\b cannot bound an Indic word: a vowel sign such as "ि" is a
    combining mark, not \\w, so "बार" inside "बारे" ends on a "boundary". Digits
    are left out on purpose, so "3गोली" still reads as a count.
    """
    chars = [chr(c) for c in range(0x0900, 0x0E00)
             if unicodedata.category(chr(c))[0] == "L"
             or unicodedata.category(chr(c)) in ("Mn", "Mc")]
    ranges: List[Tuple[str, str]] = []
    for ch in chars:
        if ranges and ord(ch) == ord(ranges[-1][1]) + 1:
            ranges[-1] = (ranges[-1][0], ch)
        else:
            ranges.append((ch, ch))
    body = "".join(a if a == b else f"{a}-{b}" for a, b in ranges)
    return f"[A-Za-z\u200c\u200d{body}]"


_LETTER_OR_MARK = _letter_or_mark_class()
_INDIC_START = rf"(?<!{_LETTER_OR_MARK})"
_INDIC_END = rf"(?!{_LETTER_OR_MARK})"


def _whole_words(words: Sequence[str]) -> str:
    """An alternation of Indic words that matches only where no letter or mark adjoins."""
    return _INDIC_START + _alternation(words) + _INDIC_END


def _alternation(words: Sequence[str]) -> str:
    """Join literal words into a regex alternation, longest first."""
    escaped = []
    for word in sorted(set(words), key=len, reverse=True):
        # A full stop in an abbreviation is often dropped or followed by a space
        # ("मि.ग्रा.", "मि. ग्रा.", "मि.ग्रा").
        escaped.append(re.escape(word).replace(r"\.", r"\.?\s?").replace(r"\ ", r"\s?"))
    return "(?:" + "|".join(escaped) + ")"


_EN_WORD = _alternation(_EN_NUMBER_WORDS)
_EN_NUMBER_PHRASE = rf"\b{_EN_WORD}(?:(?:\s+|-)(?:and\s+)?(?:a\s+)?{_EN_WORD})*\b"
_HI_NUMBER_PHRASE = _whole_words(_HI_NUMBER_WORDS)
# Not preceded by a letter, so "SpO2 drop" or "B12" does not start a dose.
_DIGITS = r"(?<![A-Za-z])\d+(?:[.,/]\d+)?"
_NUMBER = rf"(?:{_DIGITS}|{_EN_NUMBER_PHRASE}|{_HI_NUMBER_PHRASE})"

_LATIN_UNIT = "(?:" + "|".join(_LATIN_UNITS) + r")(?![A-Za-z])"
# Unit words are not closed on the right: they follow a number, which already
# anchors them, and they take suffixes (Telugu plural "-లు", Tamil case endings)
# that a closing boundary would cut off.
_INDIC_UNIT = _alternation([w for words in UNIT_WORDS.values() for w in words])

_NUMERIC_DOSE_RE = rf"{_NUMBER}\s*(?:{_LATIN_UNIT}|%|{_INDIC_UNIT})"

_COUNT_DOSE_RE = (
    r"\b(?:half|quarter|third|one|a|an)\s+(?:of\s+a\s+)?"
    r"(?:" + "|".join(_COUNT_NOUNS_AFTER_ARTICLE) + r")\b"
    rf"|{_NUMBER}\s+(?:of\s+a\s+)?"
    r"(?:" + "|".join(_COUNT_NOUNS_AFTER_NUMBER) + r")\b"
    rf"|(?:{_DIGITS}|{_HI_NUMBER_PHRASE})\s*{_whole_words(_HI_COUNT_NOUNS)}"
)

_INTERVAL = rf"(?:\d+(?:\s*(?:-|–|to)\s*\d+)?|{_EN_NUMBER_PHRASE})"
_FREQUENCY_RE = (
    r"\b(?:once|twice|thrice|three\s+times|four\s+times)\s+"
    r"(?:a\s+|per\s+|every\s+)?(?:day|daily|night|morning|evening|"
    # No trailing boundary: "twice weekly" must match as "twice week".
    r"bedtime|meals?|mealtime|week|hourly|hours)"
    rf"|\bevery\s+{_INTERVAL}\s*(?:hours?|hrs?|minutes?|mins?|days?)\b"
    rf"|\b{_INTERVAL}\s*times\s+(?:a|per|each)\s+(?:day|daily|week)\b"
    rf"|\b{_INTERVAL}\s*[-–]?\s*hourly\b"
    r"|\bq\s?\d+\s?h(?:rs?|ourly)?\b"
    r"|\bprn\b|\bas\s+needed\b|\bstat\b|\bod\b|\bpo\b|\bIV\b|\bIM\b|\bSC\b"
    r"|\bnocte\b|\bmane\b"
    # Upper case only: "bid" and "bd" are ordinary words in lower case.
    r"|(?-i:\b(?:BD|BID|TDS|TID|QDS|QID)\b)"
    r"|" + _whole_words([w for words in _INDIC_FREQUENCY_WORDS.values() for w in words])
)

_TITRATION_RE = (
    r"\btitrate\b|\btitrating\b|\bstep\s*up\b|\bstep\s*down\b"
    r"|\bescalate\b|\bincrease\s+the\s+(?:dose|dosage)\b"
    r"|\bdecrease\s+the\s+(?:dose|dosage)\b|\bstart\s+at\b.*\b(?:mg|ml)\b"
    r"|\bmaximum\s+dose\b|\bmax\s+dose\b"
)

_START_STOP_RE = (
    r"\bstart\s+(?:the\s+)?\w+\s+(?:at\s+)?\d"
    r"|\bstop\s+(?:the\s+)?\w+\b|\bdiscontinue\b"
    r"|\bswitch\s+to\b|\bsubstitute\s+with\b"
)

_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    (NUMERIC_DOSE, re.compile(_NUMERIC_DOSE_RE, re.I)),
    (FRACTION_DOSE, re.compile(_COUNT_DOSE_RE, re.I)),
    (FREQUENCY, re.compile(_FREQUENCY_RE, re.I)),
    (TITRATION, re.compile(_TITRATION_RE, re.I)),
    (START_STOP, re.compile(_START_STOP_RE, re.I)),
)

# Spans that contain a number but are not doses. A match is excused only where
# it overlaps one of these spans; the rest of the sentence is still checked.
_NOT_A_DOSE: Tuple["re.Pattern[str]", ...] = (
    re.compile(
        r"\b\d+\s*(?:chapter|section|page|step|grade|level|site visit|site)s?\b"
        r"|\b(?:chapter|section|page|step|grade|level|stage|class|type)\s*(?:\d+|[IV]+)[a-c]?\b",
        re.I,
    ),
    re.compile(rf"\b(?:age|aged)\s*(?:\d+|{_EN_NUMBER_PHRASE})", re.I),
    re.compile(r"\b\d+\s*(?:mm|cm)\s*(?:diameter|width|length|depth)\b", re.I),
)

_SNIPPET_CONTEXT = 40


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoseFinding:
    """
    One dose-shaped span found in a text.

    `pattern` is the matched text itself (the name is kept for the safety
    record, which already stores it). `start` and `end` index into the text
    that was passed to `detect_doses`.
    """

    reason: str
    snippet: str
    pattern: str
    start: int = -1
    end: int = -1

    def to_dict(self) -> Dict[str, str]:
        """The finding as stored in the safety record."""
        return {"reason": self.reason, "snippet": self.snippet, "pattern": self.pattern}


def detect_doses(text: Optional[str] = None) -> List[DoseFinding]:
    """
    Find every dose-shaped span in `text`.

    Accepts None and the empty string, returning no findings, because callers
    reach it from response handlers where a missing body is a normal state.

    Returns findings ordered by reason strength (see DOSE_REASONS), one per
    distinct matched text. An empty list means the text states no dose.
    """
    if not text:
        return []

    excused = [m.span() for p in _NOT_A_DOSE for m in p.finditer(text)]
    findings: List[DoseFinding] = []
    for reason, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < e and s < end for s, e in excused):
                continue
            snippet = text[max(0, start - _SNIPPET_CONTEXT):end + _SNIPPET_CONTEXT].strip()
            findings.append(
                DoseFinding(
                    reason=reason,
                    snippet=snippet,
                    pattern=match.group(0),
                    start=start,
                    end=end,
                )
            )
    return dedupe_findings(findings)


def contains_dose(text: Optional[str] = None) -> bool:
    """True when `text` states a dose in any shape `detect_doses` knows."""
    return bool(detect_doses(text))


def dedupe_findings(findings: Sequence[DoseFinding]) -> List[DoseFinding]:
    """Keep the first (strongest) finding for each distinct matched text."""
    seen: set = set()
    out: List[DoseFinding] = []
    for finding in findings:
        key = finding.pattern.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(finding)
    return out
