#!/usr/bin/env python3
"""
Palli Sahayak Dosage Guard
==========================
Enforces the study restriction that the system never provides specific medication
dosages, while still answering general drug questions.

Rationale (ADR-0004):
    The grant proposal states the system "never provides specific medication dosages"
    and the study protocol (SS7.7) requires that this restriction be *enforced and
    tested* before any participant uses the build.  The boundary is drawn at the
    *dose*, not at the OTC/prescription distinction, because:

    1. "OTC" is not a safety property for this cohort.  Participants have advanced
       kidney and liver disease, frailty and dementia, where a freely available
       medicine can still be unsafe.
    2. There is no Rx/OTC classification data in this repository to test against.
    3. A dose is a dose regardless of how it was purchased.  "Paracetamol 500mg once
       daily" breaches the promise exactly as a morphine instruction does.
    4. A dose boundary is machine-checkable, which SS7.7 requires.

Answered freely (tier 1):
    What a medicine is, what class it belongs to, what it treats, general
    precautions, what symptoms warrant a doctor, non-drug management, and how to
    take a medicine a prescriber has already supplied.

Refused and deferred (tier 2):
    Any specific dose, strength, titration step, escalation schedule,
    drug-interaction judgement, or start/stop decision.

Severity ordering:
    An emergency always wins.  A refusal must never swallow a myocardial
    infarction.  When a query is both a dosage question and an emergency trigger,
    the emergency path runs and the deferral never fires.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dose detection
# ---------------------------------------------------------------------------

# Numeric dose with a unit. Covers SI units, common abbreviations and the
# Indian pharmaceutical conventions (lakh, crore are not doses but appear in
# narrative text and are handled separately).
_UNIT_ALTERNATION = (
    r"(?:mg|mcg|µg|ug|gm|g|ml|mL|IU|cc|%|units?)"
)

# Quantities expressed as fractions, which are doses even without a unit.
_FRACTION_DOSE = (
    r"(?:half|quarter|third|one|a|an|\d+(?:\.\d+)?)\s+"
    r"(?:tablet|tab|tablets|capsule|cap|capsules|pill|pills|"
    r"dose|drop|drops|spoonful|teaspoon|tsp|tablespoon|tbsp|syrup|sachet|satchet)"
)

# Frequency and timing instructions.
_FREQUENCY = (
    r"(?:once|twice|thrice|four\s+times|three\s+times|four\s+times)\s+"
    r"(?:a\s+|per\s+|every\s+)?(?:day|daily|daily\b|night|morning|evening|"
    r"bedtime|meals?|mealtime|week|hourly|hours)"
    r"|\bevery\s+\d+\s*(?:hours?|hrs?|minutes?|mins?|days?)\b"
    r"|\b\d+\s*times\s+(?:a|per)\s+(?:day|daily|week)\b"
    r"|\bprn\b|\bas\s+needed\b|\bstat\b|\bod\b|\bpo\b|\bIV\b|\bIM\b|\bSC\b"
)

# Titration and escalation language.
_TITRATION = (
    r"\btitrate\b|\btitrating\b|\bstep\s*up\b|\bstep\s*down\b"
    r"|\bescalate\b|\bincrease\s+the\s+(?:dose|dosage)\b"
    r"|\bdecrease\s+the\s+(?:dose|dosage)\b|\bstart\s+at\b.*\b(?:mg|ml)\b"
    r"|\bmaximum\s+dose\b|\bmax\s+dose\b"
)

# Start / stop / switch decisions.
_START_STOP = (
    r"\bstart\s+(?:the\s+)?\w+\s+(?:at\s+)?\d"
    r"|\bstop\s+(?:the\s+)?\w+\b|\bdiscontinue\b"
    r"|\bswitch\s+to\b|\bsubstitute\s+with\b"
)

# Dose-shaped patterns, evaluated in order.  UNIT is the most reliable signal and
# is checked first so that the reported reason is the strongest one available.
_DOSE_PATTERNS: List[Tuple[str, str]] = [
    ("numeric_dose", rf"\b\d+(?:\.\d+)?\s*{_UNIT_ALTERNATION}\b"),
    ("fraction_dose", rf"\b{_FRACTION_DOSE}\b"),
    ("frequency", _FREQUENCY),
    ("titration", _TITRATION),
    ("start_stop", _START_STOP),
    # Devanagari and other Indic numerals are not matched by \d, so normalise them
    # before the numeric pattern is applied.  See _normalise_indic_digits.
]

_INDIC_DIGIT_MAP = str.maketrans(
    {
        "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
        "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
        "౦": "0", "౧": "1", "౨": "2", "౩": "3", "౪": "4",
        "౫": "5", "౬": "6", "౭": "7", "౮": "8", "౯": "9",
        "೦": "0", "೧": "1", "೨": "2", "೩": "3", "೪": "4",
        "೫": "5", "೬": "6", "೭": "7", "೮": "8", "೯": "9",
        "൦": "0", "൧": "1", "൨": "2", "൩": "3", "൪": "4",
        "൫": "5", "൬": "6", "൭": "7", "൮": "8", "൯": "9",
        "੦": "0", "੧": "1", "੨": "2", "੩": "3", "੪": "4",
        "੫": "5", "੬": "6", "੭": "7", "੮": "8", "੯": "9",
        "૦": "0", "૧": "1", "૨": "2", "૩": "3", "૪": "4",
        "૫": "5", "૬": "6", "૭": "7", "૮": "8", "૯": "9",
        "௦": "0", "௧": "1", "௨": "2", "௩": "3", "௪": "4",
        "௫": "5", "௬": "6", "௭": "7", "௮": "8", "௯": "9",
        "੦": "0", "੧": "1", "੨": "2", "੩": "3", "੪": "4",
        "੫": "5", "੬": "6", "੭": "7", "੮": "8", "੯": "9",
        "০": "0", "১": "1", "২": "2", "৩": "3", "৪": "4",
        "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9",
        "౦": "0",
    }
)

# Indic and English unit words, so that a dose written entirely in a regional
# language is still caught.
_INDIC_UNITS = (
    r"(?:\u092e\u093f\.?\s?\u0917\u094d\u0930\u093e\.?|\u092e\u093f\u0932\u0940\u0917\u094d\u0930\u093e|\u0917\u094d\u0930\u093e\u092e|\u092e\u093f\u0932\u0940)"
    r"|(?:\u0cae\u0cbf\.?\s?\u0c97\u0ccd\u0cb0\u0cbe\.?|\u0c97\u0ccd\u0cb0\u0cbe\u0c82|\u0cae\u0cbf\u0cb2\u0c80)"
    r"|(?:\u0bae\u0bbf\.?\s?\u0b95\u0bbf\.?|\u0bae\u0bbf\u0bb2\u0bbf\.?\s?\u0b95\u0bbf\u0bb0\u0bbe\u0bae|\u0bae\u0bbf\u0bb2\u0bbf\u0bb2\u0bbf\u0b9f\u0bb0)"
    r"|(?:\u0c2e\u0c3f\.?\s?\u0c17\u0c4d\u0c30\u0c3e\.?|\u0c17\u0c4d\u0c30\u0c3e\u0c2e\u0c4d|\u0c2e\u0c3f\u0c32\u0c40)"
    r"|(?:\u092e\u093f\u0917\u094d\u0930\u093e|\u0a17\u0a4d\u0a30\u0cbe\u0a2e|\u0a2e\u0a3f\u0a32\u0a40)"
    r"|(?:\u0aae\u0abf\u0a97\u0acd\u0ab0\u0abe|\u0a17\u0acd\u0ab0\u0abe\u0a2e|\u0a2e\u0abf\u0a32\u0a40)"
    r"|(?:\u0aae\u0abf\u0ab2\u0acd\u0aaf\u0acd\u0ab0\u0abe\u0a2e|\u0aae\u0abf\u0a32\u0ab0\u0a3f\u0a1f\u0ab0)"
    r"|(?:\u09ae\u09bf\u09b2\u09bf\u0997\u09cd\u09b0\u09be\u09ae|\u0997\u09cd\u0a30\u09be\u09ae|\u09ae\u09bf\u09b2\u09bf)"
    r"|(?:\u0e21\u0e34\u0e25\u0e25\u0e34\u0e01\u0e23\u0e31\u0e21|\u0e21\u0e34\u0e25\u0e25\u0e34\u0e25\u0e34\u0e15\u0e23)"
)

_INDIC_FREQUENCY = (
    r"(?:दिन|रोज़|रोज|दिनों|सुबह|शाम|रात|बार)"
    r"|(?:ದಿನ|ದೀನ|ಬಾರ)"
    r"|(?:தினம்|தின|நாள்|வாரம்|முறை)"
    r"|(?:రోజు|రోజ|తరువాత|సార్లు)"
    r"|(?:ਰੋਜ਼|ਰੋਜ|ਵਾਰ)"
    r"|(?:રોજ|દિવસ|વાર)"
    r"|(?:ଦିନ|�ରୋଜ)"
)

_COMPILED: List[Tuple[str, re.Pattern]] = [
    ("numeric_dose", re.compile(
        rf"\d+(?:\.\d+)?\s*(?:{_UNIT_ALTERNATION}|{_INDIC_UNITS})", re.I
    )),
    ("fraction_dose", re.compile(rf"\b{_FRACTION_DOSE}\b", re.I)),
    ("frequency", re.compile(rf"(?:{_FREQUENCY})|(?:{_INDIC_FREQUENCY})", re.I)),
    ("titration", re.compile(_TITRATION, re.I)),
    ("start_stop", re.compile(_START_STOP, re.I)),
]

# Phrases that contain a numeric token but are not doses.  Without these, prose
# such as "Chapter 5" or "2 hours of care" would trip the guard.
_FALSE_POSITIVE_PATTERNS = [
    re.compile(r"\b\d+\s*(?:chapter|section|page|step|grade|level|site|site visit)s?\b", re.I),
    re.compile(r"\b(?:age|aged)\s*\d+\b", re.I),
    re.compile(r"\b\d+\s*(?:mm|cm)\s*(?:diameter|width|length|depth)\b", re.I),
]


def _normalise_indic_digits(text: str) -> str:
    """Map Indic and Eastern-Arabic numerals to ASCII so \\b\\d+ can match them."""
    return text.translate(_INDIC_DIGIT_MAP)


def _is_false_positive(sentence: str) -> bool:
    return any(p.search(sentence) for p in _FALSE_POSITIVE_PATTERNS)


@dataclass
class DoseFinding:
    """A single dose-shaped span found in a response."""
    reason: str
    snippet: str
    pattern: str

    def to_dict(self) -> Dict[str, str]:
        return {"reason": self.reason, "snippet": self.snippet, "pattern": self.pattern}


def detect_doses(text: str) -> List[DoseFinding]:
    """
    Find dose-shaped spans in `text`.

    Returns findings ordered by pattern reliability. An empty list means the text
    is clear of specific dosage instructions.
    """
    if not text:
        return []

    normalised = _normalise_indic_digits(text)
    findings: List[DoseFinding] = []

    for reason, pattern in _COMPILED:
        for match in pattern.finditer(normalised):
            start, end = match.span()
            snippet = normalised[max(0, start - 40):min(len(normalised), end + 40)].strip()
            sentence = _sentence_around(normalised, start, end)
            if _is_false_positive(sentence):
                continue
            findings.append(
                DoseFinding(reason=reason, snippet=snippet, pattern=match.group(0))
            )

    return _dedupe(findings)


def _sentence_around(text: str, start: int, end: int) -> str:
    left = max(text.rfind(".", 0, start), text.rfind("\n", 0, start))
    right_candidates = [i for i in (text.find(".", end), text.find("\n", end)) if i != -1]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left + 1:right].strip()


def _dedupe(findings: List[DoseFinding]) -> List[DoseFinding]:
    """Drop findings whose span is already covered by an earlier, stronger pattern."""
    seen: set = set()
    out: List[DoseFinding] = []
    for f in findings:
        key = f.pattern.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


# ---------------------------------------------------------------------------
# Deferral copy
# ---------------------------------------------------------------------------

# The deferral must never imply that a human has been contacted. Protocol SS7.6
# forbids implying a transfer has succeeded before receipt is confirmed, and no
# notification path exists yet.  The wording below is therefore about the advice,
# not about a dispatch that the system cannot perform.
_REFUSAL_EN = (
    "I understand this is what you need right now, and I am not able to help with "
    "the specific quantity or schedule of a medicine — deciding that needs someone who "
    "can see the patient.\n\n"
    "What I can do is help you prepare for that conversation:\n"
    "• Write down the symptoms, when they started, and what has already been given\n"
    "• Bring the medicine packets or prescription along\n"
    "• Raise it at the next consultation — with the palliative care doctor, the primary "
    "health centre, or the OPD you are registered with\n\n"
    "If the person is in severe distress right now — struggling to breathe, unresponsive, "
    "or in pain that cannot be settled — call 108 for an ambulance rather than waiting "
    "for the next visit."
)

_REFUSAL_LOCALIZED = {
    "hi": (
        "मैं समझ सकता हूँ कि यह अभी आपको ज़रूरी है। लेकिन किसी दवा की खास मात्रा या समय-सारणी "
        "मैं नहीं बता सकता — यह तय करने के लिए मरीज़ को देखने वाले चिकित्सक की ज़रूरत है।\n\n"
        "मैं इस बातचीत की तैयारी में मदद कर सकता हूँ:\n"
        "• लक्षण, उन्हें कब शुरू हुए, और अब तक क्या दिया गया — लिख लें\n"
        "• दवा के पाउच या प्रिस्क्रिप्शन साथ ले जाएँ\n"
        "• अगली बार डॉक्टर से मिलने पर यह बात रखें — पैलिएटिव केयर डॉक्टर, प्राथमिक स्वास्थ्य "
        "केंद्र, या जिस OPD में पंजीकरण है\n\n"
        "अगर मरीज़ को अभी बहुत तेज़ तकलीफ़ है — साँस लेने में दिक्कत, बेहोशी, या न ठीक होने "
        "वाला दर्द — तो अगली बार का इंतज़ार न करें, 108 पर एम्बुलेंस बुलाएँ।"
    ),
    "bn": (
        "আমি বুঝতে পারছি এটি এখন আপনার দরকার। তবে কোনো ওষুধের নির্দিষ্ট পরিমাণ বা সময়সূচি "
        "আমি বলতে পারি না — সেটা ঠিক করতে রোগীকে দেখেন এমন চিকিৎসক দরকার।\n\n"
        "আমি এই কথোপকথনের প্রস্তুতিতে সাহায্য করতে পারি:\n"
        "• লক্ষণ, কখন শুরু হয়েছে, এখন পর্যন্ত কী দেওয়া হয়েছে — লিখে রাখুন\n"
        "• ওষুধের প্যাকেট বা প্রেসক্রিপশন সঙ্গে নিন\n"
        "• পরবর্তী সাক্ষাতে ডাক্তারের সঙ্গে এ বিষয়ে কথা বলুন — প্যালিয়েটিভ কেয়ার ডাক্তার, "
        "প্রাথমিক স্বাস্থ্য কেন্দ্র, বা আপনার নিবন্ধিত OPD\n\n"
        "রোগী এখন খুব কষ্টে থাকলে — শ্বাসকষ্ট, অচেতন, বা সামলানো যায় না এমন ব্যথা — "
        "পরের দিনের জন্য অপেক্ষা না করে 108 নম্বরে অ্যাম্বুলেন্স ডাকুন।"
    ),
    "ta": (
        "இது இப்போது உங்களுக்கு தேவை என்று புரிகிறது. ஆனால் மருந்தின் குறிப்பிட்ட அளவு அல்லது "
        "நேரத்தை நான் தெரிவிக்க முடியாது — அதைத் தீர்மானிக்க நோயாளியைப் பார்க்கும் மருத்துவர் தேவை.\n\n"
        "இந்தப் பேச்சுக்கு தயார்படுத உதவலாம்:\n"
        "• அறிகுறிகள், எப்போது தொடங்கியது, இதுவரை என்ன கொடுக்கப்பட்டது — எழுதி வையுங்கள்\n"
        "• மருந்துப் பொதி அல்லது மருந்துச்சான்று எடுத்துச் செல்லுங்கள்\n"
        "• அடுத்த சந்திப்பில் மருத்துவரிடம் இதை முன்வைக்கவும் — பாலியேட்டிவ் கேர் மருத்துவர், "
        "முதல்நிலை சுகாதார மையம், அல்லது நீங்கள் பதிவு செய்த OPD\n\n"
        "நோயாளர் இப்போது மிகவும் வேதனையில் இருந்தால் — மூச்சுத் திணறல், நினைவிழப்பு, அல்லது "
        "சமாளிக்க முடியாத வலி — அடுத்த விஜயத்தை காத்திருக்காமல் 108-ல் ஆம்புலன்ஸை அழைக்கவும்."
    ),
}

# Languages the guard degrades to English for rather than refusing to answer.
# Sarvam Bulbul v3 covers 11 languages; the guard carries translated copy for the
# highest-volume ones and falls back safely for the rest.
_FALLBACK_LANGUAGE = "en"


def _refusal_for(language: str) -> str:
    """Return the deferral copy for `language`, falling back to English."""
    if not language:
        return _REFUSAL_EN
    primary = language.split("-")[0].split("_")[0].lower()
    return _REFUSAL_LOCALIZED.get(primary, _REFUSAL_EN)


def _redaction_note(language: str) -> str:
    """Short explanation appended after a partial redaction."""
    primary = (language or "").split("-")[0].split("_")[0].lower()
    return _REDACTION_NOTE.get(primary, _REDACTION_NOTE["en"])


def _is_critical_alert(alert: Any) -> bool:
    """
    Only a CRITICAL alert may override the dosage restriction.

    The emergency detector also fires at HIGH severity, and "severe pain" is one
    of its HIGH keywords. Allowing HIGH to disable the guard would let any dose
    question phrased as severe pain bypass the restriction entirely, which is a
    far more common phrasing than a genuine life-threatening emergency.
    """
    if alert is None:
        return False
    level = getattr(alert, "level", None)
    level_value = getattr(level, "value", level)
    if isinstance(level_value, str) and level_value.lower() in (
        "critical",
        "life_threatening",
    ):
        return True
    return bool(getattr(alert, "contact_emergency_services", False))




# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

# Explicit instructions to summon urgent help. Used only as a fallback when no
# emergency detector is injected, and matched tightly on purpose: requiring the
# imperative plus the vehicle/number prevents ordinary prose such as "in an
# emergency seek help" from switching the guard off.
_EMERGENCY_INSTRUCTION_RE = re.compile(
    r"(?:call|dial|phone|ring)\s*(?:\s*an?)?\s*"
    r"(?:108|102|ambulance|emergency\s+(?:service|services|number))"
    r"|\bambulance\s+(?:now|immediately)\b"
    r"|\bemergency\s+(?:services|help)\s+(?:now|immediately)\b",
    re.I,
)


@dataclass
class DosageGuardResult:
    """Outcome of applying the guard to a single response."""
    blocked: bool
    response: str
    findings: List[DoseFinding] = field(default_factory=list)
    reason: Optional[str] = None
    emergency_overrode: bool = False
    redacted: bool = False
    removed_units: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blocked": self.blocked,
            "reason": self.reason,
            "findings": [f.to_dict() for f in self.findings],
            "emergency_overrode": self.emergency_overrode,
            "redacted": self.redacted,
            "removed_units": self.removed_units,
        }


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

# Shortest remainder worth keeping. Roughly one and a half sentences: below this
# the survivor is an orphaned clause and a plain deferral reads better. Set by
# measurement, not taste - two clean sentences of real clinical content run to
# roughly 110 characters, which a higher threshold was discarding.
_MIN_REMAINING_CHARS = 80

# Appended when specific dosing was removed from an otherwise useful answer, so
# the user is not left wondering what the silence means.
_REDACTION_NOTE = {
    "en": "\n\n_(The specific amount and schedule of any medicine are not included here. Take that part of your question to your next consultation.)_",
    "hi": "\n\n_(किसी भी दवा की खास मात्रा और समय-सारणी यहाँ नहीं दी गई है। यह सवाल अपनी अगली डॉक्टर की विज़िट में पूछें।)_",
    "bn": "\n\n_(ওষুধের নির্দিষ্ট পরিমাণ ও সময়সূচি এখানে দেওয়া হয়নি। এই প্রশ্নটি আপনার পরবর্তী ডাক্তারের সাক্ষাতে করুন।)_",
    "ta": "\n\n_(எந்த மருந்தின் குறிப்பிட்ட அளவு மற்றும் நேரம் இங்கே தரப்படவில்லை. இந்தக் கேள்வியை அடுத்த மருத்துவர் சந்திப்பில் கேளுங்கள்.)_",
}

# A trailing colon marks a heading whose contents may all be removed.
_HEADING_RE = re.compile(r"^[\s#*\-–—]*[^\n:]{0,60}:\s*$")

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।!?])\s+")


def _split_units(text: str) -> List[str]:
    """
    Split a response into independently removable units.

    Every line is sentence-split, so that redacting one clause does not discard
    the rest of the line. A list marker ("1.", "-", "\u2022") is re-attached to
    the first sentence of its line so that removing a numbered item does not
    leave an orphaned "1." in the output.
    """
    units: List[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            units.append("")
            continue
        marker_match = re.match(r"^((?:[\-\u2022\*]|\d+[.)])\s+)", stripped)
        marker = marker_match.group(1) if marker_match else ""
        body = stripped[len(marker):] if marker else stripped
        sentences = [s for s in _SENTENCE_SPLIT_RE.split(body) if s.strip()]
        if not sentences:
            units.append(stripped)
            continue
        units.append(marker + sentences[0].strip())
        units.extend(sent.strip() for sent in sentences[1:])
    return units


def _drop_orphan_headings(units: List[str]) -> List[str]:
    """
    Remove a heading whose entire block beneath it was redacted.

    A heading is orphaned when the next non-blank unit is either another heading
    or nothing at all, which means everything the heading introduced is gone.
    """
    out = list(units)
    i = 0
    while i < len(out):
        if out[i] and _HEADING_RE.match(out[i]):
            j = i + 1
            while j < len(out) and out[j] == "":
                j += 1
            block_gone = j >= len(out) or (
                bool(out[j]) and _HEADING_RE.match(out[j])
            )
            if block_gone:
                del out[i:j]
                continue
        i += 1
    return out


def redact_doses(response: str) -> Tuple[str, List[DoseFinding], int]:
    """
    Remove dose-bearing units from `response`, keeping the remainder.

    Returns the redacted text, the findings that triggered redaction, and the
    number of units removed.
    """
    units = _split_units(response)
    kept: List[str] = []
    findings: List[DoseFinding] = []
    removed = 0

    for unit in units:
        if not unit:
            kept.append(unit)
            continue
        unit_findings = detect_doses(unit)
        if unit_findings:
            findings.extend(unit_findings)
            removed += 1
            continue
        kept.append(unit)

    kept = _drop_orphan_headings(kept)

    # Collapse runs of blank lines left behind by removal.
    collapsed: List[str] = []
    for unit in kept:
        if not unit and collapsed and not collapsed[-1]:
            continue
        collapsed.append(unit)

    text = "\n".join(collapsed).strip()
    return text, _dedupe(findings), removed


class DosageGuard:
    """
    Applies the dosage restriction to generated responses.

    The guard is deliberately a *post-generation* filter rather than a prompt
    instruction.  Protocol SS7.7 states that "a prompt instruction or a successful
    connection test alone does not satisfy these checks", so the restriction is
    enforced on the response text itself and is independently testable.
    """

    def __init__(self, emergency_system: Optional[object] = None):
        # Injected rather than imported to avoid a circular dependency on
        # safety_enhancements, which imports this module.
        self.emergency_system = emergency_system

    def apply(
        self,
        response: str,
        language: str = "en",
        query: str = "",
    ) -> DosageGuardResult:
        """
        Filter `response`, returning either the original text or the deferral.

        If the query is an emergency trigger the response is passed through
        untouched and `emergency_overrode` is set, because suppressing an urgent
        instruction in order to enforce a style restriction is the worse failure.
        """
        if self._is_emergency(query, response):
            return DosageGuardResult(
                blocked=False,
                response=response,
                emergency_overrode=True,
            )

        findings = detect_doses(response)
        if not findings:
            return DosageGuardResult(blocked=False, response=response)

        logger.warning(
            "Dosage restriction applied (reason=%s, findings=%d, lang=%s)",
            findings[0].reason,
            len(findings),
            language,
        )

        # Attempt sentence-level redaction first, so that a tier-1 question is not
        # answered with a tier-2 refusal when only part of the answer was dosing.
        redacted, redacted_findings, removed = redact_doses(response)
        if redacted and len(redacted) >= _MIN_REMAINING_CHARS:
            # Safety net: re-run the detector over the redaction. Redaction must
            # never be the thing that lets a dose through, so if anything still
            # matches we fall back to refusing the whole answer.
            residue = detect_doses(redacted)
            if not residue:
                return DosageGuardResult(
                    blocked=True,
                    response=redacted + _redaction_note(language),
                    findings=_dedupe(findings),
                    reason=findings[0].reason,
                    redacted=True,
                    removed_units=removed,
                )
            logger.warning(
                "Redaction left %d dose match(es); escalating to full refusal",
                len(residue),
            )

        return DosageGuardResult(
            blocked=True,
            response=_refusal_for(language),
            findings=_dedupe(findings),
            reason=findings[0].reason,
            removed_units=removed,
        )

    def _is_emergency(self, query: str, response: str) -> bool:
        """
        Decide whether the emergency path takes precedence over the deferral.

        The detector is authoritative. The response-text fallback is deliberately
        narrow: it matches an explicit instruction to summon help, never the bare
        word "emergency". A loose match here would silently disable the entire
        dosage restriction on any answer that merely mentions emergencies, which
        is a failure mode that must not exist in a safety control.
        """
        if self.emergency_system is not None:
            try:
                alert = self.emergency_system.detect_emergency(
                    query, "dosage-guard", "en"
                )
            except Exception:  # pragma: no cover - must never break a response
                logger.warning(
                    "Emergency detector failed inside DosageGuard; "
                    "falling back to response-text check",
                    exc_info=True,
                )
            else:
                if _is_critical_alert(alert):
                    return True

        return bool(_EMERGENCY_INSTRUCTION_RE.search(response or ""))


_dosage_guard: Optional[DosageGuard] = None


def get_dosage_guard(emergency_system: Optional[object] = None) -> DosageGuard:
    """Get or create the DosageGuard singleton."""
    global _dosage_guard
    if _dosage_guard is None:
        _dosage_guard = DosageGuard(emergency_system=emergency_system)
    return _dosage_guard