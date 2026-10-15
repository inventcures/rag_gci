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
    the emergency path runs and the deferral never fires.  Only a CRITICAL
    alert on the question, passed in by the caller, invokes this Emergency
    Override; the answer's own words never do.  When it applies, the 108
    guidance in the participant's Supported Language is put first.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

import dose_detector
import supported_languages

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dose detection
# ---------------------------------------------------------------------------

# Every dose pattern lives in dose_detector. DoseFinding and detect_doses are
# re-exported here because callers and the safety record already import them
# from this module.
DoseFinding = dose_detector.DoseFinding
detect_doses = dose_detector.detect_doses
_dedupe = dose_detector.dedupe_findings


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


def _is_critical_alert(alert: Optional[Any]) -> bool:
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


def _alert_level(alert: Optional[Any]) -> str:
    """The alert's level as the safety record stores it, "none" without one."""
    if alert is None:
        return "none"
    level = getattr(alert, "level", None)
    level_value = getattr(level, "value", level)
    return str(level_value).lower() if level_value else "none"


# ---------------------------------------------------------------------------
# Emergency guidance
# ---------------------------------------------------------------------------

# Put first in the answer when the Emergency Override applies, so a caller in a
# life-threatening emergency hears where to get help before anything else. One
# sentence, with 108 in it, in every Supported Language. The en, hi, bn, ta and
# gu lines are the call-to-action of the emergency detector's own CRITICAL
# message (safety_enhancements.EmergencyDetectionSystem._create_alert), which
# reads them from here. The mr, te, kn, ml, pa and or lines are new and are
# listed for clinical review.
EMERGENCY_GUIDANCE: Dict[str, str] = {
    "en": "Call emergency services (108/102) immediately!",
    "hi": "तुरंत आपातकालीन सेवाओं (108/102) को कॉल करें!",
    "bn": "অবিলম্বে জরুরী সেবায় (108/102) কল করুন!",
    "ta": "உடனடியாக அவசர சேவைகளை (108/102) அழைக்கவும்!",
    "gu": "તાત્કાલિક આપત્તિ સેવાઓ (108/102) ને કૉલ કરો!",
    "mr": "ताबडतोब आपत्कालीन सेवांना (108/102) कॉल करा!",
    "te": "వెంటనే అత్యవసర సేవలకు (108/102) కాల్ చేయండి!",
    "kn": "ತಕ್ಷಣ ತುರ್ತು ಸೇವೆಗಳಿಗೆ (108/102) ಕರೆ ಮಾಡಿ!",
    "ml": "ഉടൻ തന്നെ അടിയന്തര സേവനങ്ങളെ (108/102) വിളിക്കുക!",
    "pa": "ਤੁਰੰਤ ਐਮਰਜੈਂਸੀ ਸੇਵਾਵਾਂ (108/102) ਨੂੰ ਕਾਲ ਕਰੋ!",
    "or": "ତୁରନ୍ତ ଜରୁରୀକାଳୀନ ସେବା (108/102) କୁ କଲ୍ କରନ୍ତୁ!",
}


def emergency_guidance_for(language: Optional[str]) -> str:
    """The 108 guidance in `language` (a code, tag or alias), English outside the eleven."""
    supported = supported_languages.get_language(language)
    code = supported.code if supported else supported_languages.ENGLISH_CODE
    return EMERGENCY_GUIDANCE.get(code, EMERGENCY_GUIDANCE["en"])


def _with_emergency_guidance(response: str, language: str) -> str:
    """`response` with the emergency guidance as its first sentence, added once."""
    guidance = emergency_guidance_for(language)
    if (response or "").lstrip().startswith(guidance):
        return response
    return f"{guidance}\n\n{response}" if response else guidance


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------


class EmergencyDetector(Protocol):
    """
    The slice of the emergency detector that DosageGuard depends on.

    Declared structurally rather than imported: safety_enhancements imports this
    module, so a direct import would be circular. Typing it as bare `object` and
    then calling a method on it would defeat the type checker, and the point of
    this module is that its dependency ordering is correct.
    """

    def detect_emergency(
        self, query: str, user_id: Optional[str] = ..., language: str = ...
    ) -> Optional[Any]: ...


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
    # The level of the alert the caller passed in ("none" without one), so the
    # safety record shows what the override decision was based on.
    emergency_level: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blocked": self.blocked,
            "reason": self.reason,
            "findings": [f.to_dict() for f in self.findings],
            "emergency_overrode": self.emergency_overrode,
            "emergency_level": self.emergency_level,
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

    def __init__(self, emergency_system: Optional[EmergencyDetector] = None):
        # Injected rather than imported to avoid a circular dependency on
        # safety_enhancements, which imports this module.
        self.emergency_system = emergency_system

    def apply(
        self,
        response: str,
        language: str = "en",
        query: str = "",
        alert: Optional[Any] = None,
    ) -> DosageGuardResult:
        """
        Filter `response`, returning either the original text or the deferral.

        `alert` is the emergency detector's alert on the question, computed once
        on the answer path and passed in (None when there was none). Only a
        CRITICAL alert invokes the Emergency Override: the response then keeps
        its text, the emergency guidance in `language` is put first, and
        `emergency_overrode` is set, because suppressing an urgent instruction
        in order to enforce the restriction is the worse failure. Nothing in the
        response text itself, such as "call 108", can invoke the override.

        `query` is the question the response answers; the override decision is
        made from `alert`, not from it.
        """
        level = _alert_level(alert)
        if _is_critical_alert(alert):
            return DosageGuardResult(
                blocked=False,
                response=_with_emergency_guidance(response, language),
                emergency_overrode=True,
                emergency_level=level,
            )

        findings = detect_doses(response)
        if not findings:
            return DosageGuardResult(blocked=False, response=response, emergency_level=level)

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
                    emergency_level=level,
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
            emergency_level=level,
        )

    def detect_alert(self, query: str, language: str = "en") -> Optional[Any]:
        """
        The emergency detector's alert on `query`, for a caller that has none yet.

        The streaming path calls this once per question and decides the override
        from the alert exactly as `apply` does. A detector failure returns None, which
        keeps the Dose Boundary in force: the safe direction for a failure.
        """
        if self.emergency_system is None or not query:
            return None
        try:
            return self.emergency_system.detect_emergency(query, None, language)
        except Exception:  # must never break a response
            logger.warning("Emergency detector failed; the Dose Boundary stays on",
                           exc_info=True)
            return None


_dosage_guard: Optional[DosageGuard] = None


def get_dosage_guard(emergency_system: Optional[EmergencyDetector] = None) -> DosageGuard:
    """Get or create the DosageGuard singleton."""
    global _dosage_guard
    if _dosage_guard is None:
        _dosage_guard = DosageGuard(emergency_system=emergency_system)
    return _dosage_guard