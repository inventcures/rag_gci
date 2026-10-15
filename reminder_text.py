#!/usr/bin/env python3
"""
Palli Sahayak Reminder Text
===========================
What a medication reminder may say: the medicine and the time, never a dose.

ADR 0004 says the system never gives a specific dose, and decision D10 reads
that strictly for reminders: a reminder does not read back even a dose the
participant or a prescriber entered, until the clinical lead decides otherwise.
The dose a person entered stays in the stored reminder; it is never spoken or
sent. Every reminder path (the voice call script, the WhatsApp scheduler and
commands, the mobile reminder route) builds its text through this module.

Dose shapes come from `dose_detector`, the one place that defines them.
"""

import logging
import re
from typing import Optional

from dose_detector import detect_doses

logger = logging.getLogger(__name__)

# Said in place of a medicine name that was nothing but a dose ("2 tablets").
GENERIC_MEDICINE = {
    "en": "your medicine",
    "hi": "अपनी दवा",
    "bn": "আপনার ওষুধ",
    "ta": "உங்கள் மருந்து",
    "gu": "તમારી દવા",
}

_EMPTY_BRACKETS_RE = re.compile(r"[\(\[\{]\s*[\)\]\}]")
_SPACE_RE = re.compile(r"[ \t]+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"[ \t]+([,.;:!?।])")
_REPEATED_PUNCT_RE = re.compile(r"([,;:])(?:\s*[,;:])+")
_EDGE_RE = re.compile(r"^[\s,;:\-–—/+]+|[\s,;:\-–—/+]+$")

# A dose found inside another dose's removal can expose a new match; a few
# passes always settle it, and the cap keeps a pathological text from looping.
_MAX_PASSES = 5


def _tidy(text: str) -> str:
    """Collapse the spacing and punctuation a removed span leaves behind."""
    text = _EMPTY_BRACKETS_RE.sub("", text)
    text = _SPACE_RE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _REPEATED_PUNCT_RE.sub(r"\1", text)
    lines = [_EDGE_RE.sub("", line) for line in text.split("\n")]
    return "\n".join(lines).strip()


def strip_doses(text: Optional[str]) -> str:
    """`text` with every dose-shaped span removed ("500mg after food" -> "after food")."""
    result = text or ""
    for _ in range(_MAX_PASSES):
        findings = [f for f in detect_doses(result) if 0 <= f.start < f.end]
        if not findings:
            return _tidy(result)
        for finding in sorted(findings, key=lambda f: f.start, reverse=True):
            result = result[:finding.start] + " " + result[finding.end:]
    # Still matching after the cap: say nothing rather than risk a dose.
    logger.warning("Reminder text kept matching a dose after %d passes; dropped it", _MAX_PASSES)
    return ""


def medicine_label(name: Optional[str], language: str = "en") -> str:
    """The medicine as a reminder may say it: its name with any dose removed."""
    label = strip_doses(name)
    return label or GENERIC_MEDICINE.get(language, GENERIC_MEDICINE["en"])


def ensure_dose_free(message: str) -> str:
    """
    The last check on a finished reminder: remove any dose that got through.

    The templates hold no dose and every inserted value is stripped first, so
    this should never change anything. If it does, a template or a caller has
    drifted, which is logged so it can be fixed.
    """
    if not detect_doses(message):
        return message
    logger.error("A reminder message still held a dose after templating; removing it")
    return strip_doses(message)
