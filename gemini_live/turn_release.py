"""
The Dose Boundary on what Live says, before the participant hears it.

Live speaks as it generates, so a dose in its reply would be heard before any
text check could run. The web voice socket therefore holds a turn's audio
until the turn completes, checks the transcript of what Live said, and only
then releases the audio. A turn that names a dose releases the guarded text
(a Redacted Answer or the Deferral) and never the audio; a turn whose speech
cannot be checked (audio with no transcript) releases the Deferral. This is
the same rule the WhatsApp Live path applies to a voice note
(whatsapp_bot._process_with_gemini_live, step 7).
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from dosage_guard import deferral_for

logger = logging.getLogger(__name__)


@dataclass
class TurnRelease:
    """What one Live turn may release to the participant."""

    audio: List[bytes] = field(default_factory=list)  # empty when withheld
    text: str = ""                    # the reply the participant may read
    withheld: bool = False            # Live's audio was not released
    dosage_blocked: bool = False      # the Dose Boundary redacted or deferred it
    # The safety record for the study row (guard_query_result's "safety").
    safety: Dict[str, Any] = field(default_factory=dict)


def check_live_turn(audio: List[bytes], spoken: str, heard: str, language: str) -> TurnRelease:
    """
    Pass one Live turn through the Dose Boundary.

    Args:
        audio: The turn's audio chunks, in order, held back so far.
        spoken: The transcript of what Live said in the turn.
        heard: The transcript of what the participant asked (the question the
            Emergency Override is detected on).
        language: The session's Supported Language.

    Returns:
        The audio to release (all of it, or none) and the text to show.
    """
    spoken = (spoken or "").strip()
    if not spoken:
        if not audio:
            return TurnRelease()
        logger.warning("Live turn has audio but no transcript; withholding it")
        return TurnRelease(text=deferral_for(language), withheld=True,
                           safety={"validation_status": "live_turn_unverifiable"})

    from safety_enhancements import guard_query_result

    check = {"status": "success", "answer": spoken, "sources": []}
    safety = guard_query_result(check, query=heard or "", language=language or "en")
    if safety is None:
        return TurnRelease(text=deferral_for(language), withheld=bool(audio),
                           safety={"validation_status": "live_turn_unverifiable"})
    if safety.dosage_blocked:
        logger.warning("Live turn redacted by the Dose Boundary (%s); audio withheld",
                       safety.dosage_reason)
        return TurnRelease(text=check["answer"], withheld=bool(audio), dosage_blocked=True,
                           safety=check["safety"])
    return TurnRelease(audio=list(audio), text=check["answer"], safety=check["safety"])
