"""
Answers released sentence by sentence, each through the Dose Boundary.

A voice caller used to hear nothing until the whole answer was written. The
model now streams; `SentenceSplitter` cuts the stream into sentences (`.`,
`?`, `!`, `।`, and line ends), and `AnswerStream` releases each one only after
the dose detector passes it, alone and joined to the sentence before it. A
failing sentence is dropped. At the end, if anything was dropped, the
redaction note follows, or the Deferral when too little is left to stand as
an answer: Redacted Answer semantics, as `DosageGuard.apply` gives the whole
text. When the question raises a CRITICAL alert the Emergency Override
applies, as it does for the whole text: the 108 guidance in the caller's
language goes out first, then the sentences unfiltered.

A channel opts in by opening `streaming_to(stream)` around its `query()` call;
the answer generator streams into the active stream instead of waiting for
the whole answer. What the channel sent is `stream.text`.
"""

from __future__ import annotations

import contextvars
import re
import time
from contextlib import contextmanager
from typing import AsyncIterator, Awaitable, Callable, Iterator, List, Optional

import dosage_guard

# A sentence ends at . ? ! or the danda, followed by space (optionally after a
# closing quote or bracket), or at a line end.
_BOUNDARY = re.compile(r"(?<=[.!?।])([\"'”’)\]]*)\s+|\n+")

# Words whose trailing full stop does not end a sentence.
_ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "st", "vs", "etc", "e.g", "i.e", "approx",
                  "no", "fig", "inj", "tab", "cap", "syp"}
_LIST_MARKER = re.compile(r"^\s*(?:\d+|[a-z])[.)]$", re.IGNORECASE)


class SentenceSplitter:
    """Feed text deltas; get back the sentences they complete."""

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, delta: str) -> List[str]:
        self._buffer += delta
        out: List[str] = []
        start = 0
        for match in _BOUNDARY.finditer(self._buffer):
            piece = self._buffer[start:match.start() + len(match.group(1) or "")]
            if match.group(0).startswith("\n") or self._ends_sentence(piece):
                if piece.strip():
                    out.append(piece.strip())
                start = match.end()
        self._buffer = self._buffer[start:]
        return out

    def flush(self) -> List[str]:
        rest, self._buffer = self._buffer.strip(), ""
        return [rest] if rest else []

    @staticmethod
    def _ends_sentence(piece: str) -> bool:
        stripped = piece.strip()
        if not stripped:
            return False
        if _LIST_MARKER.match(stripped):
            return False
        last = stripped.split()[-1].rstrip(".").lower()
        return last not in _ABBREVIATIONS


class AnswerStream:
    """Releases guarded sentences to `send` as the model writes them."""

    def __init__(
        self,
        send: Callable[[str], Awaitable[None]],
        *,
        query: str = "",
        language: str = "en",
        guard: Optional[dosage_guard.DosageGuard] = None,
        max_words: Optional[int] = None,
    ):
        self._send = send
        self.language = language
        guard = guard or _default_guard()
        # The alert is computed once, on the question, as on the whole-text
        # path; only a CRITICAL one invokes the Emergency Override.
        self.alert = guard.detect_alert(query, language) if guard else None
        self.emergency = dosage_guard._is_critical_alert(self.alert)
        self.max_words = max_words
        self.sent: List[str] = []
        self.redacted = 0
        self.truncated = False
        self.used = False
        self.started = time.perf_counter()
        self.first_sent_s: Optional[float] = None
        self._previous = ""

    @property
    def text(self) -> str:
        """Everything sent so far, as one string."""
        return " ".join(self.sent)

    async def _release(self, sentence: str) -> None:
        if self.max_words is not None:
            words = sum(len(s.split()) for s in self.sent)
            if words + len(sentence.split()) > self.max_words:
                self.truncated = True
                return
        if self.emergency and not self.sent:
            # Under the override the caller hears the 108 guidance before
            # anything else, as `DosageGuard.apply` puts it first.
            guidance = dosage_guard.emergency_guidance_for(self.language)
            self.sent.append(guidance)
            await self._send(guidance)
        self.sent.append(sentence)
        if self.first_sent_s is None:
            self.first_sent_s = time.perf_counter() - self.started
        await self._send(sentence)

    def _admits(self, sentence: str) -> bool:
        if self.emergency:
            return True
        if dosage_guard.detect_doses(sentence):
            return False
        # A dose split across two sentences ("Give 10." "mg at night.").
        return not (self._previous and dosage_guard.detect_doses(f"{self._previous} {sentence}"))

    async def offer(self, sentence: str) -> None:
        if self.truncated:
            return
        if self._admits(sentence):
            await self._release(sentence)
            self._previous = sentence
        else:
            self.redacted += 1

    async def close(self) -> None:
        """The redaction note, or the Deferral, after a dropped sentence."""
        if not self.redacted:
            return
        if len(self.text) >= dosage_guard._MIN_REMAINING_CHARS:
            closing = dosage_guard._redaction_note(self.language).strip()
        else:
            closing = dosage_guard._refusal_for(self.language)
        self.sent.append(closing)
        await self._send(closing)

    async def consume(self, deltas: AsyncIterator[str],
                      clean: Callable[[str], str] = lambda s: s) -> str:
        """Stream `deltas` through the splitter and the guard; returns the text
        sent. If the stream fails before anything was sent, the error is raised
        (the caller has nothing yet and can fall back); after that, what was
        sent stands and the stream just ends."""
        self.used = True
        splitter = SentenceSplitter()
        try:
            async for delta in deltas:
                for sentence in splitter.feed(delta):
                    cleaned = clean(sentence).strip()
                    if cleaned:
                        await self.offer(cleaned)
            for sentence in splitter.flush():
                cleaned = clean(sentence).strip()
                if cleaned:
                    await self.offer(cleaned)
        except Exception:
            if not self.sent:
                raise
        await self.close()
        return self.text


def _default_guard() -> Optional[dosage_guard.DosageGuard]:
    try:
        from safety_enhancements import get_safety_manager

        return get_safety_manager().dosage_guard
    except Exception:
        return dosage_guard.get_dosage_guard()


_active: "contextvars.ContextVar[Optional[AnswerStream]]" = contextvars.ContextVar(
    "answer_stream", default=None)


@contextmanager
def streaming_to(stream: AnswerStream) -> Iterator[AnswerStream]:
    """Answers generated inside this block stream into `stream`."""
    token = _active.set(stream)
    try:
        yield stream
    finally:
        _active.reset(token)


def active_stream() -> Optional[AnswerStream]:
    """The stream the current answer should go to, if it has not been used."""
    stream = _active.get()
    return stream if stream is not None and not stream.used else None
