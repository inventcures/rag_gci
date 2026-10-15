"""
Per-stage timing traces, one JSON line per span, one file per day.

Every request gets a trace: a root span (the channel's request or voice turn)
and child spans for each stage (embed, search, generate, guard, save ...). Each
span is written when it ends, as one line in `<trace dir>/<YYYY-MM-DD>.jsonl`,
so a day's traces can be summarised with `scripts/trace_summary.py` or read by
an agent measuring a change.

    with trace_request("bolna", user_id=caller):      # root, or child if nested
        with span("search", n_results=5):
            ...

`span()` and `trace_request()` work in sync and async code alike. The parent is
carried in a context variable, so spans opened inside `asyncio.to_thread` or a
task created from inside a trace still attach to it.

What a span may carry is deliberately narrow. Spans are not a second study log
and must never become a PHI store: attribute values are numbers, booleans, or
short strings, and attributes whose key names text a person wrote or heard
(question, answer, transcript, prompt, phone, name ...) are dropped, whatever
their value. The participant is the study pseudonym, never the raw id.

Settings (environment):
    TRACING_ENABLED   "0" turns tracing off (spans still time, nothing is written)
    TRACE_DIR         where the daily files go (default data/traces)
"""

from __future__ import annotations

import contextvars
import functools
import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

# Keys that name free text a participant wrote, said or heard, or identify them.
# Matched as substrings of the lowercased key.
_BLOCKED_KEY_PARTS = (
    "question", "answer", "query", "text", "transcript", "prompt", "message",
    "content", "phone", "number", "name", "email", "address", "user_id",
    "patient_id", "caller",
)
_MAX_STR = 64

_current: contextvars.ContextVar[Optional["Span"]] = contextvars.ContextVar(
    "palli_trace_span", default=None
)
_write_lock = threading.Lock()


def _enabled() -> bool:
    return os.getenv("TRACING_ENABLED", "1") != "0"


def trace_dir() -> Path:
    return Path(os.getenv("TRACE_DIR", "data/traces"))


def _clean_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for key, value in attrs.items():
        lowered = key.lower()
        if any(part in lowered for part in _BLOCKED_KEY_PARTS):
            continue
        if value is None or isinstance(value, (bool, int, float)):
            clean[key] = value
        elif isinstance(value, str):
            clean[key] = value[:_MAX_STR]
    return clean


_release_id: Optional[str] = None


def _current_release_id() -> str:
    global _release_id
    if _release_id is None:
        try:
            from study_release import get_study_release
            _release_id = get_study_release().release_id
        except Exception:
            _release_id = "unknown"
    return _release_id


def _pseudonym(user_id: Optional[str]) -> str:
    if not user_id:
        return "anonymous"
    try:
        from study_logging import pseudonymise
        return pseudonymise(str(user_id))
    except Exception:
        return "unavailable"


class Span:
    """One timed stage. `ms` is set when the span ends."""

    __slots__ = ("name", "trace_id", "span_id", "parent_id", "channel",
                 "participant", "attrs", "status", "started", "ms")

    def __init__(self, name: str, parent: Optional["Span"], channel: Optional[str],
                 participant: Optional[str], attrs: Dict[str, Any]):
        self.name = name
        self.span_id = uuid.uuid4().hex[:16]
        self.parent_id = parent.span_id if parent else None
        self.trace_id = parent.trace_id if parent else uuid.uuid4().hex[:16]
        self.channel = channel or (parent.channel if parent else "unknown")
        self.participant = participant or (parent.participant if parent else "anonymous")
        self.attrs = _clean_attrs(attrs)
        self.status = "ok"
        self.started = time.perf_counter()
        self.ms: Optional[float] = None

    def set(self, **attrs: Any) -> None:
        """Add attributes after the span has started (same rules as at start)."""
        self.attrs.update(_clean_attrs(attrs))

    def to_record(self) -> Dict[str, Any]:
        return {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "trace": self.trace_id,
            "span": self.span_id,
            "parent": self.parent_id,
            "name": self.name,
            "channel": self.channel,
            "participant": self.participant,
            "release": _current_release_id(),
            "ms": self.ms,
            "status": self.status,
            **({"attrs": self.attrs} if self.attrs else {}),
        }


# Open append descriptors, by file path. A turn writes 10-20 spans; opening,
# creating the directory and closing for each one was most of their cost (more
# on a network volume). One os.write on an O_APPEND descriptor per line, so
# lines from several processes cannot interleave.
_fds: Dict[str, int] = {}


def _fd_for(path: Path) -> int:
    key = str(path)
    fd = _fds.get(key)
    if fd is not None:
        try:
            if os.fstat(fd).st_nlink > 0:
                return fd
        except OSError:
            pass
        # The file was removed (retention, a test's cleanup): reopen.
        _close_fd(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(key, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    # Only one file is written at a time: yesterday's (or an old TRACE_DIR's)
    # is done with.
    for other in [k for k in _fds if k != key]:
        _close_fd(other)
    _fds[key] = fd
    return fd


def _close_fd(key: str) -> None:
    fd = _fds.pop(key, None)
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass


def close_trace_files() -> None:
    """Close the open trace files (shutdown, tests)."""
    with _write_lock:
        for key in list(_fds):
            _close_fd(key)


def _write(record: Dict[str, Any]) -> None:
    if not _enabled():
        return
    try:
        path = trace_dir() / f"{record['ts'][:10]}.jsonl"
        line = (json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8")
        with _write_lock:
            os.write(_fd_for(path), line)
    except Exception:
        # Tracing must never cost a caller their answer.
        logger.debug("trace write failed", exc_info=True)


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Span]:
    """Time one stage as a child of the current span (or as a root if none)."""
    parent = _current.get()
    current = Span(name, parent, None, None, attrs)
    token = _current.set(current)
    try:
        yield current
    except BaseException as exc:
        current.status = "cancelled" if type(exc).__name__ == "CancelledError" else "error"
        current.set(error=type(exc).__name__)
        raise
    finally:
        current.ms = round((time.perf_counter() - current.started) * 1000, 2)
        _current.reset(token)
        _write(current.to_record())


@contextmanager
def trace_request(channel: str, user_id: Optional[str] = None,
                  name: str = "request", **attrs: Any) -> Iterator[Span]:
    """
    Root span for one request or voice turn on a channel.

    Nested inside an existing trace (query() called from a channel handler that
    already opened one), this becomes a child span and keeps the outer channel.
    """
    parent = _current.get()
    if parent is not None:
        with span(name, **attrs) as child:
            yield child
        return
    root = Span(name, None, channel, _pseudonym(user_id), attrs)
    token = _current.set(root)
    try:
        yield root
    except BaseException as exc:
        root.status = "cancelled" if type(exc).__name__ == "CancelledError" else "error"
        root.set(error=type(exc).__name__)
        raise
    finally:
        root.ms = round((time.perf_counter() - root.started) * 1000, 2)
        _current.reset(token)
        _write(root.to_record())


def traced(name: str):
    """Decorator: run an async function inside `span(name)`."""
    def decorate(func):
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any):
            with span(name):
                return await func(*args, **kwargs)
        return wrapper
    return decorate


def record_span(name: str, ms: float, **attrs: Any) -> float:
    """Write an already-timed stage as a child of the current span; returns ms."""
    current = Span(name, _current.get(), None, None, attrs)
    current.ms = round(ms, 2)
    _write(current.to_record())
    return current.ms


def current_span() -> Optional[Span]:
    """The innermost open span, to add attributes from deeper code."""
    return _current.get()


def current_trace_id() -> Optional[str]:
    active = _current.get()
    return active.trace_id if active else None
