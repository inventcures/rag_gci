"""
Side work that runs after the reply.

Bookkeeping (conversation history, observation extraction, the answer-path
record, analytics and expert-sampling fan-out, the shadow-mode KG check) used to
run inline, so every caller waited for it before hearing an answer. It now runs
as supervised background tasks:

  * scheduled with `after_reply(name, fn)` once the answer is ready; each task
    runs inside a `post.<name>` span of the request's trace
  * failures are logged and counted (`stats()`), never silently dropped
  * the tasks are held, so they are not garbage-collected mid-flight, and
    `drain()` waits for them (shutdown, tests, scripts)

What must exist before the caller hears the answer (the study interaction log,
the Dose Boundary) does not go through here.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, Optional, Set

from observability import span

logger = logging.getLogger(__name__)

_tasks: Set[asyncio.Task] = set()
_counts: Dict[str, int] = {"started": 0, "completed": 0, "failed": 0}


async def _supervise(name: str, fn: Callable[[], Awaitable[Any]]) -> None:
    _counts["started"] += 1
    try:
        with span(f"post.{name}"):
            await fn()
        _counts["completed"] += 1
    except asyncio.CancelledError:
        _counts["failed"] += 1
        logger.warning("Post-reply work %s was cancelled before it finished", name)
        raise
    except Exception:
        _counts["failed"] += 1
        logger.exception("Post-reply work %s failed", name)


def after_reply(name: str, fn: Callable[[], Awaitable[Any]]) -> Optional[asyncio.Task]:
    """Run `fn()` in the background, supervised. Returns the task, or None
    when there is no running loop (then the work is skipped and counted)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _counts["failed"] += 1
        logger.error("Post-reply work %s skipped: no running event loop", name)
        return None
    task = loop.create_task(_supervise(name, fn), name=f"post.{name}")
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return task


async def drain(timeout: Optional[float] = 10.0) -> bool:
    """Wait for scheduled post-reply work, including work it schedules.
    Returns False if some was still running at the timeout."""
    loop = asyncio.get_running_loop()
    deadline = None if timeout is None else loop.time() + timeout
    while True:
        pending = {t for t in _tasks if not t.done() and t is not asyncio.current_task()}
        if not pending:
            return True
        remaining = None if deadline is None else deadline - loop.time()
        if remaining is not None and remaining <= 0:
            return False
        await asyncio.wait(pending, timeout=remaining)


def stats() -> Dict[str, int]:
    """Started, completed and failed counts, and how many are running now."""
    return {**_counts, "running": sum(1 for t in _tasks if not t.done())}
