"""
Document uploads indexed as background jobs.

Parsing, chunking and embedding a guideline is seconds to minutes of CPU. Run
inside the upload request, on the event loop, it froze every live call for as
long as it took. An upload now saves the file, queues a job and returns its id;
jobs run one at a time, their blocking work in worker threads, and
`GET /api/upload/{job_id}` reports queued, running, done or failed.
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

from post_work import after_reply

# Jobs kept for status queries; older ones are forgotten.
MAX_JOBS = 200

_jobs: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_one_at_a_time: Optional[asyncio.Lock] = None
_lock_loop: Optional[asyncio.AbstractEventLoop] = None

_UNSAFE = re.compile(r"[^A-Za-z0-9._ \-()ऀ-෿]+")


def safe_upload_name(filename: Optional[str]) -> str:
    """A filename that stays inside the uploads folder.

    Directory parts (either separator) are dropped, `..` and leading dots go,
    anything but letters, digits, Indic letters and `._ -()` becomes `_`, and a
    short random prefix keeps two uploads of the same name apart."""
    name = (filename or "").replace("\\", "/").split("/")[-1]
    name = _UNSAFE.sub("_", name).replace("..", "_").strip(" .")
    if not name:
        name = "upload"
    return f"{uuid.uuid4().hex[:8]}_{name[:120]}"


def _lock() -> asyncio.Lock:
    global _one_at_a_time, _lock_loop
    loop = asyncio.get_running_loop()
    if _one_at_a_time is None or _lock_loop is not loop:
        _one_at_a_time, _lock_loop = asyncio.Lock(), loop
    return _one_at_a_time


def submit(description: str, work: Callable[[], Awaitable[Dict[str, Any]]]) -> str:
    """Queue `work` and return the job id. Jobs run one at a time."""
    job_id = uuid.uuid4().hex[:12]
    job = {"job_id": job_id, "status": "queued", "description": description,
           "queued_at": time.time(), "started_at": None, "finished_at": None,
           "result": None, "error": None}
    _jobs[job_id] = job
    while len(_jobs) > MAX_JOBS:
        _jobs.popitem(last=False)

    async def run() -> None:
        async with _lock():
            job["status"] = "running"
            job["started_at"] = time.time()
            try:
                result = await work()
                failed = isinstance(result, dict) and result.get("status") == "error"
                job["result"] = result
                job["status"] = "failed" if failed else "done"
                if failed:
                    job["error"] = result.get("error")
            except Exception as exc:  # recorded on the job, then re-raised for post_work
                job["status"] = "failed"
                job["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                job["finished_at"] = time.time()

    after_reply("ingest", run)
    return job_id


def status(job_id: str) -> Optional[Dict[str, Any]]:
    job = _jobs.get(job_id)
    return dict(job) if job else None
