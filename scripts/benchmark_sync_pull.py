#!/usr/bin/env python3
"""
Time mobile sync pull and the patient list over synthetic patients.

Builds N patient records (default 1,000, 20 observations each) spread over ten
app users, then times, for one user, inside a trace so the spans land in the
trace files:

  * first sync (last_sync_at=0)
  * a routine sync where nothing changed since the last one
  * the patient list, twice (cold, then warm)

    python scripts/benchmark_sync_pull.py --patients 1000
    python scripts/trace_summary.py --dir <printed trace dir>
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from observability import record_span, trace_request, trace_dir  # noqa: E402
from personalization.longitudinal_memory import (  # noqa: E402
    CareTeamMember,
    DataSourceType,
    LongitudinalMemoryManager,
    LongitudinalPatientRecord,
    TimestampedObservation,
)
from sync.delta_tracker import DeltaTracker  # noqa: E402

USERS = [f"asha-{i}" for i in range(10)]


async def build(manager: LongitudinalMemoryManager, patients: int, observations: int) -> None:
    start = datetime.now() - timedelta(days=90)
    for i in range(patients):
        record = LongitudinalPatientRecord(patient_id=f"pt-{i:05d}")
        record.add_care_team_member(CareTeamMember(provider_id=USERS[i % len(USERS)], name="asha", role="asha"))
        for j in range(observations):
            record.add_observation(TimestampedObservation(
                observation_id=f"o-{i}-{j}", timestamp=start + timedelta(days=j * 4),
                source_type=DataSourceType.PATIENT_REPORTED, source_id="bench", reported_by="app",
                category="symptom", entity_name="pain", value=3, value_text="",
            ))
        await manager.save_record(record)
    old = time.time() - 3600
    for path in manager.storage_path.glob("*_longitudinal.json"):
        os.utime(path, (old, old))


async def timed(name: str, coro):
    started = time.perf_counter()
    result = await coro
    return result, record_span(name, (time.perf_counter() - started) * 1000)


async def main(patients: int, observations: int) -> None:
    store = tempfile.mkdtemp(prefix="sync-bench-")
    manager = LongitudinalMemoryManager(storage_path=store)
    await build(manager, patients, observations)
    tracker = DeltaTracker(manager, None, None)
    user = USERS[0]
    rows = []
    with trace_request("benchmark/sync_pull", name="benchmark"):
        result, ms = await timed("sync_pull.first", tracker.get_changes_since(user, 0))
        rows.append(("first sync", ms, len(result["patients"]), len(result["observations"])))
        result, ms = await timed("sync_pull.unchanged", tracker.get_changes_since(user, time.time() - 60))
        rows.append(("routine sync, nothing changed", ms, len(result["patients"]), len(result["observations"])))
        result, ms = await timed("patient_list.cold", manager.get_patients_for_user(user))
        rows.append(("patient list (cold)", ms, len(result), 0))
        result, ms = await timed("patient_list.warm", manager.get_patients_for_user(user))
        rows.append(("patient list (warm)", ms, len(result), 0))
    print(f"{patients} patients x {observations} observations, user {user}")
    print("| call | ms | patients returned | observations returned |")
    print("|---|---|---|---|")
    for label, ms, n_patients, n_obs in rows:
        print(f"| {label} | {ms:.1f} | {n_patients} | {n_obs} |")
    print(f"traces: {trace_dir()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--patients", type=int, default=1000)
    parser.add_argument("--observations", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(main(args.patients, args.observations))
