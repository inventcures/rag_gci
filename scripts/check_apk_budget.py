#!/usr/bin/env python3
"""
Measure the release APK and judge whether a field download is viable.

Criterion 8 asks whether the download is viable on the stated field connection.
Nobody can answer that by reading a build file, so this measures the artifact and
does the arithmetic. It is a script rather than a test because it needs a release
build, which a unit test run should not trigger.

    ./gradlew :app:assembleRelease && python3 scripts/check_apk_budget.py

Exits non-zero when the artifact is over budget. That is the enforcement. A comment
in a build file saying "the budget is X" is a wish; this is the check that fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APK_DIR = "app/build/outputs/apk/release"

# Ceiling chosen from the download arithmetic below, not from taste. A 2 GB device
# over a 2G link is the field constraint, and 1 MB was reachable inside a coffee
# break on the slowest connection anyone actually uses.
BUDGET_BYTES = 6 * 1024 * 1024

# Effective throughput after protocol overhead, not the headline radio number. The
# field links are 2G GPRS and 2.5G EDGE; quoted rates are burst and never sustained.
CONNECTIONS = {
    "2G GPRS (sustained)": 20_000,      # bytes per second
    "2.5G EDGE (sustained)": 80_000,
}


def human(seconds: float) -> str:
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes}m {rest:02d}s" if minutes else f"{rest}s"


def main() -> int:
    # glob once, on the directory. An earlier version interpolated the wildcard into
    # the path and then globbed again, which looks for *.apk/*.apk and reports no
    # artifact at all while one sits right there.
    apks = sorted((ROOT / "android-app" / APK_DIR).glob("*.apk"))
    if not apks:
        print("No release APK found. Run ./gradlew :app:assembleRelease first.")
        return 2

    apk = max(apks, key=lambda p: p.stat().st_size)
    size = apk.stat().st_size
    print(f"APK:      {apk.relative_to(ROOT)}")
    print(f"size:     {size / 1_048_576:.2f} MB ({size:,} bytes)")
    print(f"budget:   {BUDGET_BYTES / 1_048_576:.0f} MB")
    print()

    print("Download time by connection:")
    for name, throughput in CONNECTIONS.items():
        seconds = size / throughput
        verdict = "viable" if seconds <= 300 else "TOO SLOW"
        print(f"  {name:24s} {human(seconds):>8s}   {verdict}")
    print()

    if size > BUDGET_BYTES:
        print(
            f"OVER BUDGET by {(size - BUDGET_BYTES) / 1_048_576:.2f} MB. "
            "Raising the ceiling does not make a 2G download viable."
        )
        return 1

    print("Within budget.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
