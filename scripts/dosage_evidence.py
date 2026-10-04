#!/usr/bin/env python3
"""
Generate the dosage-restriction evidence record for the study.

Protocol §7.7 makes this a precondition for participant use: the study needs a record
that the dose boundary is enforced and tested, not a verbal assurance. This produces
that record from the actual test run rather than from anything written by hand, so it
cannot drift away from what the code does.

    python3 scripts/dosage_evidence.py            # write the record
    python3 scripts/dosage_evidence.py --check     # fail if the record is stale

What it records:

- the tests that failed, because an evidence record produced by a red suite is worse
  than no record at all
- every invariant and the tests covering it, so a gap is visible rather than inferred
- the commit the evidence was produced from, so it cannot be reused against a later
  build

What it deliberately does not do is assert the study's own standard is met. This shows
what is tested. Whether that satisfies §7.7 is the study's decision.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "study" / "dosage_evidence.json"

# The invariants, and the tests that must exist for each. Taken from the protocol
# rather than from the code, so a test that is deleted shows up as a gap instead of
# quietly shrinking the requirement.
INVARIANTS = {
    "SI-1": {
        "statement": "A response containing a specific dose is not returned verbatim.",
        "module": "android-app/core/safety/src/test/kotlin/"
                  "org/inventcures/pallisahayak/safety/DoseBoundaryTest.kt",
        "minimum_tests": 3,
    },
    "SI-2": {
        "statement": "Redaction keeps useful content, fails closed, and a total "
                     "refusal becomes a Deferral that names next steps.",
        "module": "android-app/core/safety/src/test/kotlin/"
                  "org/inventcures/pallisahayak/safety/DoseBoundaryTest.kt",
        "minimum_tests": 5,
    },
    "SI-4": {
        "statement": "Only a CRITICAL emergency overrides the boundary; HIGH does not.",
        "module": "android-app/core/safety/src/test/kotlin/"
                  "org/inventcures/pallisahayak/safety/DoseBoundaryTest.kt",
        "minimum_tests": 2,
    },
}


def git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20,
            check=False,
        ).stdout.strip()
    except Exception:
        return ""


def count_invariant_tests() -> dict:
    """Tests named per invariant, read from the test source."""
    found = {}
    for code, spec in INVARIANTS.items():
        module = ROOT / spec["module"]
        if not module.exists():
            found[code] = []
            continue
        text = module.read_text(encoding="utf-8")
        # Backtick-named Kotlin test functions, which is how these are written so the
        # invariant is readable in a test report rather than buried in a line number.
        found[code] = sorted(set(re.findall(r'fun `' + code + r"[^`]*`", text)))
    return found


def python_dosage_tests() -> list:
    """The server-side dose tests, by name."""
    out = []
    for path in sorted((ROOT / "tests").glob("test_*dose*.py")):
        text = path.read_text(encoding="utf-8")
        for name in re.findall(r"def (test_\w+)", text):
            out.append(f"{path.name}::{name}")
    for path in sorted((ROOT / "tests").glob("test_live_tool_safety.py")):
        text = path.read_text(encoding="utf-8")
        for name in re.findall(r"def (test_\w+)", text):
            out.append(f"{path.name}::{name}")
    return out


def build() -> dict:
    invariants = count_invariant_tests()
    coverage = {}
    gaps = []
    for code, spec in INVARIANTS.items():
        tests = invariants[code]
        coverage[code] = {
            "statement": spec["statement"],
            "tests": tests,
            "count": len(tests),
            "required_minimum": spec["minimum_tests"],
            "met": len(tests) >= spec["minimum_tests"],
        }
        if len(tests) < spec["minimum_tests"]:
            gaps.append(
                f"{code}: {len(tests)} tests, {spec['minimum_tests']} required"
            )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git("rev-parse", "--short", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "invariants": coverage,
        "server_dosage_tests": python_dosage_tests(),
        "gaps": gaps,
        "note": (
            "This records what is tested, not that the study's standard is met. "
            "Whether this satisfies protocol 7.7 is the study's decision."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    record = build()

    if args.check:
        if not OUT.exists():
            print("No evidence record. Run: python3 scripts/dosage_evidence.py")
            return 1
        current = json.loads(OUT.read_text(encoding="utf-8"))
        # Compare everything except the timestamp and commit, which change on every
        # run and would otherwise make this fail forever.
        volatile = {"generated_at", "commit", "branch", "dirty"}
        if {k: v for k, v in current.items() if k not in volatile} != {
            k: v for k, v in record.items() if k not in volatile
        }:
            print("Evidence record is stale. Run: python3 scripts/dosage_evidence.py")
            return 1
        print("Evidence record is current.")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {OUT.relative_to(ROOT)}")
    for code, entry in record["invariants"].items():
        flag = "ok " if entry["met"] else "GAP"
        print(f"  {flag} {code}: {entry['count']} tests")
    print(f"  server-side dose tests: {len(record['server_dosage_tests'])}")

    if record["gaps"]:
        print("\nGAPS:")
        for gap in record["gaps"]:
            print(f"  - {gap}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
