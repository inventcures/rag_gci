#!/usr/bin/env python3
"""
Summarise a day's per-stage traces: p50/p95 per stage, per channel.

Reads `<TRACE_DIR>/<date>.jsonl` (written by observability/tracing.py) and
prints one markdown table per channel: each span name with its count, median,
p95, and its share of the channel's root-span time. Share is computed from the
sum of span durations, so nested spans (e.g. `llm` inside `generate`) both show
the time they cover; it is a "where does the time go" view, not a partition.

    python scripts/trace_summary.py                     # today, data/traces
    python scripts/trace_summary.py --date 2026-10-14 --dir data/traces
    python scripts/trace_summary.py --json > summary.json

Used to measure every latency change: run the benchmark or replay traffic,
summarise, change one thing, summarise again.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _percentile(values: List[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, round(pct / 100 * len(ordered) + 0.5) - 1))
    return ordered[index]


def load(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def summarise(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """{channel: {"roots": n, "root_ms_total": x, "stages": {name: stats}}}"""
    by_channel: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    root_total: Dict[str, float] = defaultdict(float)
    root_count: Dict[str, int] = defaultdict(int)
    for record in records:
        ms = record.get("ms")
        if ms is None:
            continue
        channel = record.get("channel", "unknown")
        by_channel[channel][record.get("name", "?")].append(float(ms))
        if record.get("parent") is None:
            root_total[channel] += float(ms)
            root_count[channel] += 1

    summary: Dict[str, Dict[str, Any]] = {}
    for channel, stages in by_channel.items():
        total = root_total.get(channel) or 0.0
        rows = {}
        for name, values in stages.items():
            rows[name] = {
                "count": len(values),
                "p50_ms": round(_percentile(values, 50), 1),
                "p95_ms": round(_percentile(values, 95), 1),
                "share": round(sum(values) / total, 3) if total else None,
            }
        summary[channel] = {"roots": root_count.get(channel, 0), "root_ms_total": round(total, 1),
                            "stages": dict(sorted(rows.items(), key=lambda kv: -kv[1]["p50_ms"]))}
    return dict(sorted(summary.items()))


def to_markdown(summary: Dict[str, Dict[str, Any]]) -> str:
    out = []
    for channel, data in summary.items():
        out.append(f"### {channel} ({data['roots']} requests)\n")
        out.append("| stage | count | p50 ms | p95 ms | share of request time |")
        out.append("|---|---|---|---|---|")
        for name, row in data["stages"].items():
            share = "" if row["share"] is None else f"{row['share'] * 100:.1f}%"
            out.append(f"| {name} | {row['count']} | {row['p50_ms']} | {row['p95_ms']} | {share} |")
        out.append("")
    return "\n".join(out)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--dir", default=os.getenv("TRACE_DIR", "data/traces"))
    parser.add_argument("--json", action="store_true", help="print JSON instead of markdown")
    args = parser.parse_args(argv)

    path = Path(args.dir) / f"{args.date}.jsonl"
    if not path.exists():
        print(f"no traces at {path}", file=sys.stderr)
        return 1
    summary = summarise(load(path))
    print(json.dumps(summary, indent=2) if args.json else to_markdown(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
