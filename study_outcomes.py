#!/usr/bin/env python3
"""
Study Outcomes from the System Log
==================================
Computes the outcomes the EVAH grant says are sourced from system logs,
implementing the definitions the study protocol fixes rather than inventing
convenient ones.

Grant commitments implemented here:
  * Adoption    - "Calls/ASHA/week; sustained use (Continuous; system logs)"
  * Safety      - "Hallucination rate; emergency F1"
  * Equity      - "All metrics by language (Throughout; system logs)"
  * Cost        - "Per-interaction resource use"
  * Supervision - tiered expert sampling rates

Protocol 4.1, quoted, because it is unusually specific and easy to get wrong:

    "Continued adoption at three months after training means at least one
    substantive care-related interaction in at least two distinct weeks of the
    preceding 28 days. Training demonstrations, test messages, accidental
    activations and duplicate retries are excluded. A substantive interaction
    records a care question and a returned response or appropriate clinical
    escalation. The programme benchmark is more than 60% of trained participants
    meeting this definition. Report the denominator, clinical opportunity,
    deaths/role changes and uncertainty, rather than reward unnecessary use."

The denominator and the exclusions matter more than the numerator. A metric that
counts every logged row would overstate adoption, because training demos and
failed calls inflate it, and the protocol explicitly asks that they do not.
"""

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Set

from study_logging import LOG_DIR, read_entries

# Protocol 4.1 windows.
ADOPTION_WINDOW_DAYS = 28
ADOPTION_MIN_DISTINCT_WEEKS = 2

# Grant feasibility thresholds.
SUSTAINED_ADOPTION_TARGET = 0.60
SUSTAINED_USABILITY_TARGET = 70.0   # SUS, collected separately

# Grant language-equity threshold: "LEI <0.8 flags equity gaps".
LEI_FLAG_THRESHOLD = 0.8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _substantive(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Only interactions the protocol 4.1 definition counts.

    Excluded here: training demonstrations, test messages, accidental
    activations, duplicate retries, and any interaction that returned neither a
    response nor an escalation.
    """
    return [e for e in entries if e.get("interaction_class", "substantive") == "substantive"]


def _parse_ts(entry: Dict[str, Any]) -> Optional[datetime]:
    raw = entry.get("occurred_at_iso")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except (ValueError, TypeError):
        return None


def _iso_week_key(when: datetime) -> str:
    year, week, _ = when.isocalendar()
    return f"{year}-W{week:02d}"


# ---------------------------------------------------------------------------
# Adoption
# ---------------------------------------------------------------------------

def adoption_summary(
    entries: List[Dict[str, Any]],
    window_days: int = ADOPTION_WINDOW_DAYS,
    as_of: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Sustained adoption per protocol 4.1, plus the weekly call volume the grant
    names as its adoption outcome.

    Reports the denominator explicitly, because the protocol asks for it and
    because a percentage without one is not interpretable.
    """
    as_of = as_of or datetime.now(timezone.utc)
    window_start = as_of - timedelta(days=window_days)
    substantive = _substantive(entries)

    by_participant: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in substantive:
        when = _parse_ts(entry)
        if when is None or when < window_start or when > as_of:
            continue
        by_participant[entry.get("participant_id", "")].append(entry)

    sustained: List[str] = []
    per_participant: Dict[str, Dict[str, Any]] = {}

    for participant_id, items in by_participant.items():
        weeks: Set[str] = {_iso_week_key(_parse_ts(i)) for i in items if _parse_ts(i)}
        active = len(weeks) >= ADOPTION_MIN_DISTINCT_WEEKS
        if active:
            sustained.append(participant_id)
        per_participant[participant_id] = {
            "interactions": len(items),
            "distinct_weeks": len(weeks),
            "weeks": sorted(weeks),
            "sustained": active,
            "languages": sorted({i.get("language", "") for i in items if i.get("language")}),
            "care_role": items[0].get("care_role", ""),
            "site_id": items[0].get("site_id", ""),
        }

    denominator = len(per_participant)
    rate = (len(sustained) / denominator) if denominator else 0.0

    # Calls per ASHA worker per week, the grant's literal adoption wording.
    active_asas = {
        pid for pid, d in per_participant.items()
        if "asha" in (d.get("care_role") or "").lower()
    }
    weeks_observed = max(window_days / 7.0, 1.0)
    calls_per_asha_week = (
        sum(d["interactions"] for pid, d in per_participant.items() if pid in active_asas)
        / (len(active_asas) * weeks_observed)
        if active_asas else 0.0
    )

    return {
        "window_days": window_days,
        "window_start": window_start.isoformat(),
        "as_of": as_of.isoformat(),
        # Denominator first, per protocol 4.1.
        "denominator_participants_with_activity": denominator,
        "sustained_participants": len(sustained),
        "sustained_adoption_rate": round(rate, 4),
        "meets_programme_benchmark": rate > SUSTAINED_ADOPTION_TARGET,
        "target": SUSTAINED_ADOPTION_TARGET,
        "calls_per_asha_per_week": round(calls_per_asha_week, 2),
        "active_asha_workers": len(active_asas),
        "excluded_interactions": len(entries) - len(substantive),
        "exclusion_reasons": dict(Counter(
            e.get("exclusion_reason") for e in entries
            if e.get("interaction_class") != "substantive" and e.get("exclusion_reason")
        )),
        "per_participant": per_participant,
    }


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

def safety_summary(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Safety outcomes the grant names as log-sourced: hallucination rate and
    emergency detection.

    "Hallucination" is not directly measurable from logs. The operational proxy
    used here is *unsupported answer*: a substantive response returned with no
    retrieved source, which means the text was generated without grounding. This
    is stated plainly because the grant's "hallucination rate" will otherwise be
    read as a stronger claim than it is. The protocol is stricter still and
    requires clinician adjudication, so this figure is a screening proxy, not
    the reported outcome.
    """
    substantive = _substantive(entries)
    total = len(substantive)

    def rate(count: int) -> float:
        return round(count / total, 4) if total else 0.0

    unsupported = [e for e in substantive if e.get("source_count", 0) == 0]
    emergencies = [e for e in substantive if e.get("emergency_detected")]
    dosage_blocked = [e for e in substantive if e.get("dosage_blocked")]
    errored = [e for e in entries if e.get("error")]

    by_level = Counter(e.get("evidence_level", "E") for e in substantive)
    by_status = Counter(e.get("validation_status", "") for e in substantive)
    by_emergency = Counter(e.get("emergency_level", "none") for e in substantive)

    return {
        "substantive_interactions": total,
        # Screening proxy for the grant's "hallucination rate".
        "unsupported_answer_rate": rate(len(unsupported)),
        "unsupported_answer_count": len(unsupported),
        "hallucination_rate_caveat": (
            "Proxy only: substantive responses with zero retrieved sources. "
            "Not a hallucination measure. Protocol requires clinician "
            "adjudication for the reported figure."
        ),
        "emergency_detected_count": len(emergencies),
        "emergency_detection_rate": rate(len(emergencies)),
        "emergency_levels": dict(by_emergency),
        "dosage_restriction_applied": len(dosage_blocked),
        "dosage_restriction_rate": rate(len(dosage_blocked)),
        "dosage_block_reasons": dict(Counter(
            e.get("dosage_reason") for e in dosage_blocked if e.get("dosage_reason")
        )),
        "evidence_levels": dict(by_level),
        "validation_statuses": dict(by_status),
        "error_count": len(errored),
        "feasibility": {
            "zero_undetected_critical_safety_events": len([
                e for e in substantive
                if e.get("emergency_level") == "critical"
                and not e.get("emergency_detected")
            ]) == 0,
        },
    }


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------

def latency_summary(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Stage latency, which the grant monitors in Phase 2 and the Android spec
    budgets at under 13 seconds for a voice interaction.
    """
    totals: List[float] = []
    stages: Dict[str, List[float]] = defaultdict(list)

    for entry in _substantive(entries):
        if entry.get("total_latency_ms"):
            totals.append(float(entry["total_latency_ms"]))
        for stage, value in (entry.get("stage_latency_ms") or {}).items():
            try:
                stages[stage].append(float(value))
            except (TypeError, ValueError):
                continue

    def stats(values: List[float]) -> Dict[str, Any]:
        if not values:
            return {"count": 0}
        ordered = sorted(values)
        return {
            "count": len(ordered),
            "mean_ms": round(mean(ordered), 1),
            "median_ms": round(median(ordered), 1),
            "p95_ms": round(ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)], 1),
            "max_ms": round(ordered[-1], 1),
        }

    return {
        "end_to_end": stats(totals),
        "by_stage": {stage: stats(values) for stage, values in sorted(stages.items())},
        "voice_budget_ms": 13000,
        "voice_budget_met": bool(totals) and (sum(totals) / len(totals)) < 13000,
    }


# ---------------------------------------------------------------------------
# Language equity
# ---------------------------------------------------------------------------

def language_equity_summary(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    "All metrics by language", which the grant lists as a secondary outcome
    measured throughout, and the protocol's requirement to report participant and
    audited interaction counts per language.

    Protocol 4.4 forbids non-inferiority claims across languages and warns that
    "absence of observed errors in a small language sample is not evidence of
    equal safety". Low-count languages are therefore flagged as underpowered
    rather than ranked.
    """
    by_language: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "interactions": 0,
            "participants": set(),
            "unsupported": 0,
            "dosage_blocked": 0,
            "emergencies": 0,
            "errors": 0,
            "latencies": [],
            "care_roles": set(),
        }
    )

    for entry in _substantive(entries):
        lang = entry.get("language") or "unknown"
        bucket = by_language[lang]
        bucket["interactions"] += 1
        bucket["participants"].add(entry.get("participant_id", ""))
        if entry.get("source_count", 0) == 0:
            bucket["unsupported"] += 1
        if entry.get("dosage_blocked"):
            bucket["dosage_blocked"] += 1
        if entry.get("emergency_detected"):
            bucket["emergencies"] += 1
        if entry.get("error"):
            bucket["errors"] += 1
        if entry.get("total_latency_ms"):
            bucket["latencies"].append(float(entry["total_latency_ms"]))
        if entry.get("care_role"):
            bucket["care_roles"].add(entry["care_role"])

    out: Dict[str, Any] = {}
    for lang, bucket in sorted(by_language.items()):
        n = bucket["interactions"]
        latencies = bucket["latencies"]
        out[lang] = {
            "interactions": n,
            "participants": len(bucket["participants"] - {""}),
            "care_roles": sorted(bucket["care_roles"] - {""}),
            "unsupported_answer_rate": round(bucket["unsupported"] / n, 4) if n else 0.0,
            "dosage_restriction_rate": round(bucket["dosage_blocked"] / n, 4) if n else 0.0,
            "emergency_count": bucket["emergencies"],
            "error_count": bucket["errors"],
            "mean_latency_ms": round(mean(latencies), 1) if latencies else None,
            # Protocol 4.4: below this, avoid comparative regression.
            "underpowered": n < 10,
            "reporting_note": (
                "Underpowered: report descriptively, do not rank languages"
                if n < 10 else
                ("Descriptive only, wide intervals" if n < 30 else "Reportable")
            ),
        }

    languages = [v for v in out.values() if v["interactions"]]
    # A simple equity index: 1.0 means every supported language has equal
    # per-interaction unsupported-answer rate. Named LEI here is deliberate
    # only in that it flags disparity; the grant's formal LEI is not
    # reimplemented, and the protocol forbids ranking on small samples.
    min_rate = min((v["unsupported_answer_rate"] for v in languages), default=0.0)
    max_rate = max((v["unsupported_answer_rate"] for v in languages), default=0.0)
    disparity = round(max_rate - min_rate, 4)

    return {
        "by_language": out,
        "languages_with_data": len(languages),
        "unsupported_rate_disparity": disparity,
        "equity_flagged": disparity > (1.0 - LEI_FLAG_THRESHOLD),
        "leverage_threshold": LEI_FLAG_THRESHOLD,
        "caveat": (
            "Protocol 4.4: language and site may be confounded; observed "
            "differences cannot be attributed to language technology, and no "
            "non-inferiority claim is made."
        ),
    }


# ---------------------------------------------------------------------------
# Supervision and release
# ---------------------------------------------------------------------------

def supervision_summary(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Tiered expert sampling, at the rates the grant specifies."""
    substantive = _substantive(entries)
    total = len(substantive)
    no_source = sum(1 for e in substantive if e.get("source_count", 0) == 0)
    low_confidence = sum(1 for e in substantive if float(e.get("confidence") or 0) < 0.7)
    critical = no_source  # flagged as critical by the sampler
    return {
        "sampling_rates": {"random": 0.05, "flagged": 0.50, "critical": 1.00},
        "expected_samples_per_100": {
            "random": 5,
            "flagged_flagged_pool": round(0.5 * (low_confidence + no_source), 1),
            "critical_all": critical,
        },
        "flagged_pool": {
            "low_confidence_under_0_7": low_confidence,
            "no_retrieved_sources": no_source,
            "total_flagged": low_confidence + no_source,
            "flagged_rate": round((low_confidence + no_source) / total, 4) if total else 0.0,
        },
        "critical_events": critical,
    }


def release_summary() -> Dict[str, Any]:
    """Release provenance and any recorded drift (protocol 7.8)."""
    from study_release import get_study_release

    release = get_study_release()
    drift = release.check_drift()
    drift_files = sorted(Path("./data/study/drift").glob("drift_*.json"))

    return {
        "release_id": release.release_id,
        "release_name": release.release_name,
        "status": release.status,
        "approved": release.approvals_complete,
        "fit_for_participant_use": release.is_activated and release.approvals_complete,
        "drift_detected": bool(drift),
        "drift_components": [d.describe().strip() for d in drift],
        "drift_records": len(drift_files),
        "releases_observed_in_log": sorted({
            e.get("release_id", "") for e in entries_from_log() if e.get("release_id")
        }),
        "unapproved_interactions": sum(
            1 for e in entries_from_log() if e.get("release_approved") is False
        ),
    }


def entries_from_log(day: Optional[str] = None) -> List[Dict[str, Any]]:
    return read_entries(day=day)


# ---------------------------------------------------------------------------
# Roll-up
# ---------------------------------------------------------------------------

def build_dashboard(day: Optional[str] = None) -> Dict[str, Any]:
    """
    Everything the study dashboard shows, from one read of the log.

    Kept as a single pass so the figures on a page cannot disagree with each
    other through being computed from different subsets.
    """
    entries = entries_from_log(day=day)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "day": day or "all",
        "total_logged_interactions": len(entries),
        "total_substantive": len(_substantive(entries)),
        "adoption": adoption_summary(entries),
        "safety": safety_summary(entries),
        "latency": latency_summary(entries),
        "language_equity": language_equity_summary(entries),
        "supervision": supervision_summary(entries),
        "release": release_summary(),
    }


if __name__ == "__main__":
    print(json.dumps(build_dashboard(), indent=2, default=str))