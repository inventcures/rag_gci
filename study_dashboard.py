#!/usr/bin/env python3
"""
Study Audit Dashboard
=====================
A separate, read-only admin view over the system log, serving the figures the
EVAH grant says are log-sourced, in the grant's own vocabulary.

Read-only by construction: this module exposes GET routes only and offers no
write, edit or delete path into the log. The study log is append-only, so an
auditor can see everything and change nothing.

What it shows, and where each number comes from:
  * Adoption   - protocol 4.1 sustained use, calls/ASHA/week (grant outcome)
  * Safety     - unsupported-answer rate, emergency detection, dosage refusals
  * Latency    - STT / RAG / TTS / end-to-end, against the 13s voice budget
  * Equity     - every metric broken out by language (grant secondary outcome)
  * Supervision- 5% random / 50% flagged / 100% critical sampling pools
  * Release    - which release served each interaction, and any drift
  * Browse     - masked, scrubbed interaction rows for spot-checking

Privacy: participant identifiers are HMAC pseudonyms and free text is scrubbed
before it reaches disk, so this view never displays a raw participant id. Free
text may still contain a personal name, because names cannot be removed reliably
without a gazetteer; that limitation is stated in the UI rather than hidden.

Mount separately from the participant-facing API:
    app.include_router(study_dashboard_router)   # prefix /admin/study
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

import study_outcomes

logger = logging.getLogger(__name__)

router = APIRouter()

PAGE_TEMPLATE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Palli Sahayak &mdash; Study Audit</title>
<style>
 :root{--bg:#0f1216;--panel:#171b21;--line:#262c35;--ink:#e6e9ef;--dim:#8b95a5;
       --ok:#3fb950;--warn:#d29922;--bad:#f85149;--accent:#58a6ff}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);
      font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
 header{padding:20px 28px;border-bottom:1px solid var(--line);display:flex;
        gap:16px;align-items:baseline;flex-wrap:wrap}
 h1{font-size:18px;margin:0;font-weight:650;letter-spacing:-.01em}
 .sub{color:var(--dim);font-size:12.5px}
 .wrap{padding:20px 28px 64px;max-width:1400px}
 .banner{padding:12px 16px;border-radius:8px;margin-bottom:20px;font-size:13px;
         border:1px solid;line-height:1.5}
 .b-red{background:rgba(248,81,73,.09);border-color:rgba(248,81,73,.4);color:#ffb3ae}
 .b-amber{background:rgba(210,153,34,.09);border-color:rgba(210,153,34,.4);color:#e3c37a}
 .b-blue{background:rgba(88,166,255,.07);border-color:rgba(88,166,255,.35);color:#a8ccff}
 h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);
    margin:32px 0 12px;font-weight:600}
 .grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(230px,1fr))}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
 .k{color:var(--dim);font-size:11.5px;text-transform:uppercase;letter-spacing:.06em}
 .v{font-size:25px;font-weight:640;margin-top:5px;letter-spacing:-.02em}
 .v.ok{color:var(--ok)} .v.warn{color:var(--warn)} .v.bad{color:var(--bad)}
 .n{color:var(--dim);font-size:12px;margin-top:4px}
 table{width:100%;border-collapse:collapse;background:var(--panel);
       border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:13px}
 th{text-align:left;padding:9px 12px;color:var(--dim);font-weight:600;font-size:11.5px;
    text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid var(--line);
    background:rgba(255,255,255,.02)}
 td{padding:9px 12px;border-bottom:1px solid rgba(38,44,53,.6);vertical-align:top}
 tr:last-child td{border-bottom:none}
 td.num{text-align:right;font-variant-numeric:tabular-nums}
 code{background:rgba(255,255,255,.05);padding:1px 5px;border-radius:4px;font-size:12px}
 .tag{display:inline-block;padding:1px 7px;border-radius:20px;font-size:11px;
      border:1px solid;font-weight:600}
 .t-ok{color:var(--ok);border-color:rgba(63,185,80,.4);background:rgba(63,185,80,.1)}
 .t-warn{color:var(--warn);border-color:rgba(210,153,34,.4);background:rgba(210,153,34,.1)}
 .t-bad{color:var(--bad);border-color:rgba(248,81,73,.4);background:rgba(248,81,73,.1)}
 .t-mut{color:var(--dim);border-color:var(--line)}
 details{background:var(--panel);border:1px solid var(--line);border-radius:10px;
         padding:12px 16px;margin-top:10px}
 summary{cursor:pointer;color:var(--dim);font-size:12.5px}
 pre{white-space:pre-wrap;word-break:break-word;font-size:12px;color:#c9d1d9;
     background:rgba(0,0,0,.28);padding:12px;border-radius:8px;overflow:auto;max-height:460px}
 .caveat{color:var(--dim);font-size:12px;margin-top:8px;font-style:italic}
</style></head><body>
<header><h1>Palli Sahayak &mdash; Study Audit</h1>
<span class="sub">read-only &middot; masked participants &middot; scrubbed free text
 &middot; generated __GENERATED__</span></header>
<div class="wrap">__BODY__</div></body></html>"""


def _card(key: str, value: Any, note: str = "", tone: str = "") -> str:
    cls = f" {tone}" if tone else ""
    note_html = f'<div class="n">{note}</div>' if note else ""
    return (
        f'<div class="card"><div class="k">{key}</div>'
        f'<div class="v{cls}">{value}</div>{note_html}</div>'
    )


def _tag(text: str, tone: str) -> str:
    return f'<span class="tag t-{tone}">{text}</span>'


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@router.get("/api")
async def dashboard_api(day: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")) -> Dict[str, Any]:
    """Full dashboard payload as JSON."""
    return study_outcomes.build_dashboard(day=day)


@router.get("/api/interactions")
async def browse_interactions(
    day: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    participant: Optional[str] = None,
    language: Optional[str] = None,
    flagged: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """
    Browse logged interactions for spot-checking.

    Reads only pseudonymised identifiers, which is what is on disk. There is no
    route here that can resolve a pseudonym back to a participant: the mapping is
    one-way by design.
    """
    entries = study_outcomes.entries_from_log(day=day)

    if participant:
        entries = [e for e in entries if e.get("participant_id") == participant]
    if language:
        entries = [e for e in entries if e.get("language") == language]
    if flagged == "dosage":
        entries = [e for e in entries if e.get("dosage_blocked")]
    elif flagged == "emergency":
        entries = [e for e in entries if e.get("emergency_detected")]
    elif flagged == "unsupported":
        entries = [e for e in entries if e.get("source_count", 0) == 0]
    elif flagged == "unapproved_release":
        entries = [e for e in entries if e.get("release_approved") is False]

    entries.sort(key=lambda e: e.get("occurred_at", 0), reverse=True)
    return {
        "total": len(entries),
        "offset": offset,
        "limit": limit,
        "returned": entries[offset:offset + limit],
    }


@router.get("/health")
async def dashboard_health() -> Dict[str, Any]:
    """Health of the audit view itself."""
    log_dir = Path("./data/study/logs")
    return {
        "status": "ok",
        "log_dir_exists": log_dir.exists(),
        "day_files": len(list(log_dir.glob("interactions-*.jsonl"))) if log_dir.exists() else 0,
        "read_only": True,
    }


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

@router.get("/", response_class=HTMLResponse)
async def dashboard_page(day: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")) -> str:
    d = study_outcomes.build_dashboard(day=day)
    adoption = d["adoption"]
    safety = d["safety"]
    latency = d["latency"]
    equity = d["language_equity"]
    supervision = d["supervision"]
    release = d["release"]

    parts: List[str] = []

    # -- release gate: the single most important thing on the page ----------
    if not release["fit_for_participant_use"]:
        parts.append(
            '<div class="banner b-red"><b>Not approved for participant use.</b> '
            f'Release <code>{release["release_id"]}</code> is status '
            f'<code>{release["status"]}</code>; clinical and technical lead '
            f'approvals are {"signed" if release["approved"] else "NOT signed"}. '
            "Protocol 7.7 requires signed approval and passed 7.7 checks before "
            "any participant touches this build.</div>"
        )
    if release["drift_detected"]:
        parts.append(
            '<div class="banner b-amber"><b>Release drift detected.</b> The running '
            "system does not match the approved record in: "
            + "; ".join(release["drift_components"])
            + f' &middot; {release["unapproved_interactions"]} interaction(s) were '
            "served under the mismatch.</div>"
        )
    parts.append(
        '<div class="banner b-blue">Figures below are computed from the system log '
        "under the study's own definitions. Free text is scrubbed of phone numbers, "
        "Aadhaar, ABHA and email before it reaches disk; participant ids are "
        "one-way HMAC pseudonyms. <b>Personal names are not removed</b> &mdash; no "
        "reliable method exists without a name gazetteer &mdash; so treat query and "
        "response text as potentially identifying.</div>"
    )

    # -- headline -----------------------------------------------------------
    parts.append('<h2>Headline</h2><div class="grid">')
    parts.append(_card(
        "Substantive interactions", f"{d['total_substantive']:,}",
        f"{d['total_logged_interactions']:,} logged &middot; "
        f"{adoption['excluded_interactions']:,} excluded by protocol 4.1",
    ))
    tone = "ok" if adoption["meets_programme_benchmark"] else "warn"
    parts.append(_card(
        "Sustained adoption", _pct(adoption["sustained_adoption_rate"]),
        f"{adoption['sustained_participants']} of "
        f"{adoption['denominator_participants_with_activity']} participants &middot; "
        f"target &gt;{_pct(adoption['target'])}", tone,
    ))
    parts.append(_card(
        "Calls / ASHA / week", adoption["calls_per_asha_per_week"],
        f"{adoption['active_asha_workers']} active ASHA workers",
    ))
    utone = "bad" if safety["unsupported_answer_rate"] > 0.15 else "ok"
    parts.append(_card(
        "Unsupported-answer rate", _pct(safety["unsupported_answer_rate"]),
        f"{safety['unsupported_answer_count']} substantive answers with no source",
        utone,
    ))
    parts.append(_card(
        "Emergencies detected", safety["emergency_detected_count"],
        "critical+high+medium", "warn" if safety["emergency_detected_count"] else "",
    ))
    dtone = "warn" if safety["dosage_restriction_applied"] else ""
    parts.append(_card(
        "Dosage refusals", safety["dosage_restriction_applied"],
        "ADR 0004 dose boundary fired", dtone,
    ))
    e2e = latency["end_to_end"]
    ltone = "ok" if latency["voice_budget_met"] else "bad"
    parts.append(_card(
        "Median voice latency", f"{e2e.get('median_ms', 0) / 1000:.1f}s"
        if e2e.get("median_ms") else "&mdash;",
        f"p95 {e2e.get('p95_ms', 0) / 1000:.1f}s &middot; budget 13.0s", ltone,
    ))
    parts.append(_card(
        "Release drift", "yes" if release["drift_detected"] else "no",
        f"{release['unapproved_interactions']:,} interactions on an unapproved build",
        "bad" if release["drift_detected"] else "ok",
    ))
    parts.append("</div>")

    # -- adoption ----------------------------------------------------------
    parts.append("<h2>Adoption &mdash; protocol 4.1</h2>")
    parts.append(
        "<p class=\"caveat\">Sustained use = at least one substantive care-related "
        "interaction in at least two distinct weeks of the preceding 28 days. "
        "Training demonstrations, test messages, accidental activations and "
        "duplicate retries are excluded. The denominator is reported first, as "
        "the protocol requires.</p>"
    )
    exclusions = adoption["exclusion_reasons"]
    parts.append(
        '<div class="grid">'
        + _card("Denominator", f"{adoption['denominator_participants_with_activity']}",
                "participants with any activity in window")
        + _card("Sustained", f"{adoption['sustained_participants']}",
                f"in &ge;2 distinct weeks of {adoption['window_days']}d")
        + _card("Excluded rows", f"{adoption['excluded_interactions']:,}",
                ", ".join(f"{k}: {v}" for k, v in exclusions.items()) or "none")
        + "</div>"
    )

    if adoption["per_participant"]:
        rows = "".join(
            "<tr>"
            f"<td><code>{pid}</code></td>"
            f"<td>{_tag(d['care_role'] or 'unknown', 'mut')}</td>"
            f"<td>{d['site_id'] or '&mdash;'}</td>"
            f"<td class='num'>{d['interactions']}</td>"
            f"<td class='num'>{d['distinct_weeks']}</td>"
            f"<td>{_tag('sustained' if d['sustained'] else 'not sustained', 'ok' if d['sustained'] else 'mut')}</td>"
            f"<td class='sub'>{', '.join(d['languages'])}</td>"
            "</tr>"
            for pid, d in sorted(
                adoption["per_participant"].items(),
                key=lambda kv: (-kv[1]["interactions"], kv[0]),
            )[:60]
        )
        parts.append(
            "<table><thead><tr><th>Participant</th><th>Care Role</th><th>Site</th>"
            "<th class='num'>Calls</th><th class='num'>Weeks</th><th>Status</th>"
            "<th>Languages</th></tr></thead><tbody>" + rows + "</tbody></table>"
        )
    else:
        parts.append(
            '<div class="banner b-blue">No interactions logged yet. Once the server '
            "is running and the mobile API is exercised, adoption populates "
            "automatically.</div>"
        )

    # -- safety ------------------------------------------------------------
    parts.append("<h2>Safety</h2>")
    parts.append(
        f'<p class="caveat">{safety["hallucination_rate_caveat"]}</p>'
    )
    parts.append(
        "<table><thead><tr><th>Measure</th><th>Value</th><th>Detail</th></tr></thead><tbody>"
        f"<tr><td>Unsupported-answer rate</td><td class='num'>{_pct(safety['unsupported_answer_rate'])}</td>"
        f"<td class='sub'>{safety['unsupported_answer_count']} of {safety['substantive_interactions']} substantive answers returned zero sources</td></tr>"
        f"<tr><td>Emergency detection rate</td><td class='num'>{_pct(safety['emergency_detection_rate'])}</td>"
        f"<td class='sub'>{safety['emergency_detected_count']} flagged</td></tr>"
        f"<tr><td>Emergency levels</td><td class='num'>{json.dumps(safety['emergency_levels'])}</td>"
        "<td class='sub'>detector severities</td></tr>"
        f"<tr><td>Dosage restriction fired</td><td class='num'>{_pct(safety['dosage_restriction_rate'])}</td>"
        f"<td class='sub'>{json.dumps(safety['dosage_block_reasons'])}</td></tr>"
        f"<tr><td>Evidence levels</td><td class='num'>{json.dumps(safety['evidence_levels'])}</td>"
        "<td class='sub'>A–E grading</td></tr>"
        f"<tr><td>Validation status</td><td class='num'>{json.dumps(safety['validation_statuses'])}</td>"
        "<td class='sub'>including dosage_restricted</td></tr>"
        f"<tr><td>Zero undetected critical events</td><td class='num'>"
        f"{'yes' if safety['feasibility']['zero_undetected_critical_safety_events'] else 'NO'}</td>"
        "<td class='sub'>grant feasibility threshold</td></tr>"
        f"<tr><td>Errors</td><td class='num'>{safety['error_count']}</td><td class='sub'>failed interactions</td></tr>"
        "</tbody></table>"
    )

    # -- latency -----------------------------------------------------------
    parts.append("<h2>Latency &mdash; grant technical monitoring</h2>")
    if latency["by_stage"]:
        rows = "".join(
            f"<tr><td><code>{stage}</code></td>"
            f"<td class='num'>{v.get('median_ms','&mdash;')}</td>"
            f"<td class='num'>{v.get('mean_ms','&mdash;')}</td>"
            f"<td class='num'>{v.get('p95_ms','&mdash;')}</td>"
            f"<td class='num'>{v.get('max_ms','&mdash;')}</td>"
            f"<td class='num'>{v.get('count','0')}</td></tr>"
            for stage, v in sorted(latency["by_stage"].items())
        )
        parts.append(
            "<table><thead><tr><th>Stage</th><th class='num'>Median ms</th>"
            "<th class='num'>Mean ms</th><th class='num'>p95 ms</th>"
            "<th class='num'>Max ms</th><th class='num'>n</th></tr></thead><tbody>"
            + rows + "</tbody></table>"
        )
        parts.append(
            '<p class="caveat">Budget is 13,000 ms end-to-end for a voice '
            f"interaction. Currently met: {latency['voice_budget_met']}.</p>"
        )
    else:
        parts.append('<div class="banner b-blue">No latency recorded yet.</div>')

    # -- equity ------------------------------------------------------------
    parts.append("<h2>Language equity &mdash; all metrics by language</h2>")
    parts.append(f'<p class="caveat">{equity["caveat"]}</p>')
    if equity["by_language"]:
        rows = "".join(
            f"<tr><td><code>{lang}</code></td>"
            f"<td class='num'>{v['interactions']}</td>"
            f"<td class='num'>{v['participants']}</td>"
            f"<td class='num'>{_pct(v['unsupported_answer_rate'])}</td>"
            f"<td class='num'>{_pct(v['dosage_restriction_rate'])}</td>"
            f"<td class='num'>{v['emergency_count']}</td>"
            f"<td class='num'>{v['mean_latency_ms'] or '&mdash;'}</td>"
            f"<td>{_tag(v['reporting_note'], 'warn' if v['underpowered'] else 'ok')}</td></tr>"
            for lang, v in equity["by_language"].items()
        )
        flag = (
            _tag("equity gap flagged", "bad")
            if equity["equity_flagged"] else _tag("no gap flagged", "ok")
        )
        parts.append(
            f'<div style="margin-bottom:8px">{flag} &nbsp; unsupported-rate '
            f"disparity across languages: {equity['unsupported_rate_disparity']}</div>"
        )
        parts.append(
            "<table><thead><tr><th>Language</th><th class='num'>Calls</th>"
            "<th class='num'>Participants</th><th class='num'>Unsupported</th>"
            "<th class='num'>Dosage block</th><th class='num'>Emerg</th>"
            "<th class='num'>Mean ms</th><th>Reporting</th></tr></thead><tbody>"
            + rows + "</tbody></table>"
        )
    else:
        parts.append('<div class="banner b-blue">No language data yet.</div>')

    # -- supervision -------------------------------------------------------
    flagged = supervision["flagged_pool"]
    parts.append("<h2>Supervision &mdash; tiered expert sampling</h2>")
    parts.append(
        "<table><thead><tr><th>Tier</th><th>Rate</th><th>Pool</th>"
        "<th class='num'>Interactions</th></tr></thead><tbody>"
        f"<tr><td>Random</td><td>5%</td><td class='sub'>all substantive</td>"
        f"<td class='num'>{safety['substantive_interactions']}</td></tr>"
        f"<tr><td>Flagged</td><td>50%</td>"
        f"<td class='sub'>{flagged['low_confidence_under_0_7']} low confidence "
        f"+ {flagged['no_retrieved_sources']} with no source</td>"
        f"<td class='num'>{flagged['total_flagged']}</td></tr>"
        f"<tr><td>Critical</td><td>100%</td>"
        "<td class='sub'>substantive with zero retrieved sources</td>"
        f"<td class='num'>{supervision['critical_events']}</td></tr>"
        "</tbody></table>"
    )

    # -- release -----------------------------------------------------------
    parts.append("<h2>Release provenance &mdash; protocol 7.8</h2>")
    parts.append(
        "<table><thead><tr><th>Field</th><th>Value</th></tr></thead><tbody>"
        f"<tr><td>Release</td><td><code>{release['release_id']}</code></td></tr>"
        f"<tr><td>Status</td><td>{_tag(release['status'], 'warn' if release['status'] != 'activated' else 'ok')}</td></tr>"
        f"<tr><td>Approvals signed</td><td>{_tag('yes' if release['approved'] else 'no', 'ok' if release['approved'] else 'bad')}</td></tr>"
        f"<tr><td>Fit for participant use</td><td>{_tag('no' if not release['fit_for_participant_use'] else 'yes', 'bad' if not release['fit_for_participant_use'] else 'ok')}</td></tr>"
        f"<tr><td>Drift</td><td>{_tag('detected' if release['drift_detected'] else 'none', 'bad' if release['drift_detected'] else 'ok')}</td></tr>"
        f"<tr><td>Drift records persisted</td><td class='num'>{release['drift_records']}</td></tr>"
        f"<tr><td>Releases seen in log</td><td class='sub'>"
        f"{', '.join(release['releases_observed_in_log']) or '&mdash;'}</td></tr>"
        f"<tr><td>Interactions on unapproved build</td><td class='num'>"
        f"{release['unapproved_interactions']:,}</td></tr>"
        "</tbody></table>"
    )
    if release["drift_components"]:
        parts.append("<pre>" + "\n".join(release["drift_components"]) + "</pre>")

    # -- audit trail -------------------------------------------------------
    parts.append("<h2>Audit</h2>")
    parts.append(
        f'<p class="caveat">Browse the masked log at '
        f'<code>/admin/study/api/interactions?flagged=dosage</code>, '
        f'<code>...=emergency</code>, <code>...=unsupported</code>, or '
        f'<code>...=unapproved_release</code>. This view is read-only; the log is '
        f"append-only and there is no endpoint that edits or deletes a row. Raw "
        f"JSON: <code>/admin/study/api</code>.</p>"
    )
    if day:
        parts.append(f'<p class="caveat">Filtered to day: <code>{day}</code>. '
                     "<code>?day=YYYY-MM-DD</code> to change.</p>")

    body = "\n".join(parts)
    return PAGE_TEMPLATE.replace("__GENERATED__", d["generated_at"]).replace("__BODY__", body)