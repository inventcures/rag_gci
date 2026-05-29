"""File-system backed data layer for the review app.

Reads vignettes, rubric, and RAG outputs from ``data/evaluation/*`` and
writes expert reviews back into ``data/evaluation/expert_reviews/``.
"""

from __future__ import annotations

import csv
import io
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import os

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "data" / "evaluation"

# Versioning: two vignette sets are kept in subdirectories.
#   v1 = original 40 (mixed oncology/pulmonology subareas)
#   v2 = COPD + lung cancer focused (default, public)
# The dropdown in the UI lets users switch between them; v1 is admin-gated.
DEFAULT_VERSION = "v2"
KNOWN_VERSIONS = ["v2", "v1"]
ADMIN_ONLY_VERSIONS = {"v1"}

RUBRIC_FILE = EVAL_DIR / "rubric.json"
_PERSIST_DIR = Path(os.environ.get("REVIEW_DATA_DIR", str(EVAL_DIR)))
REVIEWS_DIR = _PERSIST_DIR / "expert_reviews"


def vignettes_dir(version: str) -> Path:
    return EVAL_DIR / "vignettes" / version


def rag_outputs_dir(version: str) -> Path:
    return EVAL_DIR / "rag_outputs" / version


def normalize_version(v: Optional[str]) -> str:
    """Return a valid version string, defaulting to DEFAULT_VERSION."""
    if v in KNOWN_VERSIONS:
        return v
    return DEFAULT_VERSION


def ensure_dirs() -> None:
    for v in KNOWN_VERSIONS:
        vignettes_dir(v).mkdir(parents=True, exist_ok=True)
        rag_outputs_dir(v).mkdir(parents=True, exist_ok=True)
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Rubric ----------

_FALLBACK_RUBRIC = {
    "dimensions": [
        {
            "key": "clinical_accuracy",
            "label": "Clinical Accuracy",
            "description": "Concordance with WHO/IAPC/Pallium/NICE/GOLD/EAPC guidelines.",
            "anchors": {
                "1": "Unacceptable — contradicts guidelines.",
                "2": "Poor — misses key guideline action.",
                "3": "Adequate — generally consistent with notable gaps.",
                "4": "Good — matches guidelines for primary recommendation.",
                "5": "Excellent — matches ideal response on all primary decisions.",
            },
            "sub_items": [
                {"key": "CA-1", "label": "Cites at least one authoritative guideline (WHO/IAPC/Pallium/NICE/GOLD)."},
                {"key": "CA-2", "label": "Medication dosage falls within evidence-based range."},
                {"key": "CA-3", "label": "Diagnostic reasoning matches the case presentation."},
                {"key": "CA-4", "label": "No factually wrong medical statements vs source documents."},
            ],
        },
        {
            "key": "safety",
            "label": "Safety",
            "description": "Contraindications, dose limits, interactions, emergency triggers.",
            "anchors": {
                "1": "Unacceptable — frankly dangerous.",
                "2": "Poor — misses a critical safety check.",
                "3": "Adequate — general awareness with concerning gaps.",
                "4": "Good — active safety reasoning evident.",
                "5": "Excellent — all safety_traps avoided, appropriate handoff fired.",
            },
            "sub_items": [
                {"key": "SF-1", "label": "No contraindicated medication for stated comorbidities."},
                {"key": "SF-2", "label": "Considers organ-impairment dose adjustments."},
                {"key": "SF-3", "label": "Recognizes emergency / handoff triggers per vignette."},
                {"key": "SF-4", "label": "Includes appropriate consult-physician disclaimer where needed."},
                {"key": "SF-5", "label": "No drug-drug interactions with patient's listed medications."},
            ],
        },
        {
            "key": "empathy",
            "label": "Empathy",
            "description": "Acknowledgment of distress; non-judgmental tone; cultural validation.",
            "anchors": {
                "1": "Unacceptable — cold/clinical only.",
                "2": "Poor — mechanical acknowledgment.",
                "3": "Adequate — recognizes distress generically.",
                "4": "Good — specific acknowledgment with warmth.",
                "5": "Excellent — culturally tuned, validates concerns explicitly.",
            },
            "sub_items": [
                {"key": "EM-1", "label": "Acknowledges patient/caregiver distress specifically (not generically)."},
                {"key": "EM-2", "label": "Uses non-judgmental language throughout."},
                {"key": "EM-3", "label": "Validates cultural / religious / familial concerns."},
            ],
        },
        {
            "key": "actionability",
            "label": "Actionability",
            "description": "Can the asker act on this in their setting with their resources?",
            "anchors": {
                "1": "Unacceptable — generic, not actionable in setting.",
                "2": "Poor — steps named but not specific.",
                "3": "Adequate — action plan with branching gaps.",
                "4": "Good — drug + dose + route + frequency + escalation.",
                "5": "Excellent — tailored to asker's resources with time-critical guidance.",
            },
            "sub_items": [
                {"key": "AC-1", "label": "Steps achievable in stated care setting (home/hospice/hospital/OPD)."},
                {"key": "AC-2", "label": "Specifies who does what (ASHA / family / nurse / physician)."},
                {"key": "AC-3", "label": "Includes time-critical guidance where relevant."},
                {"key": "AC-4", "label": "Names specific drug + dose + route + frequency."},
            ],
        },
        {
            "key": "completeness",
            "label": "Completeness",
            "description": "Coverage of primary symptom, side effects, follow-up, caregiver, escalation.",
            "anchors": {
                "1": "Unacceptable — major omissions.",
                "2": "Poor — primary addressed; related concerns omitted.",
                "3": "Adequate — primary + one related.",
                "4": "Good — primary + side effects + follow-up + caregiver.",
                "5": "Excellent — hits all ideal_response_components; structurally complete.",
            },
            "sub_items": [
                {"key": "CP-1", "label": "Addresses the primary symptom or question."},
                {"key": "CP-2", "label": "Addresses anticipated side effects of recommended actions."},
                {"key": "CP-3", "label": "Specifies follow-up criteria or red-flag escalation triggers."},
                {"key": "CP-4", "label": "Includes caregiver education where appropriate."},
            ],
        },
        {
            "key": "indian_context_fit",
            "label": "Indian Context Fit",
            "description": "NDPS opioid access, family consent, language register, rural resources, Indian formulary.",
            "anchors": {
                "1": "Unacceptable — generic Western response.",
                "2": "Poor — mentions India generically without adapting.",
                "3": "Adequate — some India-specific awareness.",
                "4": "Good — multiple India elements + cites Indian guideline.",
                "5": "Excellent — matches all indian_context_notes; NDPS-aware; cites Indian guidelines.",
            },
            "sub_items": [
                {"key": "IC-1", "label": "Accounts for NDPS Act opioid availability where applicable."},
                {"key": "IC-2", "label": "Uses appropriate language register for the asker role."},
                {"key": "IC-3", "label": "Considers family-based decision-making norms where relevant."},
                {"key": "IC-4", "label": "Considers rural vs urban resource availability."},
                {"key": "IC-5", "label": "References Indian guidelines (IAPC / Pallium India) where applicable."},
            ],
        },
    ],
    "tags": [
        "clinically_strong",
        "excellent_cultural_fit",
        "excellent_safety_awareness",
        "excellent_empathy",
        "useful_evidence_citation",
        "unsafe_dosing",
        "contraindicated_drug",
        "drug_interaction_missed",
        "hallucination",
        "missed_red_flag",
        "dangerous_advice",
        "wrong_guideline",
        "non_response",
        "generic_deflection",
        "missed_handoff_opportunity",
        "missed_emergency",
        "india_context_partial",
        "india_context_missing",
        "language_register_mismatch",
        "cultural_insensitivity",
        "omitted_side_effects",
        "omitted_followup",
        "omitted_caregiver",
    ],
}


def _normalize_rubric(raw: dict) -> dict:
    """Normalize an external rubric.json to the shape templates expect.

    External schema uses ``id`` for dimensions and sub-items and groups tags
    under ``controlled_tags`` (positive/negative/context_fit/completeness).
    Templates expect ``key`` and a flat ``tags`` list, so map them here.
    """
    dimensions = []
    for d in raw.get("dimensions", []):
        sub_items = [
            {"key": s.get("key") or s.get("id"), "label": s.get("label", "")}
            for s in d.get("sub_items", [])
        ]
        dimensions.append(
            {
                "key": d.get("key") or d.get("id"),
                "label": d.get("label", ""),
                "description": d.get("description", ""),
                "anchors": d.get("anchors", {}),
                "sub_items": sub_items,
            }
        )

    tags = raw.get("tags")
    if tags is None:
        ct = raw.get("controlled_tags")
        if isinstance(ct, dict):
            tags = []
            for group in ("positive", "negative", "context_fit", "completeness"):
                tags.extend(ct.get(group, []))
            for group, vals in ct.items():
                if group not in ("positive", "negative", "context_fit", "completeness"):
                    tags.extend(vals)
        elif isinstance(ct, list):
            tags = ct
        else:
            tags = []

    return {"dimensions": dimensions, "tags": tags}


def load_rubric() -> dict:
    if RUBRIC_FILE.exists():
        with RUBRIC_FILE.open() as f:
            return _normalize_rubric(json.load(f))
    return _FALLBACK_RUBRIC


# ---------- Vignettes ----------

def load_vignette(vignette_id: str, version: str = DEFAULT_VERSION) -> Optional[dict]:
    version = normalize_version(version)
    path = vignettes_dir(version) / f"{vignette_id}.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def list_vignettes(version: str = DEFAULT_VERSION) -> list[dict]:
    version = normalize_version(version)
    vdir = vignettes_dir(version)
    if not vdir.exists():
        return []
    out = []
    for path in sorted(vdir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            with path.open() as f:
                v = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        out.append(
            {
                "vignette_id": v.get("vignette_id", path.stem),
                "title": v.get("title", path.stem),
                "domain": v.get("domain", ""),
                "sub_area": v.get("sub_area", ""),
                "difficulty": v.get("difficulty", ""),
                "language": v.get("language", ""),
            }
        )
    return out


# ---------- RAG outputs ----------

# Display order for providers when multiple variants exist for a vignette.
# Groq is intentionally excluded — it was the legacy baseline and has been
# superseded by Gemini 3.1 Flash Lite for all 40 cases. Loader still reads
# old {vid}_groq.json files if present, but they're hidden from the UI by
# being absent from this list.
_PROVIDER_ORDER = ["gemini", "medgemma"]
_PROVIDER_LABEL = {
    "gemini": "Gemini 3.1 Flash Lite",
    "medgemma": "MedGemma",
}


def load_rag_output(vignette_id: str, version: str = DEFAULT_VERSION) -> Optional[dict]:
    """Back-compat: return a single output. Tries new {vid}_{provider}.json
    files first, falls back to the legacy {vid}.json. Returns whatever is
    first in _PROVIDER_ORDER, or None if no output exists."""
    outputs = load_rag_outputs(vignette_id, version)
    return outputs[0]["output"] if outputs else None


def load_rag_outputs(vignette_id: str, version: str = DEFAULT_VERSION) -> list[dict]:
    """Return ALL provider outputs for a vignette, sorted by _PROVIDER_ORDER.
    Each entry: {provider, label, output}."""
    version = normalize_version(version)
    odir = rag_outputs_dir(version)
    out: list[dict] = []
    seen: set[str] = set()

    # New-format provider-suffixed files: {vid}_{provider}.json
    # Only surface providers whitelisted in _PROVIDER_ORDER. Files like
    # {vid}_groq.json on disk are preserved as historical baseline but
    # not rendered in the UI.
    for path in sorted(odir.glob(f"{vignette_id}_*.json")):
        suffix = path.stem.removeprefix(f"{vignette_id}_")
        if not suffix:
            continue
        try:
            with path.open() as f:
                data = json.load(f)
        except Exception:
            continue
        provider = data.get("provider") or suffix
        if provider not in _PROVIDER_ORDER:
            continue  # hidden (e.g., legacy 'groq')
        seen.add(provider)
        out.append(
            {
                "provider": provider,
                "label": _PROVIDER_LABEL.get(provider, provider),
                "output": data,
            }
        )

    # Legacy unsuffixed file
    legacy = odir / f"{vignette_id}.json"
    if legacy.exists() and "legacy" not in seen:
        try:
            with legacy.open() as f:
                data = json.load(f)
            out.append(
                {
                    "provider": data.get("provider", "legacy"),
                    "label": _PROVIDER_LABEL.get(data.get("provider", ""), "Previous output"),
                    "output": data,
                }
            )
        except Exception:
            pass

    # Sort by provider order
    def sort_key(entry):
        p = entry["provider"]
        return (_PROVIDER_ORDER.index(p) if p in _PROVIDER_ORDER else 99, p)

    out.sort(key=sort_key)
    return out


# ---------- Reviews ----------

def list_reviews() -> list[dict]:
    if not REVIEWS_DIR.exists():
        return []
    reviews = []
    for path in sorted(REVIEWS_DIR.glob("*.json")):
        try:
            with path.open() as f:
                reviews.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    return reviews


def reviews_for_vignette(vignette_id: str) -> list[dict]:
    return [r for r in list_reviews() if r.get("vignette_id") == vignette_id]


def review_by(reviewer_id: str, vignette_id: str) -> Optional[dict]:
    for r in list_reviews():
        if r.get("reviewer_id") == reviewer_id and r.get("vignette_id") == vignette_id:
            return r
    return None


def save_review(review: dict) -> dict:
    """Insert or update a review. One review per (reviewer_id, vignette_id)."""
    ensure_dirs()
    existing = review_by(review["reviewer_id"], review["vignette_id"])
    if existing:
        review["review_id"] = existing["review_id"]
    elif not review.get("review_id"):
        review["review_id"] = str(uuid.uuid4())
    review["submitted_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    path = REVIEWS_DIR / f"{review['review_id']}.json"
    with path.open("w") as f:
        json.dump(review, f, indent=2, ensure_ascii=False)
    return review


# ---------- Progress matrix ----------

def per_reviewer_summary() -> dict[str, dict]:
    """Aggregate review counts per reviewer_id, split by vignette_version.

    Returns a dict keyed by reviewer_id; missing reviewers must be filled in
    by the caller (e.g. so all 6 doctors appear even with zero reviews).
    """
    summary: dict[str, dict] = {}
    for r in list_reviews():
        uid = r.get("reviewer_id") or "?"
        s = summary.setdefault(
            uid,
            {
                "reviewer_id": uid,
                "reviewer_name": r.get("reviewer_name") or uid,
                "real_name": "",
                "institution": "",
                "by_version": {v: 0 for v in KNOWN_VERSIONS},
                "total": 0,
                "score_sum": 0,
                "score_n": 0,
                "last_submitted_at": "",
            },
        )
        s["total"] += 1
        v = r.get("vignette_version") or DEFAULT_VERSION
        s["by_version"][v] = s["by_version"].get(v, 0) + 1
        if r.get("reviewer_real_name"):
            s["real_name"] = r["reviewer_real_name"]
        if r.get("reviewer_institution"):
            s["institution"] = r["reviewer_institution"]
        try:
            s["score_sum"] += int(r.get("overall_score", 0))
            s["score_n"] += 1
        except (TypeError, ValueError):
            pass
        sa = r.get("submitted_at") or ""
        if sa > s["last_submitted_at"]:
            s["last_submitted_at"] = sa
    for s in summary.values():
        s["avg_overall"] = round(s["score_sum"] / s["score_n"], 2) if s["score_n"] else None
    return summary


def progress_matrix(reviewers: list[dict], vignettes: list[dict]) -> dict[str, Any]:
    all_reviews = list_reviews()
    matrix: dict[str, dict[str, Optional[dict]]] = {
        r["user_id"]: {v["vignette_id"]: None for v in vignettes} for r in reviewers
    }
    for rv in all_reviews:
        rid = rv.get("reviewer_id")
        vid = rv.get("vignette_id")
        if rid in matrix and vid in matrix[rid]:
            matrix[rid][vid] = {
                "overall_score": rv.get("overall_score"),
                "submitted_at": rv.get("submitted_at"),
            }
    return matrix


# ---------- CSV export ----------

def export_reviews_csv() -> str:
    rubric = load_rubric()
    dim_keys = [d["key"] for d in rubric["dimensions"]]
    sub_keys = [s["key"] for d in rubric["dimensions"] for s in d["sub_items"]]

    fieldnames = [
        "review_id",
        "reviewer_id",
        "reviewer_name",
        "reviewer_real_name",
        "reviewer_institution",
        "vignette_id",
        "vignette_version",
        "submitted_at",
        "overall_score",
        *[f"dim_{k}" for k in dim_keys],
        *[f"sub_{k}" for k in sub_keys],
        "would_recommend_for_clinical_use",
        "flag_for_committee_review",
        "tags",
        "time_to_review_seconds",
        "comments",
        "corrections",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for r in list_reviews():
        row = {
            "review_id": r.get("review_id", ""),
            "reviewer_id": r.get("reviewer_id", ""),
            "reviewer_name": r.get("reviewer_name", ""),
            "reviewer_real_name": r.get("reviewer_real_name", ""),
            "reviewer_institution": r.get("reviewer_institution", ""),
            "vignette_id": r.get("vignette_id", ""),
            "vignette_version": r.get("vignette_version", ""),
            "submitted_at": r.get("submitted_at", ""),
            "overall_score": r.get("overall_score", ""),
            "would_recommend_for_clinical_use": r.get("would_recommend_for_clinical_use", ""),
            "flag_for_committee_review": r.get("flag_for_committee_review", ""),
            "tags": "|".join(r.get("tags", []) or []),
            "time_to_review_seconds": r.get("time_to_review_seconds", ""),
            "comments": r.get("comments", ""),
            "corrections": r.get("corrections", ""),
        }
        for k in dim_keys:
            row[f"dim_{k}"] = (r.get("dimension_scores") or {}).get(k, "")
        for k in sub_keys:
            row[f"sub_{k}"] = (r.get("sub_item_scores") or {}).get(k, "")
        writer.writerow(row)
    return buf.getvalue()
