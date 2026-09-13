"""Strict contracts for the clinical gateway (not a clinical decision model)."""

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import re
from typing import Any, Dict, List, Literal, Optional
import unicodedata

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GovernanceError(ValueError):
    """A safe, machine-readable denial reason without request contents."""

    def __init__(self, code: str, http_status: int = 409):
        self.code = code
        self.http_status = http_status
        super().__init__(code)


@dataclass(frozen=True)
class Actor:
    id: str
    tenant: str
    roles: frozenset[str]
    purposes: frozenset[str] = frozenset({"information", "quality_review"})
    clinical_authority: bool = False

    def require(self, *roles: str) -> None:
        if not self.id or not self.tenant or not self.roles.intersection(roles):
            raise GovernanceError("role_not_authorized", 403)

    def permit(self, purpose: str) -> None:
        if purpose not in self.purposes:
            raise GovernanceError("purpose_not_authorized", 403)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        raise GovernanceError("invalid_timestamp", 422) from None
    if result.tzinfo is None:
        raise GovernanceError("timezone_required", 422)
    return result


class Rights(StrictModel):
    """Recorded authorization, not independent legal verification."""

    authorized: Literal[True]
    authorization_reference: str = Field(min_length=1, max_length=500)
    purposes: List[str] = Field(min_length=1, max_length=20)
    external_processing: bool = False
    processors: List[str] = Field(default_factory=list, max_length=30)
    redistribution: bool = False
    valid_until: str

    @model_validator(mode="after")
    def validate_expiry(self):
        timestamp(self.valid_until)
        if self.external_processing and not self.processors:
            raise ValueError("External processing requires named permitted processors")
        return self


class EvidenceAnchor(StrictModel):
    source_id: str
    normalized_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    excerpt: str = Field(min_length=1, max_length=20000)
    page: Optional[int] = Field(default=None, ge=1)
    block_id: Optional[str] = None
    role: Literal[
        "passage", "table_cell", "header", "footnote", "algorithm", "exception"
    ] = "passage"

    @model_validator(mode="after")
    def ordered(self):
        if self.end <= self.start:
            raise ValueError("Evidence interval must be ordered")
        return self


class Condition(StrictModel):
    """Bounded boolean expression; a missing fact never implies permission."""

    op: Literal[
        "all", "any", "not", "eq", "ne", "lt", "lte", "gt", "gte", "in", "not_in"
    ]
    field: Optional[str] = Field(default=None, pattern=r"^[a-z][a-z0-9_]{0,79}$")
    value: Any = None
    children: List["Condition"] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def valid_shape(self):
        if self.op in ("all", "any", "not"):
            if self.field is not None or self.value is not None or not self.children:
                raise ValueError("Boolean conditions require children only")
            if self.op == "not" and len(self.children) != 1:
                raise ValueError("not requires one child")
        else:
            if not self.field or self.children or self.value is None:
                raise ValueError("A comparison needs a field and explicit value")
            values = self.value if isinstance(self.value, list) else [self.value]
            if self.op in ("in", "not_in") and (
                not isinstance(self.value, list) or not values
            ):
                raise ValueError("Membership requires a nonempty list")
            if self.op not in ("in", "not_in") and isinstance(self.value, list):
                raise ValueError("Scalar comparison requires a scalar")
            for value in values:
                if type(value) not in (str, int, float, bool):
                    raise ValueError("Only scalar values are supported")
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("Nonfinite values are not supported")
            if self.op in ("lt", "lte", "gt", "gte") and type(self.value) not in (
                int,
                float,
            ):
                raise ValueError("Ordered comparison requires a number")
        return self


def condition_fields(condition: Condition, depth: int = 0) -> set[str]:
    if depth > 10:
        raise GovernanceError("condition_too_deep", 422)
    if condition.field:
        return {condition.field}
    return set().union(
        *(condition_fields(child, depth + 1) for child in condition.children)
    )


def equal(left: Any, right: Any) -> bool:
    """Avoid Python's True == 1 when checking clinical variables."""
    if type(left) in (int, float) and type(right) in (int, float):
        return left == right
    return type(left) is type(right) and left == right


def applicability(condition: Condition, context: Dict[str, Any]) -> Dict[str, Any]:
    condition_fields(condition)  # Validate depth before recursive evaluation.

    def evaluate(node):
        if node.children:
            results = [evaluate(child) for child in node.children]
            values = [value for value, _ in results]
            missing = set().union(*(gap for _, gap in results))
            if node.op == "not":
                return (None if values[0] is None else not values[0]), missing
            if node.op == "all":
                return (
                    False if False in values else None if None in values else True
                ), missing
            return (
                True if True in values else None if None in values else False
            ), missing
        actual = context.get(node.field)
        if actual is None or type(actual) not in (str, bool, int, float):
            return None, {node.field}
        if isinstance(actual, float) and not math.isfinite(actual):
            return None, {node.field}
        if node.op in ("lt", "lte", "gt", "gte"):
            if type(actual) not in (int, float):
                return None, {node.field}
            return {
                "lt": actual < node.value,
                "lte": actual <= node.value,
                "gt": actual > node.value,
                "gte": actual >= node.value,
            }[node.op], set()
        matched = (
            any(equal(actual, value) for value in node.value)
            if node.op in ("in", "not_in")
            else equal(actual, node.value)
        )
        return (not matched if node.op in ("ne", "not_in") else matched), set()

    applies, missing = evaluate(condition)
    return {
        "status": "unknown"
        if applies is None
        else "applies"
        if applies
        else "does_not_apply",
        "missing_variables": sorted(missing),
    }


class RecommendationDraft(StrictModel):
    statement: str = Field(min_length=1, max_length=10000)
    evidence: List[EvidenceAnchor] = Field(min_length=1, max_length=50)
    condition: Condition
    qualifiers: Dict[str, str] = Field(default_factory=dict)
    exceptions: List[str] = Field(default_factory=list, max_length=50)
    alternatives: List[str] = Field(default_factory=list, max_length=50)
    purposes: List[str] = Field(min_length=1, max_length=20)
    explanations: Dict[str, str] = Field(default_factory=dict)
    keywords: List[str] = Field(min_length=1, max_length=100)
    high_risk: bool = False
    valid_until: str
    # Untrusted imported metadata is retained only as data, never as authority.
    import_metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_draft(self):
        condition_fields(self.condition)
        timestamp(self.valid_until)
        if any(
            not re.fullmatch(r"[a-z]{2,3}(?:-[A-Z]{2})?", key)
            or not text.strip()
            or len(text) > 10000
            for key, text in self.explanations.items()
        ):
            raise ValueError("Invalid language or explanation")
        if not any(keyword.strip() for keyword in self.keywords):
            raise ValueError("Search keywords required")
        return self


class ReviewPolicy(StrictModel):
    mode: Literal[
        "off", "sampled", "user_requested", "policy_triggered", "required"
    ] = "off"
    sample_percent: int = Field(default=10, ge=0, le=100)
    trigger_high_risk: bool = True
    retain_request_seconds: int = Field(default=0, ge=0, le=86400 * 30)
    session_ttl_seconds: int = Field(default=3600, ge=60, le=86400)


def unicode_offsets(text: str, start: int, end: int) -> Dict[str, List[int]]:
    if (
        type(start) is not int
        or type(end) is not int
        or not 0 <= start < end <= len(text)
    ):
        raise GovernanceError("invalid_codepoint_interval", 422)
    return {
        "codepoints": [start, end],
        "utf8": [len(text[:start].encode("utf-8")), len(text[:end].encode("utf-8"))],
        "utf16": [
            len(text[:start].encode("utf-16-le")) // 2,
            len(text[:end].encode("utf-16-le")) // 2,
        ],
    }


def terms(text: str) -> set[str]:
    """Keep Indic combining marks within words; no clinical interpretation."""
    words, word = set(), []
    for char in text.casefold():
        if unicodedata.category(char)[0] in "LMN":
            word.append(char)
        elif word:
            words.add("".join(word))
            word = []
    if word:
        words.add("".join(word))
    return words
