#!/usr/bin/env python3
"""
Retrieval index identity and drift detection
=============================================
Identifies which embedding model a vector index was built with, and refuses to
search it when that model does not match the configured one.

Why this exists
---------------
The live index was built with `BAAI/bge-small-en-v1.5` at 384 dimensions while the
corpus it serves is palliative care guidance for users of eleven languages.
Measured across all eleven, it scores 1.00 hit@5 on English and **0.00 on every
Indic language**. Nothing errors. The system simply returns confident, unfounded
clinical advice to a Marathi question.

Nothing in the existing code would catch that. `get_or_create_collection` reuses
whichever collection is already on disk regardless of the embedding function it was
created with, and the auto-rebuild path triggers on corruption rather than on a
model change. So changing the configured model does not reindex, and querying
proceeds against stale vectors regardless.

This module makes the mismatch explicit and actionable.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ADR 0003, chosen by measurement: mean hit@5 0.96 against 0.72 for LaBSE and 0.09
# for the previous English-only index. See evaluation/retrieval_eval/results.json.
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_DIMENSION = 1024

# Dimensionality for models we may need to reason about. ChromaDB records nothing
# about which model built a collection, so the dimension is the only observable
# proxy available from the index itself.
KNOWN_DIMENSIONS = {
    "BAAI/bge-m3": 1024,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "google/embeddinggemma-300m": 768,
}


@dataclass(frozen=True)
class EmbeddingIdentity:
    """Which embedding model an index was built with."""

    model: str
    dimension: int

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("EmbeddingIdentity requires a model name")
        if self.dimension <= 0:
            raise ValueError("EmbeddingIdentity requires a positive dimension")

    def to_dict(self) -> Dict[str, Any]:
        return {"model": self.model, "dimension": self.dimension}


def load_identity_from_config(config: Optional[Dict[str, Any]]) -> EmbeddingIdentity:
    """
    Resolve the configured embedding identity.

    A single source of truth, so the pipeline, the reindex script and the release
    record cannot disagree about which model is meant to be running. The previous
    arrangement had three sources disagreeing at once.
    """
    section = (config or {}).get("embedding") or {}
    model = str(section.get("model") or DEFAULT_EMBEDDING_MODEL)
    dimension = section.get("dimension") or KNOWN_DIMENSIONS.get(model) or DEFAULT_EMBEDDING_DIMENSION
    return EmbeddingIdentity(model=model, dimension=int(dimension))


@dataclass
class IndexDrift:
    """
    The result of comparing the configured identity against the live index.

    Drift is never merely advisory here: a mismatch means every answer is being
    retrieved through the wrong geometry.
    """

    expected: EmbeddingIdentity
    observed: Optional[EmbeddingIdentity]
    reasons: List[str] = field(default_factory=list)

    @property
    def has_drift(self) -> bool:
        return bool(self.reasons)

    @property
    def remediation(self) -> str:
        if not self.reasons:
            return "No action required."
        return (
            "Reindex the corpus with the configured embedding model "
            f"({self.expected.model}, {self.expected.dimension}d) before querying. "
            "Do not query this index in the meantime: a mismatch returns confident "
            "but ungrounded results rather than an error."
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected": self.expected.to_dict(),
            "observed": self.observed.to_dict() if self.observed else None,
            "has_drift": self.has_drift,
            "reasons": list(self.reasons),
            "remediation": self.remediation,
        }

    def raise_if_drifted(self) -> None:
        """
        Refuse to continue on a mismatched index.

        Callers that search anyway produce answers that look healthy, which is the
        failure this whole module exists to prevent.
        """
        if self.has_drift:
            raise IndexMismatchError(self)


class IndexMismatchError(RuntimeError):
    """The live index was not built with the configured embedding model."""

    def __init__(self, drift: IndexDrift):
        observed = (
            f"{drift.observed.model} at {drift.observed.dimension}d"
            if drift.observed else "an unobservable index"
        )
        super().__init__(
            f"Embedding index mismatch: configured {drift.expected.model} at "
            f"{drift.expected.dimension}d, index was built with {observed}. "
            f"{drift.remediation}"
        )
        self.drift = drift


def detect_drift(
    expected: EmbeddingIdentity,
    observed: Optional[EmbeddingIdentity],
) -> IndexDrift:
    """
    Compare the configured identity against what the index actually holds.

    An unobservable index counts as drift rather than as agreement. Defaulting to
    "no news is good news" here would reintroduce exactly the silent failure this
    guards against.
    """
    reasons: List[str] = []

    if observed is None:
        reasons.append("index_unobservable")
        return IndexDrift(expected=expected, observed=None, reasons=reasons)

    if observed.dimension != expected.dimension:
        reasons.append("dimension_mismatch")

    if _normalise(observed.model) != _normalise(expected.model):
        reasons.append("model_mismatch")

    return IndexDrift(expected=expected, observed=observed, reasons=reasons)


def _normalise(model: str) -> str:
    """Compare model names tolerantly: a local path is the same model as its repo."""
    name = str(model).rstrip("/").split("/")[-1] if "/" in str(model) else str(model)
    return name.strip().lower()


def observe_index(collection: Any) -> Optional[EmbeddingIdentity]:
    """
    Read whatever identity the live index can tell us about itself.

    ChromaDB does not record which model built a collection, so the only reliable
    observation is dimensionality. Where that is ambiguous, we deliberately return
    None rather than guessing a model, and the caller treats unobservable as drift.
    """
    if collection is None:
        return None

    try:
        sample = collection.get(limit=1, include=["embeddings"])
    except Exception:
        logger.warning("Could not sample the index to observe its dimension", exc_info=True)
        return None

    embeddings = sample.get("embeddings") if isinstance(sample, dict) else None
    # ChromaDB returns a numpy array here, so truthiness is ambiguous and must not
    # be used. Length is the only safe test.
    if embeddings is None or len(embeddings) == 0 or len(embeddings[0]) == 0:
        # An empty collection is not yet drifted; it simply has nothing to say.
        return None

    dimension = len(embeddings[0])
    candidates = [m for m, d in KNOWN_DIMENSIONS.items() if d == dimension]
    model = candidates[0] if len(candidates) == 1 else f"unknown-{dimension}d"
    return EmbeddingIdentity(model=model, dimension=dimension)


def check_live_index(
    collection: Any,
    expected: EmbeddingIdentity,
) -> IndexDrift:
    """
    Compare the configured identity against a live ChromaDB collection.

    The collection's own metadata is preferred over inference from dimensionality,
    because a record written at index time is evidence rather than a guess.
    """
    observed = None
    try:
        metadata = getattr(collection, "metadata", None) or {}
        model = metadata.get("embedding_model")
        if model:
            observed = EmbeddingIdentity(
                model=str(model),
                dimension=int(metadata.get("embedding_dimension") or expected.dimension),
            )
    except Exception:
        logger.warning("Could not read index metadata", exc_info=True)

    if observed is None:
        observed = observe_index(collection)

    return detect_drift(expected, observed)


def write_index_metadata(
    collection: Any,
    identity: EmbeddingIdentity,
) -> None:
    """
    Record the identity on the collection at reindex time.

    Without this, the only available observation is dimensionality, and two models
    sharing 1024 dimensions are indistinguishable. Writing it now means a later
    mismatch is diagnosable rather than merely detectable.
    """
    collection.modify(metadata={
        "embedding_model": identity.model,
        "embedding_dimension": identity.dimension,
    })
