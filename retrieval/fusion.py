"""
Reciprocal Rank Fusion, shared by every retriever that needs to merge rankings.

RRF scores an item by summing 1 / (k + rank) over every list it appears in, with
rank counted from 1. It uses only rank positions, never raw scores, which is why
it suits this system: a Chroma L2 distance, a PageIndex node selection, a Neo4j
fact and a GraphRAG entity score live on scales that cannot be compared directly.

Worked example with k = 60:

    vector list:  [A, B, C]      kg list: [C, D]
    A = 1/61                     = 0.01639
    B = 1/62                     = 0.01613
    C = 1/63 + 1/61              = 0.03227   (in both lists, so it wins)
    D = 1/62                     = 0.01613

Previously the only copy lived in context1_integration/tools.py. It now lives
here and that module calls it, so there is exactly one fusion function in the
codebase.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Generic, Hashable, List, Optional, Sequence, TypeVar

T = TypeVar("T")

DEFAULT_RRF_K = 60


@dataclass
class FusedItem(Generic[T]):
    """One fused result: the item, its RRF score, and its rank in each list."""

    item: T
    score: float
    # list name -> 1-based rank within that list
    ranks: Dict[str, int] = field(default_factory=dict)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[T]],
    key: Callable[[T], Hashable],
    k: int = DEFAULT_RRF_K,
    weights: Optional[Sequence[float]] = None,
    names: Optional[Sequence[str]] = None,
) -> List[FusedItem[T]]:
    """
    Fuse several ranked lists into one, best first.

    Args:
        ranked_lists: each list ordered best first.
        key: identity of an item; items with the same key are the same result.
        k: RRF damping constant. Larger k flattens the advantage of a top rank.
        weights: optional per-list multiplier, default 1.0 for every list.
        names: optional per-list name used in `FusedItem.ranks`.

    Returns:
        Fused items sorted by descending score. Ties keep first-seen order, so
        the output is deterministic for a given input. The representative item
        for a key is the first one seen.
    """
    if k < 0:
        raise ValueError("RRF k must be non-negative")
    if weights is not None and len(weights) != len(ranked_lists):
        raise ValueError("weights must have one entry per ranked list")
    if names is not None and len(names) != len(ranked_lists):
        raise ValueError("names must have one entry per ranked list")

    fused: Dict[Hashable, FusedItem[T]] = {}
    order: Dict[Hashable, int] = {}

    for list_idx, ranked in enumerate(ranked_lists):
        weight = 1.0 if weights is None else float(weights[list_idx])
        name = names[list_idx] if names is not None else str(list_idx)
        seen_in_this_list = set()
        for position, item in enumerate(ranked):
            item_key = key(item)
            # A duplicate within one list counts once, at its best rank.
            if item_key in seen_in_this_list:
                continue
            seen_in_this_list.add(item_key)
            rank = position + 1
            contribution = weight / (k + rank)
            if item_key not in fused:
                fused[item_key] = FusedItem(item=item, score=0.0)
                order[item_key] = len(order)
            fused[item_key].score += contribution
            fused[item_key].ranks[name] = rank

    return sorted(
        fused.values(),
        key=lambda f: (-f.score, order[key(f.item)]),
    )
