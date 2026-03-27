"""Context-1 tool suite for agentic retrieval against ChromaDB and knowledge graph."""

import re
import logging
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    score: float
    source: str
    token_estimate: int = 0

    def __post_init__(self):
        if self.token_estimate == 0:
            self.token_estimate = len(self.text.split())


@dataclass
class TokenBudgetTracker:
    """Tracks token usage against the Context-1 budget."""

    budget: int = 32_768
    soft_threshold: int = 24_576
    hard_cutoff: int = 28_672
    _used: int = 0

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self._used)

    @property
    def at_soft_threshold(self) -> bool:
        return self._used >= self.soft_threshold

    @property
    def at_hard_cutoff(self) -> bool:
        return self._used >= self.hard_cutoff

    def consume(self, tokens: int) -> bool:
        """Consume tokens. Returns False if hard cutoff exceeded."""
        if self._used + tokens > self.hard_cutoff:
            return False
        self._used += tokens
        return True

    def release(self, tokens: int) -> None:
        self._used = max(0, self._used - tokens)


def _estimate_tokens(text: str) -> int:
    return len(text.split())


def _reciprocal_rank_fusion(
    ranked_lists: List[List[ChunkResult]],
    k: int = 60,
) -> List[ChunkResult]:
    """Combine multiple ranked result lists using RRF."""
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, ChunkResult] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            rrf_score = 1.0 / (k + rank + 1)
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + rrf_score
            if chunk.chunk_id not in chunk_map or rrf_score > chunk_map[chunk.chunk_id].score:
                chunk_map[chunk.chunk_id] = chunk

    for cid, total_score in scores.items():
        if cid in chunk_map:
            chunk_map[cid].score = total_score

    return sorted(chunk_map.values(), key=lambda c: c.score, reverse=True)


class Context1Tools:
    """Implements the four Context-1 retrieval tools against existing ChromaDB."""

    def __init__(
        self,
        chroma_collection,
        knowledge_graph_rag=None,
        top_k: int = 50,
    ):
        self._collection = chroma_collection
        self._kg_rag = knowledge_graph_rag
        self._top_k = top_k

    async def search_corpus(
        self,
        query: str,
        top_k: Optional[int] = None,
        seen_ids: Optional[Set[str]] = None,
    ) -> List[ChunkResult]:
        """
        Hybrid BM25-style keyword + dense vector search via RRF.
        Returns top-k results deduplicated against seen_ids.
        """
        k = top_k or self._top_k
        seen = seen_ids or set()

        vector_results = self._vector_search(query, n_results=k)
        keyword_results = self._keyword_search(query, n_results=k)

        fused = _reciprocal_rank_fusion([vector_results, keyword_results])

        deduped = [c for c in fused if c.chunk_id not in seen][:k]

        logger.info(
            f"search_corpus: query='{query[:60]}...' "
            f"vector={len(vector_results)} keyword={len(keyword_results)} "
            f"fused={len(fused)} deduped={len(deduped)}"
        )
        return deduped

    async def grep_corpus(
        self,
        pattern: str,
        max_matches: int = 5,
    ) -> List[ChunkResult]:
        """Regex pattern matching against all document chunks."""
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error:
            logger.warning(f"Invalid regex pattern: {pattern}")
            return []

        all_data = self._collection.get(include=["documents", "metadatas"])
        if not all_data or not all_data.get("documents"):
            return []

        results = []
        documents = all_data["documents"]
        metadatas = all_data.get("metadatas", [{}] * len(documents))
        ids = all_data.get("ids", [str(i) for i in range(len(documents))])

        for doc_text, meta, doc_id in zip(documents, metadatas, ids):
            if compiled.search(doc_text or ""):
                results.append(ChunkResult(
                    chunk_id=doc_id,
                    text=doc_text,
                    metadata=meta or {},
                    score=1.0,
                    source="grep",
                ))
                if len(results) >= max_matches:
                    break

        logger.info(f"grep_corpus: pattern='{pattern}' matches={len(results)}")
        return results

    async def read_document(
        self,
        doc_id: str,
    ) -> Optional[ChunkResult]:
        """Full document/chunk retrieval by ID from ChromaDB."""
        try:
            result = self._collection.get(ids=[doc_id], include=["documents", "metadatas"])
        except Exception as e:
            logger.warning(f"read_document failed for {doc_id}: {e}")
            return None

        if not result or not result.get("documents") or not result["documents"]:
            return None

        return ChunkResult(
            chunk_id=doc_id,
            text=result["documents"][0],
            metadata=result.get("metadatas", [{}])[0] or {},
            score=1.0,
            source="read",
        )

    def prune_chunks(
        self,
        chunks: List[ChunkResult],
        budget_tracker: TokenBudgetTracker,
        min_score: float = 0.0,
    ) -> List[ChunkResult]:
        """
        Remove low-relevance chunks to stay within token budget.
        Releases tokens from the budget tracker for pruned chunks.
        """
        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)

        kept = []
        pruned_tokens = 0
        for chunk in sorted_chunks:
            if chunk.score <= min_score:
                pruned_tokens += chunk.token_estimate
                continue
            if budget_tracker.at_soft_threshold and chunk.score < 0.3:
                pruned_tokens += chunk.token_estimate
                continue
            kept.append(chunk)

        if pruned_tokens > 0:
            budget_tracker.release(pruned_tokens)
            logger.info(
                f"prune_chunks: kept={len(kept)} pruned={len(sorted_chunks) - len(kept)} "
                f"freed={pruned_tokens} tokens"
            )
        return kept

    def _vector_search(self, query: str, n_results: int) -> List[ChunkResult]:
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=n_results,
            )
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

        if not results or not results.get("documents") or not results["documents"][0]:
            return []

        chunks = []
        documents = results["documents"][0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            score = 1.0 / (1.0 + dist) if dist is not None else 0.5
            chunk_id = ids[i] if i < len(ids) else f"vec_{i}"
            chunks.append(ChunkResult(
                chunk_id=chunk_id,
                text=doc,
                metadata=meta or {},
                score=score,
                source="vector",
            ))
        return chunks

    def _keyword_search(self, query: str, n_results: int) -> List[ChunkResult]:
        """BM25-style keyword search using ChromaDB where filter on document text."""
        keywords = [w.lower() for w in query.split() if len(w) > 3]
        if not keywords:
            return []

        try:
            all_data = self._collection.get(include=["documents", "metadatas"])
        except Exception as e:
            logger.warning(f"Keyword search data fetch failed: {e}")
            return []

        if not all_data or not all_data.get("documents"):
            return []

        documents = all_data["documents"]
        metadatas = all_data.get("metadatas", [{}] * len(documents))
        ids = all_data.get("ids", [str(i) for i in range(len(documents))])

        scored = []
        for doc_text, meta, doc_id in zip(documents, metadatas, ids):
            if not doc_text:
                continue
            doc_lower = doc_text.lower()
            hit_count = sum(1 for kw in keywords if kw in doc_lower)
            if hit_count > 0:
                score = hit_count / len(keywords)
                scored.append(ChunkResult(
                    chunk_id=doc_id,
                    text=doc_text,
                    metadata=meta or {},
                    score=score,
                    source="keyword",
                ))

        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:n_results]
