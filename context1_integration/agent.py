"""Context-1 agentic multi-hop retrieval agent."""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set

from context1_integration.config import Context1Config
from context1_integration.tools import (
    Context1Tools,
    ChunkResult,
    TokenBudgetTracker,
    _estimate_tokens,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunks: List[ChunkResult]
    total_tokens: int
    hops_used: int
    subqueries: List[str]
    context_text: str
    metadatas: List[Dict]


class Context1RetrievalAgent:
    """
    Agentic multi-hop retrieval using Context-1 patterns.
    Decomposes complex queries into subqueries, iteratively searches
    using the Context-1 tool suite, and manages a bounded token budget.
    """

    def __init__(
        self,
        config: Context1Config,
        chroma_collection=None,
        knowledge_graph_rag=None,
    ):
        self._config = config
        self._groq_api_key = os.getenv("GROQ_API_KEY")
        self._tools: Optional[Context1Tools] = None
        self._chroma_collection = chroma_collection
        self._kg_rag = knowledge_graph_rag

        if chroma_collection is not None:
            self._tools = Context1Tools(
                chroma_collection=chroma_collection,
                knowledge_graph_rag=knowledge_graph_rag,
                top_k=config.top_k_results,
            )

    def set_collection(self, chroma_collection, knowledge_graph_rag=None) -> None:
        self._chroma_collection = chroma_collection
        self._kg_rag = knowledge_graph_rag
        self._tools = Context1Tools(
            chroma_collection=chroma_collection,
            knowledge_graph_rag=knowledge_graph_rag,
            top_k=self._config.top_k_results,
        )

    async def multi_hop_retrieve(
        self,
        query: str,
        max_hops: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Decompose a complex query into subqueries and iteratively search.
        Uses bounded token budget with soft threshold pruning.
        """
        if self._tools is None:
            logger.warning("Context1 tools not initialized, returning empty result")
            return RetrievalResult(
                chunks=[], total_tokens=0, hops_used=0,
                subqueries=[], context_text="", metadatas=[],
            )

        hops = max_hops or self._config.max_hops
        budget = TokenBudgetTracker(
            budget=self._config.token_budget,
            soft_threshold=self._config.soft_threshold,
            hard_cutoff=self._config.hard_cutoff,
        )

        subqueries = await self._decompose_query(query)
        if not subqueries:
            subqueries = [query]

        all_chunks: List[ChunkResult] = []
        seen_ids: Set[str] = set()
        search_calls = 0

        for hop_idx, subquery in enumerate(subqueries):
            if hop_idx >= hops:
                break
            if budget.at_hard_cutoff:
                logger.info(f"Hard cutoff reached at hop {hop_idx}")
                break
            if search_calls >= self._config.max_search_calls:
                logger.info(f"Max search calls ({self._config.max_search_calls}) reached")
                break

            results = await self._tools.search_corpus(
                query=subquery,
                top_k=self._config.top_k_results,
                seen_ids=seen_ids,
            )
            search_calls += 1

            for chunk in results:
                if not budget.consume(chunk.token_estimate):
                    break
                all_chunks.append(chunk)
                seen_ids.add(chunk.chunk_id)

            if budget.at_soft_threshold:
                all_chunks = self._tools.prune_chunks(
                    all_chunks, budget, min_score=0.1,
                )

            follow_up = await self._generate_follow_up(query, subquery, all_chunks)
            if follow_up and follow_up != subquery:
                if search_calls < self._config.max_search_calls and not budget.at_hard_cutoff:
                    extra = await self._tools.search_corpus(
                        query=follow_up,
                        top_k=self._config.top_k_results // 2,
                        seen_ids=seen_ids,
                    )
                    search_calls += 1
                    for chunk in extra:
                        if not budget.consume(chunk.token_estimate):
                            break
                        all_chunks.append(chunk)
                        seen_ids.add(chunk.chunk_id)

        all_chunks = self._tools.prune_chunks(all_chunks, budget, min_score=0.05)
        all_chunks.sort(key=lambda c: c.score, reverse=True)

        context_text = "\n\n".join(
            f"Source: {c.metadata.get('filename', 'unknown')} "
            f"(chunk {c.metadata.get('chunk_index', 0) + 1})\n{c.text}"
            for c in all_chunks
        )

        metadatas = [
            {
                "filename": c.metadata.get("filename", "unknown"),
                "chunk_index": c.metadata.get("chunk_index", 0),
                "total_chunks": c.metadata.get("total_chunks", 1),
                "score": c.score,
                "source": c.source,
            }
            for c in all_chunks
        ]

        logger.info(
            f"Multi-hop retrieval complete: {len(all_chunks)} chunks, "
            f"{budget.used} tokens, {search_calls} searches, "
            f"{len(subqueries)} subqueries"
        )

        return RetrievalResult(
            chunks=all_chunks,
            total_tokens=budget.used,
            hops_used=min(len(subqueries), hops),
            subqueries=subqueries,
            context_text=context_text,
            metadatas=metadatas,
        )

    async def _decompose_query(self, query: str) -> List[str]:
        """Use Groq LLM to decompose a complex query into subqueries."""
        if not self._groq_api_key:
            return self._rule_based_decompose(query)

        try:
            import aiohttp

            prompt = (
                "Decompose this clinical query into 2-4 independent subqueries "
                "for document retrieval. Return ONLY a JSON array of strings.\n\n"
                f"Query: {query}\n\nSubqueries:"
            )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._groq_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._config.groq_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2,
                        "max_tokens": 256,
                    },
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"Groq decompose failed: {resp.status}")
                        return self._rule_based_decompose(query)

                    data = await resp.json()
                    raw = data["choices"][0]["message"]["content"].strip()
                    subqueries = json.loads(raw)
                    if isinstance(subqueries, list) and all(isinstance(s, str) for s in subqueries):
                        return subqueries
        except Exception as e:
            logger.warning(f"LLM decomposition failed, using rule-based: {e}")

        return self._rule_based_decompose(query)

    async def _generate_follow_up(
        self,
        original_query: str,
        last_subquery: str,
        current_chunks: List[ChunkResult],
    ) -> Optional[str]:
        """Generate a follow-up subquery based on retrieved context."""
        if not self._groq_api_key or not current_chunks:
            return None

        try:
            import aiohttp

            context_summary = " ".join(
                c.text[:200] for c in current_chunks[-3:]
            )
            prompt = (
                "Given the original question and the partial context retrieved so far, "
                "generate ONE follow-up search query to fill gaps. "
                "Return ONLY the query string, nothing else.\n\n"
                f"Original question: {original_query}\n"
                f"Last search: {last_subquery}\n"
                f"Context so far: {context_summary[:500]}\n\n"
                "Follow-up query:"
            )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._groq_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._config.groq_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "max_tokens": 64,
                    },
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.debug(f"Follow-up generation failed: {e}")

        return None

    def _rule_based_decompose(self, query: str) -> List[str]:
        """Fallback decomposition using sentence splitting and keyword extraction."""
        import re

        parts = re.split(r'[,;]|\band\b|\bbut\b|\bwhile\b|\bgiven\b', query, flags=re.IGNORECASE)
        subqueries = [p.strip() for p in parts if len(p.strip()) > 10]

        if len(subqueries) < 2:
            subqueries = [query]

        return subqueries[:4]
