"""
PageIndex Query Engine for Palli Sahayak

Reasoning-based retrieval over document tree indexes. Two-step process:
1. LLM reasons over tree summaries to select relevant nodes
2. Full text extracted from selected nodes as context

Usage:
    from pageindex_integration.config import PageIndexConfig
    from pageindex_integration.storage import PageIndexStorage
    from pageindex_integration.query_engine import PageIndexQueryEngine

    config = PageIndexConfig()
    storage = PageIndexStorage(config)
    engine = PageIndexQueryEngine(config, storage)
    result = await engine.search("What are morphine dosing guidelines?")
"""

import asyncio
import re
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from pageindex_integration.config import PageIndexConfig
from pageindex_integration.storage import PageIndexStorage
from pageindex_integration.llm_adapter import LLMAdapter
from pageindex_integration.utils import (
    flatten_tree,
    strip_text_from_tree,
    extract_node_texts,
    count_tokens,
)

logger = logging.getLogger(__name__)


TREE_NAVIGATION_PROMPT = """You are a document analyst. Given a hierarchical tree structure of a document and a user query, identify which sections (nodes) contain information relevant to answering the query.

DOCUMENT: {filename}

TREE STRUCTURE (titles and summaries only, no full text):
{tree_structure}

USER QUERY: {query}

INSTRUCTIONS:
1. Analyze each node's title and summary to determine relevance to the query
2. Select up to {max_nodes} most relevant nodes
3. Prefer leaf nodes (more specific) over parent nodes (more general) when both are relevant
4. If a parent node's children don't individually cover the query but the parent does, select the parent

Return a JSON object with:
- "thinking": Your reasoning about which sections are relevant and why (1-2 sentences)
- "node_list": Array of node_id strings for the most relevant sections

Return ONLY the JSON object, no other text:"""


@dataclass
class PageIndexSearchResult:
    """Result of a PageIndex tree search."""
    query: str
    context: str
    selected_nodes: List[Dict[str, Any]]
    reasoning: str
    doc_sources: List[Dict[str, Any]]
    confidence: float
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "context": self.context,
            "selected_nodes": self.selected_nodes,
            "reasoning": self.reasoning,
            "doc_sources": self.doc_sources,
            "confidence": self.confidence,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


class PageIndexQueryEngine:
    """
    Reasoning-based retrieval over document tree indexes.

    The context feeds into the existing answer generation pipeline
    in simple_rag_server.py (_generate_answer_with_citations).
    """

    def __init__(self, config: PageIndexConfig, storage: PageIndexStorage):
        self._config = config
        self._storage = storage
        self._llm = LLMAdapter(config.llm)
        self._cache: Optional[Any] = None
        # Per tree: the navigation JSON (text stripped) and the node map,
        # built once per tree instead of on every query.
        self._prepared: Dict[str, tuple] = {}

        if config.search.cache_enabled:
            try:
                from graphrag_integration.utils import QueryCache
                self._cache = QueryCache(
                    maxsize=config.search.cache_maxsize,
                    ttl_seconds=config.search.cache_ttl_seconds,
                )
            except ImportError:
                logger.debug("QueryCache not available from graphrag_integration, caching disabled")

    async def search(
        self,
        query: str,
        doc_ids: Optional[List[str]] = None,
    ) -> PageIndexSearchResult:
        """
        Search across indexed trees.

        Args:
            query: User query
            doc_ids: Optional list of doc_ids to search (None = all)

        Returns:
            PageIndexSearchResult with context, sources, reasoning
        """
        start_time = time.time()

        if self._cache:
            cached = self._cache.get(query, "pageindex")
            if cached is not None:
                logger.debug(f"Cache hit for: {query[:50]}...")
                return cached

        entries = self._storage.list_trees()
        if doc_ids:
            entries = [e for e in entries if e.doc_id in doc_ids]

        completed_entries = [e for e in entries if e.status == "completed"]
        if not completed_entries:
            return PageIndexSearchResult(
                query=query,
                context="",
                selected_nodes=[],
                reasoning="No tree indexes available.",
                doc_sources=[],
                confidence=0.0,
                duration_ms=0.0,
            )

        all_selected_nodes = []
        all_sources = []
        all_reasoning = []

        # One navigation call per document, run together (bounded), so a
        # query costs about the slowest document, not the sum of them.
        limit = asyncio.Semaphore(max(1, self._config.search.max_concurrent_docs))

        async def navigate(entry):
            prepared = self._prepare(entry.doc_id)
            if prepared is None:
                return None
            tree_json, node_map = prepared
            async with limit:
                node_ids, reasoning = await self._navigate_tree(
                    query, None, entry.filename, tree_json=tree_json)
            return node_ids, reasoning, node_map

        outcomes = await asyncio.gather(*(navigate(e) for e in completed_entries))

        for entry, outcome in zip(completed_entries, outcomes):
            if outcome is None:
                continue
            node_ids, reasoning, node_map = outcome
            all_reasoning.append(f"[{entry.filename}]: {reasoning}")

            if not node_ids:
                continue

            token_budget = self._config.search.max_context_tokens // max(len(completed_entries), 1)
            extracted = extract_node_texts(node_ids, node_map, max_tokens=token_budget)

            for ext in extracted:
                ext["filename"] = entry.filename
                ext["doc_id"] = entry.doc_id
                all_selected_nodes.append(ext)

            all_sources.append({
                "doc_id": entry.doc_id,
                "filename": entry.filename,
                "selected_nodes": len(extracted),
                "page_ranges": [
                    f"pg {e['start_page']}-{e['end_page']}" for e in extracted
                ],
            })

        context_parts = []
        for node in all_selected_nodes:
            source_label = f"Source: {node['filename']}"
            if node.get("start_page"):
                source_label += f" (pg {node['start_page']}-{node['end_page']})"
            context_parts.append(f"{source_label}\n{node['text']}")

        context = "\n\n".join(context_parts)
        duration_ms = (time.time() - start_time) * 1000
        confidence = min(1.0, len(all_selected_nodes) / max(self._config.search.max_nodes_per_query, 1))

        result = PageIndexSearchResult(
            query=query,
            context=context,
            selected_nodes=all_selected_nodes,
            reasoning="\n".join(all_reasoning),
            doc_sources=all_sources,
            confidence=confidence,
            duration_ms=duration_ms,
        )

        if self._cache:
            self._cache.set(query, "pageindex", result)

        return result

    def _prepare(self, doc_id: str) -> Optional[tuple]:
        """(navigation JSON, node map) for a tree, from the cache while the
        stored tree is unchanged."""
        tree = self._storage.load_tree(doc_id)
        if not tree:
            return None
        cached = self._prepared.get(doc_id)
        if cached is not None and cached[0] is tree:
            return cached[1], cached[2]
        tree_json = self._tree_json(tree, doc_id)
        node_map = flatten_tree(tree)
        self._prepared[doc_id] = (tree, tree_json, node_map)
        return tree_json, node_map

    @staticmethod
    def _tree_json(tree: Dict[str, Any], filename: str) -> str:
        """The tree as the navigation prompt shows it: no text, capped."""
        tree_json = json.dumps(strip_text_from_tree(tree), indent=2)
        tree_tokens = count_tokens(tree_json)
        if tree_tokens > 50000:
            logger.warning(f"Tree for {filename} is {tree_tokens} tokens, truncating")
            tree_json = tree_json[:200000]
        return tree_json

    async def _navigate_tree(
        self,
        query: str,
        tree: Optional[Dict[str, Any]],
        filename: str,
        tree_json: Optional[str] = None,
    ) -> tuple:
        """
        LLM reasons over tree to select relevant node_ids.

        Returns: (list of node_ids, reasoning string)
        """
        if tree_json is None:
            tree_json = self._tree_json(tree, filename)

        prompt = TREE_NAVIGATION_PROMPT.format(
            filename=filename,
            tree_structure=tree_json,
            query=query,
            max_nodes=self._config.search.max_nodes_per_query,
        )

        try:
            response = await self._llm.chat_async(
                [{"role": "user", "content": prompt}],
                max_tokens=1024,
            )

            json_str = _extract_json(response)
            if json_str:
                parsed = json.loads(json_str)
                node_ids = parsed.get("node_list", parsed.get("nodes", []))
                reasoning = parsed.get("thinking", parsed.get("reasoning", ""))
                return node_ids, reasoning

            logger.warning(f"Failed to parse tree navigation response for {filename}")
            return [], f"Parse error: {response[:200]}"

        except Exception as e:
            logger.error(f"Tree navigation failed for {filename}: {e}")
            return [], str(e)

    def get_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"cache": None}
        if self._cache:
            stats["cache"] = self._cache.get_stats()
        return stats

    def __repr__(self) -> str:
        return f"PageIndexQueryEngine(provider={self._config.llm.provider})"


def _extract_json(text: str) -> Optional[str]:
    """Extract JSON object from LLM response (may contain markdown fences)."""
    text = text.strip()
    if text.startswith("{"):
        return text

    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r'\{[^{}]*"node_list"[^{}]*\}', text, re.DOTALL)
    if match:
        return match.group(0)

    return None
