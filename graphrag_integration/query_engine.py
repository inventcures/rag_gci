"""
GraphRAG Query Engine for Palli Sahayak

Provides query capabilities using Global, Local, DRIFT, and Basic search strategies.

Usage:
    from graphrag_integration.query_engine import GraphRAGQueryEngine
    from graphrag_integration.config import GraphRAGConfig

    config = GraphRAGConfig.from_yaml("./data/graphrag/settings.yaml")
    engine = GraphRAGQueryEngine(config)

    # Global search for holistic questions
    result = await engine.global_search("What are the main approaches to pain management?")

    # Local search for entity-specific questions
    result = await engine.local_search("What are the side effects of morphine?")

    # DRIFT search for complex multi-hop questions
    result = await engine.drift_search("How should pain be managed in a patient with renal failure?")

    # Auto-select best method
    result = await engine.auto_search("What medications treat nausea?")
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field

from graphrag_integration.config import GraphRAGConfig
from graphrag_integration.data_loader import GraphRAGDataLoader

logger = logging.getLogger(__name__)


class SearchMethod(Enum):
    """
    Available search methods.

    GLOBAL: Holistic corpus-wide queries using community reports
    LOCAL: Entity-focused queries using entity relationships
    DRIFT: Dynamic Retrieval with Iterative Focusing and Traversal
    BASIC: Simple vector similarity search
    """
    GLOBAL = "global"
    LOCAL = "local"
    DRIFT = "drift"
    BASIC = "basic"


@dataclass
class SearchResult:
    """
    Container for search results.

    Attributes:
        query: Original query string
        response: Generated response text
        method: Search method used
        sources: List of source documents/text units
        entities: List of relevant entities
        communities: List of relevant communities
        confidence: Confidence score (0.0-1.0)
        metadata: Additional metadata
    """
    query: str
    response: str
    method: SearchMethod
    sources: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    communities: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "response": self.response,
            "method": self.method.value,
            "sources": self.sources,
            "entities": self.entities,
            "communities": self.communities,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class GraphRAGQueryEngine:
    """
    Query engine for GraphRAG.

    Supports multiple search strategies optimized for different query types:
    - Global: Best for holistic, thematic questions
    - Local: Best for entity-specific questions
    - DRIFT: Best for complex multi-hop reasoning
    - Basic: Best for simple factual queries

    Attributes:
        config: GraphRAG configuration
        data_loader: Data loader for parquet files

    Example:
        engine = GraphRAGQueryEngine(config)

        # Initialize (loads data)
        await engine.initialize()

        # Global search
        result = await engine.global_search(
            "What are the main approaches to pain management?"
        )

        # Local search
        result = await engine.local_search(
            "What are the side effects of morphine?"
        )

        # DRIFT search
        result = await engine.drift_search(
            "How should pain be managed in a patient with renal failure?"
        )

        # Auto-select method
        result = await engine.auto_search("What is morphine?")
    """

    def __init__(self, config: GraphRAGConfig):
        """
        Initialize query engine.

        Args:
            config: GraphRAG configuration
        """
        self.config = config
        self.data_loader = GraphRAGDataLoader(config)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize query engine and load data."""
        if self._initialized:
            return

        await self.data_loader.load_all()
        self._initialized = True
        logger.info("Query engine initialized")

    async def global_search(
        self,
        query: str,
        community_level: int = 2,
        response_type: str = "comprehensive"
    ) -> SearchResult:
        """
        Perform global search across entire corpus.

        Uses community reports to synthesize corpus-wide themes.
        Best for holistic questions requiring broad understanding.

        Args:
            query: User query
            community_level: Community hierarchy level (0=granular, higher=broader)
            response_type: Response format ("comprehensive" or "brief")

        Returns:
            SearchResult with response and sources

        Example:
            result = await engine.global_search(
                "What are the main themes in palliative care guidelines?"
            )
        """
        await self.initialize()

        try:
            from graphrag.api.query import global_search as graphrag_global

            native_config = self.config.to_graphrag_config()

            result = await graphrag_global(
                config=native_config,
                query=query,
            )

            return SearchResult(
                query=query,
                response=result.response,
                method=SearchMethod.GLOBAL,
                sources=self._extract_sources(result),
                entities=[],
                communities=self._extract_communities(result),
                confidence=self._calculate_confidence(result),
                metadata={
                    "community_level": community_level,
                    "response_type": response_type,
                },
            )

        except ImportError:
            logger.warning("GraphRAG not installed, using fallback search")
            return await self._fallback_global_search(query, community_level)

        except Exception as e:
            logger.error(f"Global search failed: {e}")
            raise

    async def local_search(
        self,
        query: str,
        top_k_entities: int = 10,
        include_community_context: bool = True
    ) -> SearchResult:
        """
        Perform local search focused on specific entities.

        Uses entity relationships and descriptions for focused retrieval.
        Best for questions about specific entities (medications, symptoms).

        Args:
            query: User query
            top_k_entities: Number of top entities to retrieve
            include_community_context: Include community reports for context

        Returns:
            SearchResult with response and sources

        Example:
            result = await engine.local_search(
                "What are the side effects of morphine?"
            )
        """
        await self.initialize()

        try:
            from graphrag.api.query import local_search as graphrag_local

            native_config = self.config.to_graphrag_config()

            result = await graphrag_local(
                config=native_config,
                query=query,
            )

            return SearchResult(
                query=query,
                response=result.response,
                method=SearchMethod.LOCAL,
                sources=self._extract_sources(result),
                entities=self._extract_entities(result),
                communities=[],
                confidence=self._calculate_confidence(result),
                metadata={
                    "top_k_entities": top_k_entities,
                    "include_community_context": include_community_context,
                },
            )

        except ImportError:
            logger.warning("GraphRAG not installed, using fallback search")
            return await self._fallback_local_search(query, top_k_entities)

        except Exception as e:
            logger.error(f"Local search failed: {e}")
            raise

    async def drift_search(
        self,
        query: str,
        n_depth: int = 3,
        primer_folds: int = 3
    ) -> SearchResult:
        """
        Perform DRIFT (Dynamic Retrieval with Iterative Focusing and Traversal) search.

        Uses multi-phase iterative search with query expansion.
        Best for complex multi-hop reasoning questions.

        Args:
            query: User query
            n_depth: Depth of iterative search (more = broader exploration)
            primer_folds: Number of primer phase folds

        Returns:
            SearchResult with response and sources

        Example:
            result = await engine.drift_search(
                "How should pain be managed in a patient with renal failure?"
            )
        """
        await self.initialize()

        try:
            from graphrag.api.query import drift_search as graphrag_drift

            native_config = self.config.to_graphrag_config()

            result = await graphrag_drift(
                config=native_config,
                query=query,
            )

            return SearchResult(
                query=query,
                response=result.response,
                method=SearchMethod.DRIFT,
                sources=self._extract_sources(result),
                entities=self._extract_entities(result),
                communities=self._extract_communities(result),
                confidence=self._calculate_confidence(result),
                metadata={
                    "n_depth": n_depth,
                    "primer_folds": primer_folds,
                },
            )

        except ImportError:
            logger.warning("GraphRAG not installed, using fallback search")
            return await self._fallback_drift_search(query)

        except Exception as e:
            logger.error(f"DRIFT search failed: {e}")
            raise

    async def basic_search(
        self,
        query: str,
        top_k: int = 5
    ) -> SearchResult:
        """
        Perform basic vector similarity search.

        Simple retrieval based on embedding similarity.
        Best for simple factual queries and baseline comparison.

        Args:
            query: User query
            top_k: Number of results to retrieve

        Returns:
            SearchResult with response and sources

        Example:
            result = await engine.basic_search("What is morphine?")
        """
        await self.initialize()

        try:
            from graphrag.api.query import basic_search as graphrag_basic

            native_config = self.config.to_graphrag_config()

            result = await graphrag_basic(
                config=native_config,
                query=query,
            )

            return SearchResult(
                query=query,
                response=result.response,
                method=SearchMethod.BASIC,
                sources=self._extract_sources(result),
                entities=[],
                communities=[],
                confidence=self._calculate_confidence(result),
                metadata={"top_k": top_k},
            )

        except ImportError:
            logger.warning("GraphRAG not installed, using fallback search")
            return await self._fallback_search(query, SearchMethod.BASIC)

        except Exception as e:
            logger.error(f"Basic search failed: {e}")
            raise

    async def auto_search(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> SearchResult:
        """
        Automatically select the best search method based on query analysis.

        Analyzes the query to determine whether it's:
        - Holistic/thematic (-> global search)
        - Entity-specific (-> local search)
        - Complex/multi-hop (-> DRIFT search)
        - Simple factual (-> basic search)

        Args:
            query: User query
            context: Optional conversation context for better method selection

        Returns:
            SearchResult from the selected method

        Example:
            result = await engine.auto_search("What is the best treatment for pain?")
            print(f"Method used: {result.method.value}")
        """
        method = self._analyze_query(query)

        if method == SearchMethod.GLOBAL:
            return await self.global_search(query)
        elif method == SearchMethod.LOCAL:
            return await self.local_search(query)
        elif method == SearchMethod.DRIFT:
            return await self.drift_search(query)
        else:
            return await self.basic_search(query)

    def _analyze_query(self, query: str) -> SearchMethod:
        """
        Analyze query to determine optimal search method.

        Uses keyword matching to classify query type.
        In production, could use LLM-based classification.

        Args:
            query: User query

        Returns:
            Recommended search method
        """
        query_lower = query.lower()

        # Global indicators: broad, thematic questions
        global_keywords = [
            "overall", "main", "themes", "summary", "generally",
            "across", "comprehensive", "all", "types of", "approaches",
            "overview", "principles", "guidelines", "broad", "holistic"
        ]

        # Local indicators: entity-specific questions
        local_keywords = [
            "specific", "what is", "tell me about", "side effects",
            "dosage", "how does", "compare", "difference between",
            "define", "describe", "explain", "details about"
        ]

        # DRIFT indicators: complex, multi-hop questions
        drift_keywords = [
            "how should", "in the context of", "considering",
            "for a patient with", "when combined with", "impact of",
            "relationship between", "if the patient", "given that",
            "complex", "multiple", "together with"
        ]

        # Count keyword matches
        global_score = sum(1 for kw in global_keywords if kw in query_lower)
        local_score = sum(1 for kw in local_keywords if kw in query_lower)
        drift_score = sum(1 for kw in drift_keywords if kw in query_lower)

        # Boost scores based on question structure
        if "?" in query and len(query.split()) > 15:
            drift_score += 1  # Longer questions often need DRIFT

        if "what are the" in query_lower or "list" in query_lower:
            local_score += 1

        # Select method based on scores
        max_score = max(global_score, local_score, drift_score)

        if max_score == 0:
            # Default to local for most queries
            return SearchMethod.LOCAL

        if global_score == max_score:
            return SearchMethod.GLOBAL
        elif drift_score == max_score:
            return SearchMethod.DRIFT
        else:
            return SearchMethod.LOCAL

    async def _fallback_search(
        self,
        query: str,
        method: SearchMethod
    ) -> SearchResult:
        """
        Fallback search using loaded parquet data directly.

        Args:
            query: User query
            method: Original search method

        Returns:
            SearchResult with basic response
        """
        entities = await self.data_loader.search_entities(query, top_k=5)

        if entities:
            response_parts = []
            for entity in entities:
                name = entity.get("title", entity.get("name", "Unknown"))
                desc = entity.get("description", "No description available")
                entity_type = entity.get("type", "")
                type_str = f" ({entity_type})" if entity_type else ""
                response_parts.append(f"**{name}**{type_str}: {desc}")
            response = "\n\n".join(response_parts)
        else:
            response = "No relevant information found in the knowledge base."

        return SearchResult(
            query=query,
            response=response,
            method=method,
            sources=[],
            entities=entities,
            communities=[],
            confidence=0.5 if entities else 0.0,
            metadata={"fallback": True},
        )

    async def _fallback_global_search(
        self,
        query: str,
        community_level: int = 2
    ) -> SearchResult:
        """
        Fallback global search using community reports.

        Args:
            query: User query
            community_level: Community level to search

        Returns:
            SearchResult with community-based response
        """
        reports = await self.data_loader.get_community_reports_by_level(community_level)

        if reports:
            # Build response from community summaries
            relevant_reports = []
            query_lower = query.lower()

            for report in reports[:5]:  # Top 5 reports
                summary = report.get("summary", "").lower()
                title = report.get("title", "").lower()
                if any(word in summary or word in title for word in query_lower.split()):
                    relevant_reports.append(report)

            if relevant_reports:
                response_parts = []
                for report in relevant_reports[:3]:
                    title = report.get("title", "Community Report")
                    summary = report.get("summary", "")[:500]
                    response_parts.append(f"**{title}**\n{summary}")
                response = "\n\n---\n\n".join(response_parts)
            else:
                response = "No community reports directly match your query."
        else:
            response = "No community reports available."

        return SearchResult(
            query=query,
            response=response,
            method=SearchMethod.GLOBAL,
            sources=[],
            entities=[],
            communities=reports[:3] if reports else [],
            confidence=0.6 if reports else 0.0,
            metadata={"fallback": True, "community_level": community_level},
        )

    async def _fallback_local_search(
        self,
        query: str,
        top_k: int = 10
    ) -> SearchResult:
        """
        Fallback local search using entities and relationships.

        Args:
            query: User query
            top_k: Number of entities to retrieve

        Returns:
            SearchResult with entity-based response
        """
        entities = await self.data_loader.search_entities(query, top_k=top_k)

        response_parts = []
        all_relationships = []

        for entity in entities[:5]:
            name = entity.get("title", entity.get("name", "Unknown"))
            desc = entity.get("description", "No description")
            entity_type = entity.get("type", "")

            # Get relationships for this entity
            rels = await self.data_loader.get_entity_relationships(name)
            all_relationships.extend(rels)

            type_str = f" ({entity_type})" if entity_type else ""
            response_parts.append(f"**{name}**{type_str}: {desc}")

            if rels:
                rel_strs = []
                for rel in rels[:3]:
                    rel_type = rel.get("type", "related to")
                    target = rel.get("target", "")
                    source = rel.get("source", "")
                    other = target if source.lower() == name.lower() else source
                    rel_strs.append(f"  - {rel_type}: {other}")
                if rel_strs:
                    response_parts.append("\n".join(rel_strs))

        if response_parts:
            response = "\n\n".join(response_parts)
        else:
            response = "No relevant entities found for your query."

        return SearchResult(
            query=query,
            response=response,
            method=SearchMethod.LOCAL,
            sources=[],
            entities=entities,
            communities=[],
            confidence=0.7 if entities else 0.0,
            metadata={"fallback": True, "top_k": top_k},
        )

    async def _fallback_drift_search(
        self,
        query: str
    ) -> SearchResult:
        """
        Fallback DRIFT search combining entity and community data.

        Args:
            query: User query

        Returns:
            SearchResult with combined response
        """
        # Get entities
        entities = await self.data_loader.search_entities(query, top_k=5)

        # Get community reports
        reports = await self.data_loader.get_community_reports_by_level(1)

        response_parts = []

        if entities:
            response_parts.append("## Relevant Entities\n")
            for entity in entities[:3]:
                name = entity.get("title", entity.get("name", "Unknown"))
                desc = entity.get("description", "")
                response_parts.append(f"- **{name}**: {desc}")

        if reports:
            response_parts.append("\n## Related Context\n")
            query_lower = query.lower()
            for report in reports[:2]:
                summary = report.get("summary", "")
                if any(word in summary.lower() for word in query_lower.split()):
                    title = report.get("title", "Context")
                    response_parts.append(f"**{title}**: {summary[:300]}...")

        if response_parts:
            response = "\n".join(response_parts)
        else:
            response = "Unable to find comprehensive information for this query."

        return SearchResult(
            query=query,
            response=response,
            method=SearchMethod.DRIFT,
            sources=[],
            entities=entities,
            communities=reports[:2] if reports else [],
            confidence=0.6 if entities or reports else 0.0,
            metadata={"fallback": True},
        )

    def _extract_sources(self, result: Any) -> List[Dict[str, Any]]:
        """Extract sources from GraphRAG result."""
        sources = []
        if hasattr(result, "context_data"):
            context = result.context_data
            if hasattr(context, "sources"):
                for source in context.sources:
                    sources.append({
                        "id": getattr(source, "id", ""),
                        "text": getattr(source, "text", ""),
                        "document": getattr(source, "document", ""),
                    })
        return sources

    def _extract_entities(self, result: Any) -> List[Dict[str, Any]]:
        """Extract entities from GraphRAG result."""
        entities = []
        if hasattr(result, "context_data"):
            context = result.context_data
            if hasattr(context, "entities"):
                for entity in context.entities:
                    entities.append({
                        "name": getattr(entity, "name", getattr(entity, "title", "")),
                        "type": getattr(entity, "type", ""),
                        "description": getattr(entity, "description", ""),
                    })
        return entities

    def _extract_communities(self, result: Any) -> List[Dict[str, Any]]:
        """Extract communities from GraphRAG result."""
        communities = []
        if hasattr(result, "context_data"):
            context = result.context_data
            if hasattr(context, "reports"):
                for report in context.reports:
                    communities.append({
                        "id": getattr(report, "id", ""),
                        "title": getattr(report, "title", ""),
                        "summary": getattr(report, "summary", ""),
                    })
        return communities

    def _calculate_confidence(self, result: Any) -> float:
        """Calculate confidence score from result."""
        # Basic confidence based on response length and source count
        confidence = 0.8

        if hasattr(result, "response"):
            response_len = len(result.response)
            if response_len < 100:
                confidence = 0.5
            elif response_len > 500:
                confidence = 0.9

        if hasattr(result, "context_data"):
            if hasattr(result.context_data, "sources"):
                source_count = len(result.context_data.sources)
                confidence = min(confidence + (source_count * 0.02), 1.0)

        return confidence

    def __repr__(self) -> str:
        return f"GraphRAGQueryEngine(initialized={self._initialized})"

    def __str__(self) -> str:
        return self.__repr__()
