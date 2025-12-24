"""
GraphRAG Data Loader for Palli Sahayak

Handles loading and caching of parquet files for query operations.

Usage:
    from graphrag_integration.data_loader import GraphRAGDataLoader
    from graphrag_integration.config import GraphRAGConfig

    config = GraphRAGConfig.from_yaml("./data/graphrag/settings.yaml")
    loader = GraphRAGDataLoader(config)

    # Load all data
    await loader.load_all()

    # Search entities
    entities = await loader.search_entities("morphine", top_k=5)

    # Get relationships
    relationships = await loader.get_entity_relationships("morphine")

    # Get statistics
    stats = await loader.get_stats()
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from graphrag_integration.config import GraphRAGConfig

logger = logging.getLogger(__name__)


class GraphRAGDataLoader:
    """
    Data loader for GraphRAG parquet files.

    Provides efficient loading and caching of indexed data for query operations.

    Attributes:
        config: GraphRAG configuration
        entities: Loaded entities DataFrame
        relationships: Loaded relationships DataFrame
        communities: Loaded communities DataFrame
        community_reports: Loaded community reports DataFrame
        text_units: Loaded text units DataFrame

    Example:
        loader = GraphRAGDataLoader(config)
        await loader.load_all()

        # Search for entities
        entities = await loader.search_entities("pain", top_k=10)

        # Get entity relationships
        rels = await loader.get_entity_relationships("morphine")

        # Get community report
        report = await loader.get_community_report("community_1")
    """

    def __init__(self, config: GraphRAGConfig):
        """
        Initialize data loader.

        Args:
            config: GraphRAG configuration
        """
        self.config = config
        self._entities = None
        self._relationships = None
        self._communities = None
        self._community_reports = None
        self._text_units = None
        self._documents = None
        self._embeddings = None
        self._loaded = False

    async def load_all(self) -> bool:
        """
        Load all data files from parquet.

        Returns:
            True if any data was loaded successfully
        """
        if self._loaded:
            return True

        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas not installed, data loading disabled")
            return False

        artifacts_dir = self.config.artifacts_dir
        loaded_any = False

        try:
            # Load entities
            entities_path = artifacts_dir / "entities.parquet"
            if entities_path.exists():
                self._entities = pd.read_parquet(entities_path)
                logger.info(f"Loaded {len(self._entities)} entities")
                loaded_any = True

            # Load relationships
            relationships_path = artifacts_dir / "relationships.parquet"
            if relationships_path.exists():
                self._relationships = pd.read_parquet(relationships_path)
                logger.info(f"Loaded {len(self._relationships)} relationships")
                loaded_any = True

            # Load communities
            communities_path = artifacts_dir / "communities.parquet"
            if communities_path.exists():
                self._communities = pd.read_parquet(communities_path)
                logger.info(f"Loaded {len(self._communities)} communities")
                loaded_any = True

            # Load community reports
            reports_path = artifacts_dir / "community_reports.parquet"
            if reports_path.exists():
                self._community_reports = pd.read_parquet(reports_path)
                logger.info(f"Loaded {len(self._community_reports)} community reports")
                loaded_any = True

            # Load text units
            text_units_path = artifacts_dir / "text_units.parquet"
            if text_units_path.exists():
                self._text_units = pd.read_parquet(text_units_path)
                logger.info(f"Loaded {len(self._text_units)} text units")
                loaded_any = True

            # Load documents
            documents_path = artifacts_dir / "documents.parquet"
            if documents_path.exists():
                self._documents = pd.read_parquet(documents_path)
                logger.info(f"Loaded {len(self._documents)} documents")
                loaded_any = True

            self._loaded = loaded_any

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

        return loaded_any

    async def search_entities(
        self,
        query: str,
        top_k: int = 10,
        entity_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search entities by query string.

        Uses text matching on entity titles and descriptions.
        In production, this would use embedding similarity.

        Args:
            query: Search query
            top_k: Maximum results to return
            entity_type: Optional filter by entity type (e.g., "Medication", "Symptom")

        Returns:
            List of matching entity dictionaries sorted by relevance score
        """
        await self.load_all()

        if self._entities is None or len(self._entities) == 0:
            return []

        query_lower = query.lower()
        query_words = query_lower.split()
        matches = []

        for _, row in self._entities.iterrows():
            score = 0.0

            # Get entity fields
            title = str(row.get("title", row.get("name", ""))).lower()
            description = str(row.get("description", "")).lower()
            entity_type_val = row.get("type", "")

            # Filter by entity type if specified
            if entity_type is not None and entity_type_val != entity_type:
                continue

            # Exact match in title (highest score)
            if query_lower in title:
                score += 3.0

            # Exact match in description
            if query_lower in description:
                score += 1.5

            # Word matches
            for word in query_words:
                if len(word) >= 3:  # Skip very short words
                    if word in title:
                        score += 0.5
                    if word in description:
                        score += 0.25

            if score > 0:
                matches.append({
                    "title": row.get("title", row.get("name", "Unknown")),
                    "name": row.get("name", row.get("title", "Unknown")),
                    "type": entity_type_val,
                    "description": row.get("description", ""),
                    "score": score,
                    "id": row.get("id", row.get("human_readable_id", "")),
                })

        # Sort by score descending
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches[:top_k]

    async def get_entity_by_name(
        self,
        entity_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific entity by name.

        Args:
            entity_name: Name of the entity

        Returns:
            Entity dictionary or None if not found
        """
        await self.load_all()

        if self._entities is None:
            return None

        entity_lower = entity_name.lower()

        for _, row in self._entities.iterrows():
            title = str(row.get("title", row.get("name", ""))).lower()
            if title == entity_lower:
                return {
                    "title": row.get("title", row.get("name", "")),
                    "name": row.get("name", row.get("title", "")),
                    "type": row.get("type", ""),
                    "description": row.get("description", ""),
                    "id": row.get("id", row.get("human_readable_id", "")),
                }

        return None

    async def get_entity_relationships(
        self,
        entity_name: str
    ) -> List[Dict[str, Any]]:
        """
        Get relationships for a specific entity.

        Args:
            entity_name: Name of the entity

        Returns:
            List of relationship dictionaries where entity is source or target
        """
        await self.load_all()

        if self._relationships is None or len(self._relationships) == 0:
            return []

        matches = []
        entity_lower = entity_name.lower()

        for _, row in self._relationships.iterrows():
            source = str(row.get("source", "")).lower()
            target = str(row.get("target", "")).lower()

            if entity_lower in source or entity_lower in target:
                matches.append({
                    "source": row.get("source", ""),
                    "target": row.get("target", ""),
                    "type": row.get("type", row.get("description", "")),
                    "description": row.get("description", ""),
                    "weight": float(row.get("weight", row.get("rank", 1.0))),
                    "id": row.get("id", row.get("human_readable_id", "")),
                })

        return matches

    async def get_community_report(
        self,
        community_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get community report by ID.

        Args:
            community_id: Community ID

        Returns:
            Community report dictionary or None if not found
        """
        await self.load_all()

        if self._community_reports is None:
            return None

        for _, row in self._community_reports.iterrows():
            row_id = str(row.get("id", row.get("community", "")))
            if row_id == str(community_id):
                return {
                    "id": row_id,
                    "title": row.get("title", ""),
                    "summary": row.get("summary", ""),
                    "full_content": row.get("full_content", row.get("content", "")),
                    "level": row.get("level", 0),
                    "rank": row.get("rank", 0),
                }

        return None

    async def get_community_reports_by_level(
        self,
        level: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get all community reports at a specific hierarchy level.

        Args:
            level: Community hierarchy level (0=most granular)

        Returns:
            List of community report dictionaries
        """
        await self.load_all()

        if self._community_reports is None:
            return []

        reports = []
        for _, row in self._community_reports.iterrows():
            if row.get("level", 0) == level:
                reports.append({
                    "id": row.get("id", row.get("community", "")),
                    "title": row.get("title", ""),
                    "summary": row.get("summary", ""),
                    "full_content": row.get("full_content", row.get("content", "")),
                    "level": level,
                    "rank": row.get("rank", 0),
                })

        # Sort by rank
        reports.sort(key=lambda x: x.get("rank", 0), reverse=True)
        return reports

    async def get_text_units(
        self,
        document_id: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get text units, optionally filtered by document.

        Args:
            document_id: Optional document ID to filter by
            top_k: Maximum number of text units to return

        Returns:
            List of text unit dictionaries
        """
        await self.load_all()

        if self._text_units is None:
            return []

        units = []
        for _, row in self._text_units.iterrows():
            if document_id is not None:
                row_doc_id = row.get("document_id", row.get("document_ids", ""))
                if str(document_id) not in str(row_doc_id):
                    continue

            units.append({
                "id": row.get("id", row.get("human_readable_id", "")),
                "text": row.get("text", row.get("chunk", "")),
                "document_id": row.get("document_id", row.get("document_ids", "")),
                "n_tokens": row.get("n_tokens", 0),
            })

            if len(units) >= top_k:
                break

        return units

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get data statistics.

        Returns:
            Dictionary with counts for each data type and load status
        """
        await self.load_all()

        return {
            "entities": len(self._entities) if self._entities is not None else 0,
            "relationships": len(self._relationships) if self._relationships is not None else 0,
            "communities": len(self._communities) if self._communities is not None else 0,
            "community_reports": len(self._community_reports) if self._community_reports is not None else 0,
            "text_units": len(self._text_units) if self._text_units is not None else 0,
            "documents": len(self._documents) if self._documents is not None else 0,
            "loaded": self._loaded,
        }

    async def reload(self) -> bool:
        """
        Force reload of all data.

        Returns:
            True if reload was successful
        """
        self._loaded = False
        self._entities = None
        self._relationships = None
        self._communities = None
        self._community_reports = None
        self._text_units = None
        self._documents = None
        self._embeddings = None

        return await self.load_all()

    def is_loaded(self) -> bool:
        """Check if data has been loaded."""
        return self._loaded

    @property
    def entities(self):
        """Get entities DataFrame (may be None if not loaded)."""
        return self._entities

    @property
    def relationships(self):
        """Get relationships DataFrame (may be None if not loaded)."""
        return self._relationships

    @property
    def communities(self):
        """Get communities DataFrame (may be None if not loaded)."""
        return self._communities

    @property
    def community_reports(self):
        """Get community reports DataFrame (may be None if not loaded)."""
        return self._community_reports

    @property
    def text_units(self):
        """Get text units DataFrame (may be None if not loaded)."""
        return self._text_units

    def __repr__(self) -> str:
        return f"GraphRAGDataLoader(loaded={self._loaded})"

    def __str__(self) -> str:
        return self.__repr__()
