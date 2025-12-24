"""
Knowledge Graph RAG Integration

Combines Knowledge Graph with RAG pipeline for enhanced retrieval.
Provides hybrid search: vector similarity + graph traversal.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from .neo4j_client import Neo4jClient
from .entity_extractor import EntityExtractor, Entity, Relationship
from .graph_builder import GraphBuilder
from .cypher_generator import CypherGenerator, CypherQuery
from .visualizer import GraphVisualizer, VisualizationData

logger = logging.getLogger(__name__)


class KnowledgeGraphRAG:
    """
    Hybrid RAG system combining vector search with knowledge graph.

    Features:
    - Automatic entity extraction from documents
    - Graph-based relationship discovery
    - Cypher query generation from natural language
    - Hybrid retrieval (vector + graph)
    - Graph visualization

    Architecture:
        Query → Entity Extraction → Cypher Generation → Graph Query
                                                            ↓
        Vector Search → Merge Results → Rerank → Response

    Usage:
        kg_rag = KnowledgeGraphRAG()
        await kg_rag.initialize()

        # Index a document
        await kg_rag.index_document(doc_id, chunks)

        # Query with hybrid retrieval
        results = await kg_rag.query("What medications treat pain?")
    """

    def __init__(
        self,
        neo4j_client: Optional[Neo4jClient] = None,
        use_llm: bool = True
    ):
        """
        Initialize Knowledge Graph RAG.

        Args:
            neo4j_client: Optional Neo4j client (creates from env if not provided)
            use_llm: Whether to use LLM for entity/query generation
        """
        self.neo4j = neo4j_client or Neo4jClient.from_env()
        self.entity_extractor = EntityExtractor(use_patterns=True)
        self.graph_builder = GraphBuilder(self.neo4j)
        self.cypher_generator = CypherGenerator(use_llm=use_llm)
        self.visualizer = GraphVisualizer()

        self._initialized = False

    async def initialize(self) -> bool:
        """
        Initialize the knowledge graph system.

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        # Try to connect to Neo4j
        if self.neo4j.is_available():
            connected = await self.neo4j.connect()
            if connected:
                # Initialize schema
                await self.graph_builder.initialize_schema()
                self._initialized = True
                logger.info("Knowledge Graph RAG initialized")
                return True
            else:
                logger.warning("Could not connect to Neo4j - running in limited mode")
        else:
            logger.warning("Neo4j not configured - running in limited mode")

        self._initialized = True  # Still mark as initialized for limited functionality
        return False

    async def close(self):
        """Close connections."""
        if self.neo4j:
            await self.neo4j.close()

    def is_graph_available(self) -> bool:
        """Check if graph database is available."""
        return self.neo4j.is_available() and self.neo4j._connected

    async def index_document(
        self,
        doc_id: str,
        chunks: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Index a document in the knowledge graph.

        Extracts entities and relationships, then adds to graph.

        Args:
            doc_id: Document identifier
            chunks: Document text chunks
            metadata: Optional document metadata

        Returns:
            Indexing results
        """
        if not self.is_graph_available():
            # Use pattern-based extraction only
            all_entities = []
            for chunk in chunks:
                entities = self.entity_extractor._extract_with_patterns(chunk)
                all_entities.extend(entities)

            entities = self.entity_extractor._deduplicate_entities(all_entities)

            return {
                "status": "limited",
                "doc_id": doc_id,
                "entities_found": len(entities),
                "entities": [e.to_dict() for e in entities],
                "graph_updated": False,
                "message": "Neo4j not available - entities extracted but not stored in graph"
            }

        # Full extraction with LLM
        all_entities = []
        all_relationships = []

        for chunk in chunks:
            entities, relationships = await self.entity_extractor.extract(chunk)
            all_entities.extend(entities)
            all_relationships.extend(relationships)

        # Deduplicate
        entities = self.entity_extractor._deduplicate_entities(all_entities)

        # Build graph
        result = await self.graph_builder.build_from_document(
            doc_id=doc_id,
            entities=entities,
            relationships=all_relationships
        )

        return {
            "status": "success",
            "doc_id": doc_id,
            "entities_found": len(entities),
            "relationships_found": len(all_relationships),
            "graph_result": result
        }

    async def query(
        self,
        question: str,
        include_vector_results: bool = True,
        vector_results: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Query the knowledge graph with natural language.

        Args:
            question: Natural language question
            include_vector_results: Whether to merge vector search results
            vector_results: Pre-computed vector search results
            top_k: Maximum results to return

        Returns:
            Query results with graph data and visualization
        """
        result = {
            "question": question,
            "graph_results": [],
            "cypher_query": None,
            "visualization": None,
            "entities_mentioned": [],
            "timestamp": datetime.now().isoformat()
        }

        # Extract entities from question
        question_entities, _ = await self.entity_extractor.extract(question, use_llm=False)
        result["entities_mentioned"] = [e.to_dict() for e in question_entities]

        # Generate and execute Cypher query
        if self.is_graph_available():
            try:
                cypher_query = await self.cypher_generator.generate(question)
                result["cypher_query"] = {
                    "query": cypher_query.query,
                    "parameters": cypher_query.parameters,
                    "template": cypher_query.template_used,
                    "confidence": cypher_query.confidence
                }

                # Execute query
                graph_results = await self.neo4j.execute_read(
                    cypher_query.query,
                    cypher_query.parameters,
                    limit=top_k
                )
                result["graph_results"] = graph_results

                # Generate visualization
                if graph_results:
                    viz_data = self.visualizer.from_query_results(graph_results)
                    result["visualization"] = viz_data.to_dict()

            except Exception as e:
                logger.error(f"Graph query failed: {e}")
                result["error"] = str(e)

        # Merge with vector results if provided
        if include_vector_results and vector_results:
            result["vector_results"] = vector_results
            result["merged_results"] = self._merge_results(
                result["graph_results"],
                vector_results
            )

        return result

    def _merge_results(
        self,
        graph_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge graph and vector results."""
        merged = []

        # Add graph results with source tag
        for r in graph_results:
            merged.append({
                **r,
                "_source": "graph",
                "_score": 0.9  # Graph results typically high relevance
            })

        # Add vector results with source tag
        for r in vector_results:
            merged.append({
                **r,
                "_source": "vector",
                "_score": r.get("score", 0.5)
            })

        # Sort by score
        merged.sort(key=lambda x: x.get("_score", 0), reverse=True)

        return merged

    async def get_entity_graph(
        self,
        entity_name: str,
        depth: int = 1
    ) -> Dict[str, Any]:
        """
        Get subgraph centered on an entity.

        Args:
            entity_name: Name of the center entity
            depth: Traversal depth

        Returns:
            Subgraph data with visualization
        """
        if not self.is_graph_available():
            return {"error": "Graph database not available"}

        # Query neighbors
        neighbors = await self.graph_builder.get_entity_neighbors(entity_name, depth)

        # Create visualization
        if isinstance(neighbors, list):
            viz_data = self.visualizer.create_subgraph(entity_name, neighbors)
        else:
            viz_data = VisualizationData()

        return {
            "entity": entity_name,
            "neighbors": neighbors,
            "visualization": viz_data.to_dict()
        }

    async def import_base_knowledge(self) -> Dict[str, Any]:
        """
        Import base palliative care knowledge into the graph.

        Returns:
            Import summary
        """
        if not self.is_graph_available():
            return {"error": "Graph database not available"}

        return await self.graph_builder.import_palliative_care_data()

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get knowledge graph statistics.

        Returns:
            Graph statistics
        """
        if not self.is_graph_available():
            return {
                "available": False,
                "message": "Graph database not connected"
            }

        stats = await self.neo4j.get_stats()
        schema = await self.neo4j.get_schema()

        return {
            "available": True,
            "stats": stats,
            "schema": schema,
            "timestamp": datetime.now().isoformat()
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Check knowledge graph health.

        Returns:
            Health status
        """
        neo4j_healthy = await self.neo4j.health_check() if self.is_graph_available() else False

        return {
            "neo4j_available": self.neo4j.is_available(),
            "neo4j_connected": self.neo4j._connected if hasattr(self.neo4j, '_connected') else False,
            "neo4j_healthy": neo4j_healthy,
            "entity_extractor": "ready",
            "cypher_generator": "ready",
            "visualizer": "ready"
        }

    def get_visualization_html(
        self,
        viz_data: VisualizationData
    ) -> str:
        """
        Generate standalone HTML visualization.

        Args:
            viz_data: Visualization data

        Returns:
            HTML string
        """
        return self.visualizer.generate_html(viz_data)

    async def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search for entities by name.

        Args:
            query: Search query
            entity_type: Optional type filter
            limit: Maximum results

        Returns:
            Matching entities
        """
        if not self.is_graph_available():
            return []

        if entity_type:
            cypher = f"""
            MATCH (n:{entity_type})
            WHERE toLower(n.name) CONTAINS toLower($query)
            RETURN n.name as name, labels(n)[0] as type
            LIMIT {limit}
            """
        else:
            cypher = f"""
            MATCH (n)
            WHERE toLower(n.name) CONTAINS toLower($query)
            RETURN n.name as name, labels(n)[0] as type
            LIMIT {limit}
            """

        return await self.neo4j.execute_read(cypher, {"query": query})

    async def get_treatments_for_symptom(
        self,
        symptom: str
    ) -> List[Dict[str, Any]]:
        """
        Get medications that treat a specific symptom.

        Args:
            symptom: Symptom name

        Returns:
            List of treatments
        """
        if not self.is_graph_available():
            return []

        cypher = """
        MATCH (m:Medication)-[r:TREATS|ALLEVIATES]->(s:Symptom)
        WHERE toLower(s.name) CONTAINS toLower($symptom)
        RETURN m.name as medication, type(r) as relationship,
               r.effectiveness as effectiveness, r.evidence as evidence,
               s.name as symptom
        ORDER BY r.effectiveness DESC
        LIMIT 10
        """

        return await self.neo4j.execute_read(cypher, {"symptom": symptom})

    async def get_side_effects(
        self,
        medication: str
    ) -> List[Dict[str, Any]]:
        """
        Get side effects of a medication.

        Args:
            medication: Medication name

        Returns:
            List of side effects
        """
        if not self.is_graph_available():
            return []

        cypher = """
        MATCH (se:SideEffect)-[r:SIDE_EFFECT_OF]->(m:Medication)
        WHERE toLower(m.name) CONTAINS toLower($medication)
        RETURN se.name as side_effect, r.frequency as frequency
        LIMIT 10
        """

        return await self.neo4j.execute_read(cypher, {"medication": medication})
