"""
Graph Builder for Palliative Care Knowledge Graph

Builds and populates the Neo4j knowledge graph from extracted entities.
Inspired by OncoGraph's GraphBuilder pattern.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .neo4j_client import Neo4jClient
from .entity_extractor import Entity, Relationship, EntityType, RelationshipType

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builds the knowledge graph from extracted entities and relationships.

    Features:
    - Creates/updates nodes with MERGE (upsert semantics)
    - Creates relationships between entities
    - Maintains provenance (source document, extraction date)
    - Supports batch operations for efficiency

    Usage:
        builder = GraphBuilder(neo4j_client)
        await builder.add_entities(entities)
        await builder.add_relationships(relationships)
        await builder.build_from_document(doc_id, entities, relationships)
    """

    def __init__(self, neo4j_client: Neo4jClient):
        """
        Initialize graph builder.

        Args:
            neo4j_client: Neo4j client instance
        """
        self.client = neo4j_client

    async def initialize_schema(self) -> Dict[str, Any]:
        """
        Initialize the graph schema with constraints and indexes.

        Returns:
            Schema creation results
        """
        results = {"constraints": [], "indexes": []}

        # Create unique constraints for each entity type
        for entity_type in EntityType:
            label = entity_type.value
            constraint_query = f"""
            CREATE CONSTRAINT {label.lower()}_name IF NOT EXISTS
            FOR (n:{label}) REQUIRE n.name IS UNIQUE
            """
            try:
                await self.client.execute_write(constraint_query)
                results["constraints"].append(f"{label}_name")
            except Exception as e:
                logger.warning(f"Constraint creation failed for {label}: {e}")

        # Create indexes for common queries
        index_queries = [
            "CREATE INDEX symptom_name_idx IF NOT EXISTS FOR (n:Symptom) ON (n.name)",
            "CREATE INDEX medication_name_idx IF NOT EXISTS FOR (n:Medication) ON (n.name)",
            "CREATE INDEX condition_name_idx IF NOT EXISTS FOR (n:Condition) ON (n.name)",
            "CREATE INDEX entity_source_idx IF NOT EXISTS FOR (n:Entity) ON (n.source_doc)",
        ]

        for query in index_queries:
            try:
                await self.client.execute_write(query)
                results["indexes"].append(query.split("INDEX ")[1].split(" IF")[0])
            except Exception as e:
                logger.warning(f"Index creation failed: {e}")

        logger.info(f"Schema initialized: {results}")
        return results

    async def add_entity(
        self,
        entity: Entity,
        source_doc: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add a single entity to the graph.

        Args:
            entity: Entity to add
            source_doc: Source document ID

        Returns:
            Created node data
        """
        properties = {
            "name": entity.name,
            **entity.properties,
            "confidence": entity.confidence,
            "created_at": datetime.now().isoformat(),
        }

        if source_doc:
            properties["source_doc"] = source_doc

        if entity.source_text:
            properties["source_text"] = entity.source_text[:500]

        return await self.client.create_node(
            label=entity.type.value,
            properties=properties,
            unique_key="name"
        )

    async def add_entities(
        self,
        entities: List[Entity],
        source_doc: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add multiple entities to the graph.

        Args:
            entities: List of entities
            source_doc: Source document ID

        Returns:
            Summary of created nodes
        """
        results = {"created": 0, "errors": 0, "entities": []}

        for entity in entities:
            result = await self.add_entity(entity, source_doc)
            if "error" in result:
                results["errors"] += 1
            else:
                results["created"] += 1
                results["entities"].append(entity.name)

        logger.info(f"Added {results['created']} entities, {results['errors']} errors")
        return results

    async def add_relationship(
        self,
        relationship: Relationship,
        source_doc: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add a relationship between entities.

        Args:
            relationship: Relationship to create
            source_doc: Source document ID

        Returns:
            Created relationship data
        """
        properties = {
            **relationship.properties,
            "confidence": relationship.confidence,
            "created_at": datetime.now().isoformat(),
        }

        if source_doc:
            properties["source_doc"] = source_doc

        return await self.client.create_relationship(
            from_label=relationship.source_type.value,
            from_key="name",
            from_value=relationship.source,
            to_label=relationship.target_type.value,
            to_key="name",
            to_value=relationship.target,
            rel_type=relationship.relationship.value,
            properties=properties
        )

    async def add_relationships(
        self,
        relationships: List[Relationship],
        source_doc: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add multiple relationships to the graph.

        Args:
            relationships: List of relationships
            source_doc: Source document ID

        Returns:
            Summary of created relationships
        """
        results = {"created": 0, "errors": 0, "warnings": 0}

        for rel in relationships:
            result = await self.add_relationship(rel, source_doc)
            if "error" in result:
                results["errors"] += 1
            elif "warning" in result:
                results["warnings"] += 1
            else:
                results["created"] += 1

        logger.info(f"Added {results['created']} relationships")
        return results

    async def build_from_document(
        self,
        doc_id: str,
        entities: List[Entity],
        relationships: List[Relationship]
    ) -> Dict[str, Any]:
        """
        Build graph from a document's extracted data.

        Args:
            doc_id: Document identifier
            entities: Extracted entities
            relationships: Extracted relationships

        Returns:
            Build summary
        """
        logger.info(f"Building graph from document {doc_id}")

        # First add all entities
        entity_results = await self.add_entities(entities, doc_id)

        # Then add relationships
        rel_results = await self.add_relationships(relationships, doc_id)

        return {
            "document_id": doc_id,
            "entities_added": entity_results["created"],
            "relationships_added": rel_results["created"],
            "errors": entity_results["errors"] + rel_results["errors"],
            "timestamp": datetime.now().isoformat()
        }

    async def add_treatment_relationship(
        self,
        medication: str,
        symptom: str,
        effectiveness: str = "effective",
        evidence: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add a TREATS relationship between medication and symptom.

        Args:
            medication: Medication name
            symptom: Symptom name
            effectiveness: Effectiveness level
            evidence: Evidence source

        Returns:
            Relationship data
        """
        # Ensure nodes exist
        await self.client.create_node(
            "Medication",
            {"name": medication},
            "name"
        )
        await self.client.create_node(
            "Symptom",
            {"name": symptom},
            "name"
        )

        # Create relationship
        return await self.client.create_relationship(
            from_label="Medication",
            from_key="name",
            from_value=medication,
            to_label="Symptom",
            to_key="name",
            to_value=symptom,
            rel_type="TREATS",
            properties={
                "effectiveness": effectiveness,
                "evidence": evidence,
                "created_at": datetime.now().isoformat()
            }
        )

    async def import_palliative_care_data(self) -> Dict[str, Any]:
        """
        Import base palliative care knowledge.

        This creates common symptom-medication relationships from
        palliative care guidelines.

        Returns:
            Import summary
        """
        # Common palliative care knowledge
        treatments = [
            # Pain management
            ("Morphine", "Pain", "first-line", "WHO Pain Ladder"),
            ("Paracetamol", "Pain", "adjuvant", "WHO Pain Ladder"),
            ("Ibuprofen", "Pain", "mild pain", "WHO Pain Ladder"),
            ("Fentanyl", "Pain", "severe pain", "Palliative Care Formulary"),
            ("Gabapentin", "Neuropathic Pain", "first-line", "NICE Guidelines"),

            # Nausea/vomiting
            ("Ondansetron", "Nausea", "first-line", "Palliative Care Formulary"),
            ("Metoclopramide", "Nausea", "prokinetic", "Palliative Care Formulary"),
            ("Haloperidol", "Nausea", "chemical causes", "Palliative Care Formulary"),

            # Breathlessness
            ("Morphine", "Breathlessness", "effective", "Cochrane Review"),
            ("Midazolam", "Breathlessness", "anxiolytic", "Palliative Care Formulary"),

            # Anxiety/depression
            ("Lorazepam", "Anxiety", "short-term", "Palliative Care Formulary"),
            ("Midazolam", "Anxiety", "acute", "Palliative Care Formulary"),

            # Constipation (often from opioids)
            ("Lactulose", "Constipation", "first-line", "Palliative Care Formulary"),
            ("Senna", "Constipation", "stimulant", "Palliative Care Formulary"),
            ("Bisacodyl", "Constipation", "stimulant", "Palliative Care Formulary"),

            # Other symptoms
            ("Dexamethasone", "Fatigue", "short-term", "Palliative Care Formulary"),
            ("Hyoscine", "Secretions", "effective", "Palliative Care Formulary"),
        ]

        results = {"added": 0, "errors": 0}

        for medication, symptom, effectiveness, evidence in treatments:
            result = await self.add_treatment_relationship(
                medication, symptom, effectiveness, evidence
            )
            if "error" in result:
                results["errors"] += 1
            else:
                results["added"] += 1

        # Add side effects
        side_effects = [
            ("Constipation", "Morphine"),
            ("Drowsiness", "Morphine"),
            ("Nausea", "Morphine"),
            ("Constipation", "Fentanyl"),
            ("Drowsiness", "Lorazepam"),
            ("Drowsiness", "Midazolam"),
        ]

        for side_effect, medication in side_effects:
            await self.client.create_node("SideEffect", {"name": side_effect}, "name")
            await self.client.create_relationship(
                from_label="SideEffect",
                from_key="name",
                from_value=side_effect,
                to_label="Medication",
                to_key="name",
                to_value=medication,
                rel_type="SIDE_EFFECT_OF",
                properties={"created_at": datetime.now().isoformat()}
            )

        logger.info(f"Imported palliative care data: {results}")
        return results

    async def get_entity_neighbors(
        self,
        entity_name: str,
        depth: int = 1
    ) -> Dict[str, Any]:
        """
        Get neighboring nodes for an entity.

        Args:
            entity_name: Name of the entity
            depth: How many hops to traverse

        Returns:
            Neighboring nodes and relationships
        """
        query = f"""
        MATCH (n {{name: $name}})
        CALL apoc.neighbors.tohop(n, '', {depth}) YIELD node
        RETURN n, collect(node) as neighbors
        """

        # Fallback if APOC not available
        fallback_query = """
        MATCH (n {name: $name})-[r]-(m)
        RETURN n, type(r) as rel_type, m
        LIMIT 50
        """

        try:
            results = await self.client.execute_read(query, {"name": entity_name})
            if results:
                return results[0]
        except Exception:
            pass

        # Try fallback
        results = await self.client.execute_read(fallback_query, {"name": entity_name})
        return {"entity": entity_name, "neighbors": results}
