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
        Import comprehensive palliative care knowledge base.

        This creates a rich knowledge graph with:
        - Symptom-medication relationships (treatments)
        - Condition-symptom relationships (causes)
        - Side effect relationships
        - Care goal and setting relationships
        - Non-pharmacological intervention relationships

        Based on:
        - WHO Pain Ladder
        - Palliative Care Formulary (PCF)
        - NICE Palliative Care Guidelines
        - IAHPC Essential Medicines List

        Returns:
            Import summary
        """
        results = {
            "treatments_added": 0,
            "side_effects_added": 0,
            "conditions_added": 0,
            "interventions_added": 0,
            "errors": 0
        }

        # =====================================================================
        # SYMPTOM-MEDICATION RELATIONSHIPS (TREATS/ALLEVIATES)
        # =====================================================================
        treatments = [
            # Pain management - WHO Pain Ladder Step 1 (Non-opioids)
            ("Paracetamol", "Pain", "first-line, mild pain", "WHO Pain Ladder Step 1"),
            ("Ibuprofen", "Pain", "mild-moderate pain", "WHO Pain Ladder Step 1"),
            ("Diclofenac", "Pain", "anti-inflammatory", "WHO Pain Ladder Step 1"),
            ("Naproxen", "Pain", "anti-inflammatory", "WHO Pain Ladder Step 1"),

            # Pain management - WHO Pain Ladder Step 2 (Weak opioids)
            ("Codeine", "Pain", "moderate pain", "WHO Pain Ladder Step 2"),
            ("Tramadol", "Pain", "moderate pain", "WHO Pain Ladder Step 2"),
            ("Dihydrocodeine", "Pain", "moderate pain", "WHO Pain Ladder Step 2"),

            # Pain management - WHO Pain Ladder Step 3 (Strong opioids)
            ("Morphine", "Pain", "first-line strong opioid", "WHO Pain Ladder Step 3"),
            ("Morphine", "Cancer Pain", "gold standard", "WHO Essential Medicines"),
            ("Oxycodone", "Pain", "alternative to morphine", "Palliative Care Formulary"),
            ("Fentanyl", "Pain", "transdermal, breakthrough", "Palliative Care Formulary"),
            ("Fentanyl", "Breakthrough Pain", "rapid onset", "Palliative Care Formulary"),
            ("Hydromorphone", "Pain", "renal impairment", "Palliative Care Formulary"),
            ("Methadone", "Pain", "complex pain, neuropathic", "Palliative Care Formulary"),
            ("Buprenorphine", "Pain", "transdermal option", "Palliative Care Formulary"),
            ("Alfentanil", "Pain", "subcutaneous infusion", "Palliative Care Formulary"),

            # Neuropathic pain
            ("Gabapentin", "Neuropathic Pain", "first-line", "NICE Neuropathic Pain"),
            ("Pregabalin", "Neuropathic Pain", "first-line", "NICE Neuropathic Pain"),
            ("Amitriptyline", "Neuropathic Pain", "first-line, low dose", "NICE Neuropathic Pain"),
            ("Duloxetine", "Neuropathic Pain", "SNRI option", "NICE Neuropathic Pain"),
            ("Carbamazepine", "Neuropathic Pain", "trigeminal neuralgia", "Palliative Care Formulary"),
            ("Lidocaine", "Neuropathic Pain", "topical patch", "Palliative Care Formulary"),
            ("Ketamine", "Neuropathic Pain", "refractory pain", "Palliative Care Formulary"),

            # Bone pain
            ("Paracetamol", "Bone Pain", "baseline analgesia", "Palliative Care Formulary"),
            ("Ibuprofen", "Bone Pain", "anti-inflammatory", "Palliative Care Formulary"),
            ("Dexamethasone", "Bone Pain", "adjuvant", "Palliative Care Formulary"),

            # Nausea and vomiting
            ("Ondansetron", "Nausea", "chemotherapy-induced", "Palliative Care Formulary"),
            ("Metoclopramide", "Nausea", "gastric stasis", "Palliative Care Formulary"),
            ("Domperidone", "Nausea", "gastric stasis, less CNS effects", "Palliative Care Formulary"),
            ("Cyclizine", "Nausea", "motion, vestibular", "Palliative Care Formulary"),
            ("Haloperidol", "Nausea", "chemical/metabolic causes", "Palliative Care Formulary"),
            ("Levomepromazine", "Nausea", "broad spectrum", "Palliative Care Formulary"),
            ("Dexamethasone", "Nausea", "raised ICP, adjuvant", "Palliative Care Formulary"),
            ("Ondansetron", "Vomiting", "5-HT3 antagonist", "Palliative Care Formulary"),
            ("Metoclopramide", "Vomiting", "prokinetic", "Palliative Care Formulary"),

            # Dyspnea (breathlessness)
            ("Morphine", "Dyspnea", "reduces perception", "Cochrane Review"),
            ("Oxycodone", "Dyspnea", "alternative opioid", "Palliative Care Formulary"),
            ("Midazolam", "Dyspnea", "anxiety component", "Palliative Care Formulary"),
            ("Lorazepam", "Dyspnea", "anxiety component", "Palliative Care Formulary"),
            ("Oxygen", "Dyspnea", "if hypoxic", "Palliative Care Formulary"),
            ("Dexamethasone", "Dyspnea", "airway obstruction", "Palliative Care Formulary"),

            # Death rattle (terminal secretions)
            ("Hyoscine", "Death Rattle", "first-line", "Palliative Care Formulary"),
            ("Glycopyrronium", "Death Rattle", "less sedating", "Palliative Care Formulary"),
            ("Atropine", "Death Rattle", "sublingual", "Palliative Care Formulary"),

            # Constipation
            ("Lactulose", "Constipation", "osmotic laxative", "Palliative Care Formulary"),
            ("Senna", "Constipation", "stimulant laxative", "Palliative Care Formulary"),
            ("Bisacodyl", "Constipation", "stimulant laxative", "Palliative Care Formulary"),
            ("Docusate", "Constipation", "softener", "Palliative Care Formulary"),
            ("Polyethylene Glycol", "Constipation", "osmotic, impaction", "Palliative Care Formulary"),
            ("Methylnaltrexone", "Constipation", "opioid-induced", "Palliative Care Formulary"),
            ("Naloxegol", "Constipation", "opioid-induced, oral", "Palliative Care Formulary"),

            # Anxiety
            ("Lorazepam", "Anxiety", "short-acting", "Palliative Care Formulary"),
            ("Diazepam", "Anxiety", "longer-acting", "Palliative Care Formulary"),
            ("Midazolam", "Anxiety", "acute, parenteral", "Palliative Care Formulary"),
            ("Clonazepam", "Anxiety", "longer duration", "Palliative Care Formulary"),
            ("Sertraline", "Anxiety", "SSRI, if prognosis allows", "Palliative Care Formulary"),
            ("Mirtazapine", "Anxiety", "also helps sleep, appetite", "Palliative Care Formulary"),

            # Depression
            ("Sertraline", "Depression", "SSRI, first-line", "Palliative Care Formulary"),
            ("Citalopram", "Depression", "SSRI option", "Palliative Care Formulary"),
            ("Mirtazapine", "Depression", "also sedating, appetite", "Palliative Care Formulary"),
            ("Amitriptyline", "Depression", "also for pain, sleep", "Palliative Care Formulary"),

            # Delirium/Agitation
            ("Haloperidol", "Delirium", "first-line", "Palliative Care Formulary"),
            ("Olanzapine", "Delirium", "atypical option", "Palliative Care Formulary"),
            ("Risperidone", "Delirium", "atypical option", "Palliative Care Formulary"),
            ("Quetiapine", "Delirium", "atypical, sedating", "Palliative Care Formulary"),
            ("Midazolam", "Agitation", "acute, parenteral", "Palliative Care Formulary"),
            ("Lorazepam", "Agitation", "acute", "Palliative Care Formulary"),
            ("Haloperidol", "Agitation", "antipsychotic", "Palliative Care Formulary"),
            ("Phenobarbital", "Agitation", "refractory", "Palliative Care Formulary"),

            # Insomnia
            ("Mirtazapine", "Insomnia", "sedating antidepressant", "Palliative Care Formulary"),
            ("Lorazepam", "Insomnia", "short-term", "Palliative Care Formulary"),
            ("Amitriptyline", "Insomnia", "low dose", "Palliative Care Formulary"),

            # Fatigue
            ("Dexamethasone", "Fatigue", "short-term benefit", "Palliative Care Formulary"),
            ("Methylprednisolone", "Fatigue", "steroid option", "Palliative Care Formulary"),

            # Anorexia/Cachexia
            ("Dexamethasone", "Anorexia", "appetite stimulant", "Palliative Care Formulary"),
            ("Dexamethasone", "Cachexia", "short-term", "Palliative Care Formulary"),
            ("Mirtazapine", "Anorexia", "appetite side effect", "Palliative Care Formulary"),

            # Seizures
            ("Midazolam", "Seizures", "acute, buccal/SC", "Palliative Care Formulary"),
            ("Diazepam", "Seizures", "rectal", "Palliative Care Formulary"),
            ("Phenobarbital", "Seizures", "maintenance, SC", "Palliative Care Formulary"),
            ("Levetiracetam", "Seizures", "maintenance", "Palliative Care Formulary"),

            # Bowel obstruction
            ("Hyoscine", "Bowel Obstruction", "antisecretory", "Palliative Care Formulary"),
            ("Octreotide", "Bowel Obstruction", "reduces secretions", "Palliative Care Formulary"),
            ("Dexamethasone", "Bowel Obstruction", "may relieve", "Palliative Care Formulary"),
            ("Haloperidol", "Bowel Obstruction", "antiemetic", "Palliative Care Formulary"),

            # Hiccups
            ("Metoclopramide", "Hiccups", "gastric cause", "Palliative Care Formulary"),
            ("Haloperidol", "Hiccups", "central cause", "Palliative Care Formulary"),
            ("Baclofen", "Hiccups", "refractory", "Palliative Care Formulary"),

            # Pruritus (itching)
            ("Ondansetron", "Pruritus", "cholestatic", "Palliative Care Formulary"),
            ("Hydroxyzine", "Pruritus", "antihistamine", "Palliative Care Formulary"),

            # Fever
            ("Paracetamol", "Fever", "antipyretic", "Palliative Care Formulary"),
            ("Ibuprofen", "Fever", "antipyretic", "Palliative Care Formulary"),
        ]

        for medication, symptom, effectiveness, evidence in treatments:
            result = await self.add_treatment_relationship(
                medication, symptom, effectiveness, evidence
            )
            if "error" not in result:
                results["treatments_added"] += 1
            else:
                results["errors"] += 1

        # =====================================================================
        # SIDE EFFECT RELATIONSHIPS
        # =====================================================================
        side_effects = [
            # Opioid side effects
            ("Constipation", "Morphine", "common"),
            ("Drowsiness", "Morphine", "common, usually transient"),
            ("Nausea", "Morphine", "common initially"),
            ("Confusion", "Morphine", "elderly, high doses"),
            ("Constipation", "Oxycodone", "common"),
            ("Drowsiness", "Oxycodone", "common"),
            ("Constipation", "Fentanyl", "less than morphine"),
            ("Drowsiness", "Fentanyl", "common"),
            ("Constipation", "Codeine", "common"),
            ("Nausea", "Codeine", "common"),
            ("Constipation", "Tramadol", "less than other opioids"),
            ("Nausea", "Tramadol", "common"),
            ("Dizziness", "Tramadol", "common"),

            # Benzodiazepine side effects
            ("Drowsiness", "Lorazepam", "common"),
            ("Confusion", "Lorazepam", "elderly"),
            ("Drowsiness", "Midazolam", "expected"),
            ("Drowsiness", "Diazepam", "common"),
            ("Confusion", "Diazepam", "elderly"),

            # Antiemetic side effects
            ("Constipation", "Ondansetron", "common"),
            ("Headache", "Ondansetron", "occasional"),
            ("Drowsiness", "Haloperidol", "dose-related"),
            ("Drowsiness", "Levomepromazine", "marked"),

            # Steroid side effects
            ("Insomnia", "Dexamethasone", "give in morning"),
            ("Confusion", "Dexamethasone", "high doses"),
            ("Agitation", "Dexamethasone", "steroid psychosis"),

            # Antidepressant side effects
            ("Drowsiness", "Mirtazapine", "beneficial for sleep"),
            ("Drowsiness", "Amitriptyline", "beneficial for sleep"),
            ("Xerostomia", "Amitriptyline", "anticholinergic"),
            ("Constipation", "Amitriptyline", "anticholinergic"),

            # Anticholinergic side effects
            ("Xerostomia", "Hyoscine", "anticholinergic"),
            ("Confusion", "Hyoscine", "elderly"),
            ("Xerostomia", "Glycopyrronium", "anticholinergic"),
        ]

        for side_effect, medication, frequency in side_effects:
            await self.client.create_node("SideEffect", {"name": side_effect}, "name")
            await self.client.create_node("Medication", {"name": medication}, "name")
            result = await self.client.create_relationship(
                from_label="SideEffect",
                from_key="name",
                from_value=side_effect,
                to_label="Medication",
                to_key="name",
                to_value=medication,
                rel_type="SIDE_EFFECT_OF",
                properties={
                    "frequency": frequency,
                    "created_at": datetime.now().isoformat()
                }
            )
            if "error" not in result:
                results["side_effects_added"] += 1

        # =====================================================================
        # CONDITION-SYMPTOM RELATIONSHIPS (CAUSES)
        # =====================================================================
        condition_symptoms = [
            # Cancer symptoms
            ("Cancer", "Pain", "common"),
            ("Cancer", "Fatigue", "very common"),
            ("Cancer", "Anorexia", "common"),
            ("Cancer", "Cachexia", "advanced disease"),
            ("Cancer", "Nausea", "common"),
            ("Metastatic Cancer", "Bone Pain", "bone metastases"),
            ("Metastatic Cancer", "Dyspnea", "lung involvement"),
            ("Lung Cancer", "Dyspnea", "primary symptom"),
            ("Lung Cancer", "Cough", "common"),
            ("Lung Cancer", "Hemoptysis", "possible"),
            ("Brain Tumor", "Headache", "raised ICP"),
            ("Brain Tumor", "Confusion", "common"),
            ("Brain Tumor", "Seizures", "possible"),

            # Heart failure symptoms
            ("Heart Failure", "Dyspnea", "cardinal symptom"),
            ("Heart Failure", "Fatigue", "common"),
            ("Heart Failure", "Edema", "fluid overload"),

            # COPD symptoms
            ("COPD", "Dyspnea", "progressive"),
            ("COPD", "Cough", "chronic"),
            ("COPD", "Fatigue", "common"),

            # Dementia symptoms
            ("Dementia", "Confusion", "defining feature"),
            ("Dementia", "Agitation", "BPSD"),
            ("Dementia", "Dysphagia", "advanced"),

            # Kidney failure symptoms
            ("Kidney Failure", "Fatigue", "common"),
            ("Kidney Failure", "Nausea", "uremia"),
            ("Kidney Failure", "Pruritus", "uremia"),
            ("Kidney Failure", "Confusion", "uremia"),

            # Liver failure symptoms
            ("Liver Failure", "Confusion", "encephalopathy"),
            ("Liver Failure", "Ascites", "common"),
            ("Liver Failure", "Jaundice", "common"),
            ("Liver Failure", "Pruritus", "cholestasis"),

            # Motor neuron disease
            ("Motor Neuron Disease", "Dysphagia", "bulbar"),
            ("Motor Neuron Disease", "Dyspnea", "respiratory weakness"),
            ("Motor Neuron Disease", "Death Rattle", "secretions"),
        ]

        for condition, symptom, notes in condition_symptoms:
            await self.client.create_node("Condition", {"name": condition}, "name")
            await self.client.create_node("Symptom", {"name": symptom}, "name")
            result = await self.client.create_relationship(
                from_label="Condition",
                from_key="name",
                from_value=condition,
                to_label="Symptom",
                to_key="name",
                to_value=symptom,
                rel_type="CAUSES",
                properties={
                    "notes": notes,
                    "created_at": datetime.now().isoformat()
                }
            )
            if "error" not in result:
                results["conditions_added"] += 1

        # =====================================================================
        # NON-PHARMACOLOGICAL INTERVENTIONS
        # =====================================================================
        interventions = [
            # Pain interventions
            ("Massage Therapy", "Pain", "complementary"),
            ("Relaxation Techniques", "Pain", "mind-body"),
            ("Physiotherapy", "Pain", "mobility, function"),
            ("Nerve Block", "Pain", "refractory cases"),
            ("Repositioning", "Pain", "pressure relief"),

            # Dyspnea interventions
            ("Relaxation Techniques", "Dyspnea", "breathing exercises"),
            ("Repositioning", "Dyspnea", "upright position"),

            # Anxiety interventions
            ("Counseling", "Anxiety", "psychological support"),
            ("Relaxation Techniques", "Anxiety", "mind-body"),
            ("Music Therapy", "Anxiety", "complementary"),
            ("Aromatherapy", "Anxiety", "complementary"),

            # Depression interventions
            ("Counseling", "Depression", "psychological support"),

            # Constipation interventions
            ("Hydration", "Constipation", "fluid intake"),

            # Pressure ulcer interventions
            ("Wound Care", "Pressure Ulcer", "essential"),
            ("Repositioning", "Pressure Ulcer", "prevention"),

            # Oral care
            ("Mouth Care", "Xerostomia", "comfort measure"),
        ]

        for intervention, symptom, notes in interventions:
            await self.client.create_node("Intervention", {"name": intervention}, "name")
            await self.client.create_node("Symptom", {"name": symptom}, "name")
            result = await self.client.create_relationship(
                from_label="Intervention",
                from_key="name",
                from_value=intervention,
                to_label="Symptom",
                to_key="name",
                to_value=symptom,
                rel_type="ALLEVIATES",
                properties={
                    "notes": notes,
                    "type": "non-pharmacological",
                    "created_at": datetime.now().isoformat()
                }
            )
            if "error" not in result:
                results["interventions_added"] += 1

        logger.info(f"Imported palliative care knowledge base: {results}")
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
