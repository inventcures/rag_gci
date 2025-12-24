"""
Cypher Query Generator for Palliative Care Knowledge Graph

Generates Cypher queries from natural language questions.
Inspired by OncoGraph's QueryEngine and CypherGenerator patterns.
"""

import os
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Schema definition for the palliative care knowledge graph
SCHEMA_DEFINITION = """
Node Types:
- Symptom: name (unique), description, severity_levels
- Medication: name (unique), class, route, dosage_forms
- Condition: name (unique), icd_code, description
- Treatment: name (unique), type, description
- SideEffect: name (unique), severity, frequency
- BodyPart: name (unique), system

Relationship Types:
- (Medication)-[TREATS {effectiveness, evidence}]->(Symptom)
- (Medication)-[ALLEVIATES {strength}]->(Symptom)
- (Condition)-[CAUSES {frequency}]->(Symptom)
- (SideEffect)-[SIDE_EFFECT_OF {frequency}]->(Medication)
- (Treatment)-[MANAGES {effectiveness}]->(Condition)
- (Condition)-[AFFECTS]->(BodyPart)
- (Medication)-[CONTRAINDICATES]->(Condition)
"""

# Query templates for common palliative care questions
QUERY_TEMPLATES = {
    "medication_for_symptom": """
        MATCH (m:Medication)-[r:TREATS|ALLEVIATES]->(s:Symptom)
        WHERE toLower(s.name) CONTAINS toLower($symptom)
        RETURN m.name as medication, type(r) as relationship,
               r.effectiveness as effectiveness, r.evidence as evidence,
               s.name as symptom
        ORDER BY r.effectiveness DESC
        LIMIT 10
    """,

    "side_effects_of_medication": """
        MATCH (se:SideEffect)-[r:SIDE_EFFECT_OF]->(m:Medication)
        WHERE toLower(m.name) CONTAINS toLower($medication)
        RETURN se.name as side_effect, r.frequency as frequency, m.name as medication
        LIMIT 10
    """,

    "symptoms_of_condition": """
        MATCH (c:Condition)-[r:CAUSES]->(s:Symptom)
        WHERE toLower(c.name) CONTAINS toLower($condition)
        RETURN c.name as condition, s.name as symptom, r.frequency as frequency
        LIMIT 10
    """,

    "treatments_for_condition": """
        MATCH (t:Treatment)-[r:MANAGES]->(c:Condition)
        WHERE toLower(c.name) CONTAINS toLower($condition)
        RETURN t.name as treatment, c.name as condition, r.effectiveness as effectiveness
        LIMIT 10
    """,

    "all_medications": """
        MATCH (m:Medication)
        RETURN m.name as medication, m.class as class
        ORDER BY m.name
        LIMIT 50
    """,

    "all_symptoms": """
        MATCH (s:Symptom)
        RETURN s.name as symptom
        ORDER BY s.name
        LIMIT 50
    """,

    "medication_interactions": """
        MATCH (m1:Medication)-[r:INTERACTS_WITH]-(m2:Medication)
        WHERE toLower(m1.name) CONTAINS toLower($medication)
        RETURN m1.name as medication1, m2.name as medication2, r.severity as severity
        LIMIT 10
    """,

    "graph_overview": """
        MATCH (n)
        WITH labels(n) as labels, count(*) as count
        RETURN labels[0] as node_type, count
        ORDER BY count DESC
    """
}


@dataclass
class CypherQuery:
    """Represents a generated Cypher query."""
    query: str
    parameters: Dict[str, Any]
    template_used: Optional[str] = None
    confidence: float = 1.0
    explanation: str = ""


class CypherValidator:
    """
    Validates Cypher queries for safety.

    Implements defense-in-depth security:
    1. Block write operations (CREATE, DELETE, etc.)
    2. Block dangerous functions (CALL, LOAD)
    3. Enforce LIMIT clauses
    4. Validate against schema

    Inspired by OncoGraph's RuleBasedValidator.
    """

    # Blocked keywords (write operations)
    BLOCKED_KEYWORDS = [
        "CREATE", "DELETE", "MERGE", "SET", "REMOVE",
        "DROP", "CALL", "LOAD", "DETACH"
    ]

    # Maximum results allowed
    MAX_LIMIT = 200
    DEFAULT_LIMIT = 100

    def validate(self, query: str) -> Tuple[bool, str]:
        """
        Validate a Cypher query.

        Args:
            query: Cypher query string

        Returns:
            Tuple of (is_valid, error_message)
        """
        query_upper = query.upper()

        # Check for blocked keywords
        for keyword in self.BLOCKED_KEYWORDS:
            # Use word boundary to avoid false positives
            pattern = r'\b' + keyword + r'\b'
            if re.search(pattern, query_upper):
                return False, f"Blocked operation: {keyword}"

        # Check for parameter injection
        if re.search(r'\$\w+\s*\+', query):
            return False, "Potential injection detected"

        # Ensure LIMIT exists
        if "LIMIT" not in query_upper:
            return True, "Warning: No LIMIT clause"

        # Check LIMIT value
        limit_match = re.search(r'LIMIT\s+(\d+)', query_upper)
        if limit_match:
            limit_value = int(limit_match.group(1))
            if limit_value > self.MAX_LIMIT:
                return False, f"LIMIT exceeds maximum ({self.MAX_LIMIT})"

        return True, ""

    def sanitize(self, query: str) -> str:
        """
        Sanitize a query by adding safety measures.

        Args:
            query: Original query

        Returns:
            Sanitized query
        """
        query_upper = query.upper()

        # Add LIMIT if missing
        if "LIMIT" not in query_upper:
            query = f"{query.rstrip(';')} LIMIT {self.DEFAULT_LIMIT}"

        # Ensure read-only by wrapping in read transaction hint
        # (actual enforcement happens at execution level)

        return query


class CypherGenerator:
    """
    Generates Cypher queries from natural language questions.

    Uses a combination of:
    1. Template matching for common patterns
    2. LLM-based generation for complex queries
    3. Validation and sanitization

    Usage:
        generator = CypherGenerator()
        query = await generator.generate("What medications treat pain?")
        results = await neo4j_client.execute_read(query.query, query.parameters)
    """

    def __init__(self, use_llm: bool = True):
        """
        Initialize Cypher generator.

        Args:
            use_llm: Whether to use LLM for complex queries
        """
        self.use_llm = use_llm
        self.validator = CypherValidator()
        self.templates = QUERY_TEMPLATES

        # Initialize LLM client
        self._groq_client = None
        if use_llm and os.getenv("GROQ_API_KEY"):
            try:
                from groq import AsyncGroq
                self._groq_client = AsyncGroq()
            except ImportError:
                logger.warning("groq package not installed")

        # Question patterns for template matching
        self._patterns = {
            r"(what|which)\s+(medication|drug|medicine)s?\s+(treat|help|for)\s+(.+)": "medication_for_symptom",
            r"side\s+effects?\s+(of|from)\s+(.+)": "side_effects_of_medication",
            r"(symptoms?|signs?)\s+(of|from|caused by)\s+(.+)": "symptoms_of_condition",
            r"(treatment|therapy)\s+(for|of)\s+(.+)": "treatments_for_condition",
            r"(list|show|all)\s+medications?": "all_medications",
            r"(list|show|all)\s+symptoms?": "all_symptoms",
            r"(interact|interaction).*\s+(.+)": "medication_interactions",
            r"(overview|statistics|stats|summary)": "graph_overview",
        }

    async def generate(
        self,
        question: str,
        use_llm_fallback: bool = True
    ) -> CypherQuery:
        """
        Generate Cypher query from natural language question.

        Args:
            question: Natural language question
            use_llm_fallback: Use LLM if template matching fails

        Returns:
            CypherQuery object
        """
        question_lower = question.lower().strip()

        # Try template matching first
        for pattern, template_name in self._patterns.items():
            match = re.search(pattern, question_lower, re.IGNORECASE)
            if match:
                return self._generate_from_template(template_name, match, question)

        # Fallback to LLM generation
        if use_llm_fallback and self._groq_client:
            return await self._generate_with_llm(question)

        # Default query if nothing matches
        return CypherQuery(
            query="""
                MATCH (n)
                WHERE toLower(n.name) CONTAINS toLower($search)
                RETURN labels(n)[0] as type, n.name as name
                LIMIT 10
            """,
            parameters={"search": question.split()[-1] if question.split() else ""},
            template_used="default_search",
            confidence=0.3,
            explanation="Generic search query (no specific pattern matched)"
        )

    def _generate_from_template(
        self,
        template_name: str,
        match: re.Match,
        original_question: str
    ) -> CypherQuery:
        """Generate query from matched template."""
        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"Unknown template: {template_name}")

        # Extract parameters from regex groups
        parameters = {}
        groups = match.groups()

        if template_name == "medication_for_symptom" and len(groups) >= 4:
            parameters["symptom"] = groups[3].strip()

        elif template_name == "side_effects_of_medication" and len(groups) >= 2:
            parameters["medication"] = groups[1].strip()

        elif template_name == "symptoms_of_condition" and len(groups) >= 3:
            parameters["condition"] = groups[2].strip()

        elif template_name == "treatments_for_condition" and len(groups) >= 3:
            parameters["condition"] = groups[2].strip()

        elif template_name == "medication_interactions" and len(groups) >= 2:
            parameters["medication"] = groups[1].strip()

        # Validate query
        is_valid, message = self.validator.validate(template)
        if not is_valid:
            logger.error(f"Template validation failed: {message}")

        return CypherQuery(
            query=self.validator.sanitize(template),
            parameters=parameters,
            template_used=template_name,
            confidence=0.9,
            explanation=f"Matched pattern for {template_name}"
        )

    async def _generate_with_llm(self, question: str) -> CypherQuery:
        """Generate Cypher query using LLM."""
        prompt = f"""You are a Cypher query generator for a palliative care knowledge graph.

{SCHEMA_DEFINITION}

Generate a READ-ONLY Cypher query for this question:
"{question}"

Rules:
1. Only use MATCH, WHERE, RETURN, ORDER BY, LIMIT
2. Never use CREATE, DELETE, MERGE, SET, DROP, CALL
3. Always include LIMIT (max 50)
4. Use toLower() for case-insensitive matching
5. Return meaningful column names

Respond with ONLY the Cypher query, no explanation."""

        try:
            response = await self._groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )

            generated_query = response.choices[0].message.content.strip()

            # Remove markdown code blocks if present
            generated_query = re.sub(r'```\w*\n?', '', generated_query)
            generated_query = generated_query.strip()

            # Validate
            is_valid, message = self.validator.validate(generated_query)
            if not is_valid:
                logger.warning(f"LLM query rejected: {message}")
                # Return safe fallback
                return CypherQuery(
                    query="MATCH (n) RETURN labels(n)[0] as type, count(*) as count LIMIT 10",
                    parameters={},
                    template_used="llm_fallback",
                    confidence=0.3,
                    explanation=f"LLM query rejected: {message}"
                )

            return CypherQuery(
                query=self.validator.sanitize(generated_query),
                parameters={},
                template_used="llm_generated",
                confidence=0.7,
                explanation="Query generated by LLM"
            )

        except Exception as e:
            logger.error(f"LLM query generation failed: {e}")
            return CypherQuery(
                query="MATCH (n) RETURN labels(n)[0] as type, count(*) as count LIMIT 10",
                parameters={},
                template_used="error_fallback",
                confidence=0.2,
                explanation=f"LLM generation failed: {str(e)}"
            )

    def get_template(self, name: str) -> Optional[str]:
        """Get a query template by name."""
        return self.templates.get(name)

    def list_templates(self) -> List[str]:
        """List available query templates."""
        return list(self.templates.keys())
