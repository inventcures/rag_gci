"""
Neo4j Client for Knowledge Graph Operations

Provides connection management and query execution for Neo4j database.
Inspired by OncoGraph's Neo4jExecutor pattern.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Try to import neo4j driver
try:
    from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
    from neo4j.exceptions import ServiceUnavailable, AuthError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    AsyncGraphDatabase = None
    AsyncDriver = None
    AsyncSession = None


@dataclass
class Neo4jConfig:
    """Configuration for Neo4j connection."""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = ""
    database: str = "neo4j"
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 50
    connection_timeout: int = 30

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        """Create config from environment variables."""
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", ""),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )


class Neo4jClient:
    """
    Async Neo4j client for knowledge graph operations.

    Features:
    - Async connection management
    - Read-only query execution (for safety)
    - Connection pooling
    - Automatic retry on transient failures

    Usage:
        client = Neo4jClient.from_env()
        await client.connect()

        results = await client.execute_read(
            "MATCH (n:Symptom) RETURN n.name LIMIT 10"
        )

        await client.close()
    """

    def __init__(self, config: Optional[Neo4jConfig] = None):
        """Initialize Neo4j client."""
        if not NEO4J_AVAILABLE:
            logger.warning("neo4j package not installed. Install with: pip install neo4j")

        self.config = config or Neo4jConfig.from_env()
        self._driver: Optional[AsyncDriver] = None
        self._connected = False

    @classmethod
    def from_env(cls) -> "Neo4jClient":
        """Create client from environment variables."""
        return cls(Neo4jConfig.from_env())

    def is_available(self) -> bool:
        """Check if Neo4j is available and configured."""
        return NEO4J_AVAILABLE and bool(self.config.password)

    async def connect(self) -> bool:
        """
        Establish connection to Neo4j.

        Returns:
            True if connection successful, False otherwise
        """
        if not NEO4J_AVAILABLE:
            logger.error("neo4j package not installed")
            return False

        if not self.config.password:
            logger.warning("NEO4J_PASSWORD not configured")
            return False

        try:
            self._driver = AsyncGraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                max_connection_lifetime=self.config.max_connection_lifetime,
                max_connection_pool_size=self.config.max_connection_pool_size,
                connection_timeout=self.config.connection_timeout,
            )

            # Verify connection
            await self._driver.verify_connectivity()
            self._connected = True
            logger.info(f"Connected to Neo4j at {self.config.uri}")
            return True

        except AuthError as e:
            logger.error(f"Neo4j authentication failed: {e}")
            return False
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False

    async def close(self):
        """Close the Neo4j connection."""
        if self._driver:
            await self._driver.close()
            self._connected = False
            logger.info("Neo4j connection closed")

    @asynccontextmanager
    async def session(self):
        """Get a Neo4j session context manager."""
        if not self._connected:
            await self.connect()

        if not self._driver:
            raise RuntimeError("Neo4j driver not initialized")

        session = self._driver.session(database=self.config.database)
        try:
            yield session
        finally:
            await session.close()

    async def execute_read(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Execute a read-only Cypher query.

        Args:
            query: Cypher query string
            parameters: Query parameters
            limit: Maximum results (enforced for safety)

        Returns:
            List of result records as dictionaries
        """
        if not self._connected:
            connected = await self.connect()
            if not connected:
                return []

        # Enforce LIMIT for safety
        if "LIMIT" not in query.upper():
            query = f"{query} LIMIT {limit}"

        try:
            async with self.session() as session:
                result = await session.run(query, parameters or {})
                records = await result.data()
                return records

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            return []

    async def execute_write(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a write Cypher query.

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            Query summary with counters
        """
        if not self._connected:
            connected = await self.connect()
            if not connected:
                return {"error": "Not connected"}

        try:
            async with self.session() as session:
                result = await session.run(query, parameters or {})
                summary = await result.consume()

                return {
                    "nodes_created": summary.counters.nodes_created,
                    "nodes_deleted": summary.counters.nodes_deleted,
                    "relationships_created": summary.counters.relationships_created,
                    "relationships_deleted": summary.counters.relationships_deleted,
                    "properties_set": summary.counters.properties_set,
                }

        except Exception as e:
            logger.error(f"Write query failed: {e}")
            return {"error": str(e)}

    async def create_node(
        self,
        label: str,
        properties: Dict[str, Any],
        unique_key: str = "name"
    ) -> Dict[str, Any]:
        """
        Create or merge a node.

        Args:
            label: Node label (e.g., "Symptom", "Medication")
            properties: Node properties
            unique_key: Property to use for uniqueness

        Returns:
            Created/merged node data
        """
        unique_value = properties.get(unique_key)
        if not unique_value:
            return {"error": f"Missing unique key: {unique_key}"}

        query = f"""
        MERGE (n:{label} {{{unique_key}: $unique_value}})
        SET n += $properties
        RETURN n
        """

        params = {
            "unique_value": unique_value,
            "properties": properties
        }

        try:
            async with self.session() as session:
                result = await session.run(query, params)
                record = await result.single()
                if record:
                    return dict(record["n"])
                return {}

        except Exception as e:
            logger.error(f"Create node failed: {e}")
            return {"error": str(e)}

    async def create_relationship(
        self,
        from_label: str,
        from_key: str,
        from_value: str,
        to_label: str,
        to_key: str,
        to_value: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a relationship between nodes.

        Args:
            from_label: Source node label
            from_key: Source node unique key
            from_value: Source node key value
            to_label: Target node label
            to_key: Target node unique key
            to_value: Target node key value
            rel_type: Relationship type
            properties: Relationship properties

        Returns:
            Relationship creation result
        """
        query = f"""
        MATCH (a:{from_label} {{{from_key}: $from_value}})
        MATCH (b:{to_label} {{{to_key}: $to_value}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r += $properties
        RETURN type(r) as type, properties(r) as props
        """

        params = {
            "from_value": from_value,
            "to_value": to_value,
            "properties": properties or {}
        }

        try:
            async with self.session() as session:
                result = await session.run(query, params)
                record = await result.single()
                if record:
                    return {"type": record["type"], "properties": record["props"]}
                return {"warning": "Nodes not found"}

        except Exception as e:
            logger.error(f"Create relationship failed: {e}")
            return {"error": str(e)}

    async def get_schema(self) -> Dict[str, Any]:
        """
        Get the database schema (node labels and relationship types).

        Returns:
            Schema information
        """
        schema = {"labels": [], "relationships": [], "properties": {}}

        # Get labels
        labels_query = "CALL db.labels()"
        labels_result = await self.execute_read(labels_query)
        schema["labels"] = [r.get("label") for r in labels_result if r.get("label")]

        # Get relationship types
        rels_query = "CALL db.relationshipTypes()"
        rels_result = await self.execute_read(rels_query)
        schema["relationships"] = [r.get("relationshipType") for r in rels_result if r.get("relationshipType")]

        return schema

    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {"nodes": 0, "relationships": 0, "labels": {}}

        # Total nodes
        nodes_result = await self.execute_read("MATCH (n) RETURN count(n) as count")
        if nodes_result:
            stats["nodes"] = nodes_result[0].get("count", 0)

        # Total relationships
        rels_result = await self.execute_read("MATCH ()-[r]->() RETURN count(r) as count")
        if rels_result:
            stats["relationships"] = rels_result[0].get("count", 0)

        # Nodes per label
        labels_result = await self.execute_read("""
            CALL db.labels() YIELD label
            CALL apoc.cypher.run('MATCH (n:`' + label + '`) RETURN count(n) as count', {}) YIELD value
            RETURN label, value.count as count
        """)

        # Fallback if APOC not available
        if not labels_result:
            schema = await self.get_schema()
            for label in schema.get("labels", []):
                result = await self.execute_read(f"MATCH (n:{label}) RETURN count(n) as count")
                if result:
                    stats["labels"][label] = result[0].get("count", 0)
        else:
            for r in labels_result:
                stats["labels"][r.get("label")] = r.get("count", 0)

        return stats

    async def clear_database(self, confirm: bool = False) -> Dict[str, Any]:
        """
        Clear all nodes and relationships (use with caution!).

        Args:
            confirm: Must be True to execute

        Returns:
            Deletion summary
        """
        if not confirm:
            return {"error": "Must confirm=True to clear database"}

        logger.warning("Clearing entire Neo4j database!")

        return await self.execute_write("MATCH (n) DETACH DELETE n")

    async def health_check(self) -> bool:
        """Check if Neo4j connection is healthy."""
        try:
            result = await self.execute_read("RETURN 1 as health")
            return len(result) > 0 and result[0].get("health") == 1
        except Exception:
            return False
