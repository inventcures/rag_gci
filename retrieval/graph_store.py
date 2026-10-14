"""
Read-only access to the clinical knowledge graph, in the two shapes the answer
path needs: facts about named entities (for retrieval) and the relationships
between specific pairs (for the cross-reference check).

`Neo4jGraphStore` is the production implementation over `knowledge_graph`'s
Neo4j client. `InMemoryGraphStore` holds the same triples in a list; tests use it,
and it documents the contract without a database.

Every query here is template Cypher with parameters. Nothing is sent to an LLM,
so neither the user's question nor the drafted answer leaves the deployment on
this path.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Relationships that carry clinical meaning for retrieval and checking. Dosage
# relationships (DOSAGE_FOR) are deliberately absent: the tool must never give a
# specific dose, so dose facts are never pulled into a prompt from the graph.
CLINICAL_REL_TYPES = (
    "TREATS",
    "ALLEVIATES",
    "MANAGES",
    "PREVENTS",
    "SIDE_EFFECT_OF",
    "CAUSES",
    "CONTRAINDICATES",
)


class GraphUnavailable(RuntimeError):
    """The graph cannot be reached; the caller must fall back, not fail."""


@dataclass
class Fact:
    """One edge: (subject)-[rel]->(object), in the direction stored."""

    subject: str
    subject_type: str
    rel: str
    object: str
    object_type: str
    props: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PairRelations:
    """What the graph knows about one (a, b) pair."""

    a_known: bool
    b_known: bool
    # (relationship type, name of the node the edge starts from)
    rels: List[Tuple[str, str]] = field(default_factory=list)


class InMemoryGraphStore:
    """A list of facts with the same contract as Neo4jGraphStore."""

    def __init__(
        self,
        facts: Sequence[Fact],
        available: bool = True,
        version: str = "in-memory",
        delay_s: float = 0.0,
    ):
        self.facts = list(facts)
        self.available = available
        self._version = version
        self.delay_s = delay_s

    def is_available(self) -> bool:
        return self.available

    async def _maybe_delay(self) -> None:
        if self.delay_s:
            import asyncio

            await asyncio.sleep(self.delay_s)

    async def facts_about(self, names: Sequence[str], limit: int = 25) -> List[Fact]:
        if not self.available:
            raise GraphUnavailable("in-memory graph marked unavailable")
        await self._maybe_delay()
        wanted = {n.lower() for n in names}
        out = [
            f
            for f in self.facts
            if (f.subject.lower() in wanted or f.object.lower() in wanted)
            and f.rel in CLINICAL_REL_TYPES
        ]
        return out[:limit]

    async def relations_between(
        self, pairs: Sequence[Tuple[str, str]]
    ) -> List[PairRelations]:
        if not self.available:
            raise GraphUnavailable("in-memory graph marked unavailable")
        await self._maybe_delay()
        names = {f.subject.lower() for f in self.facts} | {
            f.object.lower() for f in self.facts
        }
        results = []
        for a, b in pairs:
            al, bl = a.lower(), b.lower()
            rels = [
                (f.rel, f.subject)
                for f in self.facts
                if {f.subject.lower(), f.object.lower()} == {al, bl}
            ]
            results.append(PairRelations(al in names, bl in names, rels))
        return results

    async def version(self) -> str:
        return self._version


class Neo4jGraphStore:
    """Template Cypher over the existing knowledge_graph Neo4j client."""

    _FACTS_CYPHER = """
    UNWIND $names AS n
    MATCH (a)-[r]-(b)
    WHERE toLower(a.name) = n AND type(r) IN $rel_types
    WITH DISTINCT r, startNode(r) AS s, endNode(r) AS e
    RETURN s.name AS subject, labels(s)[0] AS subject_type, type(r) AS rel,
           e.name AS object, labels(e)[0] AS object_type,
           r.evidence AS evidence, r.effectiveness AS effectiveness,
           r.frequency AS frequency
    LIMIT $limit
    """

    _PAIRS_CYPHER = """
    UNWIND $pairs AS p
    OPTIONAL MATCH (a) WHERE toLower(a.name) = p.a
    WITH p, collect(DISTINCT a) AS as_
    OPTIONAL MATCH (b) WHERE toLower(b.name) = p.b
    WITH p, as_, collect(DISTINCT b) AS bs
    OPTIONAL MATCH (x)-[r]-(y) WHERE x IN as_ AND y IN bs
    RETURN p.i AS i, size(as_) > 0 AS a_known, size(bs) > 0 AS b_known,
           collect(DISTINCT CASE WHEN r IS NULL THEN NULL
                   ELSE [type(r), startNode(r).name] END) AS rels
    LIMIT 100
    """

    _VERSION_CYPHER = """
    OPTIONAL MATCH (m:KGMeta) WITH m ORDER BY m.updated_at DESC LIMIT 1
    CALL { MATCH (n) RETURN count(n) AS nodes }
    CALL { MATCH ()-[r]->() RETURN count(r) AS rels }
    RETURN m.version AS label, nodes, rels
    LIMIT 1
    """

    def __init__(self, neo4j_client: Any, version_label: Optional[str] = None, version_ttl_s: float = 300.0):
        self.client = neo4j_client
        self.version_label = version_label
        self.version_ttl_s = version_ttl_s
        self._version_cache: Optional[Tuple[float, str]] = None

    @classmethod
    def from_kg_rag(cls, kg_rag: Any, version_label: Optional[str] = None) -> "Neo4jGraphStore":
        return cls(kg_rag.neo4j, version_label=version_label)

    def is_available(self) -> bool:
        # Configured AND connected. Checking the connection flag rather than
        # attempting a connect keeps a down database from costing a connect
        # timeout on every user turn.
        try:
            return bool(self.client.is_available() and getattr(self.client, "_connected", False))
        except Exception:
            return False

    async def _read(self, query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_available():
            raise GraphUnavailable("Neo4j is not connected")
        try:
            return await self.client.execute_read(query, params, raise_errors=True)
        except Exception as e:
            raise GraphUnavailable(f"Neo4j read failed: {e}") from e

    async def facts_about(self, names: Sequence[str], limit: int = 25) -> List[Fact]:
        rows = await self._read(
            self._FACTS_CYPHER,
            {
                "names": [n.lower() for n in names],
                "rel_types": list(CLINICAL_REL_TYPES),
                "limit": int(limit),
            },
        )
        return [
            Fact(
                subject=r.get("subject") or "",
                subject_type=r.get("subject_type") or "",
                rel=r.get("rel") or "",
                object=r.get("object") or "",
                object_type=r.get("object_type") or "",
                props={
                    k: r.get(k)
                    for k in ("evidence", "effectiveness", "frequency")
                    if r.get(k)
                },
            )
            for r in rows
        ]

    async def relations_between(
        self, pairs: Sequence[Tuple[str, str]]
    ) -> List[PairRelations]:
        rows = await self._read(
            self._PAIRS_CYPHER,
            {"pairs": [{"i": i, "a": a.lower(), "b": b.lower()} for i, (a, b) in enumerate(pairs)]},
        )
        by_index = {r.get("i"): r for r in rows}
        out = []
        for i in range(len(pairs)):
            r = by_index.get(i) or {}
            out.append(
                PairRelations(
                    a_known=bool(r.get("a_known")),
                    b_known=bool(r.get("b_known")),
                    rels=[(t, s) for t, s in (r.get("rels") or [])],
                )
            )
        return out

    async def version(self) -> str:
        """
        A label for the graph snapshot that answered.

        An explicit KG_VERSION wins. Otherwise a KGMeta node's version, else the
        node and relationship counts, which change whenever the graph is rebuilt
        or extended. Cached so the release log costs nothing on the hot path.
        """
        if self.version_label:
            return self.version_label
        now = time.monotonic()
        if self._version_cache and now - self._version_cache[0] < self.version_ttl_s:
            return self._version_cache[1]
        try:
            rows = await self._read(self._VERSION_CYPHER, {})
            row = rows[0] if rows else {}
            label = row.get("label") or f"neo4j:n{row.get('nodes', 0)}-r{row.get('rels', 0)}"
        except GraphUnavailable:
            label = "unavailable"
        self._version_cache = (now, label)
        return label
