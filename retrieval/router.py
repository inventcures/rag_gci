"""
Routed retrieval for the `kg`, `graphrag` and `fused` RAG methods.

Each retriever returns a ranked list of passages; the lists are merged with the
shared Reciprocal Rank Fusion function and the top passages become the context
the answer is drafted from.

    kg        vector + Neo4j knowledge graph
    graphrag  vector + GraphRAG index
    fused     vector + PageIndex (when enabled) + knowledge graph + GraphRAG

Vector retrieval is always one of the lists. Graph retrievers run in parallel
with it under their own timeouts, and any that fail, time out or are not
configured are left out and recorded; the answer is then drafted from what did
come back. If vector retrieval itself fails, `retrieve` returns None and
`query()` falls back to the unchanged default vector path.

Data residency: the knowledge-graph retriever runs template Cypher against the
deployment's own Neo4j and the GraphRAG retriever reads the local parquet index
(no LLM, no embedding API). The user's question is not sent anywhere new on
either path. GraphRAG's own `local_search` / `global_search` call the model
configured in data/graphrag/settings.yaml (OpenAI by default, which is not on the
approved list), so they are deliberately not used here.

Dose safety: graph and GraphRAG text is passed through the dosage guard's
redaction before it can reach a prompt, and the knowledge-graph retriever never
reads dosage relationships.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from retrieval.embedder import vector_search
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.graph_store import Fact, GraphUnavailable
from retrieval.settings import RetrievalSettings

logger = logging.getLogger(__name__)

# The same relevance cut the default vector path applies (simple_rag_server).
VECTOR_RELEVANCE_THRESHOLD = 1.5

_REL_PHRASES = {
    "TREATS": "is used to treat",
    "ALLEVIATES": "can relieve",
    "MANAGES": "is used to manage",
    "PREVENTS": "can prevent",
    "SIDE_EFFECT_OF": "is a recognised side effect of",
    "CAUSES": "can cause",
    "CONTRAINDICATES": "is a contraindication for",
}


@dataclass
class Passage:
    """One retrieved unit of context, from any retriever."""

    key: str
    text: str
    retriever: str
    metadata: Dict[str, Any]
    distance: Optional[float] = None


@dataclass
class RetrieverOutcome:
    name: str
    status: str  # ok | empty | unavailable | timeout | error | not_configured
    passages: List[Passage] = field(default_factory=list)
    latency_ms: float = 0.0
    detail: str = ""


@dataclass
class RoutedRetrieval:
    """What query() needs to draft an answer, plus the record for the log."""

    passages: List[Passage]
    outcomes: List[RetrieverOutcome]
    rag_method: str
    fusion: Dict[str, Any]
    latency_ms: float

    @property
    def context_text(self) -> str:
        return "\n\n".join(_format_passage(p) for p in self.passages)

    @property
    def metadatas(self) -> List[Dict[str, Any]]:
        return [p.metadata for p in self.passages]

    @property
    def distances(self) -> List[float]:
        # Only real similarity distances feed the evidence badge. Graph facts have
        # none, and inventing one would overstate the evidence.
        return [p.distance for p in self.passages if p.distance is not None]

    @property
    def active(self) -> List[str]:
        return [o.name for o in self.outcomes if o.status in ("ok", "empty")]

    @property
    def failed(self) -> Dict[str, str]:
        return {
            o.name: o.status
            for o in self.outcomes
            if o.status not in ("ok", "empty")
        }

    def record(self) -> Dict[str, Any]:
        return {
            "rag_method": self.rag_method,
            "retrievers_requested": [o.name for o in self.outcomes],
            "retrievers_active": self.active,
            "retrievers_failed": self.failed,
            "contributing": sorted({p.retriever for p in self.passages}),
            "fusion": self.fusion,
            "stage_latency_ms": {
                f"retrieve_{o.name}": round(o.latency_ms, 1) for o in self.outcomes
            },
            "retrieval_latency_ms": round(self.latency_ms, 1),
        }


def _format_passage(p: Passage) -> str:
    meta = p.metadata
    if p.retriever == "vector":
        return f"Source: {meta['filename']} (chunk {meta['chunk_index'] + 1})\n{p.text}"
    if p.retriever == "pageindex":
        return f"Source: {meta['filename']} (pg {meta.get('page_range', '')})\n{p.text}"
    return f"Source: {meta['filename']}\n{p.text}"


def _redact_doses(text: str) -> str:
    """Strip dose sentences from graph-derived text before it reaches a prompt."""
    try:
        from dosage_guard import detect_doses, redact_doses

        if not detect_doses(text):
            return text
        redacted, _, _ = redact_doses(text)
        return redacted or ""
    except Exception:
        logger.exception("Dose redaction of graph context failed; dropping the text")
        return ""


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------


class VectorRetriever:
    name = "vector"

    def __init__(self, vector_db: Any, embedder: Any = None):
        self.vector_db = vector_db
        self.embedder = embedder

    async def retrieve(self, query: str, top_k: int) -> List[Passage]:
        # Off the event loop, so the graph retrievers run alongside it; the
        # embedder's cache means the probe's embedding is reused here.
        results = await vector_search(self.vector_db, self.embedder, query, top_k)
        docs = (results.get("documents") or [[]])[0] or []
        metas = (results.get("metadatas") or [[]])[0] or []
        dists = (results.get("distances") or [[]])[0] or [0.0] * len(docs)
        passages = []
        for doc, meta, dist in zip(docs, metas, dists):
            if dist > VECTOR_RELEVANCE_THRESHOLD:
                continue
            meta = dict(meta or {})
            meta.setdefault("filename", "unknown")
            meta.setdefault("chunk_index", 0)
            meta.setdefault("total_chunks", 1)
            meta["retriever"] = "vector"
            passages.append(
                Passage(
                    key=f"vector:{meta['filename']}:{meta['chunk_index']}",
                    text=doc,
                    retriever="vector",
                    metadata=meta,
                    distance=float(dist),
                )
            )
        return passages


class PageIndexRetriever:
    name = "pageindex"

    def __init__(self, engine: Any):
        self.engine = engine

    async def retrieve(self, query: str, top_k: int) -> List[Passage]:
        result = await self.engine.search(query)
        passages = []
        for node in (result.selected_nodes or [])[:top_k]:
            page_range = f"{node.get('start_page', 0)}-{node.get('end_page', 0)}"
            filename = node.get("filename", "")
            passages.append(
                Passage(
                    key=f"pageindex:{filename}:{page_range}",
                    text=node.get("text", ""),
                    retriever="pageindex",
                    metadata={
                        "filename": filename,
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "page_range": page_range,
                        "retriever": "pageindex",
                    },
                )
            )
        return passages


class KnowledgeGraphRetriever:
    """
    Facts about the clinical entities named in the question.

    Entities come from the pattern lexicon (no LLM). One passage per entity,
    ranked so that facts linking two entities from the question come first:
    for "Is morphine OK for breathlessness?" the morphine-dyspnoea edge leads.
    """

    name = "kg"
    SOURCE_NAME = "Clinical knowledge graph"

    def __init__(self, graph: Any, extractor: Any = None, facts_limit: int = 40):
        self.graph = graph
        self._extractor = extractor
        self.facts_limit = facts_limit

    @property
    def extractor(self):
        if self._extractor is None:
            from retrieval.kg_check import ClaimExtractor

            self._extractor = ClaimExtractor()
        return self._extractor

    async def retrieve(self, query: str, top_k: int) -> List[Passage]:
        if self.graph is None or not self.graph.is_available():
            raise GraphUnavailable("knowledge graph unavailable")
        mentions = self.extractor.mentions(query)
        names = list(dict.fromkeys(m.name for m in mentions))
        if not names:
            return []
        facts: List[Fact] = await self.graph.facts_about(names, limit=self.facts_limit)
        asked = {n.lower() for n in names}

        by_entity: Dict[str, List[Fact]] = {n: [] for n in names}
        for f in facts:
            for n in names:
                if n.lower() in (f.subject.lower(), f.object.lower()):
                    by_entity[n].append(f)

        scored = []
        for n, fs in by_entity.items():
            if not fs:
                continue
            # Facts that join two asked-about entities first.
            fs.sort(
                key=lambda f: -int(f.subject.lower() in asked and f.object.lower() in asked)
            )
            links = sum(1 for f in fs if f.subject.lower() in asked and f.object.lower() in asked)
            text = _redact_doses(" ".join(_fact_sentence(f) for f in fs[:8]))
            if not text:
                continue
            scored.append((links, n, text))

        scored.sort(key=lambda t: -t[0])
        return [
            Passage(
                key=f"kg:{n.lower()}",
                text=text,
                retriever="kg",
                metadata={
                    "filename": self.SOURCE_NAME,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "entity": n,
                    "retriever": "kg",
                },
            )
            for _, n, text in scored[:top_k]
        ]


def _fact_sentence(f: Fact) -> str:
    phrase = _REL_PHRASES.get(f.rel, f.rel.lower().replace("_", " "))
    sentence = f"{f.subject} {phrase} {f.object}"
    evidence = f.props.get("evidence")
    note = f.props.get("effectiveness") or f.props.get("frequency")
    extras = [str(x) for x in (note, evidence) if x]
    if extras:
        sentence += f" ({'; '.join(extras)})"
    return sentence + "."


class GraphRAGRetriever:
    """Entity descriptions and relationships from the local GraphRAG index."""

    name = "graphrag"
    SOURCE_NAME = "GraphRAG index"

    def __init__(self, engine: Any):
        self.engine = engine

    async def retrieve(self, query: str, top_k: int) -> List[Passage]:
        loader = getattr(self.engine, "data_loader", None)
        if loader is None:
            raise GraphUnavailable("GraphRAG index not loaded")
        entities = await loader.search_entities(query, top_k=top_k)
        passages = []
        for ent in entities:
            name = str(ent.get("title") or ent.get("name") or "")
            if not name:
                continue
            parts = [f"{name}: {ent.get('description', '')}".strip()]
            rels = await loader.get_entity_relationships(name)
            for r in rels[:5]:
                desc = r.get("description") or ""
                if desc:
                    parts.append(desc)
            text = _redact_doses(" ".join(parts))
            if not text:
                continue
            passages.append(
                Passage(
                    key=f"graphrag:{name.lower()}",
                    text=text,
                    retriever="graphrag",
                    metadata={
                        "filename": self.SOURCE_NAME,
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "entity": name,
                        "retriever": "graphrag",
                    },
                )
            )
        return passages


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

METHOD_RETRIEVERS = {
    "kg": ("vector", "kg"),
    "graphrag": ("vector", "graphrag"),
    "fused": ("vector", "pageindex", "kg", "graphrag"),
}


class RetrievalRouter:
    """Runs the retrievers a method names, in parallel, and fuses them."""

    def __init__(self, settings: RetrievalSettings, retrievers: Optional[Dict[str, Any]] = None):
        self.settings = settings
        self.retrievers: Dict[str, Any] = dict(retrievers or {})

    def attach(self, name: str, retriever: Any) -> None:
        self.retrievers[name] = retriever

    def detach(self, name: str) -> None:
        self.retrievers.pop(name, None)

    def _timeout_for(self, name: str) -> Optional[float]:
        return {
            "kg": self.settings.kg_timeout_s,
            "graphrag": self.settings.graphrag_timeout_s,
            "pageindex": self.settings.pageindex_timeout_s,
        }.get(name)

    async def _run(self, name: str, query: str) -> RetrieverOutcome:
        started = time.perf_counter()
        retriever = self.retrievers.get(name)
        if retriever is None:
            return RetrieverOutcome(name, "not_configured", detail=f"{name} retriever not attached")
        try:
            coro = retriever.retrieve(query, self.settings.per_retriever_top_k)
            timeout = self._timeout_for(name)
            passages = await (asyncio.wait_for(coro, timeout) if timeout else coro)
            status = "ok" if passages else "empty"
            return RetrieverOutcome(name, status, passages, (time.perf_counter() - started) * 1000)
        except asyncio.TimeoutError:
            logger.warning("Retriever %s timed out; continuing without it", name)
            return RetrieverOutcome(name, "timeout", latency_ms=(time.perf_counter() - started) * 1000,
                                    detail=f"exceeded {self._timeout_for(name)}s")
        except GraphUnavailable as e:
            logger.warning("Retriever %s unavailable; continuing without it: %s", name, e)
            return RetrieverOutcome(name, "unavailable", latency_ms=(time.perf_counter() - started) * 1000,
                                    detail=str(e))
        except Exception as e:
            logger.exception("Retriever %s failed; continuing without it", name)
            return RetrieverOutcome(name, "error", latency_ms=(time.perf_counter() - started) * 1000,
                                    detail=f"{type(e).__name__}: {e}")

    async def retrieve(self, query: str, top_k: int, method: Optional[str] = None) -> Optional[RoutedRetrieval]:
        """
        Fused passages for `method`, or None when the caller must use the
        default vector path (vector retrieval failed).
        """
        method = method or self.settings.rag_method
        names: Sequence[str] = METHOD_RETRIEVERS.get(method, ("vector",))
        started = time.perf_counter()
        outcomes = list(await asyncio.gather(*(self._run(n, query) for n in names)))

        vector = next((o for o in outcomes if o.name == "vector"), None)
        if vector is None or vector.status not in ("ok", "empty"):
            logger.warning("Vector retrieval failed in routed mode %s; falling back to default path", method)
            return None

        lists = [o.passages for o in outcomes if o.passages]
        list_names = [o.name for o in outcomes if o.passages]
        fused = reciprocal_rank_fusion(
            lists, key=lambda p: p.key, k=self.settings.rrf_k, names=list_names
        )
        passages = [f.item for f in fused[:top_k]]
        for f in fused[:top_k]:
            f.item.metadata["rrf_score"] = round(f.score, 6)
            f.item.metadata["rrf_ranks"] = dict(f.ranks)

        return RoutedRetrieval(
            passages=passages,
            outcomes=outcomes,
            rag_method=method,
            fusion=self.settings.fusion_description(),
            latency_ms=(time.perf_counter() - started) * 1000,
        )
