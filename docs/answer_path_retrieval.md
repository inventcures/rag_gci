# Knowledge graph and GraphRAG on the answer path

Before this change the knowledge graph (`knowledge_graph/`, Neo4j) and Microsoft
GraphRAG (`graphrag_integration/`) were reachable only through their own
`/api/kg/*` and `/api/graphrag/*` endpoints and admin tabs. `SimpleRAGPipeline.query()`,
which every channel calls, routed only between ChromaDB vector search, PageIndex
and a vector + PageIndex hybrid, so nothing a user heard was informed by the graph.

This document covers what is now wired, how to turn it on, and what it costs.

**Nothing changes unless a flag or config turns it on.** With the defaults
(`rag_method: vector`, `kg_check: off`) `query()` takes the same code path as
before: no router is built, no checker is built, no graph is queried. The only
additions on the default path are stage timings and one append-only line in the
retrieval log per interaction (below).

## Turning it on

Three equivalent ways, highest precedence first:

```bash
# 1. CLI
python simple_rag_server.py --rag-method fused --kg-check block

# 2. Environment
RAG_METHOD=kg KG_CHECK_MODE=shadow python simple_rag_server.py

# 3. config.yaml
retrieval:
  rag_method: kg
  kg_check:
    mode: flag
```

Neo4j is read from `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, as the existing
`/api/kg/*` endpoints already do. The graph must be populated (for example with
`POST /api/kg/import-base`, which loads the base palliative-care graph from
`graph_builder.py`). GraphRAG needs `data/graphrag/settings.yaml` and an index in
`data/graphrag/output/`.

Enabling either setting changes the live release fingerprint
(`retrieval.rag_method`, `retrieval.kg_check`), so the server logs RELEASE DRIFT
against `data/study/release.yaml` until the record is updated and re-approved.
That is intended: the study needs to know which interactions ran on which path.

## RAG methods

| `--rag-method` | Retrievers | Combination |
|---|---|---|
| `vector` (default) | ChromaDB | unchanged |
| `pageindex` | PageIndex | unchanged |
| `hybrid` | ChromaDB + PageIndex | unchanged (concatenated) |
| `kg` | ChromaDB + Neo4j knowledge graph | Reciprocal Rank Fusion |
| `graphrag` | ChromaDB + GraphRAG index | Reciprocal Rank Fusion |
| `fused` | ChromaDB + PageIndex + Neo4j + GraphRAG | Reciprocal Rank Fusion |

All fusion goes through one function, `retrieval/fusion.py:reciprocal_rank_fusion`
(k = 60 by default, `retrieval.rrf_k`). `context1_integration/tools.py` now calls
it too, so there is one RRF implementation in the codebase.

How the routed methods retrieve:

- **Vector** is always one of the lists, with the same 1.5 L2 relevance cut as the
  default path. If vector retrieval fails, `query()` falls back to the default
  vector path, exactly as if no flag were set.
- **Knowledge graph**: clinical entities in the question are found with the
  `EntityExtractor` pattern lexicon (no LLM), then template Cypher fetches their
  treatment, side-effect, cause and contraindication edges. One passage per
  entity; edges that join two entities from the question rank first ("morphine"
  and "breathlessness" puts *Morphine is used to treat Dyspnea* at the top).
- **GraphRAG**: entity descriptions and relationships from the local parquet index
  (`data_loader.search_entities`). GraphRAG's own `local_search`/`global_search`
  are **not** used: they send the question to the LLM in
  `data/graphrag/settings.yaml` (Gemini API, not region-pinned).
- **PageIndex** (`fused` only): its selected nodes, when PageIndex is enabled.

Each graph retriever runs in parallel with vector search under its own timeout
(`retrieval.timeouts_s`, 2 s for the graph and GraphRAG). One that is down, slow,
or not configured is left out, the reason is logged, and the answer is drafted
from what came back.

## Knowledge-graph cross-reference check (protocol K26)

After the answer is drafted, `retrieval/kg_check.py` extracts medication claims
and checks each against Neo4j in a single query.

```
"Morphine also relieves constipation."
  -> claim: Morphine treats Constipation
  -> graph: (Constipation)-[:SIDE_EFFECT_OF]->(Morphine)
  -> verdict: contradicted
```

| Verdict | Meaning |
|---|---|
| supported | the graph has the edge the claim asserts |
| contradicted | the graph has an edge saying the opposite: a "treats" claim for a recorded side effect or contraindication, a "does not cause" claim for a recorded side effect, a "causes" claim for something the graph says it treats |
| unsupported | no edge either way, or the graph does not know one of the entities |

A relation is read only from the words between the two mentions, so
"Morphine can cause constipation, so give senna" yields *Morphine causes
Constipation* and nothing about senna. Claims are extracted from English text;
answers in other languages produce no claims and pass through unchanged.

| `--kg-check` | Effect on the answer |
|---|---|
| `off` (default) | not run |
| `shadow` | none; result logged. Use this first, to measure before acting. |
| `flag` | a caution line appended when any claim is contradicted or unsupported |
| `block` | sentences with contradicted claims removed, with a caution line; with `block_unsupported: true` unsupported ones too. If too little is left, a safe "please consult your doctor" answer replaces it. |

If Neo4j is down, slow (`kg_check.timeout_s`, 1.5 s) or errors, the check reports
`unavailable`, `timeout` or `error`, the drafted answer goes out unchanged, and
the result is logged. It never fails a turn. An interaction with a contradicted
claim is also put in the study log's 50% "flagged" expert-review tier, whatever
the mode did with it.

## Same path for every channel

Web, WhatsApp, the Gemini Live `search_medical_knowledge` tool, the Sarvam voice
fallback and Retell all call `SimpleRAGPipeline.query()`. The retrieval settings
and the check live behind that one method, so all channels behave identically.
Previously the voice handler wrote `rag_method="vector"` into the study log
regardless of configuration; it now records what `query()` actually used.

## Safety and residency

- **Doses.** The graph retriever never reads `DOSAGE_FOR` edges, and all graph and
  GraphRAG text passes through the dosage guard's redaction before it can reach a
  prompt. The existing dose boundary on the voice paths is unchanged; tests drive
  a dose-bearing draft through the Live tool and the Sarvam handler in `kg` mode
  and assert no dose reaches the user.
- **Data residency.** The graph retriever and the check run template Cypher
  against the deployment's own Neo4j; the GraphRAG retriever reads local files.
  No new external service sees a question or an answer.

## What is recorded per interaction

`query()` adds `retrieval` and `kg_check` to its result and appends one line to
`data/study/retrieval/retrieval-YYYY-MM-DD.jsonl` (`RETRIEVAL_AUDIT_DIR` overrides)
for every interaction on every channel:

```json
{"release_id": "...", "participant_id": "<keyed pseudonym>", "status": "success",
 "retrieval": {"rag_method": "kg", "retrievers_active": ["vector", "kg"],
               "retrievers_failed": {}, "fusion": {"method": "rrf", "k": 60, "per_retriever_top_k": 5},
               "kg_version": "neo4j:n118-r173",
               "stage_latency_ms": {"retrieve_vector": 4.1, "retrieve_kg": 9.8, "retrieval": 12.0,
                                    "generation": 2100.0, "kg_check": 10.8, "total": 2130.4}},
 "kg_check": {"mode": "block", "status": "ok", "action": "blocked", "contradicted": 1,
              "claims": [{"claim": "Morphine treats Constipation", "verdict": "contradicted", "...": "..."}]}}
```

No question or answer text is written, and the participant id is the same HMAC
pseudonym the study log uses. The study log (`InteractionLog`) gains the same two
fields where it is written (mobile and voice). The KG version is `KG_VERSION` if
set, else a `KGMeta` node's `version`, else the node and relationship counts.

## Latency

`scripts/benchmark_answer_path.py` runs a fixed set of 12 questions through
`query()` in each configuration. The run below used the live 263-chunk Chroma
index and a real Neo4j 5.26 loaded with the base palliative-care graph (118
nodes, 173 relationships). Query embedding and LLM generation take the same time
on the old and new paths, so the offline run excludes them: each question is
searched with a stored chunk embedding and the LLM returns a canned draft. What
remains is exactly what the new modes add. `--pipeline live` measures the whole
path in the deployment's environment.

| Configuration | retrieval ms (median / p95) | K26 check ms (median / p95) | total ms (median / p95) | added vs old (median) | contradicted claims | answers edited |
|---|---|---|---|---|---|---|
| old: vector | 4.0 / 7.1 | not run | 6.0 / 9.3 | +0.0 | 0 | 0 |
| vector + K26 check (block) | 4.1 / 7.3 | 19.8 / 52.5 | 27.9 / 64.2 | +21.9 | 2 | 2 |
| kg | 15.4 / 25.4 | not run | 17.8 / 28.1 | +11.8 | 0 | 0 |
| kg + K26 check (block) | 12.0 / 22.2 | 10.8 / 14.8 | 26.1 / 34.8 | +20.1 | 2 | 2 |
| fused + K26 check (block) | 10.1 / 14.9 | 8.9 / 12.3 | 22.4 / 27.9 | +16.4 | 2 | 2 |

PageIndex and GraphRAG had no index in this environment, so `fused` here is
vector + graph; PageIndex adds its own LLM tree search when enabled (capped by
`timeouts_s.pageindex`, 8 s). The two contradicted claims are the ones planted in
the set (Q4 "morphine relieves constipation", Q12 "morphine does not cause
constipation"); none of the other 21 claims was marked contradicted.

Failure cases:

- **Neo4j killed mid-session** (`kg` + `block`): all 12 answers delivered, median
  10.7 ms, max 18.1 ms; retriever and check both reported `unavailable`.
- **Neo4j hanging**: bounded by the timeouts, at most 2.0 s for retrieval (in
  parallel with vector search) plus 1.5 s for the check, well inside the 30 s
  voice budget.

## Limits

- Claim extraction is pattern-based and English-only. It finds the relations the
  graph models (treats, side effect, contraindication) between entities in the
  lexicon; anything else is not checked rather than guessed at.
- "Unsupported" mostly measures graph coverage. With the 118-node base graph many
  true statements will be unsupported, which is why `block` removes only
  contradicted claims unless `block_unsupported` is set, and why `shadow` is the
  sensible first setting.
- If Neo4j is unreachable at startup the graph stays off until restart; the
  server does not reconnect in the background.
