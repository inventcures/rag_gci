# Palli Sahayak: Enhanced Architecture Incorporating Context-1, Always-On Memory, and HyperAgents

**Version**: 0.1.0
**Date**: 27 March 2026
**Status**: Research Integration Spec
**Builds on**: `v0_27march2026_0923st_210s_detailed-android-app-specs.md` (Android app) + `v50_20260327_0947ist_210s__changes-to-existing-rag_gci-codebase_detailed-specs.md` (backend changes)
**Research Sources**:
- Chroma Context-1 (March 2026): Self-editing agentic search model
- Google Always-On Memory Agent: Three-tier persistent memory architecture
- Meta HyperAgents (March 2026): Self-improving agent framework

---

## 1. Executive Summary

This document extends the Palli Sahayak Android app and backend architecture with insights from three cutting-edge research developments. Each addresses a specific gap in the current design:

| Research | Gap Addressed | Integration Point |
|----------|--------------|-------------------|
| **Context-1** (ChromaDB) | Multi-hop clinical retrieval quality | Backend RAG pipeline enhancement |
| **Always-On Memory** (Google) | Longitudinal patient memory consolidation | Backend + Android offline memory |
| **HyperAgents** (Meta) | Self-improving clinical response quality | Backend meta-agent layer |

**Key architectural insight**: These three systems are complementary, not competing. Context-1 improves *retrieval*, Always-On Memory improves *longitudinal context*, and HyperAgents improve *the improvement process itself*. Together they create a self-improving clinical decision support system with persistent patient memory and state-of-the-art retrieval.

---

## 2. Context-1 Integration: Agentic Multi-Hop Clinical Retrieval

### 2.1 What Context-1 Is

Context-1 is a 20B parameter agentic search model from ChromaDB that performs multi-hop document retrieval. It decomposes complex queries into subqueries, iteratively searches corpora, and returns ranked document sets. Key metrics:

- 400-500 tokens/second inference (MXFP4 quantized on B200)
- 0.94 prune accuracy (selectively removes irrelevant documents)
- 2.56 parallel tool calls per turn (68% more efficient than base)
- Matches frontier models at ~10x lower cost
- Training: SFT + CISPO (Clipped Importance-Sampled Policy Optimization) RL

### 2.2 Why It Matters for Palli Sahayak

The current RAG pipeline uses three parallel retrieval methods (ChromaDB vector, Neo4j graph, GraphRAG communities) unified via Reciprocal Rank Fusion. This works well for single-hop queries ("How to manage pain?") but struggles with multi-hop clinical reasoning:

- "What non-opioid alternatives exist for a patient with liver impairment and moderate pain who is already on paracetamol?"
- "If morphine causes constipation and the patient already has bowel obstruction, what alternative strong opioid avoids this?"

These require chaining 2-4 retrieval steps, exactly what Context-1 is designed for.

### 2.3 Integration Architecture

Context-1 cannot run on-device (20B parameters). It integrates as a **server-side retrieval enhancement** within the existing RAG pipeline:

```
Current:  Query → [ChromaDB | Neo4j | GraphRAG] → RRF → LLM → Response
Enhanced: Query → ComplexityClassifier
            ├── [Simple] → Current pipeline (unchanged)
            └── [Multi-hop] → Context-1 Agent → LLM → Response
```

### 2.4 Implementation: Backend Changes

#### New file: `context1_integration/agent.py`

```python
class Context1RetrievalAgent:
    """
    Agentic multi-hop retrieval using Context-1 patterns.
    Implements the tool suite and context management from the Context-1 paper.
    """

    TOOLS = {
        "search_corpus": "Hybrid BM25 + dense vector via RRF, top 50 results",
        "grep_corpus": "Regex pattern matching, up to 5 matches",
        "read_document": "Full document retrieval with reranking",
        "prune_chunks": "Context management through selective removal",
    }

    TOKEN_BUDGET = 32_768
    SOFT_THRESHOLD = 24_576
    HARD_CUTOFF = 28_672

    async def multi_hop_retrieve(self, query: str, corpus_id: str) -> RetrievalResult:
        """
        Decompose complex query into subqueries and iteratively search.
        Uses bounded token budget with three pressure mechanisms.
        """
        # 1. Decompose query into subqueries
        # 2. For each subquery, call search_corpus with deduplication
        # 3. Apply soft threshold pruning
        # 4. Return ranked, deduplicated chunks
```

#### New file: `context1_integration/complexity_classifier.py`

```python
class QueryComplexityClassifier:
    """
    Routes queries to simple (current pipeline) or multi-hop (Context-1).
    Classification based on: constraint count, entity references, temporal reasoning.
    """

    MULTI_HOP_INDICATORS = [
        "alternative", "instead of", "if.*then", "while also",
        "contraindicated", "interaction with", "given that",
    ]

    def classify(self, query: str) -> QueryComplexity:
        # Count constraint indicators
        # If >= 2 constraints or medical entity chains detected → MULTI_HOP
        # Otherwise → SIMPLE
```

#### Changes to `simple_rag_server.py`

```python
# In the query() method, before current retrieval:
complexity = complexity_classifier.classify(query_text)
if complexity == QueryComplexity.MULTI_HOP:
    result = await context1_agent.multi_hop_retrieve(query_text, corpus_id="palliative_care")
else:
    result = await current_rag_pipeline(query_text)  # unchanged
```

### 2.5 Android App Impact

**No changes to the Android app.** Context-1 operates entirely server-side. The app continues to call `/api/mobile/v1/query` — the response quality improves for complex multi-hop queries without any client changes.

The offline cache should include pre-computed multi-hop queries in the top 20 cache bundle, generated using Context-1 at bundle build time.

### 2.6 Key Insight from Context-1 Paper: Reward Structure for Palliative Care

The Context-1 CISPO reward function weights recall 16:1 over precision, reflecting that missing a relevant document is far worse than including an irrelevant one. This maps directly to clinical safety: **missing a contraindication is worse than showing an extra reference**.

Recommendation: Adopt the same recall-heavy reward weighting when training or fine-tuning any retrieval component for palliative care.

---

## 3. Always-On Memory Integration: Persistent Patient Context

### 3.1 What Always-On Memory Is

Google's Always-On Memory Agent uses a three-tier architecture:

1. **IngestAgent**: Processes multimodal inputs, extracts structured data
2. **ConsolidateAgent**: Background synthesis every 30 minutes, discovers cross-connections
3. **QueryAgent**: Retrieves and synthesizes with source citations

Critical design choice: **No vector embeddings. SQLite stores memories as structured records.** Consolidation creates explicit connection records between related memories. The LLM reasons over structure, not similarity.

### 3.2 Why It Matters for Palli Sahayak

The current longitudinal memory system (`personalization/longitudinal_memory.py`) stores observations as flat JSON files. It has no consolidation — if a patient reports pain 15 times over 3 months, there are 15 separate records with no synthesized insight like "pain trending upward, correlating with medication change on date X."

The Always-On Memory pattern addresses this directly:
- **Raw observations** remain immutable (ASHA records symptom during visit)
- **Consolidated insights** are generated periodically (system discovers patterns)
- **Query responses** cite specific observations (traceable, auditable)

### 3.3 Integration Architecture

```
Current:
  Observation → JSON file → LongitudinalPatientRecord → Manual query

Enhanced:
  Observation → IngestAgent → Raw Memory (SQLite)
                     ↓
              ConsolidateAgent (every 30 min)
                     ↓
              Consolidated Insights (SQLite)
                     ↓
              QueryAgent → Synthesized response with citations
```

### 3.4 Implementation: Backend Changes

#### New file: `memory_agents/__init__.py`

```python
"""
Always-On Memory Agent System for Palli Sahayak.
Three-tier architecture: Ingest → Consolidate → Query.
Based on Google's Always-On Memory Agent pattern.
"""
```

#### New file: `memory_agents/ingest_agent.py`

```python
class PatientIngestAgent:
    """
    Processes incoming observations and extracts structured memory records.
    Supports: voice transcripts, symptom reports, medication logs, vital signs.
    """

    async def ingest_observation(self, observation: TimestampedObservation) -> MemoryRecord:
        """
        Extract structured memory from observation.
        Returns: MemoryRecord with summary, entities, topics, importance_score.
        """
        # Use Groq LLM to extract:
        # - Summary: "Patient reports moderate pain (6/10) in lower back"
        # - Entities: ["pain", "lower_back", "moderate"]
        # - Topics: ["symptom_management", "pain_assessment"]
        # - Importance: 0.7 (scale 0-1, based on severity and clinical relevance)
        # - Source: observation_id for traceability
```

#### New file: `memory_agents/consolidate_agent.py`

```python
class PatientConsolidateAgent:
    """
    Background consolidation agent. Runs every 30 minutes.
    Discovers cross-connections between patient memories.
    Generates synthesized insights without discarding raw data.
    """

    CONSOLIDATION_INTERVAL_MINUTES = 30

    async def consolidate_patient(self, patient_id: str) -> List[ConsolidatedInsight]:
        """
        Review unconsolidated memories, generate insights.

        Examples of insights generated:
        - "Pain severity has increased from Mild to Moderate over 2 weeks"
        - "Morphine started on Day 5; constipation reported on Day 8 (likely side effect)"
        - "Patient's appetite improving since anti-emetic added"
        """
        raw_memories = await self.get_unconsolidated(patient_id)
        # Group by entity and temporal proximity
        # Use LLM to reason about connections
        # Generate explicit connection records
        # Mark memories as consolidated
```

#### New file: `memory_agents/query_agent.py`

```python
class PatientQueryAgent:
    """
    Retrieves and synthesizes answers from patient memory with source citations.
    Traverses both raw memories and consolidated insights.
    """

    async def query(self, patient_id: str, question: str) -> MemoryQueryResult:
        """
        Answer questions about a patient's history with citations.

        Example:
          Q: "How has the patient's pain changed since starting morphine?"
          A: "Pain decreased from 7/10 to 4/10 over the first week (observations
              #45, #52, #58). However, constipation was reported on day 3
              (observation #48), which is a known morphine side effect."
        """
```

### 3.5 Android App Changes

#### Observation timeline enhancement

The `PatientDetailScreen` gains a **"Insights" tab** alongside the raw observation timeline:

```kotlin
// In feature-patient/ui/PatientDetailScreen.kt
// Add tab: "Timeline" | "Insights"
// Insights tab shows consolidated insights from the server
// Each insight cites specific observations (tappable to jump to raw record)
```

#### New API endpoints for memory agents

```
GET  /api/mobile/v1/patient/{id}/insights     → Consolidated insights
POST /api/mobile/v1/patient/{id}/query-memory  → Natural language query against patient memory
```

#### Offline consolidation on device

For offline scenarios (CCF, CCHRC), a lightweight version of the ConsolidateAgent can run on-device:

```kotlin
// In core-evaluation or a new core-memory module:
class LocalConsolidationEngine {
    /**
     * Simple rule-based consolidation for offline use:
     * - Detect severity trends (3+ observations of same symptom)
     * - Flag medication-symptom temporal correlations
     * - No LLM needed — pattern matching on Room DB queries
     */
    suspend fun consolidatePatient(patientId: String): List<LocalInsight>
}
```

### 3.6 Key Insight: No Vector DB for Patient Memory

The Always-On Memory paper's most counterintuitive finding: **SQLite + LLM reasoning outperforms vector retrieval for structured memory**. This validates Palli Sahayak's existing approach (JSON file storage) while suggesting an upgrade path:

- Current: JSON files → manual query
- Proposed: SQLite (on device via Room) → LLM-assisted consolidation → cited responses

The Room database already exists in the Android app. The consolidation layer adds *reasoning* on top of *storage*.

---

## 4. HyperAgents Integration: Self-Improving Clinical Quality

### 4.1 What HyperAgents Are

Meta's HyperAgents (March 2026) introduces self-referential agents where:
- The **task agent** solves the clinical query
- The **meta agent** modifies both itself and the task agent
- The **improvement procedure itself** is editable (recursive self-enhancement)

Key result: "DGM-H improves performance over time and outperforms baselines without self-improvement or open-ended exploration."

### 4.2 Why It Matters for Palli Sahayak

The EVAH evaluation generates thousands of clinical interactions across 4 sites. Currently, the system learns nothing from these interactions — the same prompt, same RAG pipeline, same response quality from day 1 to day 240.

HyperAgents enable the system to **improve its own clinical response quality** over time:
- Learn which prompt structures produce physician-approved responses
- Discover which retrieval strategies work best for different query types
- Adapt medication guidance based on regional formulary differences across sites
- Self-improve safety detection based on false positive/negative feedback

### 4.3 Integration Architecture: Clinical Meta-Agent

```
                    ┌─────────────────────────────────┐
                    │         Meta Agent               │
                    │  (Self-improvement procedure)    │
                    │                                   │
                    │  - Evaluates response quality     │
                    │  - Modifies prompts               │
                    │  - Adjusts retrieval weights      │
                    │  - Evolves safety thresholds      │
                    │  - Archives improvement steps     │
                    └──────────┬──────────────────────┘
                               │ modifies
                    ┌──────────▼──────────────────────┐
                    │         Task Agent                │
                    │  (Clinical response pipeline)    │
                    │                                   │
                    │  - RAG retrieval                  │
                    │  - Response generation            │
                    │  - Safety validation              │
                    │  - Evidence grading               │
                    └──────────────────────────────────┘
```

### 4.4 Implementation: Backend Changes

#### New file: `meta_agent/clinical_meta_agent.py`

```python
class ClinicalMetaAgent:
    """
    HyperAgent-inspired meta-agent for Palli Sahayak.

    Implements recursive self-improvement within clinical safety boundaries:
    1. Evaluates response quality using physician feedback (expert sampling)
    2. Generates prompt/retrieval modifications
    3. Tests modifications in sandbox
    4. Archives successful improvements as "stepping stones"
    5. Periodically improves the improvement procedure itself

    Safety constraints (non-modifiable):
    - Emergency detection thresholds can only become MORE sensitive
    - Medication dosage ranges can only be validated, never relaxed
    - Human handoff triggers can only be added, never removed
    - All modifications logged for audit trail
    """

    SAFETY_CONSTRAINTS = {
        "emergency_sensitivity": "monotonically_increasing",
        "dosage_validation": "read_only",
        "handoff_triggers": "append_only",
        "evidence_thresholds": "monotonically_increasing",
    }

    async def evaluate_and_improve(self, batch: List[EvaluatedResponse]) -> List[Improvement]:
        """
        Given a batch of physician-evaluated responses:
        1. Identify patterns in high/low scoring responses
        2. Generate prompt modifications
        3. Test in sandbox with held-out examples
        4. If improvement > threshold, archive as stepping stone
        """

    async def meta_improve(self) -> MetaImprovement:
        """
        Improve the improvement procedure itself.
        Example: If the agent discovers that modifying retrieval weights
        produces better results than prompt editing, it increases the
        weight of retrieval modification in its strategy.
        """
```

#### New file: `meta_agent/improvement_archive.py`

```python
class ImprovementArchive:
    """
    Persistent archive of successful improvements (stepping stones).
    Each improvement records:
    - What was changed (prompt, retrieval weight, safety threshold)
    - Why (evaluation metrics before/after)
    - When (timestamp, evaluation batch)
    - Cross-domain applicability score
    """

    async def archive_improvement(self, improvement: Improvement) -> str:
        """Store improvement with before/after metrics."""

    async def get_transferable_improvements(self, domain: str) -> List[Improvement]:
        """
        Find improvements from other domains that may transfer.
        Example: A prompt improvement discovered at CMC Vellore (Tamil)
        may transfer to CCHRC Silchar (Bengali) if it's language-agnostic.
        """
```

### 4.5 Integration with EVAH Evaluation

The HyperAgent architecture directly enhances the EVAH evaluation design:

**Expert sampling (5%/50%/100%) becomes training data**:
- Physician-scored responses feed the meta-agent's evaluation loop
- The 2,400 vignette crossover responses provide controlled before/after data
- Cross-site learning workshops (Month 6) can be informed by meta-agent insights

**Self-improvement timeline**:
- Months 3-4 (Training): Meta-agent observes supervised deployment, builds baseline
- Months 5-8 (Active): Meta-agent begins improving after sufficient evaluation data
- Month 9-10 (Analysis): Meta-agent improvements quantified as part of study outcomes

### 4.6 Android App Impact

**No direct changes to the Android app.** The meta-agent operates entirely server-side. However, the app gains indirect benefits:
- Response quality improves over the evaluation period
- Evidence badge accuracy improves as thresholds are refined
- Emergency detection sensitivity improves through accumulated feedback

One addition: a **"Rate this response"** quick feedback mechanism in the voice query result screen:

```kotlin
// In feature-query/ui/VoiceQueryScreen.kt, after showing response:
Row {
    IconButton(onClick = { viewModel.rateResponse(queryId, positive = true) }) {
        Icon(Icons.Default.ThumbUp, "Helpful")
    }
    IconButton(onClick = { viewModel.rateResponse(queryId, positive = false) }) {
        Icon(Icons.Default.ThumbDown, "Not helpful")
    }
}
```

This lightweight feedback feeds the meta-agent's evaluation pipeline.

### 4.7 Safety Guardrails for Self-Improvement

From the HyperAgents paper: "All experiments were conducted with safety precautions (e.g., sandboxing, human oversight)."

For clinical deployment, additional constraints:

| Constraint | Implementation |
|------------|---------------|
| Sandbox testing | All improvements tested against held-out physician-scored examples before deployment |
| Clinical boundary | Safety thresholds can only become stricter, never relaxed |
| Human gate | Monthly clinician review of all meta-agent modifications |
| Audit trail | Every modification logged with before/after metrics |
| Rollback | One-click rollback to any previous stepping stone |
| Site isolation | Improvements tested at one site before cross-site deployment |

---

## 5. Unified Architecture: Three Systems Working Together

### 5.1 Data Flow

```
ASHA Worker asks clinical question via Android app
    │
    ▼
[1] Context-1 Complexity Classifier
    │
    ├── Simple query ──► Standard RAG (ChromaDB + Neo4j + GraphRAG)
    │
    └── Multi-hop query ──► Context-1 Agentic Retrieval
                              ├── search_corpus (hybrid BM25+dense)
                              ├── grep_corpus (pattern matching)
                              ├── read_document (full retrieval)
                              └── prune_chunks (context management)
    │
    ▼
[2] Response Generation (Groq LLM) with Patient Context
    │
    ├── Patient Memory Query Agent ──► Always-On Memory
    │     ├── Raw observations (immutable)
    │     ├── Consolidated insights (30-min synthesis)
    │     └── Cited connections
    │
    ▼
[3] Clinical Validation + Safety
    │
    ▼
[4] Evidence Badge + Response Delivery
    │
    ▼
[5] HyperAgent Meta-Evaluation (async, background)
    ├── Physician scoring feeds improvement loop
    ├── User feedback (thumbs up/down) feeds evaluation
    └── Meta-agent archives successful improvements
```

### 5.2 What Each System Contributes

| Phase | System | Contribution |
|-------|--------|-------------|
| Retrieval | **Context-1** | Multi-hop reasoning for complex queries (contraindications, drug interactions, conditional recommendations) |
| Context | **Always-On Memory** | Patient history with consolidated insights ("pain trending up since medication change"), cited observations |
| Improvement | **HyperAgents** | System improves response quality over time using evaluation data, without manual prompt engineering |
| Presentation | **Android App** | Voice-first interface, offline cache, evaluation instrumentation (unchanged architecture) |

### 5.3 Implementation Priority

| Priority | Component | Effort | Impact | When |
|----------|-----------|--------|--------|------|
| 1 (High) | Always-On Memory consolidation | Medium | High — directly improves clinical context for every query | Pre-evaluation (Month 1-2) |
| 2 (High) | Context-1 multi-hop routing | Medium | High — improves complex query answers | Pre-evaluation (Month 2-3) |
| 3 (Medium) | User feedback in Android app | Low | Medium — enables meta-agent training data | Month 3-4 |
| 4 (Medium) | HyperAgent meta-evaluation | High | High — but requires evaluation data to train | Month 5+ |
| 5 (Low) | On-device consolidation | Medium | Medium — only matters for offline at CCF/CCHRC | Month 4-5 |

---

## 6. Updated Backend File Manifest

### New modules (in addition to existing mobile_api, auth, sync, evaluation, offline):

```
context1_integration/
├── __init__.py
├── agent.py                    # Context-1 agentic retrieval
├── complexity_classifier.py    # Routes simple vs multi-hop queries
├── tools.py                    # search_corpus, grep_corpus, read_document, prune_chunks
└── config.py                   # Token budgets, thresholds

memory_agents/
├── __init__.py
├── ingest_agent.py             # Observation → structured memory extraction
├── consolidate_agent.py        # Background synthesis (30-min interval)
├── query_agent.py              # Natural language query with citations
├── memory_store.py             # SQLite-backed memory storage
└── scheduler.py                # Background consolidation scheduler

meta_agent/
├── __init__.py
├── clinical_meta_agent.py      # HyperAgent for clinical quality improvement
├── improvement_archive.py      # Stepping stone persistence
├── sandbox.py                  # Safe testing of modifications
├── safety_constraints.py       # Non-relaxable clinical boundaries
└── evaluator.py                # Response quality scoring
```

### Modified existing files:

```
simple_rag_server.py            # Add complexity routing, memory agent init, meta-agent hooks
personalization/longitudinal_memory.py  # Connect to memory_agents for consolidation
config.yaml                     # Add context1, memory_agents, meta_agent config sections
```

### Android app changes:

```
feature/feature-query/ui/VoiceQueryScreen.kt     # Add thumbs up/down feedback
feature/feature-patient/ui/PatientDetailScreen.kt # Add "Insights" tab
core/core-network/api/PalliSahayakApiService.kt   # Add insights + memory query endpoints
core/core-network/dto/ApiModels.kt                 # Add InsightsResponse, MemoryQueryResponse
```

---

## 7. Key Takeaways for EVAH Evaluation

### From Context-1:
- **Recall > Precision** for clinical retrieval (16:1 weighting). Missing a contraindication is worse than showing an extra reference.
- **Bounded context budgets** prevent hallucination from context overflow. Apply 32K token limit to RAG context window.
- **Parallel tool calling** (2.56 calls/turn) reduces latency. Apply to parallel retrieval from ChromaDB + Neo4j + GraphRAG.

### From Always-On Memory:
- **No vector DB for patient memory.** SQLite + LLM reasoning outperforms vector similarity for structured longitudinal data. The Room database on Android is the right choice.
- **Consolidation creates clinical value.** Raw observations alone are data; consolidated insights are knowledge.
- **Source citations enable audit.** Every synthesized insight must trace to specific observations for clinical accountability.

### From HyperAgents:
- **Self-improvement must be clinically constrained.** Safety thresholds can only become stricter, never relaxed.
- **Evaluation data is training data.** The EVAH study's expert sampling and vignette crossover design directly feed the meta-agent.
- **Cross-domain transfer is valuable.** Improvements discovered at one site should transfer to others, accelerating system-wide quality.
- **The improvement procedure itself should improve.** Don't just optimize prompts — optimize the process of optimizing prompts.

---

**End of Document**

*Version 0.1.0 | 27 March 2026 | Enhanced Architecture with Context-1, Always-On Memory, and HyperAgents*
