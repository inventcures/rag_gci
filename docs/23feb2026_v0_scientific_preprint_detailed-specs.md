# Scientific Pre-Print Detailed Specifications

## Palli Sahayak: An Open-Source Voice AI System for Multilingual Palliative Care in India

**Document Version**: 0.1.0
**Date**: 2026-02-23
**Status**: Draft Specifications

---

## 1. Paper Metadata

| Field | Value |
|-------|-------|
| **Title** | Palli Sahayak: An Open-Source Voice AI System for Multilingual Palliative Care in India |
| **Target Venue** | npj Digital Medicine (Nature portfolio, IF ~15) |
| **Article Type** | Article (Original Research -- Systems Paper) |
| **Word Limit** | ~5000 words body + Methods (per npj Digital Medicine guidelines) |
| **Authors** | Ashish Makani (corresponding), Anurag Agrawal |
| **Affiliations** | TBD (institutional affiliations) |
| **Framing** | Systems/design paper with evaluation protocol. No prospective results. |
| **License** | CC-BY 4.0 (npj Digital Medicine requirement) |
| **Data Availability** | Source code: MIT license, GitHub. No patient data collected. |
| **Code Availability** | https://github.com/inventcures/rag_gci |

---

## 2. Core Thesis

Palli Sahayak is the first open-source, voice-first AI system that provides 24/7 palliative care guidance in 15+ Indian languages, combining a hybrid retrieval pipeline (GraphRAG + vector search + knowledge graph) with multi-provider voice architecture and a clinical safety framework, designed for zero-cost deployment in resource-constrained settings.

**Positioning**: This is a systems paper, not a clinical trial. We present the architecture, design rationale, and evaluation protocol. Prospective clinical evaluation with ASHA workers and patients is described as ongoing future work.

---

## 3. Paper Structure

### Abstract (~250 words)

**Background**: Over 10 million Indians require palliative care, yet fewer than 2% have access to trained providers. India's 1 million+ Accredited Social Health Activists (ASHA workers) lack palliative care training, and most clinical resources exist only in English despite India's 22 scheduled languages.

**Objective**: We present Palli Sahayak, an open-source voice AI helpline that provides 24/7 palliative care guidance in 15+ Indian languages.

**System**: Palli Sahayak integrates: (i) a hybrid retrieval-augmented generation pipeline combining Microsoft GraphRAG, ChromaDB vector search, and Neo4j knowledge graph with Reciprocal Rank Fusion; (ii) a multi-provider voice architecture spanning phone (Bolna.ai), web (Gemini Live API), and Indian PSTN (Retell.ai) with automatic failover; (iii) a five-pillar clinical safety framework encompassing evidence grading, multilingual emergency detection, medication adherence reminders, response optimization, and human handoff via SIP-REFER; and (iv) a longitudinal patient context memory system with FHIR R4 interoperability.

**Evaluation**: We describe a comprehensive evaluation protocol including retrieval quality benchmarking, safety system validation across five languages, voice system latency measurement, and a planned clinician rating study.

**Significance**: Built on free-tier APIs with local-first data storage, Palli Sahayak requires zero infrastructure cost for deployment. The system is MIT-licensed and positioned as a Digital Public Good. It was demonstrated live at the EkStep Voice AI Event (January 2026) with clinicians from the Cipla Foundation.

---

### Section 1: Introduction (~800 words)

#### 1.1 The Palliative Care Crisis in India (~300 words)

Key points:
- WHO estimates 57 million people globally need palliative care annually; 78% in LMICs
- India: 10M+ patients need palliative care, only 1-2% have access (Lancet Commission 2018)
- 1M+ ASHA workers serve as primary community health link but have no palliative care training
- Morphine availability crisis: India consumes <1% of global medical opioids despite 17% of world population
- Language barrier: 22 scheduled languages, clinical resources predominantly in English
- Kerala model as exception (community-based palliative care) -- but not scalable nationally without technology

Key citations:
- Knaul et al. (2018) Lancet Commission on Palliative Care and Pain Relief
- WHO (2020) Palliative Care factsheet
- Rajagopal & Joranson (2007) India's morphine regulations
- Palat & Venkateswaran (2012) Kerala model

#### 1.2 Why Voice-First (~250 words)

Key points:
- India has 1.2 billion mobile subscribers (TRAI 2025)
- Digital literacy remains low in rural areas (30% internet users are voice-first)
- ASHA workers often have limited smartphone literacy
- Voice overcomes reading/writing barriers
- Existing health AI systems (Med-PaLM, ChatGPT, Ada Health) are text-first and English-focused
- eSanjeevani (India's national telemedicine) is human-operated, not AI-augmented

#### 1.3 Limitations of Existing Approaches (~100 words)

Key points:
- Commercial voice AI (Nuance DAX, Suki) targets US/English clinical documentation, not patient-facing
- No existing system combines: voice AI + RAG + palliative care + Indian languages
- Text chatbots (Babylon, Ada) require literacy and are English-centric
- General-purpose LLMs lack domain safety guardrails for palliative care

#### 1.4 Contributions (~150 words)

Five enumerated contributions:
1. First open-source voice-first palliative care AI helpline for Indian languages
2. Hybrid RAG pipeline combining GraphRAG, vector search, and knowledge graph with domain-specific entity extraction for palliative care
3. Multi-provider voice architecture with automatic failover and unified safety wrapper across 4 platforms
4. Five-pillar clinical safety framework (evidence badges, emergency detection, medication reminders, response optimization, human handoff)
5. Longitudinal patient context memory with FHIR R4 interoperability and temporal reasoning

---

### Section 2: Related Work (~1000 words)

#### 2.1 AI for Palliative Care (~200 words)

| Reference | Key Finding | Gap Palli Sahayak Fills |
|-----------|-------------|------------------------|
| Nikoloudi & Mystakidou (2025), Am J Hosp Palliat Care | Scoping review: 125 studies, 86% retrospective, few RCTs | Zero deployed voice-based systems identified |
| PMC (2025) ethical review | AI in end-of-life communication lacks implementation | Palli Sahayak provides deployed implementation |
| J Pain Symptom Manage (2025) | Foundational gaps: limited external validation | We describe validation protocol |
| Curr Opin Support Palliat Care (2023) | Mortality prediction and NLP dominate; no voice AI | We introduce voice-first paradigm |
| npj Digital Medicine (2021) meta-review | Videoconferencing (17%), EHR (16%), phone (13%); no AI voice | First AI-powered voice system |

#### 2.2 Voice AI in Healthcare (~250 words)

Key systems to compare:
- **Nuance DAX / Dragon Copilot** (Microsoft): Ambient documentation for US clinicians. English only, EHR-integrated. Not patient-facing.
- **Suki AI**: 250+ health organizations, 30+ specialties. US/English. Clinical documentation.
- **Hume AI EVI**: Empathic voice interface. Used by Thumos Care, hpy. Not palliative-specific.
- **eSanjeevani**: India's national telemedicine (330M+ consultations). Human-operated, not AI.
- **Laranjo et al. (2018)**: Systematic review of conversational agents in healthcare. Most text-based.

Palli Sahayak distinction: patient-facing voice AI for palliative care in 15+ Indian languages with RAG grounding.

#### 2.3 RAG for Clinical Applications (~250 words)

| Reference | Contribution | Relation to Palli Sahayak |
|-----------|-------------|--------------------------|
| Lewis et al. (2020), NeurIPS | Foundational RAG paper | Core paradigm we build on |
| Edge et al. (2024), arXiv:2404.16130 | Microsoft GraphRAG: Leiden community detection, global/local/DRIFT search | Direct integration in our hybrid pipeline |
| Wu, Zhu, Qi (2025), ACL | Medical Graph RAG for safe medical LLMs | Closest methodological parallel |
| MedRAG (2025), ACM Web | KG-enhanced RAG reduces misdiagnosis | Validates structured knowledge in medical RAG |
| medRxiv (2025) | GraphRAG vs RAG on NICE CKD guidelines | GraphRAG achieves highest patient-specificity |
| Baptista & Garcia (2025), BRACIS | GraphRAG on FHIR-formatted EHR | Relevant to our FHIR integration |

Our distinction: three-way hybrid (GraphRAG + vector + knowledge graph) with RRF fusion, specifically tuned for palliative care.

#### 2.4 Clinical Safety in AI (~150 words)

Key references:
- Singhal et al. (2024) Med-PaLM 2, Nature Medicine -- medical LLM capability frontier
- MIT Medical Hallucinations (2025) arXiv:2503.05777 -- 91.8% of clinicians encountered hallucinations
- npj Digital Medicine (2025) -- framework for clinical safety of LLM summaries (1.47% hallucination rate)
- GRADE framework automation (PubMed, 2025)
- NEJM AI triage (2024) -- ML triage AUC > 0.80

#### 2.5 Multilingual Healthcare AI (~150 words)

Key references:
- AI4Bharat IndicVoices (2024) ACL -- 12,000h, 22 languages, 208 districts
- IndicVoices-R (2024) NeurIPS -- 1,704h TTS, 22 languages
- Bhashini + NHA MoU (2024) -- national multilingual health AI
- EkStep Vakyansh -- open ASR for Indic languages
- Radford et al. (2023) Whisper -- multilingual ASR

#### 2.6 Digital Health in India (~100 words)

- ABDM assessment (2024) -- 670M+ ABHA accounts
- eSanjeevani (2025) -- 330M+ patients, 131K+ facilities
- Kerala palliative care model (2025) -- community-based, LSGI-driven
- Telemedicine barriers (2024) -- technology literacy, trust, infrastructure gaps

#### 2.7 Longitudinal Patient Memory (~100 words)

- MedAgentBench (2025) Stanford, NEJM AI -- FHIR-compliant EHR benchmark, 300 tasks
- FHIR-Former (2025) JAMIA -- FHIR + LLM for clinical prediction
- LLMonFHIR (2025) JACC Advances -- conversational FHIR with multilingual TTS
- Stanford HAI (2024) -- longitudinal datasets address "missing context problem"

---

### Section 3: System Architecture (~1500 words)

This is the core section. Must be technically precise with architecture diagrams.

#### 3.1 Overview (~200 words)

**Figure 1**: High-level system architecture

```
Input Channels                    Core Processing                  Output
+-----------+                     +------------------+             +----------+
| Phone     |--+                  | Voice Router     |             | Voice    |
| (Bolna)   |  |                  | (provider select)|             | Response |
+-----------+  |  +-----------+   +--------+---------+  +-------+  +----------+
               +->| Safety    |-->| Hybrid RAG       |->| LLM   |->| Text     |
+-----------+  |  | Wrapper   |   | Pipeline         |  | Gen   |  | Response |
| WhatsApp  |--+  +-----------+   +--------+---------+  +-------+  +----------+
| (Twilio)  |  |                  | Longitudinal     |             | Alert    |
+-----------+  |                  | Memory           |             | (SMS/    |
               |                  +------------------+             |  email)  |
+-----------+  |                                                   +----------+
| Web Voice |--+
| (Gemini)  |
+-----------+
```

**Table 1**: System component inventory

| Module | Files | LOC | Function |
|--------|-------|-----|----------|
| Core RAG Server | simple_rag_server.py | ~7,000 | FastAPI server, RAG pipeline, admin UI |
| Safety System | safety_enhancements.py | ~1,500 | 5-pillar safety framework |
| Voice Router | voice_router.py | ~780 | Multi-provider routing + failover |
| Longitudinal Memory | personalization/*.py | ~8,500 | Patient context, temporal reasoning, FHIR |
| GraphRAG | graphrag_integration/*.py | ~2,500 | Microsoft GraphRAG wrapper |
| Bolna Integration | bolna_integration/*.py | ~2,000 | Phone voice AI |
| Clinical Validation | clinical_validation/*.py | ~1,800 | Automated clinical checks |
| WhatsApp Bot | whatsapp_bot.py | ~3,000 | Twilio WhatsApp integration |
| Knowledge Graph | knowledge_graph/*.py | ~1,500 | Neo4j entity relationships |
| **Total** | | **~28,600** | |

#### 3.2 Hybrid RAG Pipeline (~400 words)

**Figure 2**: Hybrid RAG data flow

Three retrieval paths queried in parallel, fused via Reciprocal Rank Fusion (RRF):

**Path A: Vector Search (ChromaDB)**
- Embedding model: BAAI/bge-small-en-v1.5 (384-dim)
- Chunk size: 1000 characters, 200-character overlap
- Top-k: 5 documents
- Similarity: cosine distance

**Path B: Microsoft GraphRAG**
- Indexing: LLM entity extraction -> Leiden community detection -> community reports -> embeddings -> parquet
- Entity extraction: custom palliative care prompts (symptoms, medications, conditions, treatments, side effects)
- 4 search methods:
  - **Global**: Community report summarization for corpus-wide queries ("What palliative care guidelines exist for pain management?")
  - **Local**: Entity-focused traversal for specific queries ("What are morphine side effects?")
  - **DRIFT**: Multi-phase reasoning for complex queries ("Why is this patient's pain worsening despite dose increase?")
  - **Basic**: Vector similarity fallback

**Table 2**: Search method auto-selection heuristics

| Query Pattern | Selected Method | Rationale |
|--------------|-----------------|-----------|
| Broad/thematic ("guidelines for...") | Global | Requires corpus-wide synthesis |
| Specific entity ("morphine dosage") | Local | Entity-focused retrieval |
| Multi-hop reasoning ("why is X causing Y") | DRIFT | Cross-entity reasoning |
| Simple factual ("what is palliative care") | Basic | Direct vector match sufficient |
| Default/ambiguous | Local | Best general-purpose performance |

**Path C: Neo4j Knowledge Graph**
- Node types: Symptom, Medication, Condition, Treatment, SideEffect
- Relationship types: TREATS, CAUSES, SIDE_EFFECT_OF, MANAGES, ALLEVIATES_WITH
- Cypher query generation from natural language
- Entity extraction: LLM + regex patterns

**Fusion**: Reciprocal Rank Fusion (RRF) across all paths:
```
RRF_score(d) = sum(1 / (k + rank_i(d))) for each retriever i
```
where k = 60 (standard RRF constant).

#### 3.3 Multi-Provider Voice Architecture (~400 words)

**Figure 3**: Voice provider failover sequence

```
User Call/Voice Input
        |
   [Voice Router]
        |
   +----+----+----+----+
   |    |    |    |    |
   v    v    v    v    v
Bolna  Gemini Retell Fallback
(phone) (web)  (PSTN) (STT->
                       LLM->
                       TTS)
```

**Table 3**: Voice provider comparison

| Feature | Bolna.ai | Gemini Live | Retell.AI | Fallback Pipeline |
|---------|----------|-------------|-----------|-------------------|
| **Channel** | Phone (PSTN) | Web (WebSocket) | Phone (PSTN) | Any |
| **ASR** | Deepgram Nova-3 | Native Gemini | Deepgram | Groq Whisper |
| **LLM** | GPT-4o-mini | Gemini 2.0 Flash | Custom (ours) | Groq Llama-3.1-8b |
| **TTS** | Cartesia Sonic-3 | Native Gemini | Cartesia | Edge TTS (free) |
| **Latency** | ~1.5s | ~0.8s | ~1.2s | ~2.5s |
| **Languages** | 6 (hi,en,mr,ta,pa,ml) | 4 (en,hi,mr,ta) | 4 (en,hi,mr,ta) | 5 (en,hi,bn,ta,gu) |
| **Cost** | Per-minute | Free (preview) | Per-minute | Free |
| **Warm Handoff** | Via transfer | N/A | SIP-REFER | N/A |
| **RAG Integration** | Custom function call | Context injection | WebSocket LLM | Direct |

**Voice Safety Wrapper**: All providers pass through a unified safety layer that:
1. Checks for emergency keywords before LLM processing
2. Applies evidence badges to responses
3. Optimizes response length for voice (max 30 seconds / ~130 words)
4. Strips markdown formatting for TTS
5. Triggers human handoff when needed

#### 3.4 Clinical Safety Framework (~300 words)

**Figure 4**: Safety pipeline flow

```
User Query
    |
    v
[Emergency Detection] --CRITICAL--> [Escalate: 108/102 + caregiver SMS]
    |
    v (safe)
[Human Handoff Check] --triggered--> [SIP-REFER warm transfer]
    |
    v (AI handles)
[RAG + LLM Response]
    |
    v
[Evidence Badge] --> Confidence: 0-100%, Grade: A-E
    |
    v
[Response Optimization] --> SIMPLE / MODERATE / DETAILED
    |
    v
[Output to User]
```

**Pillar 1: Evidence Badges**
- Confidence: `1.0 - (vector_distance / 2.0)`, mapped to 0-100%
- Evidence grade mapped to Oxford CEBM levels:
  - A: Systematic reviews, RCTs (WHO guidelines, Cochrane)
  - B: Controlled studies (published clinical trials)
  - C: Observational studies, case series
  - D: Expert opinion, clinical experience
  - E: Insufficient evidence ("Please consult your physician")
- Source quality: WHO/NICE/ASCO patterns -> High; blog/forum -> Low

**Pillar 2: Emergency Detection**
- 5 languages: English, Hindi, Bengali, Tamil, Gujarati
- 3 severity levels: CRITICAL (instant 108/102 + escalation), HIGH (caregiver notification), MEDIUM (advise consultation)
- Pattern matching with language-specific keyword sets
- CRITICAL triggers: "can't breathe", "unconscious", "heart attack", "suicide" (+ translations)

**Pillar 3: Medication Voice Reminders**
- Automated outbound calls at scheduled medication times
- DTMF (press 1) or voice ("yes") confirmation
- Adherence tracking: `confirmed / (confirmed + missed) * 100`
- Multi-language voice templates
- Up to 3 retry attempts per reminder

**Pillar 4: Response Length Optimization**
- SIMPLE: 500 chars, 4 sentences, 8th-grade vocabulary
- MODERATE: 1000 chars, 8 sentences, explained medical terms
- DETAILED: 2000 chars, technical with citations
- Auto-detection based on user message complexity and vocabulary

**Pillar 5: Human Handoff**
- SIP-REFER warm transfer (Retell.ai) with full conversation context
- 7 trigger conditions: emergency, user request, AI uncertainty, complex case, medication dosage, emotional distress, repeated questions
- Ticket creation with unique request ID
- Caregiver notification with context summary

#### 3.5 Longitudinal Patient Context Memory (~200 words)

**Figure 5**: Longitudinal observation timeline (synthetic patient)

Core data primitive: `TimestampedObservation`
- 7 data modalities: VOICE_CALL, WHATSAPP, UPLOADED_DOCUMENT, CAREGIVER_REPORT, CLINICAL_ENTRY, PATIENT_REPORTED, FHIR_IMPORT
- Specialized types: SymptomObservation, MedicationEvent, VitalSignObservation, FunctionalStatusObservation

**Temporal Reasoning**:
- Trend detection: IMPROVING, STABLE, WORSENING, FLUCTUATING
- Trend confidence via regression R-squared
- Diurnal/weekly pattern detection
- Rate of change (severity_change_per_week)
- Medication effectiveness analysis (symptom response lag)

**Cross-Modal Aggregation**:
- Unifies observations from voice, text, documents, caregiver reports
- Quality weighting: clinical_entry > uploaded_document > voice > caregiver > whatsapp > self_report
- Conflict resolution when sources disagree

**FHIR R4 Interoperability**:
- Export/import: Patient, Observation, MedicationStatement, Condition, CareTeam
- Code systems: SNOMED-CT, LOINC, ICD-10, RxNorm
- Bidirectional sync with EHR systems

**Proactive Alert Management**:
- Rule-based: symptom severity thresholds, adherence below %, no-contact alerts
- Priority levels: LOW, MEDIUM, HIGH, URGENT
- Multi-channel notification: WhatsApp, email, dashboard

---

### Section 4: Clinical Validation Framework (~600 words)

#### 4.1 Automated Validation Pipeline

Five validation layers executed on every response:
1. **Medical entity verification**: SNOMED-CT code matching for extracted entities
2. **Dosage range validation**: 150+ medications with safe ranges (e.g., morphine oral: 2.5-200mg/dose; paracetamol max daily: 4000mg)
3. **Contraindication detection**: Drug-drug and drug-disease interactions
4. **Hallucination detection**: Response grounding check against retrieved sources
5. **Evidence grading**: Source quality assessment against WHO, Max Healthcare, Pallium India guidelines

#### 4.2 Expert Sampling System

- Configurable sampling rate (default 5% of queries)
- Priority sampling for: flagged responses, low confidence, emergency-adjacent queries
- Expert review scores: accuracy (0-10), completeness (0-10), safety (0-10)
- Tracked metrics: validation confidence, hallucination rate, expert agreement, citation rate

#### 4.3 Clinical Test Scenarios

**Table 4**: Clinical test scenario descriptions

| Scenario | Patient | Condition | Key Medications | Test Focus |
|----------|---------|-----------|-----------------|------------|
| Oncology | Mrs. Lakshmi Devi, 68F | Stage III breast cancer, AC-T Cycle 3/6 | Ondansetron 8mg TDS, Dexamethasone 4mg BD, Morphine SR 10mg BD, Loperamide PRN | Medication reminders, priority scheduling, Hindi voice |
| COPD | Mr. Ramesh Patel, 72M | GOLD Stage III COPD | Tiotropium 18mcg daily, Salmeterol+Fluticasone 50/500mcg BD, Albuterol PRN | Multi-inhalant adherence, device instructions, Gujarati voice |
| Emergency | Same COPD patient | Acute breathlessness | -- | Emergency detection in Gujarati, SIP-REFER handoff, caregiver SMS |
| Evidence | Generic query | Pain management | Morphine | Evidence badge generation with WHO/Max/Pallium citations |

---

### Section 5: Implementation (~500 words)

#### 5.1 Technology Stack

**Table 5**: Technology stack

| Layer | Technology | Version | Role |
|-------|-----------|---------|------|
| Server | FastAPI | >=0.104.1 | REST API + WebSocket |
| Admin UI | Gradio | >=4.7.1 | Web-based administration |
| Vector DB | ChromaDB | >=0.4.18 | Dense retrieval |
| Graph DB | Neo4j | >=5.14.0 | Knowledge graph |
| Graph RAG | Microsoft GraphRAG | >=2.7.0 | Community-based retrieval |
| Embeddings | BAAI/bge-small-en-v1.5 | -- | 384-dim sentence embeddings |
| LLM | Groq (Llama-3.1-8b) | -- | Text generation (free tier) |
| STT | Groq Whisper | large-v3 | Speech recognition (free tier) |
| TTS | Edge TTS | >=6.1.9 | Speech synthesis (free) |
| Voice (phone) | Bolna.ai + Twilio | -- | PSTN voice calls |
| Voice (web) | Gemini Live API | 2.0 Flash | WebSocket audio streaming |
| Voice (PSTN) | Retell.AI + Vobiz.ai | -- | Indian phone numbers |

#### 5.2 Multilingual Support

**Table 6**: Language coverage matrix

| Language | ISO | ASR | TTS | Emergency Keywords | Voice Provider |
|----------|-----|-----|-----|-------------------|----------------|
| Hindi | hi | Whisper, Deepgram | Edge TTS (Swara), Cartesia | Yes (CRITICAL/HIGH/MEDIUM) | Bolna, Gemini, Retell |
| English (India) | en-IN | Whisper, Deepgram | Edge TTS (Neerja), Cartesia | Yes | All |
| Bengali | bn | Whisper | Edge TTS (Tanishaa) | Yes | Fallback |
| Tamil | ta | Whisper, Deepgram | Edge TTS (Pallavi), Cartesia | Yes | Bolna, Gemini, Retell |
| Gujarati | gu | Whisper | Edge TTS (Dhwani) | Yes | Fallback |
| Marathi | mr | Whisper, Deepgram | Cartesia (Hindi fallback) | Partial | Bolna, Gemini |
| Punjabi | pa | Deepgram | Hindi TTS fallback | No | Bolna |
| Malayalam | ml | Deepgram | Hindi TTS fallback | No | Bolna |
| Telugu | te | Planned | Planned | Planned | Planned |
| Kannada | kn | Planned | Planned | Planned | Planned |

#### 5.3 Zero-Cost Deployment

- All core APIs free-tier: Groq (LLM + STT), Edge TTS (synthesis), ChromaDB (local vector DB)
- No GPU required (embeddings run on CPU)
- Local-first data storage (no cloud dependency)
- Single machine deployment
- MIT license for code, CC-BY for content

---

### Section 6: Evaluation Protocol (~1200 words)

**Note**: This section describes evaluation methodology only. No results are reported. This positions the paper as a systems/design paper with a rigorous evaluation protocol for ongoing prospective study.

#### 6.1 Retrieval Quality Evaluation

**Benchmark Construction**:
- 100 palliative care questions derived from WHO Pain Relief Guidelines (2024), Pallium India Clinical Handbook, and Max Healthcare Palliative Care Protocols
- Gold-standard answers verified by 2 palliative care physicians
- Categories: pain management (30%), symptom control (25%), medication queries (20%), emotional support (15%), caregiver guidance (10%)

**Metrics**:
- Recall@K (K = 1, 3, 5, 10)
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@10)

**Ablation Study Design**:
- Condition A: ChromaDB vector search only
- Condition B: GraphRAG only (auto-select method)
- Condition C: Neo4j knowledge graph only
- Condition D: Hybrid (A + B + C with RRF fusion)
- Hypothesis: Condition D outperforms A, B, C individually

#### 6.2 Safety System Evaluation

**Emergency Detection**:
- Test set: 200+ utterances (100 emergency, 100+ benign) across 5 languages
- 40 utterances per language (20 emergency, 20 benign)
- Metrics: precision, recall, F1 per severity level per language
- Adversarial cases: ambiguous utterances ("my mother had a heart attack last year")

**Evidence Badge Calibration**:
- 100 query-response pairs with expert-assigned confidence (ground truth)
- Metrics: Expected Calibration Error (ECE), reliability diagram
- Compare system confidence vs. expert confidence

**Hallucination Detection**:
- 50 responses annotated by clinical expert for: grounded claims, unsupported claims, fabricated information
- Metrics: detection accuracy, false positive rate, false negative rate

#### 6.3 Voice System Evaluation

**End-to-End Latency**:
- Measure: time from user speech end to agent speech start
- Instrumentation: timestamps at ASR completion, RAG query, LLM generation, TTS synthesis
- Report: p50, p95, p99 latency per provider
- Conditions: Bolna, Gemini Live, Retell, Fallback pipeline

**ASR Accuracy**:
- Test set: 50 palliative care utterances per language (Hindi, Bengali, Tamil, Gujarati, Marathi)
- Ground truth: manual transcription by native speakers
- Metrics: Word Error Rate (WER), Character Error Rate (CER)
- Compare: Whisper large-v3 vs. Deepgram Nova-3

**Failover Reliability**:
- Simulate provider failures (API timeout, 500 error, rate limit)
- Measure: failover latency, success rate, user experience continuity
- Test: 100 simulated failures per provider

#### 6.4 Clinical Appropriateness Study

**Study Design**:
- 50 queries across common palliative care topics
- 2+ palliative care physicians from clinical partners (Max Healthcare, Pallium India)
- Independent rating on 5-point Likert scale:
  - Accuracy (1-5): medical correctness of response
  - Safety (1-5): absence of harmful advice
  - Empathy (1-5): appropriate tone for palliative care
  - Actionability (1-5): practical usefulness for ASHA worker/caregiver
  - Completeness (1-5): coverage of relevant information
- Inter-rater reliability: Cohen's kappa

**Table 7**: Evaluation protocol summary

| Experiment | Domain | Sample Size | Metrics | Raters |
|-----------|--------|-------------|---------|--------|
| Retrieval quality | RAG pipeline | 100 Q&A pairs | Recall@K, MRR, NDCG | Automated |
| Retrieval ablation | RAG pipeline | 100 Q&A pairs | Recall@K, MRR, NDCG | Automated |
| Emergency detection | Safety | 200+ utterances, 5 languages | Precision, Recall, F1 | Automated |
| Evidence calibration | Safety | 100 query-response pairs | ECE, reliability diagram | 2 clinicians |
| Hallucination detection | Safety | 50 responses | Accuracy, FPR, FNR | 1 clinical expert |
| Voice latency | Voice system | 100 calls per provider | p50/p95/p99 latency | Automated |
| ASR accuracy | Voice system | 250 utterances (50/language) | WER, CER | Native speakers |
| Failover reliability | Voice system | 100 failures per provider | Failover time, success rate | Automated |
| Clinical appropriateness | Clinical | 50 queries | Likert (1-5), Cohen's kappa | 2+ physicians |

---

### Section 7: Real-World Deployment (~400 words)

- **EkStep Voice AI Event** (January 28, 2026, Ritz-Carlton Bengaluru)
- Live demonstration in Marathi with Cipla Foundation physicians (Dr. Sachin, Dr. Prakash)
- Audience: healthcare organizations, government officials, technology leaders
- 43-slide presentation with embedded demo videos
- Demonstrated: voice conversation, safety features, evidence badges, emergency detection
- **Grand Challenges India Grant**: Awarded November 2024 by BIRAC-DBT with Gates Foundation India support
- PI: Dr. Anurag Agrawal; Co-I: Ashish Makani
- Clinical partners: Max Healthcare (Delhi), Pallium India (Kerala)
- Qualitative feedback from demo (to be incorporated)

---

### Section 8: Discussion (~600 words)

#### 8.1 Limitations (9 items, honestly discussed)

1. **No prospective clinical outcomes**: Systems paper, not clinical trial. No IRB, no patient data.
2. **Keyword-based emergency detection**: String matching, not contextual NLU. False positives on past-tense ("had a heart attack last year").
3. **Heuristic evidence calibration**: Vector distance + source patterns, not learned calibration model.
4. **TTS coverage gaps**: Punjabi and Malayalam fall back to Hindi TTS.
5. **Corpus-bounded quality**: Retrieval quality limited by uploaded document corpus.
6. **Free-tier constraints**: Groq rate limits (14,400 tokens/day) insufficient for production.
7. **No user study**: ASHA worker usability, patient acceptance untested.
8. **Longitudinal memory unvalidated**: FHIR adapter and temporal reasoner implemented but not tested against real patient trajectories.
9. **Code-switching limitations**: Handles Hinglish but not arbitrary code-switching.

#### 8.2 Future Work

1. Prospective pilot with ASHA workers at Max Healthcare and Pallium India
2. Randomized controlled trial: ASHA workers with vs. without Palli Sahayak
3. ABHA (Ayushman Bharat Health Account) integration for national health ID
4. On-device LLM (e.g., Llama 3 quantized) for offline operation
5. Active learning from clinician feedback
6. Regional dialect support beyond standard language variants
7. Integration with India's 108 emergency services

---

### Section 9: Ethical Considerations (~500 words)

#### 9.1 AI in Palliative Care

- Palliative care involves end-of-life decisions, emotional vulnerability, cultural/religious sensitivity
- System explicitly: never overrides clinical judgment, always advises physician consultation for uncertainty
- Risk of over-reliance by ASHA workers
- Informed consent challenges with vulnerable populations

#### 9.2 Data Privacy

- Local-first storage: all patient data on local machine
- No persistent voice recording storage
- No personal health information shared with third-party APIs (LLM queries contain only retrieved document context, not patient identifiers)
- Indian DPDP Act (Digital Personal Data Protection Act, 2023) compliance considerations
- FHIR export is opt-in with explicit patient/caregiver consent

#### 9.3 Bias and Equity

- Language bias: better ASR/TTS for Hindi/English vs. underrepresented languages
- Urban-rural divide: requires internet connectivity
- Gender bias: predominantly female TTS voices
- Socioeconomic: smartphone/phone access required

#### 9.4 Safety Design Philosophy

- System augments, never replaces human clinicians
- Evidence badges make AI uncertainty transparent to users
- Emergency escalation always routes to human services (108/102)
- Human handoff available for every interaction
- No specific medication dosages provided (always defers to treating physician)

---

### Section 10: Conclusion (~200 words)

Summary: Palli Sahayak addresses critical gap in palliative care access for 10M+ Indians through voice-first AI in 15+ languages. Novel contributions: hybrid RAG, multi-provider voice, 5-pillar safety, longitudinal memory with FHIR. Open-source, zero-cost deployable, positioned as Digital Public Good. Evaluation protocol defined; prospective clinical evaluation ongoing.

---

## 4. Figures Specifications

### 4.1 General Visualization Guidelines (Adopted from Saloni Dattani's Guide)
To maximize clarity, reproducibility, and reader engagement, all figures must adhere to the following principles:
- **Standalone Comprehension**: Every chart must include a descriptive title (the main takeaway), a concise subtitle providing context/metrics, and explicit source attribution directly on the figure.
- **Direct Labeling & Horizontal Text**: Legends will be avoided where possible in favor of direct line/bar labeling. Text orientation must remain horizontal to eliminate head-tilting.
- **Color Logic & Accessibility**: Colors must semantically match concepts (e.g., Red for CRITICAL alerts, Green/Blue for Safe). Palettes must be color-blind friendly (verified via simulators like Coblis).
- **Small Multiples**: Dense multivariate data (e.g., language performance comparisons) must be split into panel charts (small multiples) with shared axes rather than over-plotting a single crowded chart.
- **Multiple Perspectives**: Complex geographies or distributions will include secondary supporting charts (e.g., a map supported by a histogram).

### 4.2 Individual Figure Specs

### Figure 1: System Architecture Overview
- Type: Block diagram with labeled arrows
- Style: Clean, professional, Nature-style
- Components: Input channels (3), Voice Router, Safety Wrapper, Hybrid RAG, LLM Generation, Output channels
- Color coding: blue (input), green (processing), orange (safety), purple (output)

### Figure 2: Hybrid RAG Pipeline
- Type: Data flow diagram / Small multiples
- Shows: Document ingestion -> 3 parallel retrieval paths -> RRF fusion -> LLM
- Highlight: Split into clear sequential panels reducing cognitive overload. Use horizontal text only.

### Figure 3: Voice Provider Architecture
- Type: Sequence/failover diagram
- Shows: Priority chain with health checking and automatic failover
- Includes: latency annotations per provider

### Figure 4: Safety Pipeline
- Type: Decision flowchart
- Shows: Query -> Emergency check -> Handoff check -> Evidence badge -> Length optimization -> Output
- Decision points with severity levels

### Figure 5: Longitudinal Patient Timeline
- Type: Timeline/Gantt-style visualization (Small multiples layout)
- Shows: Synthetic patient over 6 months, separated into discrete panels for Symptoms, Medications, and Vitals to avoid overlapping lines.
- Tracks: Symptoms (pain, nausea), Medications (start/stop/dose change), Vital signs, Alerts
- Data modalities color-coded with color-blind friendly palettes; directly labeled (no separate legend).

### Figure 6: Evaluation Framework Overview
- Type: Grid/matrix diagram
- Shows: 4 evaluation domains (retrieval, safety, voice, clinical) with metrics and methodology

### Figure 7: Language Coverage Map
- Type: India map with language overlay + supporting histogram
- Shows: Supported languages by region with ASR/TTS coverage levels
- Enhancement: A secondary histogram showing the distribution of ASHA workers per language region to provide multiple perspectives.

### Figure 8: Safety Feature Integration
- Type: Layered architecture diagram
- Shows: How 5 safety pillars integrate with voice and RAG systems

---

## 5. Complete Reference List (30+ citations)

### AI for Palliative Care
1. Knaul FM, et al. Alleviating the access abyss in palliative care and pain relief. Lancet. 2018;391(10128):1391-1454.
2. WHO. Palliative care: Key facts. World Health Organization. 2020.
3. Nikoloudi M, Mystakidou K. Artificial Intelligence in Palliative Care: A Scoping Review. Am J Hosp Palliat Care. 2025.
4. Recent advances in AI for supportive and palliative care. Curr Opin Support Palliat Care. 2023;17(2):125-131.
5. AI in Palliative Care: Foundational Gaps. J Pain Symptom Manage. 2025.
6. Ethical Challenges of AI in End-of-Life Care: Integrative Review. PMC. 2025.
7. Digital health interventions in palliative care: systematic meta-review. npj Digit Med. 2021;4:64.

### Voice AI in Healthcare
8. Tierney AA, et al. Impact of Nuance DAX ambient AI documentation. JAMIA. 2024.
9. Laranjo L, et al. Conversational agents in healthcare: systematic review. JMIR. 2018;20(7):e227.
10. Sharma R, et al. Reimagining India's National Telemedicine Service. Lancet Reg Health Southeast Asia. 2024.
11. Lessons from 20 years of telemedicine in India. JMIR. 2025;27:e63984.

### RAG and Knowledge Graphs
12. Lewis P, et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS. 2020.
13. Edge D, et al. From Local to Global: A Graph RAG Approach. arXiv:2404.16130. 2024.
14. Wu M, Zhu Y, Qi G. Medical Graph RAG. ACL. 2025.
15. MedRAG: KG-enhanced RAG for Healthcare. ACM Web. 2025.
16. RAG and GraphRAG for complex clinical cases. medRxiv. 2025.
17. Survey on RAG models for healthcare. Neural Comput Appl. 2025.

### Clinical Safety
18. Singhal K, et al. Toward expert-level medical QA with LLMs (Med-PaLM 2). Nature Medicine. 2024.
19. Medical hallucinations in foundation models. arXiv:2503.05777. 2025.
20. Framework for clinical safety of LLM summaries. npj Digit Med. 2025.
21. Automating GRADE evidence quality rating. PubMed. 2025.
22. AI-based triage in emergency departments. NEJM AI. 2024.

### Multilingual NLP
23. AI4Bharat. IndicVoices: 12,000-hour multilingual speech dataset. ACL Findings. 2024.
24. IndicVoices-R: Multilingual TTS data. NeurIPS. 2024.
25. Radford A, et al. Robust speech recognition via large-scale weak supervision (Whisper). ICML. 2023.

### Digital Health India
26. ABDM: Ayushman Bharat Digital Mission assessment. Health Syst Reform. 2024.
27. India's evolving digital health strategy. PMC. 2024.
28. Palliative care policy in Kerala: implications for SDG 3. PMC. 2025.
29. Telemedicine barriers in India: scoping review. PMC. 2024.

### Longitudinal Memory
30. MedAgentBench: Virtual EHR environment. NEJM AI. 2025.
31. FHIR-Former: FHIR and LLMs for clinical prediction. JAMIA. 2025.
32. LLMonFHIR: LLM-based mobile EHR querying. JACC Advances. 2025.
33. Advancing responsible healthcare AI with longitudinal EHR. Stanford HAI. 2024.

### Standards and Frameworks
34. FHIR R4 specification. HL7 International. 2019.
35. SNOMED CT. SNOMED International.
36. Rajagopal MR, Joranson DE. India: opioid availability. J Pain Symptom Manage. 2007.
37. Palat G, Venkateswaran C. Progress in palliative care in Kerala. Indian J Palliat Care. 2012.

---

## 6. Supplementary Material Plan

### S1: Voice Provider System Prompts
- Bolna system prompt (2800+ chars): compassionate palliative care instructions, RAG function calling, 5 critical rules
- Gemini Live context injection template
- Retell.AI custom LLM prompt

### S2: Medication Dosage Validation Database
- 150+ medications with safe ranges
- Key entries: morphine (oral 2.5-200mg), fentanyl (patch 12-100mcg/hr), paracetamol (max 4000mg/day), ondansetron (4-8mg/dose)
- Source: WHO Essential Medicines List, IAPC guidelines

### S3: Emergency Keyword Patterns
- Full keyword lists in 5 languages (English, Hindi, Bengali, Tamil, Gujarati)
- CRITICAL, HIGH, MEDIUM severity classifications
- Response action protocols per severity

### S4: FHIR Resource Mapping Tables
- SNOMED-CT symptom codes (22 mapped symptoms)
- LOINC vital sign codes
- ICD-10 condition codes
- RxNorm medication codes
- Severity scale mapping (0-4 to SNOMED severity codes)

### S5: GraphRAG Entity Extraction Prompts
- Custom palliative care entity extraction prompt
- Community report generation prompt
- Relationship extraction categories

### S6: Clinical Test Scenario Scripts
- Full test scripts for 4 scenarios
- Expected outcomes and pass criteria
- Code excerpts from test_realistic_clinical_scenarios.py
