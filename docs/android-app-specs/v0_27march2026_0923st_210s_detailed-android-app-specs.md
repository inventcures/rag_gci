# Palli Sahayak Android App: Detailed Technical Specification

**Version**: 0.1.0
**Date**: 27 March 2026
**Authors**: Ashish Makani, Dr. Anurag Agrawal (PI), KCDH-A Team
**Status**: Draft — Pre-Implementation Spec
**Companion Document**: `v50_20260327_0947ist_210s__changes-to-existing-rag_gci-codebase_detailed-specs.md` (backend changes)
**Informing Document**: `v4_proposal_draft.pdf` (EVAH Pathway A Proposal)
**Reference**: [DeepWiki: rag_gci Architecture](https://deepwiki.com/inventcures/rag_gci)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Research Synthesis: Lessons from HCW Mobile Apps](#2-research-synthesis)
3. [Target Users, Sites, and Constraints](#3-target-users-sites-and-constraints)
4. [Technology Stack](#4-technology-stack)
5. [Architecture Overview](#5-architecture-overview)
6. [Module Structure](#6-module-structure)
7. [Data Layer](#7-data-layer)
8. [Domain Layer](#8-domain-layer)
9. [UI Layer](#9-ui-layer)
10. [Voice Integration](#10-voice-integration)
11. [Offline-First Architecture](#11-offline-first-architecture)
12. [Security and Privacy](#12-security-and-privacy)
13. [FHIR and ABDM Integration](#13-fhir-and-abdm-integration)
14. [Evaluation Instrumentation](#14-evaluation-instrumentation)
15. [Build, CI, and Release](#15-build-ci-and-release)
16. [Phased Implementation Plan](#16-phased-implementation-plan)
17. [CLAUDE.md for Android Project](#17-claudemd-for-android-project)
18. [Risk Mitigation](#18-risk-mitigation)
19. [Appendices](#19-appendices)

---

## 1. Executive Summary

### 1.1 Purpose

This specification defines a native Android application for **Palli Sahayak** ("Companion in Care") -- a voice-first AI clinical decision support tool that provides evidence-based palliative care guidance to frontline health workers, family caregivers, and patients in India.

The app serves as the primary mobile interface for the **EVAH Pathway A evaluation** -- a 12-month, $1M mixed-methods implementation science study across four Indian clinical sites with approximately 200 frontline users. The evaluation measures usability (SUS), adoption (calls/ASHA/week), safety (hallucination rate, emergency F1), and workflow integration (time-motion, disruption score).

### 1.2 Scope

The Android app:
- Provides **voice-first clinical query** with hands-free operation during ASHA home visits
- Works **offline** in rural/remote areas (CCF Coimbatore, CCHRC Silchar) with local caching and background sync
- Supports **9 Indian languages** (Tamil, Telugu, Kannada, Tulu, Malayalam, Bengali, Assamese, Hindi, English)
- Collects **evaluation data** (SUS scores, time-motion, clinical vignette crossover, interaction logs) for the EVAH study
- Integrates with **ABDM/ABHA** for India's digital health ecosystem via FHIR R4
- Manages **medication reminders**, **patient records**, and **care team coordination**

### 1.3 What This App Is Not

- Not a replacement for the existing telephony (Bolna), WhatsApp (Twilio), or web (Gradio) channels -- it is an additional client surface
- Not a standalone system -- it is a mobile client for the existing FastAPI backend (~30,000 lines Python)
- No clinical logic in the app -- all RAG pipeline, clinical validation, and evidence generation runs server-side
- No software development is proposed under EVAH funds (per proposal: "all funds support evaluation and evidence generation") -- this spec is for pre-evaluation development under the existing GCI grant

### 1.4 Key Design Principles

| Principle | Rationale |
|-----------|-----------|
| Voice-first, not screen-first | ASHA workers use the tool during active home visits; hands must be free |
| Sub-13-second interactions | Simple App benchmark for CHW encounters at scale in India |
| Offline-first, sync-later | CCF and CCHRC serve rural/remote communities with intermittent connectivity |
| PIN-based auth, not passwords | Validated by mSakhi for low-literacy ASHA workers; device sharing is common |
| Evidence badges on every response | Five-pillar safety framework is non-negotiable |
| Evaluation-instrumented from day one | Every interaction generates data for RE-AIM analysis |

---

## 2. Research Synthesis: Lessons from HCW Mobile Apps

### 2.1 Reference Applications Studied

The following platforms were analyzed for architectural patterns, offline strategies, UI/UX for low-literacy users, and field deployment lessons:

#### Tier 1: Primary References (directly relevant)

**Simple App (simpledotorg)** -- The single best technical reference for Palli Sahayak's Android app.
- **Scale**: 7M+ patients managed across India, Bangladesh, Ethiopia, Sri Lanka
- **Stack**: Kotlin, Mobius (MVI) architecture, SQLite+NDK, WorkManager sync
- **Key metric**: 13-second per-encounter interaction time
- **Field lesson**: "Healthcare workers are too busy to enter detailed data" -- the fewer taps and screens, the higher the adoption
- **Offline**: Full offline-first with periodic sync; average sync interval is 15 minutes
- **Languages**: 16 translations, including Hindi, Tamil, Kannada, Bengali
- **Published evidence**: BMJ research on scale-up factors in Indian primary healthcare
- **Implication for Palli Sahayak**: Our voice-first approach has a natural advantage -- speaking is faster than typing. Target the same 13-second benchmark.

**OpenSRP FHIR Core** -- Best FHIR integration reference.
- **Stack**: Kotlin, Google Android FHIR SDK, Keycloak auth
- **FHIR**: On-device FHIR resource store with server sync
- **Min API**: 26 (Android 8.0), matching our target
- **Implication**: Use Google Android FHIR SDK for ABDM interoperability, not as primary store

**mSakhi / ReMiND** -- Most relevant India-specific precedent for ASHA workers.
- **Validation**: Proved that voice/audio guidance works for low-literacy ASHAs
- **Field lessons**: Device sharing is common (needs session-based auth), supervisor engagement drives adoption, local dialect audio matters, incentive alignment is critical
- **Implication**: PIN-based auth (not password), session timeout after 5 minutes, voice prompts in local dialect

#### Tier 2: Architecture References

**Dimagi CommCare** -- Gold standard for CHW apps globally (2,000+ projects, 80+ countries).
- **Offline**: OpenRosa/XForms with complete offline form submission
- **Media**: Embedded audio/video prompts in workflows for low-literacy users
- **Sync**: Heartbeat-based sync with conflict resolution
- **Implication**: Pre-recorded audio prompts for onboarding and training modules

**Medic (formerly Medic Mobile)** -- Community Health Toolkit (CHT).
- **Architecture**: Progressive web app approach with CouchDB/PouchDB offline sync
- **Task management**: Automated task scheduling for CHW follow-ups
- **SMS/voice**: Integrated SMS and voice alongside app for multi-channel reach
- **Implication**: Consider push notification-based task reminders for follow-up visits

**DHIS2 Android Capture App** -- WHO-recommended health information system.
- **Offline**: Complete offline data capture with granular sync control
- **Large datasets**: Pagination and lazy loading for resource-constrained devices
- **SDK**: Provides an Android SDK for custom app development
- **Implication**: Cursor-based pagination for observation history; lazy load older records

#### Tier 3: Evaluation & Data Collection References

**IDinsight** tools and SurveyCTO customizations.
- **Quality assurance**: Audio audit of enumerator interactions
- **GPS metadata**: Automatic location capture for field verification
- **Implication**: Record interaction timestamps and optional GPS for time-motion analysis

**ODK (Open Data Kit) / KoboToolbox** -- XForm-based data collection.
- **Offline**: Robust offline form submission with queue management
- **Media**: Audio, photo, and video capture within forms
- **Multi-language**: Dynamic language switching within forms
- **Implication**: Clinical vignette assessment module follows ODK patterns for structured data capture

**REDCap Mobile** -- Clinical research data capture.
- **Integration**: Tight integration with research workflows and statistical analysis
- **Export**: Direct export to R, SAS, SPSS formats
- **Implication**: Evaluation data export as CSV compatible with R/lme4 for multi-level modeling

### 2.2 Cross-Cutting Design Patterns Synthesized

| Pattern | Best Practice (from research) | Palli Sahayak Implementation |
|---------|-------------------------------|------------------------------|
| Offline-first | Room/SQLite + WorkManager (Simple App) | Room + SQLCipher + WorkManager with 15-min periodic sync |
| Low-literacy UI | Large buttons (56dp+), icon-based nav, audio feedback (CommCare, mSakhi) | 96dp voice button, evidence badge icons, TTS auto-play |
| Voice/audio | Pre-recorded prompts + live STT/TTS (mSakhi) | Sarvam AI STT/TTS (22 languages) + on-device Android SpeechRecognizer fallback |
| Device compatibility | Min SDK 26, target 2GB RAM devices (Simple App, OpenSRP) | Min SDK 26, target SDK 35, tested on Redmi 10A / Samsung A03 |
| Data security | SQLCipher encryption, AndroidKeyStore (OpenSRP) | SQLCipher + EncryptedSharedPreferences + certificate pinning |
| Auth for shared devices | PIN/biometric per-session (mSakhi field finding) | 4-digit PIN with Argon2id hash, 5-minute auto-lock |
| Sync conflict resolution | Server-wins for clinical data (Simple App) | Server-wins for mutable records, append-only for observations |
| FHIR interop | Google Android FHIR SDK (OpenSRP) | FHIR SDK for export/import, Room as primary store |
| Evaluation data | Audio audit + GPS (IDinsight/SurveyCTO) | Timestamped interaction logs, optional GPS, SUS questionnaire |
| APK size | Under 25MB for 2G download (field constraint) | Target 20MB with dynamic feature delivery for FHIR module |

### 2.3 India-Specific Context

**ASHA Worker Reality**:
- 1M+ ASHAs across India; exclusively women; basic smartphone access ($100-200 devices)
- Typical devices: Redmi 10A (2GB RAM), Samsung Galaxy A03 (3GB RAM), Jio phones
- Network: 4G in urban/peri-urban, 2G/3G in rural, intermittent/absent in remote (Barak Valley)
- Device sharing is common -- ASHAs share phones with family members
- Digital literacy varies widely -- baseline assessed as covariate in EVAH study

**ABDM/ABHA Integration**:
- Ayushman Bharat Digital Mission (ABDM) mandates ABHA health IDs (14-digit)
- FHIR R4 is the interoperability standard
- NHA provides Health Facility Registry (HFR) and Health Professional Registry (HPR)
- Palli Sahayak's existing FHIR adapter (`personalization/fhir_adapter.py`) maps to ABDM-compatible resources

**Language Landscape at Evaluation Sites**:

| Site | State | Languages | Language Family |
|------|-------|-----------|----------------|
| CMC Vellore | Tamil Nadu | Tamil, Telugu, English | Dravidian |
| KMC Manipal (MAHE) | Karnataka | Kannada, Tulu, English | Dravidian |
| CCF Coimbatore | Tamil Nadu | Tamil, Malayalam, English | Dravidian |
| CCHRC Silchar | Assam | Bengali, Assamese, Hindi | Indo-Aryan |

**Sarvam AI Language Support**:
- STT (Saaras v3): All 22 constitutionally scheduled languages
- TTS (Bulbul v3): 11 languages with 30+ voices
- For STT-only languages (Assamese), TTS falls back to Hindi
- All processing on Indian sovereign compute infrastructure

**DPDP Act 2023 Compliance**:
- Explicit informed consent required before data collection
- Right to erasure must be implemented
- Data localization: processing on Indian servers (Sarvam AI)
- No persistent voice recordings stored (per EVAH proposal section 10.1)

---

## 3. Target Users, Sites, and Constraints

### 3.1 User Groups

#### ASHA Workers (Primary Users)
- **Count**: ~80 across 4 sites (50/site target, 60 recruited with 20% attrition buffer)
- **Context**: Conduct twice-weekly home visits to palliative care patients
- **Literacy**: Variable; baseline digital literacy assessed as covariate
- **Device**: Basic Android smartphone ($100-200), often shared with family
- **Network**: Varies from 4G (CMC Vellore) to intermittent 2G (CCHRC Silchar)
- **Primary need**: Hands-free voice query during home visits for symptom management guidance
- **Interaction pattern**: 2-5 queries per home visit, 4-8 visits per week

#### Family Caregivers (Secondary Users)
- **Count**: ~80 across 4 sites
- **Context**: Provide 24/7 care between ASHA visits; trained and supervised by ASHAs
- **Literacy**: Moderate; more comfortable with text input than ASHAs
- **Device**: Personal smartphone, moderate quality
- **Network**: Home-based, more consistent connectivity
- **Primary need**: Medication reminders, symptom tracking, basic clinical guidance between visits

#### Patients (Tertiary Users)
- **Count**: ~40 across 4 sites (stable condition, willing to engage)
- **Context**: Use tool for self-management between visits
- **Literacy**: Variable; voice-first is essential
- **Device**: Often shared with caregiver
- **Primary need**: Medication reminders, basic symptom guidance in native language

### 3.2 Device Constraints

| Constraint | Specification |
|------------|---------------|
| Min Android version | API 26 (Android 8.0 Oreo) |
| Target Android version | API 35 (Android 15) |
| RAM | 2GB minimum, 3GB recommended |
| Storage | 32GB minimum (app target: 50MB installed with cache) |
| Screen | 5.0" minimum, 720p |
| Network | Must work offline; sync when connected (2G minimum for sync) |
| Battery | Full workday (8 hours) with moderate use |
| Camera | Not required for core features |
| GPS | Optional (for time-motion metadata) |
| Microphone | Required (voice-first design) |
| Speaker | Required (TTS playback) |

### 3.3 Reference Devices for Testing

| Device | RAM | Storage | Android | Price | Site Relevance |
|--------|-----|---------|---------|-------|---------------|
| Redmi 10A | 2GB | 32GB | 12 | $100 | CCF, CCHRC |
| Samsung Galaxy A03 | 3GB | 32GB | 12 | $120 | All sites |
| Redmi Note 12 | 4GB | 64GB | 13 | $160 | CMC, KMC |
| Samsung Galaxy A14 | 4GB | 64GB | 13 | $180 | CMC, KMC |

---

## 4. Technology Stack

### 4.1 Non-Negotiable Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Language | Kotlin 2.0+ (K2 compiler) | Modern, null-safe, coroutine-native |
| UI Framework | Jetpack Compose (Material3) | Declarative, preview-able, no XML |
| Architecture | MVVM + UDF | ViewModel -> StateFlow -> Composable |
| DI | Hilt (Dagger) | Compile-time DI, first-class Compose support |
| Local DB | Room + SQLCipher | Type-safe ORM with encryption at rest |
| Async | Kotlin Coroutines + Flow | Structured concurrency, reactive streams |
| Network | Retrofit + OkHttp | Type-safe HTTP, interceptors, certificate pinning |
| Images | Coil | Kotlin-first, Compose integration, disk cache |
| Background | WorkManager | Battery-aware, survives process death |
| Navigation | Navigation Compose | Type-safe, deep link support |
| Build | Gradle Kotlin DSL + Version Catalog | Reproducible builds, centralized versions |

### 4.2 Additional Libraries

| Purpose | Library | Version |
|---------|---------|---------|
| FHIR | Google Android FHIR SDK (Engine + Data Capture) | 1.0.0 |
| Encryption | SQLCipher (`net.zetetic:android-database-sqlcipher`) | 4.5.6 |
| Secure Storage | `androidx.security:security-crypto` | 1.1.0-alpha06 |
| Biometrics | `androidx.biometric:biometric` | 1.2.0-alpha05 |
| JSON | Moshi | 1.15.1 |
| Audio Recording | AudioRecord (Android SDK) | N/A |
| Audio Playback | AudioTrack / MediaPlayer (Android SDK) | N/A |
| On-Device STT | Android SpeechRecognizer | N/A |
| On-Device TTS | Android TextToSpeech | N/A |
| WebSocket | OkHttp WebSocket | 4.12.0 |
| Testing: Unit | JUnit 4 + MockK + Turbine | Latest |
| Testing: UI | Compose UI Test | Latest (from BOM) |
| Testing: E2E | Maestro | Latest |
| Crash Reporting | Firebase Crashlytics | Latest |
| Analytics | Firebase Analytics (evaluation only) | Latest |

### 4.3 What We Explicitly Do NOT Use

| Excluded | Reason |
|----------|--------|
| Java | Kotlin-only codebase |
| XML layouts | Jetpack Compose only |
| RxJava | Kotlin Coroutines + Flow |
| LiveData | StateFlow (cold by default, lifecycle-aware collection in Compose) |
| Dagger (raw) | Hilt (Dagger wrapper with reduced boilerplate) |
| Glide / Picasso | Coil (Kotlin-first) |
| SharedPreferences (raw) | EncryptedSharedPreferences or DataStore |
| Firebase Realtime DB | Room + custom sync (our backend is not Firebase) |
| Flutter / React Native | Native Kotlin for audio performance on budget devices |

---

## 5. Architecture Overview

### 5.1 System Context

```
                    ┌──────────────────────────────────┐
                    │       Palli Sahayak Backend       │
                    │         (FastAPI + Python)        │
                    │                                   │
                    │  ┌─────────┐ ┌────────────────┐  │
                    │  │  RAG    │ │  Safety &       │  │
                    │  │ Pipeline│ │  Clinical       │  │
                    │  │         │ │  Validation     │  │
                    │  └─────────┘ └────────────────┘  │
                    │  ┌─────────┐ ┌────────────────┐  │
                    │  │ Sarvam  │ │ Longitudinal   │  │
                    │  │ STT/TTS │ │ Memory + FHIR  │  │
                    │  └─────────┘ └────────────────┘  │
                    │  ┌─────────┐ ┌────────────────┐  │
                    │  │Knowledge│ │ Mobile API      │  │
                    │  │ Graph   │ │ (NEW - v1)      │  │
                    │  └─────────┘ └────────────────┘  │
                    └───────────────┬──────────────────┘
                                   │ HTTPS / WSS
                    ┌──────────────┴──────────────────┐
                    │      Android App (This Spec)     │
                    │                                   │
                    │  ┌─────────┐ ┌────────────────┐  │
                    │  │  Voice  │ │  Offline Cache  │  │
                    │  │ Engine  │ │  + Sync Engine  │  │
                    │  └─────────┘ └────────────────┘  │
                    │  ┌─────────┐ ┌────────────────┐  │
                    │  │ Room DB │ │  Evaluation     │  │
                    │  │(SQLCipher│ │  Instrumentation│  │
                    │  └─────────┘ └────────────────┘  │
                    │  ┌─────────────────────────────┐  │
                    │  │   Jetpack Compose UI Layer   │  │
                    │  │  (Voice Query, Dashboard,    │  │
                    │  │   Medication, Patient, etc.)  │  │
                    │  └─────────────────────────────┘  │
                    └──────────────────────────────────┘
```

### 5.2 Data Flow: Voice Query (Primary Interaction)

```
User taps microphone
        │
        ▼
AudioRecorder captures PCM (16kHz, 16-bit, mono)
        │
        ├── [ONLINE] ──► POST /api/mobile/v1/query/voice
        │                     │
        │                     ▼
        │               Sarvam STT (Saaras v3)
        │                     │
        │                     ▼
        │               SimpleRAGPipeline.query()
        │                 ├── ChromaDB vector search
        │                 ├── Neo4j graph traversal
        │                 └── GraphRAG community search
        │                     │
        │                     ▼
        │               ClinicalValidator.validate()
        │                     │
        │                     ▼
        │               SafetyEnhancementsManager
        │                 ├── Evidence badge
        │                 ├── Emergency detection
        │                 └── Expert sampling
        │                     │
        │                     ▼
        │               Response JSON + TTS audio (base64)
        │                     │
        │                     ▼
        │               App displays response + plays audio
        │
        └── [OFFLINE] ──► Android SpeechRecognizer
                              │
                              ▼
                         Local keyword matching
                              │
                              ├── [Emergency] ──► Show emergency banner + 108/112 dialer
                              │
                              └── [Normal] ──► Match against QueryCacheEntity (top 20)
                                                  │
                                                  ├── [Cache hit] ──► Display cached response
                                                  │
                                                  └── [Cache miss] ──► "Connect to network for
                                                                        this query" message
```

### 5.3 Architectural Decisions Record

| ID | Decision | Rationale | Alternatives Considered |
|----|----------|-----------|------------------------|
| AD-1 | Native Kotlin over cross-platform | Direct AudioRecord/AudioTrack for low-latency voice; SpeechRecognizer fallback; AlarmManager reliability on budget devices | Flutter (audio unreliable on budget Android), React Native (bridge overhead) |
| AD-2 | Room + SQLCipher as primary store | Backend data model richer than FHIR (voice metadata, evaluation logs, vignettes); full query control; validated by Simple App at 7M patient scale | FHIR SDK as primary (too limited), CouchDB (no Kotlin support) |
| AD-3 | WorkManager for sync | Survives process death, respects Doze mode, battery-aware; Google's recommendation; validated by Simple App | Custom SyncAdapter (deprecated), AlarmManager (battery issues), JobScheduler (WorkManager wraps it) |
| AD-4 | Single Activity + Compose Navigation | Standard modern Android; reduced memory on budget devices; simplified deep linking | Multi-Activity (higher memory, fragment overhead) |
| AD-5 | PIN auth, not password | mSakhi field validation with ASHA workers; device sharing reality; low-literacy users | Password (too complex), no auth (unsafe for PHI), biometric-only (not all devices support) |
| AD-6 | Server-side clinical logic only | Safety-critical: RAG pipeline, clinical validation, evidence badges run server-side; app is a presentation layer | On-device LLM (too large for 2GB RAM), local RAG (storage/performance constraints) |
| AD-7 | Pre-cached top 20 queries | EVAH proposal section 10.3 mandates offline functionality; 20 queries cover ~60% of ASHA usage based on pilot data | Full offline RAG (impossible on 2GB), no offline (unacceptable for CCHRC) |

---

## 6. Module Structure

### 6.1 Multi-Module Gradle Project

```
palli-sahayak-android/
│
├── app/                                    # Application composition root
│   ├── build.gradle.kts
│   └── src/main/kotlin/.../
│       ├── PalliSahayakApplication.kt      # Hilt @HiltAndroidApp
│       ├── MainActivity.kt                 # Single Activity host
│       └── navigation/
│           └── PalliSahayakNavGraph.kt     # Root NavHost
│
├── core/
│   ├── core-common/                        # Shared utilities
│   │   └── src/main/kotlin/.../
│   │       ├── result/Result.kt            # Sealed class: Success | Error | Loading
│   │       ├── dispatchers/DispatcherProvider.kt
│   │       ├── extensions/                 # Kotlin extension functions
│   │       └── constants/AppConstants.kt
│   │
│   ├── core-model/                         # Domain models (no framework deps)
│   │   └── src/main/kotlin/.../
│   │       ├── user/UserRole.kt, UserProfile.kt
│   │       ├── patient/Patient.kt, Observation.kt, SeverityLevel.kt
│   │       ├── medication/MedicationReminder.kt, AdherenceStats.kt
│   │       ├── query/QueryResult.kt, EvidenceLevel.kt, EmergencyLevel.kt
│   │       ├── careteam/CareTeamMember.kt
│   │       ├── evaluation/SusScore.kt, VignetteResponse.kt, InteractionEvent.kt
│   │       ├── sync/SyncStatus.kt, SyncResult.kt
│   │       └── fhir/FhirBundle.kt, AbhaId.kt
│   │
│   ├── core-data/                          # Room DB, DataStore, repositories
│   │   └── src/main/kotlin/.../
│   │       ├── database/
│   │       │   ├── PalliSahayakDatabase.kt # Room DB with SQLCipher
│   │       │   ├── entity/                 # Room entities (8 tables)
│   │       │   ├── dao/                    # Room DAOs
│   │       │   └── converter/              # TypeConverters
│   │       ├── datastore/
│   │       │   └── UserPreferencesDataStore.kt
│   │       ├── repository/                 # Repository implementations
│   │       │   ├── QueryRepository.kt
│   │       │   ├── PatientRepository.kt
│   │       │   ├── MedicationRepository.kt
│   │       │   ├── CareTeamRepository.kt
│   │       │   ├── EvaluationRepository.kt
│   │       │   └── UserRepository.kt
│   │       └── network/NetworkMonitor.kt
│   │
│   ├── core-network/                       # Retrofit services, OkHttp
│   │   └── src/main/kotlin/.../
│   │       ├── api/
│   │       │   └── PalliSahayakApiService.kt  # All backend endpoint mappings
│   │       ├── interceptor/
│   │       │   ├── AuthInterceptor.kt         # JWT token injection
│   │       │   └── ConnectivityInterceptor.kt # Offline detection
│   │       ├── dto/                            # Network DTOs (request/response)
│   │       │   ├── QueryRequest.kt, QueryResponse.kt
│   │       │   ├── SyncPushRequest.kt, SyncPullResponse.kt
│   │       │   ├── AuthRequest.kt, AuthResponse.kt
│   │       │   └── ... (one per endpoint group)
│   │       ├── mapper/                         # DTO <-> Domain model mappers
│   │       └── di/NetworkModule.kt             # Hilt module
│   │
│   ├── core-voice/                         # Voice engine abstraction
│   │   └── src/main/kotlin/.../
│   │       ├── VoiceEngine.kt              # Interface
│   │       ├── ServerVoiceEngine.kt        # Sarvam STT/TTS via API
│   │       ├── OnDeviceVoiceEngine.kt      # Android SpeechRecognizer/TTS
│   │       ├── VoiceWebSocketManager.kt    # Gemini Live streaming
│   │       ├── AudioRecorder.kt            # PCM capture (16kHz, mono)
│   │       ├── AudioPlayer.kt              # Playback with waveform data
│   │       ├── EmergencyKeywordDetector.kt # Local keyword matching (5 langs)
│   │       ├── LanguageMapper.kt           # Short code -> BCP-47 mapping
│   │       └── di/VoiceModule.kt
│   │
│   ├── core-security/                      # Auth, encryption, compliance
│   │   └── src/main/kotlin/.../
│   │       ├── PinManager.kt              # Argon2id PIN hashing
│   │       ├── BiometricHelper.kt         # BiometricPrompt wrapper
│   │       ├── EncryptionHelper.kt        # SQLCipher key management
│   │       ├── TokenManager.kt            # JWT storage in EncryptedSharedPrefs
│   │       ├── SessionManager.kt          # Auto-lock after 5 min
│   │       └── di/SecurityModule.kt
│   │
│   ├── core-sync/                          # WorkManager sync engine
│   │   └── src/main/kotlin/.../
│   │       ├── SyncManager.kt             # Orchestrates all sync workers
│   │       ├── SyncWorker.kt              # Base sync worker
│   │       ├── ObservationSyncWorker.kt   # Sync observations
│   │       ├── MedicationSyncWorker.kt    # Sync medication reminders
│   │       ├── PatientSyncWorker.kt       # Sync patient records
│   │       ├── EvaluationSyncWorker.kt    # Sync evaluation data
│   │       ├── CacheBundleSyncWorker.kt   # Download offline cache bundle
│   │       ├── ConflictResolver.kt        # Server-wins resolution
│   │       └── di/SyncModule.kt
│   │
│   ├── core-fhir/                          # FHIR SDK integration
│   │   └── src/main/kotlin/.../
│   │       ├── FhirEngineSetup.kt         # Initialize FHIR engine
│   │       ├── FhirResourceMapper.kt      # Room entities <-> FHIR resources
│   │       ├── FhirExporter.kt            # Export patient as FHIR Bundle
│   │       ├── FhirImporter.kt            # Import FHIR Bundle
│   │       ├── AbhaIdValidator.kt         # 14-digit ABHA validation
│   │       ├── SnomedCodes.kt             # Palliative care SNOMED CT mappings
│   │       └── di/FhirModule.kt
│   │
│   ├── core-evaluation/                    # EVAH study instrumentation
│   │   └── src/main/kotlin/.../
│   │       ├── InteractionLogger.kt       # Structured event logging
│   │       ├── TimeMotionTracker.kt       # Automatic timing capture
│   │       ├── SusQuestionnaire.kt        # 10-item SUS in 9 languages
│   │       ├── EvaluationDataExporter.kt  # CSV export for R/lme4
│   │       └── di/EvaluationModule.kt
│   │
│   └── core-ui/                            # Design system
│       └── src/main/kotlin/.../
│           ├── theme/
│           │   ├── PalliSahayakTheme.kt
│           │   ├── Color.kt
│           │   ├── Typography.kt
│           │   └── Shape.kt
│           └── component/
│               ├── VoiceButton.kt          # 96dp FAB with waveform
│               ├── EvidenceBadge.kt        # Color-coded confidence badge
│               ├── EmergencyBanner.kt      # Red banner with 108 call
│               ├── SyncStatusChip.kt       # Online/offline indicator
│               ├── LanguagePicker.kt       # 9-language selector
│               ├── PatientCard.kt          # Patient summary card
│               ├── ObservationRow.kt       # Timeline entry
│               ├── MedicationReminderCard.kt
│               ├── LikertScale.kt          # 1-5 scale for SUS/vignettes
│               └── LoadingOverlay.kt       # Shimmer/skeleton loading
│
├── feature/
│   ├── feature-onboarding/                 # First-launch setup
│   │   └── src/main/kotlin/.../
│   │       ├── ui/
│   │       │   ├── LanguageSelectScreen.kt
│   │       │   ├── RoleSelectScreen.kt
│   │       │   ├── AbhaLinkScreen.kt
│   │       │   ├── PinSetupScreen.kt
│   │       │   ├── ConsentScreen.kt        # DPDP Act consent
│   │       │   └── TrainingScreen.kt       # Audio-guided tutorial
│   │       ├── OnboardingViewModel.kt
│   │       └── navigation/OnboardingNavGraph.kt
│   │
│   ├── feature-home/                       # Role-specific dashboards
│   │   └── src/main/kotlin/.../
│   │       ├── ui/
│   │       │   ├── AshaDashboardScreen.kt
│   │       │   ├── CaregiverDashboardScreen.kt
│   │       │   └── PatientDashboardScreen.kt
│   │       ├── HomeViewModel.kt
│   │       └── navigation/HomeNavGraph.kt
│   │
│   ├── feature-query/                      # Voice/text clinical query
│   │   └── src/main/kotlin/.../
│   │       ├── ui/
│   │       │   ├── VoiceQueryScreen.kt     # PRIMARY SCREEN
│   │       │   ├── TextQueryScreen.kt      # Fallback for noisy env
│   │       │   └── QueryResultCard.kt      # Response display
│   │       ├── domain/
│   │       │   ├── SubmitVoiceQueryUseCase.kt
│   │       │   ├── CheckEmergencyUseCase.kt
│   │       │   └── GetOfflineCacheUseCase.kt
│   │       ├── QueryViewModel.kt
│   │       └── navigation/QueryNavGraph.kt
│   │
│   ├── feature-medication/                 # Reminders + adherence
│   │   └── src/main/kotlin/.../
│   │       ├── ui/
│   │       │   ├── ReminderListScreen.kt
│   │       │   ├── CreateReminderScreen.kt
│   │       │   └── AdherenceDashboardScreen.kt
│   │       ├── domain/
│   │       │   └── CreateMedicationReminderUseCase.kt
│   │       ├── worker/MedicationAlarmReceiver.kt
│   │       ├── MedicationViewModel.kt
│   │       └── navigation/MedicationNavGraph.kt
│   │
│   ├── feature-patient/                    # Patient records + timeline
│   │   └── src/main/kotlin/.../
│   │       ├── ui/
│   │       │   ├── PatientListScreen.kt
│   │       │   ├── PatientDetailScreen.kt
│   │       │   ├── ObservationTimelineScreen.kt
│   │       │   └── RecordObservationScreen.kt
│   │       ├── domain/
│   │       │   └── RecordObservationUseCase.kt
│   │       ├── PatientViewModel.kt
│   │       └── navigation/PatientNavGraph.kt
│   │
│   ├── feature-careteam/                   # Care team management
│   │   └── src/main/kotlin/.../
│   │       ├── ui/
│   │       │   ├── CareTeamScreen.kt
│   │       │   └── AddMemberScreen.kt
│   │       ├── CareTeamViewModel.kt
│   │       └── navigation/CareTeamNavGraph.kt
│   │
│   ├── feature-vignette/                   # Clinical vignette assessment
│   │   └── src/main/kotlin/.../
│   │       ├── ui/
│   │       │   ├── VignetteListScreen.kt
│   │       │   ├── VignetteDetailScreen.kt
│   │       │   └── VignetteResponseScreen.kt
│   │       ├── domain/
│   │       │   └── SubmitVignetteResponseUseCase.kt
│   │       ├── VignetteViewModel.kt
│   │       └── navigation/VignetteNavGraph.kt
│   │
│   └── feature-settings/                   # App settings + data export
│       └── src/main/kotlin/.../
│           ├── ui/
│           │   ├── SettingsScreen.kt
│           │   ├── LanguageSettingsScreen.kt
│           │   ├── DataExportScreen.kt
│           │   ├── SusScreen.kt            # SUS questionnaire
│           │   └── AboutScreen.kt
│           ├── domain/
│           │   ├── CollectSusScoreUseCase.kt
│           │   └── ExportFhirBundleUseCase.kt
│           ├── SettingsViewModel.kt
│           └── navigation/SettingsNavGraph.kt
│
├── gradle/
│   └── libs.versions.toml                  # Centralized version catalog
│
├── build.gradle.kts                        # Root build config
├── settings.gradle.kts                     # Module declarations
├── CLAUDE.md                               # AI coding assistant rules
└── README.md
```

### 6.2 Module Dependency Graph

```
app ──► feature-* (all feature modules)
feature-* ──► core-model, core-data, core-ui, core-common
feature-query ──► core-voice, core-evaluation
feature-medication ──► core-sync
feature-patient ──► core-sync, core-fhir
feature-vignette ──► core-evaluation
feature-settings ──► core-fhir, core-evaluation, core-security
feature-onboarding ──► core-security, core-fhir

core-data ──► core-model, core-network, core-common
core-network ──► core-model, core-security, core-common
core-voice ──► core-network, core-model, core-common
core-sync ──► core-data, core-network, core-common
core-fhir ──► core-model, core-data, core-common
core-evaluation ──► core-model, core-data, core-common
core-security ──► core-common
core-ui ──► core-model, core-common
```

**Rule**: No circular dependencies. Feature modules never depend on each other. Core modules may depend on other core modules but not on feature modules.

---

## 7. Data Layer

### 7.1 Room Database

#### Database Definition

```kotlin
@Database(
    entities = [
        UserEntity::class,
        PatientEntity::class,
        ObservationEntity::class,
        MedicationReminderEntity::class,
        CareTeamMemberEntity::class,
        QueryCacheEntity::class,
        InteractionLogEntity::class,
        VignetteResponseEntity::class,
    ],
    version = 1,
    exportSchema = true
)
@TypeConverters(Converters::class)
abstract class PalliSahayakDatabase : RoomDatabase() {
    abstract fun userDao(): UserDao
    abstract fun patientDao(): PatientDao
    abstract fun observationDao(): ObservationDao
    abstract fun medicationReminderDao(): MedicationReminderDao
    abstract fun careTeamMemberDao(): CareTeamMemberDao
    abstract fun queryCacheDao(): QueryCacheDao
    abstract fun interactionLogDao(): InteractionLogDao
    abstract fun vignetteResponseDao(): VignetteResponseDao
}
```

Encrypted via SQLCipher:
```kotlin
val passphrase = EncryptionHelper.getDatabaseKey(context)
val factory = SupportFactory(passphrase)
Room.databaseBuilder(context, PalliSahayakDatabase::class.java, "palli_sahayak.db")
    .openHelperFactory(factory)
    .build()
```

### 7.2 Entity Definitions

#### UserEntity
Maps to backend `personalization/user_profile.py` UserProfile class.

```kotlin
@Entity(tableName = "users")
data class UserEntity(
    @PrimaryKey val userId: String,
    val name: String,
    val phoneHash: String,                    // SHA-256 hashed phone
    val role: String,                          // asha_worker, caregiver, patient
    val language: String,                      // BCP-47: ta-IN, kn-IN, etc.
    val communicationStyle: String,            // simple, detailed, clinical, empathetic
    val voiceSpeed: String,                    // slow, normal, fast
    val siteId: String,                        // cmc_vellore, kmc_manipal, ccf_coimbatore, cchrc_silchar
    val abhaId: String?,                       // 14-digit ABHA health ID
    val digitalLiteracyScore: Int?,            // Baseline covariate (0-100)
    val createdAt: Long,
    val updatedAt: Long,
    val lastSyncAt: Long,
    val syncStatus: String                     // pending, synced, conflict
)
```

#### PatientEntity
Maps to `LongitudinalPatientRecord` in `personalization/longitudinal_memory.py`.

```kotlin
@Entity(tableName = "patients")
data class PatientEntity(
    @PrimaryKey val patientId: String,
    val name: String,
    val primaryCondition: String?,             // cancer type, COPD, etc.
    val conditionStage: String?,               // stage I-IV, early/advanced
    val careLocation: String?,                 // home, hospice, hospital
    val assignedUserId: String?,               // ASHA worker or caregiver
    val createdAt: Long,
    val updatedAt: Long,
    val lastSyncAt: Long,
    val syncStatus: String
)
```

#### ObservationEntity
Maps to `TimestampedObservation` and subclasses (`SymptomObservation`, `MedicationObservation`, `VitalSignObservation`, `FunctionalStatusObservation`).

```kotlin
@Entity(
    tableName = "observations",
    foreignKeys = [ForeignKey(
        entity = PatientEntity::class,
        parentColumns = ["patientId"],
        childColumns = ["patientId"],
        onDelete = ForeignKey.CASCADE
    )],
    indices = [
        Index("patientId"),
        Index("timestamp"),
        Index("syncStatus"),
        Index("category")
    ]
)
data class ObservationEntity(
    @PrimaryKey val observationId: String,
    val patientId: String,
    val timestamp: Long,
    val sourceType: String,                    // voice_call, app, caregiver_report, clinical_entry
    val reportedBy: String,                    // patient, caregiver, system, provider
    val category: String,                      // symptom, medication, vital_sign, functional_status, emotional
    val entityName: String,                    // pain, morphine, blood_pressure, etc.
    val value: String?,                        // severity level or numeric value
    val valueText: String?,                    // human-readable description
    val severity: Int?,                        // 0-4 (NONE to VERY_SEVERE)
    val location: String?,                     // body location for symptoms
    val duration: String?,                     // e.g., "3 days", "since morning"
    val metadata: String?,                     // JSON for extra fields
    val createdAt: Long,
    val syncStatus: String
)
```

#### MedicationReminderEntity
Maps to `MedicationVoiceReminder` in `medication_voice_reminders.py`.

```kotlin
@Entity(
    tableName = "medication_reminders",
    foreignKeys = [ForeignKey(
        entity = PatientEntity::class,
        parentColumns = ["patientId"],
        childColumns = ["patientId"],
        onDelete = ForeignKey.CASCADE
    )],
    indices = [Index("patientId"), Index("scheduledTime"), Index("syncStatus")]
)
data class MedicationReminderEntity(
    @PrimaryKey val reminderId: String,
    val patientId: String,
    val medicationName: String,
    val dosage: String,
    val frequency: String?,                    // once_daily, twice_daily, etc.
    val scheduledTime: Long,                   // next scheduled reminder
    val language: String,
    val callStatus: String,                    // scheduled, pending, completed, missed, failed
    val callAttempts: Int,
    val patientConfirmed: Boolean,
    val confirmationMethod: String?,           // dtmf_1, voice_yes, missed_call
    val isActive: Boolean,
    val createdAt: Long,
    val updatedAt: Long,
    val lastSyncAt: Long,
    val syncStatus: String
)
```

#### CareTeamMemberEntity
Maps to `CareTeamMember` in `personalization/longitudinal_memory.py`.

```kotlin
@Entity(
    tableName = "care_team_members",
    foreignKeys = [ForeignKey(
        entity = PatientEntity::class,
        parentColumns = ["patientId"],
        childColumns = ["patientId"],
        onDelete = ForeignKey.CASCADE
    )],
    indices = [Index("patientId")]
)
data class CareTeamMemberEntity(
    @PrimaryKey val memberId: String,
    val patientId: String,
    val name: String,
    val role: String,                          // doctor, nurse, asha_worker, caregiver, volunteer, social_worker
    val organization: String?,
    val phoneNumber: String?,
    val primaryContact: Boolean,
    val createdAt: Long,
    val updatedAt: Long,
    val lastSyncAt: Long,
    val syncStatus: String
)
```

#### QueryCacheEntity
Offline cache for top 20 clinical queries.

```kotlin
@Entity(
    tableName = "query_cache",
    indices = [Index("language"), Index("expiresAt")]
)
data class QueryCacheEntity(
    @PrimaryKey val queryHash: String,         // SHA-256 of normalized query text
    val queryText: String,
    val language: String,
    val responseText: String,
    val responseJson: String,                  // Full response including sources, evidence
    val evidenceLevel: String,                 // A, B, C, D, E
    val sources: String,                       // JSON array of source citations
    val cachedAt: Long,
    val expiresAt: Long,                       // Cache TTL: 7 days default
    val accessCount: Int
)
```

#### InteractionLogEntity
Evaluation data for RE-AIM analysis.

```kotlin
@Entity(
    tableName = "interaction_logs",
    indices = [Index("userId"), Index("sessionId"), Index("timestamp"), Index("syncStatus")]
)
data class InteractionLogEntity(
    @PrimaryKey val logId: String,
    val userId: String,
    val sessionId: String,
    val eventType: String,                     // See Section 14.1 for full event taxonomy
    val eventData: String?,                    // JSON payload
    val timestamp: Long,
    val durationMs: Long?,                     // Event duration (for time-motion)
    val language: String,
    val siteId: String,
    val isOffline: Boolean,
    val syncStatus: String
)
```

#### VignetteResponseEntity
Clinical vignette crossover assessment (EVAH section 6.4).

```kotlin
@Entity(
    tableName = "vignette_responses",
    indices = [Index("userId"), Index("syncStatus")]
)
data class VignetteResponseEntity(
    @PrimaryKey val responseId: String,
    val userId: String,
    val vignetteId: String,                    // e.g., "V01" through "V20"
    val vignetteTitle: String,
    val withTool: Boolean,                     // true = with Palli Sahayak, false = without
    val responseText: String?,                 // Transcribed response
    val responseAudioPath: String?,            // Local encrypted audio file
    val startedAt: Long,
    val completedAt: Long,
    val durationMs: Long,                      // completedAt - startedAt
    val metadata: String?,                     // JSON: confidence, tool_queries_made, etc.
    val createdAt: Long,
    val syncStatus: String
)
```

### 7.3 Repository Pattern

All repositories follow the Single Source of Truth (SSOT) pattern:

```kotlin
interface PatientRepository {
    fun getPatients(): Flow<List<Patient>>
    fun getPatient(patientId: String): Flow<Patient?>
    fun getPatientWithObservations(patientId: String): Flow<PatientWithObservations?>
    suspend fun createPatient(patient: Patient): Result<Patient>
    suspend fun updatePatient(patient: Patient): Result<Patient>
    suspend fun syncPatients(): Result<SyncResult>
}

class PatientRepositoryImpl @Inject constructor(
    private val patientDao: PatientDao,
    private val observationDao: ObservationDao,
    private val apiService: PalliSahayakApiService,
    private val networkMonitor: NetworkMonitor,
) : PatientRepository {

    override fun getPatients(): Flow<List<Patient>> =
        patientDao.getAllPatients().map { entities -> entities.map { it.toDomain() } }

    override suspend fun createPatient(patient: Patient): Result<Patient> {
        val entity = patient.toEntity(syncStatus = SyncStatus.PENDING)
        patientDao.insert(entity)
        return Result.Success(patient)
    }

    override suspend fun syncPatients(): Result<SyncResult> {
        // 1. Push pending local changes
        val pending = patientDao.getPendingSync()
        if (pending.isNotEmpty()) {
            apiService.syncPush(SyncPushRequest(patients = pending.map { it.toDto() }))
            patientDao.updateSyncStatus(pending.map { it.patientId }, SyncStatus.SYNCED)
        }
        // 2. Pull server changes since last sync
        val lastSync = patientDao.getLastSyncTimestamp() ?: 0L
        val serverChanges = apiService.syncPull(lastSyncAt = lastSync)
        // 3. Resolve conflicts (server wins for mutable data)
        serverChanges.patients.forEach { dto ->
            patientDao.upsert(dto.toEntity(syncStatus = SyncStatus.SYNCED))
        }
        return Result.Success(SyncResult(pushed = pending.size, pulled = serverChanges.patients.size))
    }
}
```

---

## 8. Domain Layer

### 8.1 Domain Models

Pure Kotlin data classes in `core-model` with no framework dependencies:

```kotlin
// User roles matching backend personalization/user_profile.py UserRole enum
enum class UserRole { ASHA_WORKER, CAREGIVER, PATIENT }

// Severity matching backend personalization/longitudinal_memory.py SeverityLevel
enum class SeverityLevel(val value: Int) {
    NONE(0), MILD(1), MODERATE(2), SEVERE(3), VERY_SEVERE(4)
}

// Evidence levels matching backend safety_enhancements.py EvidenceBadge
enum class EvidenceLevel(val label: String, val recommendation: String) {
    A("Strong Evidence", "Based on WHO guidelines or meta-analyses"),
    B("Good Evidence", "Based on clinical studies"),
    C("Moderate Evidence", "Based on expert consensus"),
    D("Limited Evidence", "Based on case reports — consult physician"),
    E("Insufficient Evidence", "Please consult your physician")
}

// Emergency level for safety escalation
enum class EmergencyLevel { NONE, LOW, HIGH, CRITICAL }

// Data source types matching backend DataSourceType enum
enum class DataSourceType { VOICE_CALL, APP, CAREGIVER_REPORT, CLINICAL_ENTRY, PATIENT_REPORTED, FHIR_IMPORT }

// Observation categories
enum class ObservationCategory { SYMPTOM, MEDICATION, VITAL_SIGN, FUNCTIONAL_STATUS, EMOTIONAL }

// Sync status for offline-first tracking
enum class SyncStatus { PENDING, SYNCED, CONFLICT }

// Communication style matching backend UserPreferences
enum class CommunicationStyle { SIMPLE, DETAILED, CLINICAL, EMPATHETIC }

// Care team roles matching backend CareTeamMember
enum class CareTeamRole { DOCTOR, NURSE, ASHA_WORKER, CAREGIVER, VOLUNTEER, SOCIAL_WORKER }

// Medication reminder status matching backend CallStatus
enum class ReminderStatus { SCHEDULED, PENDING, CALLING, CONNECTED, COMPLETED, CONFIRMED, MISSED, FAILED, RETRYING }
```

### 8.2 Core Domain Objects

```kotlin
data class QueryResult(
    val answer: String,
    val sources: List<Source>,
    val evidenceLevel: EvidenceLevel,
    val emergencyLevel: EmergencyLevel,
    val confidence: Float,                     // 0.0 - 1.0
    val language: String,
    val validationStatus: String,              // passed, flagged, review_required
    val disclaimer: String?,
    val audioBase64: String?,                  // TTS audio for auto-play
)

data class Source(
    val document: String,
    val page: Int?,
    val relevanceScore: Float,
    val snippet: String?,
)

data class Patient(
    val patientId: String,
    val name: String,
    val primaryCondition: String?,
    val conditionStage: String?,
    val careLocation: String?,
    val assignedUserId: String?,
    val observations: List<Observation>,
    val careTeam: List<CareTeamMember>,
    val medicationReminders: List<MedicationReminder>,
)

data class Observation(
    val observationId: String,
    val patientId: String,
    val timestamp: Long,
    val sourceType: DataSourceType,
    val reportedBy: String,
    val category: ObservationCategory,
    val entityName: String,
    val severity: SeverityLevel?,
    val valueText: String?,
    val location: String?,
)

data class MedicationReminder(
    val reminderId: String,
    val patientId: String,
    val medicationName: String,
    val dosage: String,
    val scheduledTime: Long,
    val language: String,
    val status: ReminderStatus,
    val patientConfirmed: Boolean,
    val isActive: Boolean,
)
```

### 8.3 Use Cases

```kotlin
class SubmitVoiceQueryUseCase @Inject constructor(
    private val voiceEngine: VoiceEngine,
    private val queryRepository: QueryRepository,
    private val emergencyDetector: EmergencyKeywordDetector,
    private val interactionLogger: InteractionLogger,
    private val timeMotionTracker: TimeMotionTracker,
    private val networkMonitor: NetworkMonitor,
) {
    suspend operator fun invoke(audioData: ByteArray, language: String): Result<QueryResult> {
        val sessionEvent = timeMotionTracker.startEvent("voice_query")

        // Step 1: Local emergency check on transcript (if available)
        // Step 2: Online -> server query; Offline -> cache lookup
        // Step 3: Log interaction event
        // Step 4: Return result with timing data

        timeMotionTracker.endEvent(sessionEvent)
        return result
    }
}

class CheckEmergencyUseCase @Inject constructor(
    private val emergencyDetector: EmergencyKeywordDetector,
) {
    operator fun invoke(transcript: String, language: String): EmergencyLevel {
        return emergencyDetector.detect(transcript, language)
    }
}

class GetOfflineCacheUseCase @Inject constructor(
    private val queryCacheDao: QueryCacheDao,
) {
    suspend operator fun invoke(queryText: String, language: String): QueryCacheEntity? {
        val hash = queryText.normalize().sha256()
        return queryCacheDao.getByHash(hash, language)
            ?: queryCacheDao.findSimilar(queryText, language) // fuzzy match fallback
    }
}

class CreateMedicationReminderUseCase @Inject constructor(
    private val medicationRepository: MedicationRepository,
    private val alarmScheduler: AlarmScheduler,
    private val interactionLogger: InteractionLogger,
) {
    suspend operator fun invoke(reminder: MedicationReminder): Result<MedicationReminder> {
        // 1. Save to local DB (SyncStatus.PENDING)
        // 2. Schedule local AlarmManager as backup
        // 3. Log interaction event
        // 4. Return result (sync will happen in background)
    }
}

class RecordObservationUseCase @Inject constructor(
    private val patientRepository: PatientRepository,
    private val interactionLogger: InteractionLogger,
) {
    suspend operator fun invoke(observation: Observation): Result<Observation> {
        // 1. Save to local DB (SyncStatus.PENDING)
        // 2. Log interaction event
        // 3. Return result
    }
}

class ExportFhirBundleUseCase @Inject constructor(
    private val fhirExporter: FhirExporter,
    private val networkMonitor: NetworkMonitor,
) {
    suspend operator fun invoke(patientId: String): Result<FhirBundle> {
        // Online: fetch from server /api/mobile/v1/fhir/export/{patient_id}
        // Offline: generate from local Room data using FhirResourceMapper
    }
}

class CollectSusScoreUseCase @Inject constructor(
    private val evaluationRepository: EvaluationRepository,
    private val interactionLogger: InteractionLogger,
) {
    suspend operator fun invoke(scores: List<Int>, userId: String, siteId: String): Result<SusScore> {
        // Validate 10 items, each 1-5
        // Calculate SUS score using standard formula
        // Save to local DB + sync
    }
}

class SubmitVignetteResponseUseCase @Inject constructor(
    private val evaluationRepository: EvaluationRepository,
    private val interactionLogger: InteractionLogger,
) {
    suspend operator fun invoke(response: VignetteResponse): Result<VignetteResponse> {
        // Save response with timing data
        // Include whether tool was used (withTool flag)
        // Sync to server for blinded physician scoring
    }
}
```

---

## 9. UI Layer

### 9.1 Navigation Graph

```kotlin
@Composable
fun PalliSahayakNavGraph(navController: NavHostController) {
    NavHost(navController, startDestination = "splash") {

        composable("splash") { SplashScreen(navController) }

        // Onboarding flow (first launch only)
        navigation(startDestination = "language_select", route = "onboarding") {
            composable("language_select") { LanguageSelectScreen(navController) }
            composable("consent") { ConsentScreen(navController) }
            composable("role_select") { RoleSelectScreen(navController) }
            composable("abha_link") { AbhaLinkScreen(navController) }
            composable("pin_setup") { PinSetupScreen(navController) }
            composable("training") { TrainingScreen(navController) }
        }

        // PIN unlock (every session)
        composable("pin_unlock") { PinUnlockScreen(navController) }

        // Main app (post-auth)
        navigation(startDestination = "home", route = "main") {
            // Home/Dashboard
            composable("home") { HomeDashboardScreen(navController) }

            // Voice Query (primary interaction)
            composable("voice_query") { VoiceQueryScreen(navController) }
            composable("text_query") { TextQueryScreen(navController) }

            // Medication
            composable("medication_reminders") { ReminderListScreen(navController) }
            composable("create_reminder/{patientId}") { CreateReminderScreen(navController) }
            composable("adherence/{patientId}") { AdherenceDashboardScreen(navController) }

            // Patient
            composable("patients") { PatientListScreen(navController) }
            composable("patient/{patientId}") { PatientDetailScreen(navController) }
            composable("observation_timeline/{patientId}") { ObservationTimelineScreen(navController) }
            composable("record_observation/{patientId}") { RecordObservationScreen(navController) }

            // Care Team
            composable("care_team/{patientId}") { CareTeamScreen(navController) }
            composable("add_team_member/{patientId}") { AddMemberScreen(navController) }

            // Vignette Assessment
            composable("vignettes") { VignetteListScreen(navController) }
            composable("vignette/{vignetteId}") { VignetteDetailScreen(navController) }
            composable("vignette_response/{vignetteId}/{withTool}") { VignetteResponseScreen(navController) }

            // Settings
            composable("settings") { SettingsScreen(navController) }
            composable("language_settings") { LanguageSettingsScreen(navController) }
            composable("data_export") { DataExportScreen(navController) }
            composable("sus_questionnaire") { SusScreen(navController) }
            composable("about") { AboutScreen(navController) }
        }
    }
}
```

### 9.2 Key Screen Designs

#### Voice Query Screen (PRIMARY — where ASHAs spend 90% of time)

```
┌────────────────────────────────────────┐
│ ◄ Back      Palli Sahayak    ● Online  │  <- SyncStatusChip
│                                         │
│  ┌────────────────────────────────────┐ │
│  │                                    │ │
│  │     [Response area - scrollable]   │ │
│  │                                    │ │
│  │  "For moderate pain, you can       │ │
│  │   consider paracetamol 500mg       │ │
│  │   every 6 hours..."               │ │
│  │                                    │ │
│  │  ┌──────────────────┐             │ │
│  │  │ 🟢 Evidence: B   │             │ │  <- EvidenceBadge
│  │  │ Good Evidence     │             │ │
│  │  └──────────────────┘             │ │
│  │                                    │ │
│  │  Sources: WHO Pain Ladder, p.12    │ │
│  │           Pallium India HB, p.45   │ │
│  │                                    │ │
│  └────────────────────────────────────┘ │
│                                         │
│              ╔═══════════╗              │
│              ║           ║              │
│              ║    🎤     ║              │  <- 96dp VoiceButton
│              ║           ║              │     (tap to speak)
│              ╚═══════════╝              │
│          "Tap to ask a question"        │
│                                         │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐     │
│  │ 🏠  │ │ 💊  │ │ 👥  │ │ ⚙️  │     │  <- Bottom nav
│  │Home │ │Meds │ │Team │ │More │     │
│  └─────┘ └─────┘ └─────┘ └─────┘     │
└────────────────────────────────────────┘
```

**During recording (voice waveform animation):**
```
┌────────────────────────────────────────┐
│                                         │
│        ~~~~ ▐▌▐▌▐▌▐▌ ~~~~             │  <- Real-time waveform
│                                         │
│              ╔═══════════╗              │
│              ║  ⏹  STOP ║              │  <- Button changes to stop
│              ╚═══════════╝              │
│          "Listening... (3.2s)"          │
└────────────────────────────────────────┘
```

**Emergency detected:**
```
┌────────────────────────────────────────┐
│ ╔══════════════════════════════════════╗│
│ ║ ⚠️  EMERGENCY DETECTED              ║│  <- EmergencyBanner (red)
│ ║                                      ║│
│ ║ Severe bleeding detected.            ║│
│ ║ Call emergency services immediately. ║│
│ ║                                      ║│
│ ║  ┌────────────────────────────┐     ║│
│ ║  │     📞  CALL 108           │     ║│  <- One-tap emergency call
│ ║  └────────────────────────────┘     ║│
│ ╚══════════════════════════════════════╝│
└────────────────────────────────────────┘
```

#### ASHA Dashboard Screen

```
┌────────────────────────────────────────┐
│ Namaste, [ASHA Name]          ● Online │
│ CMC Vellore · Tamil                     │
│                                         │
│ ┌──────────────────────────────────────┐│
│ │  ╔═══════════════════════════════╗   ││
│ │  ║    🎤  Ask a Question         ║   ││  <- Primary CTA (voice query)
│ │  ╚═══════════════════════════════╝   ││
│ └──────────────────────────────────────┘│
│                                         │
│ Today's Visits                          │
│ ┌──────────────┐ ┌──────────────┐      │
│ │ Mrs. Lakshmi │ │ Mr. Kumar    │      │  <- PatientCards
│ │ Pain: ▲ Mod  │ │ Nausea: → Mild│     │
│ │ Next: 2:00pm │ │ Next: 4:00pm │      │
│ └──────────────┘ └──────────────┘      │
│                                         │
│ Medication Reminders        3 pending   │
│ ┌──────────────────────────────────────┐│
│ │ 💊 Morphine 10mg - Mrs. Lakshmi     ││
│ │    Scheduled: 2:00 PM               ││
│ └──────────────────────────────────────┘│
│                                         │
│ Alerts                      1 active    │
│ ┌──────────────────────────────────────┐│
│ │ ⚡ Mr. Kumar: Pain increased from    ││
│ │    Mild to Moderate over 3 days      ││
│ └──────────────────────────────────────┘│
│                                         │
│  🏠     💊      👥      ⚙️             │
│ Home   Meds    Team    More            │
└────────────────────────────────────────┘
```

### 9.3 Design System (`core-ui`)

#### Color Palette (Material3, high contrast for outdoor use)

```kotlin
val PalliSahayakLightColorScheme = lightColorScheme(
    primary = Color(0xFF1B5E20),              // Dark green — trust, health
    onPrimary = Color.White,
    primaryContainer = Color(0xFFA5D6A7),
    secondary = Color(0xFF0D47A1),            // Blue — clinical confidence
    error = Color(0xFFC62828),                // Red — emergency/danger
    background = Color(0xFFFAFAFA),
    surface = Color.White,
)

// Evidence badge colors (matching backend safety_enhancements.py)
val EvidenceBadgeColors = mapOf(
    EvidenceLevel.A to Color(0xFF1B5E20),     // Dark green
    EvidenceLevel.B to Color(0xFF2E7D32),     // Green
    EvidenceLevel.C to Color(0xFFF57F17),     // Amber
    EvidenceLevel.D to Color(0xFFE65100),     // Orange
    EvidenceLevel.E to Color(0xFFC62828),     // Red
)
```

#### Typography (accessible sizing)

```kotlin
val PalliSahayakTypography = Typography(
    displayLarge = TextStyle(fontSize = 32.sp, fontWeight = FontWeight.Bold),     // Screen titles
    headlineMedium = TextStyle(fontSize = 24.sp, fontWeight = FontWeight.SemiBold), // Section headers
    bodyLarge = TextStyle(fontSize = 18.sp, lineHeight = 26.sp),                  // Primary text (response)
    bodyMedium = TextStyle(fontSize = 16.sp, lineHeight = 24.sp),                 // Secondary text
    labelLarge = TextStyle(fontSize = 16.sp, fontWeight = FontWeight.Medium),     // Button text
    labelMedium = TextStyle(fontSize = 14.sp),                                    // Metadata
)
```

#### Touch Targets

- **Primary actions** (voice button, emergency call): 96dp minimum
- **Secondary actions** (patient cards, reminder items): 56dp minimum
- **Tertiary actions** (settings items, navigation): 48dp minimum (Material3 spec)

### 9.4 Accessibility

- All interactive elements have `contentDescription` for TalkBack
- Evidence badges use both color and icon (not color-alone)
- Text contrast ratio: 7:1 minimum (WCAG AAA) for body text
- Voice button has haptic feedback on tap
- Response text auto-plays as TTS (configurable in settings)
- Skeleton/shimmer loading states (no spinner-only screens)

---

## 10. Voice Integration

### 10.1 VoiceEngine Interface

```kotlin
interface VoiceEngine {
    suspend fun speechToText(audio: ByteArray, language: String): Result<String>
    suspend fun textToSpeech(text: String, language: String): Result<ByteArray>
    suspend fun streamVoiceQuery(audio: ByteArray, language: String): Flow<VoiceStreamEvent>
    fun isAvailable(): Boolean
}

sealed class VoiceStreamEvent {
    data class TranscriptUpdate(val text: String, val isFinal: Boolean) : VoiceStreamEvent()
    data class ResponseText(val text: String) : VoiceStreamEvent()
    data class ResponseAudio(val audioData: ByteArray) : VoiceStreamEvent()
    data class Error(val message: String) : VoiceStreamEvent()
    data object Complete : VoiceStreamEvent()
}
```

### 10.2 ServerVoiceEngine (Primary — Online)

Calls Sarvam AI via the backend's mobile API:

```kotlin
class ServerVoiceEngine @Inject constructor(
    private val apiService: PalliSahayakApiService,
    private val languageMapper: LanguageMapper,
) : VoiceEngine {

    override suspend fun speechToText(audio: ByteArray, language: String): Result<String> {
        val bcp47 = languageMapper.toBcp47(language) // "ta" -> "ta-IN"
        val response = apiService.voiceQuery(
            audio = audio.toRequestBody("audio/wav".toMediaType()),
            language = bcp47,
        )
        return Result.Success(response.transcript)
    }

    override suspend fun textToSpeech(text: String, language: String): Result<ByteArray> {
        val bcp47 = languageMapper.toBcp47(language)
        val response = apiService.textToSpeech(
            TtsRequest(text = text, language = bcp47)
        )
        return Result.Success(Base64.decode(response.audioBase64, Base64.DEFAULT))
    }
}
```

### 10.3 OnDeviceVoiceEngine (Fallback — Offline)

```kotlin
class OnDeviceVoiceEngine @Inject constructor(
    @ApplicationContext private val context: Context,
) : VoiceEngine {

    private val speechRecognizer = SpeechRecognizer.createSpeechRecognizer(context)
    private val tts = TextToSpeech(context, null)

    override suspend fun speechToText(audio: ByteArray, language: String): Result<String> =
        suspendCancellableCoroutine { cont ->
            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, language)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            }
            speechRecognizer.setRecognitionListener(object : RecognitionListener {
                override fun onResults(results: Bundle?) {
                    val text = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()
                    if (text != null) cont.resume(Result.Success(text))
                    else cont.resume(Result.Error(Exception("No transcript")))
                }
                override fun onError(error: Int) {
                    cont.resume(Result.Error(Exception("STT error: $error")))
                }
                // ... other callbacks
            })
            speechRecognizer.startListening(intent)
        }

    override fun isAvailable(): Boolean =
        SpeechRecognizer.isRecognitionAvailable(context)
}
```

### 10.4 Language Mapping

Matches `sarvam_integration/config.py` language code mappings:

```kotlin
object LanguageMapper {
    private val codeMap = mapOf(
        "ta" to "ta-IN",      // Tamil
        "te" to "te-IN",      // Telugu
        "kn" to "kn-IN",      // Kannada
        "ml" to "ml-IN",      // Malayalam
        "bn" to "bn-IN",      // Bengali
        "as" to "as-IN",      // Assamese
        "hi" to "hi-IN",      // Hindi
        "en" to "en-IN",      // English (India)
        "tu" to "tu-IN",      // Tulu (mapped to Kannada for STT/TTS)
    )

    // Sarvam TTS voice mapping (matching config.py SARVAM_VOICE_MAP)
    private val voiceMap = mapOf(
        "hi-IN" to "meera",   // Female Hindi voice
        "ta-IN" to "meera",   // Female Tamil voice
        "bn-IN" to "meera",   // Female Bengali voice
        "kn-IN" to "meera",   // Female Kannada voice
        "en-IN" to "meera",   // Female English voice
        "ml-IN" to "meera",   // Female Malayalam voice
        "te-IN" to "meera",   // Female Telugu voice
    )

    fun toBcp47(shortCode: String): String = codeMap[shortCode] ?: "en-IN"
    fun getVoice(bcp47: String): String = voiceMap[bcp47] ?: "meera"
}
```

### 10.5 Emergency Keyword Detection (Local)

Runs on-device before any server call, matching `safety_enhancements.py` emergency patterns:

```kotlin
class EmergencyKeywordDetector {
    private val emergencyPatterns = mapOf(
        "en" to listOf("bleeding", "unconscious", "not breathing", "chest pain", "seizure", "suicide", "severe pain"),
        "hi" to listOf("खून बह रहा", "बेहोश", "सांस नहीं", "छाती में दर्द", "दौरा", "तेज दर्द"),
        "ta" to listOf("இரத்தப்போக்கு", "மயக்கம்", "மூச்சு விடவில்லை", "நெஞ்சு வலி"),
        "bn" to listOf("রক্তপাত", "অচেতন", "শ্বাস নেই", "বুকে ব্যথা"),
        "kn" to listOf("ರಕ್ತಸ್ರಾವ", "ಪ್ರಜ್ಞೆ ತಪ್ಪಿದ", "ಉಸಿರಾಟ ಇಲ್ಲ"),
    )

    fun detect(transcript: String, language: String): EmergencyLevel {
        val normalized = transcript.lowercase().trim()
        val patterns = emergencyPatterns[language] ?: emergencyPatterns["en"]!!
        val matched = patterns.any { normalized.contains(it) }
        return if (matched) EmergencyLevel.CRITICAL else EmergencyLevel.NONE
    }
}
```

### 10.6 Audio Recording

```kotlin
class AudioRecorder {
    companion object {
        const val SAMPLE_RATE = 16000          // 16kHz (Sarvam requirement)
        const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
    }

    private var audioRecord: AudioRecord? = null
    private val _waveformData = MutableStateFlow<FloatArray>(floatArrayOf())
    val waveformData: StateFlow<FloatArray> = _waveformData

    fun startRecording(): Flow<ByteArray> = callbackFlow {
        val bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT, bufferSize
        )
        audioRecord?.startRecording()

        val buffer = ByteArray(bufferSize)
        val outputStream = ByteArrayOutputStream()

        while (isActive) {
            val read = audioRecord?.read(buffer, 0, bufferSize) ?: break
            if (read > 0) {
                outputStream.write(buffer, 0, read)
                _waveformData.value = buffer.toFloatArray() // for waveform visualization
            }
        }

        trySend(outputStream.toByteArray())
        close()
    }

    fun stopRecording() {
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
    }
}
```

---

## 11. Offline-First Architecture

### 11.1 Cache Strategy

| Data Type | Cache Location | TTL | Refresh Strategy |
|-----------|---------------|-----|------------------|
| Top 20 clinical queries | `QueryCacheEntity` (Room) | 7 days | CacheBundleSyncWorker on app launch |
| Top 50 symptom-treatment pairs | `QueryCacheEntity` | 7 days | Bundled with query cache |
| Emergency keywords (all languages) | Hardcoded in APK | N/A | Updated with app releases |
| Patient records | `PatientEntity` (Room) | N/A | Sync on connect, periodic 15-min |
| Observations | `ObservationEntity` (Room) | N/A | Append-only, sync on connect |
| Medication reminders | `MedicationReminderEntity` (Room) | N/A | Sync on connect, AlarmManager backup |
| User profile | `UserEntity` (Room) + DataStore | N/A | Sync on login |
| Clinical vignettes (20) | `VignetteEntity` (Room) | N/A | Downloaded during onboarding |
| TTS voice packs | Android TTS system | N/A | Downloaded during onboarding |

### 11.2 Sync Engine (WorkManager)

```kotlin
class SyncManager @Inject constructor(
    private val workManager: WorkManager,
) {
    fun schedulePeriodicSync() {
        val constraints = Constraints.Builder()
            .setRequiredNetworkType(NetworkType.CONNECTED)
            .build()

        val syncRequest = PeriodicWorkRequestBuilder<SyncWorker>(15, TimeUnit.MINUTES)
            .setConstraints(constraints)
            .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 30, TimeUnit.SECONDS)
            .build()

        workManager.enqueueUniquePeriodicWork(
            "periodic_sync",
            ExistingPeriodicWorkPolicy.KEEP,
            syncRequest
        )
    }

    fun triggerImmediateSync() {
        val syncRequest = OneTimeWorkRequestBuilder<SyncWorker>()
            .setConstraints(Constraints.Builder()
                .setRequiredNetworkType(NetworkType.CONNECTED)
                .build())
            .build()

        workManager.enqueueUniqueWork(
            "immediate_sync",
            ExistingWorkPolicy.REPLACE,
            syncRequest
        )
    }
}
```

### 11.3 Conflict Resolution

| Entity | Strategy | Rationale |
|--------|----------|-----------|
| Observations | Append-only (no conflicts) | Observations are immutable once created; both device and server versions accepted, deduplicated by `observationId` |
| Patient records | Server-wins | Clinical data is authoritative from server; clinician may update records from admin UI |
| Care team | Server-wins | Managed by supervising physician |
| Medication reminders | Device-wins for schedule, server-wins for status | Local AlarmManager is ground truth for when to remind; server tracks confirmation |
| Interaction logs | Append-only | Device generates all logs; server only receives |
| Vignette responses | Device-wins | Captured on device; server only receives for scoring |
| Query cache | Server-wins | Cache is generated server-side and downloaded |

### 11.4 Connectivity Monitor

```kotlin
class NetworkMonitor @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    private val connectivityManager = context.getSystemService<ConnectivityManager>()

    val isOnline: StateFlow<Boolean> = callbackFlow {
        val callback = object : ConnectivityManager.NetworkCallback() {
            override fun onAvailable(network: Network) { trySend(true) }
            override fun onLost(network: Network) { trySend(false) }
        }
        connectivityManager?.registerDefaultNetworkCallback(callback)
        awaitClose { connectivityManager?.unregisterNetworkCallback(callback) }
    }.stateIn(CoroutineScope(Dispatchers.Default), SharingStarted.Eagerly, false)
}
```

---

## 12. Security and Privacy

### 12.1 Authentication Flow

```
App Launch
    │
    ├── [First Launch] ──► Onboarding
    │                       ├── Language Select
    │                       ├── DPDP Consent
    │                       ├── Role Select
    │                       ├── ABHA Link (optional)
    │                       ├── PIN Setup (4-digit)
    │                       │     └── Argon2id hash stored locally
    │                       └── Training
    │                             └── POST /api/mobile/v1/auth/register
    │                                   └── Returns JWT token
    │
    └── [Returning User] ──► PIN Unlock Screen
                              ├── Verify against local Argon2id hash
                              ├── On success: unlock app, refresh JWT if expired
                              └── On 5 failures: wipe local data, force re-onboarding
```

### 12.2 PIN Management

```kotlin
class PinManager @Inject constructor(
    private val encryptedPrefs: SharedPreferences, // EncryptedSharedPreferences
) {
    fun setPin(pin: String) {
        val salt = SecureRandom().generateSeed(16)
        val hash = Argon2id.hash(pin.toByteArray(), salt)
        encryptedPrefs.edit {
            putString("pin_hash", hash.encoded)
            putString("pin_salt", salt.base64())
        }
    }

    fun verifyPin(pin: String): Boolean {
        val storedHash = encryptedPrefs.getString("pin_hash", null) ?: return false
        val salt = encryptedPrefs.getString("pin_salt", null)?.fromBase64() ?: return false
        return Argon2id.verify(storedHash, pin.toByteArray(), salt)
    }
}
```

### 12.3 Data Encryption

| Layer | Method |
|-------|--------|
| Room database | SQLCipher AES-256 (key from AndroidKeyStore) |
| SharedPreferences | EncryptedSharedPreferences (AES-256 GCM) |
| JWT token | Stored in EncryptedSharedPreferences |
| Audio files (temporary) | EncryptedFile API, deleted after upload |
| Network | HTTPS with certificate pinning (OkHttp CertificatePinner) |
| FHIR data | Encrypted at rest via Room; in-transit via HTTPS |

### 12.4 DPDP Act 2023 Compliance

| Requirement | Implementation |
|-------------|---------------|
| Explicit consent | ConsentScreen during onboarding with language-appropriate explanation |
| Purpose limitation | Only collect data for clinical decision support and EVAH evaluation |
| Data minimization | Phone numbers stored as SHA-256 hashes; no persistent voice recordings |
| Right to erasure | Settings > "Delete My Data" clears local DB and sends deletion to server |
| Data localization | Sarvam AI processes on Indian servers; backend deployed in India |
| Breach notification | Firebase Crashlytics monitors for security-relevant crashes |
| Children's data | Not applicable (all participants are adults) |

### 12.5 Session Management

```kotlin
class SessionManager @Inject constructor(
    private val tokenManager: TokenManager,
) {
    companion object {
        const val AUTO_LOCK_TIMEOUT_MS = 5 * 60 * 1000L // 5 minutes
        const val MAX_PIN_ATTEMPTS = 5
    }

    private var lastActivityTimestamp = System.currentTimeMillis()
    private var failedPinAttempts = 0

    fun isSessionValid(): Boolean =
        (System.currentTimeMillis() - lastActivityTimestamp) < AUTO_LOCK_TIMEOUT_MS

    fun recordActivity() { lastActivityTimestamp = System.currentTimeMillis() }

    fun recordFailedPin(): Boolean {
        failedPinAttempts++
        if (failedPinAttempts >= MAX_PIN_ATTEMPTS) {
            wipeLocalData()
            return false // force re-onboarding
        }
        return true
    }
}
```

---

## 13. FHIR and ABDM Integration

### 13.1 Google Android FHIR SDK Usage

The FHIR SDK is used as a **secondary store** for interoperability, not as the primary database.

```kotlin
class FhirEngineSetup @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    val fhirEngine: FhirEngine by lazy {
        FhirEngineProvider.getInstance(context)
    }

    fun initialize() {
        FhirEngineProvider.init(
            FhirEngineConfiguration(
                enableEncryptionIfSupported = true,
                databaseErrorStrategy = DatabaseErrorStrategy.RECREATE_AT_OPEN,
            )
        )
    }
}
```

### 13.2 FHIR Resource Mapping

Maps Room entities to FHIR R4 resources using the same SNOMED CT codes as `personalization/fhir_adapter.py`:

```kotlin
object SnomedCodes {
    // Matching fhir_adapter.py SYMPTOM_SNOMED_CODES (lines 49-63)
    val SYMPTOM_CODES = mapOf(
        "pain" to "22253000",
        "nausea" to "422587007",
        "vomiting" to "422400008",
        "breathlessness" to "267036007",
        "dyspnea" to "267036007",
        "anxiety" to "48694002",
        "depression" to "35489007",
        "constipation" to "14760008",
        "insomnia" to "193462001",
        "fatigue" to "84229001",
        "appetite_loss" to "79890006",
        "confusion" to "40917007",
        "cough" to "49727002",
        "diarrhea" to "62315008",
        "edema" to "267038008",
        "fever" to "386661006",
        "itching" to "418290006",
        "mouth_sores" to "26284000",
    )

    // Matching fhir_adapter.py SEVERITY_FHIR_CODES (lines 66-72)
    val SEVERITY_CODES = mapOf(
        SeverityLevel.NONE to "260413007",     // None
        SeverityLevel.MILD to "255604002",     // Mild
        SeverityLevel.MODERATE to "6736007",   // Moderate
        SeverityLevel.SEVERE to "24484000",    // Severe
        SeverityLevel.VERY_SEVERE to "442452003", // Life-threatening
    )
}
```

### 13.3 ABDM / ABHA Integration

```kotlin
class AbhaIdValidator {
    fun validate(abhaId: String): Boolean {
        // ABHA health ID: 14-digit numeric (XX-XXXX-XXXX-XXXX)
        val cleaned = abhaId.replace("-", "").replace(" ", "")
        return cleaned.length == 14 && cleaned.all { it.isDigit() }
    }
}
```

ABHA ID is stored as a FHIR Patient identifier:
```kotlin
fun addAbhaIdentifier(patient: Patient, abhaId: String): Patient {
    val identifier = Identifier().apply {
        system = "https://healthid.abdm.gov.in"
        value = abhaId
        type = CodeableConcept().apply {
            coding = listOf(Coding("http://terminology.hl7.org/CodeSystem/v2-0203", "MR", "Medical Record"))
        }
    }
    patient.identifier.add(identifier)
    return patient
}
```

---

## 14. Evaluation Instrumentation

### 14.1 Interaction Event Taxonomy

Every user interaction generates a structured event for the EVAH evaluation:

```kotlin
enum class InteractionEventType(val code: String) {
    // Session events
    SESSION_START("session_start"),
    SESSION_END("session_end"),
    PIN_UNLOCK("pin_unlock"),

    // Query events (maps to "Adoption: calls/ASHA/week")
    QUERY_VOICE_START("query_voice_start"),
    QUERY_VOICE_RECORDING_COMPLETE("query_voice_recording_complete"),
    QUERY_STT_COMPLETE("query_stt_complete"),
    QUERY_SUBMITTED("query_submitted"),
    QUERY_RESPONSE_RECEIVED("query_response_received"),
    QUERY_TTS_PLAYED("query_tts_played"),
    QUERY_OFFLINE_CACHE_HIT("query_offline_cache_hit"),
    QUERY_OFFLINE_CACHE_MISS("query_offline_cache_miss"),

    // Safety events (maps to "Safety: hallucination rate, emergency F1")
    EMERGENCY_DETECTED("emergency_detected"),
    EMERGENCY_CALL_INITIATED("emergency_call_initiated"),
    HANDOFF_TRIGGERED("handoff_triggered"),
    EVIDENCE_BADGE_SHOWN("evidence_badge_shown"),

    // Language events (maps to "Language equity: all metrics by language")
    LANGUAGE_DETECTED("language_detected"),
    LANGUAGE_SWITCHED("language_switched"),
    STT_LANGUAGE_MISMATCH("stt_language_mismatch"),

    // Patient management events
    OBSERVATION_CREATED("observation_created"),
    MEDICATION_REMINDER_CREATED("medication_reminder_created"),
    MEDICATION_REMINDER_CONFIRMED("medication_reminder_confirmed"),
    PATIENT_RECORD_VIEWED("patient_record_viewed"),
    CARE_TEAM_CONTACTED("care_team_contacted"),

    // Evaluation-specific events
    SUS_COMPLETED("sus_completed"),
    VIGNETTE_STARTED("vignette_started"),
    VIGNETTE_COMPLETED("vignette_completed"),

    // Sync events
    SYNC_STARTED("sync_started"),
    SYNC_COMPLETED("sync_completed"),
    SYNC_FAILED("sync_failed"),
    WENT_OFFLINE("went_offline"),
    CAME_ONLINE("came_online"),

    // Navigation events
    SCREEN_VIEW("screen_view"),
    FEATURE_USED("feature_used"),
}
```

### 14.2 Time-Motion Tracker

Automatically captures timing data for the EVAH "Workflow integration: Time-motion" primary outcome:

```kotlin
class TimeMotionTracker @Inject constructor(
    private val interactionLogger: InteractionLogger,
) {
    data class TimedEvent(
        val eventName: String,
        val startTime: Long = System.currentTimeMillis(),
        var endTime: Long? = null,
    ) {
        val durationMs: Long? get() = endTime?.let { it - startTime }
    }

    private val activeEvents = ConcurrentHashMap<String, TimedEvent>()

    fun startEvent(eventName: String): TimedEvent {
        val event = TimedEvent(eventName)
        activeEvents[eventName] = event
        return event
    }

    fun endEvent(event: TimedEvent) {
        event.endTime = System.currentTimeMillis()
        activeEvents.remove(event.eventName)
        interactionLogger.log(
            eventType = "time_motion_${event.eventName}",
            durationMs = event.durationMs,
        )
    }
}
```

**Timed segments for voice query interaction:**
1. `tap_to_recording_start` — Microphone button tap to AudioRecord start
2. `recording_duration` — Recording start to stop
3. `stt_latency` — Audio sent to transcript received
4. `server_query_latency` — Query submitted to response received
5. `tts_latency` — Response text to audio playback start
6. `total_interaction` — Microphone tap to response audio complete
7. `comprehension_time` — Response shown to next user action

### 14.3 SUS Questionnaire

10-item System Usability Scale, translated into all 9 site languages:

```kotlin
class SusQuestionnaire {
    val items = listOf(
        SusItem(1, "I think that I would like to use this system frequently"),
        SusItem(2, "I found the system unnecessarily complex"),
        SusItem(3, "I thought the system was easy to use"),
        SusItem(4, "I think that I would need the support of a technical person to use this system"),
        SusItem(5, "I found the various functions in this system were well integrated"),
        SusItem(6, "I thought there was too much inconsistency in this system"),
        SusItem(7, "I would imagine that most people would learn to use this system very quickly"),
        SusItem(8, "I found the system very cumbersome to use"),
        SusItem(9, "I felt very confident using the system"),
        SusItem(10, "I needed to learn a lot of things before I could get going with this system"),
    )

    fun calculateScore(responses: List<Int>): Float {
        // SUS scoring: odd items (1,3,5,7,9): score - 1; even items (2,4,6,8,10): 5 - score
        // Sum * 2.5 = SUS score (0-100)
        var sum = 0
        responses.forEachIndexed { index, score ->
            sum += if ((index + 1) % 2 == 1) score - 1 else 5 - score
        }
        return sum * 2.5f
    }
}
```

**Voice-read option**: Each SUS item can be read aloud via TTS for low-literacy participants.
**Target**: SUS score above 70 (above-average usability, per EVAH section 8).

### 14.4 Clinical Vignette Assessment

Implements the EVAH crossover design (section 6.4):

```kotlin
data class ClinicalVignette(
    val vignetteId: String,                    // V01 through V20
    val title: String,
    val scenario: String,                      // Clinical scenario text
    val scenarioAudioPath: String?,            // Pre-recorded audio (all 9 languages)
    val expectedDomain: String,                // symptom_management, medication, emotional_support, etc.
    val difficulty: String,                    // basic, intermediate, complex
)
```

**Assignment logic** (server-side, per EVAH section 6.4):
- 20 vignettes total per participant
- 10 with Palli Sahayak tool access, 10 without
- Randomized, counterbalanced assignment
- Assignment downloaded during onboarding and cached locally

**Scoring** (server-side, by blinded physicians):
- Clinical accuracy (1-5 Likert)
- Safety (1-5 Likert)
- Empathy (1-5 Likert)
- Actionability (1-5 Likert)
- Completeness (1-5 Likert)

### 14.5 Data Export

```kotlin
class EvaluationDataExporter @Inject constructor(
    private val interactionLogDao: InteractionLogDao,
    private val vignetteResponseDao: VignetteResponseDao,
) {
    suspend fun exportInteractionLogs(startDate: Long, endDate: Long): File {
        val logs = interactionLogDao.getLogsInRange(startDate, endDate)
        return writeCsv("interaction_logs", logs) { log ->
            listOf(log.logId, log.userId, log.sessionId, log.eventType,
                   log.timestamp.toString(), log.durationMs?.toString() ?: "",
                   log.language, log.siteId, log.isOffline.toString())
        }
    }

    suspend fun exportVignetteResponses(): File {
        val responses = vignetteResponseDao.getAll()
        return writeCsv("vignette_responses", responses) { r ->
            listOf(r.responseId, r.userId, r.vignetteId, r.withTool.toString(),
                   r.responseText ?: "", r.startedAt.toString(), r.completedAt.toString(),
                   r.durationMs.toString())
        }
    }
}
```

---

## 15. Build, CI, and Release

### 15.1 Version Catalog (`gradle/libs.versions.toml`)

```toml
[versions]
kotlin = "2.0.21"
agp = "8.5.2"
compose-bom = "2024.12.01"
compose-compiler = "1.5.15"
hilt = "2.51.1"
room = "2.6.1"
retrofit = "2.11.0"
okhttp = "4.12.0"
coil = "2.7.0"
work = "2.9.1"
navigation = "2.8.4"
moshi = "1.15.1"
sqlcipher = "4.5.6"
security-crypto = "1.1.0-alpha06"
biometric = "1.2.0-alpha05"
fhir-engine = "1.0.0"
fhir-data-capture = "1.1.0"
datastore = "1.1.1"
lifecycle = "2.8.7"
coroutines = "1.9.0"
firebase-bom = "33.6.0"

# Testing
junit = "4.13.2"
mockk = "1.13.12"
turbine = "1.1.0"

[libraries]
# Compose
compose-bom = { group = "androidx.compose", name = "compose-bom", version.ref = "compose-bom" }
compose-material3 = { group = "androidx.compose.material3", name = "material3" }
compose-ui = { group = "androidx.compose.ui", name = "ui" }
compose-ui-tooling = { group = "androidx.compose.ui", name = "ui-tooling" }
compose-ui-tooling-preview = { group = "androidx.compose.ui", name = "ui-tooling-preview" }
compose-navigation = { group = "androidx.navigation", name = "navigation-compose", version.ref = "navigation" }

# DI
hilt-android = { group = "com.google.dagger", name = "hilt-android", version.ref = "hilt" }
hilt-compiler = { group = "com.google.dagger", name = "hilt-android-compiler", version.ref = "hilt" }
hilt-navigation-compose = { group = "androidx.hilt", name = "hilt-navigation-compose", version = "1.2.0" }
hilt-work = { group = "androidx.hilt", name = "hilt-work", version = "1.2.0" }

# Data
room-runtime = { group = "androidx.room", name = "room-runtime", version.ref = "room" }
room-compiler = { group = "androidx.room", name = "room-compiler", version.ref = "room" }
room-ktx = { group = "androidx.room", name = "room-ktx", version.ref = "room" }
datastore-preferences = { group = "androidx.datastore", name = "datastore-preferences", version.ref = "datastore" }

# Network
retrofit-core = { group = "com.squareup.retrofit2", name = "retrofit", version.ref = "retrofit" }
retrofit-moshi = { group = "com.squareup.retrofit2", name = "converter-moshi", version.ref = "retrofit" }
okhttp-core = { group = "com.squareup.okhttp3", name = "okhttp", version.ref = "okhttp" }
okhttp-logging = { group = "com.squareup.okhttp3", name = "logging-interceptor", version.ref = "okhttp" }
moshi-kotlin = { group = "com.squareup.moshi", name = "moshi-kotlin", version.ref = "moshi" }
moshi-codegen = { group = "com.squareup.moshi", name = "moshi-kotlin-codegen", version.ref = "moshi" }

# Image
coil-compose = { group = "io.coil-kt", name = "coil-compose", version.ref = "coil" }

# Security
sqlcipher = { group = "net.zetetic", name = "android-database-sqlcipher", version.ref = "sqlcipher" }
security-crypto = { group = "androidx.security", name = "security-crypto", version.ref = "security-crypto" }
biometric = { group = "androidx.biometric", name = "biometric", version.ref = "biometric" }

# Background
work-runtime = { group = "androidx.work", name = "work-runtime-ktx", version.ref = "work" }

# FHIR
fhir-engine = { group = "com.google.android.fhir", name = "engine", version.ref = "fhir-engine" }
fhir-data-capture = { group = "com.google.android.fhir", name = "data-capture", version.ref = "fhir-data-capture" }

# Lifecycle
lifecycle-viewmodel = { group = "androidx.lifecycle", name = "lifecycle-viewmodel-compose", version.ref = "lifecycle" }
lifecycle-runtime = { group = "androidx.lifecycle", name = "lifecycle-runtime-compose", version.ref = "lifecycle" }

# Coroutines
coroutines-android = { group = "org.jetbrains.kotlinx", name = "kotlinx-coroutines-android", version.ref = "coroutines" }
coroutines-test = { group = "org.jetbrains.kotlinx", name = "kotlinx-coroutines-test", version.ref = "coroutines" }

# Firebase
firebase-bom = { group = "com.google.firebase", name = "firebase-bom", version.ref = "firebase-bom" }
firebase-crashlytics = { group = "com.google.firebase", name = "firebase-crashlytics-ktx" }
firebase-analytics = { group = "com.google.firebase", name = "firebase-analytics-ktx" }

# Testing
junit = { group = "junit", name = "junit", version.ref = "junit" }
mockk = { group = "io.mockk", name = "mockk-android", version.ref = "mockk" }
turbine = { group = "app.cash.turbine", name = "turbine", version.ref = "turbine" }
compose-ui-test = { group = "androidx.compose.ui", name = "ui-test-junit4" }
compose-ui-test-manifest = { group = "androidx.compose.ui", name = "ui-test-manifest" }
room-testing = { group = "androidx.room", name = "room-testing", version.ref = "room" }
work-testing = { group = "androidx.work", name = "work-testing", version.ref = "work" }

[plugins]
android-application = { id = "com.android.application", version.ref = "agp" }
android-library = { id = "com.android.library", version.ref = "agp" }
kotlin-android = { id = "org.jetbrains.kotlin.android", version.ref = "kotlin" }
kotlin-compose = { id = "org.jetbrains.kotlin.plugin.compose", version.ref = "kotlin" }
hilt = { id = "com.google.dagger.hilt.android", version.ref = "hilt" }
ksp = { id = "com.google.devtools.ksp", version = "2.0.21-1.0.27" }
firebase-crashlytics = { id = "com.google.firebase.crashlytics", version = "3.0.2" }
gms = { id = "com.google.gms.google-services", version = "4.4.2" }
```

### 15.2 Build Variants

```kotlin
android {
    buildTypes {
        debug {
            isDebuggable = true
            applicationIdSuffix = ".debug"
            buildConfigField("String", "BASE_URL", "\"http://10.0.2.2:8000\"") // Emulator localhost
        }
        create("staging") {
            isDebuggable = true
            applicationIdSuffix = ".staging"
            buildConfigField("String", "BASE_URL", "\"https://staging.pallisahayak.org\"")
        }
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            buildConfigField("String", "BASE_URL", "\"https://api.pallisahayak.org\"")
        }
    }

    flavorDimensions += "site"
    productFlavors {
        create("cmcVellore") { dimension = "site"; buildConfigField("String", "SITE_ID", "\"cmc_vellore\"") }
        create("kmcManipal") { dimension = "site"; buildConfigField("String", "SITE_ID", "\"kmc_manipal\"") }
        create("ccfCoimbatore") { dimension = "site"; buildConfigField("String", "SITE_ID", "\"ccf_coimbatore\"") }
        create("cchrcSilchar") { dimension = "site"; buildConfigField("String", "SITE_ID", "\"cchrc_silchar\"") }
        create("allSites") { dimension = "site"; buildConfigField("String", "SITE_ID", "\"all\"") }
    }
}
```

### 15.3 Testing Strategy

| Layer | Framework | Scope |
|-------|-----------|-------|
| Unit | JUnit 4 + MockK + Turbine | All use cases, repositories (Room in-memory), ViewModels (StateFlow via Turbine) |
| Integration | MockWebServer | Retrofit services against mock server responses |
| UI | Compose UI Test | Critical flows: voice query, SUS, vignette, onboarding |
| E2E | Maestro | Device testing: full voice query flow, offline mode, sync |
| Performance | Macrobenchmark | Cold start (<3s on Redmi 10A), voice query (<13s) |

### 15.4 APK Size Budget

| Component | Estimated Size |
|-----------|---------------|
| Kotlin stdlib + AndroidX | 3MB |
| Jetpack Compose | 4MB |
| Room + SQLCipher | 3MB |
| Retrofit + OkHttp + Moshi | 2MB |
| Hilt | 1MB |
| Firebase | 2MB |
| FHIR SDK (dynamic feature) | 5MB (on demand) |
| App code + resources | 3MB |
| **Total** | **~23MB** (under 25MB target) |

---

## 16. Phased Implementation Plan

### Phase 1: Foundation (Weeks 1-3)

**Goal**: Project scaffold, core modules, Room database, Retrofit services.

**Files to create**:
- Root build files: `settings.gradle.kts`, `build.gradle.kts`, `gradle/libs.versions.toml`
- `app/` module with `PalliSahayakApplication.kt`, `MainActivity.kt`
- `core-common/`: `Result.kt`, `DispatcherProvider.kt`, `AppConstants.kt`
- `core-model/`: All domain enums and data classes (Section 8)
- `core-data/`: `PalliSahayakDatabase.kt`, all 8 Entity classes, all 8 DAOs, `NetworkMonitor.kt`
- `core-network/`: `PalliSahayakApiService.kt` (all endpoint signatures), `AuthInterceptor.kt`, `NetworkModule.kt`
- `core-security/`: `PinManager.kt`, `EncryptionHelper.kt`, `TokenManager.kt`, `SessionManager.kt`
- `core-ui/`: `PalliSahayakTheme.kt`, `Color.kt`, `Typography.kt`

**Verification**: `./gradlew assembleDebug` succeeds. Room schema export generates 8 tables. Retrofit interface compiles.

### Phase 2: Voice Engine (Weeks 4-5)

**Goal**: Audio recording, Sarvam STT/TTS integration, on-device fallback, emergency detection.

**Files to create**:
- `core-voice/`: `VoiceEngine.kt`, `ServerVoiceEngine.kt`, `OnDeviceVoiceEngine.kt`, `AudioRecorder.kt`, `AudioPlayer.kt`, `LanguageMapper.kt`, `EmergencyKeywordDetector.kt`, `VoiceModule.kt`

**Verification**: Unit tests for `LanguageMapper`, `EmergencyKeywordDetector`. Integration test for `ServerVoiceEngine` against mock Sarvam API.

### Phase 3: Primary Interaction Flow (Weeks 6-8)

**Goal**: Voice query end-to-end, onboarding, dashboards.

**Files to create**:
- `feature-onboarding/`: 6 screens + ViewModel + nav graph
- `feature-query/`: `VoiceQueryScreen.kt`, `QueryViewModel.kt`, 3 use cases
- `feature-home/`: 3 dashboard screens + ViewModel
- `app/`: `PalliSahayakNavGraph.kt`

**Verification**: Full flow: launch -> onboard -> set PIN -> voice query -> see response with evidence badge. Offline: cached response shown. Emergency: banner + 108 call button.

### Phase 4: Medication & Patient Management (Weeks 9-11)

**Goal**: Medication reminders, patient records, care team, observation recording, sync engine.

**Files to create**:
- `feature-medication/`: 3 screens + ViewModel + use case + AlarmReceiver
- `feature-patient/`: 4 screens + ViewModel + use case
- `feature-careteam/`: 2 screens + ViewModel
- `core-sync/`: `SyncManager.kt`, 5 sync workers, `ConflictResolver.kt`

**Verification**: Create reminder -> local alarm fires. Record observation -> appears in timeline. Sync worker runs on connect.

### Phase 5: FHIR & ABDM (Weeks 12-13)

**Goal**: FHIR R4 export/import, ABHA ID linking.

**Files to create**:
- `core-fhir/`: `FhirEngineSetup.kt`, `FhirResourceMapper.kt`, `SnomedCodes.kt`, `AbhaIdValidator.kt`
- Update `feature-onboarding/`: `AbhaLinkScreen.kt`
- Update `feature-settings/`: `DataExportScreen.kt`

**Verification**: Export patient as FHIR Bundle -> validate R4 schema. Import test Bundle -> verify in Room.

### Phase 6: Evaluation Instrumentation (Weeks 14-16)

**Goal**: SUS, time-motion, vignettes, interaction logging, data export.

**Files to create**:
- `core-evaluation/`: `InteractionLogger.kt`, `TimeMotionTracker.kt`, `SusQuestionnaire.kt`, `EvaluationDataExporter.kt`
- `feature-vignette/`: 3 screens + ViewModel + use case
- Update `feature-settings/`: `SusScreen.kt`

**Verification**: Complete SUS flow (10 items, score calculated). Complete vignette (with/without tool, timing captured). Export CSV, verify format for R/lme4.

### Phase 7: Polish & Field Readiness (Weeks 17-18)

**Goal**: Performance optimization, accessibility, site-specific builds, field testing.

**Tasks**:
- APK size optimization (target <25MB)
- Cold start profiling (target <3s on Redmi 10A)
- TalkBack audit
- Battery drain measurement (8-hour simulated workday)
- 4 site-specific build flavors
- Field testing protocol documentation
- Crash reporting setup (Firebase Crashlytics)

**Verification**: Install on Redmi 10A. Full workflow on 2G simulation. Battery: <15% drain over 8 hours of moderate use.

---

## 17. CLAUDE.md for Android Project

This file should be placed at the root of the Android project repository:

```markdown
# Android project rules

## Stack (non-negotiable)
- Kotlin only. No Java.
- Jetpack Compose (no XML layouts)
- MVVM + UDF: ViewModel -> StateFlow -> Composable
- Hilt for DI
- Room + SQLCipher + Kotlin Coroutines (no RxJava)
- Retrofit + OkHttp for network
- Coil for images
- Min SDK 26, target SDK 35

## Architecture rules
- One feature = one Gradle module (feature/auth/, feature/home/ etc.)
- ViewModels own no Android framework types except SavedStateHandle
- Repository pattern: no direct Room/Retrofit calls from ViewModels
- All suspend fns run in viewModelScope or dedicated CoroutineScope
- Sealed class for UI state: Loading | Success | Error
- Single Source of Truth: Room is the SSOT. Network writes go to Room first.
- All clinical logic is server-side. The app is a presentation + caching layer.

## Code style
- No magic numbers -- use named constants or resource tokens
- Composables must be preview-able (no side effects in preview)
- Functions > 40 lines should be split
- All public functions have type hints on parameters and return
- Prefer immutable data classes for domain models
- Use when() exhaustively on sealed classes

## Voice-first design rules
- Voice button must be accessible from every main screen
- Every text response must have a TTS auto-play option
- Emergency detection runs locally before any server call
- Offline mode must provide cached responses, never a blank screen

## Evaluation rules
- Every user interaction must be logged via InteractionLogger
- Time-motion tracking on all query flows
- Never skip evaluation instrumentation "for simplicity"

## Build
- Run ./gradlew build before declaring a task done
- Run ./gradlew lint and fix all warnings before PR
- APK size must stay under 25MB (check with ./gradlew :app:assembleRelease)
```

---

## 18. Risk Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|-----------|
| Budget phones (2GB RAM) crash under load | High | Medium | Strict module boundaries; lazy feature loading; Coil disk cache; Room indices; profile on Redmi 10A |
| Sarvam API latency on 2G networks | High | High | 10-second client timeout; on-device STT fallback; streaming partial response display |
| FHIR SDK bloats APK beyond 25MB | Medium | Medium | Dynamic feature module; load FHIR only when export/import needed |
| SQLCipher performance on budget CPUs | Medium | Low | WAL journal mode; batch inserts; `PRAGMA cipher_memory_security = OFF` |
| ASHA workers forget PIN | Medium | High | Recovery via supervising physician; site coordinator can reset |
| Connectivity worse than expected at CCHRC | High | Medium | Multi-SIM failover (Jio+Airtel); cache top 20 queries + 50 treatments; 2G-only mode |
| TTS quality varies by language | Medium | Medium | Pre-recorded audio for onboarding; Sarvam voices validated per language; fallback to Hindi TTS |
| Device sharing causes data leakage | High | Medium | PIN per session; 5-min auto-lock; no PHI in notifications; wipe after 5 failed PINs |
| Play Store review delays deployment | Medium | Low | Direct APK distribution via site coordinators; Play Store as secondary channel |
| Battery drain from voice recording + sync | Medium | Medium | WorkManager respects Doze; voice recording uses efficient PCM; sync only when connected |

---

## 19. Appendices

### Appendix A: Backend API Endpoint Reference

Full mapping of all backend endpoints the Android app calls. See companion document `v50_20260327_0947ist_210s__changes-to-existing-rag_gci-codebase_detailed-specs.md` for the new `/api/mobile/v1/*` endpoints.

### Appendix B: Language Code Reference

| Language | ISO 639-1 | BCP-47 | Sarvam STT | Sarvam TTS | On-Device STT | On-Device TTS |
|----------|-----------|--------|------------|------------|---------------|---------------|
| Tamil | ta | ta-IN | Yes | Yes | Limited | Yes |
| Telugu | te | te-IN | Yes | Yes | No | Yes |
| Kannada | kn | kn-IN | Yes | Yes | No | Yes |
| Tulu | - | - | No (use kn) | No (use kn) | No | No |
| Malayalam | ml | ml-IN | Yes | Yes | No | Yes |
| Bengali | bn | bn-IN | Yes | Yes | Limited | Yes |
| Assamese | as | as-IN | Yes | No (use hi) | No | No |
| Hindi | hi | hi-IN | Yes | Yes | Yes | Yes |
| English | en | en-IN | Yes | Yes | Yes | Yes |

### Appendix C: EVAH Outcome-to-Feature Mapping

| EVAH Outcome | Measure | App Feature |
|-------------|---------|-------------|
| Usability | SUS score | `feature-settings/SusScreen.kt` |
| Adoption | Calls/ASHA/week | `core-evaluation/InteractionLogger.kt` (QUERY_* events) |
| Safety | Hallucination rate, emergency F1 | `core-voice/EmergencyKeywordDetector.kt` + server-side validation |
| Workflow integration | Time-motion, disruption score | `core-evaluation/TimeMotionTracker.kt` |
| Knowledge | Pre-post palliative care test | External instrument (not in-app) |
| Confidence | Self-efficacy scale | External instrument (not in-app) |
| Clinical appropriateness | Likert 1-5 on vignettes | `feature-vignette/VignetteResponseScreen.kt` |
| Language equity | All metrics by language | `InteractionLogger` captures language on every event |
| Cost per consultation | Total cost / consultations | Derived from server logs + budget data |
| Cross-group patterns | By user group | `UserEntity.role` as stratification variable |

### Appendix D: Existing Backend Reuse Mapping

| Backend Component | File | Android Reuse Strategy |
|-------------------|------|----------------------|
| RAG Pipeline | `simple_rag_server.py` | Call via `/api/mobile/v1/query` (thin wrapper) |
| Safety Framework | `safety_enhancements.py` | Server-side; evidence badges displayed in `EvidenceBadge.kt` |
| Clinical Validation | `clinical_validation/validator.py` | Server-side; validation status shown in QueryResult |
| User Profiles | `personalization/user_profile.py` | Mirror as `UserEntity` in Room; sync via mobile API |
| Longitudinal Memory | `personalization/longitudinal_memory.py` | Mirror observations in Room; sync via mobile API |
| FHIR Adapter | `personalization/fhir_adapter.py` | Reuse SNOMED codes in `SnomedCodes.kt`; call via mobile API |
| Medication Reminders | `medication_voice_reminders.py` | Mirror in Room + local AlarmManager; sync via mobile API |
| Sarvam STT/TTS | `sarvam_integration/client.py` | Call via `/api/mobile/v1/query/voice` (server proxies to Sarvam) |
| Knowledge Graph | `knowledge_graph/` | Call via `/api/mobile/v1/` endpoints; cache top 50 treatments |
| Analytics | `analytics/usage_analytics.py` | Send interaction logs via `/api/mobile/v1/evaluation/logs` |

---

**End of Document**

*Version 0.1.0 | 27 March 2026 | Palli Sahayak Android App Detailed Specification*
