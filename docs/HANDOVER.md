# Palli Sahayak — Resume Brief

**Purpose:** paste this into a fresh session after context compaction to resume work
without re-deriving anything.
**Written:** 4 October 2026, at ~57% context.
**Repo:** `/home/tp53/showmethecode/rag_gci` · branch `main` · remote `inventcures/rag_gci`

---

## 1. Working agreement (non-negotiable)

**The user takes product and design decisions. I take software decisions and code like a
senior software engineer.**

- Never decide product or design unilaterally. Who the users are, what a screen shows, what
  to prioritise, whether something ships, naming — these are the user's. Bring a
  recommendation with trade-offs.
- Own module boundaries, data structures, interfaces, error handling, concurrency, build
  layout, test strategy, library choice, code structure. Decide these directly and state the
  reasoning so it can be overridden cheaply.
- Investigate before asking. If a fact is in the filesystem or a tool, look it up.
- Surface contradictions rather than resolving them quietly.
- This is a **clinical safety system under an IRB-approved study protocol.** Safety, consent
  and privacy are constraints, not trade-offs against features.

Recorded in `AGENTS.md` (project) and `~/.pi/agent/AGENTS.md` (cross-project).

---

## 2. What this project is

Palli Sahayak: a voice-first clinical decision support system giving palliative care guidance
to ASHA Workers, Family Caregivers and Patients in India, across eleven Indic languages.
Evaluating under an **EVAH grant** (Wellcome, Gates, Novo Nordisk) with a 12-month
**study protocol v3** across four sites (CMC Vellore, KMC Manipal, CCF Coimbatore,
CCHRC Silchar).

Python FastAPI backend exists and is instrumented. The Android client did not exist until
this session.

---

## 3. Source of truth — read these, not this file, for detail

| What | Where |
|---|---|
| Architecture decisions | `docs/adr/0001` … `0007` |
| Domain glossary (17 terms) | `CONTEXT.md` |
| Implementation spec (86 user stories) | `docs/android-app-specs/v1_03oct2026_android-app-implementation-spec.md` |
| Technical spec, amended to 0.2.0 | `docs/android-app-specs/v0_27march2026_…_detailed-android-app-specs.md` |
| Tickets | `/home/tp53/.scratch/android-app-v1/issues/01…08-*.md` |
| Study release record | `data/study/release.yaml` |
| Grant PDF text | `/tmp/grant.txt` (extracted, 7,582 lines) |
| Protocol PDF text | `/tmp/protocol.txt` (extracted, 1,264 lines) |

**`.scratch/` is outside the repo** and will not survive a repo-based resume. If it is
missing, recreate from §6 below.

---

## 4. Architecture decisions (7 ADRs — all resolved, do not relitigate)

1. **0001** One app, three Care Roles, enforced server-side.
2. **0002** Eleven languages, bounded by **TTS not STT**. No Hindi substitution, ever.
   `tu-IN` removed (not a Sarvam language). `mr/od/pa/gu-IN` added. Bulbul v3 speakers
   (`priya`/`karun`), not the deprecated `meera`.
3. **0003** Cross-language retrieval via **bge-m3**. Query translation is **fallback only**.
4. **0004** **Dose Boundary** — never state a specific dose/strength/titration/escalation/
   interaction/start-stop, regardless of OTC status. Redact sentence-level, keep the
   remainder; full Deferral if too little survives. **Emergency Override is CRITICAL only.**
5. **0005** Offline = cached answer bundle + local history. On-device embeddings ruled out
   with numbers. Disabled mic + icon + plain text, not a banner.
6. **0006** Care Role is an **analysis label, not an access boundary**. One Household Record.
   Backend-admin-only override. Never imply privacy.
7. **0007** Gemini 3.8 Live voice + automatic Sarvam fallback; Gemini 3.5 Flash @
   `asia-south1` for RAG.

**Two more decided in T3, recorded in code and commit messages rather than ADRs** (consider
writing ADR-0008 if they warrant one):
- Generated API client lives in its own module so the contract is a real seam.
- The Dosage Guard is duplicated client-side as defence in depth.

---

## 5. Ticket status

```bash
T1  Retrieval — bge-m3 end to end                      ✅ COMPLETE  8/8
T2  Project and contract seam                          ✅ COMPLETE  5/5
T3  Walking skeleton + client safety core              🔶 PARTIAL  (safety done, data layer scaffolded only)
T4  Voice — push-to-talk, then real-time               ⬜ not started
T5  Voice resilience — fallback + admin override       ⬜
T6  Language, household and roles                      ⬜
T7  Offline — history, state, cached bundle            ⬜
T8  Accessibility contract + release gate              ⬜
```

### T1 — complete
- Live index rebuilt: 263 chunks, bge-m3, 1024-dim, identity written to the collection.
- **Measured:** mean hit@5 **0.96** across 11 languages (was **0.09**); lowest Punjabi 0.78.
  The old index scored **0.00 on every Indic language** and 1.00 on English.
- Hindi/Tamil/Kannada/Bengali morphine queries each return the correct passage.
- All 80 eval outputs regenerated: **0 doses**, 41 partial redactions, 0 full refusals.
- Offline questions: 20 × 11 languages, correct scripts, no dosage question.
- Translation is now fallback-only, gated on a distance probe.

### T2 — complete
- Debug APK builds (~10.4 MB).
- `verifyApiContract` compares server schema to the committed contract and **fails the
  build naming the changed field**. Proven with two injected renames.
- `verifyGeneratedClient` fails if committed Kotlin is stale vs the contract.
- Comparison resolves `$ref` into `components` — without that, component-level renames pass
  silently.

### T3 — safety foundation done, rest outstanding
**Done:** `:core:safety` (pure Kotlin, no Android) — `DoseDetector`, `Redactor`,
`DoseBoundary`, `EmergencyDetector`, `DeferralCopy`. **23 tests, 0 failures.**

**Outstanding:**
- Room entity + DAO (the `InteractionEntity` is where **SI-5 instrumentation** must land)
- Repository + Retrofit implementation
- ViewModel exposing `SafetyResult`
- Compose screen: ask, answer, **Redacted Answer rendered distinctly**
- SI-5 instrumentation on every interaction
- Robolectric test through the real stack

---

## 6. Tickets still to be written (if `.scratch/` is lost)

```
04 Voice — push-to-talk, then real-time
   Blocked by: 03
   - held press records audio, returns a spoken answer without any text input
   - answer shown as text as well as spoken, and replayable
   - clear indication of listening and of finished speaking
   - spoken output stops immediately
   - a real-time conversation endpoint exists, grounded through a tool call back to the
     same retrieval and safety path
   - the app can interrupt real-time playback
   - answers over the voice path still pass the Dose Boundary and the safety invariants

05 Voice resilience — fallback and backend-admin override
   Blocked by: 04
   - fallback triggers on first-token timeout AND per-turn stall, not just failed connection
   - interface shows which voice mode is active
   - backend admin can force the India-resident path deployment-wide
   - site coordinators and field staff cannot
   - every override records actor, timestamp, previous and new value, required reason,
     append-only
   - audit view reports interactions per voice path
   - every interaction records which voice path answered

06 Language, household and roles
   Blocked by: 03
   - language chosen once and remembered
   - all eleven supported, every screen and spoken prompt in each
   - unsupported language refused plainly; never substituted
   - household registrable; observation against the right member
   - everyone in a household reads the same record
   - role switch one tap, no PIN, never shown with a lock or shield
   - Active Role as a persistent large icon
   - three role icons differ in silhouette and posture: `medical_services` (ASHA Worker),
     `diversity_2` (Family Caregiver), `bed` (Patient)
   - a participant cannot read another household's record

07 Offline — history, state and cached bundle
   Blocked by: 03, 01
   - past interactions readable offline, survive the app being closed
   - Redacted Answer renders distinctly in history
   - offline state persistent, icon + colour + text
   - mic visible but disabled with an explanation
   - scheduled medication reminders keep firing offline
   - anticipated questions answered offline in the user's language
   - a question outside the bundle explained, not silent
   - cached answers marked as such
   - clean recovery when signal returns

08 Accessibility contract and release gate
   Blocked by: 01, 06, 07
   - primary actions completable with pictures alone
   - enlarged touch targets; no gesture-only actions
   - important states carried by colour as well as text
   - no screen requires English
   - legible at largest font on a small screen
   - app refuses to start against an unapproved release
   - release drift visible before any participant uses it
   - installs and runs on the low-end reference device
   - evidence recorded that the dosage restriction is enforced and tested
```

---

## 7. Toolchain — not derivable from the repo

Everything is **user-local under `~/tools`** (no root, no sudo on this machine):

```bash
export JAVA_HOME=~/tools/jdk17            # Temurin 17.0.20.1
export ANDROID_HOME=~/tools/android-sdk   # platform-35, build-tools 35.0.0, platform-tools
export PATH=~/tools/gradle-8.11.1/bin:$PATH
```

Gradle wrapper exists at `android-app/gradlew`. **It has not been exercised** — only the
local Gradle install was used, because the wrapper's distribution-URL validation check
failed without reliable network.

API key: written to `.env` as `GEMINI_API_KEY`, gitignored, perms 600. **It works** —
authenticated against `gemini-3.5-flash`. Two cautions:
- I **inferred** the variable name. The key starts `AQ.`, not Google's usual `AIza`. If it
  belongs to another service, move it.
- **The key is in the session transcript on disk.** Rotate when convenient.
- `STUDY_LOG_SALT` is **not** set, so participant ids are ephemeral and not linkable across
  restarts. Must be set before any participant use — sustained adoption is unmeasurable
  without it.

Also installed: 25 official Google Android skills at `~/.pi/agent/skills/android/`
(from github.com/android/skills). CommCare cloned at `/tmp/commcare-android` for reference.

---

## 8. Hard-won state that is NOT in the repo

### The highest-value findings of the whole session

**A. The live index was English-only and nobody noticed.** 0.00 hit@5 on all ten Indic
languages, 1.00 on English. A Marathi question returned confident, unfounded palliative
guidance. No error anywhere. This is *the* defect everything else responded to.

**B. The mobile API had never worked.** `mobile_api/router.py` called
`safety_manager.process_response(...)` on both `/query` and `/query/voice`, but
`SafetyEnhancementsManager` never defined it. Every call raised `AttributeError` → HTTP 500.
Fixed by adding `process_response`.

**C. The eval harness had been running nothing.** `run_rag_on_vignettes.py` globbed
`VIGNETTES_DIR/*.json` non-recursively, but vignettes live in `v1/` and `v2/`. It exited
"No vignettes found" while *appearing* to work. Nine ids also collide across versions, so a
naive recursive glob would have dropped nine of eighty.

**D. The eval harness never applied the safety layer.** It called `rag_pipeline.query()`
directly. That is why **51 of 80** stored outputs contained doses — they were never run
through the Dose Boundary.

**E. The eval runner also had a stale guard**: `_generate_answer_with_citations` checked
`GROQ_API_KEY` *before* reaching its own Gemini-first routing, so a Gemini-configured
deployment never called Gemini.

**F. Nothing was ever wired to anything.** `RealtimeMetrics` (four-stage latency),
`UsageAnalytics`, `clinical_validation.ExpertSampler` (5/50/100) and `interaction_logs/`
were all written and called by nothing. The grant's entire analytic plan depended on them.

### Do-not-repeat list

Every one of these was **silent**, and every one was caught by *executing* rather than
reading. This is the pattern worth preserving.

- **Thai digits missing** from the Kotlin dose detector → `๑๐ มิลลิกรัม` passed the guard.
  Python had them; Kotlin didn't.
- **Bengali units missing entirely** from the Kotlin unit alternation → `১০ মিলিগ্রাম`
  passed. Found by checking which Unicode scripts the alternation *actually* covered.
- **`import org.junit.Test`** (JUnit 4) against a Jupiter engine → **zero tests ran and the
  build reported SUCCESS.** Caught only by checking the expected test count, not the exit
  code.
- **Two hand-typed copies of the same Kannada phrase** diverged by one codepoint
  (`ಜ`+virama+`ಞ` vs `ಜ`+`ೆ`). Fixed structurally: shared constants + a test that sweeps the
  detector's own list.
- **Tamil transliteration written where Tamil script belonged**, in patient-facing copy.
  Caught by diffing against HEAD, not by reading.
- **`if not embeddings` on a numpy array** — ambiguous truth value, crashes.
- **Optional fields emitted as non-nullable** with a `null` default — generated Kotlin that
  doesn't compile.
- **The multipart body is a `$ref`**, so voice-route parts were invisible until resolved.
- **`find_spec("google")` succeeds** even when `google.genai` doesn't, because it is a
  namespace package.
- **v25 longitudinal context** created files (`data/comprehension_profiles/`,
  `data/handoff_requests/`, `data/longitudinal/`) during tests. Untracked; gitignore or
  clean as desired.
- `git stash -u` then `pop` **fails** when untracked runtime files already exist. Check
  state before popping.

### Known limitations, deliberately not hidden

- **The weak-evidence probe is not discriminative.** With 263 chunks every string has a near
  neighbour under the 1.5 threshold, so gibberish also reads as "not weak". It is a
  weak-evidence signal, not a relevance classifier. Becomes meaningful as the corpus grows.
- **All Indic content is engineering-authored** — offline questions (11 languages),
  deferral copy, emergency phrases, retrieval probes. Structurally tested; fluently
  unverified. **A real gate before the study build, and it grows with every language
  feature added.**
- **The eval harness carries its own inline translations**, so native review of *that* is a
  data edit in one module rather than a config swap. The offline bundle already accepts an
  external set.
- **Corpus is 263 chunks from 3 PDFs; 35 of those are a table of contents.** bge-m3's 0.96
  was measured against that ceiling. Confirm this is the intended knowledge base.
- **§7.7 remains the participant-use gate.** The guard is enforced and tested, but the
  release is `status: draft` with both approvals unsigned, and the §7.7 checks have not
  been formally passed.
- **Residency needs written confirmation from Google.** Live voice leaves India; the LLM and
  embeddings do not. A selected region is not itself evidence, and Google's docs warn
  endpoints "don't guarantee data residency".

---

## 9. Test state — real numbers

```
649 passed, 2 failed        (pytest now completes; previously it aborted at collection)
23 Kotlin safety tests, 0 failures
```

Both Python failures are in `test_voice_safety_integration.py`
(`test_voice_optimization`, `test_retell_integration`) and **predate this work** —
`voice_safety_wrapper.py` last changed in `f1b4823`.

I previously reported "105 tests green" — that was **my 6 suites only**, never a full run.
Do not quote a number that was not measured.

Baseline fixes in `b93ec19`: `pytest-asyncio` installed, `asyncio_mode = auto`,
`tests/conftest.py` skips modules whose optional dependency is absent, and a run that
collects nothing now says so and fails.

---

## 10. Immediate next steps

**Agreed plan:** build the repository in T4 using Room, letting it absorb T3's data-layer
sub-task, rather than doing T3-then-T4 as separate passes over the same code. Confirmed by
the user.

1. Room `InteractionEntity` — **SI-5 lives here**: participant id, care role, locale,
   connectivity, bundle-vs-live, voice path used, release id, safety outcome.
2. DAO + database, then repository over the generated Retrofit client.
3. Robolectric test through the real stack (mock only network/clock/hardware; **never our own
   code** — CommCare's rule).
4. Commit; write a fresh handover.
5. Then T4's push-to-talk slice: record → `POST /query/voice` → TTS playback.

The `:core:data` module is scaffolded and wired into `settings.gradle.kts`, with Room and
KSP dependencies declared. **Nothing has been written in it yet** and it has not been
compiled — expect dependency-resolution to need attention on first build.

---

## 11. Working practices worth keeping

- **Check the expected test count, not the exit code.** Two separate false greens this
  session (JUnit 4 import; pytest collecting nothing).
- **Run the real corpus.** The eval harness running zero vignettes looked identical to
  running eighty.
- **Prove a guard fails.** A contract check that cannot fail is decoration. Injected drift
  to prove both gates.
- **Prefer one source over two copies.** The Kannada divergence existed because detector and
  test each held a string.
- **Look up the real value; do not quote the comment.** `config.yaml` said the Live API was
  US/EU only — true but stale; Google's current table showed Seoul too, and that Gemini
  3.8/3.7/3.6 Flash have **no** Asia-Pacific availability at all, which is why the RAG LLM is
  3.5 Flash.
- **Diff against HEAD when editing Indic text.** Two corruptions caught this way.
- **Commit per ticket and push.** Every ticket lands green, so history doubles as a demo log.
