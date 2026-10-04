# Palli Sahayak — Resume Brief

**Purpose:** paste this into a fresh session after context compaction to resume work
without re-deriving anything.
**Written:** 4 October 2026. Updated at the end of the T4 build.
**Commit at update:** `7f33386` on `main`, clean and pushed.
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
| Tickets | `docs/tickets/01…08-*.md` (in the repository) |
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
T3  Walking skeleton + client safety core              ✅ COMPLETE
T4  Voice — push-to-talk, then real-time               ✅ COMPLETE
T5  Voice resilience — fallback + admin override       🔶 arbitration, override, audit done; Gemini transport outstanding
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

### T3 — complete

**Done.** The Android client runs end to end. A typed question returns a grounded
answer, the safety invariants hold on that first answer, and the interaction is
recorded.

- `:core:safety` — `DoseDetector`, `Redactor`, `DoseBoundary`, `EmergencyDetector`,
  `DeferralCopy`. 23 tests.
- `:core:data` — Room entities and DAOs, `PalliDatabase`, `TextScrubber`,
  `PalliSahayakRepository`, `AskViewModel`, `ServiceLocator`.
- `:core:api` — Kotlin client generated from the contract.
- `app` — `AskScreen` with the three answer kinds rendered distinctly, `AppRoot`,
  `MainActivity`. Debug APK builds at 10.6 MB.
- **All six Robolectric tests are green.** 35 Kotlin tests pass.

### T4 — complete

Voice works end to end, and voice obeys the same safety rules as text.

- **Push-to-talk** — 16-bit PCM at 16 kHz mono, press-and-hold mic, spoken and
  readable answer, replay, and a stop control that shows while audio plays.
- **`/api/mobile/v1/ws/voice`** — session, grounding through the same pipeline, and
  the same `SafetyEnhancementsManager` as the text route.
- **`voice_session.py`** — transcript and answer in one exchange. The client cannot
  know what the user said, so a separate transcript round trip buys nothing.
- **`voice_router.py`** — Live by default with Sarvam fallback. Arbitration is pure
  and tested apart from any provider.
- **`voice_provider.py`** — the admin override. Requires an actor and a reason, and is
  audited with the previous value.
- **Dashboard** — a voice panel showing the selected provider, whether Live is
  reachable, and the fallback rate.

**Still not connected:** `gemini_live/service.py` has no `stream_audio` method, so
`LiveVoiceProvider` has nothing to call. That adapter is the last mile of real-time
voice. Everything either side of it is built and tested.

### T5 — complete in code, blocked on one dependency

Arbitration, fallback rate, operator override and audit trail all exist. They are
now also **connected**, which they were not before. See the findings below.

Remaining: `google-genai` is not installed, so `LiveAvailability.probe()` correctly
reports Live unavailable and every turn falls back to Sarvam. Installing it needs a
decision (see section 10).

## Findings that were expensive to learn

**The cause of the red Robolectric tests was Room, not the network.** Room's suspend
DAO methods dispatch to its transaction executor, and under Robolectric that executor
never resumes the continuation, so the repository hung inside `insert()`. The tests
were reporting an incomplete repository as a safety failure. Fixed with a direct
executor in the test database, and by stubbing the API so only the network is faked.

**Three diagnoses on the way were wrong, and the pattern repeats.** I blamed Moshi,
then the network, then believed Robolectric had swallowed the instrumentation. Each
was wrong. A throwaway test proved Moshi fine; replacing the server proved the network
fine; and `android.util.Log` is swallowed by Robolectric, so `println` is required when
instrumenting. Guessing costs more than measuring.

**A reporting bug the tests caught.** `SafetyResult` has no `answer_kind` field, so a
`getattr` fallback reported every voice answer as `ANSWER` regardless of what the
guard did. The safety layer was working while the reporting claimed otherwise, which
would have under-reported every refusal. Voice now derives the kind from
`dosage_blocked`.

**A comment that lied about the code.** `InteractionEntity` claimed the app
pseudonymises participant ids the way the server does. It does not, and should not:
`mobile_api/router.py:70` shows registration issues `uuid4()`, so the device never
receives a raw identifier. The code was right and the comment was wrong.

**A dashboard number that meant nothing.** `get_voice_router` was added by a patch
whose anchor lived in another file, so it never existed, and the fallback counter
returned zero forever. It now raises when no router is registered, and the panel says
"no traffic yet" rather than "0 percent fallback".

**Fabricated audit entries, from my own smoke test.** Testing the admin endpoints
wrote two entries into the voice provider audit trail under a real person's name,
describing an incident that never happened. A study audit trail with invented
entries is worse than an empty one. Both files are now gitignored as runtime state.

**The voice router was dead code.** `set_voice_router` had no callers,
`LiveVoiceProvider` was constructed nowhere, and `handle_audio` was reachable only
from a provider nothing built. The Android app's real voice route, `POST
/query/voice`, transcribed with Sarvam directly, hardcoded its own grounding and
safety, and never consulted the router. The socket route read a transcript from the
client, which cannot hear itself. So every field turn bypassed arbitration, never
saw an operator override, and reported `voice_path=live` while using Sarvam
throughout. Both routes now go through `VoiceRouter`.

**Both voice routes would have returned HTTP 500 on every turn.** They called
`rag_pipeline.query` with keywords the real pipeline does not accept: `language` on
the socket path, and `language` plus `query_text` on the REST path. The real
signature is `(question, conversation_id, user_id, top_k, source_language)`. The REST
route then read `result.answer` off what is a `Dict`. Every test passed because the
fakes accepted `**kw`, so they were more forgiving than the code they stood in for.
`FakeRag` now mirrors the real signature and a guard test fails the build if the two
drift apart again.

**The docs reference was stale the moment it was written.** It embedded the commit
hash it was generated from, so its own staleness gate could never pass. The gate now
excludes build time and commit from the comparison, and is tested in both
directions.

**A build directory outside `.gitignore`.** The first MkDocs build wrote an untracked
`site/` at the repository root. A committed copy of built docs is how a stale page
becomes permanent.

**The repository is public.** Verified: `gh repo view` reports `visibility: PUBLIC`.
Checked before assuming: `uploads/` has no tracked files, `data/study/` holds only
the release manifest, `.env` was never committed, and a scan of all history for
credential-shaped strings found nothing. So there is no PHI or secret exposure
today. The live risk is forward-looking, which is why CI now runs gitleaks. Whether
the protocol and grant documents should be published at all is a study decision,
not an engineering one.

**All five documentation diagrams failed to render twice** before I checked, for
reasons unrelated to the diagrams: a missing headless Chrome, then a sandbox
restriction. A diagram that silently fails looks deliberate. Render them.

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
**Now resolved:** all four are wired through `study_logging.py` and feed
`/admin/study`. Listed here because it was the pattern, not because it is open.

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
Python  686 passed, 2 failed
Kotlin   35 passed, 0 failed
APK builds 10.6 MB · both contract gates green · 29 REST routes plus /ws/voice
```

The 2 Python failures are `test_voice_optimization` and `test_retell_integration` in
`tests/test_voice_safety_integration.py`. They predate this work;
`voice_safety_wrapper.py` last changed in `f1b4823`.

I previously quoted "105 tests green", which was only my own suites and never a full
run, because collection aborted. Do not quote a number that was not measured.

## 10. What is next

### Everything green, as of 2026-10-04

```
Python  716 passed, 0 failed
CI      all four jobs green: Python, Android, docs staleness, gitleaks
Docs    https://inventcures.github.io/rag_gci/  rebuilt on every push
Kotlin  35 passed
```

The last two red Python tests are fixed, and both were real defects that had been
reported as pre-existing and left for a long time.

**Handoff requests went unmatched whenever phrased naturally.**
`check_handoff_needed` compared keywords by exact substring, and "speak to doctor"
is not a substring of "speak to a doctor". Asking for a human in the ordinary way
meant being answered by the bot. Matching now ignores articles and politeness
fillers.

**Voice responses were not bounded in time.** `optimize_for_voice` truncated on
word count alone, and one long token is one word, so a 3000-character input was
spoken in full. There is now a character budget, cut on a word boundary.

**The Live tool-call path was a way around the dose boundary.** Ticket 04 asks for
real-time voice to ground through a tool call to the same retrieval and safety path.
That handler returned raw retrieval text to Gemini with no safety manager anywhere on
the route. It reads as the safe one precisely because it is grounded in the verified
knowledge base. Filtering now happens on this server before the model sees anything,
which is also what ADR 0007 requires, since grounding must stay in India.

### The environment

`requirements.txt` has **never** been installable. Three entries cannot resolve:
`ktem` and `kotaemon` are 404 on PyPI because they ship with the vendored
`kotaemon-main` tree, and `whisper-openai>=20231107` names a version that was never
published. CI filters those three out of the real file rather than keeping a second
list, so a dependency added for a feature reaches CI without being added twice.

```
uv venv .venv --python 3.12 --system-site-packages
uv pip install --python .venv/bin/python google-genai pyyaml setuptools
```

`uv`, not `python -m venv`: this machine is PEP 668 managed with no `ensurepip`, and
`python -m venv` fails outright with an unhelpful first line. A `.pth` file puts the
repository root on the path; editable installs were tried and rejected.

### CI notes for whoever touches it next

- The Gradle wrapper pointed at `file:///tmp/gradle.zip`. It resolved on exactly one
  machine and nowhere else, so Android CI could never have run. Now points at 8.11.1.
- The Android SDK is installed explicitly. `android-actions/setup-android` checks the
  version of the sdkmanager baked into the runner image and aborts when it disagrees,
  which is a failure of the image rather than of this project.
- Both contract gates shell out to `scripts/export_openapi_contract.py`, which
  imports the FastAPI app and therefore `auth` and therefore PyJWT. Hand-kept
  dependency lists missed it twice.
- `build_docs.py` reads routes with `ast`, not by importing, because the import made
  the generated page depend on what was installed.
- It also sorts its glob. Unsorted, the page depended on filesystem ordering and
  differed between a working copy and a fresh clone.

### Known limits, stated plainly

- **No Live turn has been exercised against the real Gemini endpoint.** Every test
  fakes the network. The transport is written and wired, not proven in anger.
- **Interruption has no test.** The server discards the half-heard turn and the client
  sends the frame, but nothing asserts the sequence.
- **Ticket 04 criterion 4 has no test.** The override is admin-only by inspection
  rather than by assertion. On a clinical system that should close.
- ~~The tickets live outside the repository.~~ **Done 2026-10-04.** They are now at
  `docs/tickets/` and version controlled. Stale copies may remain at
  `/home/tp53/.scratch/android-app-v1/issues/`; the repository wins.
- **The pi-lens type checker reports root-level imports as unresolved and ignores
  `pyrightconfig.json` entirely**, verified by measurement: standalone `npx pyright`
  reports zero errors on the same files, `refreshRunners: all` changes nothing, and a
  dedicated config in `tests/` changed nothing. Tool-side fix required. The code is
  clean.

### Needs you, and I cannot do it from here

- **The repository is public.** No PHI or secret exposure, verified by scanning all
  history and confirming `uploads/` holds no tracked files. Whether the protocol and
  grant documents should be published is a study decision, not an engineering one.
- `STUDY_LOG_SALT` is unset. Participant ids are not linkable across restarts, and
  sustained adoption is unmeasurable without it.
- The Gemini API key is in `.env` and in the session transcript on disk. Rotate it.
- Residency needs written confirmation from Google. A selected region is not itself
  evidence.

## 11. Working practices worth keeping

- **Check the expected test count, not the exit code.** Two false greens: a JUnit 4
  import that produced a passing build with zero tests run, and a pytest run
  reporting success while collecting nothing.
- **Run the real corpus.** The evaluation harness running zero vignettes looked
  identical to running eighty.
- **Prove a guard fails.** A check that cannot fail is decoration. Both contract gates
  were verified with injected drift.
- **Render the diagrams.** All five failed twice before I checked.
- **One source, never two copies.** The Kannada phrase diverged by one codepoint
  because detector and test each held a string.
- **Do not smoke-test against a persistent audit trail.** It fabricated entries under
  a real name.
- **Instrument with `println`, not `Log`, under Robolectric.**
- **Commit per logical chunk.** Every commit lands green.

