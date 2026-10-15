# Code review round 3: decision log

Decisions taken during the round-3 review of main at `6cb47ff`. The round used Matt Pocock's `code-review`, `to-spec` and `to-tickets` skills. The review, spec and tickets are in the project files under `research/code-review-3/`:

- `review.md`
- `spec.md`
- `tickets/01-26`

The project lead asked that any choice made on their behalf be written down here with the reason, so there is an audit trail of the assumptions and decisions.

| # | Decision | Why |
|---|---|---|
| D1 | Fixed point is the empty tree, so the review covers every tracked file at `6cb47ff`. | The skill reviews a diff since a fixed point; tp53 asked for the entire codebase, and the empty tree is the fixed point that makes the diff equal the whole repo. |
| D2 | Ran the two axes (Standards, Spec) per area: five areas, ten sub-agents, instead of one pair. | About 100k lines of first-party code does not fit one sub-agent's context. The axes stay separate, as the skill requires. |
| D3 | Excluded `kotaemon-main/` (vendored), `android-app/app/build/` (build output), `cache/`, `out/`, and the deck and video generators. | None of it is code the project writes or ships to participants. |
| D4 | Raised each sub-agent's word limit from 400 to 500. | Each report covers a whole area, not one diff. |
| D5 | Standards sources: AGENTS.md, CLAUDE.md "Code Quality Standards", CONTEXT.md glossary, ADRs 0001-0007, HANDOVER.md working agreement, docs/decisions/latency.md. | The repo has no CODING_STANDARDS.md or CONTRIBUTING.md; these are the files that say how code should be written. |
| D6 | Known items were given to every sub-agent as "do not re-report" (latency decisions, round-2 plan, handoff keywords, voice thinking level). | Avoids re-raising work that is already decided or waiting on the clinical lead. |
| D7 | Tracker: one Markdown file per ticket in the project's shared folder, as the latency round did, and the spec as a Markdown file beside them. | `/setup-matt-pocock-skills` was never run, so no tracker is configured; the latency round set this convention and tp53 accepted it. |
| D8 | Skipped to-spec's "check the seams with the user" and to-tickets' "quiz the user" steps; the chosen seams and breakdown are recorded here instead. | tp53 said to go with Claude's judgement when unsure and keep an audit trail. |
| D9 | Test seams: the Dose Boundary's `apply` as the one answer gate every outgoing answer passes through, and the FastAPI app through a test client. Android keeps its repository-with-fake-API seam. | to-spec asks for the highest seams and as few as possible. Both already exist and have tests built on them (the dose-guard, text-path and mobile-ownership tests), so no new seam is needed. |
| D10 | Reminders name the medicine and the time, never a dose, even a dose the participant or a prescriber entered. The clinical lead is asked to confirm. | ADR 0004 says the system never gives a specific dose, and Android ticket 07 says the stored reminder text never contains one. Reading back a user-entered dose is ambiguous under the ADR's carve-out, so the safer reading ships until the clinical lead decides. |
| D11 | Phone providers that send audio or text outside India (Bolna's Deepgram, OpenAI and Cartesia; Retell's analysis) are off in the study release unless the release record lists them. | Identifiable data must stay India-hosted. None of these providers appears in ADR 0007's residency table or the release record, so off is the default that cannot break the rule. Turning one on becomes a deliberate release-record change. |
| D12 | Built-but-unused subsystems (memory agents, meta agent, Context-1, the alert manager's fake "sent" paths, the dead Live provider) are removed or not constructed, rather than wired in. | Wiring them in would change answers mid-study and needs its own spec. Leaving them in place keeps code that claims behaviour it does not have. |
| D13 | 26 tickets, with the language work run as expand–contract (ticket 14 expands the table and moves WhatsApp over; ticket 15 migrates the other channels and deletes the old tables). The prefactor that merges the dose detectors is folded into ticket 01. | to-tickets wants vertical slices, with wide refactors run as expand–contract. The language tables are spread across about 12 files, so one change cannot stay green. |
| D14 | Out of scope: the mechanical type-hint, docstring and broad-except sweep outside the touched paths, and splitting `simple_rag_server.py` into routers. | Neither changes behaviour, and the split would raise review cost mid-study. Both are recorded in the review for a later round. |
| D15 | The Android comment density versus AGENTS.md "no in-code comments" is surfaced and left for the project lead, not changed. | The comments carry the ADR reasoning. Which rule wins is the project lead's call, not a review fix. |
| D16 | Corrected project memory: Retell is dose-guarded (latency ticket 02), so "Retell lacks the dose guard" is no longer an open item. | `retell_integration/custom_llm_server.py` calls `guard_query_result` on every turn. |

## Implementation decisions

Decisions taken while implementing the 26 tickets with Matt Pocock's `implement` skill, one ticket per sub-agent in its own worktree, merged onto the wave's branch as `implement-spec` describes.

| # | Decision | Why |
|---|---|---|
| D17 | Changes that tighten safety or fix a correctness defect (a leaked dose, a missing check, a wrong language, an invented distance, a wrong release id) ship on, with no setting that restores the old behaviour. A change that adds a new answer-shaping choice goes behind a setting that is off by default, unless its ticket says it ships on. Every participant-visible change is listed in the release notes below. | A setting that can switch the Dose Boundary back to its leaky form is a way to give a dose, which ADR 0004 forbids. The running study is protected by the versioned release, not by settings: nothing reaches participants until a new release record is approved, and drift detection reports a build that is not the approved one. |
| D18 | One pull request per dependency wave (five waves). Each PR is merged as soon as its CI is green, before the next wave starts. Every ticket inside a wave is its own commit. | tp53 asked for this on 2026-10-15. Merging each wave before starting the next means no PR is stacked on another one, and each wave builds on a main that is already green. |
| D19 | The seams are the ones D9 recorded; no sub-agent asks to confirm them. | The `tdd` skill asks for seams to be agreed before tests are written. D9 agreed them in the spec. |

### Wave 1 (tickets 01, 05, 07, 08, 10, 11, 14, 19, 20, 21)

| # | Decision | Why |
|---|---|---|
| D20 | (01) One module, `dose_detector.py`, owns every dose pattern; a test fails if another module compiles a dose-matching regex. New shapes fold into the five existing reason names. | Study rows store the reason names, so they stay comparable across releases. |
| D21 | (01) Indic vocabulary words match only as whole words (no letter or combining mark beside them), and every inflected schedule form the old substring match caught is listed explicitly. Unit words stay open on the right for suffixes. | Substring matching redacted ordinary Hindi, Marathi and Punjabi sentences ("के बारे में", "घरात", "ਪਰਿਵਾਰ"). Listing the inflections keeps the boundary at least as strong as before. |
| D22 | (05) Every route on the main server has a row in `auth/policy.py` (public, participant, admin or own verification). A route with no row is served admin-only and fails a test. Admin tokens are signed with a separate `ADMIN_JWT_SECRET`, and browsers sign in with HTTP Basic, using the admin token as the password. | A mobile token that claims the admin role cannot become an admin. Basic sign-in lets the Gradio panel and study dashboard work without a new login page. |
| D23 | (07) An unset webhook secret returns 403 and a bad or missing signature returns 401. `/api/set_language` uses Twilio verification. Sarvam uses HMAC-SHA256 over the body with `SARVAM_WEBHOOK_SECRET`. | The ticket allows either status code, and 403 tells the operator the problem is server configuration. The repo had no Sarvam signing scheme. |
| D24 | (08) The refresh token goes in the request body, and the session ends at PIN sign-in plus 144 hours whatever the number of refreshes. The PIN lock lifts 15 minutes after the last wrong try. | Without a fixed end, refreshing would keep a session alive forever without the PIN. The lock time is a constant, not a setting, because it is a safety fix. |
| D25 | (10) The corpus snapshot is a hash of the document manifest. The record's `knowledge.snapshot_sha256` is left unrecorded, so it shows as drift until the PI records it. | The tracked manifest (7 documents) does not match what the record describes (3 PDFs), so writing the live hash in would approve a corpus nobody reviewed. |
| D26 | (11) The index-mismatch refusal is returned as a normal answer, marked with `refusal_reason: "index_mismatch"`. It is checked after the emergency check, and recovery needs a restart. | Channels replace any non-success status with their own fallback, and Gemini Live would then answer from its own knowledge. An emergency still gets its escalation. |
| D27 | (14) The Supported Language table is a Python module. Odia's code is `or`, and Sarvam's `od-IN` is kept as an alias. WhatsApp speaks through Sarvam (India-hosted) first and falls back to Edge. A typed message with no `/lang` goes by its script. | It fails at import if malformed and needs no loader. The ticket asks for the India-hosted voice where one exists. |
| D28 | (19) A mode with no retrieval distance gives zero retrieval credit to the evidence badge. | Any stand-in value would overstate the evidence. |
| D29 | (20) The vendored `kotaemon-main/` tree, its entry points and `setup.py` are deleted. The CI count check now reads the real total and fails below a floor of 1000. | Once the entry points were gone, nothing loaded the tree. `pip install .` would have run the Kotaemon installer. The old count check read a per-file line and never saw the total. |
| D30 | (21) Every error body is `{"status": "error", "error": ...}`; an unhandled error says only "Internal server error". A torn metadata file stops the server from starting. | Exception text can carry paths or internal detail. Failing loudly is what the spec asks for. |

**What participants and operators will notice (wave 1, all need a study release):**
- Doses written in words, spelled-out units, counts of tablets, puffs or drops, "N-hourly", q4h or BD, and Hindi एमजी are now withheld in every Supported Language. "Stage IV" sentences are no longer wrongly withheld. Ordinary Indic sentences containing words like "बारे" are no longer withheld.
- While the embedding index doesn't match the release's model, every channel says the service is temporarily unavailable, in the participant's language.
- WhatsApp speaks every language in its own Sarvam voice instead of Edge or Hindi. Bengali and Gujarati voice notes go to Live in their own language. A romanised (Latin-script) message with no `/lang` set is answered in English only.
- PageIndex and hybrid evidence badges use real distances, so PageIndex answers usually show a lower evidence level. A long WhatsApp answer ends with the whole truncation notice.
- A locked mobile account can sign in again 15 minutes after the last wrong PIN, and the app asks for the PIN again at most six days after sign-in.
- Operators must set `ADMIN_JWT_SECRET`, `BOLNA_WEBHOOK_SECRET` with `RAG_API_KEY`, `RETELL_API_KEY`, `SARVAM_WEBHOOK_SECRET`, and `TWILIO_AUTH_TOKEN` with `PUBLIC_BASE_URL`, or the matching routes refuse every request. CORS allows no origins until `server.cors_allowed_origins` names them.
- Check before go-live: Bolna's custom function calls to `/api/bolna/query` must carry both a signature and the Bearer key. If Bolna does not sign those calls, they will be refused.

**For the clinical lead's review:** the index-unavailable message (ten languages), the WhatsApp "voice unavailable" and "language set" messages (eleven languages) and the English unsupported-language notice, all machine-drafted, and the Indic schedule words added to the dose detector.

### Wave 2 (tickets 02, 04, 06, 09, 15, 22)

| # | Decision | Why |
|---|---|---|
| D31 | (02) The Emergency Override still lets the dose through, with the 108 guidance put first, in the participant's language. Only a CRITICAL alert on the question triggers it. | This is what CONTEXT.md, ADR 0004 and the ticket say. Whether doses should also be removed under the override is the clinical lead's call. |
| D32 | (04) All eleven languages get their own Deferral and Redacted Answer note. The Hindi Deferral's "अगली बार" became "अगली मुलाक़ात", and new copy avoids the Indic words for "time" and "day". | The detector reads those words as dose frequencies, so the old Hindi Deferral contained a dose match of its own. |
| D33 | (06) Reminder text is built in one place (`reminder_text.py`). Doses are cleaned span by span, including from the medicine name. A mobile reminder needs a phone number from the caller. | Running the reminder through `DosageGuard.apply` would replace the whole reminder with the Deferral. A name can carry a dose too ("Paracetamol 500mg"). The token carries no phone number. |
| D34 | (09) The household is the authenticated `user_id`. An auditor is anyone holding an admin token. Sync push acknowledges each item by (kind, index). Reminders reach the app without the dose. | A participant chooses their own mobile role at registration, so that role cannot grant the full export. Logs and vignettes can arrive without an id. Sending the stored dose would open a new dose path, which D10 rules out. |
| D35 | (15) Bolna and Retell get voices only for the languages the old tables really voiced (hi, en, mr, ta). Every other language raises an error instead of falling back to Hindi. Live does not speak Odia or Punjabi. | No Cartesia voice ids were invented. An operator should check whether more Indic voices exist before adding them to the table. |
| D36 | (22) The Android app registers with a PIN, keeps its tokens encrypted with an Android Keystore key, and refreshes once when it gets a 401. A failed refresh sends the participant back to the PIN only when the server refuses it, not when the network is down. The mobile voice socket now needs an access token. The release base URL defaults to `https://api.pallisahayak.org/`. | Being offline is normal at the study sites. androidx.security is deprecated and adds to the APK size. The host is the one the Android spec names; confirm it before a release build. |

**What participants will notice (wave 2, all need a study release):**
- An answer that says "call 108" no longer lets a dose through. Every CRITICAL question gets the 108 guidance first, in the participant's language, on every channel.
- Marathi, Kannada, Malayalam, Odia, Punjabi, Telugu and Gujarati participants get the Deferral and note in their own language. A Hindi web question gets the Hindi Deferral.
- Reminder calls and WhatsApp reminders name the medicine and the time, never a dose. The Hindi, Tamil and Gujarati call closings say "take care".
- The app's reminder list and sync now return the household's own reminders, without the dose. A participant's export shows only their own records.
- Assamese leaves the web voice picker. Live refuses Odia, Punjabi and Assamese plainly instead of answering in English. Bolna no longer builds Punjabi or Malayalam agents that spoke Hindi.
- The Android app shows a refusal screen for an unapproved release, then a language picker, Care Role and PIN. Its new strings are English only so far.
- Over-redaction to know about: the detector treats bare Indic day words (दिन) as a schedule, so some ordinary sentences are withheld. This errs on the safe side and is listed for the clinical lead.

**For the clinical lead's review:** `dosage_guard.EMERGENCY_GUIDANCE` (mr, te, kn, ml, pa, or), the Deferral and note copy for mr, kn, ml, or, pa, te and gu, the two reworded Hindi lines, the reminder lines and closings, and the 11 voice greetings.

### Wave 3 (tickets 03, 12, 23, 24)

| # | Decision | Why |
|---|---|---|
| D37 | (03) Every route that returns model- or graph-derived text passes the Dose Boundary, which guards every string of a structured payload, not only an "answer" field. The Emergency Override applies to the answer field only. The route table marks the rest as `NO_ANSWER_TEXT` (stored records, the caller's own transcripts, metadata). | A dose can sit in a KG entity or a community report as easily as in an answer. Withholding a dose in a side field under a CRITICAL question errs in the safe direction. |
| D38 | (03) The Gradio admin tabs call the route functions in process instead of over HTTP to localhost. | Since ticket 05 those HTTP calls carry no admin token and got 401, so every tab showed "Error". |
| D39 | (12) The study release record names the Live model, and nothing else can choose it. Config, the client and `set_model` are refused. `GEMINI_LIVE_MODEL` stays as a development override that shows as drift. The record's `gemini-3.8-live` was corrected to `gemini-3.1-flash-live-preview`. | The service never ran `gemini-3.8-live` and would refuse to start with it. HANDOVER's verified round trip is on `gemini-3.1-flash-live-preview`. **The PI and technical lead must confirm this model id**, because the record is what they approve. |
| D40 | (12) Web voice holds each Live turn's audio until the turn is complete, then checks the transcript at the Dose Boundary. A blocked turn releases the Deferral, and audio with no transcript is withheld. Web voice sessions are keyed to the token or are anonymous; the frame's `user_id` is ignored. Translator mode is deleted. | Live's audio and transcript are not aligned, so only the whole turn can be checked, as on WhatsApp. **This adds latency: speech starts only when the turn ends.** |
| D41 | (23) The Android client trusts a CRITICAL from the server and takes the higher of the two levels. Its dose detector mirrors the server's rows. Offline it answers from a downloaded bundle and queues counts-only interaction logs. Sync runs in a process-scoped coroutine, not WorkManager. | ADR 0004 says the server decides the emergency. WorkManager can't be fetched here and costs APK budget; the trade is that a push waits for the app process. |
| D42 | (23) The release status is cached per build with no expiry, so an approved build opens with no signal. | Being offline is normal at the study sites. A revoked release keeps opening offline only until the phone next has signal, and offline it can only answer from the bundle. |
| D43 | (24) Settings has a working Active Role switch and language list. The chosen language sets both the answer language and the screen language. Hindi strings ship now; the other ten languages follow the same pattern. Odia goes out as `or-IN`. | Recreating the Activity would re-run the release gate, which refuses offline. Machine drafts of nine more scripts would only add to clinical review. |

**What participants and operators will notice (wave 3, all need a study release):**
- Mobile patient-memory answers and insights, and admin KG, GraphRAG and PageIndex output, now withhold doses. Those responses gain a `safety` field, and the mobile ones accept a `language`.
- Temporal recommendations no longer suggest an opioid rotation; that wording now defers to the care team.
- Web voice: a turn that speaks a dose plays the Deferral instead. Speech starts only when Gemini finishes the turn, which makes every answer slower to start. The web page can no longer pick a model or use translator mode.
- The Android app shows the 108 banner for emergency questions in every language and for any question the server flags CRITICAL. Offline, saved questions get their saved answer, labelled as saved, and an approved build opens with no signal.
- The Android Settings screen switches Active Role and language. On a Hindi phone, or with Hindi chosen, every screen is in Hindi.

**For the PI and technical lead:** confirm the Live model id in `study/release.yaml` (D39), accept or reject the web voice latency trade (D40), and reconcile ADR 0007 and HANDOVER line 76, which still say "3.8 Live". Commit the Room schema `4.json` from the first real Android build.

**For the clinical lead's review:** the 76 machine-drafted Hindi app strings in `android-app/app/src/main/res/values-hi/strings.xml`, and the reworded temporal recommendations (including whether "continue the current regimen" still reads as a start/stop judgement).

### Wave 4 (tickets 13, 16, 17, 18)

| # | Decision | Why |
|---|---|---|
| D44 | (13) Every channel (web, web voice, WhatsApp, Bolna, Retell, mobile text and voice) writes one study row per answer through `study_logging.record_answer`, with `voice_path` (text, live, fallback, cache, phone) and the Dose Boundary outcome. Callers with no participant identity are recorded as `"anonymous"`, not as a pseudonym. Retell's row is written when its streamed turn ends; every other channel writes before sending. | A refusal that no row records can't be counted, and a per-session pseudonym would look like a participant in the adoption counts. |
| D45 | (16) Live gets 4 s to start and 8 s to finish a turn (ADR 0007's constants). On a miss, the Live session is closed and Sarvam answers. The operator's forced-Sarvam override now governs web voice, WhatsApp voice notes and the mobile routes. The fallback rate counts only Live failing on its own. | A stalled session used to be waited on for ever. Closing it, not reusing it, means its late audio can never reach the caller. The deadlines stay out of release.yaml so the fingerprint is unchanged. |
| D46 | (16) A web voice turn that the Dose Boundary withholds is now spoken, on the Sarvam voice, in the session's language. A language with no Sarvam voice stays text only. | Before, the caller heard silence and only the web page showed the Deferral. |
| D47 | (17) Bolna and Retell serve only when the release record lists every outside service they use and the provider is switched on. The default release lists none, so both refuse with a 503 that names the missing services. Turning one on without listing it shows as drift. | This is D11's recorded default. The platforms and their analysis services count as outside India because their hosting is unknown. |
| D48 | (17) Bolna's LLM is now this server, through Bolna's custom-LLM option, and the unguarded `/api/bolna/query` route is deleted. Retell post-call analysis is off by default. | Bolna's hosted OpenAI LLM rewrote answers after the Dose Boundary and had no hook to check them. |
| D49 | (18) In the kg, graphrag and fused modes, a non-English question is translated to English for the graph retrievers only; vector search keeps the question as written (ADR 0003). It ships on, with no setting. | Graph entities are English, so Indic questions got no graph facts at all. A wrong-language search is a correctness defect (D17), and the graph modes are already off by default. |
| D50 | (18) GraphRAG's own model calls are off unless every model is pinned to Vertex asia-south1. With the shipped settings, searches answer from the local index and indexing is refused. `graphrag==3.3.0` and the chromadb and sentence-transformers versions are pinned. | The shipped settings use the global Gemini API endpoint, which is not region-pinned. The study needs a fixed release. |

**What participants and operators will notice (wave 4, all need a study release):**
- Researchers get a study row from every channel, with the voice path, and mobile voice refusals are now counted.
- Web voice: if Live goes quiet, the caller hears a Sarvam answer after 4 to 8 seconds instead of silence. A withheld turn is now spoken. A forced Sarvam override now really applies to web and WhatsApp voice.
- Bolna and Retell phone lines, and their reminder calls, refuse in the default release.
- In the graph modes only, Indic questions now get graph facts in the answer context.
- Admin GraphRAG searches answer from the local index, and indexing is refused.

**Before a release that turns on a phone provider:** list its outside services in `study/release.yaml`, and register the custom LLM in Bolna (model `palli-sahayak-guarded`, one URL per language, `RAG_API_KEY` as the key). That registration was built from Bolna's documentation and has not been tried against a live Bolna account.

**For the clinical lead's review:** nothing new to translate. The Bolna and Retell decision (D11) still waits on them.

### Wave 5 (tickets 25, 26)

| # | Decision | Why |
|---|---|---|
| D51 | (25) One Care Role vocabulary (`asha_worker`, `family_caregiver`, `patient`) on the server and the app. Registration and the mobile question routes refuse any other value with 422. A study row records the Active Role the app sends, else the registered role; a role outside the three is recorded as unknown (`""`), never mapped. | Mapping "caregiver" or "healthcare_worker" onto a Care Role would be a guess in the study record. The app only sends the three, so the refusal cannot push a phone into its offline fallback. |
| D52 | (25) Keyword role detection is deleted, the sync default of `asha_worker` is gone, and observations are attributed to the speaker's role instead of "patient". | Nothing called the keyword detector, and a guessed or defaulted role corrupts who asked what. |
| D53 | (26) Nothing claims a notification that did not happen. "A human caregiver has been notified" is removed from the en and hi CRITICAL and HIGH actions (the 108 guidance stays). The care-team notify route, the admin Notify tab and `AlertManager.send_alert` are deleted. | Each one reported a notification as sent after only a log line. |
| D54 | (26) Start-up no longer builds the memory agents, the meta agent or Context-1 (D12); their packages stay. Patient insights answer empty and query-memory answers 503. `/feedback` stores feedback with no LLM call. Dead code is removed: the unused Live provider in `voice_router.py` and the WhatsApp Cloud API handlers (with their two dose tests, so the evidence record lists 83 server-side dose tests instead of 85). | D12 asks only that they are not constructed. Deleting a dead path deletes the tests that drove it; no live path lost a dose test. |
| D55 | (26) Adherence is counted per patient: reminders carry an optional `patient_id`, and stats filter to it under the caller's own id. | The route passed the patient id as the user id, so stats were always empty. |

**What participants and researchers will notice (wave 5, all need a study release):**
- Emergency replies in English and Hindi no longer say a caregiver was notified.
- The app's adherence screen shows real counts per patient. App feedback is stored.
- Patient-memory answers in the app are unavailable (503) and insights are empty, since the memory agents no longer start.
- Study rows record the Active Role; a role outside the three reads as unknown. Registering with any other role is refused.
- Admin: the Notify tab is gone.

**For the clinical lead:** the handoff copy still says "A healthcare professional will contact you shortly", but no one is assigned to handoff requests (see the 2026-10-15 handoff decision). That promise needs an owner or a rewording.
