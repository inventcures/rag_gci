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
