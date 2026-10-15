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
