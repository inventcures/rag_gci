# Latency work: decision log

Decisions taken while implementing the latency tickets
(`research/latency-tickets/` in the project files; plan in
`research/latency_review_and_plan.md`). The project lead pre-approved the plan's
recommendations and asked that any choice made without them be written down
here, with the reason, so it can be revisited.

Each entry: what was decided, the alternatives, and why.

## 01: Per-stage traces

- **Hand-rolled spans written as JSONL, not the OpenTelemetry SDK.** The span
  fields mirror OpenTelemetry's (trace id, span id, parent, name, duration,
  attributes) so an OTLP exporter can be added later without touching call
  sites. Adding the SDK now would add three dependencies and an exporter
  process for what is, today, "append a line to a file". Revisit if traces need
  to go to a hosted backend.
- **One file per UTC day under `data/traces/` (`TRACE_DIR`), git-ignored.** Same
  shape as the retrieval audit, so the same retention handling applies.
- **Attributes are allow-by-type, deny-by-name.** Values must be numbers,
  booleans or strings of at most 64 characters, and any key containing
  question, answer, query, text, transcript, prompt, message, content, phone,
  number, name, email, address, user_id, patient_id or caller is dropped. This
  is deliberately over-eager: a dropped timing attribute costs nothing, a
  leaked question costs the study. The participant is the study pseudonym.
- **Channel from the HTTP path for requests; one trace per turn for sockets.**
  A Retell or Live socket lasts a whole call, so a socket-level trace would be
  one 10-minute span. The voice handlers open a trace per turn instead.
- **PHI-bearing INFO logs moved to DEBUG** (question text, answer text, phone
  numbers, message bodies, chunk metadata). No study tool reads these stdlib
  logs; the study rows come from the study log and the retrieval audit.

## 02: Retell answers every turn, dose-guarded

- **The handler owns `accept()`.** The endpoint accepts only on the
  "unavailable" path, to send the error frame. The handler takes an
  `accepted=` flag for callers that already accepted.
- **Dose guard = `guard_query_result`, before `optimize_for_voice`.** It is the
  same call Bolna and the web API make, so all channels share one boundary. A
  guard exception becomes the spoken error message, never the draft.
- **Turns run as tasks; a newer `response_id` cancels the older turn.** Retell
  sends a new `response_required` on barge-in. Answering the old one would
  speak a stale answer over the caller.
- **Dropped `conversation_context=`** rather than adding the parameter to
  `query()`. The pipeline already keys history on `conversation_id`, and
  widening `query()` would change every channel.
- **Not done here:** an acknowledgement chunk before the answer, and streaming.
  Both belong to ticket 13.

## 03: Each Bolna caller is their own participant

- **Identity order: `from_number`, then `execution_id`, `call_sid`, `call_id`.**
  The number keeps one participant across a caller's calls, which is what the
  study's per-participant rows and the conversation history want. A call id is
  the fallback so one call at least stays together.
- **No identity means a one-off id (`bolna_anon_<random>`), never a shared one.**
  Losing history for an unidentified call is cheaper than mixing two callers.
- **The agent config sends `from_number` and `execution_id`** as Bolna system
  variables (`%(var)s`). Agents created before this change keep the old params
  until they are recreated; until then their calls land on the one-off id. An
  unfilled `%(...)s` value is treated as missing.
- **Existing pooled `bolna_bolna_call` records are left as they are.** They
  cannot be split after the fact.
- **Not verified against a live Bolna call** (no Bolna key in this
  environment). Check one call's request body once a key is available.
