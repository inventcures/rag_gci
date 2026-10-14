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
