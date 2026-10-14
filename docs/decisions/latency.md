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

## 04: Mobile shows only the caller's patients

- **Ownership = the care team.** A patient belongs to the app users whose user
  id is a `provider_id` on its care team. No new field, and the admin UI's
  care-team tab is already the way to assign a patient.
- **A write to a new patient id makes it the writer's; a write to another care
  team's patient is rejected** (sync push counts it as rejected; POST
  `/observation` answers 404).
- **Records with no app user on the care team are visible to no app user.**
  This hides records created before this change from the app until an admin
  adds the ASHA Worker to the care team. Claiming them on first write would let
  any user take over any existing patient. Revisit if pre-study app records
  need migrating (the store was empty in this repo).
- **404, not 403, for another care team's patient**, so the API does not
  confirm the id exists.
- **Every per-patient endpoint checks ownership**: patient detail, care team
  (read and add), adherence, FHIR export, insights, memory query. FHIR import is
  not checked yet: it writes, it does not read, and it needs its own decision on
  who owns imported patients.
- **One store.** The app now gets the answer path's `LongitudinalMemoryManager`
  (`data/longitudinal`) instead of a second instance on
  `data/longitudinal_memory`. Two instances also had two in-memory caches that
  could disagree.
- **Delta by file modification time** (2-second slack), then per-observation
  `timestamp > last_sync_at` as before. An observation pushed late with an old
  timestamp is still not pulled by another device; that was true before and
  needs a per-observation modified time to fix.
- **Saves are atomic** (write to `.tmp`, then rename) so the sync scan never
  reads half a file.
- Also fixed in passing: GET `/patient/{id}` called a method that did not
  exist, and POST `/observation` called `add_observation` with the wrong
  arguments. Both always failed.

## 05: Release status reflects the approved record

- **The endpoint serves the loaded release singleton** (`get_study_release()`),
  the same object the server stamps onto every interaction, instead of an
  empty `StudyRelease()`.
- **Drift is cached on the release.** `verify_on_startup` computes it (it
  already ran at boot), `client_view` reuses it, `refresh_drift()` or
  `reload_study_release()` recomputes. A per-request git call and Chroma client
  on an unauthenticated endpoint was both slow and an easy way to load the
  server.
- **Still unauthenticated**, as before: the app needs the gate before it has a
  session, and the payload carries no participant data.
- **Gate behaviour change, study team sign-off pending.** With today's
  `release.yaml` (status `draft`, approvals unsigned) the app sees exactly what
  it saw before: not usable with participants. The difference appears only once
  the record is activated and signed: the app will then open, where before it
  never could. That is the intended behaviour, but the ticket asks the study
  team to confirm it, so it is flagged on the PR.

## 06: Embed once, off the event loop, with one model

- **Search by `query_embeddings`, embedded by the collection's own embedding
  function in a worker thread** (`retrieval/embedder.py`). Same function, same
  vectors as `query_texts`; a test on a real Chroma collection checks ids and
  distances are identical.
- **One bge-m3 instance.** Chroma's `SentenceTransformerEmbeddingFunction`
  caches models by name in a class dict; the server seeds it with the model it
  already loaded (from `EMBEDDING_MODEL_PATH` when set), so the collection's
  function no longer loads a second copy from the hub. The function's persisted
  config still names `BAAI/bge-m3`, so the existing collection opens without an
  embedding-function conflict. Expected saving is one model copy (about 2.2 GB
  for bge-m3 in fp32); not measured here because the hub is blocked in this
  environment and bge-m3 cannot be downloaded.
- **The probe is the main search.** A non-English question is searched once
  with `top_k`; the probe reads the top distance from that result and the vector
  path reuses it. Translation triggers on the same threshold (1.5) and the same
  top-1 distance. The one theoretical difference: Chroma's HNSW search with k=5
  explores at least as many candidates as k=1, so its top-1 can only be equal
  or closer. On a corpus of a few hundred chunks HNSW is effectively exact.
- **Exact-text cache, 512 entries, in process.** Covers repeated questions and
  the routed methods' second look at the same text. Not shared across workers
  (ticket 18).
- **Removed per-query debug work**: a `vector_db.count()` SQLite call and a list
  of every document id, run on every query for a DEBUG log line.
- **The Live classifier's embedding moved off the loop** (`should_query_rag` in
  a thread). It already shared the pipeline's model.
- **Not verified here:** the retrieval harness's hit@5 in the eleven languages
  (needs bge-m3 and the corpus). By construction the vectors are identical; run
  `evaluation/retrieval_eval` once on a machine with the model to confirm.

## 07: Side work runs after the reply

- **`post_work.after_reply(name, fn)`**: supervised asyncio tasks, held so they
  are not collected mid-flight, each in a `post.<name>` span of the request's
  trace. Failures are logged and counted; `/health` shows the counts. Shutdown
  drains them (10 s). Chosen over FastAPI `BackgroundTasks` because the answer
  path is shared by sockets and webhooks, not only HTTP routes.
- **Moved after the reply:** conversation history, observation extraction, the
  answer-path record (the retrieval audit row), the shadow-mode KG check, and
  the study logger's analytics and expert-sampling fan-out.
- **Stays before the reply:** the study interaction row
  (`StudyInteractionLogger.write`), the Dose Boundary, and KG checks in `flag`
  and `block` mode, which can change the answer. The retrieval and K26 record
  is still attached to the result before the reply; only writing it moves.
- **Shadow KG check:** the reply carries `{"mode": "shadow", "status":
  "pending"}`; the audit row gets the finished check.
- **Observation extraction reads the question only** (`Patient: <question>`).
  Extracting from the generated answer recorded the assistant's words as the
  patient's observations. Nothing found means nothing written: no record is
  created.
- **Conversation history is an append-only JSONL** next to the old JSON file
  (`conversations.jsonl`), written from a worker thread. The legacy JSON is
  still read at startup, then the log. Memory keeps the last 50 turns per
  conversation; the full history stays on disk.
- **Per-call ids are marked ephemeral** (`participants.ephemeral()`, prefix
  `session:`): Retell calls without a number, Bolna call ids, mobile voice
  sessions without a signed-in user, Gemini Live sessions without a known
  person. They are still traced and logged, but get no longitudinal record and
  no comprehension profile. Gemini Live sessions opened through the session
  manager now carry the person (the WhatsApp number), matching the text path.
  Mobile voice now keys on the signed-in user, matching mobile text.
- **Comprehension profiles are saved from a timer thread, coalesced (1 s), and
  atomically**, instead of rewriting every profile on the event loop per
  message.
- **Longitudinal saves lock per patient and write in a worker thread.**
- **`observations_extracted` is gone from the query result**: it is not known
  at reply time, and nothing read it.
