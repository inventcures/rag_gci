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

## 08: WhatsApp acks fast and runs its steps concurrently

- **Both `/webhook` routes acknowledge Twilio immediately** and hand the message
  to `post_work.after_reply`. Each message gets its own `whatsapp` trace.
- **Dedupe on Twilio's `MessageSid`**, last 4,096 ids in memory. In-process
  only: with several workers (ticket 18) this needs a shared store.
- **Message order is kept** (English, then the user's language, then audio),
  but the translation starts while the English text is being sent, and speech
  synthesis starts while the translated text is being sent. The voice-note
  confirmation is sent while the answer is drafted. All fixed one-second
  sleeps are gone. Twilio's synchronous client runs in a worker thread.
- **Live voice notes:**
  - The reply is collected until `TURN_COMPLETE` (or `INTERRUPTED`), within
    `WHATSAPP_LIVE_REPLY_DEADLINE_S` (default 20 s). The old loop compared
    against string markers that never matched, so it ran until the session
    closed. The note is ended with `audio_stream_end` so Live does not wait for
    silence.
  - Grounding is the guarded answer (after `guard_query_result`), sent as
    context with `turn_complete=False`, so Live does not reply to the grounding
    itself. It used to inject `context_used`, which is a chunk count.
  - **Live's audio is sent only if its output transcript passes the Dose
    Boundary.** If the transcript has a dose, or there is no transcript or no
    audio, or the deadline passes, the guarded text is sent and spoken with TTS
    instead. Audio cannot be redacted, so unverifiable audio is never sent.
  - Transcription failure now goes straight to the traditional pipeline:
    without a transcript the reply cannot be grounded or checked.
  - No separate translation of Live's reply: Live already speaks the user's
    language.
- **Not verified against Twilio or a live Gemini session** (no Twilio keys
  here). Covered by simulated-session tests.

## 09: Offline bundle built once and cached

- **Bundle key = hash of language, release id, corpus version and the question
  set.** Corpus version is the indexed document ids with their chunk counts,
  so a re-index, an added or a removed document rebuilds. The key is the ETag.
- **Kept in memory and on disk** (`data/cache/mobile/bundle_<lang>_<key>.json`),
  so a restart serves the file instead of rebuilding. Older files for the
  language are deleted when a new one is written. Writes are atomic.
- **One build per language at a time** (per-language lock); concurrent first
  requests wait for it. Within a build, four answers are drafted at once
  instead of twenty in a row.
- **Build answers use the participant id `system:offline_bundle`.** The
  retrieval audit gets a `kind` field: `build` for these, `interaction` for
  everything else. System ids get no per-person records.
- **Still built on first request, not at startup.** Startup building would
  spend twenty answers per language on every deploy, including languages no
  one requests. Revisit if the first download's wait matters in the field.

## 10: Uploads index in the background

- **`POST /api/upload` returns a job id at once; `GET /api/upload/{job_id}`
  reports queued, running, done or failed.** This changes the response: it no
  longer carries the indexing result. Nothing in the repo read it; the docs
  are updated. Revisit if an external script relies on the old shape.
- **Jobs run one at a time.** Two guidelines embedding at once would double
  the CPU taken from live calls, and Chroma writes serialise anyway. The last
  200 jobs are kept for status queries, in memory: a restart forgets them.
- **Blocking work runs in worker threads:** copy and parse, then embedding in
  batches of 32 chunks, then persist, verify and the metadata save. Each batch
  is one `vector_db.add`, so the GIL is released between batches. Chunk ids
  and metadata are identical to a single add.
- **Stored filenames are sanitised:** directory parts and `..` are dropped,
  unusual characters become `_`, Indic letters are kept, and a random prefix
  keeps repeated names apart.
- **The Gradio admin upload is unchanged:** it already runs in Gradio's worker
  thread with its own loop.
- **Not measured with real bge-m3** (the model cannot be downloaded here). The
  loop-blocking test uses a slow fake parser and collection; the worst stall
  stays under 50 ms.

## 11: Thinking level and deadlines

- **Measured against the live model** (`scripts/benchmark_thinking.py`,
  results in `docs/latency/thinking-11.md`). The default thinking level had no
  token headroom: production answers were cut off mid-sentence after ~11.5 s.
  `max_tokens` now means visible text; thinking headroom is added on top
  (512 / 2048 / 4096 / 8192 by level).
- **Answers: `low` on text, `minimal` in a voice turn.** `low` matched `high`
  in judged quality. `minimal` is the only level that fits the voice deadline,
  and it lost to `high` on 28 of 40 judged comparisons. We take that cost on
  voice because the alternative is the timeout fallback on most voice turns.
  Ticket 12 changes the voice prompt anyway; **a clinician review of voice
  answers is required before the release is activated.** Revisit if the
  review finds minimal-level voice answers unsafe or unhelpful: the option
  then is `low` with ticket 13's streaming, which starts speaking at ~7.5 s.
- **Translation and clarification: `minimal`.** Query translations were
  judged equivalent at every level.
- **Deadlines:** voice turn 7 s (`LLM_VOICE_DEADLINE_S`), query translation
  3 s, answer translation 8 s (WhatsApp only; `minimal` p95 4.5 s, so 3 s
  would have failed most), everything else 45 s. The fallback provider gets
  what is left of the same budget, not a fresh 45 s.
- **Voice channels:** Retell, Bolna, the Gemini Live grounding tool and
  post-turn query, and mobile voice run their answer inside
  `within_voice_deadline`. A turn past it gets the channel's existing spoken
  fallback (Retell timeout message, Bolna partial answer, Live "no
  information", mobile voice fallback answer).
- **A failed generation no longer reaches the caller as text.** `query()` used
  to return "Error generating response: ..." as the answer. It now returns a
  fixed "could not prepare an answer, ask again or speak to your doctor"
  answer, with `model_used: "error"` kept for the record.
- **Release record:** thinking levels and deadlines are in the fingerprint
  and in `data/study/release.yaml`, marked pending study approval. This
  ticket changes answer content and ships in the next versioned release
  together with 12-14.
- **Not done here:** time to first token in the request trace. Calls are not
  streamed until ticket 13, so the `llm` span measures the whole call (with
  `thinking` and `budget_s` attributes). The benchmark measured first token
  by streaming.

## 12: Voice-length answers

- **Voice turns get their own prompt** (`VOICE_ANSWER_INSTRUCTION`): 60-80
  spoken words, no lists, markdown or citations. The first sentence keeps
  paragraph 1's meaning, including the dose referral to the doctor or nurse.
  A voice turn is anything inside `within_voice_deadline` (ticket 11), so
  Retell, Bolna, the Gemini Live tool and mobile voice all get it. Text
  channels keep the text prompt unchanged.
- **Measured on the live model** (`docs/latency/voice-answers-12.md`):
  output tokens down 83%, 74 words at p50 (87 max), whole answer in 1.4 s at
  p50 against 10.6 s for the text answer. No voice answer reached the safety
  net; 2 drafts had a dose before the guard and none after; 37 of 40 told the
  caller to confirm the dose with the doctor or nurse.
- **Thinking stays `minimal` on voice.** With the voice prompt it is close to
  `high` (18 wins, 22 losses), but the judge prefers `low` (minimal lost 30 of
  40 against it). `low` takes 6.6 s at p50 and 8.9 s at p95, past the 7 s
  deadline. **Open for the clinical lead:** if the review prefers `low`, set
  `llm.thinking.voice_answer: low` and `llm.deadlines.voice_s: 10`, and the
  caller waits about 7 s for the first word instead of 1 s. Ticket 13's
  acknowledgement fills some of that silence on Retell.
- **The text evidence badge is not appended on voice.** It is a formatted
  block (emoji, a rule line) that was either read aloud or cut. The badge's
  fields now go in `safety_enhancements.evidence_badge` on every channel.
  This field was `true` before, so Retell and Bolna crashed into their
  fallback truncation when they read it as a dict. They now speak their short
  evidence note when the badge says so.
- **Truncation is a safety net at 60 s** (`VOICE_SAFETY_NET_S`, ~130 words),
  raised from 30 s, so a voice-length answer plus its evidence note is never
  cut. Mobile voice now applies it too: TTS gets the voice-length text, and
  that same text is recorded and returned.
- **The Dose Boundary's redaction note is now spoken.** The voice cleanup
  deleted everything in parentheses, including the note `_(...)_` that follows
  a removed dose sentence. It is unwrapped first.
- **Clinician review:** a sheet with all 40 spoken answers is in project files
  (`research/latency-evals/voice-answers-review.md`). It has to be filled in
  and recorded before the release is activated.

## 13: Streamed, sentence-guarded answers

- **`answer_stream.py`:** the model streams; `SentenceSplitter` cuts on
  `. ? !` and `।` (not decimals, common abbreviations or list numbers) and
  on line ends. Each sentence is released only if the dose detector passes
  it alone and joined to the previous sentence (a dose split across a
  boundary). A failing sentence is dropped; at the end the redaction note
  follows, or the Deferral if less than the guard's minimum is left. This is
  `DosageGuard.apply`'s Redacted Answer behaviour, one sentence at a time. An
  emergency question passes untouched, as with the whole text.
- **The answer path streams only when a channel opts in** (`streaming_to`).
  Retell does. The answer the pipeline returns is exactly the text sent; the
  channel guards and sends any text added after generation on its own.
- **Retell:** sentences go out as `content_complete: false` messages, then a
  final message with the evidence note. A barge-in cancels the turn's task,
  which closes the model stream; no later sentence of the old turn is sent.
  The handler keeps what it sent as `session.last_spoken`.
- **Acknowledgement after 1.5 s of silence, not immediately.** Measured, the
  first sentence arrives at 1.13 s at p50; an immediate "one moment" would
  delay the answer it announces. It still covers slow turns, and covers the
  wait if voice thinking is ever raised to `low` (ticket 12).
- **Study log:** Retell writes no study interaction log (only the mobile
  routes do), so there is nothing to record the spoken text in yet.
  `last_spoken` is what such a record must use when Retell joins the study.
- **Web and mobile text opt-in not exposed yet.** The mechanism is one
  `streaming_to` around `query()`, but a new endpoint changes the mobile API
  contract the Android app is generated from. Left for when the app wants it.
- **The model stream** (`LLMClient.stream`) has the same budget as
  `generate`: each wait for the next piece is bounded by what is left. The
  fallback provider is used only if Gemini fails before its first text. The
  `llm` span records `first_text_ms`, so the trace now has time to first
  token per purpose for streamed calls.
- **Found, not fixed (safety code, needs its own review):** the dose detector
  flags the Tamil word "மருந்தின்" (of the medicine) and the Hindi "अगली बार"
  (next time) as frequencies. So the guard's own Tamil redaction note and
  Tamil and Hindi Deferrals contain "doses" by its own test, and Tamil text
  mentioning medicine is over-redacted. Over-redaction is the safe failure,
  but Live's Tamil audio on WhatsApp (ticket 08) will often fall back to TTS.
