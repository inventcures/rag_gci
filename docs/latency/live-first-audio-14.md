# Gemini Live: end of speech to first audio

`scripts/benchmark_live_first_audio.py`, 6 spoken questions x 2, model gemini-3.1-flash-live-preview, audio streamed in real time then silence. Seconds from the end of speech to the first audio chunk.

| variant | n | p50 | p95 | max | RAG tool ran | spoke before the tool | failed to connect |
|---|---|---|---|---|---|---|---|
| before | 12 | 3.46 | 3.76 | 4.49 | 11 | 0 | 0 |
| after | 12 | 3.17 | 3.59 | 3.60 | 10 | 4 | 0 |

## Reading it

- **The caller hears something within about a second in half the "after"
  turns** (6 of 12: four acknowledgements before the tool call, at
  0.80-1.18 s, and two short clarifying questions), against 1 of 12 before.
  The other "after" turns are where they were (3.2-3.6 s), because the
  model went straight to the tool. The prompt allows the acknowledgement
  ("may"), and the model takes it in roughly a third to a half of turns
  (4 of 12 in this run and in an earlier run of the same prompt).
- **Those ~3.4 s are the tool path:** the model's tool call, retrieval, the
  voice-length answer (about 1.3 s on the live model), the guard, and the
  model speaking it.
- **Pre-connecting does not show here,** and should not: in this benchmark
  "before" connects at the first audio chunk, and the handshake (~0.6 s)
  finishes while the caller is still speaking, so it was already hidden. It
  matters when the caller's first chunk is sent before the connection is up
  and is queued; it costs nothing when unused (see the decision log).
- **Turns without the tool** (2 in "after", 1 in "before") were all
  clarifying questions or an acknowledgement still waiting for the tool when
  the 12 s window closed; none gave medical advice without it in this run.
