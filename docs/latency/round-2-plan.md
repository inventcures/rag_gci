# Latency round 2: what is left, ranked

Written 2026-10-15, after the 18 tickets of round 1 (PR #5).

## Where a turn's time goes now

| | p50 | source |
|---|---|---|
| Our code on a turn (retrieve, fuse, guard, record) | 5-10 ms | `baseline-01.md` |
| Query embedding (bge-m3, CPU) | ~190 ms, in a thread | round-1 research |
| Text answer, thinking `low` | 7.5 s to first token, 10.8 s whole | `thinking-11.md` |
| Voice answer, `minimal`, 80 words | 1.4 s whole | `voice-answers-12.md` |
| Answer translation (WhatsApp) | 3.8 s, after the answer | `thinking-11.md` |
| Live tool turn | ~1 s first audio on half the turns, else 3.4 s | `live-first-audio-14.md` |

Our side is under 1% of a text turn. What is left is Gemini time, waits in
series, and stalls that hurt other callers. Production traces have not been
summarised yet, so the shares below are measured on benchmarks, not on real
traffic.

## Ranked by time saved for the effort

| # | Item | Saves | Effort | Changes what users get | Status |
|---|---|---|---|---|---|
| 1 | Summarise a week of production traces (`scripts/trace_summary.py data/traces/*.jsonl`) and run `scripts/benchmark_answer_path.py --pipeline live` on the server | tells us where p95 really is | an hour on the server | no | needs the server |
| 2 | Handoff keywords stop the answer for ordinary questions ("scared", "alone", "caregiver", "interaction"), and nobody resolves the handoff | an answer instead of none | small code change, a clinical decision | **yes** | built behind `safety.handoff_mode`, off; needs the clinical lead and a study release |
| 3 | WhatsApp in Hindi and other languages: translation waits for the whole 10.8 s English answer. Stream the English answer and translate each paragraph as it completes | ~2-3 s on the translated message | medium | **yes** (translation by paragraph) | proposed, needs an eval and a study release |
| 4 | Stream text answers to the Android app (`AnswerStream` from round 1, as server-sent events) | first words at ~7.5 s instead of 10.8 s | medium, mostly Android | no (same text, sooner) | proposed |
| 5 | Text answers think for ~7 s before the first word at `low`. Judge `minimal` with a tighter prompt against `low` | ~6 s per text answer if it holds up | small code, needs the Gemini key for the eval | **yes** | proposed, needs the key |
| 6 | Embedding runtime parity gate (ticket 17), ONNX or OpenVINO | ~100 ms per turn, inferred | an hour on the server | no if fp32 | needs the server |
| 7 | Event-loop stalls: handoff store rewrite (351 ms at 300 requests and growing), `/health` index count, per-span file open | other callers stop waiting | small | no | **done in this PR** |
| 8 | Patient context fetched alongside retrieval; Gemini client and token ready at startup | first answer after start; latent per turn | small | no | **done in this PR** |
| 9 | Shared state (SQLite WAL) so the server can run more than one worker | throughput only | large | no | not now: Gemini-bound, one worker is enough until a load test says otherwise |
| 10 | Hedged Gemini requests for the tail | p95 only | small | no | not now: doubles the cost of hedged calls, and needs item 1's numbers first |

## Still open from round 1

- KG Cypher `EXPLAIN` check against the deployment graph (`NEO4J_URI`).
- Gemini Live can answer a medical turn without calling the knowledge-base
  tool, and nothing notices. A check on the turn's output transcript, as its
  own safety ticket.
- Voice thinking stays `minimal` until the clinician review of voice answers.
