# Retell: time to first spoken sentence

`scripts/benchmark_retell_stream.py`, 34 vignette turns (6 more were answered before retrieval as emergencies or out of scope), live model, one turn at a time. Seconds from response_required to the message Retell speaks; provider STT and TTS excluded. Retrieval by word overlap on the live index.

| measure | p50 | p95 |
|---|---|---|
| first speech (ack or sentence) | 1.13 | 1.45 |
| first answer sentence | 1.13 | 1.45 |
| answer complete, streamed | 1.44 | 1.71 |
| answer complete, unstreamed | 1.41 | 2.11 |

## Reading it

- The first answer sentence reaches Retell at 1.13 s at p50 (target: under
  1.2 s), 1.45 s at p95.
- Streaming saves less than it would have before tickets 11 and 12: a voice
  answer is now ~75 words and written in ~1.4 s, so the whole answer is not
  much later than its first sentence. At p95 the first sentence comes 0.7 s
  before an unstreamed answer would.
- The acknowledgement waits 1.5 s. At 0.6 s it played on nearly every turn
  and queued ~1.5 s of "one moment" ahead of a sentence that was 0.4 s away.
  It now plays only on a slow turn.
- Not included: query embedding (bge-m3 cannot be downloaded here; ~0.2 s on
  CPU per the Rust research), provider STT and TTS.
