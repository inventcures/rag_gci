# Offline is a cached question bundle plus local history, not local retrieval

When Palli Sahayak has no connectivity, it shows the interaction history stored on
the device and the answers to a small set of anticipated questions. It does not
answer new questions. The microphone control stays visible but disabled, with a
plain-language explanation, rather than failing silently or hiding the control.

The bundle is the grant's own "offline caching of 20 most common questions". A
lexical index with a bilingual Indic-to-English palliative term lexicon is
deferred to v1.1 as the honest answer to "why can't I ask anything else".

## Rejected: frozen corpus vectors plus on-device query embedding

The corpus is 263 chunks, so frozen vectors are ~263 KB at int8 and fit anywhere.
The blocker is the other half of the operation. Searching requires embedding the
incoming question, which means running the encoder on the phone. bge-m3 is 568M
parameters: ~600 MB int8, ~1.1 GB fp16. Against the study's target devices that is
two problems, not one:

- **Download.** At an effective 50 kbps, a 2G link characteristic of CCHRC
  Barak Valley, 600 MB is roughly 27 hours; even a generous 200 kbps is 7 hours.
  Nobody downloads that before a home visit.
- **Memory.** The reference device is a 2 GB Redmi. The weights alone need
  600 MB to 1 GB resident before runtime overhead, on a phone also running the
  operating system and the app.

It is worth being precise about a common misreading: the 25 MB figure in the
Android specification is a self-imposed design target, not a platform limit — Google
Play permits roughly 200 MB. Even so, raising the target to 500 MB does not save
this. The 2G download time and the RAM kill it on their own.

Freezing the vectors without also solving query embedding produces vectors the
device cannot use, and defeats the purpose, because the reason for offline is that
the network is gone.

## Rejected: managed retrieval

Vertex AI Search, now branded Agent Search, would remove the retrieval code from
the deployment. It was rejected on the study protocol rather than on capability.
§7.5 requires a stable version identifier for the searchable knowledge-base
snapshot and §7.8 requires the release record to name it; a managed index is opaque,
can change chunking or swap embedding models on a platform upgrade, and cannot be
pinned to a commit. §7.7 additionally requires source-withdrawal and
provider-failure checks, which require owning the index. Bring-your-own-embeddings
is capped at 768 dimensions, so bge-m3 at 1024 would need Matryoshka truncation —
paying for the model locally while losing control of the index, which is strictly
worse than self-hosting.

## Consequences

- The offline affordance is a design question about legibility, not a technical
  one. Users may be semi-literate, so the state must be conveyed by icon and
  colour together with plain text rather than by text alone.
- A disabled control that explains itself teaches the reason. A transient banner
  gets scrolled past and leaves the user unsure whether the tap registered.
- Redacted Answers must render distinctly from Answers in offline history, so that a
  refusal saved weeks earlier is never mistaken for clinical advice.
- Scheduled medication reminders already on the device must continue to fire
  offline; they are not a network feature.
- The bundle is built server-side by `offline/cache_builder.py`, which exists but
  is not wired to any route. Its schema is a shared contract between the server
  and the client, which is part of why the app lives in this repository rather than
  a separate one.
- The v1.1 lexical upgrade is cheap at this corpus size: a BM25 inverted index over
  263 chunks plus a curated term lexicon is a few megabytes and needs no model.
  It should be built only if the v1 bundle proves insufficient, and measured against
  the same harness used for the embedding evaluation.