# EmbeddingGemma 2 replaces bge-m3, gated on measurement

Retrieval moves from `BAAI/bge-m3` (1024d) to `google/embeddinggemma-2` (768d), on the condition that it matches or beats bge-m3 on ADR 0003's 99-probe, eleven-language evaluation, both overall and in every language. The `Retrieval eval` workflow enforces the condition. On a pass it rebuilds the knowledge index and commits it with the measurement. On a failure it stays red, and this change does not merge.

## Why consider it

EmbeddingGemma 2 was released on 6 Oct 2026 under Apache 2.0. Every figure below is from Google's model card, Hugging Face card or launch posts unless marked otherwise.

- **Multilingual text.** It scores 61.36 on MTEB Multilingual v2, Mean(Task). bge-m3 scores 59.56 (from the EmbeddingGemma 1 paper, arXiv 2509.20354).
- **The closest public evidence for our task.** XTREME-UP has an Indic query retrieve an English passage, which is what this app does. On it, EmbeddingGemma 1 scored 47.7 MRR@10 against bge-m3's 26.9, and won on Hindi, Marathi, Kannada, Malayalam, Gujarati, Tamil and Punjabi. Google says version 2 "retains" version 1's multilingual accuracy. That is a claim, not a measurement on these languages.
- **Smaller.** The text path is 270M parameters, against bge-m3's 568M. The 740M total includes vision and audio encoders, which we do not load.
- **Matryoshka output.** The 768d vectors can be truncated to 512, 256 or 128 dimensions. That reopens ADR 0005's offline question.

## Why it is gated, not simply adopted

- Google publishes no Indic result for version 2 and names no Indic language.
- **Odia** is the one supported language where EmbeddingGemma 1 lost to bge-m3 on XTREME-UP (21.2 against 23.1). **Bengali and Telugu** are not in XTREME-UP at all.
- Benchmarks have already misled this project once. LaBSE, tuned on 22 Indian languages, lost to bge-m3 0.72 to 0.96 on this corpus (ADR 0003).

## Consequences

- **Prompts.** EmbeddingGemma 2 is asymmetric. Queries are embedded as `task: search result | query: …` and passages as `title: none | text: …`. Missing prompts do not error; they lower quality silently. `retrieval_embeddings.py` routes Chroma's document embedding and `embed_query` to the right prompt, so every existing `query_texts=` call site is covered. Per-model defaults live there because the server builds its pipeline without `config.yaml`.
- **Reindex.** The index is rebuilt at 768d. Chroma refuses a collection persisted with a different embedding function, so the server now opens such an index as it is and lets the drift check refuse retrieval, instead of failing to start.
- **Dependencies.** `sentence-transformers>=6.1.0`, required by the model, and `chromadb>=1.5.0` for `embed_query`.
- **Precision.** Run it in float32 or bfloat16, never float16. The model card warns float16 gives NaN or degraded embeddings.
- **Residency.** Unchanged. The weights are open and self-hosted, so no query or passage text leaves the deployment.
- **Not yet calibrated.** `CROSS_LANGUAGE_FALLBACK_DISTANCE` (1.5) was set for the previous model. Cosine distances from a new model have a different spread, so the weak-retrieval fallback should be re-checked against the gate's results.
- **Native-speaker review.** ADR 0003's caveat still applies. Nine probes per language cannot separate two models that both score around 0.95, so the probe set needs a native-speaker review before the study build.
