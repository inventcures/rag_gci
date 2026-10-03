# Cross-language retrieval replaces query translation

Retrieval matches an Indic query directly against the knowledge base using a multilingual embedding model, instead of translating the query into English first. Query translation stays only as a fallback for weak retrieval, never as the default path.

Two facts force this. The corpus is English and the live index is `BAAI/bge-small-en-v1.5` at 384 dimensions, which handles English only. And `translate_query_to_english()` still calls `llama-3.1-8b-instant` on Groq, while the RAG LLM moved to Gemini 3.1 Flash Lite. Without `GROQ_API_KEY` that call returns `status: "error"`, and `simple_rag_server.py:1261` treats that as "proceed with the original query". The result is that untranslated Marathi gets embedded by an English model against an English index and returns confident, unfounded palliative advice. The failure is silent.

Dropping the translation hop also removes an LLM round-trip from every query, which is what the "sub-13-second interactions" principle in §1.4 of the Android spec actually asks for.

## Consequences

- `data/chroma_db` must be reindexed with **bge-m3**, chosen by measurement rather than by vendor claim. See the decision below and `evaluation/retrieval_eval/`.
- Reindexing 263 chunks takes about 295 s on CPU. Query encoding runs at about 104 ms, comfortably inside the 13-second voice interaction budget.
- Three sources disagreed about the previous embedding model, and `config.yaml:20` said `all-MiniLM-L6-v2`, `simple_rag_server.py:817` tried `google/embeddinggemma-300m`, and the live index said `bge-small-en-v1.5`. All three were English-only. The release record now names one model and the pipeline fails rather than searching a mismatched index.
- Self-hosted rather than managed, because the study protocol (§7.5, §7.8) requires a versioned knowledge-base snapshot and an immutable release identifier. A managed RAG index cannot be pinned to a commit.
- Frozen corpus vectors run to about 263 KB at int8 with bge-m3's 1024 dimensions, so shipping them to the Android client is free. On-device query embedding is the open question, and it depends on shipping the model itself, which is constrained by the 2 GB / 2G target devices in §3.1 and §3.3.
- Punjabi scored 0.78 hit@5 in the evaluation, the weakest supported language and the only one below 0.89. On a 99-query evaluation that is noise, but it is the language to watch.

## Measured result: bge-m3 over LaBSE

`evaluation/retrieval_eval/run_eval.py` runs 9 clinical probes across all 11 supported languages against the real 263-chunk corpus, with gold anchored to specific chunk indices so a model cannot score well by retrieving a thematically adjacent passage.

| | bge-m3 | LaBSE | bge-small-en-v1.5 (previous) |
|---|---|---|---|
| mean hit@5 | **0.96** | 0.72 | 0.09 |
| hit@1 | **0.73** | 0.32 | 0.07 |
| MRR@10 | **0.84** | 0.49 | 0.08 |
| dimensions | 1024 | 471 | 384 |
| parameters | 568M | 471M | 33M |
| reindex, 263 chunks on CPU | 295 s | 91 s | 30 s |
| query latency | 104 ms | 54 ms | 12 ms |

The third column is the finding that matters. The model the live index was built with scored **0.00 on every one of the ten Indic languages** while scoring 1.00 on English. A Marathi query retrieved none of its gold passages and returned confident, unfounded palliative advice.

LaBSE was expected to win on its home turf, having been tuned on 22 Indian languages. It lost, and lost worst on the languages it claims, at 0.56 for Kannada and 0.56 for Malayalam against bge-m3's 1.00 and 0.89. The tuning claim did not survive contact with clinical palliative vocabulary.

This evaluation is enough to choose a model and not enough to publish. Ninety-nine queries is a small sample, the translations were authored by engineers rather than reviewed by native speakers, and the gold labels are engineering judgement. Protocol §7.7 requires the release record to name the actual retriever and embedding model with evidence, so the probe set should be reviewed by a native speaker per language before the study build. The harness is in place for that re-run.
