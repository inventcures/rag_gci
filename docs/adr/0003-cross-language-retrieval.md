# Cross-language retrieval replaces query translation

Retrieval matches an Indic query directly against the knowledge base using a multilingual embedding model, instead of translating the query into English first. Query translation is retained only as a fallback for weak retrieval, never as the default path.

Two facts force this. The corpus is English and the live index is `BAAI/bge-small-en-v1.5` at 384 dimensions — English-only. And `translate_query_to_english()` still calls `llama-3.1-8b-instant` on Groq, while the RAG LLM moved to Gemini 3.1 Flash Lite. Without `GROQ_API_KEY` that call returns `status: "error"`, which `simple_rag_server.py:1261` treats as "proceed with the original query" — so untranslated Marathi is embedded by an English model against an English index and returns confident, ungrounded palliative advice. The failure is silent.

Removing the translation hop also removes an LLM round-trip from every query, which is what the "sub-13-second interactions" principle in §1.4 of the Android spec actually requires.

## Consequences

- `data/chroma_db` must be reindexed. At 263 chunks this takes minutes on CPU; `rebuild_embeddings.py` and `force_rebuild_embeddings.py` already exist.
- Note that three sources currently disagree about the embedding model — `config.yaml:20` says `all-MiniLM-L6-v2`, `simple_rag_server.py:817` tries `google/embeddinggemma-300m`, and the live collection says `bge-small-en-v1.5`. All three are English-only.
- Self-hosted rather than managed, because the study protocol (§7.5, §7.8) requires a versioned knowledge-base snapshot and an immutable release identifier. A managed RAG index cannot be pinned to a commit.
- Frozen corpus vectors are ~263 KB at int8 with bge-m3's 1024 dimensions, so shipping them to the Android client is free. The open question is query embedding on-device, which needs the model itself and is bounded by the 2 GB / 2G target devices in §3.1 and §3.3.
- The embedding model choice must be settled empirically against the real corpus. Neither vendor aggregate claims — Gemini Embedding 2's "over 100 languages", or any bge-m3 benchmark — speaks to these eleven languages with this clinical vocabulary.