# Text LLM

Every text-generation call goes through one client, `llm/client.py`:

- answer drafting and query translation (`simple_rag_server.py`)
- knowledge-graph entity extraction and Cypher generation (`knowledge_graph/`)
- PageIndex tree search (`pageindex_integration/`)
- GraphRAG indexing and search (`data/graphrag/settings.yaml`)
- memory, meta and Context-1 agents, smart clarification

Before this, those calls were spread across Groq (Llama 3.1, Qwen3), OpenAI
(gpt-4o, gpt-4o-mini), a MedGemma endpoint and Gemini, each with its own key and
error handling. Those paths are gone.

Gemini Live voice (the model the study release names, `models.voice_primary.model`) is a speech model on its own session and is
not part of this client.

## Configuration

`config.yaml`, with environment variables taking precedence:

```yaml
llm:
  model: "gemini-3.5-flash"       # LLM_MODEL
  location: "asia-south1"         # LLM_LOCATION (Vertex AI only)
  timeout_s: 45
  fallback:
    enabled: false                # LLM_FALLBACK_ENABLED
    provider: "deepseek"
    model: "deepseek-flash"       # LLM_FALLBACK_MODEL
    base_url: "https://api.deepseek.com"
```

| Variable | Effect |
|---|---|
| `GOOGLE_CLOUD_PROJECT` | Use Vertex AI, pinned to `llm.location` |
| `GEMINI_API_KEY` | Use the Gemini API instead (no region pinning) |
| `DEEPSEEK_API_KEY` | Needed only when the fallback is enabled |

## Behaviour

- Gemini is called first. DeepSeek is called only when `fallback.enabled` is true,
  a key is set, and Gemini failed or timed out.
- `generate()` never raises. A failure returns `ok=False` with the reason, and each
  caller keeps its existing degraded path (template Cypher, untranslated query,
  rule-based agent output, the "please consult your doctor" answer).
- `<think>` blocks are stripped from replies.

## Why the default is 3.5 Flash with no fallback

The study requires identifiable data to stay India-hosted (ADR 0007).

- Gemini 3.8 Flash has no Asia-Pacific region, so it would send questions outside
  India. 3.5 Flash is the newest Flash served from `asia-south1`.
- DeepSeek is hosted in China and is not on the approved processor list.

To switch, set `model: "gemini-3.8-flash"` and/or `fallback.enabled: true`. Both
change the release fingerprint (`llm.model`, `llm.location`, `llm.fallback` in
`study_release.py`), so the server reports RELEASE DRIFT until
`data/study/release.yaml` is updated and re-approved.
