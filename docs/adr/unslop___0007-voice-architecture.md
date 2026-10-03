# Gemini Live for voice with a Sarvam fallback, and an India-resident RAG LLM

Real-time voice conversation is the primary interaction. It runs on the Gemini Live API, with the Sarvam streaming path as an automatic fallback whenever the Live session fails or stalls. The RAG language model is Gemini 3.5 Flash pinned to `asia-south1`, and embeddings are self-hosted bge-m3.

## Why the LLM is 3.5 Flash and not 3.8

This was verified against Google's current Vertex AI model-endpoint locations table, not against the February comment in `config.yaml:107` claiming the Live API is "ONLY available in US/Europe regions". That comment is now out of date in one respect and correct in another, and both halves matter.

Available in Asia-Pacific, checked on the current documentation:

| Model | Asia-Pacific availability |
|---|---|
| Gemini 3.8 Flash | none |
| Gemini 3.7 Flash | none |
| Gemini 3.6 Flash | none |
| **Gemini 3.5 Flash** | **Mumbai, Singapore, Tokyo, Sydney** |
| Gemini 2.5 Flash | Mumbai, Singapore, Tokyo, Sydney, Seoul |
| Gemini Embedding (`gemini-embedding-001`) | all seven, including Mumbai |
| Gemini 3.8 Live (Live API) | none |
| Chirp and TTS models | none |

Gemini 3.5 Flash is the newest model available in India, so it is the RAG LLM. The model previously in use, `gemini-3.1-flash-lite`, is also unavailable in Asia-Pacific, so inference was not India-resident before this decision and is now.

## Why not the Sarvam-only path

The alternative was to route all voice through Sarvam on our own server, keeping every stream in India. It was rejected because it forfeits barge-in and interruption, which matter for a hands-free tool used during a home visit, and the grant names response latency as a monitored outcome.

## The residency position, stated precisely

The grant promises that "Sarvam AI servers located in India (data residency)" mitigates voice privacy risk. Gemini Live does not satisfy that promise for raw voice audio, and no Indian region offers the Live API. What the architecture delivers is narrower and should be described exactly this way rather than as a blanket residency claim:

| Stage | Provider | Where it runs |
|---|---|---|
| Voice, primary | Gemini Live | US, Europe, Seoul, Dammam, São Paulo |
| Voice, fallback | Sarvam | India |
| Language model | Gemini 3.5 Flash | **India (`asia-south1`)** |
| Embeddings | bge-m3 | **local** |
| Knowledge base | ChromaDB | **local** |

Clinical reasoning and grounding, which carry the patient content that matters most, stay in India. Only the audio stream leaves. Selecting an Indian region is not on its own evidence of in-region processing. The same documentation warns that endpoints "don't guarantee data residency" and that some models do not use a regional endpoint. That distinction needs confirming in writing with Google before the IRB sees the claim.

## Consequences

- **The fallback must trigger on health, not reachability.** A Live session that connects and then stalls mid-answer is worse than one that never connects, so detection needs a first-token timeout and per-turn stall detection, not just a connection check.
- **The provider used is recorded on every interaction.** Log rows must show which path answered, so the analysis can report how often India-resident and non-resident paths ran. That figure is also the evidence for the funder conversation about the residency deviation.
- **The fallback is automatic and not operator-toggleable.** A toggle that a site coordinator can flip in the field will eventually be flipped, at which point the residency position silently becomes false.
- Fallback changes the interaction shape. The Sarvam path is turn-based and cannot barge in, so the UI must indicate which mode is active, because the difference is perceptible and a user mid-sentence will otherwise assume the app has frozen.
- Pinning the LLM to `asia-south1` is a study-release property rather than a runtime default, and belongs in `data/study/release.yaml` so that drift is detected if an environment variable points it elsewhere.
- Revisit when the Live API gains an Indian region. The check is one table lookup and the answer should be re-verified rather than remembered.
