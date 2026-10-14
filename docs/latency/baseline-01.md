# Latency baseline (ticket 01)

Recorded on 14 Oct 2026 at the commit that added tracing, before any latency change.

**How:** `scripts/benchmark_answer_path.py --pipeline offline --reps 3`, then
`scripts/trace_summary.py` over the traces it wrote. The offline pipeline uses
the live Chroma index (263 chunks) with stored chunk embeddings and a canned
LLM draft. That means it measures everything on our side of the answer path
except query embedding and the Gemini call. Neo4j was not available, so the
`kg` and `fused` rows show the vector-only fallback.

**What this baseline does not cover:** query embedding (bge-m3; Hugging Face
downloads are blocked in the cloud session) and the Gemini call. Both dominate a
real turn. Re-run with `--pipeline live` on the deployment to get the full
picture. Per-turn traces from production (`data/traces/`) show every channel.

### benchmark/fused + K26 check (block) (36 requests)

| stage | count | p50 ms | p95 ms | share of request time |
|---|---|---|---|---|
| benchmark | 36 | 7.8 | 10.8 | 100.0% |
| answer | 36 | 7.6 | 10.5 | 98.0% |
| retrieve | 36 | 4.2 | 6.1 | 54.5% |
| router.retrieve | 36 | 3.9 | 5.7 | 49.9% |
| kg_check | 36 | 0.9 | 1.8 | 12.8% |
| record | 36 | 0.3 | 0.5 | 3.5% |
| generate | 36 | 0.0 | 0.0 | 0.1% |

### benchmark/kg (36 requests)

| stage | count | p50 ms | p95 ms | share of request time |
|---|---|---|---|---|
| benchmark | 36 | 4.1 | 5.5 | 100.0% |
| answer | 36 | 4.0 | 5.3 | 97.5% |
| retrieve | 36 | 2.7 | 3.7 | 65.0% |
| router.retrieve | 36 | 2.5 | 3.4 | 60.3% |
| record | 36 | 0.1 | 0.3 | 3.9% |
| generate | 36 | 0.0 | 0.0 | 0.0% |

### benchmark/kg + K26 check (block) (36 requests)

| stage | count | p50 ms | p95 ms | share of request time |
|---|---|---|---|---|
| benchmark | 36 | 6.1 | 8.0 | 100.0% |
| answer | 36 | 6.0 | 7.9 | 98.2% |
| retrieve | 36 | 3.5 | 5.0 | 58.4% |
| router.retrieve | 36 | 3.3 | 4.6 | 54.4% |
| kg_check | 36 | 0.6 | 0.9 | 10.7% |
| record | 36 | 0.2 | 0.3 | 3.2% |
| generate | 36 | 0.0 | 0.0 | 0.1% |

### benchmark/old: vector (36 requests)

| stage | count | p50 ms | p95 ms | share of request time |
|---|---|---|---|---|
| benchmark | 36 | 5.4 | 7.9 | 100.0% |
| answer | 36 | 5.2 | 7.7 | 97.7% |
| retrieve | 36 | 3.4 | 5.1 | 65.4% |
| search | 36 | 3.0 | 4.7 | 58.4% |
| record | 36 | 0.2 | 0.3 | 3.5% |
| generate | 36 | 0.0 | 0.0 | 0.1% |

### benchmark/vector + K26 check (block) (36 requests)

| stage | count | p50 ms | p95 ms | share of request time |
|---|---|---|---|---|
| benchmark | 36 | 6.5 | 8.3 | 100.0% |
| answer | 36 | 6.3 | 8.1 | 97.9% |
| retrieve | 36 | 3.2 | 4.5 | 51.4% |
| search | 36 | 2.9 | 4.0 | 45.6% |
| kg_check | 36 | 0.9 | 1.4 | 14.1% |
| record | 36 | 0.2 | 0.4 | 4.0% |
| generate | 36 | 0.0 | 0.0 | 0.1% |

### direct (5 requests)

| stage | count | p50 ms | p95 ms | share of request time |
|---|---|---|---|---|
| answer | 5 | 6.8 | 25.9 | 100.0% |
| retrieve | 5 | 3.9 | 11.4 | 42.5% |
| router.retrieve | 3 | 3.5 | 3.6 | 16.3% |
| search | 2 | 3.2 | 3.2 | 10.2% |
| kg_check | 3 | 1.1 | 20.2 | 37.4% |
| record | 5 | 0.2 | 0.3 | 2.1% |
| generate | 5 | 0.0 | 0.0 | 0.1% |

