#!/usr/bin/env python3
"""
Latency of the answer path, old (vector only) against the new retrieval modes,
on a fixed question set.

Each configuration runs every question `--reps` times through
SimpleRAGPipeline.query(), the method every channel calls, and the per-stage
timings it records (retrieval, generation, kg_check, total) are summarised as
median and p95. Results print as a markdown table and, with --out, are written
to a file.

Two pipelines:

  --pipeline live      The real SimpleRAGPipeline: real query embedding, real
                       Chroma, real LLM. Needs the deployment's environment
                       (model weights, API keys).
  --pipeline offline   The live Chroma index (copied to a temp dir, never
                       written), queried by vector with a stand-in query
                       embedding (the stored embedding of the chunk sharing most
                       words with the question), and a canned draft answer in
                       place of the LLM. Measures exactly what the new modes add
                       (graph retrieval, fusion, the K26 check) on real Chroma and
                       real Neo4j, without model weights or API keys. Query
                       embedding and generation are identical in old and new
                       paths, so they are excluded rather than estimated.

Neo4j comes from NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD, as in the server.
With --load-base-kg the base palliative-care graph is imported first.

    python scripts/benchmark_answer_path.py --pipeline offline --load-base-kg \
        --out docs/answer_path_latency_results.md
"""

import argparse
import asyncio
import os
import re
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Fixed question set: the vignette themes the study cares about (pain, opioid
# side effects, breathlessness, nausea, terminal care), phrased the way ASHA
# workers and caregivers ask. Each carries a draft answer for --pipeline offline
# that includes checkable claims, one of them deliberately contradicted (Q4).
QUESTIONS: List[Tuple[str, str]] = [
    ("What can we give for severe cancer pain at home?",
     "Morphine is the first-line strong opioid for cancer pain. Paracetamol helps with mild pain. "
     "Morphine commonly causes constipation, so a laxative should be started."),
    ("My father is on morphine and has not passed stool for 4 days. What should we do?",
     "Constipation is a common side effect of morphine. Senna is used to treat constipation. "
     "Lactulose prevents opioid-induced constipation."),
    ("How can breathlessness be eased in a patient with lung cancer?",
     "Morphine helps reduce breathlessness. Lung cancer often causes breathlessness. "
     "A fan and sitting upright can also help."),
    ("Is morphine good for constipation?",
     "Morphine relieves constipation in most patients. Senna is used to treat constipation."),
    ("What helps with nausea and vomiting in advanced cancer?",
     "Ondansetron helps with nausea. Metoclopramide is used for vomiting from gastric stasis. "
     "Haloperidol can control nausea from metabolic causes."),
    ("The patient is very drowsy after starting morphine. Is that expected?",
     "Drowsiness is a common side effect of morphine and usually settles in a few days. "
     "Tell the doctor if the patient cannot be woken."),
    ("Can we give ibuprofen to a patient with kidney failure?",
     "Ibuprofen should be avoided in kidney failure. Paracetamol is safer for mild pain."),
    ("What can help with anxiety in the last days of life?",
     "Lorazepam helps with anxiety. Midazolam is used for acute agitation. "
     "Calm reassurance and family presence also help."),
    ("How do we manage noisy breathing near the end of life?",
     "Hyoscine is used to treat the death rattle. Repositioning the patient on their side can help."),
    ("What medicine helps nerve pain?",
     "Gabapentin helps neuropathic pain. Amitriptyline is also used for neuropathic pain."),
    ("Patient is confused and agitated at night. What can be done?",
     "Haloperidol is used for delirium. Check for constipation and urine retention, which can cause agitation."),
    ("Does morphine cause addiction when used for pain in cancer?",
     "When morphine is used for cancer pain under medical supervision, addiction is uncommon. "
     "Morphine does not cause constipation."),
]

CONFIGS: List[Tuple[str, str, str]] = [
    # (label, rag_method, kg_check)
    ("old: vector", "vector", "off"),
    ("vector + K26 check (block)", "vector", "block"),
    ("kg", "kg", "off"),
    ("kg + K26 check (block)", "kg", "block"),
    ("fused + K26 check (block)", "fused", "block"),
]


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower())}


class ProxyEmbeddingCollection:
    """A Chroma collection queried by a stored chunk embedding per question."""

    def __init__(self, collection, query_embeddings: Dict[str, List[float]]):
        self.collection = collection
        self.query_embeddings = query_embeddings

    def query(self, query_texts, n_results):
        emb = self.query_embeddings[query_texts[0]]
        return self.collection.query(query_embeddings=[emb], n_results=n_results)

    def count(self):
        return self.collection.count()


def offline_pipeline(answers: Dict[str, str]):
    import chromadb

    from simple_rag_server import SimpleRAGPipeline

    tmp = Path(tempfile.mkdtemp(prefix="bench-chroma-"))
    shutil.copytree(ROOT / "data" / "chroma_db", tmp / "chroma_db")
    client = chromadb.PersistentClient(path=str(tmp / "chroma_db"))
    collection = client.get_collection("documents")
    stored = collection.get(include=["documents", "embeddings"])

    query_embeddings = {}
    for question, _ in QUESTIONS:
        qw = _words(question)
        best = max(
            range(len(stored["documents"])),
            key=lambda i: len(qw & _words(stored["documents"][i] or "")),
        )
        query_embeddings[question] = list(stored["embeddings"][best])

    p = SimpleRAGPipeline.__new__(SimpleRAGPipeline)
    p.vector_db = ProxyEmbeddingCollection(collection, query_embeddings)
    p.document_metadata = {}
    p.conversations = {}
    p.context_injector = None
    p.cross_modal_aggregator = None
    p.translation_fallback_count = 0
    p.retrieval_router = None
    p.kg_checker = None
    p.graph_store = None

    async def generate(question, context, metadatas, should_fuse=False):
        return answers[question], "canned"

    p._generate_answer_with_citations = generate
    return p


def live_pipeline():
    from simple_rag_server import SimpleRAGPipeline

    return SimpleRAGPipeline()


async def connect_graph(load_base: bool):
    from knowledge_graph import KnowledgeGraphRAG

    kg = KnowledgeGraphRAG(use_llm=False)
    connected = await kg.initialize()
    if connected and load_base:
        await kg.import_base_knowledge()
    return kg, connected


def graphrag_engine():
    try:
        from graphrag_integration import GraphRAGConfig, GraphRAGQueryEngine

        settings_path = ROOT / "data" / "graphrag" / "settings.yaml"
        if not settings_path.exists():
            return None
        return GraphRAGQueryEngine(GraphRAGConfig.from_yaml(str(settings_path)))
    except Exception:
        return None


def summarise(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return statistics.median(ordered), p95


async def run(args) -> str:
    import simple_rag_server
    from retrieval.settings import RetrievalSettings, set_active_settings

    os.environ.setdefault("RETRIEVAL_AUDIT_DIR", tempfile.mkdtemp(prefix="bench-audit-"))
    answers = dict(QUESTIONS)
    pipeline = offline_pipeline(answers) if args.pipeline == "offline" else live_pipeline()
    kg, connected = await connect_graph(args.load_base_kg)
    engine = graphrag_engine()
    if engine is not None:
        try:
            await engine.initialize()
        except Exception:
            engine = None

    rows = []
    for label, method, check in CONFIGS:
        s = set_active_settings(RetrievalSettings(rag_method=method, kg_check_mode=check))
        simple_rag_server._rag_method = method
        pipeline.retrieval_router = None
        pipeline.kg_checker = None
        pipeline.configure_answer_path(s, kg_rag=kg, graphrag_engine=engine)

        # One untimed warm-up per configuration (connection pools, HNSW load).
        await pipeline.query(QUESTIONS[0][0])

        stages: Dict[str, List[float]] = {}
        outcomes = {"contradicted": 0, "blocked": 0, "graph_failed": 0}
        for _ in range(args.reps):
            for question, _ in QUESTIONS:
                started = time.perf_counter()
                result = await pipeline.query(question)
                wall = (time.perf_counter() - started) * 1000
                timings = result.get("retrieval", {}).get("stage_latency_ms", {})
                for key in ("retrieval", "generation", "kg_check"):
                    if key in timings:
                        stages.setdefault(key, []).append(float(timings[key]))
                stages.setdefault("total", []).append(wall)
                kc = result.get("kg_check", {})
                outcomes["contradicted"] += int(kc.get("contradicted", 0) or 0)
                outcomes["blocked"] += int(kc.get("action") == "blocked")
                outcomes["graph_failed"] += int(bool(result.get("retrieval", {}).get("retrievers_failed")))
        rows.append((label, stages, outcomes))

    base_total = summarise(rows[0][1]["total"])[0]
    lines = [
        f"Pipeline: `{args.pipeline}`; Neo4j connected: {connected}; GraphRAG index: "
        f"{'loaded' if engine is not None and getattr(engine.data_loader, 'is_loaded', lambda: False)() else 'none'}; "
        f"{len(QUESTIONS)} questions x {args.reps} reps per configuration.",
        "",
        "| Configuration | retrieval ms (median / p95) | K26 check ms (median / p95) | total ms (median / p95) | added vs old (median) | contradicted claims | answers edited |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, stages, outcomes in rows:
        r = summarise(stages.get("retrieval", []))
        c = summarise(stages.get("kg_check", []))
        t = summarise(stages["total"])
        check = f"{c[0]:.1f} / {c[1]:.1f}" if stages.get("kg_check") else "not run"
        lines.append(
            f"| {label} | {r[0]:.1f} / {r[1]:.1f} | {check} | {t[0]:.1f} / {t[1]:.1f} | "
            f"{t[0] - base_total:+.1f} | {outcomes['contradicted'] // args.reps} | {outcomes['blocked'] // args.reps} |"
        )
    await kg.close()
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline", choices=["offline", "live"], default="offline")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--load-base-kg", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    import logging

    logging.disable(logging.WARNING)
    table = asyncio.run(run(args))
    print(table)
    if args.out:
        args.out.write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
