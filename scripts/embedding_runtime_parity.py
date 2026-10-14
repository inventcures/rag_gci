#!/usr/bin/env python3
"""
The parity gate for switching the embedding runtime (ticket 17).

For each runtime asked for, on this machine: embed the live corpus and the
retrieval-eval queries (every probe in every Supported Language), and report

  * hit@5 per language, and the largest drop against torch
  * query embed p50 and p95 (one query at a time, as the answer path does)
  * how close its corpus vectors are to torch's (mean and minimum cosine);
    below 0.999 the vectors changed and the switch needs a reindex

A runtime passes when no language's hit@5 is more than 0.01 below torch.
Run it on the deployment's hardware: the speed numbers are only true there.

    EMBEDDING_MODEL_PATH=/srv/models/bge-m3 python scripts/embedding_runtime_parity.py \\
        --runtimes torch onnx openvino_int8 tei --out docs/latency/embedding-runtime-17.md

Exit status 1 if any runtime fails the gate.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retrieval.embedding_runtime import RUNTIMES, load_embedding_model  # noqa: E402

PARITY_TOLERANCE = 0.01
SAME_VECTORS_COSINE = 0.999


def hit_at_5(q_emb: np.ndarray, doc_emb: np.ndarray, queries: List[dict]) -> Dict[str, float]:
    sims = q_emb @ doc_emb.T
    per: Dict[str, List[int]] = {}
    for i, q in enumerate(queries):
        top = set(int(j) for j in np.argsort(-sims[i])[:5])
        per.setdefault(q["language"], []).append(1 if q["gold"] & top else 0)
    return {lang: sum(v) / len(v) for lang, v in sorted(per.items())}


def measure(runtime: str, source: str, corpus: List[str], queries: List[dict]) -> Dict[str, Any]:
    started = time.perf_counter()
    model = load_embedding_model(source, runtime)
    load_s = time.perf_counter() - started
    doc_emb = np.asarray(model.encode(corpus, normalize_embeddings=True, batch_size=32))
    q_emb = np.asarray(model.encode([q["text"] for q in queries], normalize_embeddings=True))
    times = []
    for q in queries:
        t = time.perf_counter()
        model.encode([q["text"]], normalize_embeddings=True)
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return {
        "runtime": runtime,
        "load_s": round(load_s, 1),
        "embed_p50_ms": round(statistics.median(times), 1),
        "embed_p95_ms": round(times[int(0.95 * (len(times) - 1))], 1),
        "hit_at_5": hit_at_5(q_emb, doc_emb, queries),
        "_doc_emb": doc_emb,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtimes", nargs="+", default=["torch", "onnx"], choices=RUNTIMES)
    ap.add_argument("--model", default=os.getenv("EMBEDDING_MODEL_PATH") or "BAAI/bge-m3")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    from evaluation.retrieval_eval.run_eval import build_queries, load_corpus

    corpus, _ = load_corpus()
    queries = build_queries()
    runtimes = ["torch"] + [r for r in args.runtimes if r != "torch"]
    results = []
    for runtime in runtimes:
        try:
            results.append(measure(runtime, args.model, corpus, queries))
        except Exception as e:  # a runtime this machine cannot load
            results.append({"runtime": runtime, "error": f"{type(e).__name__}: {e}"})
        print(json.dumps({k: v for k, v in results[-1].items() if not k.startswith("_")}), flush=True)

    base = results[0]
    if "error" in base:
        print("torch failed; nothing to compare against", file=sys.stderr)
        return 1
    failed = False
    lines = ["# Embedding runtime parity", "",
             f"`scripts/embedding_runtime_parity.py`, model `{args.model}`, {len(corpus)} chunks, "
             f"{len(queries)} queries. Pass: no language's hit@5 more than {PARITY_TOLERANCE} below torch.",
             "", "| runtime | embed p50 ms | p95 ms | worst hit@5 drop | vectors vs torch (mean / min cos) | gate |",
             "|---|---|---|---|---|---|"]
    for r in results:
        if "error" in r:
            failed = True
            lines.append(f"| {r['runtime']} | - | - | - | - | not loadable: {r['error'][:80]} |")
            continue
        drop = max(base["hit_at_5"][lang] - r["hit_at_5"].get(lang, 0.0) for lang in base["hit_at_5"])
        cos = np.sum(base["_doc_emb"] * r["_doc_emb"], axis=1)
        same = float(cos.min()) >= SAME_VECTORS_COSINE
        ok = drop <= PARITY_TOLERANCE
        failed |= not ok
        lines.append(f"| {r['runtime']} | {r['embed_p50_ms']} | {r['embed_p95_ms']} | {drop:+.3f} | "
                     f"{float(cos.mean()):.4f} / {float(cos.min()):.4f} | "
                     f"{'pass' if ok else 'FAIL'}{'' if same else ', reindex needed'} |")
    lines += ["", "hit@5 by language:", "", "| language | " + " | ".join(r["runtime"] for r in results) + " |",
              "|---|" + "---|" * len(results)]
    for lang in base["hit_at_5"]:
        lines.append(f"| {lang} | " + " | ".join(
            f"{r['hit_at_5'].get(lang, 0):.2f}" if "hit_at_5" in r else "-" for r in results) + " |")
    print("\n".join(lines))
    if args.out:
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
