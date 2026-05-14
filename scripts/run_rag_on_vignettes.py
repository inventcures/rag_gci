#!/usr/bin/env python3
"""
Run the Palli Sahayak RAG pipeline against all 40 vignettes.

Loads each vignette JSON from data/evaluation/vignettes/, calls
SimpleRAGPipeline.query() with the asker_query in the vignette's
language, and stores the output to data/evaluation/rag_outputs/{id}.json
for downstream expert review.

Output schema per file:
{
    "vignette_id": str,
    "language": str,
    "asker_query": str,
    "answer": str,
    "sources": [{document, relevance, snippet}],
    "evidence_level": str,
    "emergency_level": str,
    "confidence": float,
    "validation": dict,
    "generated_at": ISO 8601 timestamp,
    "rag_method": str,
    "elapsed_ms": int,
    "error": str | None
}

Usage:
    cd /Users/tp53/Documents/tp53_AA/llms4palliative_gci/demo_feb2025/rag_gci
    source venv/bin/activate
    python3 scripts/run_rag_on_vignettes.py
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VIGNETTES_DIR = ROOT / "data" / "evaluation" / "vignettes"
OUTPUTS_DIR = ROOT / "data" / "evaluation" / "rag_outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_vignettes() -> Dict[str, dict]:
    """Load all vignette JSON files."""
    vignettes = {}
    for filepath in sorted(VIGNETTES_DIR.glob("*.json")):
        if filepath.name.startswith("_"):
            continue  # skip _schema.json etc.
        try:
            with open(filepath) as f:
                v = json.load(f)
            vignettes[v["vignette_id"]] = v
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
    return vignettes


def build_query_input(vignette: dict) -> str:
    """
    Construct the input string for the RAG pipeline.
    Uses the asker_query as the primary question, optionally prefixed
    with a brief case context summary so the RAG has clinical context.
    """
    context_parts = []
    pd = vignette.get("patient_demographics", {})
    cc = vignette.get("clinical_context", {})

    age = pd.get("age", "")
    sex = pd.get("sex", "")
    diagnosis = cc.get("primary_diagnosis", "")
    meds = cc.get("current_medications", [])
    comorbs = cc.get("comorbidities", [])

    summary = (
        f"Patient: {age}{sex}, diagnosis: {diagnosis}. "
        f"Current medications: {', '.join(meds) if meds else 'none'}. "
        f"Comorbidities: {', '.join(comorbs) if comorbs else 'none'}. "
        f"Presenting: {vignette.get('presenting_complaint', '')}"
    )
    asker_query = vignette.get("asker_query", "")
    return f"{summary}\n\nQuestion: {asker_query}"


async def run_one(vignette: dict, rag_pipeline) -> Dict[str, Any]:
    """Run a single vignette through the RAG pipeline."""
    vid = vignette["vignette_id"]
    language = vignette.get("language", "en-IN")
    short_lang = language.split("-")[0]  # "en-IN" -> "en"
    asker_query = vignette.get("asker_query", "")
    query_input = build_query_input(vignette)

    logger.info(f"[{vid}] Running RAG (lang={language})")
    started_at = time.time()

    try:
        result = await rag_pipeline.query(
            question=query_input,
            user_id=f"vignette_eval__{vid}",
            top_k=5,
            source_language=short_lang,
        )
    except Exception as e:
        logger.exception(f"[{vid}] RAG query raised exception")
        return {
            "vignette_id": vid,
            "language": language,
            "asker_query": asker_query,
            "answer": "",
            "sources": [],
            "evidence_level": "E",
            "emergency_level": "none",
            "confidence": 0.0,
            "validation": None,
            "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
            "rag_method": "unknown",
            "elapsed_ms": int((time.time() - started_at) * 1000),
            "error": str(e),
        }

    elapsed_ms = int((time.time() - started_at) * 1000)

    # Normalize fields across simple_rag_server.query() return shapes
    answer = result.get("answer", "") if isinstance(result, dict) else str(result)
    sources_raw = result.get("sources", []) if isinstance(result, dict) else []
    sources = []
    for s in sources_raw:
        if isinstance(s, dict):
            sources.append({
                "document": s.get("document") or s.get("source") or s.get("filename") or "",
                "page": s.get("page"),
                "relevance": s.get("relevance") or s.get("relevance_score") or s.get("score") or 0,
                "snippet": s.get("snippet") or s.get("text") or s.get("content", "")[:300],
            })
        else:
            sources.append({"document": str(s), "page": None, "relevance": 0, "snippet": ""})

    evidence_level = "C"
    emergency_level = "none"
    confidence = 0.0
    validation = None

    if isinstance(result, dict):
        eb = result.get("evidence_badge")
        if isinstance(eb, dict):
            evidence_level = eb.get("level", "C")
            confidence = eb.get("confidence_score", 0.0)
        ea = result.get("emergency_alert")
        if isinstance(ea, dict):
            emergency_level = ea.get("level", "none")
        validation = result.get("validation_result")
        if "confidence" in result and not confidence:
            confidence = float(result.get("confidence") or 0.0)

    return {
        "vignette_id": vid,
        "language": language,
        "asker_query": asker_query,
        "answer": answer,
        "sources": sources,
        "evidence_level": evidence_level,
        "emergency_level": emergency_level,
        "confidence": confidence,
        "validation": validation,
        "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "rag_method": result.get("rag_method", "vector") if isinstance(result, dict) else "unknown",
        "elapsed_ms": elapsed_ms,
        "error": None,
    }


async def main() -> int:
    # Lazy import — only after sys.path is set
    from dotenv import load_dotenv
    load_dotenv()
    from simple_rag_server import SimpleRAGPipeline  # noqa

    logger.info("Initializing SimpleRAGPipeline...")
    rag_pipeline = SimpleRAGPipeline()

    vignettes = load_vignettes()
    if not vignettes:
        logger.error("No vignettes found in %s", VIGNETTES_DIR)
        return 1
    logger.info("Loaded %d vignettes", len(vignettes))

    # Groq free tier: 6000 TPM. Each query uses ~2-3k tokens (context +
    # response), so space queries to stay under the limit. Also retry on
    # empty answer (which indicates the LLM call rate-limited inside the
    # pipeline and we only got an empty string back).
    INTER_REQUEST_DELAY_S = float(os.environ.get("RAG_RUNNER_DELAY_S", "30"))
    MAX_RETRIES = 2

    results = []
    for i, (vid, v) in enumerate(sorted(vignettes.items()), 1):
        for attempt in range(1, MAX_RETRIES + 2):
            logger.info("[%d/%d] %s (attempt %d)", i, len(vignettes), vid, attempt)
            result = await run_one(v, rag_pipeline)
            answer = result.get("answer", "") or ""
            if answer.strip() and not result.get("error"):
                break
            if attempt <= MAX_RETRIES:
                wait = 30 * attempt
                logger.warning(
                    "[%s] Empty/errored response (likely rate-limited); "
                    "sleeping %ds before retry", vid, wait,
                )
                await asyncio.sleep(wait)

        out_path = OUTPUTS_DIR / f"{vid}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        results.append((vid, result.get("error"), result.get("elapsed_ms")))

        if i < len(vignettes) and INTER_REQUEST_DELAY_S > 0:
            await asyncio.sleep(INTER_REQUEST_DELAY_S)

    # Summary
    failures = [r for r in results if r[1]]
    successes = [r for r in results if not r[1]]
    total_ms = sum(r[2] or 0 for r in results)
    avg_ms = total_ms // max(len(results), 1)

    print("\n" + "=" * 70)
    print(f"RAG run complete: {len(successes)}/{len(results)} successful")
    print(f"Total elapsed: {total_ms / 1000:.1f}s  (avg {avg_ms} ms per case)")
    print(f"Failures: {len(failures)}")
    for vid, err, _ in failures:
        print(f"  - {vid}: {err}")
    print("=" * 70)

    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
