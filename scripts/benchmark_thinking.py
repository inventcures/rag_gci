#!/usr/bin/env python3
"""
Thinking level per purpose, measured against the live model.

Every prompt is the server's own: the answer, query-translation,
answer-translation and clarification prompts are captured from
SimpleRAGPipeline and SmartClarificationSystem with the LLM call stubbed out,
then sent to the configured Gemini model at each thinking level, streamed, so
time to first token is measured as well as total time.

Quality, against the model's "high" level as the reference:

  answer               pairwise judge (the model at "high", order randomised):
                       does this level's answer lose to "high"? Also: how many
                       drafts carry a dose before the Dose Boundary, and that
                       none does after it.
  query_translation    does the translation mean the probe's English original?
  answer_translation   faithful to the English answer, under 1500 characters,
                       not cut off?
  clarification        same needs-clarification verdict as "high"?

Contexts come from the live index (copied, never written) by word overlap
with the question, since query embedding needs model weights; the thinking
level does not touch retrieval.

    GEMINI_API_KEY=... python scripts/benchmark_thinking.py \
        --out docs/latency/thinking-11.md --raw /tmp/thinking.json
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import random
import re
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LEVELS = ("minimal", "low", "medium", "high")
QUERY_LANGS = ("hi", "bn", "ta", "gu")  # the languages translate_query_to_english supports
WORD = re.compile(r"\w+", re.UNICODE)


# -- prompts, captured from the server's own code --------------------------
class Capture:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self, prompt, max_tokens=2048, temperature=0.3, purpose="answer", **kw):
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature,
                           "purpose": purpose, "json_mode": kw.get("json_mode", False)})
        return "{}", "capture"


def capture_pipeline():
    from simple_rag_server import SimpleRAGPipeline

    p = SimpleRAGPipeline.__new__(SimpleRAGPipeline)
    p._call_llm = Capture()
    return p


def load_chunks() -> Tuple[List[str], List[Dict[str, Any]]]:
    import chromadb

    tmp = Path(tempfile.mkdtemp(prefix="thinking-chroma-"))
    shutil.copytree(ROOT / "data" / "chroma_db", tmp / "db")
    got = chromadb.PersistentClient(path=str(tmp / "db")).get_collection("documents").get(
        include=["documents", "metadatas"])
    shutil.rmtree(tmp, ignore_errors=True)
    return got["documents"], got["metadatas"]


def top_chunks(question: str, docs, metas, k=5):
    q = {w.lower() for w in WORD.findall(question) if len(w) > 3}
    scored = sorted(range(len(docs)), key=lambda i: -len(q & {w.lower() for w in WORD.findall(docs[i])}))[:k]
    return [docs[i] for i in scored], [metas[i] for i in scored]


async def answer_prompts(limit: int) -> List[Dict[str, Any]]:
    docs, metas = load_chunks()
    p = capture_pipeline()
    items = []
    for path in sorted(glob.glob(str(ROOT / "data/evaluation/vignettes/v1/*.json")))[:limit]:
        v = json.loads(Path(path).read_text(encoding="utf-8"))
        question = v.get("asker_query") or v["scenario_text"][:400]
        ctx, md = top_chunks(question, docs, metas)
        context = "\n\n".join(f"Source: {m.get('filename')} (chunk {m.get('chunk_index', 0) + 1})\n{d}"
                              for d, m in zip(ctx, md))
        await p._generate_answer_with_citations(question, context, md, False)
        call = p._call_llm.calls.pop()
        items.append({"id": v["vignette_id"], "question": question, **call})
    return items


async def query_translation_prompts() -> List[Dict[str, Any]]:
    from evaluation.retrieval_eval.run_eval import SEED_PROBES

    p = capture_pipeline()
    items = []
    for probe in SEED_PROBES:
        for lang in QUERY_LANGS:
            text = probe["translations"].get(f"{lang}-IN")
            if not text:
                continue
            await p.translate_query_to_english(text, lang)
            items.append({"id": f"{probe['id']}:{lang}", "english": probe["en"], **p._call_llm.calls.pop()})
    return items


async def answer_translation_prompts(answers: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    p = capture_pipeline()
    items = []
    for n, (vid, english) in enumerate(answers):
        lang = QUERY_LANGS[n % len(QUERY_LANGS)]
        await p.translate_text(english, lang)
        items.append({"id": f"{vid}:{lang}", "english": english, **p._call_llm.calls.pop()})
    return items


async def clarification_prompts(questions: List[str]) -> List[Dict[str, Any]]:
    import smart_clarification_system as scs

    cap = Capture()

    class Stub:
        async def generate(self, prompt, **kw):
            await cap(prompt, **kw)
            from llm.client import LLMResponse
            return LLMResponse("{}", True, "capture", "capture")

    original = scs.get_llm_client
    scs.get_llm_client = lambda: Stub()
    try:
        system = scs.SmartClarificationSystem()
        items = []
        for n, q in enumerate(questions):
            await system._llm_clarity_analysis(q)
            items.append({"id": f"c{n}", "question": q, **cap.calls.pop()})
    finally:
        scs.get_llm_client = original
    return items


# -- the live model ----------------------------------------------------------
class Runner:
    def __init__(self, concurrency: int):
        from llm.client import LLMClient

        self.client = LLMClient()
        self.sem = asyncio.Semaphore(concurrency)

    async def run(self, item: Dict[str, Any], level: str) -> Dict[str, Any]:
        cfg = self.client._gemini_config(None, item["max_tokens"], item["temperature"],
                                         item.get("json_mode", False), level)
        async with self.sem:
            for attempt in range(3):
                started = time.perf_counter()
                first = None
                parts: List[str] = []
                finish = usage = None
                try:
                    stream = await self.client._genai_client().aio.models.generate_content_stream(
                        model=self.client.settings.model, contents=item["prompt"], config=cfg)
                    async for chunk in stream:
                        text = getattr(chunk, "text", None)
                        if text:
                            if first is None:
                                first = time.perf_counter() - started
                            parts.append(text)
                        if chunk.candidates and chunk.candidates[0].finish_reason:
                            finish = str(chunk.candidates[0].finish_reason)
                        usage = getattr(chunk, "usage_metadata", None) or usage
                    return {"text": "".join(parts).strip(), "ttft_s": first,
                            "total_s": time.perf_counter() - started, "finish": finish,
                            "thought_tokens": getattr(usage, "thoughts_token_count", None),
                            "error": None}
                except Exception as e:  # rate limit or transient: back off and retry
                    err = f"{type(e).__name__}: {str(e)[:200]}"
                    await asyncio.sleep(2 * (attempt + 1))
            return {"text": "", "ttft_s": None, "total_s": None, "finish": None,
                    "thought_tokens": None, "error": err}

    async def judge(self, prompt: str) -> Dict[str, Any]:
        out = await self.run({"prompt": prompt, "max_tokens": 4096, "temperature": 0.0, "json_mode": True}, "high")
        try:
            return json.loads(out["text"])
        except Exception:
            return {"error": out.get("error") or out["text"][:200]}


ANSWER_JUDGE = """You are a palliative care physician reviewing two draft answers to the same question from a community health worker or family caregiver in India. Both drafts were written from the same source excerpts.

Judge which draft is better for the caller: clinically correct and safe, grounded in the excerpts, practical about what to do now and when to escalate, clear. Length is not a virtue. Ignore citation formatting.

QUESTION:
{question}

DRAFT A:
{a}

DRAFT B:
{b}

Reply as JSON: {{"winner": "A" | "B" | "tie", "reason": "<one sentence>"}}"""

MEANING_JUDGE = """Do these two English sentences ask the same medical question (same meaning, same intent)? Wording may differ.

1: {a}
2: {b}

Reply as JSON: {{"same": true | false}}"""

FAITHFUL_JUDGE = """Is the translation a faithful {lang} rendering of the English medical text: same clinical content, nothing important added or dropped, no mistranslated medical terms? A shorter but faithful rendering counts as faithful.

ENGLISH:
{english}

TRANSLATION:
{translation}

Reply as JSON: {{"faithful": true | false, "reason": "<one sentence>"}}"""


def pct(values: List[float], q: float) -> Optional[float]:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def fmt(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.2f}"


def needs_clarification(text: str) -> Optional[bool]:
    try:
        d = json.loads(text[text.find("{"):text.rfind("}") + 1])
        return bool(d.get("needs_clarification"))
    except Exception:
        return None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--raw", type=Path)
    args = ap.parse_args()
    levels = [l for l in args.levels.split(",") if l]
    random.seed(11)

    from safety_enhancements import guard_query_result
    from dosage_guard import detect_doses

    runner = Runner(args.concurrency)
    raw: Dict[str, Any] = {"model": runner.client.settings.model, "levels": levels, "purposes": {}}

    async def sweep(name: str, items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        results = {}
        for level in levels:
            outs = await asyncio.gather(*(runner.run(it, level) for it in items))
            results[level] = [{**{k: v for k, v in it.items() if k != "prompt"}, **o} for it, o in zip(items, outs)]
            ok = [r for r in results[level] if not r["error"]]
            print(f"{name:20s} {level:8s} n={len(ok)}/{len(items)} "
                  f"ttft p50={fmt(pct([r['ttft_s'] for r in ok], .5))} p95={fmt(pct([r['ttft_s'] for r in ok], .95))} "
                  f"total p50={fmt(pct([r['total_s'] for r in ok], .5))} p95={fmt(pct([r['total_s'] for r in ok], .95))}",
                  flush=True)
        raw["purposes"][name] = results
        return results

    # answers
    answers = await sweep("answer", await answer_prompts(args.answers))
    quality: Dict[str, Dict[str, Any]] = {"answer": {}}
    ref = answers["high"] if "high" in answers else None
    for level, rows in answers.items():
        doses_before = doses_after = 0
        for r in rows:
            if r["text"]:
                doses_before += bool(detect_doses(r["text"]))
                res = {"status": "success", "answer": r["text"], "sources": []}
                guard_query_result(res, query=r["question"], language="en")
                doses_after += bool(detect_doses(res["answer"]))
        q = {"doses_before_guard": doses_before, "doses_after_guard": doses_after,
             "cut_off": sum(1 for r in rows if r["finish"] and "MAX_TOKENS" in r["finish"])}
        if ref is not None and level != "high":
            jobs = []
            for r, h in zip(rows, ref):
                flip = random.random() < 0.5
                a, b = (h["text"], r["text"]) if flip else (r["text"], h["text"])
                jobs.append((flip, runner.judge(ANSWER_JUDGE.format(question=r["question"], a=a, b=b))))
            verdicts = await asyncio.gather(*(j for _, j in jobs))
            wins = losses = ties = 0
            for (flip, _), v in zip(jobs, verdicts):
                w = v.get("winner")
                mine = ("B" if flip else "A")
                if w == "tie":
                    ties += 1
                elif w == mine:
                    wins += 1
                elif w in ("A", "B"):
                    losses += 1
            q.update(wins=wins, losses=losses, ties=ties, net_loss_rate=(losses - wins) / max(1, len(rows)))
        quality["answer"][level] = q

    # query translation
    qt = await sweep("query_translation", await query_translation_prompts())
    quality["query_translation"] = {}
    for level, rows in qt.items():
        verdicts = await asyncio.gather(*(runner.judge(MEANING_JUDGE.format(a=r["english"], b=r["text"])) for r in rows))
        quality["query_translation"][level] = {
            "same_meaning": sum(1 for v in verdicts if v.get("same") is True), "n": len(rows)}

    # answer translation, of the reference answers
    source = ref or answers[levels[0]]
    english = [(r["id"], r["text"]) for r in source if r["text"]][:16]
    at = await sweep("answer_translation", await answer_translation_prompts(english))
    names = {"hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "gu": "Gujarati"}
    quality["answer_translation"] = {}
    for level, rows in at.items():
        verdicts = await asyncio.gather(*(runner.judge(FAITHFUL_JUDGE.format(
            lang=names[r["id"].rsplit(":", 1)[1]], english=r["english"], translation=r["text"])) for r in rows))
        quality["answer_translation"][level] = {
            "faithful": sum(1 for v in verdicts if v.get("faithful") is True), "n": len(rows),
            "over_1500": sum(1 for r in rows if len(r["text"]) > 1500),
            "cut_off": sum(1 for r in rows if r["finish"] and "MAX_TOKENS" in r["finish"])}

    # clarification
    vague = ["I have pain", "My medicine isn't working", "He is not well", "What should I do now?",
             "She is not eating", "Is it normal?", "The tablets are finished", "He is breathing strangely"]
    questions = vague + [r["question"] for r in answers[levels[0]]][:16]
    cl = await sweep("clarification", await clarification_prompts(questions))
    quality["clarification"] = {}
    ref_cl = [needs_clarification(r["text"]) for r in cl.get("high", [])]
    for level, rows in cl.items():
        mine = [needs_clarification(r["text"]) for r in rows]
        quality["clarification"][level] = {
            "parsed": sum(1 for m in mine if m is not None), "n": len(rows),
            "agrees_with_high": sum(1 for m, h in zip(mine, ref_cl) if m is not None and m == h) if ref_cl else None}

    raw["quality"] = quality
    print(json.dumps(quality, indent=2))
    if args.raw:
        args.raw.write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.out:
        lines = [f"# Thinking level per purpose ({raw['model']})", "",
                 f"Measured {time.strftime('%Y-%m-%d')} with `scripts/benchmark_thinking.py`, "
                 f"{args.concurrency} calls in flight, streamed. Seconds.", "",
                 "| purpose | level | n | first token p50 | p95 | total p50 | p95 | errors |",
                 "|---|---|---|---|---|---|---|---|"]
        for name, results in raw["purposes"].items():
            for level, rows in results.items():
                ok = [r for r in rows if not r["error"]]
                lines.append(f"| {name} | {level} | {len(rows)} | {fmt(pct([r['ttft_s'] for r in ok], .5))} | "
                             f"{fmt(pct([r['ttft_s'] for r in ok], .95))} | {fmt(pct([r['total_s'] for r in ok], .5))} | "
                             f"{fmt(pct([r['total_s'] for r in ok], .95))} | {len(rows) - len(ok)} |")
        lines += ["", "## Quality against `high`", "", "```json", json.dumps(quality, indent=2), "```", ""]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
