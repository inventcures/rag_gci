#!/usr/bin/env python3
"""
Voice-length answers against the text answers they replace, on the live model.

The same 40 vignette prompts (scripts/benchmark_thinking.py) are answered with
the text prompt at the text level ("low") and with the voice prompt at the
voice level ("minimal"), as a voice turn would. Reported: output tokens, words,
time to first token and total, how many voice answers the channel safety net
(voice_safety_wrapper.fit_for_voice) would still cut, how many drafts carried a
dose before the Dose Boundary and after it, and how many voice answers say to
confirm the dose with the doctor or nurse.

It also writes a clinician review sheet: each vignette's question with its
voice answer after the Dose Boundary and the channel cleanup, as a caller
would hear it.

    GEMINI_API_KEY=... python scripts/benchmark_voice_answers.py \\
        --out docs/latency/voice-answers-12.md --sheet /tmp/voice-review.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_thinking import Runner, answer_prompts, fmt, pct  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--sheet", type=Path)
    ap.add_argument("--raw", type=Path)
    args = ap.parse_args()

    from dosage_guard import detect_doses
    from safety_enhancements import guard_query_result
    from voice_safety_wrapper import VOICE_SAFETY_NET_S, clean_for_voice, fit_for_voice

    runner = Runner(args.concurrency)
    settings = runner.client.settings
    text_items = await answer_prompts(args.answers, voice=False)
    voice_items = await answer_prompts(args.answers, voice=True)
    text = await asyncio.gather(*(runner.run(it, settings.thinking_for("answer")) for it in text_items))
    voice = await asyncio.gather(*(runner.run(it, settings.thinking_for("answer", voice=True)) for it in voice_items))

    rows = []
    for it, t, v in zip(voice_items, text, voice):
        guarded = {"status": "success", "answer": v["text"], "sources": []}
        guard_query_result(guarded, query=it["question"], language="en")
        spoken = fit_for_voice(guarded["answer"], VOICE_SAFETY_NET_S)
        rows.append({
            "id": it["id"], "question": it["question"],
            "text_tokens": t["output_tokens"], "voice_tokens": v["output_tokens"],
            "text_words": len(t["text"].split()), "voice_words": len(v["text"].split()),
            "voice_ttft_s": v["ttft_s"], "voice_total_s": v["total_s"],
            "text_ttft_s": t["ttft_s"], "text_total_s": t["total_s"],
            "cut_by_safety_net": spoken.rstrip(". ") != clean_for_voice(guarded["answer"]).rstrip(". "),
            "dose_before_guard": bool(detect_doses(v["text"])),
            "dose_after_guard": bool(detect_doses(guarded["answer"])),
            "dose_referral": any(w in v["text"].lower() for w in ("doctor or nurse", "nurse or doctor")),
            "finish": v["finish"], "error": v["error"] or t["error"],
            "spoken": spoken,
        })

    ok = [r for r in rows if not r["error"] and r["text_tokens"] and r["voice_tokens"]]
    text_tokens = sum(r["text_tokens"] for r in ok)
    voice_tokens = sum(r["voice_tokens"] for r in ok)
    summary = {
        "n": len(rows), "errors": len(rows) - len(ok),
        "output_tokens_text": text_tokens, "output_tokens_voice": voice_tokens,
        "output_token_drop": round(1 - voice_tokens / max(1, text_tokens), 3),
        "voice_words_p50": pct([r["voice_words"] for r in ok], .5),
        "voice_words_max": max((r["voice_words"] for r in ok), default=None),
        "text_words_p50": pct([r["text_words"] for r in ok], .5),
        "voice_first_token_p50_s": pct([r["voice_ttft_s"] for r in ok], .5),
        "voice_total_p50_s": pct([r["voice_total_s"] for r in ok], .5),
        "voice_total_p95_s": pct([r["voice_total_s"] for r in ok], .95),
        "text_first_token_p50_s": pct([r["text_ttft_s"] for r in ok], .5),
        "text_total_p50_s": pct([r["text_total_s"] for r in ok], .5),
        "cut_by_safety_net": sum(r["cut_by_safety_net"] for r in ok),
        "cut_off_by_token_cap": sum(1 for r in ok if r["finish"] and "MAX_TOKENS" in r["finish"]),
        "dose_before_guard": sum(r["dose_before_guard"] for r in ok),
        "dose_after_guard": sum(r["dose_after_guard"] for r in ok),
        "dose_referral": sum(r["dose_referral"] for r in ok),
    }
    print(json.dumps(summary, indent=2))
    if args.raw:
        args.raw.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1),
                            encoding="utf-8")
    if args.out:
        lines = [f"# Voice-length answers ({settings.model})", "",
                 f"`scripts/benchmark_voice_answers.py`, {len(rows)} vignettes, {args.concurrency} calls "
                 "in flight. Text: text prompt at the text level. Voice: voice prompt at the voice level.", "",
                 "| | text | voice |", "|---|---|---|",
                 f"| output tokens (sum) | {text_tokens} | {voice_tokens} |",
                 f"| words p50 | {summary['text_words_p50']} | {summary['voice_words_p50']} |",
                 f"| first token p50 (s) | {fmt(summary['text_first_token_p50_s'])} | "
                 f"{fmt(summary['voice_first_token_p50_s'])} |",
                 f"| total p50 (s) | {fmt(summary['text_total_p50_s'])} | {fmt(summary['voice_total_p50_s'])} |",
                 "", "```json", json.dumps(summary, indent=2), "```", ""]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines), encoding="utf-8")
    if args.sheet:
        lines = ["# Voice answers for clinician review", "",
                 "Each answer is exactly what a caller would hear: generated at voice length, "
                 "after the Dose Boundary and the channel's speech cleanup. For each, mark "
                 "safe / unsafe and helpful / unhelpful, with a note.", ""]
        for r in rows:
            lines += [f"## {r['id']}", "", f"**Question:** {r['question']}", "",
                      f"**Spoken answer ({len(r['spoken'].split())} words):** {r['spoken']}", "",
                      "- Safe: [ ] yes [ ] no", "- Helpful: [ ] yes [ ] no", "- Note:", ""]
        args.sheet.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
