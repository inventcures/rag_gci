#!/usr/bin/env python3
"""
Time to first spoken sentence on Retell, against the live model.

Each vignette question is sent to the real Retell handler as a
response_required event, over a fake socket, with the real answer path
(SimpleRAGPipeline.query, voice prompt, streaming, the Dose Boundary) and the
live Gemini model. Retrieval returns the vignette's top chunks from the live
index by word overlap (query embedding needs model weights; it is the same
for streamed and unstreamed answers). Provider STT and TTS are outside this
measurement, as the ticket specifies.

Reported per turn: time to the first message Retell would speak (the
acknowledgement or a sentence), time to the first answer sentence, and time
to the end of the answer, with the unstreamed time to the whole answer for
comparison (the same turn with streaming disabled).

    GEMINI_API_KEY=... python scripts/benchmark_retell_stream.py --out docs/latency/retell-stream-13.md
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_thinking import fmt, load_chunks, pct, top_chunks  # noqa: E402


class LexicalCollection:
    def __init__(self, docs, metas):
        self.docs, self.metas = docs, metas

    def query(self, query_texts, n_results):
        found, metas = top_chunks(query_texts[0], self.docs, self.metas, k=n_results)
        return {"documents": [found],
                "metadatas": [[{"filename": m.get("filename", "doc"), "chunk_index": m.get("chunk_index", 0),
                                "total_chunks": m.get("total_chunks", 1)} for m in metas]],
                "distances": [[0.4] * len(found)]}

    def count(self):
        return len(self.docs)


def pipeline(docs, metas):
    from simple_rag_server import SimpleRAGPipeline

    p = SimpleRAGPipeline.__new__(SimpleRAGPipeline)
    p.vector_db = LexicalCollection(docs, metas)
    p.document_metadata = {}
    p.conversations = {}
    p.context_injector = None
    p.cross_modal_aggregator = None
    p.translation_fallback_count = 0
    p.retrieval_router = None
    p.kg_checker = None
    p.graph_store = None
    p.conversation_file = Path(tempfile.mkdtemp(prefix="retell-bench-")) / "conversations.json"
    return p


class Socket:
    def __init__(self):
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.sent = []

    async def accept(self):
        pass

    async def receive_text(self):
        item = await self.inbox.get()
        if item is None:
            from fastapi import WebSocketDisconnect
            raise WebSocketDisconnect()
        return json.dumps(item)

    async def send_json(self, data):
        self.sent.append((asyncio.get_running_loop().time(), data))


async def one_turn(handler_cls, rag, question: str, ack_text: str, stream: bool):
    import answer_stream

    handler = handler_cls(rag_pipeline=rag)
    ws = Socket()
    call = asyncio.create_task(handler.handle_websocket(ws, "bench"))
    await ws.inbox.put({"interaction_type": "call_details", "call": {"metadata": {"language": "en-IN"}}})
    original = answer_stream.active_stream
    if not stream:
        import simple_rag_server
        simple_rag_server.active_stream = lambda: None
    loop = asyncio.get_running_loop()
    asked = loop.time()
    await ws.inbox.put({"interaction_type": "response_required", "response_id": 1,
                        "transcript": [{"role": "user", "content": question}]})
    while not any(d.get("content_complete") for _, d in ws.sent if d.get("response_type") == "response"):
        await asyncio.sleep(0.01)
    if not stream:
        import simple_rag_server
        simple_rag_server.active_stream = original
    await ws.inbox.put(None)
    await call
    responses = [(t - asked, d) for t, d in ws.sent if d.get("response_type") == "response"]
    first_any = responses[0][0]
    first_sentence = next((t for t, d in responses if d["content"].strip() and ack_text not in d["content"]), None)
    return first_any, first_sentence, responses[-1][0]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    from retell_integration.custom_llm_server import RetellCustomLLMHandler

    docs, metas = load_chunks()
    rag = pipeline(docs, metas)
    ack = RetellCustomLLMHandler()._get_ack_message("en")
    questions = []
    for path in sorted(glob.glob(str(ROOT / "data/evaluation/vignettes/v1/*.json")))[:args.turns]:
        v = json.loads(Path(path).read_text(encoding="utf-8"))
        questions.append(v.get("asker_query") or v["scenario_text"][:400])

    streamed, whole = [], []
    skipped = 0
    for q in questions:  # one at a time: per-turn latency, not throughput
        s = await one_turn(RetellCustomLLMHandler, rag, q, ack, stream=True)
        if s[2] < 0.05:
            # Answered before retrieval (an emergency or out-of-scope reply).
            skipped += 1
            continue
        streamed.append(s)
        whole.append(await one_turn(RetellCustomLLMHandler, rag, q, ack, stream=False))
        print(f"streamed first={streamed[-1][0]:.2f} sentence={streamed[-1][1] or -1:.2f} "
              f"end={streamed[-1][2]:.2f} | whole end={whole[-1][2]:.2f}", flush=True)

    rows = {
        "first speech (ack or sentence)": [s[0] for s in streamed],
        "first answer sentence": [s[1] for s in streamed if s[1] is not None],
        "answer complete, streamed": [s[2] for s in streamed],
        "answer complete, unstreamed": [w[2] for w in whole],
    }
    lines = ["# Retell: time to first spoken sentence", "",
             f"`scripts/benchmark_retell_stream.py`, {len(streamed)} vignette turns ({skipped} more were "
             "answered before retrieval as emergencies or out of scope), live model, one turn "
             "at a time. Seconds from response_required to the message Retell speaks; provider STT and TTS "
             "excluded. Retrieval by word overlap on the live index.", "",
             "| measure | p50 | p95 |", "|---|---|---|"]
    for name, values in rows.items():
        lines.append(f"| {name} | {fmt(pct(values, .5))} | {fmt(pct(values, .95))} |")
    print("\n".join(lines))
    if args.out:
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
