"""
Uploads index in the background.

Before this, /api/upload parsed, chunked and embedded the whole document inside
the request, on the event loop: every live call stalled for as long as a
guideline took to index, and the filename came straight from the client.

What these pin down:

  * a hostile filename is stored inside uploads/ with no separators or `..`
  * a job moves queued -> running -> done (or failed), one job at a time
  * the event loop stays responsive (< 50 ms) while a slow parse and embed run
  * batched indexing adds exactly the chunks and ids a single add would
  * the server's upload route queues a job and has a status route
"""

import asyncio
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ingest_jobs  # noqa: E402
import post_work  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class SafeNames(unittest.TestCase):
    def test_hostile_names_stay_inside_uploads(self):
        uploads = Path("uploads").resolve()
        for raw in ["../../etc/passwd", "a\\b/c.pdf", "..", "", None, "/abs/x.pdf",
                    "..\\..\\win.ini", "ok name (v2).pdf", "दर्द प्रबंधन.pdf"]:
            name = ingest_jobs.safe_upload_name(raw)
            self.assertNotIn("/", name, raw)
            self.assertNotIn("\\", name, raw)
            self.assertNotIn("..", name, raw)
            self.assertEqual((uploads / name).resolve().parent, uploads, raw)

    def test_readable_names_survive_and_repeats_differ(self):
        a = ingest_jobs.safe_upload_name("WHO pain guide.pdf")
        b = ingest_jobs.safe_upload_name("WHO pain guide.pdf")
        self.assertTrue(a.endswith("_WHO pain guide.pdf"))
        self.assertNotEqual(a, b)
        self.assertTrue(ingest_jobs.safe_upload_name("दर्द.pdf").endswith("_दर्द.pdf"))


class Jobs(unittest.TestCase):
    def test_lifecycle_and_one_at_a_time(self):
        async def go():
            running = []
            peak = []
            gate = asyncio.Event()

            def work(tag, fail=None):
                async def run():
                    running.append(tag)
                    peak.append(len(running))
                    await gate.wait()
                    running.remove(tag)
                    if fail == "raise":
                        raise ValueError("bad pdf")
                    if fail == "result":
                        return {"status": "error", "error": "parse failed"}
                    return {"status": "success", "tag": tag}
                return run

            ok = ingest_jobs.submit("a.pdf", work("a"))
            bad = ingest_jobs.submit("b.pdf", work("b", "result"))
            boom = ingest_jobs.submit("c.pdf", work("c", "raise"))
            self.assertEqual(ingest_jobs.status(ok)["status"], "queued")
            await asyncio.sleep(0.02)
            self.assertEqual(ingest_jobs.status(ok)["status"], "running")
            self.assertEqual(ingest_jobs.status(bad)["status"], "queued")
            gate.set()
            await post_work.drain(5)
            self.assertEqual(max(peak), 1)
            self.assertEqual(ingest_jobs.status(ok)["status"], "done")
            self.assertEqual(ingest_jobs.status(ok)["result"]["tag"], "a")
            self.assertEqual(ingest_jobs.status(bad)["status"], "failed")
            self.assertEqual(ingest_jobs.status(bad)["error"], "parse failed")
            self.assertEqual(ingest_jobs.status(boom)["status"], "failed")
            self.assertIn("bad pdf", ingest_jobs.status(boom)["error"])
            self.assertIsNone(ingest_jobs.status("nope"))

        asyncio.run(go())

    def test_status_is_a_copy(self):
        async def go():
            async def work():
                return {"status": "success"}
            job = ingest_jobs.submit("x", work)
            ingest_jobs.status(job)["status"] = "tampered"
            self.assertEqual(ingest_jobs.status(job)["status"], "queued")
            await post_work.drain(5)

        asyncio.run(go())


class SlowCollection:
    def __init__(self, delay):
        self.delay = delay
        self.adds = []
        self.store = {}

    def add(self, documents, metadatas, ids):
        time.sleep(self.delay)
        self.adds.append(list(ids))
        for i, d, m in zip(ids, documents, metadatas):
            self.store[i] = (d, m)

    def get(self, ids=None, **kw):
        return {"ids": [i for i in ids if i in self.store]}


class SlowProcessor:
    def __init__(self, chunks, delay):
        self.chunks = chunks
        self.delay = delay

    def process_file(self, path):
        time.sleep(self.delay)
        return {"status": "success", "chunks": list(self.chunks), "page_count": 3}


def ingest_pipeline(chunks, parse_delay=0.0, add_delay=0.0):
    from simple_rag_server import SimpleRAGPipeline

    tmp = Path(tempfile.mkdtemp(prefix="ingest-"))
    p = SimpleRAGPipeline.__new__(SimpleRAGPipeline)
    p.vector_db = SlowCollection(add_delay)
    p.document_processor = SlowProcessor(chunks, parse_delay)
    p.document_metadata = {}
    p.documents_dir = tmp / "documents"
    p.documents_dir.mkdir()
    p.metadata_file = tmp / "metadata.json"
    src = tmp / "guide.pdf"
    src.write_bytes(b"%PDF fake")
    return p, str(src)


class AddDocuments(unittest.TestCase):
    def test_batches_add_every_chunk_once(self):
        chunks = [f"chunk {i}" for i in range(75)]
        p, src = ingest_pipeline(chunks)
        result = asyncio.run(p.add_documents([src], {"source": "test"}))
        self.assertEqual(result["results"][0]["status"], "success")
        self.assertEqual(result["results"][0]["chunks"], 75)
        doc_id = result["results"][0]["doc_id"]
        self.assertEqual([len(b) for b in p.vector_db.adds], [32, 32, 11])
        flat = [i for b in p.vector_db.adds for i in b]
        self.assertEqual(flat, [f"{doc_id}_chunk_{i}" for i in range(75)])
        self.assertEqual(p.vector_db.store[f"{doc_id}_chunk_40"][0], "chunk 40")
        self.assertEqual(p.vector_db.store[f"{doc_id}_chunk_40"][1]["total_chunks"], 75)
        self.assertEqual(p.document_metadata[doc_id]["chunk_count"], 75)
        self.assertTrue(p.metadata_file.exists())

    def test_event_loop_stays_responsive_while_indexing(self):
        p, src = ingest_pipeline([f"c{i}" for i in range(64)], parse_delay=0.3, add_delay=0.15)

        async def go():
            worst = 0.0
            done = asyncio.Event()

            async def keepalive():
                nonlocal worst
                loop = asyncio.get_running_loop()
                while not done.is_set():
                    t = loop.time()
                    await asyncio.sleep(0.01)
                    worst = max(worst, loop.time() - t - 0.01)

            ka = asyncio.create_task(keepalive())
            result = await p.add_documents([src])
            done.set()
            await ka
            return result, worst

        result, worst = asyncio.run(go())
        self.assertEqual(result["results"][0]["status"], "success")
        self.assertLess(worst, 0.05, f"event loop blocked {worst * 1000:.0f} ms")


class ServerRoutes(unittest.TestCase):
    def test_upload_queues_and_reports(self):
        source = (ROOT / "simple_rag_server.py").read_text(encoding="utf-8")
        start = source.index('@app.post("/api/upload")')
        body = source[start:source.index("PAGEINDEX RAG ENDPOINTS", start)]
        self.assertIn("ingest_jobs.safe_upload_name(file.filename)", body)
        self.assertIn("ingest_jobs.submit(", body)
        self.assertIn('@app.get("/api/upload/{job_id}")', body)
        self.assertNotIn("await rag_pipeline.add_documents", body)


if __name__ == "__main__":
    unittest.main()
