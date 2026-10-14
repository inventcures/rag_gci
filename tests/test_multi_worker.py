"""
The server can run as more than one process.

Before this, the app was built inside `main()` (no factory for uvicorn
workers), and each process kept conversation history only in its own memory:
with two workers, a caller whose next turn landed on the other process lost
the conversation.

What these pin down:

  * `create_app` is a factory uvicorn can call per worker; `--workers N` runs
    it that way
  * a turn recorded by one process is in the other's history on its next
    turn, read from the shared append-only log; a process does not read its
    own turns back twice; a line still being written is left for later
  * two processes appending long rows to one log never split each other's
    rows (one write() per row)
  * more than one worker refuses to start on the local Chroma store
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def worker(data_dir, writer):
    from simple_rag_server import SimpleRAGPipeline

    p = SimpleRAGPipeline.__new__(SimpleRAGPipeline)
    p.conversation_file = Path(data_dir) / "conversations.json"
    p._writer = writer
    p.conversations = p._load_conversations()
    return p


def turn(question):
    return {"timestamp": "t", "user_id": "u", "question": question, "answer": "a", "sources": []}


class SharedConversationHistory(unittest.TestCase):
    def test_a_turn_on_one_worker_is_history_on_the_other(self):
        data = tempfile.mkdtemp()
        a, b = worker(data, "A"), worker(data, "B")

        a._remember_turn("call-1", turn("breathless at night"))
        a._append_conversation("call-1", turn("breathless at night"))
        b._catch_up_conversations()
        self.assertEqual([t["question"] for t in b.conversations["call-1"]], ["breathless at night"])

        b._remember_turn("call-1", turn("and restless"))
        b._append_conversation("call-1", turn("and restless"))
        a._catch_up_conversations()
        b._catch_up_conversations()
        for p in (a, b):
            self.assertEqual([t["question"] for t in p.conversations["call-1"]],
                             ["breathless at night", "and restless"])

    def test_a_restarted_worker_loads_everything(self):
        data = tempfile.mkdtemp()
        a = worker(data, "A")
        for q in ("one", "two"):
            a._append_conversation("c", turn(q))
        self.assertEqual([t["question"] for t in worker(data, "C").conversations["c"]], ["one", "two"])

    def test_a_line_still_being_written_waits(self):
        data = tempfile.mkdtemp()
        b = worker(data, "B")
        log = Path(data) / "conversations.jsonl"
        row = json.dumps({"conversation_id": "c", "turn": turn("half"), "w": "A"})
        with open(log, "a") as f:
            f.write(row[:20])
        b._catch_up_conversations()
        self.assertNotIn("c", b.conversations)
        with open(log, "a") as f:
            f.write(row[20:] + "\n")
        b._catch_up_conversations()
        self.assertEqual(b.conversations["c"][0]["question"], "half")


def _append_rows(path, tag, n):
    from study_logging import append_line

    for i in range(n):
        append_line(Path(path), json.dumps({"w": tag, "i": i, "pad": tag * 30000}))


class SharedAppendOnlyLog(unittest.TestCase):
    def test_two_processes_appending_long_rows_do_not_interleave(self):
        import multiprocessing as mp

        path = Path(tempfile.mkdtemp()) / "interactions.jsonl"
        procs = [mp.get_context("fork").Process(target=_append_rows, args=(str(path), t, 150))
                 for t in ("a", "b")]
        for p in procs:
            p.start()
        for p in procs:
            p.join(30)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(rows), 300)
        self.assertEqual(sorted((r["w"], r["i"]) for r in rows),
                         sorted((t, i) for t in "ab" for i in range(150)))


class Factory(unittest.TestCase):
    def test_create_app_is_the_worker_factory(self):
        import inspect

        import simple_rag_server as srv

        self.assertIn("args", inspect.signature(srv.create_app).parameters)
        source = inspect.getsource(srv.main)
        self.assertIn('"simple_rag_server:create_app"', source)
        self.assertIn("factory=True", source)
        self.assertEqual(srv._parse_args(["--workers", "2", "--rag-method", "vector"]).workers, 2)
        self.assertEqual(srv._parse_args([]).workers, int(os.getenv("WEB_CONCURRENCY", "1")))

    def test_several_workers_need_a_chroma_server(self):
        from unittest import mock

        import simple_rag_server as srv

        env = {k: v for k, v in os.environ.items() if k != "CHROMA_HOST"}
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.object(sys, "argv", ["srv", "--workers", "2"]), \
             mock.patch.object(srv.uvicorn, "run") as run:
            with self.assertRaises(SystemExit):
                srv.main()
            run.assert_not_called()
        with mock.patch.dict(os.environ, {"CHROMA_HOST": "chroma"}), \
             mock.patch.object(sys, "argv", ["srv", "--workers", "2", "--no-ngrok"]), \
             mock.patch.object(srv.uvicorn, "run") as run:
            srv.main()
            self.assertIn("--workers 2", os.environ["PALLI_SERVER_ARGS"])
        self.assertEqual(run.call_args.args[0], "simple_rag_server:create_app")
        self.assertEqual(run.call_args.kwargs["workers"], 2)


if __name__ == "__main__":
    unittest.main()
