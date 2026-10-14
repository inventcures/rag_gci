"""
The embedding runtime is a configuration switch behind a parity gate.

What these pin down:

  * `embedding.runtime` (or EMBEDDING_RUNTIME) picks torch, onnx,
    openvino_int8 or tei; anything else, or nothing, is torch (rollback is
    one config change)
  * each runtime loads the same model through sentence-transformers' backend,
    or the TEI sidecar's /embed, behind the same `encode`
  * the runtime is in the release fingerprint, so running another than the
    record shows as drift
  * the parity gate fails a runtime that loses more than 0.01 hit@5 in any
    language, and says when the vectors changed enough to need a reindex
"""

import json
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import embedding_runtime as er  # noqa: E402


class Selection(unittest.TestCase):
    def test_config_env_and_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EMBEDDING_RUNTIME", None)
            self.assertEqual(er.configured_runtime({}), "torch")
            self.assertEqual(er.configured_runtime({"embedding": {"runtime": "onnx"}}), "onnx")
            self.assertEqual(er.configured_runtime({"embedding": {"runtime": "cuda-magic"}}), "torch")
        with mock.patch.dict(os.environ, {"EMBEDDING_RUNTIME": "openvino_int8"}):
            self.assertEqual(er.configured_runtime({"embedding": {"runtime": "onnx"}}), "openvino_int8")

    def test_each_runtime_loads_through_its_backend(self):
        calls = []

        class FakeST:
            def __init__(self, source, **kwargs):
                calls.append((source, kwargs))

        with mock.patch("sentence_transformers.SentenceTransformer", FakeST):
            er.load_embedding_model("/m/bge-m3", "torch")
            er.load_embedding_model("/m/bge-m3", "onnx")
            er.load_embedding_model("/m/bge-m3", "openvino_int8")
        self.assertEqual(calls[0], ("/m/bge-m3", {}))
        self.assertEqual(calls[1], ("/m/bge-m3", {"backend": "onnx"}))
        self.assertEqual(calls[2][1]["backend"], "openvino")
        self.assertIn("qint8", calls[2][1]["model_kwargs"]["file_name"])


class TEI(unittest.TestCase):
    def test_encode_matches_the_sentence_transformers_shape(self):
        import httpx

        seen = []

        def handler(request):
            body = json.loads(request.content)
            seen.append(body)
            return httpx.Response(200, json=[[1.0, 0.0, 0.0] for _ in body["inputs"]])

        encoder = er.TEIEncoder("http://tei:8080")
        encoder._client = httpx.Client(transport=httpx.MockTransport(handler))
        batch = encoder.encode(["a", "b", "c"], batch_size=2, normalize_embeddings=True)
        self.assertEqual(batch.shape, (3, 3))
        self.assertEqual([len(b["inputs"]) for b in seen], [2, 1])
        self.assertTrue(seen[0]["normalize"])
        self.assertEqual(encoder.encode("one").shape, (3,))
        self.assertEqual(encoder.encode(["x"], convert_to_numpy=False), [[1.0, 0.0, 0.0]])


class Fingerprint(unittest.TestCase):
    def test_runtime_is_fingerprinted_and_drift_is_seen(self):
        from study_release import StudyRelease, load_yaml

        from pathlib import Path

        path = Path("data/study/release.yaml")
        release = StudyRelease(record=load_yaml(path), path=path)
        self.assertEqual(release.live_fingerprint()["embedding.runtime"], "torch")
        with mock.patch.dict(os.environ, {"EMBEDDING_RUNTIME": "onnx"}):
            drift = {f.component: f for f in release.check_drift()}
        self.assertIn("embedding.runtime", drift)
        self.assertEqual((drift["embedding.runtime"].recorded, drift["embedding.runtime"].actual),
                         ("torch", "onnx"))


class Gate(unittest.TestCase):
    def test_hit_at_5_by_language(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
        import embedding_runtime_parity as gate

        docs = np.eye(6)
        queries = [{"language": "hi-IN", "gold": {0}}, {"language": "hi-IN", "gold": {5}},
                   {"language": "ta-IN", "gold": {1}}]
        q = np.array([docs[0], docs[0], docs[1]])
        self.assertEqual(gate.hit_at_5(q, docs, queries), {"hi-IN": 0.5, "ta-IN": 1.0})

    def test_a_runtime_that_loses_recall_fails(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
        import embedding_runtime_parity as gate

        corpus = [f"chunk {i}" for i in range(8)]
        queries = [{"language": lang, "text": f"chunk {i}", "gold": {i}}
                   for lang in ("hi-IN", "ta-IN") for i in range(8)]

        class Model:
            def __init__(self, noisy):
                self.noisy = noisy

            def encode(self, texts, **kw):
                out = []
                for t in texts:
                    i = int(t.split()[-1])
                    v = np.zeros(8)
                    # The noisy runtime confuses chunk 7 for chunk 0 in queries.
                    v[0 if (self.noisy and t == "chunk 7" and kw.get("batch_size") is None) else i] = 1
                    out.append(v)
                return np.array(out)

        def load(source, runtime):
            return Model(noisy=runtime == "onnx")

        def run(loader):
            with mock.patch.object(gate, "load_embedding_model", loader), \
                 mock.patch("evaluation.retrieval_eval.run_eval.load_corpus", return_value=(corpus, [])), \
                 mock.patch("evaluation.retrieval_eval.run_eval.build_queries", return_value=queries), \
                 mock.patch.object(sys, "argv", ["gate", "--runtimes", "torch", "onnx"]):
                return gate.main()

        self.assertEqual(run(lambda source, runtime: Model(noisy=False)), 0)
        self.assertEqual(run(load), 1)  # chunk 7 lost: hit@5 down 0.125 in both languages


if __name__ == "__main__":
    unittest.main()
