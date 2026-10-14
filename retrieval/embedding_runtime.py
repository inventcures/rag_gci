"""
Which runtime computes embeddings: PyTorch, ONNX, OpenVINO int8, or a TEI sidecar.

The model is fixed by configuration (ADR 0003: bge-m3); this is how it runs.
`embedding.runtime` in config.yaml (or EMBEDDING_RUNTIME) picks one:

  torch          sentence-transformers on PyTorch (the default, and rollback)
  onnx           sentence-transformers with the ONNX Runtime backend
  openvino_int8  sentence-transformers with OpenVINO, int8-quantised weights
  tei            Hugging Face text-embeddings-inference over HTTP
                 (EMBEDDING_TEI_URL, default http://127.0.0.1:8080)

Whatever the runtime, the object returned has the `encode` the server and
Chroma's embedding function call, so nothing else changes. A runtime is only
allowed in a study release after `scripts/embedding_runtime_parity.py` shows
hit@5 within 0.01 of torch in every Supported Language; one whose vectors
differ from the index on disk (int8) needs a reindex with it
(scripts/reindex_with_configured_model.py). The runtime is part of the
release fingerprint, so running a different one than the record shows as
drift.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

RUNTIMES = ("torch", "onnx", "openvino_int8", "tei")
DEFAULT_RUNTIME = "torch"
DEFAULT_TEI_URL = "http://127.0.0.1:8080"

# The int8 OpenVINO export sentence-transformers produces with
# `export_static_quantized_openvino_model` (and ships for some models).
OPENVINO_INT8_FILE = "openvino/openvino_model_qint8_quantized.xml"


def configured_runtime(config: Optional[Dict[str, Any]] = None) -> str:
    """EMBEDDING_RUNTIME, else `embedding.runtime`, else torch."""
    runtime = os.getenv("EMBEDDING_RUNTIME") or ((config or {}).get("embedding") or {}).get("runtime")
    runtime = str(runtime or DEFAULT_RUNTIME).strip().lower()
    if runtime not in RUNTIMES:
        logger.warning("Unknown embedding runtime %r; using %s", runtime, DEFAULT_RUNTIME)
        return DEFAULT_RUNTIME
    return runtime


def load_embedding_model(source: str, runtime: str = DEFAULT_RUNTIME, **kwargs: Any) -> Any:
    """The model at `source` (hub name or local path) on `runtime`."""
    if runtime == "tei":
        return TEIEncoder(os.getenv("EMBEDDING_TEI_URL", DEFAULT_TEI_URL))

    from sentence_transformers import SentenceTransformer

    if runtime == "onnx":
        return SentenceTransformer(source, backend="onnx", **kwargs)
    if runtime == "openvino_int8":
        return SentenceTransformer(source, backend="openvino",
                                   model_kwargs={"file_name": OPENVINO_INT8_FILE}, **kwargs)
    return SentenceTransformer(source, **kwargs)


class TEIEncoder:
    """`encode` against a text-embeddings-inference sidecar (`POST /embed`).

    TEI normalises by default, as the server's bge-m3 use expects; the
    `normalize_embeddings` argument is passed through as `normalize`."""

    def __init__(self, url: str, timeout_s: float = 10.0):
        import httpx

        self.url = url.rstrip("/")
        self._client = httpx.Client(timeout=timeout_s)

    def encode(self, sentences: Any, batch_size: int = 32, normalize_embeddings: bool = True,
               convert_to_numpy: bool = True, **_: Any) -> Any:
        import numpy as np

        single = isinstance(sentences, str)
        texts: List[str] = [sentences] if single else list(sentences)
        vectors: List[Sequence[float]] = []
        for start in range(0, len(texts), batch_size):
            response = self._client.post(f"{self.url}/embed", json={
                "inputs": texts[start:start + batch_size], "normalize": normalize_embeddings})
            response.raise_for_status()
            vectors.extend(response.json())
        array = np.asarray(vectors, dtype=np.float32)
        out = array[0] if single else array
        return out if convert_to_numpy else out.tolist()

    def get_sentence_embedding_dimension(self) -> Optional[int]:
        try:
            return len(self.encode("dimension probe"))
        except Exception:
            return None
