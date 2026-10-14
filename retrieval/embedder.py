"""
Query embedding: once per text, off the event loop, from the one loaded model.

The answer path used to hand Chroma `query_texts`, so Chroma embedded the
question synchronously on the event loop, and a non-English question was
embedded twice (the cross-language probe, then the search). A bge-m3 embedding
is tens to hundreds of milliseconds of CPU; on the event loop that is every
other caller's audio and keepalives waiting.

`QueryEmbedder` embeds in a worker thread, keeps a small exact-text cache, and
searches with `query_embeddings`, so Chroma never embeds on the request path.
It uses the collection's own embedding function, so the vectors are the ones
Chroma would have computed from `query_texts`.
"""

from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence

from observability import span

DEFAULT_CACHE_SIZE = 512


def share_loaded_model(model_name: str, model: Any) -> None:
    """Make Chroma's SentenceTransformerEmbeddingFunction reuse `model`.

    The function keeps a class-level cache keyed by model name and loads from
    the hub on a miss. Seeding it means the collection's embedding function
    (whose persisted config names the model, e.g. "BAAI/bge-m3") uses the
    instance the server already loaded, from a local path if configured: one
    copy in memory, and no hub access at startup.
    """
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    SentenceTransformerEmbeddingFunction.models[model_name] = model


class QueryEmbedder:
    """Embeds query texts in a worker thread, with an exact-text LRU cache."""

    def __init__(self, embed: Callable[[Sequence[str]], Any], cache_size: int = DEFAULT_CACHE_SIZE):
        self._embed = embed
        self._cache: "OrderedDict[str, List[float]]" = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _cached(self, text: str) -> Optional[List[float]]:
        with self._lock:
            vector = self._cache.get(text)
            if vector is not None:
                self._cache.move_to_end(text)
                self.hits += 1
            return vector

    def _store(self, text: str, vector: List[float]) -> None:
        with self._lock:
            self._cache[text] = vector
            self._cache.move_to_end(text)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    def _embed_one(self, text: str) -> List[float]:
        return [float(x) for x in self._embed([text])[0]]

    async def embed(self, text: str) -> List[float]:
        vector = self._cached(text)
        if vector is not None:
            return vector
        with span("embed"):
            vector = await asyncio.to_thread(self._embed_one, text)
        self.misses += 1
        self._store(text, vector)
        return vector

    async def search(self, collection: Any, text: str, n_results: int) -> Dict[str, Any]:
        """Chroma query by embedding, in a worker thread."""
        vector = await self.embed(text)
        return await asyncio.to_thread(collection.query, query_embeddings=[vector], n_results=n_results)


async def vector_search(collection: Any, embedder: Optional[QueryEmbedder], text: str,
                        n_results: int) -> Dict[str, Any]:
    """Search by text, off the event loop: through the embedder when there is
    one, else by `query_texts` (collections without a known embedding
    function, and test doubles)."""
    if embedder is not None:
        return await embedder.search(collection, text, n_results)
    return await asyncio.to_thread(collection.query, query_texts=[text], n_results=n_results)
