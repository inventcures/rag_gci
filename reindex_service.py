#!/usr/bin/env python3
"""
Re-embedding migration
======================
Rebuilds a ChromaDB collection using a different embedding model.

Scope of this migration
-----------------------
The corpus source documents are not available in this environment: the permanent
document store and the uploads directory are both empty, and the collection's
`file_path` metadata points at a macOS temporary directory that no longer exists.
So the migration re-embeds the **stored chunk text** rather than re-chunking the
originals.

That is the correct operation here. What changes is the embedding geometry; what
must not change is the chunking. Re-chunking would alter the passages a clinician
has already reviewed, so it is deliberately out of scope for a model migration and
would need its own review.

Writing the embedding identity onto the collection afterwards means a later
mismatch is diagnosable from metadata rather than inferred from dimensionality,
which cannot distinguish two models that share a dimensionality.

The one way to rebuild
----------------------
This service is the only code outside the pipeline that rebuilds the index (round-3
review ticket 20 removed the seven root rebuild and cleanup scripts, one of which
rebuilt with an English-only model). It refuses any embedding model except the one
the study release records (`data/study/release.yaml`, models.embedding), so a
rebuild cannot bring back an index the approved release did not measure.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from retrieval_index import KNOWN_DIMENSIONS, EmbeddingIdentity, write_index_metadata

logger = logging.getLogger(__name__)

# The collection is staged under this name and swapped in only once the replacement
# has been written successfully, so a failed migration leaves the old index usable.
STAGING_SUFFIX = "__reindexing"


class ReleaseModelMismatch(ValueError):
    """A rebuild was asked for with an embedding model the study release does not record."""


def release_identity(release: Any = None) -> EmbeddingIdentity:
    """
    The embedding identity the study release records.

    Raises ReleaseModelMismatch when the release names no embedding model: with
    nothing approved to compare against, no rebuild can be shown to be the
    approved one.
    """
    if release is None:
        from study_release import get_study_release

        release = get_study_release()
    section = (release.record.get("models") or {}).get("embedding") or {}
    model = section.get("model")
    if not model:
        raise ReleaseModelMismatch(
            "the study release records no embedding model (models.embedding.model "
            "in data/study/release.yaml); refusing to rebuild the index"
        )
    dimension = section.get("dimensions") or section.get("dimension") or KNOWN_DIMENSIONS.get(model)
    if not dimension:
        raise ReleaseModelMismatch(
            f"the study release names {model} but no dimensionality; refusing to rebuild"
        )
    return EmbeddingIdentity(model=str(model), dimension=int(dimension))


def require_release_model(identity: EmbeddingIdentity, release: Any = None) -> EmbeddingIdentity:
    """Return the release's identity, or raise if `identity` is not it."""
    approved = release_identity(release)
    if identity != approved:
        raise ReleaseModelMismatch(
            f"refusing to rebuild with {identity.model} at {identity.dimension}d: "
            f"the study release records {approved.model} at {approved.dimension}d"
        )
    return approved


@dataclass
class ReindexResult:
    chunks_written: int
    duration_seconds: float
    identity: EmbeddingIdentity
    replaced: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunks_written": self.chunks_written,
            "duration_seconds": self.duration_seconds,
            "identity": self.identity.to_dict(),
            "replaced": self.replaced,
            "error": self.error,
        }


def read_stored_chunks(collection: Any) -> List[Dict[str, Any]]:
    """Read every chunk's stored text and metadata out of an existing collection."""
    data = collection.get(include=["documents", "metadatas"])
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    chunks = []
    for i, text in enumerate(documents):
        if not text:
            continue
        chunks.append({
            "text": text,
            "metadata": metadatas[i] if i < len(metadatas) else {},
        })
    return chunks


def load_model(identity: EmbeddingIdentity, local_path: Optional[str] = None,
               runtime: Optional[str] = None):
    """
    Load the embedding model for `identity`, on the configured runtime
    (retrieval/embedding_runtime.py; torch unless set).

    A local path is preferred when supplied so a 2.3 GB download is not required
    on every migration.
    """
    from retrieval.embedding_runtime import configured_runtime, load_embedding_model

    source = local_path or identity.model
    if source != identity.model and Path(source).is_dir():
        logger.info("Loading embedding model from %s", source)
    return load_embedding_model(source, runtime or configured_runtime())


def _embedding_function(identity: EmbeddingIdentity, local_model_path: Optional[str] = None):
    """
    The ChromaDB embedding function for `identity`.

    The collection must be created with the same function the pipeline queries
    with, because the pipeline passes `query_texts` and Chroma embeds them using
    the function persisted at creation time. Creating the collection without one
    records "default", and the pipeline then refuses to start with an embedding
    function conflict.
    """
    from chromadb.utils import embedding_functions

    source = local_model_path or identity.model
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=source)


def reindex_with_identity(
    chroma_path: str,
    collection_name: str,
    identity: EmbeddingIdentity,
    local_model_path: Optional[str] = None,
    batch_size: int = 32,
    runtime: Optional[str] = None,
    release: Any = None,
) -> ReindexResult:
    """
    Rebuild `collection_name` so its vectors come from `identity`.

    `identity` must be the study release's embedding identity (`release`, or the
    loaded study release when None); anything else raises ReleaseModelMismatch
    before the store is opened. Writes into a staging collection first and swaps
    only on success, so an interrupted migration cannot leave the deployment with
    no index at all.
    """
    require_release_model(identity, release)

    import chromadb

    started = time.time()

    client = chromadb.PersistentClient(path=chroma_path)
    source = client.get_collection(collection_name)
    chunks = read_stored_chunks(source)

    if not chunks:
        return ReindexResult(
            chunks_written=0,
            duration_seconds=round(time.time() - started, 1),
            identity=identity,
            replaced=False,
            error="source collection holds no chunk text to re-embed",
        )

    logger.info("Re-embedding %d chunks with %s", len(chunks), identity.model)
    model = load_model(identity, local_model_path, runtime)
    vectors = model.encode(
        [c["text"] for c in chunks],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    actual = len(vectors[0])
    if actual != identity.dimension:
        # Silently storing the wrong width would recreate exactly the class of
        # problem this migration exists to remove.
        return ReindexResult(
            chunks_written=0,
            duration_seconds=round(time.time() - started, 1),
            identity=identity,
            replaced=False,
            error=(
                f"model produced {actual}d vectors but identity declares "
                f"{identity.dimension}d; refusing to write a mismatched index"
            ),
        )

    staging_name = f"{collection_name}{STAGING_SUFFIX}"
    _drop_collection(client, staging_name)

    staging = client.create_collection(
        name=staging_name,
        embedding_function=_embedding_function(identity, local_model_path),
    )
    staging.add(
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        embeddings=[list(map(float, v)) for v in vectors],
        documents=[c["text"] for c in chunks],
        metadatas=[_clean_metadata(c["metadata"]) for c in chunks],
    )
    write_index_metadata(staging, identity)

    written = staging.count()
    logger.info("Staged %d chunks as %s", written, staging_name)

    _restore_name(client, staging_name, collection_name, written, identity,
                 local_model_path)

    return ReindexResult(
        chunks_written=written,
        duration_seconds=round(time.time() - started, 1),
        identity=identity,
        replaced=True,
    )


def reindex_for_release(
    chroma_path: str,
    collection_name: str,
    release: Any = None,
    local_model_path: Optional[str] = None,
    batch_size: int = 32,
    runtime: Optional[str] = None,
) -> ReindexResult:
    """Rebuild `collection_name` with the embedding model the study release records."""
    identity = release_identity(release)
    return reindex_with_identity(
        chroma_path=chroma_path,
        collection_name=collection_name,
        identity=identity,
        local_model_path=local_model_path,
        batch_size=batch_size,
        runtime=runtime,
        release=release,
    )


def chunk_doc_id(chunk_id: str) -> str:
    """The document a chunk belongs to: the pipeline names chunks `<doc_id>_chunk_<n>`."""
    return chunk_id.rsplit("_chunk_", 1)[0]


def prune_orphaned_chunks(
    chroma_path: str,
    collection_name: str,
    known_doc_ids: Set[str],
) -> List[str]:
    """
    Delete the chunks of documents that are no longer in the document metadata.

    Embeds nothing, so it cannot change the index's model. Returns the removed
    chunk ids. Replaces the old root `cleanup_vector_db.py`.
    """
    import chromadb

    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(collection_name)
    ids = collection.get(include=[]).get("ids") or []
    orphaned = [chunk_id for chunk_id in ids if chunk_doc_id(chunk_id) not in known_doc_ids]
    if orphaned:
        collection.delete(ids=orphaned)
    logger.info("Pruned %d orphaned chunks of %d", len(orphaned), len(ids))
    return orphaned


def _restore_name(
    client: Any,
    staging_name: str,
    collection_name: str,
    written: int,
    identity: EmbeddingIdentity,
    local_model_path: Optional[str] = None,
) -> None:
    """
    Put the staged collection back under the canonical name.

    ChromaDB's rename support has varied across versions, so this copies the
    staging collection into the canonical name rather than assuming a rename
    exists. Copying is cheap here: the corpus is a few hundred chunks.
    """
    _drop_collection(client, collection_name)

    try:
        staged = client.get_collection(staging_name)
    except Exception:
        logger.error("Staging collection vanished; index is now empty")
        return

    data = staged.get(include=["documents", "metadatas", "embeddings"])
    restored = client.create_collection(
        name=collection_name,
        embedding_function=_embedding_function(identity, local_model_path),
    )
    restored.add(
        ids=data["ids"],
        embeddings=[list(map(float, v)) for v in data["embeddings"]],
        documents=data["documents"],
        metadatas=[_clean_metadata(m) for m in (data.get("metadatas") or [])],
    )
    write_index_metadata(restored, identity)

    _drop_collection(client, staging_name)

    logger.info("Restored collection %s with %d chunks", collection_name, written)


def _clean_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    ChromaDB rejects None and non-scalar metadata values.

    Source metadata came from a different toolchain and contains both, so it is
    coerced rather than assumed well-formed.
    """
    out: Dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            out[str(key)] = value
        else:
            out[str(key)] = str(value)
    return out


def _drop_collection(client: Any, name: str) -> None:
    """Delete a collection if it exists, logging rather than swallowing the reason."""
    try:
        client.delete_collection(name)
    except Exception as exc:
        logger.debug("Collection %s not dropped: %s", name, exc)
