"""
PageIndex Storage Layer for Palli Sahayak

Manages tree index JSON files on disk. Each document gets a
{doc_id}_tree.json in data/pageindex/trees/. A master index at
data/pageindex/index.json maps doc_ids to metadata.

Usage:
    from pageindex_integration.config import PageIndexConfig
    from pageindex_integration.storage import PageIndexStorage

    config = PageIndexConfig()
    storage = PageIndexStorage(config)
    storage.save_tree("doc1", tree_dict, {"filename": "guide.pdf", "page_count": 100})
    tree = storage.load_tree("doc1")
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

from pageindex_integration.config import PageIndexConfig

logger = logging.getLogger(__name__)


@dataclass
class TreeIndexEntry:
    """Metadata for a single tree index."""
    doc_id: str
    filename: str
    tree_path: str
    indexed_at: str
    node_count: int
    page_count: int
    status: str
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PageIndexStorage:
    """
    Manages tree index JSON files on disk.

    Each document gets a {doc_id}_tree.json in data/pageindex/trees/.
    A master index at data/pageindex/index.json maps doc_ids to metadata.
    """

    def __init__(self, config: PageIndexConfig):
        self._config = config
        config.ensure_directories()
        self._index: Dict[str, TreeIndexEntry] = {}
        self._load_index()

    def _load_index(self) -> None:
        if self._config.index_file.exists():
            try:
                with open(self._config.index_file, 'r') as f:
                    raw = json.load(f)
                self._index = {
                    k: TreeIndexEntry(**v) for k, v in raw.items()
                }
                logger.info(f"Loaded {len(self._index)} tree index entries")
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Failed to load tree index: {e}")
                self._index = {}
        else:
            self._index = {}

    def _save_index(self) -> None:
        with open(self._config.index_file, 'w') as f:
            json.dump(
                {k: v.to_dict() for k, v in self._index.items()},
                f, indent=2
            )

    def save_tree(
        self,
        doc_id: str,
        tree: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Path:
        """Persist tree JSON and update index."""
        tree_filename = f"{doc_id}_tree.json"
        tree_path = self._config.trees_dir / tree_filename

        with open(tree_path, 'w') as f:
            json.dump(tree, f, indent=2)

        from pageindex_integration.utils import tree_stats
        stats = tree_stats(tree)

        self._index[doc_id] = TreeIndexEntry(
            doc_id=doc_id,
            filename=metadata.get("filename", ""),
            tree_path=str(tree_path),
            indexed_at=datetime.now().isoformat(),
            node_count=stats["node_count"],
            page_count=metadata.get("page_count", 0),
            status="completed",
        )
        self._save_index()
        logger.info(f"Saved tree for {doc_id}: {stats['node_count']} nodes")
        return tree_path

    def load_tree(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Load tree JSON by doc_id. Returns None if not found."""
        entry = self._index.get(doc_id)
        if not entry or entry.status != "completed":
            return None

        tree_path = Path(entry.tree_path)
        if not tree_path.exists():
            logger.warning(f"Tree file missing for {doc_id}: {tree_path}")
            return None

        with open(tree_path, 'r') as f:
            return json.load(f)

    def has_tree(self, doc_id: str) -> bool:
        entry = self._index.get(doc_id)
        return entry is not None and entry.status == "completed"

    def delete_tree(self, doc_id: str) -> bool:
        entry = self._index.get(doc_id)
        if not entry:
            return False

        tree_path = Path(entry.tree_path)
        if tree_path.exists():
            tree_path.unlink()

        del self._index[doc_id]
        self._save_index()
        return True

    def set_status(self, doc_id: str, status: str, error: str = "") -> None:
        """Update or create an indexing status entry for a document."""
        if doc_id in self._index:
            self._index[doc_id].status = status
            self._index[doc_id].error = error
        else:
            self._index[doc_id] = TreeIndexEntry(
                doc_id=doc_id,
                filename="",
                tree_path="",
                indexed_at=datetime.now().isoformat(),
                node_count=0,
                page_count=0,
                status=status,
                error=error,
            )
        self._save_index()

    def list_trees(self) -> List[TreeIndexEntry]:
        return list(self._index.values())

    def get_stats(self) -> Dict[str, Any]:
        completed = [e for e in self._index.values() if e.status == "completed"]
        return {
            "total_trees": len(self._index),
            "completed": len(completed),
            "building": sum(1 for e in self._index.values() if e.status == "building"),
            "failed": sum(1 for e in self._index.values() if e.status == "failed"),
            "total_nodes": sum(e.node_count for e in completed),
            "total_pages": sum(e.page_count for e in completed),
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"PageIndexStorage(trees={stats['total_trees']}, nodes={stats['total_nodes']})"
