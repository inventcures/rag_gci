"""
PageIndex Tree Builder for Palli Sahayak

Builds hierarchical tree indexes from documents. Supports PDF (via pymupdf
TOC + text extraction), Markdown (heading hierarchy), DOCX (heading styles),
and plain text (LLM-generated structure).

Usage:
    from pageindex_integration.config import PageIndexConfig
    from pageindex_integration.storage import PageIndexStorage
    from pageindex_integration.tree_builder import PageIndexTreeBuilder

    config = PageIndexConfig()
    storage = PageIndexStorage(config)
    builder = PageIndexTreeBuilder(config, storage)
    result = await builder.build_tree("/path/to/doc.pdf", "doc_id_123")
"""

import os
import re
import json
import logging
import tempfile
from pathlib import Path
from enum import Enum
from typing import Dict, List, Any, Optional, Callable

from pageindex_integration.config import PageIndexConfig
from pageindex_integration.storage import PageIndexStorage
from pageindex_integration.llm_adapter import LLMAdapter
from pageindex_integration.utils import count_tokens, tree_stats

logger = logging.getLogger(__name__)


class IndexingStatus(str, Enum):
    PENDING = "pending"
    BUILDING = "building"
    COMPLETED = "completed"
    FAILED = "failed"


class PageIndexTreeBuilder:
    """
    Builds hierarchical tree indexes from documents.

    The tree building process:
    1. Extract document structure (TOC, headings, or LLM-detected sections)
    2. Extract text per section with page boundaries
    3. Generate LLM summaries for each node
    4. Persist tree JSON via PageIndexStorage
    """

    def __init__(
        self,
        config: PageIndexConfig,
        storage: PageIndexStorage,
        on_progress: Optional[Callable[[str, int], None]] = None,
    ):
        self._config = config
        self._storage = storage
        self._llm = LLMAdapter(config.llm)
        self._on_progress = on_progress
        self._status: Dict[str, IndexingStatus] = {}
        self._node_counter = 0

    def _next_node_id(self) -> str:
        self._node_counter += 1
        return f"{self._node_counter:04d}"

    def _reset_counter(self) -> None:
        self._node_counter = 0

    async def build_tree(
        self,
        file_path: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build tree index for a single document.

        Returns:
            Dict with keys: status, node_count, tree_path, error
        """
        metadata = metadata or {}
        self._reset_counter()
        self._status[doc_id] = IndexingStatus.BUILDING
        self._storage.set_status(doc_id, "building")

        try:
            ext = Path(file_path).suffix.lower()

            if ext == ".pdf":
                tree = await self._build_tree_pdf(file_path)
            elif ext == ".md":
                tree = await self._build_tree_markdown(file_path)
            elif ext == ".docx":
                tree = await self._build_tree_docx(file_path)
            elif ext == ".txt":
                tree = await self._build_tree_text(file_path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")

            if self._config.tree_build.add_node_summary:
                await self._generate_summaries(tree)

            tree_path = self._storage.save_tree(doc_id, tree, {
                "filename": metadata.get("filename", Path(file_path).name),
                "page_count": metadata.get("page_count", 0),
            })

            self._status[doc_id] = IndexingStatus.COMPLETED
            stats = tree_stats(tree)

            logger.info(f"Tree built for {doc_id}: {stats['node_count']} nodes")
            return {
                "status": "completed",
                "node_count": stats["node_count"],
                "tree_path": str(tree_path),
                "error": "",
            }

        except Exception as e:
            self._status[doc_id] = IndexingStatus.FAILED
            self._storage.set_status(doc_id, "failed", str(e))
            logger.error(f"Tree build failed for {doc_id}: {e}")
            return {
                "status": "failed",
                "node_count": 0,
                "tree_path": "",
                "error": str(e),
            }

    async def _build_tree_pdf(self, file_path: str) -> Dict[str, Any]:
        """Build tree from PDF using TOC extraction or page-based fallback."""
        import fitz  # pymupdf

        doc = fitz.open(file_path)
        toc = doc.get_toc()

        if toc:
            tree = self._toc_to_tree(toc, doc)
        else:
            tree = self._pages_to_tree(doc)

        doc.close()
        return tree

    def _toc_to_tree(self, toc: List, doc) -> Dict[str, Any]:
        """Convert PDF TOC to tree structure with text from pages."""
        root = {
            "node_id": self._next_node_id(),
            "title": doc.metadata.get("title", "") or Path(doc.name).stem,
            "level": 0,
            "start_page": 1,
            "end_page": len(doc),
            "summary": "",
            "text": "",
            "children": [],
        }

        entries = []
        for i, (level, title, page_num) in enumerate(toc):
            next_page = toc[i + 1][2] if i + 1 < len(toc) else len(doc) + 1
            entries.append({
                "level": level,
                "title": title.strip(),
                "start_page": page_num,
                "end_page": next_page - 1,
            })

        stack = [root]
        for entry in entries:
            node = {
                "node_id": self._next_node_id(),
                "title": entry["title"],
                "level": entry["level"],
                "start_page": entry["start_page"],
                "end_page": entry["end_page"],
                "summary": "",
                "text": self._extract_pages_text(
                    doc, entry["start_page"], entry["end_page"]
                ),
                "children": [],
            }

            while len(stack) > 1 and stack[-1]["level"] >= entry["level"]:
                stack.pop()

            stack[-1]["children"].append(node)
            stack.append(node)

        return root

    def _pages_to_tree(self, doc) -> Dict[str, Any]:
        """Fallback: group pages into fixed-size nodes when no TOC is available."""
        max_pages = self._config.tree_build.max_pages_per_node
        root = {
            "node_id": self._next_node_id(),
            "title": doc.metadata.get("title", "") or Path(doc.name).stem,
            "level": 0,
            "start_page": 1,
            "end_page": len(doc),
            "summary": "",
            "text": "",
            "children": [],
        }

        for start in range(0, len(doc), max_pages):
            end = min(start + max_pages, len(doc))
            node = {
                "node_id": self._next_node_id(),
                "title": f"Pages {start + 1}-{end}",
                "level": 1,
                "start_page": start + 1,
                "end_page": end,
                "summary": "",
                "text": self._extract_pages_text(doc, start + 1, end),
                "children": [],
            }
            root["children"].append(node)

        return root

    def _extract_pages_text(self, doc, start_page: int, end_page: int) -> str:
        """Extract text from a page range (1-indexed)."""
        texts = []
        for page_num in range(max(0, start_page - 1), min(end_page, len(doc))):
            page = doc[page_num]
            texts.append(page.get_text())
        return "\n".join(texts)

    async def _build_tree_markdown(self, file_path: str) -> Dict[str, Any]:
        """Build tree from Markdown using heading hierarchy."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        headings = [
            (m.start(), len(m.group(1)), m.group(2).strip())
            for m in heading_pattern.finditer(content)
        ]

        root = {
            "node_id": self._next_node_id(),
            "title": Path(file_path).stem,
            "level": 0,
            "start_page": 1,
            "end_page": 1,
            "summary": "",
            "text": "",
            "children": [],
        }

        if not headings:
            root["text"] = content
            return root

        stack = [root]
        for i, (pos, level, title) in enumerate(headings):
            next_pos = headings[i + 1][0] if i + 1 < len(headings) else len(content)
            text = content[pos:next_pos].strip()
            text = re.sub(r'^#{1,6}\s+.+\n?', '', text, count=1).strip()

            node = {
                "node_id": self._next_node_id(),
                "title": title,
                "level": level,
                "start_page": 1,
                "end_page": 1,
                "summary": "",
                "text": text,
                "children": [],
            }

            while len(stack) > 1 and stack[-1]["level"] >= level:
                stack.pop()

            stack[-1]["children"].append(node)
            stack.append(node)

        return root

    async def _build_tree_docx(self, file_path: str) -> Dict[str, Any]:
        """Build tree from DOCX by converting heading styles to markdown."""
        import docx as python_docx

        doc = python_docx.Document(file_path)
        md_lines = []
        for para in doc.paragraphs:
            style = para.style.name.lower()
            if style.startswith("heading"):
                try:
                    level = int(style.replace("heading", "").strip())
                    md_lines.append(f"{'#' * level} {para.text}")
                except ValueError:
                    md_lines.append(para.text)
            else:
                md_lines.append(para.text)

        md_content = "\n".join(md_lines)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp:
            tmp.write(md_content)
            tmp_path = tmp.name

        try:
            tree = await self._build_tree_markdown(tmp_path)
        finally:
            os.unlink(tmp_path)

        tree["title"] = Path(file_path).stem
        return tree

    async def _build_tree_text(self, file_path: str) -> Dict[str, Any]:
        """Build tree from plain text using LLM-detected section boundaries."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if count_tokens(content) < 2000:
            return {
                "node_id": self._next_node_id(),
                "title": Path(file_path).stem,
                "level": 0,
                "start_page": 1,
                "end_page": 1,
                "summary": "",
                "text": content,
                "children": [],
            }

        detect_prompt = (
            "Analyze the following text and identify distinct sections or topics.\n"
            'Return a JSON array of objects with "title" and "start_char" '
            "(character offset where section begins).\n\n"
            f"Text (first 3000 chars):\n{content[:3000]}\n\n"
            "Return ONLY valid JSON array, no other text:"
        )

        try:
            response = await self._llm.chat_async(
                [{"role": "user", "content": detect_prompt}],
                max_tokens=1024,
            )
            sections = json.loads(response.strip().strip("`").strip())
            if isinstance(sections, dict):
                sections = sections.get("sections", [])
        except (json.JSONDecodeError, RuntimeError):
            chunk_size = 5000
            sections = [
                {"title": f"Section {i // chunk_size + 1}", "start_char": i}
                for i in range(0, len(content), chunk_size)
            ]

        root = {
            "node_id": self._next_node_id(),
            "title": Path(file_path).stem,
            "level": 0,
            "start_page": 1,
            "end_page": 1,
            "summary": "",
            "text": "",
            "children": [],
        }

        for i, section in enumerate(sections):
            start = section.get("start_char", 0)
            end = sections[i + 1]["start_char"] if i + 1 < len(sections) else len(content)
            text = content[start:end].strip()

            if text:
                root["children"].append({
                    "node_id": self._next_node_id(),
                    "title": section.get("title", f"Section {i + 1}"),
                    "level": 1,
                    "start_page": 1,
                    "end_page": 1,
                    "summary": "",
                    "text": text,
                    "children": [],
                })

        return root

    async def _generate_summaries(self, tree: Dict[str, Any]) -> None:
        """Generate LLM summaries for nodes that have text but no summary."""
        nodes_to_summarize = []

        def _collect(node: Dict[str, Any]):
            if node.get("text") and not node.get("summary"):
                nodes_to_summarize.append(node)
            for child in node.get("children", []):
                _collect(child)

        _collect(tree)

        if not nodes_to_summarize:
            return

        logger.info(f"Generating summaries for {len(nodes_to_summarize)} nodes")

        for i, node in enumerate(nodes_to_summarize):
            text_preview = node["text"][:2000]
            prompt = (
                "Summarize the following section in 1-2 sentences. "
                "Focus on key medical concepts, treatments, or recommendations.\n\n"
                f"Section title: {node.get('title', 'Untitled')}\n"
                f"Text:\n{text_preview}\n\n"
                "Summary:"
            )

            try:
                summary = await self._llm.chat_async(
                    [{"role": "user", "content": prompt}],
                    max_tokens=self._config.tree_build.summary_max_tokens,
                )
                node["summary"] = summary.strip()
            except Exception as e:
                logger.warning(f"Failed to generate summary for node {node['node_id']}: {e}")
                node["summary"] = node.get("title", "")

            if self._on_progress:
                pct = int((i + 1) / len(nodes_to_summarize) * 100)
                self._on_progress("Summarizing nodes", pct)

    async def batch_index(
        self,
        documents: List[Dict[str, Any]],
        on_progress: Optional[Callable[[int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Batch index multiple documents.

        Args:
            documents: List of dicts with keys: doc_id, file_path, metadata
            on_progress: Progress callback (0-100)
        """
        results = []
        for i, doc in enumerate(documents):
            result = await self.build_tree(
                file_path=doc["file_path"],
                doc_id=doc["doc_id"],
                metadata=doc.get("metadata", {}),
            )
            results.append({"doc_id": doc["doc_id"], **result})

            if on_progress:
                on_progress(int((i + 1) / len(documents) * 100))

        completed = [r for r in results if r["status"] == "completed"]
        failed = [r for r in results if r["status"] == "failed"]

        return {
            "total": len(documents),
            "completed": len(completed),
            "failed": len(failed),
            "results": results,
        }

    def get_status(self, doc_id: str) -> str:
        return self._status.get(doc_id, IndexingStatus.PENDING).value

    def __repr__(self) -> str:
        active = sum(1 for s in self._status.values() if s == IndexingStatus.BUILDING)
        return f"PageIndexTreeBuilder(active={active})"
