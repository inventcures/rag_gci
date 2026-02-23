"""
Shared utilities for PageIndex integration.

Provides token counting, tree manipulation, and context extraction helpers.
"""

import copy
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken, with char-based fallback."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    except ImportError:
        return len(text) // 4


def flatten_tree(tree: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Flatten a tree into a node_id -> node mapping for O(1) lookup."""
    node_map = {}

    def _walk(node: Dict[str, Any]):
        node_id = node.get("node_id")
        if node_id:
            node_map[node_id] = node
        for child in node.get("children", []):
            _walk(child)

    if isinstance(tree, list):
        for node in tree:
            _walk(node)
    else:
        _walk(tree)

    return node_map


def strip_text_from_tree(tree: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-copy tree removing 'text' fields.
    LLM only needs titles and summaries during tree navigation.
    """
    stripped = copy.deepcopy(tree)

    def _strip(node: Dict[str, Any]):
        node.pop("text", None)
        for child in node.get("children", []):
            _strip(child)

    if isinstance(stripped, list):
        for node in stripped:
            _strip(node)
    else:
        _strip(stripped)

    return stripped


def extract_node_texts(
    node_ids: List[str],
    node_map: Dict[str, Dict[str, Any]],
    max_tokens: int = 8000,
) -> List[Dict[str, Any]]:
    """
    Extract text content from selected nodes, respecting token budget.

    Returns list of dicts: node_id, title, text, start_page, end_page
    """
    results = []
    total_tokens = 0

    for nid in node_ids:
        node = node_map.get(nid)
        if not node:
            logger.warning(f"Node {nid} not found in tree")
            continue

        text = node.get("text", "")
        tokens = count_tokens(text)

        if total_tokens + tokens > max_tokens:
            remaining = max_tokens - total_tokens
            if remaining > 100:
                ratio = remaining / max(tokens, 1)
                truncated = text[:int(len(text) * ratio)]
                results.append({
                    "node_id": nid,
                    "title": node.get("title", ""),
                    "text": truncated + "\n[... truncated ...]",
                    "start_page": node.get("start_page", 0),
                    "end_page": node.get("end_page", 0),
                })
            break

        results.append({
            "node_id": nid,
            "title": node.get("title", ""),
            "text": text,
            "start_page": node.get("start_page", 0),
            "end_page": node.get("end_page", 0),
        })
        total_tokens += tokens

    return results


def tree_stats(tree: Dict[str, Any]) -> Dict[str, Any]:
    """Compute statistics about a tree structure."""
    node_count = 0
    max_depth = 0
    total_text_len = 0

    def _walk(node: Dict[str, Any], depth: int):
        nonlocal node_count, max_depth, total_text_len
        node_count += 1
        max_depth = max(max_depth, depth)
        total_text_len += len(node.get("text", ""))
        for child in node.get("children", []):
            _walk(child, depth + 1)

    if isinstance(tree, list):
        for node in tree:
            _walk(node, 0)
    else:
        _walk(tree, 0)

    return {
        "node_count": node_count,
        "max_depth": max_depth,
        "total_text_chars": total_text_len,
    }
