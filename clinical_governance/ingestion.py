"""Local, evidence-preserving ingestion. No automatic clinical interpretation."""

import hashlib
from typing import Any, Dict

from .models import GovernanceError
from .store import require


MAX_SOURCE_BYTES = 20 * 1024 * 1024


def ingest_text(content: bytes, media_type: str = "text/plain") -> Dict[str, Any]:
    require(0 < len(content) <= MAX_SOURCE_BYTES, "source_size_limit", 422)
    try:
        text = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise GovernanceError("utf8_source_required", 422) from None
    require(bool(text.strip()) and "\x00" not in text, "invalid_source_text", 422)
    return {"original": content, "normalized_text": text, "media_type": media_type,
            "normalizer": "utf8-identity-codepoints-v1", "pages": [], "blocks": [],
            "coverage_gaps": ["clinical_semantics_and_completeness_require_review"]}


def ingest_pdf(content: bytes) -> Dict[str, Any]:
    require(0 < len(content) <= MAX_SOURCE_BYTES and content.startswith(b"%PDF-"), "invalid_pdf", 422)
    try:
        import pymupdf
    except ImportError:
        raise GovernanceError("pdf_dependency_unavailable", 503) from None
    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            require(not document.needs_pass, "encrypted_pdf_not_supported", 422)
            require(0 < len(document) <= 200, "pdf_page_limit", 422)
            parts, pages, blocks, gaps = [], [], [], ["clinical_semantics_and_completeness_require_review"]
            offset = 0
            for index, page in enumerate(document):
                header = f"## Page {index + 1}\n"
                parts.append(header)
                offset += len(header)
                page_start, count = offset, 0
                for block_index, block in enumerate(page.get_text("blocks", sort=True)):
                    if len(block) > 6 and block[6] != 0:
                        continue
                    text = block[4]
                    if not isinstance(text, str) or not text.strip():
                        continue
                    start = offset
                    parts.append(text)
                    offset += len(text)
                    blocks.append({"id": f"p{index + 1}-b{block_index}", "page": index + 1,
                                   "start": start, "end": offset, "bbox": list(block[:4]), "kind": "text"})
                    parts.append("\n")
                    offset += 1
                    count += 1
                pages.append({"page": index + 1, "start": page_start, "end": offset,
                              "width": page.rect.width, "height": page.rect.height})
                if not count:
                    gaps.append(f"page_{index + 1}:no_text_ocr_required")
                if page.get_images():
                    gaps.append(f"page_{index + 1}:images_or_scans_require_review")
                try:
                    tables = page.find_tables().tables
                    if tables:
                        # Preserve original cells/geometry rather than invent links
                        # between text offsets and a table detector's cells.
                        pages[-1]["tables"] = [{"bbox": list(table.bbox), "cells": table.extract()} for table in tables]
                        gaps.append(f"page_{index + 1}:table_header_footnote_relationships_require_review")
                except Exception:
                    gaps.append(f"page_{index + 1}:table_detection_unavailable")
                require(offset <= 2_000_000, "normalized_text_limit", 422)
            text = "".join(parts)
            return {"original": content, "normalized_text": text, "media_type": "application/pdf",
                    "normalizer": f"pymupdf-{pymupdf.VersionBind}-sorted-blocks-codepoints-v1",
                    "pages": pages, "blocks": blocks, "coverage_gaps": gaps}
    except GovernanceError:
        raise
    except Exception:
        raise GovernanceError("pdf_parse_failed", 422) from None


def render_pdf_page(content: bytes, page: int) -> bytes:
    """Render a bounded PNG; never embed an uploaded PDF's active content."""
    try:
        import pymupdf
        with pymupdf.open(stream=content, filetype="pdf") as document:
            require(type(page) is int and 1 <= page <= len(document), "page_unavailable", 404)
            selected = document[page - 1]
            scale = min(1.5, 1600 / max(selected.rect.width, selected.rect.height))
            return selected.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).tobytes("png")
    except GovernanceError:
        raise
    except Exception:
        raise GovernanceError("pdf_render_failed", 422) from None
