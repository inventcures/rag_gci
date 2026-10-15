"""
An uploaded document is extracted and chunked, or refused with a reason.

Moved from the root `test_upload.py` (round-3 ticket 20), a debugging script that
copied a blank PDF into uploads/, printed what the processor returned and
asserted nothing. The processor is what every admin upload goes through before
the chunks are embedded.

What these pin down, through SimpleDocumentProcessor.process_file:

  * a long Hindi text document becomes several chunks, each taken from the
    document, together covering all of it
  * a PDF with no text (the old script's blank page) is refused, not indexed empty
  * an unsupported file type is refused by name
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_rag_server import SimpleDocumentProcessor  # noqa: E402

BLANK_PDF = (
    b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n"
    b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n"
    b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\nxref\n0 4\n"
    b"0000000000 65535 f\n0000000009 00000 n\n0000000074 00000 n\n0000000120 00000 n\n"
    b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n178\n%%EOF"
)
HINDI_TEXT = "दबाव के घावों से बचने के लिए रोगी को बार-बार करवट दिलाएं। " * 80


class UploadedDocuments(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="upload-"))
        self.processor = SimpleDocumentProcessor()

    def write(self, name: str, content) -> str:
        path = self.dir / name
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        return str(path)

    def test_a_long_hindi_document_is_chunked_without_losing_text(self):
        result = self.processor.process_file(self.write("care.md", HINDI_TEXT))
        self.assertEqual(result["status"], "success", result)
        self.assertEqual(result["text"], HINDI_TEXT)
        chunks = result["chunks"]
        self.assertGreater(len(chunks), 1)
        self.assertEqual(result["chunk_count"], len(chunks))
        for chunk in chunks:
            self.assertIn(chunk.strip(), HINDI_TEXT)
        self.assertTrue(HINDI_TEXT.strip().startswith(chunks[0].strip()))
        self.assertTrue(HINDI_TEXT.strip().endswith(chunks[-1].strip()))

    def test_a_pdf_with_no_text_is_refused(self):
        result = self.processor.process_file(self.write("blank.pdf", BLANK_PDF))
        self.assertEqual(result["status"], "error")
        self.assertIn("No text extracted", result["error"])

    def test_an_unsupported_file_type_is_refused_by_name(self):
        result = self.processor.process_file(self.write("tool.exe", b"MZ"))
        self.assertEqual(result["status"], "error")
        self.assertIn(".exe", result["error"])


if __name__ == "__main__":
    unittest.main()
