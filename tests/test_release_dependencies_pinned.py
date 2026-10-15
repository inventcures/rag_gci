#!/usr/bin/env python3
"""
The retrieval stack is pinned to exact versions in requirements.txt.

The embedding vectors depend on the chromadb and sentence-transformers
versions, so a study release built from an unpinned file can retrieve
differently from the one that was approved. GraphRAG was missing altogether,
so its code silently fell back to a mock (ticket 18).
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PINNED = ("graphrag", "chromadb", "sentence-transformers")


def requirement_specs(path: Path) -> dict:
    """{name: version specifier} for each requirement line in `path`."""
    specs = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.match(r"([A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*(.*)", line)
        if match:
            specs[match.group(1).lower()] = match.group(3).strip()
    return specs


class RequirementsPinTheRetrievalStack(unittest.TestCase):
    def test_graphrag_chromadb_and_sentence_transformers_are_pinned_exactly(self):
        specs = requirement_specs(ROOT / "requirements.txt")
        for name in PINNED:
            with self.subTest(name=name):
                self.assertIn(name, specs, f"{name} is not in requirements.txt")
                self.assertRegex(specs[name], r"^==\d+(\.\d+)*$",
                                 f"{name} is not pinned to an exact version")

    def test_the_simple_requirements_pin_the_same_embedding_stack(self):
        full = requirement_specs(ROOT / "requirements.txt")
        simple = requirement_specs(ROOT / "requirements_simple.txt")
        for name in ("chromadb", "sentence-transformers"):
            with self.subTest(name=name):
                self.assertEqual(simple.get(name), full.get(name))


if __name__ == "__main__":
    unittest.main()
