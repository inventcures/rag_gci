#!/usr/bin/env python3
"""
Tests for the prompt-aware embedding function (ADR 0008).

EmbeddingGemma 2 expects a query prefix on questions and a document prefix on
passages. Missing prefixes do not error; they quietly lower retrieval quality. So
the routing is tested end to end through Chroma: passages added to a collection
must be embedded with the document prompt, and `query_texts=` searches with the
query prompt.
"""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval_embeddings import (  # noqa: E402
    PromptedSentenceTransformerEmbeddingFunction,
    load_embedding_settings,
)


class FakeModel:
    """Records which prompt each encode call used. Returns unit vectors."""

    def __init__(self, dim=8):
        self.dim = dim
        self.calls = []

    def encode(self, texts, prompt_name=None, **kwargs):
        self.calls.append((prompt_name, list(texts)))
        return [np.ones(self.dim) / np.sqrt(self.dim) for _ in texts]


class TestSettings(unittest.TestCase):
    def test_default_model_gets_query_and_document_prompts(self):
        s = load_embedding_settings({})
        self.assertEqual(s.model, "google/embeddinggemma-2")
        self.assertEqual(s.query_prompt_name, "SearchQuery")
        self.assertEqual(s.document_prompt_name, "Document")

    def test_default_model_loads_text_path_only(self):
        s = load_embedding_settings({})
        self.assertEqual(s.config_kwargs, {"vision_config": None, "audio_config": None})

    def test_local_copy_keeps_the_model_prompts(self):
        s = load_embedding_settings({}, model_source="/srv/models/embeddinggemma-2")
        self.assertEqual(s.model, "/srv/models/embeddinggemma-2")
        self.assertEqual(s.query_prompt_name, "SearchQuery")

    def test_symmetric_model_gets_no_prompts(self):
        s = load_embedding_settings({"embedding": {"model": "BAAI/bge-m3", "dimension": 1024}})
        self.assertIsNone(s.query_prompt_name)
        self.assertIsNone(s.document_prompt_name)
        self.assertEqual(s.config_kwargs, {})

    def test_config_can_switch_prompts_off_explicitly(self):
        s = load_embedding_settings({"embedding": {"query_prompt_name": None}})
        self.assertIsNone(s.query_prompt_name)
        self.assertEqual(s.document_prompt_name, "Document")


class TestPromptRouting(unittest.TestCase):
    def test_documents_and_queries_use_different_prompts(self):
        model = FakeModel()
        ef = PromptedSentenceTransformerEmbeddingFunction.from_settings(
            load_embedding_settings({}), model=model
        )
        ef(["passage"])
        ef.embed_query(["question"])
        self.assertEqual(model.calls, [("Document", ["passage"]), ("SearchQuery", ["question"])])

    def test_symmetric_model_passes_no_prompt(self):
        model = FakeModel()
        ef = PromptedSentenceTransformerEmbeddingFunction("BAAI/bge-m3", model=model)
        ef(["passage"])
        ef.embed_query(["question"])
        self.assertEqual([c[0] for c in model.calls], [None, None])

    def test_chroma_routes_add_and_query_through_the_right_prompt(self):
        import chromadb

        tmp = tempfile.mkdtemp()
        try:
            model = FakeModel()
            ef = PromptedSentenceTransformerEmbeddingFunction.from_settings(
                load_embedding_settings({}), model=model
            )
            client = chromadb.PersistentClient(path=tmp)
            col = client.create_collection("documents", embedding_function=ef)
            col.add(ids=["a"], documents=["Always give a laxative with morphine."])
            col.query(query_texts=["मॉर्फीन के साथ क्या देना चाहिए?"], n_results=1)
            self.assertEqual([c[0] for c in model.calls], ["Document", "SearchQuery"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_config_round_trips(self):
        ef = PromptedSentenceTransformerEmbeddingFunction.from_settings(load_embedding_settings({}))
        rebuilt = PromptedSentenceTransformerEmbeddingFunction.build_from_config(ef.get_config())
        self.assertEqual(rebuilt.get_config(), ef.get_config())

    def test_changing_prompts_is_refused_as_a_config_update(self):
        ef = PromptedSentenceTransformerEmbeddingFunction.from_settings(load_embedding_settings({}))
        new = dict(ef.get_config(), query_prompt_name=None)
        with self.assertRaises(ValueError):
            ef.validate_config_update(ef.get_config(), new)


class TestAdoptionGate(unittest.TestCase):
    """ADR 0008: EmbeddingGemma 2 must match or beat bge-m3 overall and per language."""

    @staticmethod
    def _result(model, mean, per_lang):
        return {
            "model": model,
            "overall": {"hit@5": mean},
            "per_language": {l: {"hit@5": v} for l, v in per_lang.items()},
        }

    def setUp(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        "evaluation", "retrieval_eval"))
        import run_eval

        self.run_eval = run_eval
        self.langs = run_eval.LANGUAGES

    def test_passes_when_equal_or_better_everywhere(self):
        inc = self._result("bge-m3", 0.96, {l: 0.89 for l in self.langs})
        cand = self._result("embeddinggemma-2", 0.97, {l: 0.89 for l in self.langs})
        self.assertEqual(self.run_eval.gate([cand, inc]), [])

    def test_fails_on_a_single_language_regression_despite_a_better_mean(self):
        inc = self._result("bge-m3", 0.96, {l: 0.89 for l in self.langs})
        per = {l: 1.0 for l in self.langs}
        per["od-IN"] = 0.78
        cand = self._result("embeddinggemma-2", 0.98, per)
        failures = self.run_eval.gate([cand, inc])
        self.assertEqual(len(failures), 1)
        self.assertIn("od-IN", failures[0])

    def test_fails_on_a_lower_mean(self):
        inc = self._result("bge-m3", 0.96, {l: 0.89 for l in self.langs})
        cand = self._result("embeddinggemma-2", 0.95, {l: 0.89 for l in self.langs})
        self.assertTrue(self.run_eval.gate([cand, inc]))


if __name__ == "__main__":
    unittest.main()
