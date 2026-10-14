#!/usr/bin/env python3
"""
Prompt-aware embedding function
===============================
The ChromaDB embedding function every index and query path uses.

Why this exists
---------------
bge-m3 is symmetric: a question and a passage go through the same `encode(text)`.
EmbeddingGemma 2 is asymmetric. It was trained with a role prefix in front of the
text, a query as `task: search result | query: ...` and a passage as
`title: none | text: ...`, and its model card warns that leaving them off reduces
quality. Nothing errors when they are missing, so the loss is silent.

Chroma's stock `SentenceTransformerEmbeddingFunction` encodes documents and queries
identically, so it cannot apply the prefixes. This one routes documents through
`__call__` and queries through `embed_query`, which Chroma calls for every
`query_texts=` search, so each existing call site gets the right prefix without
changing.

The prompt names come from config (`embedding.query_prompt_name` and
`embedding.document_prompt_name`), so a symmetric model runs through the same
class with both left unset.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings, Space
from chromadb.utils.embedding_functions import register_embedding_function

logger = logging.getLogger(__name__)

# What each model needs when config does not say, keyed by the last path segment so
# a local copy (EMBEDDING_MODEL_PATH=/srv/models/embeddinggemma-2) gets the same
# treatment as the Hugging Face id. The server builds its pipeline without
# config.yaml, so these defaults are what production actually runs with.
#
# EmbeddingGemma 2 prompt names are from its Hugging Face model card. The
# config_kwargs load the 270M text path instead of the 740M multimodal model.
MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "embeddinggemma-2": {
        "query_prompt_name": "SearchQuery",
        "document_prompt_name": "Document",
        "config_kwargs": {"vision_config": None, "audio_config": None},
    },
}


def _model_defaults(model: str) -> Dict[str, Any]:
    key = str(model).rstrip("/").split("/")[-1].strip().lower()
    return MODEL_DEFAULTS.get(key, {})


@dataclass(frozen=True)
class EmbeddingSettings:
    """How to load and prompt the configured embedding model."""

    model: str
    query_prompt_name: Optional[str] = None
    document_prompt_name: Optional[str] = None
    # Passed to the model config on load. EmbeddingGemma 2 takes
    # {"vision_config": None, "audio_config": None} to load the 270M text path
    # instead of the full 740M multimodal model.
    config_kwargs: Dict[str, Any] = field(default_factory=dict)


def load_embedding_settings(
    config: Optional[Dict[str, Any]],
    model_source: Optional[str] = None,
) -> EmbeddingSettings:
    """
    Read the embedding section of config.yaml.

    `model_source` overrides where the weights load from (EMBEDDING_MODEL_PATH)
    without changing the prompts, which belong to the model, not to its location.
    """
    from retrieval_index import load_identity_from_config

    section = (config or {}).get("embedding") or {}
    identity = load_identity_from_config(config)
    defaults = _model_defaults(identity.model)

    def pick(key: str) -> Any:
        # A key present in config wins, even when set to null, so a symmetric
        # model can switch prompts off explicitly.
        return section[key] if key in section else defaults.get(key)

    return EmbeddingSettings(
        model=model_source or identity.model,
        query_prompt_name=pick("query_prompt_name") or None,
        document_prompt_name=pick("document_prompt_name") or None,
        config_kwargs=dict(pick("config_kwargs") or {}),
    )


@register_embedding_function
class PromptedSentenceTransformerEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    A sentence-transformers embedding function with separate query and document prompts.

    Vectors are always L2-normalised, so cosine, inner product and L2 rank
    identically whatever space the collection uses.
    """

    # One loaded model per (source, config) across every collection in the process.
    _models: Dict[str, Any] = {}

    def __init__(
        self,
        model_name: str,
        query_prompt_name: Optional[str] = None,
        document_prompt_name: Optional[str] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        model: Any = None,
    ):
        self.model_name = model_name
        self.query_prompt_name = query_prompt_name
        self.document_prompt_name = document_prompt_name
        self.config_kwargs = dict(config_kwargs or {})
        self.device = device
        self._model = model

    @classmethod
    def from_settings(
        cls, settings: EmbeddingSettings, model: Any = None
    ) -> "PromptedSentenceTransformerEmbeddingFunction":
        return cls(
            model_name=settings.model,
            query_prompt_name=settings.query_prompt_name,
            document_prompt_name=settings.document_prompt_name,
            config_kwargs=settings.config_kwargs,
            model=model,
        )

    @property
    def model(self) -> Any:
        """The underlying SentenceTransformer, loaded on first use."""
        if self._model is None:
            key = f"{self.model_name}|{sorted(self.config_kwargs.items())}"
            if key not in self._models:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model %s", self.model_name)
                kwargs: Dict[str, Any] = {"device": self.device}
                if self.config_kwargs:
                    kwargs["config_kwargs"] = self.config_kwargs
                self._models[key] = SentenceTransformer(self.model_name, **kwargs)
            self._model = self._models[key]
        return self._model

    def _encode(self, texts: List[str], prompt_name: Optional[str]) -> Embeddings:
        kwargs: Dict[str, Any] = {
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": False,
        }
        if prompt_name:
            kwargs["prompt_name"] = prompt_name
        vectors = self.model.encode(list(texts), **kwargs)
        return [np.asarray(v, dtype=np.float32) for v in vectors]

    def __call__(self, input: Documents) -> Embeddings:
        """Embed passages for the index."""
        return self._encode(list(input), self.document_prompt_name)

    def embed_query(self, input: Documents) -> Embeddings:
        """Embed search queries. Chroma calls this for `query_texts=`."""
        return self._encode(list(input), self.query_prompt_name)

    @staticmethod
    def name() -> str:
        return "prompted_sentence_transformer"

    def default_space(self) -> Space:
        return "cosine"

    def supported_spaces(self) -> List[Space]:
        return ["cosine", "l2", "ip"]

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "query_prompt_name": self.query_prompt_name,
            "document_prompt_name": self.document_prompt_name,
            "config_kwargs": self.config_kwargs,
            "device": self.device,
        }

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "PromptedSentenceTransformerEmbeddingFunction":
        return PromptedSentenceTransformerEmbeddingFunction(
            model_name=config["model_name"],
            query_prompt_name=config.get("query_prompt_name"),
            document_prompt_name=config.get("document_prompt_name"),
            config_kwargs=config.get("config_kwargs") or {},
            device=config.get("device") or "cpu",
        )

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        if not config.get("model_name"):
            raise ValueError("prompted_sentence_transformer requires model_name")

    def validate_config_update(
        self, old_config: Dict[str, Any], new_config: Dict[str, Any]
    ) -> None:
        # Changing the model or its prompts changes the geometry of every stored
        # vector, which is a reindex, not a config update.
        for key in ("model_name", "query_prompt_name", "document_prompt_name"):
            if key in new_config and new_config[key] != old_config.get(key):
                raise ValueError(
                    f"Changing {key} requires a reindex: "
                    "scripts/reindex_with_configured_model.py"
                )
