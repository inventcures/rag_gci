"""Configuration for Context-1 agentic retrieval."""

from dataclasses import dataclass, field
from typing import Optional

TOKEN_BUDGET = 32_768
SOFT_THRESHOLD = 24_576
HARD_CUTOFF = 28_672
MAX_SEARCH_CALLS = 8
TOKENS_PER_SEARCH = 4_000
TOP_K_RESULTS = 50


@dataclass
class Context1Config:
    """Configuration for Context-1 multi-hop retrieval agent."""

    token_budget: int = TOKEN_BUDGET
    soft_threshold: int = SOFT_THRESHOLD
    hard_cutoff: int = HARD_CUTOFF
    max_search_calls: int = MAX_SEARCH_CALLS
    tokens_per_search: int = TOKENS_PER_SEARCH
    top_k_results: int = TOP_K_RESULTS
    max_hops: int = 4
    chroma_persist_dir: str = "./data/chroma_db"
    collection_name: str = "rag_documents"
    groq_model: str = "llama-3.1-8b-instant"
    enabled: bool = True
    recall_weight: float = 16.0
    precision_weight: float = 1.0

    @classmethod
    def from_dict(cls, data: dict) -> "Context1Config":
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)
