"""
Context-1 Integration Module for Palli Sahayak.
Agentic multi-hop retrieval using ChromaDB Context-1 patterns.
Routes complex queries through iterative search with token budget management.
"""

from context1_integration.config import Context1Config, TOKEN_BUDGET, SOFT_THRESHOLD, HARD_CUTOFF
from context1_integration.complexity_classifier import QueryComplexityClassifier, QueryComplexity
from context1_integration.agent import Context1RetrievalAgent

__all__ = [
    "Context1Config",
    "QueryComplexityClassifier",
    "QueryComplexity",
    "Context1RetrievalAgent",
    "TOKEN_BUDGET",
    "SOFT_THRESHOLD",
    "HARD_CUTOFF",
]

__version__ = "1.0.0"
