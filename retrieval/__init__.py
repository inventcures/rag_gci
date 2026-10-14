"""
Answer-path retrieval: shared Reciprocal Rank Fusion, the knowledge-graph and
GraphRAG retrievers, the knowledge-graph cross-reference check (protocol K26) and
the per-interaction retrieval record. See docs/answer_path_retrieval.md.

Submodules are imported directly (`from retrieval.fusion import ...`) so that
importing one does not pull in the others' dependencies.
"""
