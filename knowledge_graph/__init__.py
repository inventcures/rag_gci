"""
Knowledge Graph Module for Palli Sahayak RAG System

This module provides knowledge graph capabilities for the palliative care RAG system,
inspired by the OncoGraph architecture (https://github.com/ib565/OncoGraph).

Features:
- Neo4j integration for graph storage
- Entity extraction from medical documents
- Cypher query generation from natural language
- Graph visualization for admin UI
- Integration with existing RAG pipeline

Architecture:
    Documents → Entity Extractor → Graph Builder → Neo4j
                                                    ↓
    User Query → Cypher Generator → Neo4j → Results
                                                    ↓
                                            Visualizer → Graph UI

Usage:
    from knowledge_graph import KnowledgeGraphRAG, Neo4jClient

    # Initialize
    kg = KnowledgeGraphRAG()

    # Extract entities from document
    entities = await kg.extract_entities(document_text)

    # Query the graph
    results = await kg.query("What medications help with pain?")

    # Get visualization data
    viz_data = kg.get_visualization_data(results)
"""

from .neo4j_client import Neo4jClient, Neo4jConfig
from .entity_extractor import EntityExtractor, Entity, Relationship
from .graph_builder import GraphBuilder
from .cypher_generator import CypherGenerator, CypherValidator
from .visualizer import GraphVisualizer, VisualizationData
from .kg_rag import KnowledgeGraphRAG

__all__ = [
    # Core classes
    "KnowledgeGraphRAG",
    "Neo4jClient",
    "Neo4jConfig",
    "EntityExtractor",
    "GraphBuilder",
    "CypherGenerator",
    "CypherValidator",
    "GraphVisualizer",
    # Data classes
    "Entity",
    "Relationship",
    "VisualizationData",
]

__version__ = "1.0.0"
