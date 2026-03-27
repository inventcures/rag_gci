"""
Offline cache module for Palli Sahayak mobile clients.

Generates pre-computed cache bundles containing:
- Top 20 clinical queries with responses (in all supported languages)
- Top 50 symptom-treatment pairs from knowledge graph
- Emergency keywords in all supported languages
"""

from offline.cache_builder import CacheBundleBuilder

__all__ = ["CacheBundleBuilder"]
