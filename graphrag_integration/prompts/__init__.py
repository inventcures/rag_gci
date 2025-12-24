"""
Custom Prompts for Palliative Care Entity Extraction

This module contains prompt templates optimized for extracting
medical entities and relationships from palliative care documents.

Prompt Files:
    - entity_extraction.txt: Entity and relationship extraction
    - community_report.txt: Community summarization
    - summarize_descriptions.txt: Entity description consolidation
    - global_search_map.txt: Global search map phase
    - global_search_reduce.txt: Global search reduce phase
    - local_search.txt: Local search context building
"""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


def get_prompt(name: str) -> str:
    """
    Load a prompt template by name.

    Args:
        name: Prompt name (without .txt extension)

    Returns:
        Prompt template string

    Raises:
        FileNotFoundError: If prompt file doesn't exist
    """
    prompt_path = PROMPTS_DIR / f"{name}.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def list_prompts() -> list:
    """List all available prompt names."""
    return [p.stem for p in PROMPTS_DIR.glob("*.txt")]
