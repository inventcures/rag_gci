#!/usr/bin/env python3
"""
Test script to verify query translation functionality
"""

import asyncio
import os
from simple_rag_server import SimpleRAGPipeline

async def test_query_translation():
    """Test the new query translation feature"""
    
    # Initialize the RAG pipeline
    rag = SimpleRAGPipeline()
    
    # Test Hindi query translation
    hindi_query = "सिर दर्द का इलाज क्या है?"
    print(f"Testing Hindi query: {hindi_query}")
    
    # Test query translation
    translation_result = await rag.translate_query_to_english(hindi_query, "hi")
    print(f"Translation result: {translation_result}")
    
    if translation_result["status"] == "success":
        print(f"Original Hindi: {translation_result['original_query']}")
        print(f"Translated English: {translation_result['translated_query']}")
    else:
        print(f"Translation failed: {translation_result}")
    
    # Test English query (should not translate)
    english_query = "What is the treatment for headache?"
    print(f"\nTesting English query: {english_query}")
    
    translation_result_en = await rag.translate_query_to_english(english_query, "en")
    print(f"Translation result: {translation_result_en}")

if __name__ == "__main__":
    asyncio.run(test_query_translation())