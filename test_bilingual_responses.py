#!/usr/bin/env python3
"""
Test script for bilingual response generation
Tests translation and multi-language support
"""

import sys
import os
import asyncio
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_rag_server import SimpleRAGPipeline


async def test_translation_functionality():
    """Test the translation functionality"""
    
    print("🌐 Testing bilingual response generation...")
    print(f"⏰ Test started at: {datetime.now()}")
    print()
    
    # Check if GROQ_API_KEY is available
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("❌ GROQ_API_KEY not found. Skipping translation tests.")
        return
    
    print(f"✅ GROQ_API_KEY found: {groq_key[:10]}...")
    
    # Initialize RAG pipeline
    try:
        rag_pipeline = SimpleRAGPipeline(data_dir="data")
        print("✅ RAG Pipeline initialized")
    except Exception as e:
        print(f"❌ Failed to initialize RAG pipeline: {e}")
        return
    
    # Test English text for translation
    english_texts = [
        "Palliative care is specialized medical care for patients with serious illness.",
        "Pain management is an important part of cancer treatment.",
        "Hospice care provides comfort and support for patients and families."
    ]
    
    # Test languages
    test_languages = ["hi", "bn", "ta", "gu"]
    
    print(f"🔍 Testing translation to {len(test_languages)} languages...")
    print()
    
    results = {}
    
    for lang in test_languages:
        results[lang] = []
        print(f"Testing translations to {lang}:")
        print("=" * 40)
        
        for i, text in enumerate(english_texts, 1):
            print(f"  Test {i}: {text[:50]}...")
            
            try:
                # Test translation
                translation_result = await rag_pipeline.translate_text(text, lang)
                
                if translation_result["status"] == "success":
                    translated = translation_result["translated_text"]
                    
                    print(f"    ✅ Translation successful")
                    print(f"    📏 Original length: {len(text)} chars")
                    print(f"    📏 Translated length: {len(translated)} chars")
                    print(f"    🎯 Under 1500 limit: {'Yes' if len(translated) <= 1500 else 'No'}")
                    print(f"    📝 Translation: {translated[:100]}...")
                    
                    results[lang].append({
                        "original": text,
                        "translated": translated,
                        "success": True,
                        "original_length": len(text),
                        "translated_length": len(translated)
                    })
                    
                else:
                    print(f"    ❌ Translation failed: {translation_result.get('error', 'Unknown error')}")
                    results[lang].append({
                        "original": text,
                        "success": False,
                        "error": translation_result.get('error', 'Unknown error')
                    })
                
            except Exception as e:
                print(f"    ❌ Exception: {e}")
                results[lang].append({
                    "original": text,
                    "success": False,
                    "error": str(e)
                })
            
            print()
        
        print()
    
    # Test full RAG pipeline with bilingual response
    print("🔍 Testing full RAG pipeline response...")
    print("=" * 50)
    
    test_query = "What is palliative care?"
    
    try:
        # Get English response
        rag_result = await rag_pipeline.query(test_query, user_id="test_user")
        
        if rag_result["status"] == "success":
            english_answer = rag_result["answer"]
            print(f"✅ RAG query successful")
            print(f"📝 English answer: {english_answer[:200]}...")
            print(f"📏 English length: {len(english_answer)} chars")
            print()
            
            # Test translation to Hindi
            print("Testing translation to Hindi:")
            translation_result = await rag_pipeline.translate_text(english_answer, "hi")
            
            if translation_result["status"] == "success":
                hindi_answer = translation_result["translated_text"]
                print(f"✅ Translation to Hindi successful")
                print(f"📝 Hindi answer: {hindi_answer[:200]}...")
                print(f"📏 Hindi length: {len(hindi_answer)} chars")
                print(f"🎯 Under 1500 limit: {'Yes' if len(hindi_answer) <= 1500 else 'No'}")
            else:
                print(f"❌ Translation failed: {translation_result.get('error')}")
        
        else:
            print(f"❌ RAG query failed: {rag_result.get('error')}")
    
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
    
    print()
    
    # Summary
    print("📊 TRANSLATION TEST SUMMARY:")
    print("=" * 50)
    
    for lang in test_languages:
        successful = sum(1 for r in results[lang] if r.get('success', False))
        total = len(results[lang])
        print(f"{lang.upper()}: {successful}/{total} translations successful")
        
        if successful > 0:
            successful_results = [r for r in results[lang] if r.get('success', False)]
            avg_length = sum(r['translated_length'] for r in successful_results) / len(successful_results)
            max_length = max(r['translated_length'] for r in successful_results)
            min_length = min(r['translated_length'] for r in successful_results)
            
            print(f"  Average length: {avg_length:.0f} chars")
            print(f"  Max length: {max_length} chars")
            print(f"  Min length: {min_length} chars")
            
            under_limit = sum(1 for r in successful_results if r['translated_length'] <= 1500)
            print(f"  Under 1500 chars: {under_limit}/{len(successful_results)}")
        
        print()
    
    print("🏁 Bilingual test completed!")


if __name__ == "__main__":
    asyncio.run(test_translation_functionality())