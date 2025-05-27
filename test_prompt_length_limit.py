#!/usr/bin/env python3
"""
Test script for prompt-based length limiting
Tests that the LLM generates responses under 1500 characters
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


async def test_prompt_length_limiting():
    """Test that the LLM respects the 1500 character limit in prompts"""
    
    print("🧪 Testing prompt-based length limiting...")
    print(f"⏰ Test started at: {datetime.now()}")
    print()
    
    # Check if GROQ_API_KEY is available
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("❌ GROQ_API_KEY not found. Skipping LLM tests.")
        return
    
    print(f"✅ GROQ_API_KEY found: {groq_key[:10]}...")
    
    # Initialize RAG pipeline
    try:
        rag_pipeline = SimpleRAGPipeline(data_dir="data")
        print("✅ RAG Pipeline initialized")
    except Exception as e:
        print(f"❌ Failed to initialize RAG pipeline: {e}")
        return
    
    # Test queries that might generate long responses
    test_queries = [
        "What is palliative care and how does it work?",
        "Tell me everything about cancer treatment options.",
        "Explain all the symptoms and treatments for chronic pain.",
        "What are all the different types of medical procedures available?",
        "Describe comprehensive cancer care from diagnosis to treatment."
    ]
    
    print(f"🔍 Testing {len(test_queries)} queries...")
    print()
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"Test {i}/{len(test_queries)}: {query[:50]}...")
        
        try:
            # Query the RAG pipeline
            result = await rag_pipeline.query(query, user_id="test_user")
            
            if result["status"] == "success":
                answer = result["answer"]
                answer_length = len(answer)
                
                print(f"  ✅ Query successful")
                print(f"  📏 Response length: {answer_length} characters")
                print(f"  🎯 Under 1500 limit: {'Yes' if answer_length <= 1500 else 'No'}")
                
                # Check for citations
                has_citation = '{' in answer and '}' in answer
                print(f"  📚 Has citation: {'Yes' if has_citation else 'No'}")
                
                # Check citation format (short format)
                if has_citation:
                    citation_short = '_pg' in answer
                    print(f"  📖 Short citation format: {'Yes' if citation_short else 'No'}")
                
                print(f"  📝 Response preview: {answer[:100]}...")
                
                if answer_length > 1500:
                    print(f"  ⚠️ WARNING: Response exceeds 1500 characters!")
                    print(f"  📄 Full response: {answer}")
                
                results.append({
                    "query": query,
                    "length": answer_length,
                    "under_limit": answer_length <= 1500,
                    "has_citation": has_citation,
                    "answer": answer
                })
                
            else:
                print(f"  ❌ Query failed: {result.get('error', 'Unknown error')}")
                results.append({
                    "query": query,
                    "length": 0,
                    "under_limit": True,
                    "has_citation": False,
                    "error": result.get('error', 'Unknown error')
                })
        
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            results.append({
                "query": query,
                "length": 0,
                "under_limit": True,
                "has_citation": False,
                "error": str(e)
            })
        
        print()
    
    # Summary
    print("📊 TEST SUMMARY:")
    print("=" * 50)
    
    successful_queries = [r for r in results if 'error' not in r]
    under_limit_count = sum(1 for r in successful_queries if r['under_limit'])
    with_citations = sum(1 for r in successful_queries if r['has_citation'])
    
    print(f"Total queries: {len(test_queries)}")
    print(f"Successful queries: {len(successful_queries)}")
    print(f"Responses under 1500 chars: {under_limit_count}/{len(successful_queries)}")
    print(f"Responses with citations: {with_citations}/{len(successful_queries)}")
    
    if successful_queries:
        avg_length = sum(r['length'] for r in successful_queries) / len(successful_queries)
        max_length = max(r['length'] for r in successful_queries)
        min_length = min(r['length'] for r in successful_queries)
        
        print(f"Average response length: {avg_length:.0f} characters")
        print(f"Max response length: {max_length} characters")
        print(f"Min response length: {min_length} characters")
    
    print()
    
    if under_limit_count == len(successful_queries) and len(successful_queries) > 0:
        print("✅ ALL RESPONSES UNDER 1500 CHARACTER LIMIT!")
    elif len(successful_queries) == 0:
        print("⚠️ No successful queries to test")
    else:
        print(f"❌ {len(successful_queries) - under_limit_count} responses exceeded the limit")
    
    print()
    print("🏁 Test completed!")


if __name__ == "__main__":
    asyncio.run(test_prompt_length_limiting())