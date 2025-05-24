#!/usr/bin/env python3
"""
Test script to verify citation functionality in RAG responses
"""

import asyncio
import requests
import json

async def test_citations():
    """Test the citation functionality"""
    
    base_url = "http://localhost:8002"
    
    # Test queries
    test_queries = [
        "What is a neuroma?",  # Should find answer in medical corpus
        "How to treat cancer?",  # Should find answer 
        "What is the capital of France?",  # Should NOT find answer (non-medical)
        "What are bedsores?",  # Should find answer
        "How to cook pasta?",  # Should NOT find answer (non-medical)
    ]
    
    print("🧪 Testing Citation Functionality")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing: '{query}'")
        print("-" * 30)
        
        try:
            # Make API call
            response = requests.post(
                f"{base_url}/query",
                json={"question": query},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('answer', '')
                sources = result.get('sources', [])
                
                print(f"📋 Answer: {answer}")
                print(f"📚 Sources: {len(sources)} documents")
                
                # Check citation format
                if '{retrieved from:' in answer.lower():
                    print("✅ Citation found")
                elif "we are afraid, we could not find" in answer.lower():
                    print("✅ Proper no-answer response")
                else:
                    print("❌ Missing citation")
                    
            else:
                print(f"❌ API Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Citation test completed!")

def test_health_endpoint():
    """Test if server is running"""
    try:
        response = requests.get("http://localhost:8002/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Server is running")
            print(f"📊 Documents: {data.get('stats', {}).get('total_documents', 0)}")
            print(f"📄 Chunks: {data.get('stats', {}).get('total_chunks', 0)}")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server not accessible: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing RAG Server with Citations")
    print("=" * 50)
    
    if test_health_endpoint():
        print("\n🚀 Running citation tests...")
        asyncio.run(test_citations())
    else:
        print("\n💡 Please start the server first:")
        print("   ./run_simple.sh --port 8002")