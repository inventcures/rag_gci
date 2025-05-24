#!/usr/bin/env python3
"""
Test Qwen-QwQ-32B integration with RAG pipeline
"""

import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

def test_qwen_api_direct():
    """Test Qwen API directly"""
    print("🧪 Testing Qwen-QwQ-32B API Integration")
    print("=" * 50)
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ GROQ_API_KEY not found in environment")
        return False
    
    # Test simple medical question
    prompt = """You are an expert medical assistant with strong reasoning capabilities.

QUESTION: What is hypertension and what causes it?

REASONING AND ANSWER:
Let me analyze this step-by-step:"""
    
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "qwen-qwq-32b",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2,
        "max_tokens": 2048
    }
    
    try:
        print("📡 Making request to Groq API with Qwen model...")
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()
            
            print("✅ Qwen-QwQ-32B response received!")
            print(f"📝 Response length: {len(answer)} characters")
            print("\n📄 Qwen Response:")
            print("-" * 30)
            print(answer[:500] + "..." if len(answer) > 500 else answer)
            print("-" * 30)
            
            return True
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"📄 Error response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_rag_with_qwen():
    """Test RAG server with Qwen integration"""
    print("\n🔍 Testing RAG Server with Qwen Integration")
    print("=" * 50)
    
    try:
        # Test server health
        health_response = requests.get("http://localhost:8002/health", timeout=5)
        if health_response.status_code != 200:
            print("❌ Server not running on port 8002")
            return False
        
        print("✅ Server is running")
        
        # Test query
        query_data = {"question": "What is a neuroma?"}
        query_response = requests.post(
            "http://localhost:8002/api/query",
            data=query_data,
            timeout=30
        )
        
        if query_response.status_code == 200:
            result = query_response.json()
            if result.get("status") == "success":
                answer = result.get("answer", "")
                print("✅ RAG query successful with Qwen!")
                print(f"📝 Answer length: {len(answer)} characters")
                print(f"📚 Sources: {len(result.get('sources', []))} documents")
                print(f"🎯 Has citation: {'retrieved from' in answer.lower()}")
                return True
            else:
                print(f"❌ RAG query failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ RAG request failed: {query_response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ RAG test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Qwen-QwQ-32B Integration Test")
    print("=" * 60)
    
    # Test 1: Direct API
    api_success = test_qwen_api_direct()
    
    # Test 2: RAG Integration
    rag_success = test_rag_with_qwen()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS:")
    print(f"🔹 Direct Qwen API: {'✅ PASS' if api_success else '❌ FAIL'}")
    print(f"🔹 RAG Integration: {'✅ PASS' if rag_success else '❌ FAIL'}")
    
    if api_success and rag_success:
        print("\n🎉 All tests passed! Qwen-QwQ-32B is working correctly.")
    elif api_success:
        print("\n⚠️ Qwen API works, but RAG server needs to be started:")
        print("   ./run_simple.sh --port 8002")
    else:
        print("\n❌ Issues detected. Check your GROQ_API_KEY and Qwen model access.")