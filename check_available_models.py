#!/usr/bin/env python3
"""
Check available models in Groq API
"""

import requests
import os
from dotenv import load_dotenv

load_dotenv()

def check_groq_models():
    """Check available models in Groq"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ GROQ_API_KEY not found")
        return
    
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        print("🔍 Checking available models in Groq...")
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            models = response.json()
            print("✅ Available models:")
            
            reasoning_models = []
            for model in models.get("data", []):
                model_id = model.get("id", "")
                print(f"  🔹 {model_id}")
                
                # Look for reasoning-capable models
                if any(keyword in model_id.lower() for keyword in ["qwen", "reasoning", "think", "llama-3", "mixtral"]):
                    reasoning_models.append(model_id)
            
            print(f"\n🧠 Recommended reasoning models:")
            for model in reasoning_models:
                print(f"  ⭐ {model}")
                
        else:
            print(f"❌ Failed to get models: {response.status_code}")
            print(f"📄 Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_groq_models()