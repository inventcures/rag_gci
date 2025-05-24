#!/usr/bin/env python3
"""
Test TTS functionality locally
"""

import asyncio
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from whatsapp_bot import EnhancedTTSService

async def test_tts():
    """Test TTS synthesis"""
    
    print("🎤 Testing TTS Service")
    print("=" * 40)
    
    # Initialize TTS service
    tts = EnhancedTTSService()
    
    # Test text in different languages
    test_cases = [
        ("Hello, this is a test message in English.", "en"),
        ("नमस्ते, यह हिंदी में एक परीक्षण संदेश है।", "hi"),
        ("আসসালামু আলাইকুম, এটি বাংলায় একটি পরীক্ষার বার্তা।", "bn"),
    ]
    
    for i, (text, lang) in enumerate(test_cases, 1):
        print(f"\n{i}. Testing {lang.upper()}:")
        print(f"   Text: {text}")
        
        result = await tts.synthesize_speech(text, lang)
        
        if result.get("audio_available"):
            print(f"   ✅ Success: {result['audio_file']}")
            print(f"   📊 File size: {result.get('file_size', 0)} bytes")
        else:
            print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 40)
    print("🎯 TTS test completed!")

if __name__ == "__main__":
    asyncio.run(test_tts())