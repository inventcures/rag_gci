#!/usr/bin/env python3
"""
Test script for WhatsApp length limiting functionality
Tests the _ensure_whatsapp_length_limit function
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from whatsapp_bot import EnhancedWhatsAppBot, EnhancedSTTService, EnhancedTTSService


class MockRAGPipeline:
    """Mock RAG pipeline for testing"""
    async def query(self, text, user_id=None):
        return {"status": "success", "answer": "Mock response"}


def test_length_limiting():
    """Test the WhatsApp length limiting functionality"""
    
    # Create mock services
    stt_service = EnhancedSTTService()
    tts_service = EnhancedTTSService()
    rag_pipeline = MockRAGPipeline()
    
    # Create WhatsApp bot
    bot = EnhancedWhatsAppBot(rag_pipeline, stt_service, tts_service)
    
    print("🧪 Testing WhatsApp message length limiting...")
    print(f"📏 Character limit: {bot.whatsapp_char_limit}")
    print()
    
    # Test 1: Short message (should pass through unchanged)
    short_msg = "This is a short message that should pass through unchanged."
    result1 = bot._ensure_whatsapp_length_limit(short_msg)
    print(f"Test 1 - Short message:")
    print(f"  Input length: {len(short_msg)}")
    print(f"  Output length: {len(result1)}")
    print(f"  Changed: {'Yes' if result1 != short_msg else 'No'}")
    print(f"  Output: {result1[:100]}...")
    print()
    
    # Test 2: Long message (should be truncated)
    long_msg = """पैलियेटिव केयर एक विशेषज्ञ चिकित्सा देखभाल है जो गंभीर बीमारी से पीड़ित रोगियों के लिए दर्द और लक्षणों से राहत प्रदान करती है। यह एक समग्र दृष्टिकोण है जो न केवल शारीरिक लक्षणों का इलाज करता है बल्कि मानसिक, सामाजिक और आध्यात्मिक जरूरतों का भी ख्याल रखता है। पैलियेटिव केयर टीम में डॉक्टर, नर्स, सामाजिक कार्यकर्ता, मनोचिकित्सक, चैप्लिन और अन्य विशेषज्ञ शामिल होते हैं। यह देखभाल किसी भी उम्र के रोगियों को प्रदान की जा सकती है और इसका उद्देश्य जीवन की गुणवत्ता में सुधार करना है। पैलियेटिव केयर में दर्द प्रबंधन, सांस की तकलीफ का इलाज, मतली और उल्टी का नियंत्रण, थकान का प्रबंधन शामिल है। यह परिवार के सदस्यों को भी सहायता प्रदान करती है और उन्हें मुकाबला करने की रणनीति सिखाती है। पैलियेटिव केयर का लक्ष्य रोगी और परिवार के लिए सबसे अच्छा संभावित जीवन स्तर प्रदान करना है। यह केयर घर पर, अस्पताल में, या विशेष पैलियेटिव केयर सुविधाओं में प्रदान की जा सकती है। पैलियेटिव केयर में नेता टीम के साथ मिलकर व्यक्तिगत देखभाल योजना बनाते हैं जो रोगी की विशिष्ट आवश्यकताओं के अनुकूल होती है। यह देखभाल निरंतर होती है और रोगी की स्थिति के अनुसार समायोजित की जाती है।""" * 3  # Make it very long
    
    result2 = bot._ensure_whatsapp_length_limit(long_msg)
    print(f"Test 2 - Long message:")
    print(f"  Input length: {len(long_msg)}")
    print(f"  Output length: {len(result2)}")
    print(f"  Truncated: {'Yes' if len(result2) < len(long_msg) else 'No'}")
    print(f"  Under limit: {'Yes' if len(result2) <= bot.whatsapp_char_limit else 'No'}")
    print(f"  Output preview: {result2[:200]}...")
    print(f"  Output ending: ...{result2[-100:]}")
    print()
    
    # Test 3: Message exactly at limit
    limit_msg = "x" * bot.whatsapp_char_limit
    result3 = bot._ensure_whatsapp_length_limit(limit_msg)
    print(f"Test 3 - Message at exact limit:")
    print(f"  Input length: {len(limit_msg)}")
    print(f"  Output length: {len(result3)}")
    print(f"  Changed: {'Yes' if result3 != limit_msg else 'No'}")
    print()
    
    # Test 4: Message slightly over limit
    over_limit_msg = "x" * (bot.whatsapp_char_limit + 10)
    result4 = bot._ensure_whatsapp_length_limit(over_limit_msg)
    print(f"Test 4 - Message slightly over limit:")
    print(f"  Input length: {len(over_limit_msg)}")
    print(f"  Output length: {len(result4)}")
    print(f"  Truncated: {'Yes' if len(result4) < len(over_limit_msg) else 'No'}")
    print(f"  Under limit: {'Yes' if len(result4) <= bot.whatsapp_char_limit else 'No'}")
    print()
    
    # Test 5: Message with sentences
    sentence_msg = "This is sentence one. This is sentence two. This is sentence three. " * 100
    result5 = bot._ensure_whatsapp_length_limit(sentence_msg)
    print(f"Test 5 - Message with sentences:")
    print(f"  Input length: {len(sentence_msg)}")
    print(f"  Output length: {len(result5)}")
    print(f"  Under limit: {'Yes' if len(result5) <= bot.whatsapp_char_limit else 'No'}")
    print(f"  Ends with period: {'Yes' if result5.rstrip().endswith('.') else 'No'}")
    print(f"  Has truncation notice: {'Yes' if 'truncated' in result5 else 'No'}")
    print()
    
    print("✅ Length limiting tests completed!")


if __name__ == "__main__":
    test_length_limiting()