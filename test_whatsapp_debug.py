#!/usr/bin/env python3
"""
Direct test of WhatsApp bot text handling to debug the issue
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_rag_server import SimpleRAGPipeline
from whatsapp_bot import EnhancedWhatsAppBot, EnhancedSTTService, EnhancedTTSService

async def test_whatsapp_bot():
    """Test the WhatsApp bot directly"""
    
    # Initialize components
    print("🔧 Initializing RAG pipeline...")
    rag_pipeline = SimpleRAGPipeline()
    
    print("🔧 Initializing services...")
    stt_service = EnhancedSTTService()
    tts_service = EnhancedTTSService()
    
    print("🔧 Initializing WhatsApp bot...")
    whatsapp_bot = EnhancedWhatsAppBot(rag_pipeline, stt_service, tts_service)
    
    # Test query
    test_number = "+917892563038"
    test_query = "tell me about bed sores and how to prevent them"
    
    print(f"\n🧪 Testing direct RAG pipeline query...")
    direct_result = await rag_pipeline.query(test_query, user_id=f"whatsapp:{test_number}")
    print(f"📊 Direct result status: {direct_result.get('status')}")
    print(f"📝 Direct answer length: {len(direct_result.get('answer', ''))}")
    print(f"📄 Direct answer: {direct_result.get('answer', '')[:200]}...")
    
    print(f"\n🧪 Testing WhatsApp bot text handler...")
    try:
        # This will call the same method that the webhook calls
        await whatsapp_bot._handle_twilio_text_message(test_number, test_query)
        print("✅ WhatsApp bot processing completed")
    except Exception as e:
        print(f"❌ WhatsApp bot error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_whatsapp_bot())