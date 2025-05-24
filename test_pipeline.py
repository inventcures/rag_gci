#!/usr/bin/env python3
"""
Test Script for RAG Pipeline
Tests all components: document processing, querying, STT, TTS, and WhatsApp integration
"""

import os
import sys
import asyncio
import json
import tempfile
from pathlib import Path
import logging

# Add kotaemon to path
sys.path.insert(0, str(Path(__file__).parent / "kotaemon-main" / "libs"))

from rag_server import RAGPipeline
from whatsapp_bot import EnhancedSTTService, EnhancedTTSService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipelineTest:
    """Test suite for RAG Pipeline"""
    
    def __init__(self):
        self.rag_pipeline = None
        self.stt_service = None
        self.tts_service = None
        self.test_results = {}
    
    async def run_all_tests(self):
        """Run all tests"""
        print("🧪 Starting RAG Pipeline Tests")
        print("=" * 50)
        
        try:
            await self.setup_services()
            await self.test_document_processing()
            await self.test_query_processing()
            await self.test_stt_service()
            await self.test_tts_service()
            await self.test_multilingual_support()
            
            self.print_results()
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            return False
        
        return all(self.test_results.values())
    
    async def setup_services(self):
        """Setup test services"""
        print("🔧 Setting up services...")
        
        try:
            # Initialize services
            self.rag_pipeline = RAGPipeline()
            self.stt_service = EnhancedSTTService()
            self.tts_service = EnhancedTTSService()
            
            self.test_results["setup"] = True
            print("✅ Services setup successful")
            
        except Exception as e:
            self.test_results["setup"] = False
            print(f"❌ Services setup failed: {e}")
            raise
    
    async def test_document_processing(self):
        """Test document processing and indexing"""
        print("\n📄 Testing document processing...")
        
        try:
            # Create a test document
            test_content = """
            # Palliative Care Guide
            
            ## What is Palliative Care?
            Palliative care is specialized medical care focused on providing relief from pain and other symptoms of serious illness. The goal is to improve quality of life for both patients and families.
            
            ## Key Principles
            1. Pain and symptom management
            2. Emotional and spiritual support
            3. Communication and decision-making support
            4. Coordination of care
            
            ## When to Consider Palliative Care
            - At diagnosis of serious illness
            - When symptoms impact quality of life
            - During treatment for serious conditions
            - When facing difficult medical decisions
            """
            
            # Save test document
            test_file = Path("test_palliative_care.md")
            with open(test_file, 'w') as f:
                f.write(test_content)
            
            # Test document indexing
            result = await self.rag_pipeline.add_documents(
                [str(test_file)],
                {"category": "test", "topic": "palliative_care"}
            )
            
            # Clean up
            test_file.unlink()
            
            if result["status"] == "success":
                self.test_results["document_processing"] = True
                print("✅ Document processing successful")
            else:
                self.test_results["document_processing"] = False
                print(f"❌ Document processing failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.test_results["document_processing"] = False
            print(f"❌ Document processing test failed: {e}")
    
    async def test_query_processing(self):
        """Test query processing and RAG responses"""
        print("\n💬 Testing query processing...")
        
        test_queries = [
            "What is palliative care?",
            "When should someone consider palliative care?",
            "What are the key principles of palliative care?",
            "How does palliative care help families?"
        ]
        
        try:
            successful_queries = 0
            
            for query in test_queries:
                result = await self.rag_pipeline.query(query, user_id="test_user")
                
                if result["status"] == "success" and result["answer"]:
                    successful_queries += 1
                    print(f"  ✅ Query: '{query[:30]}...' - Got response")
                else:
                    print(f"  ❌ Query: '{query[:30]}...' - Failed")
            
            success_rate = successful_queries / len(test_queries)
            
            if success_rate >= 0.8:  # 80% success rate
                self.test_results["query_processing"] = True
                print(f"✅ Query processing successful ({success_rate:.1%} success rate)")
            else:
                self.test_results["query_processing"] = False
                print(f"❌ Query processing failed ({success_rate:.1%} success rate)")
                
        except Exception as e:
            self.test_results["query_processing"] = False
            print(f"❌ Query processing test failed: {e}")
    
    async def test_stt_service(self):
        """Test Speech-to-Text service"""
        print("\n🎤 Testing STT service...")
        
        try:
            # Check if API key is available
            if not os.getenv("GROQ_API_KEY"):
                self.test_results["stt_service"] = False
                print("❌ STT test skipped - GROQ_API_KEY not set")
                return
            
            # Test language detection
            test_texts = {
                "hi": "यह एक हिंदी वाक्य है",
                "bn": "এটি একটি বাংলা বাক্য",
                "ta": "இது ஒரு தமிழ் வாக்கியம்",
                "gu": "આ એક ગુજરાતી વાક્ય છે",
                "en": "This is an English sentence"
            }
            
            correct_detections = 0
            
            for lang, text in test_texts.items():
                detected = self.stt_service._detect_language(text)
                if detected == lang:
                    correct_detections += 1
                    print(f"  ✅ Language detection: {lang} - Correct")
                else:
                    print(f"  ❌ Language detection: {lang} - Detected as {detected}")
            
            detection_rate = correct_detections / len(test_texts)
            
            if detection_rate >= 0.8:
                self.test_results["stt_service"] = True
                print(f"✅ STT service test passed ({detection_rate:.1%} accuracy)")
            else:
                self.test_results["stt_service"] = False
                print(f"❌ STT service test failed ({detection_rate:.1%} accuracy)")
                
        except Exception as e:
            self.test_results["stt_service"] = False
            print(f"❌ STT service test failed: {e}")
    
    async def test_tts_service(self):
        """Test Text-to-Speech service"""
        print("\n🔊 Testing TTS service...")
        
        try:
            test_phrases = {
                "hi": "यह एक परीक्षण है",
                "bn": "এটি একটি পরীক্ষা",
                "ta": "இது ஒரு சோதனை",
                "gu": "આ એક પરીક્ષા છે",
                "en": "This is a test"
            }
            
            successful_generations = 0
            
            for lang, text in test_phrases.items():
                result = await self.tts_service.synthesize_speech(text, lang)
                
                if result["status"] == "success":
                    successful_generations += 1
                    print(f"  ✅ TTS generation: {lang} - Success")
                    
                    # Clean up audio file if created
                    if result.get("audio_file") and Path(result["audio_file"]).exists():
                        Path(result["audio_file"]).unlink()
                else:
                    print(f"  ❌ TTS generation: {lang} - Failed")
            
            success_rate = successful_generations / len(test_phrases)
            
            if success_rate >= 0.8:
                self.test_results["tts_service"] = True
                print(f"✅ TTS service test passed ({success_rate:.1%} success rate)")
            else:
                self.test_results["tts_service"] = False
                print(f"❌ TTS service test failed ({success_rate:.1%} success rate)")
                
        except Exception as e:
            self.test_results["tts_service"] = False
            print(f"❌ TTS service test failed: {e}")
    
    async def test_multilingual_support(self):
        """Test multilingual query support"""
        print("\n🌍 Testing multilingual support...")
        
        try:
            multilingual_queries = [
                ("What is pain management?", "en"),
                ("दर्द प्रबंधन क्या है?", "hi"),
                ("ব্যথা ব্যবস্থাপনা কি?", "bn"),
                ("வலி மேலாண்மை என்றால் என்ன?", "ta"),
                ("પીડા વ્યવસ્થાપન શું છે?", "gu")
            ]
            
            successful_responses = 0
            
            for query, lang in multilingual_queries:
                # Test query processing
                rag_result = await self.rag_pipeline.query(query, user_id=f"test_user_{lang}")
                
                if rag_result["status"] == "success" and rag_result["answer"]:
                    # Test TTS response
                    tts_result = await self.tts_service.synthesize_speech(
                        rag_result["answer"], lang
                    )
                    
                    if tts_result["status"] == "success":
                        successful_responses += 1
                        print(f"  ✅ Multilingual: {lang} - Complete pipeline success")
                        
                        # Clean up audio file
                        if tts_result.get("audio_file") and Path(tts_result["audio_file"]).exists():
                            Path(tts_result["audio_file"]).unlink()
                    else:
                        print(f"  ⚠️ Multilingual: {lang} - Query OK, TTS failed")
                else:
                    print(f"  ❌ Multilingual: {lang} - Query failed")
            
            success_rate = successful_responses / len(multilingual_queries)
            
            if success_rate >= 0.6:  # Lower threshold for multilingual
                self.test_results["multilingual_support"] = True
                print(f"✅ Multilingual support test passed ({success_rate:.1%} success rate)")
            else:
                self.test_results["multilingual_support"] = False
                print(f"❌ Multilingual support test failed ({success_rate:.1%} success rate)")
                
        except Exception as e:
            self.test_results["multilingual_support"] = False
            print(f"❌ Multilingual support test failed: {e}")
    
    def print_results(self):
        """Print test results summary"""
        print("\n" + "=" * 50)
        print("🧪 TEST RESULTS SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title():.<30} {status}")
        
        print("-" * 50)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {passed_tests/total_tests:.1%}")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED! The RAG pipeline is ready to use.")
        elif passed_tests >= total_tests * 0.8:
            print("\n⚠️ Most tests passed. Check failed tests and configuration.")
        else:
            print("\n❌ Multiple tests failed. Please check your setup and configuration.")
        
        print("\n📋 Next Steps:")
        if not os.getenv("GROQ_API_KEY"):
            print("• Set GROQ_API_KEY in .env file for full functionality")
        
        print("• Upload your documents via the admin UI")
        print("• Test with real queries")
        print("• Configure WhatsApp webhook for bot integration")


async def main():
    """Main test function"""
    print("🚀 RAG Pipeline Test Suite")
    print("Testing document processing, querying, STT, TTS, and multilingual support")
    print()
    
    tester = RAGPipelineTest()
    success = await tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())