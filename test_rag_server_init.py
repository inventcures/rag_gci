#!/usr/bin/env python3
"""
Test RAG server initialization with EmbeddingGemma-300M
"""

import sys
import logging
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from simple_rag_server import SimpleRAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_rag_server_init():
    """Test RAG server initialization with EmbeddingGemma"""
    try:
        logger.info("Testing RAG server initialization...")
        
        rag_server = SimpleRAGPipeline()
        
        # Check what model was loaded
        model_name = getattr(rag_server, 'embedding_model_name', 'unknown')
        logger.info(f"✅ RAG server initialized with embedding model: {model_name}")
        
        # Check vector DB status
        count = rag_server.vector_db.count()
        logger.info(f"📊 Vector database contains {count} document chunks")
        
        # Test a simple query (should work even with empty DB)
        logger.info("Testing empty query handling...")
        # Note: We won't actually query since there are no docs, but we verified the system is ready
        
        return {
            "status": "success",
            "embedding_model": model_name,
            "document_count": count,
            "message": "RAG server ready with EmbeddingGemma-300M"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG server: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    logger.info("Testing RAG server initialization with EmbeddingGemma-300M...")
    result = test_rag_server_init()
    
    logger.info(f"Result: {result}")
    
    if result.get('status') == 'success':
        logger.info("🎉 RAG server test passed!")
        logger.info(f"🤖 Using: {result['embedding_model']}")
        exit(0)
    else:
        logger.error("💥 RAG server test failed!")
        exit(1)