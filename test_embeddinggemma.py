#!/usr/bin/env python3
"""
Test EmbeddingGemma-300M loading with HF token
"""

import os
import logging
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_embeddinggemma():
    """Test loading EmbeddingGemma-300M"""
    try:
        hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
        if not hf_token:
            logger.error("No HF token found in environment")
            return False
            
        logger.info("Testing EmbeddingGemma-300M loading...")
        logger.info(f"HF token found: {hf_token[:10]}...")
        
        # Try to load the model
        model = SentenceTransformer('google/embeddinggemma-300m', token=hf_token)
        logger.info("✅ Successfully loaded EmbeddingGemma-300M!")
        
        # Test embedding generation
        test_text = "This is a test sentence for embedding generation."
        embedding = model.encode([test_text])
        
        logger.info(f"✅ Generated embedding with shape: {embedding.shape}")
        logger.info(f"✅ Embedding dimension: {embedding.shape[1]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load EmbeddingGemma-300M: {e}")
        return False

if __name__ == "__main__":
    success = test_embeddinggemma()
    if success:
        logger.info("🎉 EmbeddingGemma-300M test passed!")
        exit(0)
    else:
        logger.error("💥 EmbeddingGemma-300M test failed!")
        exit(1)