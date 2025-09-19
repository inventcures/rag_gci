#!/usr/bin/env python3
"""
Script to rebuild embeddings using EmbeddingGemma-300M for all existing documents
"""

import asyncio
import sys
from pathlib import Path
import logging

# Add the current directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from simple_rag_server import SimpleRAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def rebuild_all_embeddings():
    """Rebuild all embeddings using the new EmbeddingGemma model"""
    try:
        logger.info("Initializing RAG server with EmbeddingGemma-300M...")
        rag_server = SimpleRAGPipeline()
        
        # Auto-rebuild will detect the model change and rebuild all embeddings
        logger.info("Starting auto-rebuild process...")
        result = await rag_server.auto_rebuild_database()
        
        if result.get('status') == 'success':
            logger.info("✅ Successfully rebuilt all embeddings with EmbeddingGemma-300M")
            logger.info(f"Rebuild details: {result}")
        else:
            logger.error(f"❌ Rebuild failed: {result}")
            
        return result
        
    except Exception as e:
        logger.error(f"❌ Error during embedding rebuild: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    logger.info("Starting embedding rebuild with EmbeddingGemma-300M...")
    result = asyncio.run(rebuild_all_embeddings())
    
    if result.get('status') == 'success':
        logger.info("🎉 Embedding rebuild completed successfully!")
        sys.exit(0)
    else:
        logger.error("💥 Embedding rebuild failed!")
        sys.exit(1)