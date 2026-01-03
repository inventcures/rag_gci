#!/usr/bin/env python3
"""
Script to force rebuild all embeddings regardless of health status
"""

import asyncio
import sys
from pathlib import Path
import logging
import json

# Add the current directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from simple_rag_server import SimpleRAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def force_rebuild_embeddings():
    """Force rebuild all embeddings using the current model configuration"""
    try:
        logger.info("Initializing RAG server...")
        rag_server = SimpleRAGPipeline()
        
        # Check current document count
        collection = rag_server.vector_db
        count = collection.count()
        logger.info(f"Current document count in vector DB: {count}")
        
        if count > 0:
            # Get all documents from metadata
            metadata_file = Path(rag_server.data_dir) / "document_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    all_documents = json.load(f)
                
                logger.info(f"Found {len(all_documents)} documents in metadata")
                
                # Force rebuild by calling the internal method
                logger.info("Starting forced rebuild...")
                result = await rag_server.rebuild_manager._rebuild_from_documents(list(all_documents.values()))
                
                logger.info(f"✅ Rebuild completed: {len(result)} documents processed")
                return {"status": "success", "documents_processed": len(result)}
            else:
                logger.warning("No document metadata found")
                return {"status": "no_documents", "message": "No documents to rebuild"}
        else:
            logger.info("Vector database is empty - nothing to rebuild")
            return {"status": "empty", "message": "Vector database is empty"}
            
    except Exception as e:
        logger.error(f"❌ Error during forced rebuild: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    logger.info("Starting forced embedding rebuild...")
    result = asyncio.run(force_rebuild_embeddings())
    
    if result.get('status') == 'success':
        logger.info(f"🎉 Forced rebuild completed successfully! Processed {result.get('documents_processed', 0)} documents")
        sys.exit(0)
    else:
        logger.error(f"💥 Forced rebuild result: {result}")
        sys.exit(1)