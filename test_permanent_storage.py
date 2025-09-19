#!/usr/bin/env python3
"""
Test the permanent document storage system
"""

import asyncio
import sys
import logging
from pathlib import Path
import tempfile

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from simple_rag_server import SimpleRAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_permanent_storage():
    """Test document storage and retrieval"""
    try:
        logger.info("Testing permanent document storage...")
        
        # Create a test document
        test_content = """
# Test Medical Document

This is a test document for the RAG system.

## Pain Management
Pain management is crucial in palliative care. Opioids should be used carefully.

## Patient Care
Quality of life is the primary focus in palliative medicine.
"""
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(test_content)
            temp_path = f.name
        
        logger.info(f"Created test document: {temp_path}")
        
        # Initialize RAG server
        rag_server = SimpleRAGPipeline()
        model_name = getattr(rag_server, 'embedding_model_name', 'unknown')
        logger.info(f"Using embedding model: {model_name}")
        
        # Check documents directory
        docs_dir = rag_server.documents_dir
        logger.info(f"Documents will be stored in: {docs_dir}")
        
        # Add document (should be stored permanently)
        logger.info("Adding document to RAG system...")
        result = await rag_server.add_documents([temp_path])
        
        if result.get("status") == "success":
            logger.info("✅ Document added successfully!")
            
            # Check what was stored
            stored_files = list(docs_dir.glob("*"))
            logger.info(f"📁 Files in storage: {len(stored_files)}")
            for f in stored_files:
                logger.info(f"  - {f.name} ({f.stat().st_size} bytes)")
            
            # Check metadata
            metadata = rag_server.document_metadata
            logger.info(f"📊 Metadata entries: {len(metadata)}")
            for doc_id, info in metadata.items():
                logger.info(f"  - {info['filename']} (stored: {info.get('stored_permanently', False)})")
            
            # Check vector database
            vector_count = rag_server.vector_db.count()
            logger.info(f"🔍 Vector database chunks: {vector_count}")
            
            # Clean up temp file
            Path(temp_path).unlink()
            logger.info("🧹 Cleaned up temporary file")
            
            return {
                "status": "success",
                "embedding_model": model_name,
                "stored_files": len(stored_files),
                "metadata_entries": len(metadata),
                "vector_chunks": vector_count
            }
        else:
            logger.error(f"❌ Failed to add document: {result}")
            return {"status": "error", "message": str(result)}
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    logger.info("Testing permanent document storage system...")
    result = asyncio.run(test_permanent_storage())
    
    logger.info(f"Test result: {result}")
    
    if result.get('status') == 'success':
        logger.info("🎉 Permanent storage test passed!")
        logger.info(f"🤖 Model: {result['embedding_model']}")
        logger.info(f"📁 Files: {result['stored_files']}")
        logger.info(f"🔍 Chunks: {result['vector_chunks']}")
        sys.exit(0)
    else:
        logger.error("💥 Test failed!")
        sys.exit(1)