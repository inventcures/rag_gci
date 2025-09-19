#!/usr/bin/env python3
"""
Re-process all documents from metadata with new embedding model
"""

import asyncio
import sys
import logging
import json
from pathlib import Path

# Add the current directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from simple_rag_server import SimpleRAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def reprocess_all_documents():
    """Re-process all documents from metadata with the current embedding model"""
    try:
        logger.info("Initializing RAG server...")
        rag_server = SimpleRAGPipeline()
        
        # Load document metadata
        metadata_file = Path(rag_server.data_dir) / "document_metadata.json"
        if not metadata_file.exists():
            logger.warning("No document metadata found")
            return {"status": "no_metadata", "message": "No document metadata found"}
        
        with open(metadata_file, 'r') as f:
            all_documents = json.load(f)
        
        logger.info(f"Found {len(all_documents)} documents in metadata")
        
        # Check which files still exist
        existing_files = []
        missing_files = []
        
        for doc_id, doc_info in all_documents.items():
            file_path = doc_info["file_path"]
            if Path(file_path).exists():
                existing_files.append((doc_id, doc_info))
                logger.info(f"✅ Found: {doc_info['filename']}")
            else:
                missing_files.append((doc_id, doc_info))
                logger.warning(f"❌ Missing: {doc_info['filename']} at {file_path}")
        
        logger.info(f"📊 Status: {len(existing_files)} existing, {len(missing_files)} missing files")
        
        if not existing_files:
            logger.error("No existing files found to reprocess")
            return {"status": "no_files", "message": "No existing files found to reprocess"}
        
        # Process existing files
        processed_count = 0
        failed_count = 0
        
        for doc_id, doc_info in existing_files:
            try:
                logger.info(f"📄 Processing: {doc_info['filename']}")
                result = await rag_server.add_documents([doc_info["file_path"]])
                
                if result.get("status") == "success":
                    processed_count += 1
                    logger.info(f"✅ Successfully processed: {doc_info['filename']}")
                else:
                    failed_count += 1
                    logger.error(f"❌ Failed to process: {doc_info['filename']} - {result}")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"❌ Error processing {doc_info['filename']}: {e}")
        
        # Check final count
        final_count = rag_server.vector_db.count()
        
        result = {
            "status": "completed",
            "total_documents": len(all_documents),
            "existing_files": len(existing_files),
            "missing_files": len(missing_files),
            "processed_successfully": processed_count,
            "failed": failed_count,
            "final_vector_count": final_count,
            "embedding_model": getattr(rag_server, 'embedding_model_name', 'unknown')
        }
        
        logger.info(f"🎉 Reprocessing completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error during reprocessing: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    logger.info("Starting document reprocessing with new embedding model...")
    result = asyncio.run(reprocess_all_documents())
    
    if result.get('status') == 'completed':
        logger.info(f"🎉 Reprocessing completed successfully!")
        logger.info(f"📊 Final stats: {result['processed_successfully']}/{result['existing_files']} documents processed")
        logger.info(f"🔍 Vector database now contains {result['final_vector_count']} chunks")
        logger.info(f"🤖 Using embedding model: {result['embedding_model']}")
        sys.exit(0)
    else:
        logger.error(f"💥 Reprocessing result: {result}")
        sys.exit(1)