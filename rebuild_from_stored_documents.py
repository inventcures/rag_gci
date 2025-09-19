#!/usr/bin/env python3
"""
Rebuild embeddings from permanently stored documents
This allows upgrading to better embedding models in the future
"""

import asyncio
import sys
import logging
import json
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from simple_rag_server import SimpleRAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def rebuild_from_stored_documents():
    """Rebuild embeddings from permanently stored documents"""
    try:
        logger.info("Initializing RAG server...")
        rag_server = SimpleRAGPipeline()
        
        # Check current embedding model
        model_name = getattr(rag_server, 'embedding_model_name', 'unknown')
        logger.info(f"🤖 Current embedding model: {model_name}")
        
        # Load document metadata
        metadata_file = Path(rag_server.data_dir) / "document_metadata.json"
        if not metadata_file.exists():
            logger.warning("No document metadata found")
            return {"status": "no_metadata", "message": "No document metadata found"}
        
        with open(metadata_file, 'r') as f:
            all_documents = json.load(f)
        
        logger.info(f"Found {len(all_documents)} documents in metadata")
        
        # Check document storage
        documents_dir = Path(rag_server.data_dir) / "documents"
        if not documents_dir.exists():
            logger.error("Documents directory not found")
            return {"status": "no_documents_dir", "message": "Documents directory not found"}
        
        # Find stored documents
        stored_files = []
        missing_files = []
        
        for doc_id, doc_info in all_documents.items():
            file_path = doc_info["file_path"]
            if Path(file_path).exists():
                stored_files.append((doc_id, doc_info))
                logger.info(f"✅ Found stored: {doc_info['filename']}")
            else:
                missing_files.append((doc_id, doc_info))
                logger.warning(f"❌ Missing: {doc_info['filename']} at {file_path}")
        
        logger.info(f"📊 Status: {len(stored_files)} stored, {len(missing_files)} missing files")
        
        if not stored_files:
            logger.error("No stored files found to rebuild from")
            return {"status": "no_stored_files", "message": "No stored files found to rebuild from"}
        
        # Clear current vector database to force rebuild
        logger.info("🗑️ Clearing current vector database...")
        current_count = rag_server.vector_db.count()
        if current_count > 0:
            # Delete and recreate collection
            try:
                rag_server.vector_db.delete(
                    ids=rag_server.vector_db.get()['ids']
                )
                logger.info(f"✅ Cleared {current_count} existing vectors")
            except Exception as e:
                logger.warning(f"Error clearing vectors: {e}")
        
        # Rebuild from stored files
        processed_count = 0
        failed_count = 0
        
        for doc_id, doc_info in stored_files:
            try:
                logger.info(f"📄 Rebuilding: {doc_info['filename']}")
                
                # Process stored document 
                result = await rag_server.add_documents([doc_info["file_path"]])
                
                if result.get("status") == "success":
                    processed_count += 1
                    logger.info(f"✅ Rebuilt: {doc_info['filename']}")
                else:
                    failed_count += 1
                    logger.error(f"❌ Failed: {doc_info['filename']} - {result}")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"❌ Error rebuilding {doc_info['filename']}: {e}")
        
        # Final status
        final_count = rag_server.vector_db.count()
        
        result = {
            "status": "completed",
            "embedding_model": model_name,
            "total_documents": len(all_documents),
            "stored_files": len(stored_files),
            "missing_files": len(missing_files),
            "processed_successfully": processed_count,
            "failed": failed_count,
            "final_vector_count": final_count,
            "cleared_old_vectors": current_count
        }
        
        logger.info(f"🎉 Rebuild completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error during rebuild: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

async def list_stored_documents():
    """List all permanently stored documents"""
    try:
        logger.info("Listing stored documents...")
        
        data_dir = Path("data")
        documents_dir = data_dir / "documents"
        metadata_file = data_dir / "document_metadata.json"
        
        if not documents_dir.exists():
            logger.info("No documents directory found")
            return []
        
        if not metadata_file.exists():
            logger.info("No metadata file found")
            return []
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        stored_docs = []
        for doc_id, doc_info in metadata.items():
            file_path = Path(doc_info["file_path"])
            exists = file_path.exists()
            size = file_path.stat().st_size if exists else 0
            
            stored_docs.append({
                "doc_id": doc_id,
                "filename": doc_info["filename"],
                "file_path": str(file_path),
                "exists": exists,
                "size_bytes": size,
                "chunk_count": doc_info.get("chunk_count", 0),
                "indexed_at": doc_info.get("indexed_at", "unknown"),
                "stored_permanently": doc_info.get("stored_permanently", False)
            })
        
        return stored_docs
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return []

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Rebuild embeddings from stored documents")
    parser.add_argument("--list", action="store_true", help="List stored documents")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild embeddings")
    
    args = parser.parse_args()
    
    if args.list:
        logger.info("Listing permanently stored documents...")
        docs = asyncio.run(list_stored_documents())
        
        if not docs:
            logger.info("No stored documents found")
        else:
            logger.info(f"Found {len(docs)} stored documents:")
            for doc in docs:
                status = "✅" if doc["exists"] else "❌"
                size_mb = doc["size_bytes"] / (1024 * 1024)
                logger.info(f"{status} {doc['filename']} ({size_mb:.1f}MB, {doc['chunk_count']} chunks)")
        
    elif args.rebuild:
        logger.info("Starting embedding rebuild from stored documents...")
        result = asyncio.run(rebuild_from_stored_documents())
        
        if result.get('status') == 'completed':
            logger.info(f"🎉 Rebuild completed successfully!")
            logger.info(f"📊 Stats: {result['processed_successfully']}/{result['stored_files']} documents processed")
            logger.info(f"🔍 Vector database now contains {result['final_vector_count']} chunks")
            logger.info(f"🤖 Using embedding model: {result['embedding_model']}")
            sys.exit(0)
        else:
            logger.error(f"💥 Rebuild failed: {result}")
            sys.exit(1)
    else:
        parser.print_help()