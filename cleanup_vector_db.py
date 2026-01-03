#!/usr/bin/env python3
"""
Clean up vector database entries for orphaned documents
"""

import sys
import logging
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
import json
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_vector_database():
    """Remove vector entries for documents that no longer exist in metadata"""
    try:
        # Load current metadata to see what should exist
        data_dir = Path("data")
        metadata_file = data_dir / "document_metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                valid_metadata = json.load(f)
            valid_doc_ids = set(valid_metadata.keys())
            logger.info(f"Found {len(valid_doc_ids)} valid documents in metadata")
        else:
            valid_doc_ids = set()
            logger.info("No metadata file found - will clear all vectors")
        
        # Connect to vector database
        vector_db_path = data_dir / "chroma_db"
        logger.info(f"Connecting to vector database at: {vector_db_path}")
        
        chroma_client = chromadb.PersistentClient(path=str(vector_db_path))
        
        # Get the collection
        try:
            collection = chroma_client.get_collection(name="documents")
            logger.info("✅ Connected to vector database")
        except Exception as e:
            logger.error(f"❌ Failed to connect to collection: {e}")
            return {"status": "error", "message": str(e)}
        
        # Get all vectors
        all_vectors = collection.get()
        total_vectors = len(all_vectors['ids'])
        logger.info(f"📊 Total vectors in database: {total_vectors}")
        
        if total_vectors == 0:
            logger.info("✅ Vector database is already empty")
            return {"status": "already_empty"}
        
        # Identify orphaned vectors
        orphaned_ids = []
        valid_ids = []
        
        for vector_id in all_vectors['ids']:
            # Extract doc_id from vector_id (format: doc_id_chunk_N)
            doc_id = vector_id.rsplit('_chunk_', 1)[0]
            
            if doc_id in valid_doc_ids:
                valid_ids.append(vector_id)
            else:
                orphaned_ids.append(vector_id)
        
        logger.info(f"📊 Vector analysis:")
        logger.info(f"  ✅ Valid vectors: {len(valid_ids)}")
        logger.info(f"  ❌ Orphaned vectors: {len(orphaned_ids)}")
        
        if len(orphaned_ids) == 0:
            logger.info("🎉 No orphaned vectors found!")
            return {
                "status": "no_orphans",
                "total_vectors": total_vectors,
                "valid_vectors": len(valid_ids)
            }
        
        # Remove orphaned vectors
        logger.info(f"🗑️ Removing {len(orphaned_ids)} orphaned vectors...")
        collection.delete(ids=orphaned_ids)
        
        # Verify cleanup
        remaining_vectors = collection.count()
        removed_count = total_vectors - remaining_vectors
        
        logger.info(f"✅ Removed {removed_count} orphaned vectors")
        logger.info(f"📊 Remaining vectors: {remaining_vectors}")
        
        return {
            "status": "cleaned",
            "total_vectors": total_vectors,
            "orphaned_removed": removed_count,
            "remaining_vectors": remaining_vectors,
            "valid_vectors": len(valid_ids)
        }
        
    except Exception as e:
        logger.error(f"❌ Error cleaning vector database: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

def show_vector_status():
    """Show current vector database status"""
    try:
        data_dir = Path("data")
        vector_db_path = data_dir / "chroma_db"
        
        logger.info("📊 Vector Database Status:")
        logger.info("=" * 40)
        
        if not vector_db_path.exists():
            logger.info("❌ Vector database directory not found")
            return
        
        chroma_client = chromadb.PersistentClient(path=str(vector_db_path))
        
        try:
            collection = chroma_client.get_collection(name="documents")
            total_vectors = collection.count()
            
            logger.info(f"📊 Total vectors: {total_vectors}")
            
            if total_vectors > 0:
                # Sample some vectors to show doc_ids
                sample_data = collection.get(limit=10)
                doc_ids_found = set()
                
                for vector_id in sample_data['ids']:
                    doc_id = vector_id.rsplit('_chunk_', 1)[0]
                    doc_ids_found.add(doc_id)
                
                logger.info(f"📄 Document IDs in vectors: {len(doc_ids_found)}")
                for doc_id in sorted(doc_ids_found):
                    logger.info(f"  - {doc_id}")
            
        except Exception as e:
            logger.info(f"❌ Error accessing collection: {e}")
            
    except Exception as e:
        logger.error(f"❌ Error checking vector status: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up vector database")
    parser.add_argument("--status", action="store_true", help="Show vector database status")
    parser.add_argument("--clean", action="store_true", help="Clean up orphaned vectors")
    
    args = parser.parse_args()
    
    if args.status:
        show_vector_status()
    elif args.clean:
        logger.info("Starting vector database cleanup...")
        result = cleanup_vector_database()
        
        logger.info(f"Cleanup result: {result}")
        
        if result.get('status') == 'cleaned':
            logger.info(f"🎉 Successfully removed {result['orphaned_removed']} orphaned vectors")
            logger.info(f"📊 Database now has {result['remaining_vectors']} vectors")
            sys.exit(0)
        elif result.get('status') == 'no_orphans':
            logger.info("🎉 No cleanup needed - all vectors are valid!")
            sys.exit(0)
        else:
            logger.error(f"💥 Cleanup failed: {result}")
            sys.exit(1)
    else:
        parser.print_help()