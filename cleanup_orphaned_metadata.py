#!/usr/bin/env python3
"""
Clean up orphaned document metadata where files no longer exist
"""

import json
import logging
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_orphaned_metadata():
    """Remove metadata entries for documents that no longer exist"""
    try:
        data_dir = Path("data")
        metadata_file = data_dir / "document_metadata.json"
        
        if not metadata_file.exists():
            logger.info("No metadata file found - nothing to clean up")
            return {"status": "no_metadata"}
        
        # Load current metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Found {len(metadata)} metadata entries")
        
        # Check which documents still exist
        existing_docs = {}
        orphaned_docs = {}
        
        for doc_id, doc_info in metadata.items():
            file_path = Path(doc_info["file_path"])
            
            if file_path.exists():
                existing_docs[doc_id] = doc_info
                logger.info(f"✅ Keeping: {doc_info['filename']} (file exists)")
            else:
                orphaned_docs[doc_id] = doc_info
                logger.warning(f"❌ Orphaned: {doc_info['filename']} (file missing at {file_path})")
        
        logger.info(f"📊 Status: {len(existing_docs)} existing, {len(orphaned_docs)} orphaned")
        
        if len(orphaned_docs) == 0:
            logger.info("🎉 No orphaned metadata found - all documents exist!")
            return {
                "status": "no_orphans",
                "total_documents": len(metadata),
                "existing_documents": len(existing_docs)
            }
        
        # Create backup
        backup_file = metadata_file.with_suffix('.json.backup')
        with open(backup_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"💾 Created backup at: {backup_file}")
        
        # Write cleaned metadata
        with open(metadata_file, 'w') as f:
            json.dump(existing_docs, f, indent=2)
        
        logger.info(f"🧹 Cleaned metadata file")
        logger.info(f"✅ Removed {len(orphaned_docs)} orphaned entries")
        logger.info(f"✅ Kept {len(existing_docs)} existing entries")
        
        return {
            "status": "cleaned",
            "total_documents": len(metadata),
            "existing_documents": len(existing_docs),
            "orphaned_removed": len(orphaned_docs),
            "backup_file": str(backup_file),
            "orphaned_list": [doc['filename'] for doc in orphaned_docs.values()]
        }
        
    except Exception as e:
        logger.error(f"❌ Error cleaning metadata: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

def show_current_status():
    """Show current document status"""
    try:
        data_dir = Path("data")
        metadata_file = data_dir / "document_metadata.json"
        documents_dir = data_dir / "documents"
        
        logger.info("📋 Current Document Status:")
        logger.info("=" * 50)
        
        # Check metadata
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            logger.info(f"📄 Metadata entries: {len(metadata)}")
            
            for doc_id, doc_info in metadata.items():
                file_path = Path(doc_info["file_path"])
                exists = "✅" if file_path.exists() else "❌"
                stored_perm = doc_info.get("stored_permanently", False)
                storage_status = "📁 Permanent" if stored_perm else "🔄 Temporary"
                
                logger.info(f"  {exists} {doc_info['filename']} - {storage_status}")
        else:
            logger.info("📄 No metadata file found")
        
        # Check physical storage
        if documents_dir.exists():
            stored_files = list(documents_dir.glob("*"))
            logger.info(f"📁 Files in storage: {len(stored_files)}")
            for f in stored_files:
                size_mb = f.stat().st_size / (1024 * 1024)
                logger.info(f"  📄 {f.name} ({size_mb:.1f}MB)")
        else:
            logger.info("📁 No documents storage directory found")
        
    except Exception as e:
        logger.error(f"❌ Error checking status: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up orphaned document metadata")
    parser.add_argument("--status", action="store_true", help="Show current document status")
    parser.add_argument("--clean", action="store_true", help="Clean up orphaned metadata")
    
    args = parser.parse_args()
    
    if args.status:
        show_current_status()
    elif args.clean:
        logger.info("Starting metadata cleanup...")
        result = cleanup_orphaned_metadata()
        
        logger.info(f"Cleanup result: {result}")
        
        if result.get('status') == 'cleaned':
            logger.info(f"🎉 Successfully cleaned up {result['orphaned_removed']} orphaned entries")
            logger.info(f"📋 Removed: {', '.join(result['orphaned_list'])}")
            logger.info(f"💾 Backup saved to: {result['backup_file']}")
            logger.info("🔄 Refresh the admin UI to see updated document list")
            sys.exit(0)
        elif result.get('status') == 'no_orphans':
            logger.info("🎉 No cleanup needed - all documents exist!")
            sys.exit(0)
        else:
            logger.error(f"💥 Cleanup failed: {result}")
            sys.exit(1)
    else:
        parser.print_help()