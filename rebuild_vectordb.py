#!/usr/bin/env python3
"""
Completely rebuild the vector database from scratch
"""

import os
import sys
import shutil
import asyncio
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_rag_server import SimpleRAGPipeline

async def rebuild_vector_database():
    """Completely rebuild the vector database"""
    
    print("🔥 REBUILDING VECTOR DATABASE FROM SCRATCH")
    print("=" * 60)
    
    # Backup current metadata
    data_dir = Path("data")
    metadata_file = data_dir / "document_metadata.json"
    
    current_docs = {}
    if metadata_file.exists():
        print("📋 Backing up current document metadata...")
        with open(metadata_file, 'r') as f:
            current_docs = json.load(f)
        print(f"  📊 Found {len(current_docs)} documents in metadata")
        
        # Print document info
        for doc_id, metadata in current_docs.items():
            print(f"    📄 {metadata['filename']} ({metadata['chunk_count']} chunks)")
    
    # Stop and remove old vector database
    print("\n🗑️ REMOVING OLD VECTOR DATABASE...")
    vector_db_path = data_dir / "chroma_db"
    if vector_db_path.exists():
        print(f"  🗑️ Deleting {vector_db_path}")
        shutil.rmtree(vector_db_path)
        print("  ✅ Old vector database deleted")
    else:
        print("  ℹ️ No existing vector database found")
    
    # Clear metadata
    print("\n🧹 CLEARING METADATA...")
    if metadata_file.exists():
        print(f"  🗑️ Clearing {metadata_file}")
        with open(metadata_file, 'w') as f:
            json.dump({}, f)
        print("  ✅ Metadata cleared")
    
    # Initialize fresh RAG pipeline
    print("\n🔧 INITIALIZING FRESH RAG PIPELINE...")
    rag_pipeline = SimpleRAGPipeline()
    print("  ✅ Fresh pipeline initialized")
    
    # Find documents to re-index
    print("\n🔍 FINDING DOCUMENTS TO RE-INDEX...")
    
    # Look for documents in uploads directory and current paths
    documents_to_index = []
    
    # Check uploads directory
    uploads_dir = Path("uploads")
    if uploads_dir.exists():
        for file_path in uploads_dir.glob("*.pdf"):
            documents_to_index.append(str(file_path))
            print(f"  📄 Found in uploads: {file_path.name}")
    
    # Check paths from old metadata
    for doc_id, metadata in current_docs.items():
        file_path = Path(metadata['file_path'])
        if file_path.exists() and str(file_path) not in documents_to_index:
            documents_to_index.append(str(file_path))
            print(f"  📄 Found from metadata: {file_path.name}")
    
    if not documents_to_index:
        print("  ⚠️ No documents found to index!")
        print("  💡 Please make sure PDF files are in the uploads/ directory")
        return
    
    print(f"\n📊 TOTAL DOCUMENTS TO INDEX: {len(documents_to_index)}")
    
    # Re-index all documents
    print("\n🔄 RE-INDEXING ALL DOCUMENTS...")
    
    for i, file_path in enumerate(documents_to_index, 1):
        print(f"\n  📄 [{i}/{len(documents_to_index)}] Processing: {Path(file_path).name}")
        try:
            result = await rag_pipeline.add_documents([file_path])
            
            if result["status"] == "success" and result["successful"] > 0:
                for res in result["results"]:
                    if res["status"] == "success":
                        print(f"    ✅ Success: {res['chunks']} chunks indexed")
                    else:
                        print(f"    ❌ Error: {res['error']}")
            else:
                print(f"    ❌ Failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"    ❌ Exception: {e}")
    
    # Verify the rebuild
    print("\n✅ VERIFYING REBUILD...")
    
    # Check metadata
    final_docs = rag_pipeline.document_metadata
    print(f"  📊 Documents in metadata: {len(final_docs)}")
    
    # Check vector database
    try:
        vector_count = rag_pipeline.vector_db.count()
        print(f"  📊 Chunks in vector DB: {vector_count}")
    except Exception as e:
        print(f"  ❌ Error checking vector DB: {e}")
    
    # Test queries
    print("\n🧪 TESTING QUERIES...")
    
    test_queries = [
        "pressure sores",
        "bed sores", 
        "मुझे bed sores के बारे में बताएं"  # Hindi query
    ]
    
    for query in test_queries:
        print(f"\n  🔍 Testing: '{query}'")
        try:
            result = await rag_pipeline.query(query)
            if result["status"] == "success":
                answer_len = len(result.get("answer", ""))
                sources_count = len(result.get("sources", []))
                if answer_len > 200:
                    print(f"    ✅ SUCCESS: {answer_len} chars, {sources_count} sources")
                    print(f"    📝 Preview: {result['answer'][:100]}...")
                else:
                    print(f"    ⚠️ SHORT ANSWER: {answer_len} chars")
                    print(f"    📝 Answer: {result['answer']}")
            else:
                print(f"    ❌ Failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"    ❌ Exception: {e}")
    
    print(f"\n🎉 VECTOR DATABASE REBUILD COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(rebuild_vector_database())