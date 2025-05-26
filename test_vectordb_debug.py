#!/usr/bin/env python3
"""
Debug vector database storage and retrieval issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import numpy as np

def test_vectordb_consistency():
    """Test if there are inconsistencies in vector DB storage/retrieval"""
    
    print("🔍 TESTING VECTOR DATABASE CONSISTENCY")
    print("=" * 60)
    
    # Initialize ChromaDB the same way as in the RAG pipeline
    vector_db_path = "data/chroma_db"
    chroma_client = chromadb.PersistentClient(path=str(vector_db_path))
    
    collection = chroma_client.get_or_create_collection(
        name="documents",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-small-en-v1.5"
        )
    )
    
    print(f"📊 Collection document count: {collection.count()}")
    
    # Test query
    hindi_query = "मुझे bed source के बारे में बताएं इनसे कैसे बचें और उनका निवारन कैसे करें इसके बारे में जानकारी नहीं"
    english_query = "tell me about bed sores and how to prevent them"
    
    print(f"\n🔍 Testing Hindi query: {hindi_query[:50]}...")
    hindi_results = collection.query(query_texts=[hindi_query], n_results=5)
    
    print(f"🔍 Testing English query: {english_query}")
    english_results = collection.query(query_texts=[english_query], n_results=5)
    
    print(f"\n📊 RESULTS COMPARISON:")
    print(f"  Hindi query results: {len(hindi_results['documents'][0]) if hindi_results['documents'] else 0}")
    print(f"  English query results: {len(english_results['documents'][0]) if english_results['documents'] else 0}")
    
    if hindi_results['distances']:
        print(f"\n📏 HINDI DISTANCES: {hindi_results['distances'][0][:5]}")
    if english_results['distances']:
        print(f"📏 ENGLISH DISTANCES: {english_results['distances'][0][:5]}")
    
    # Test if we can retrieve specific chunks
    print(f"\n🔍 TESTING SPECIFIC CHUNK RETRIEVAL:")
    try:
        # Try to get some specific chunks that should exist
        sample_chunks = collection.get(limit=3)
        if sample_chunks and sample_chunks['ids']:
            print(f"  ✅ Can retrieve chunks: {len(sample_chunks['ids'])} chunks found")
            print(f"  📝 Sample IDs: {sample_chunks['ids'][:3]}")
            
            # Test getting specific chunks
            specific_result = collection.get(ids=sample_chunks['ids'][:1])
            if specific_result and specific_result['documents']:
                print(f"  ✅ Specific chunk retrieval works")
                print(f"  📝 Sample content: {specific_result['documents'][0][:100]}...")
            else:
                print(f"  ❌ Specific chunk retrieval failed")
        else:
            print(f"  ❌ No chunks found in collection!")
    except Exception as e:
        print(f"  ❌ Error retrieving chunks: {e}")
    
    # Test if embeddings are being created properly
    print(f"\n🧪 TESTING EMBEDDING CREATION:")
    try:
        # Create embedding function manually
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-small-en-v1.5"
        )
        
        test_embedding = embed_fn([hindi_query])
        print(f"  ✅ Manual embedding creation works: {len(test_embedding)} embeddings")
        print(f"  📏 Embedding dimension: {len(test_embedding[0]) if test_embedding else 'N/A'}")
        
        # Compare with direct SentenceTransformer
        st_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        st_embedding = st_model.encode([hindi_query])
        print(f"  📏 SentenceTransformer dimension: {st_embedding.shape}")
        
        # Check if they're similar
        if len(test_embedding) > 0 and len(test_embedding[0]) == st_embedding.shape[1]:
            similarity = np.dot(test_embedding[0], st_embedding[0]) / (
                np.linalg.norm(test_embedding[0]) * np.linalg.norm(st_embedding[0])
            )
            print(f"  🎯 Embedding similarity: {similarity:.6f} (should be close to 1.0)")
        
    except Exception as e:
        print(f"  ❌ Embedding test failed: {e}")

if __name__ == "__main__":
    test_vectordb_consistency()