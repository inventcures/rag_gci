#!/usr/bin/env python3
"""
Calculate actual distances for Hindi bed sores query to determine optimal threshold
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_rag_server import SimpleRAGPipeline

async def calculate_distances():
    """Calculate actual distances for the Hindi query"""
    
    # Initialize RAG pipeline
    print("🔧 Initializing RAG pipeline...")
    rag_pipeline = SimpleRAGPipeline()
    
    # Test current query
    test_query = "pressure sores"
    
    print(f"📝 Query: {test_query}")
    print(f"📊 Total documents in corpus: {len(rag_pipeline.document_metadata)}")
    
    # Get search results directly from vector DB
    print("🔍 Querying vector database...")
    search_results = rag_pipeline.vector_db.query(
        query_texts=[test_query],
        n_results=10  # Get more results to see distance distribution
    )
    
    if search_results['documents'] and search_results['documents'][0]:
        contexts = search_results['documents'][0]
        metadatas = search_results['metadatas'][0] if 'metadatas' in search_results else []
        distances = search_results['distances'][0] if 'distances' in search_results else []
        
        print(f"\n📊 Found {len(contexts)} results:")
        print("=" * 80)
        
        for i, (context, meta, distance) in enumerate(zip(contexts, metadatas, distances)):
            filename = meta.get('filename', 'Unknown') if meta else 'Unknown'
            chunk_idx = meta.get('chunk_index', 'N/A') if meta else 'N/A'
            
            print(f"Result {i+1}:")
            print(f"  📏 Distance: {distance:.6f}")
            print(f"  📄 File: {filename}")
            print(f"  🔢 Chunk: {chunk_idx}")
            print(f"  📝 Content preview: {context[:100]}...")
            print("-" * 40)
        
        # Calculate suggested threshold
        if distances:
            min_distance = min(distances)
            max_distance = max(distances)
            median_distance = sorted(distances)[len(distances)//2]
            
            print(f"\n📈 DISTANCE STATISTICS:")
            print(f"  🔽 Minimum (most similar): {min_distance:.6f}")
            print(f"  📊 Median: {median_distance:.6f}")
            print(f"  🔼 Maximum (least similar): {max_distance:.6f}")
            
            # Suggest thresholds
            print(f"\n💡 THRESHOLD SUGGESTIONS:")
            print(f"  🎯 Conservative (top 1-2 results): {distances[1] if len(distances) > 1 else distances[0]:.6f}")
            print(f"  ⚖️ Balanced (top 3-5 results): {distances[min(4, len(distances)-1)]:.6f}")
            print(f"  🔓 Liberal (top 7-10 results): {distances[min(6, len(distances)-1)]:.6f}")
            
            # Look for bed sore related content
            bed_sore_results = []
            for i, (context, meta, distance) in enumerate(zip(contexts, metadatas, distances)):
                context_lower = context.lower()
                if any(term in context_lower for term in ['bed sore', 'pressure sore', 'decubitus', 'ulcer', 'skin', 'prevent']):
                    bed_sore_results.append((i, distance, context[:200]))
            
            if bed_sore_results:
                print(f"\n🎯 BED SORE RELEVANT RESULTS FOUND ({len(bed_sore_results)}):")
                for i, distance, preview in bed_sore_results:
                    print(f"  Result {i+1}: distance={distance:.6f}")
                    print(f"    Preview: {preview}...")
                
                max_relevant_distance = max(dist for _, dist, _ in bed_sore_results)
                print(f"\n🎯 RECOMMENDED THRESHOLD FOR BED SORES: {max_relevant_distance + 0.01:.6f}")
            else:
                print(f"\n⚠️ No obvious bed sore content found in top results")
                print(f"   Recommended threshold (conservative): {distances[2] if len(distances) > 2 else distances[-1]:.6f}")
    
    else:
        print("❌ No search results returned!")

if __name__ == "__main__":
    asyncio.run(calculate_distances())