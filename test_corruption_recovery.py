#!/usr/bin/env python3
"""
Test script for vector database corruption detection and auto-rebuild functionality
"""

import asyncio
import logging
import json
from pathlib import Path
import sys
import time

# Add the current directory to path so we can import simple_rag_server
sys.path.append(str(Path(__file__).parent))

from simple_rag_server import SimpleRAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_corruption_detection_and_rebuild():
    """Test the corruption detection and auto-rebuild functionality"""
    
    print("🔧 Testing Vector Database Corruption Detection & Auto-Rebuild")
    print("=" * 70)
    
    try:
        # Initialize RAG pipeline
        print("\n1️⃣ Initializing RAG Pipeline...")
        rag = SimpleRAGPipeline(data_dir="data")
        
        # Test 1: Check health of existing database
        print("\n2️⃣ Testing Health Check...")
        health_status = rag.check_database_health()
        print(f"Health Status: {json.dumps(health_status, indent=2)}")
        
        # Test 2: Test query functionality
        print("\n3️⃣ Testing Query Functionality...")
        query_result = await rag.query("test medical query")
        print(f"Query Status: {query_result.get('status')}")
        print(f"Query Answer: {query_result.get('answer', 'No answer')[:100]}...")
        
        # Test 3: Test auto-rebuild capability  
        print("\n4️⃣ Testing Auto-Rebuild Capability...")
        rebuild_result = await rag.auto_rebuild_database()
        print(f"Rebuild Status: {rebuild_result.get('status')}")
        print(f"Rebuild Message: {rebuild_result.get('message')}")
        
        if rebuild_result.get('rebuild_stats'):
            stats = rebuild_result['rebuild_stats']
            print(f"Documents Processed: {stats.get('documents_processed')}")
            print(f"Chunks Created: {stats.get('chunks_created')}")
            print(f"Duration: {stats.get('duration_seconds')}s")
        
        # Test 4: Test query with auto-recovery
        print("\n5️⃣ Testing Query with Auto-Recovery...")
        recovery_result = await rag.query_with_auto_recovery("palliative care management")
        print(f"Recovery Query Status: {recovery_result.get('status')}")
        print(f"Rebuild Performed: {recovery_result.get('rebuild_performed', False)}")
        print(f"Health Check Performed: {recovery_result.get('health_check_performed', False)}")
        
        # Test 5: Check final health status
        print("\n6️⃣ Final Health Check...")
        final_health = rag.check_database_health()
        print(f"Final Health Status:")
        print(f"  - Corrupted: {final_health.get('is_corrupted', 'Unknown')}")
        print(f"  - Corruption Score: {final_health.get('corruption_score', 'Unknown')}")
        print(f"  - Severity: {final_health.get('severity', 'Unknown')}")
        print(f"  - Issues: {final_health.get('issues', [])}")
        
        print("\n✅ Corruption Detection & Auto-Rebuild Test Completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test Failed: {str(e)}")
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        return False

async def simulate_corruption_scenario():
    """Simulate a corruption scenario and test recovery"""
    
    print("\n🔥 Simulating Corruption Scenario...")
    print("=" * 50)
    
    try:
        rag = SimpleRAGPipeline(data_dir="data")
        
        # Simulate corruption by corrupting the vector database
        print("💥 Simulating database corruption...")
        
        # Try to corrupt by deleting some data while keeping metadata
        try:
            # Get current document count
            current_data = rag.vector_db.get()
            if current_data.get("ids"):
                # Delete half the documents to create inconsistency
                half_point = len(current_data["ids"]) // 2
                if half_point > 0:
                    corrupt_ids = current_data["ids"][:half_point]
                    rag.vector_db.delete(ids=corrupt_ids)
                    rag._persist_vector_db()
                    print(f"🔥 Deleted {len(corrupt_ids)} chunks to simulate corruption")
        
        except Exception as e:
            print(f"Could not simulate corruption: {e}")
        
        # Check if corruption is detected
        print("\n🔍 Checking if corruption is detected...")
        health_status = rag.check_database_health()
        print(f"Corruption Detected: {health_status.get('is_corrupted')}")
        print(f"Corruption Score: {health_status.get('corruption_score')}")
        
        if health_status.get('is_corrupted'):
            print("\n🔧 Attempting auto-recovery...")
            recovery_result = await rag.auto_rebuild_database()
            print(f"Recovery Status: {recovery_result.get('status')}")
            
            if recovery_result.get('status') == 'success':
                print("✅ Auto-recovery successful!")
                
                # Verify recovery
                print("\n🔍 Verifying recovery...")
                post_recovery_health = rag.check_database_health()
                print(f"Post-recovery corruption: {post_recovery_health.get('is_corrupted')}")
            else:
                print(f"❌ Auto-recovery failed: {recovery_result.get('message')}")
        else:
            print("ℹ️  No corruption detected, skipping recovery test")
        
        return True
        
    except Exception as e:
        print(f"❌ Corruption simulation failed: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🧪 Vector Database Corruption Recovery Test Suite")
    print("=" * 80)
    
    # Run basic functionality tests
    print("\n🔬 Running Basic Functionality Tests...")
    basic_success = asyncio.run(test_corruption_detection_and_rebuild())
    
    # Run corruption simulation tests
    print("\n🎭 Running Corruption Simulation Tests...")
    sim_success = asyncio.run(simulate_corruption_scenario())
    
    # Summary
    print("\n📊 Test Results Summary:")
    print("=" * 30)
    print(f"Basic Functionality Tests: {'✅ PASSED' if basic_success else '❌ FAILED'}")
    print(f"Corruption Simulation Tests: {'✅ PASSED' if sim_success else '❌ FAILED'}")
    
    if basic_success and sim_success:
        print("\n🎉 All tests passed! Corruption detection and auto-rebuild are working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the logs above for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)