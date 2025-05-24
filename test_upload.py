#!/usr/bin/env python3
"""
Test script to debug file upload issues
"""

import os
import sys
from pathlib import Path
import tempfile
import shutil

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_file_upload():
    """Test file upload functionality"""
    
    print("🧪 Testing file upload functionality")
    print("=" * 50)
    
    # Create a test PDF file
    test_content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\nxref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000074 00000 n\n0000000120 00000 n\ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n178\n%%EOF"
    
    # Create temporary file (simulating Gradio upload)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(test_content)
        temp_file_path = temp_file.name
    
    print(f"📄 Created test file: {temp_file_path}")
    
    # Test the upload directory creation
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    print(f"📁 Upload directory: {upload_dir}")
    
    # Test copying file (simulating the upload process)
    try:
        from datetime import datetime
        import uuid
        
        # Create unique destination filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        source_path = Path(temp_file_path)
        dest_filename = f"{timestamp}_{unique_id}_{source_path.name}"
        dest_path = upload_dir / dest_filename
        
        print(f"🎯 Destination: {dest_path}")
        
        # Test the copy operation
        shutil.copy2(temp_file_path, dest_path)
        
        print(f"✅ File copied successfully!")
        print(f"📊 File size: {dest_path.stat().st_size} bytes")
        
        # Test if we can read the file
        if dest_path.exists():
            print("✅ File exists and is readable")
            
            # Test document processing
            try:
                from simple_rag_server import SimpleDocumentProcessor
                processor = SimpleDocumentProcessor()
                
                result = processor.process_file(str(dest_path))
                print(f"📝 Document processing result: {result['status']}")
                
                if result['status'] == 'success':
                    print(f"📄 Extracted text length: {len(result['text'])}")
                    print(f"🧩 Number of chunks: {result['chunk_count']}")
                else:
                    print(f"❌ Processing error: {result['error']}")
                    
            except Exception as e:
                print(f"❌ Document processing failed: {e}")
        
        # Cleanup
        os.unlink(temp_file_path)
        dest_path.unlink()
        print("🧹 Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Upload test failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        try:
            os.unlink(temp_file_path)
        except:
            pass
            
        return False

def test_gradio_file_types():
    """Test different file types that Gradio might provide"""
    
    print("\n🔍 Testing Gradio file type handling")
    print("=" * 50)
    
    # Create test file
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
        temp_file.write(b"This is a test file for Gradio upload testing.")
        temp_file_path = temp_file.name
    
    # Test 1: String path (most common Gradio format)
    print(f"Test 1: String path - {temp_file_path}")
    print(f"  Type: {type(temp_file_path)}")
    print(f"  Exists: {Path(temp_file_path).exists()}")
    
    # Test 2: Simulated Gradio file object
    class MockGradioFile:
        def __init__(self, path):
            self.name = path
            self.orig_name = Path(path).name
    
    mock_file = MockGradioFile(temp_file_path)
    print(f"Test 2: Mock Gradio file object")
    print(f"  Type: {type(mock_file)}")
    print(f"  Has name: {hasattr(mock_file, 'name')}")
    print(f"  Has orig_name: {hasattr(mock_file, 'orig_name')}")
    
    # Cleanup
    os.unlink(temp_file_path)
    print("✅ File type tests completed")

if __name__ == "__main__":
    print("🚀 Starting upload tests...\n")
    
    # Run tests
    upload_success = test_file_upload()
    test_gradio_file_types()
    
    print(f"\n📋 Test Summary:")
    print(f"File Upload: {'✅ PASS' if upload_success else '❌ FAIL'}")
    
    if upload_success:
        print("\n💡 File upload mechanism is working correctly!")
        print("If you're still seeing errors, it might be a Gradio version issue.")
        print("Try running the server and check the console logs for more details.")
    else:
        print("\n❌ File upload mechanism has issues.")
        print("Please check the error messages above for debugging.")