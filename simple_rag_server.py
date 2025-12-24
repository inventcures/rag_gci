#!/usr/bin/env python3
"""
Simplified RAG Server - No Database Required
A lightweight RAG pipeline with admin UI and WhatsApp bot integration
Uses only file-based storage for maximum simplicity
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import traceback
import pickle
import hashlib
import uuid
import subprocess
import time
import signal
import shutil

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Gemini Live imports (optional, for voice conversations)
try:
    from gemini_live import (
        GeminiLiveService,
        GeminiLiveSession,
        GeminiLiveError,
        SessionManager,
        AudioHandler,
        get_config as get_gemini_config,
    )
    GEMINI_LIVE_AVAILABLE = True
except ImportError:
    GEMINI_LIVE_AVAILABLE = False
    GeminiLiveService = None
    SessionManager = None

# Core RAG components
import chromadb
from chromadb.utils import embedding_functions
import requests
import aiohttp
from sentence_transformers import SentenceTransformer

# Document processing
import PyPDF2
import docx
from pathlib import Path
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleDocumentProcessor:
    """Simple document processor without heavy dependencies"""
    
    def __init__(self):
        self.supported_extensions = {'.txt', '.md', '.pdf', '.docx'}
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single file and extract text"""
        try:
            file_path = Path(file_path)
            
            if file_path.suffix.lower() not in self.supported_extensions:
                return {
                    "status": "error",
                    "error": f"Unsupported file type: {file_path.suffix}"
                }
            
            # Extract text and get page count
            extraction_result = self._extract_text_with_metadata(file_path)
            text = extraction_result["text"]
            page_count = extraction_result.get("page_count", 1)
            
            if not text.strip():
                return {
                    "status": "error", 
                    "error": "No text extracted from file"
                }
            
            # Simple chunking
            chunks = self._chunk_text(text)
            
            return {
                "status": "success",
                "file_path": str(file_path),
                "text": text,
                "chunks": chunks,
                "chunk_count": len(chunks),
                "page_count": page_count
            }
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _extract_text_with_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from file and return metadata including page count"""
        
        if file_path.suffix.lower() in {'.txt', '.md'}:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return {"text": text, "page_count": 1}
        
        elif file_path.suffix.lower() == '.pdf':
            text = ""
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                page_count = len(reader.pages)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return {"text": text, "page_count": page_count}
        
        elif file_path.suffix.lower() == '.docx':
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            # For DOCX, estimate pages based on character count (rough estimate)
            estimated_pages = max(1, len(text) // 3000)  # ~3000 chars per page
            return {"text": text, "page_count": estimated_pages}
        
        else:
            return {"text": "", "page_count": 1}
    
    def _extract_text(self, file_path: Path) -> str:
        """Extract text from file based on extension (legacy method)"""
        result = self._extract_text_with_metadata(file_path)
        return result["text"]
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Simple text chunking"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start + chunk_size//2, end - 200), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks


class VectorDBHealthManager:
    """Handles vector database health monitoring and corruption detection"""
    
    def __init__(self, vector_db, document_metadata: Dict, data_dir: str):
        self.vector_db = vector_db
        self.document_metadata = document_metadata
        self.data_dir = data_dir
        self.corruption_indicators = []
        
    def detect_corruption(self) -> Dict[str, Any]:
        """Comprehensive corruption detection"""
        corruption_score = 0
        issues = []
        
        try:
            # Test 1: Basic database connectivity
            db_accessible = self._test_db_connectivity()
            if not db_accessible:
                corruption_score += 50
                issues.append("Database not accessible")
            
            # Test 2: Metadata consistency
            metadata_consistent = self._validate_metadata_sync()
            if not metadata_consistent:
                corruption_score += 30
                issues.append("Metadata inconsistent with vector DB")
            
            # Test 3: Query functionality
            query_functional = self._test_query_functionality()
            if not query_functional:
                corruption_score += 40
                issues.append("Query functionality broken")
            
            # Test 4: Embedding quality check
            embeddings_valid = self._check_embedding_quality()
            if not embeddings_valid:
                corruption_score += 25
                issues.append("Embedding quality degraded")
            
            # Test 5: Index integrity
            index_intact = self._validate_index_integrity()
            if not index_intact:
                corruption_score += 35
                issues.append("Index integrity compromised")
            
            is_corrupted = corruption_score >= 50
            
            return {
                "is_corrupted": is_corrupted,
                "corruption_score": corruption_score,
                "issues": issues,
                "severity": "critical" if corruption_score >= 80 else "moderate" if corruption_score >= 50 else "minor",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error during corruption detection: {e}")
            return {
                "is_corrupted": True,
                "corruption_score": 100,
                "issues": [f"Detection failed: {str(e)}"],
                "severity": "critical",
                "timestamp": datetime.now().isoformat()
            }
    
    def _test_db_connectivity(self) -> bool:
        """Test basic database connectivity"""
        try:
            # Try to get collection info
            collection = self.vector_db
            collection.count()
            return True
        except Exception as e:
            logger.warning(f"DB connectivity test failed: {e}")
            return False
    
    def _validate_metadata_sync(self) -> bool:
        """Check if metadata matches vector database contents"""
        try:
            collection = self.vector_db
            db_count = collection.count()
            
            # Calculate expected count from metadata
            expected_count = sum(meta.get("chunk_count", 0) for meta in self.document_metadata.values())
            
            # Allow 10% variance for minor inconsistencies
            variance_threshold = max(1, expected_count * 0.1)
            return abs(db_count - expected_count) <= variance_threshold
            
        except Exception as e:
            logger.warning(f"Metadata sync validation failed: {e}")
            return False
    
    def _test_query_functionality(self) -> bool:
        """Test if query operations work correctly"""
        try:
            collection = self.vector_db
            
            # Try a simple query
            results = collection.query(
                query_texts=["test query"],
                n_results=1
            )
            
            # Check if query returned expected structure
            return (
                isinstance(results, dict) and
                "documents" in results and
                "metadatas" in results and
                "distances" in results
            )
            
        except Exception as e:
            logger.warning(f"Query functionality test failed: {e}")
            return False
    
    def _check_embedding_quality(self) -> bool:
        """Check if embeddings are producing reasonable distances"""
        try:
            collection = self.vector_db
            
            # Get a small sample of documents
            sample_results = collection.query(
                query_texts=["medical"],
                n_results=min(5, collection.count())
            )
            
            if not sample_results.get("distances") or not sample_results["distances"][0]:
                return False
            
            distances = sample_results["distances"][0]
            
            # Check for anomalous distance patterns
            # Valid distances should be between 0 and 2 for cosine similarity
            valid_distances = all(0 <= d <= 2 for d in distances)
            
            # Check for reasonable variance (not all identical)
            if len(distances) > 1:
                distance_variance = max(distances) - min(distances)
                has_variance = distance_variance > 0.001
            else:
                has_variance = True
            
            return valid_distances and has_variance
            
        except Exception as e:
            logger.warning(f"Embedding quality check failed: {e}")
            return False
    
    def _validate_index_integrity(self) -> bool:
        """Check if the database index is intact"""
        try:
            collection = self.vector_db
            
            # Try to access different operations that use the index
            total_count = collection.count()
            
            if total_count == 0:
                return len(self.document_metadata) == 0
            
            # Try to peek at data
            peek_result = collection.peek(limit=1)
            return bool(peek_result.get("ids"))
            
        except Exception as e:
            logger.warning(f"Index integrity check failed: {e}")
            return False


class AutoRebuildManager:
    """Handles automatic vector database rebuilding"""
    
    def __init__(self, rag_pipeline):
        self.rag_pipeline = rag_pipeline
        self.rebuild_in_progress = False
        self.last_rebuild_time = None
        
    async def auto_rebuild_if_needed(self) -> Dict[str, Any]:
        """Automatically rebuild vector DB if corruption detected"""
        
        if self.rebuild_in_progress:
            return {
                "status": "already_rebuilding",
                "message": "Rebuild already in progress"
            }
        
        # Check for corruption
        health_manager = VectorDBHealthManager(
            self.rag_pipeline.vector_db,
            self.rag_pipeline.document_metadata,
            self.rag_pipeline.data_dir
        )
        
        health_status = health_manager.detect_corruption()
        
        if not health_status["is_corrupted"]:
            return {
                "status": "healthy",
                "message": "Vector database is healthy, no rebuild needed",
                "health_status": health_status
            }
        
        logger.warning(f"Corruption detected: {health_status}")
        
        # Perform automatic rebuild
        return await self._perform_auto_rebuild(health_status)
    
    async def _perform_auto_rebuild(self, corruption_info: Dict) -> Dict[str, Any]:
        """Perform the actual rebuild process"""
        
        self.rebuild_in_progress = True
        rebuild_start_time = datetime.now()
        
        try:
            logger.info("🔧 Starting automatic vector database rebuild...")
            
            # Step 1: Backup current state
            backup_result = await self._backup_current_state()
            if not backup_result["success"]:
                raise Exception(f"Backup failed: {backup_result['error']}")
            
            # Step 2: Collect all source documents
            documents_to_rebuild = await self._collect_source_documents()
            if not documents_to_rebuild:
                raise Exception("No source documents found for rebuild")
            
            # Step 3: Clear corrupted database
            await self._clear_corrupted_database()
            
            # Step 4: Rebuild from source documents
            rebuild_results = await self._rebuild_from_documents(documents_to_rebuild)
            
            # Step 5: Validate rebuilt database
            validation_result = await self._validate_rebuilt_database()
            
            if not validation_result["is_valid"]:
                # Restore from backup if rebuild failed
                await self._restore_from_backup(backup_result["backup_path"])
                raise Exception(f"Rebuild validation failed: {validation_result['errors']}")
            
            self.last_rebuild_time = datetime.now()
            rebuild_duration = (self.last_rebuild_time - rebuild_start_time).total_seconds()
            
            logger.info(f"✅ Vector database rebuilt successfully in {rebuild_duration:.1f}s")
            
            return {
                "status": "success",
                "message": f"Vector database rebuilt successfully in {rebuild_duration:.1f}s",
                "corruption_info": corruption_info,
                "rebuild_stats": {
                    "documents_processed": len(documents_to_rebuild),
                    "chunks_created": sum(result.get("chunks_added", 0) for result in rebuild_results),
                    "duration_seconds": rebuild_duration
                },
                "timestamp": self.last_rebuild_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Auto-rebuild failed: {e}")
            return {
                "status": "error",
                "message": f"Auto-rebuild failed: {str(e)}",
                "corruption_info": corruption_info,
                "timestamp": datetime.now().isoformat()
            }
        
        finally:
            self.rebuild_in_progress = False
    
    async def _backup_current_state(self) -> Dict[str, Any]:
        """Backup current metadata and database state"""
        try:
            backup_dir = Path(self.rag_pipeline.data_dir) / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"metadata_backup_{timestamp}.json"
            
            # Backup metadata
            with open(backup_path, 'w') as f:
                json.dump(self.rag_pipeline.document_metadata, f, indent=2)
            
            return {
                "success": True,
                "backup_path": str(backup_path),
                "timestamp": timestamp
            }
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _collect_source_documents(self) -> List[Dict]:
        """Collect all source documents that need to be re-indexed"""
        documents = []
        
        try:
            uploads_dir = Path(self.rag_pipeline.uploads_dir)
            
            for doc_id, metadata in self.rag_pipeline.document_metadata.items():
                file_path = uploads_dir / metadata["filename"]
                
                if file_path.exists():
                    documents.append({
                        "doc_id": doc_id,
                        "file_path": str(file_path),
                        "metadata": metadata
                    })
                else:
                    logger.warning(f"Source file not found: {file_path}")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error collecting source documents: {e}")
            return []
    
    async def _clear_corrupted_database(self):
        """Clear the corrupted vector database"""
        try:
            # Get all IDs and delete them
            all_data = self.rag_pipeline.vector_db.get()
            if all_data.get("ids"):
                self.rag_pipeline.vector_db.delete(ids=all_data["ids"])
            
            # Force persistence
            self.rag_pipeline._persist_vector_db()
            
            logger.info("🗑️ Cleared corrupted vector database")
            
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            raise
    
    async def _rebuild_from_documents(self, documents: List[Dict]) -> List[Dict]:
        """Rebuild vector database from source documents"""
        rebuild_results = []
        
        for doc_info in documents:
            try:
                filename = doc_info.get('filename', doc_info.get('metadata', {}).get('filename', 'unknown'))
                logger.info(f"📄 Processing: {filename}")
                
                # Re-process the document
                result = await self.rag_pipeline.add_documents([doc_info["file_path"]])
                rebuild_results.append(result)
                
                if result.get("status") != "success":
                    logger.warning(f"Failed to rebuild {filename}: {result}")
                
            except Exception as e:
                filename = doc_info.get('filename', doc_info.get('metadata', {}).get('filename', 'unknown'))
                logger.error(f"Error rebuilding document {filename}: {e}")
                rebuild_results.append({
                    "status": "error",
                    "filename": filename,
                    "error": str(e)
                })
        
        return rebuild_results
    
    async def _validate_rebuilt_database(self) -> Dict[str, Any]:
        """Validate that the rebuilt database is functional"""
        try:
            health_manager = VectorDBHealthManager(
                self.rag_pipeline.vector_db,
                self.rag_pipeline.document_metadata,
                self.rag_pipeline.data_dir
            )
            
            health_status = health_manager.detect_corruption()
            
            return {
                "is_valid": not health_status["is_corrupted"],
                "health_status": health_status,
                "errors": health_status["issues"] if health_status["is_corrupted"] else []
            }
            
        except Exception as e:
            return {
                "is_valid": False,
                "errors": [f"Validation failed: {str(e)}"]
            }
    
    async def _restore_from_backup(self, backup_path: str):
        """Restore from backup if rebuild fails"""
        try:
            with open(backup_path, 'r') as f:
                backup_metadata = json.load(f)
            
            self.rag_pipeline.document_metadata = backup_metadata
            self.rag_pipeline._save_metadata()
            
            logger.info(f"🔄 Restored metadata from backup: {backup_path}")
            
        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")


class SimpleRAGPipeline:
    """Simplified RAG Pipeline without database dependencies"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Storage paths
        self.vector_db_path = self.data_dir / "chroma_db"
        self.metadata_file = self.data_dir / "document_metadata.json"
        self.conversation_file = self.data_dir / "conversations.json"
        self.uploads_dir = "uploads"  # Directory where uploaded files are stored
        self.documents_dir = self.data_dir / "documents"  # Permanent document storage
        self.documents_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.document_processor = SimpleDocumentProcessor()
        self.embedding_model = None
        self.vector_db = None
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.translation_model = "gemma2-9b-it"  # Model for query translation - better Indic language support
        
        # Check for MedGemma preference from environment
        use_medgemma = os.getenv("USE_MEDGEMMA", "false").lower() == "true"
        if use_medgemma:
            self.response_model = "medgemma"  # Use MedGemma for English response generation
            logger.info("🩺 MedGemma model selected for English response generation")
        else:
            self.response_model = "gemma2-9b-it"  # Use default Gemma for response generation
            logger.info("💎 Default Gemma model selected for response generation")
        
        # Document metadata and conversation storage
        self.document_metadata = self._load_metadata()
        self.conversations = self._load_conversations()
        
        self._initialize_components()
    
    def _copy_to_permanent_storage(self, source_path: str) -> str:
        """Copy uploaded file to permanent storage and return new path"""
        try:
            source_path = Path(source_path)
            if not source_path.exists():
                raise FileNotFoundError(f"Source file not found: {source_path}")
            
            # Generate unique filename to avoid conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"{timestamp}_{source_path.name}"
            permanent_path = self.documents_dir / unique_filename
            
            # Copy file to permanent storage
            shutil.copy2(source_path, permanent_path)
            logger.info(f"📁 Copied document to permanent storage: {permanent_path}")
            
            return str(permanent_path)
            
        except Exception as e:
            logger.error(f"Failed to copy file to permanent storage: {e}")
            # Return original path as fallback
            return str(source_path)
    
    def _initialize_components(self):
        """Initialize embedding model and vector database"""
        try:
            # Check for Hugging Face token and choose appropriate model
            hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
            if hf_token:
                try:
                    logger.info("HF token found, attempting to load EmbeddingGemma-300M...")
                    self.embedding_model = SentenceTransformer('google/embeddinggemma-300m', token=hf_token)
                    self.embedding_model_name = "google/embeddinggemma-300m"
                    logger.info("✅ Successfully loaded EmbeddingGemma-300M")
                except Exception as e:
                    logger.warning(f"Failed to load EmbeddingGemma with token: {e}")
                    logger.info("Falling back to all-MiniLM-L6-v2...")
                    self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    self.embedding_model_name = "all-MiniLM-L6-v2"
            else:
                logger.info("No HF token found, using free model all-MiniLM-L6-v2...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_model_name = "all-MiniLM-L6-v2"
            
            # Initialize ChromaDB
            logger.info("Initializing vector database...")
            chroma_client = chromadb.PersistentClient(path=str(self.vector_db_path))
            
            # Create or get collection
            # Create embedding function based on the model we're using
            if hasattr(self, 'embedding_model_name'):
                embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.embedding_model_name
                )
            else:
                embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            
            self.vector_db = chroma_client.get_or_create_collection(
                name="documents",
                embedding_function=embedding_func
            )
            
            # Initialize corruption detection and auto-rebuild managers
            self.health_manager = VectorDBHealthManager(
                self.vector_db, 
                self.document_metadata, 
                str(self.data_dir)
            )
            self.rebuild_manager = AutoRebuildManager(self)
            
            logger.info("RAG Pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise
    
    def _persist_vector_db(self):
        """Force persistence of vector database changes"""
        try:
            # ChromaDB automatically persists, but we can trigger a manual persist if needed
            if hasattr(self.vector_db, '_client') and hasattr(self.vector_db._client, 'persist'):
                self.vector_db._client.persist()
            logger.debug("Vector database persistence triggered")
        except Exception as e:
            logger.warning(f"Could not force vector DB persistence: {e}")
    
    def _verify_document_in_vector_db(self, doc_id: str) -> bool:
        """Verify if a document's chunks exist in the vector database"""
        try:
            if doc_id not in self.document_metadata:
                return False
            
            chunk_count = self.document_metadata[doc_id]["chunk_count"]
            chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(chunk_count)]
            
            # Try to get the chunks from vector DB
            try:
                result = self.vector_db.get(ids=chunk_ids)
                return len(result['ids']) == chunk_count
            except Exception as e:
                logger.warning(f"Error checking vector DB for doc {doc_id}: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error verifying document in vector DB: {e}")
            return False
    
    def _load_metadata(self) -> Dict:
        """Load document metadata from file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_metadata(self):
        """Save document metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.document_metadata, f, indent=2)
    
    def _load_conversations(self) -> Dict:
        """Load conversations from file"""
        if self.conversation_file.exists():
            try:
                with open(self.conversation_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_conversations(self):
        """Save conversations to file"""
        with open(self.conversation_file, 'w') as f:
            json.dump(self.conversations, f, indent=2)
    
    async def remove_document(self, doc_id: str) -> Dict[str, Any]:
        """Remove a document from the RAG index"""
        try:
            if doc_id not in self.document_metadata:
                return {
                    "status": "error",
                    "error": f"Document with ID {doc_id} not found"
                }
            
            # Get document info
            doc_info = self.document_metadata[doc_id]
            chunk_count = doc_info["chunk_count"]
            
            # Generate chunk IDs to delete
            chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(chunk_count)]
            
            # Remove from vector database
            try:
                self.vector_db.delete(ids=chunk_ids)
                # Force persistence to ensure deletion is committed
                self._persist_vector_db()
            except Exception as e:
                logger.warning(f"Error deleting from vector DB: {e}")
                # Continue with metadata cleanup even if vector deletion fails
            
            # Remove from metadata
            filename = doc_info["filename"]
            del self.document_metadata[doc_id]
            
            # Save updated metadata
            self._save_metadata()
            
            logger.info(f"Successfully removed document: {filename} ({chunk_count} chunks)")
            
            return {
                "status": "success",
                "message": f"Document '{filename}' removed successfully",
                "chunks_removed": chunk_count
            }
            
        except Exception as e:
            logger.error(f"Error removing document: {e}")
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    async def add_documents(self, file_paths: List[str], metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Add documents to the RAG index"""
        try:
            results = []
            
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                # Copy to permanent storage first
                permanent_path = self._copy_to_permanent_storage(file_path)
                
                # Process document from permanent storage
                doc_result = self.document_processor.process_file(permanent_path)
                
                if doc_result["status"] != "success":
                    results.append({
                        "file_path": file_path,
                        "status": "error",
                        "error": doc_result["error"]
                    })
                    continue
                
                # Generate document ID based on permanent path
                doc_id = hashlib.md5(permanent_path.encode()).hexdigest()
                
                # Check if document already exists and remove it first
                if doc_id in self.document_metadata:
                    logger.info(f"Document {doc_id} already exists, removing old version first...")
                    await self.remove_document(doc_id)
                
                # Prepare chunks for indexing
                chunks = doc_result["chunks"]
                chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
                
                # Create metadata for each chunk
                chunk_metadata = []
                for i, chunk in enumerate(chunks):
                    meta = {
                        "file_path": permanent_path,  # Use permanent path
                        "original_path": file_path,   # Keep original path for reference
                        "filename": Path(file_path).name,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "page_count": doc_result.get("page_count", 1),
                        "doc_id": doc_id,
                        **(metadata or {})
                    }
                    chunk_metadata.append(meta)
                
                # Add to vector database with error handling
                try:
                    self.vector_db.add(
                        documents=chunks,
                        metadatas=chunk_metadata,
                        ids=chunk_ids
                    )
                    # Force persistence after adding
                    self._persist_vector_db()
                except Exception as e:
                    logger.error(f"Error adding document to vector DB: {e}")
                    results.append({
                        "file_path": file_path,
                        "status": "error",
                        "error": f"Vector database error: {str(e)}"
                    })
                    continue
                
                # Store document metadata including page count
                self.document_metadata[doc_id] = {
                    "file_path": permanent_path,  # Store permanent path
                    "original_path": file_path,   # Keep track of original upload path
                    "filename": Path(file_path).name,
                    "chunk_count": len(chunks),
                    "page_count": doc_result.get("page_count", 1),
                    "metadata": metadata or {},
                    "indexed_at": datetime.now().isoformat(),
                    "stored_permanently": True
                }
                
                # Verify the document was properly indexed
                if self._verify_document_in_vector_db(doc_id):
                    results.append({
                        "file_path": file_path,
                        "status": "success",
                        "doc_id": doc_id,
                        "chunks": len(chunks)
                    })
                    logger.info(f"Successfully indexed and verified: {file_path} ({len(chunks)} chunks)")
                else:
                    # Remove from metadata if verification failed
                    del self.document_metadata[doc_id]
                    results.append({
                        "file_path": file_path,
                        "status": "error",
                        "error": "Document indexing verification failed"
                    })
                    logger.error(f"Document indexing verification failed for: {file_path}")
            
            # Save metadata
            self._save_metadata()
            
            return {
                "status": "success",
                "results": results,
                "total_files": len(file_paths),
                "successful": len([r for r in results if r["status"] == "success"])
            }
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    async def query(self, question: str, conversation_id: Optional[str] = None, 
                   user_id: Optional[str] = None, top_k: int = 5, source_language: str = "en") -> Dict[str, Any]:
        """Query the RAG pipeline"""
        try:
            # Check if we have any documents indexed
            total_docs = len(self.document_metadata)
            logger.info(f"Query: '{question}' - Total documents in metadata: {total_docs}")
            
            # Debug: Print all document IDs in metadata
            doc_ids = list(self.document_metadata.keys())
            logger.info(f"Document IDs in metadata: {doc_ids}")
            
            # Debug: Check vector database collection count
            try:
                db_count = self.vector_db.count()
                logger.info(f"Vector database contains {db_count} documents")
            except Exception as e:
                logger.warning(f"Could not get vector DB count: {e}")
            
            # Translate query to English if needed for better embedding matching
            query_for_search = question  # Default to original question
            if source_language != "en":
                logger.info(f"Translating {source_language} query to English for embedding")
                translation_result = await self.translate_query_to_english(question, source_language)
                if translation_result["status"] in ["success", "fallback"]:
                    query_for_search = translation_result["translated_query"]
                    logger.info(f"Using translated query for search: '{query_for_search}'")
                else:
                    logger.warning(f"Query translation failed, using original: {translation_result.get('error', 'Unknown error')}")
            
            # Retrieve relevant documents using translated query
            search_results = self.vector_db.query(
                query_texts=[query_for_search],
                n_results=top_k
            )
            
            logger.info(f"Vector DB query returned {len(search_results.get('documents', []))} result sets")
            if search_results.get('documents'):
                logger.info(f"First result set has {len(search_results['documents'][0])} documents")
                # Debug: Print some metadata from search results
                if search_results.get('metadatas') and search_results['metadatas'][0]:
                    first_meta = search_results['metadatas'][0][0] if search_results['metadatas'][0] else {}
                    logger.info(f"First result metadata sample: {first_meta}")
                if search_results.get('distances'):
                    distances = search_results['distances'][0]
                    logger.info(f"Search distances: {distances[:3]}...")  # First 3 distances
            
            if not search_results['documents'] or not search_results['documents'][0]:
                logger.warning("No search results found in vector database")
                return {
                    "status": "success",
                    "answer": "We are afraid, we could not find the answer to your query in our medical corpus. Please consult a qualified medical doctor or visit your nearest hospital, with your query.",
                    "sources": [],
                    "conversation_id": conversation_id
                }
            
            # Format context with enhanced metadata
            contexts = search_results['documents'][0]
            metadatas = search_results['metadatas'][0]
            distances = search_results['distances'][0] if 'distances' in search_results else [0] * len(contexts)
            
            # Check relevance threshold (you can adjust this value)
            relevance_threshold = 1.5  # Lower is more similar
            relevant_contexts = []
            relevant_metadatas = []
            
            for i, (context, meta, distance) in enumerate(zip(contexts, metadatas, distances)):
                logger.info(f"Context {i}: distance={distance:.4f}, threshold={relevance_threshold}")
                if distance <= relevance_threshold:
                    relevant_contexts.append(context)
                    relevant_metadatas.append(meta)
                    logger.info(f"  ✅ Context {i} ACCEPTED (distance {distance:.4f} <= {relevance_threshold})")
                else:
                    logger.info(f"  ❌ Context {i} REJECTED (distance {distance:.4f} > {relevance_threshold})")
            
            logger.info(f"Total relevant contexts found: {len(relevant_contexts)}/{len(contexts)}")
            
            # If no relevant contexts found
            if not relevant_contexts:
                return {
                    "status": "success",
                    "answer": "I could not find any supporting documents in my corpus. Please consult a qualified medical doctor or visit the nearest hospital.",
                    "model_used": "system",
                    "sources": [],
                    "context_used": 0,
                    "conversation_id": conversation_id,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Intelligent context fusion based on distance similarity
            relevant_distances = [distances[i] for i, (_, _, distance) in enumerate(zip(contexts, metadatas, distances)) if distance <= relevance_threshold]
            filtered_contexts, filtered_metadatas, should_fuse = self._intelligent_context_fusion(
                relevant_contexts, relevant_metadatas, relevant_distances
            )
            
            context_text = "\n\n".join([
                f"Source: {meta['filename']} (chunk {meta['chunk_index']+1})\n{doc}"
                for doc, meta in zip(filtered_contexts, filtered_metadatas)
            ])
            
            # Generate answer using Groq with citation instructions
            answer, model_used = await self._generate_answer_with_citations(question, context_text, filtered_metadatas, should_fuse)
            
            # DISABLED: Don't override similarity threshold with LLM's uncertainty
            # If similarity threshold passed, trust that we have relevant context
            # if self._is_no_answer_response(answer):
            #     return {
            #         "status": "success", 
            #         "answer": "We are afraid, we could not find the answer to your query in our medical corpus. Please consult a qualified medical doctor or visit your nearest hospital, with your query.",
            #         "sources": [],
            #         "conversation_id": conversation_id
            #     }
            
            # Format sources
            sources = []
            for meta in relevant_metadatas:
                source = {
                    "filename": meta['filename'],
                    "chunk_index": meta['chunk_index'],
                    "total_chunks": meta['total_chunks']
                }
                if source not in sources:
                    sources.append(source)
            
            # Store conversation
            if conversation_id:
                if conversation_id not in self.conversations:
                    self.conversations[conversation_id] = []
                
                self.conversations[conversation_id].append({
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id,
                    "question": question,
                    "answer": answer,
                    "sources": sources
                })
                
                self._save_conversations()
            
            return {
                "status": "success",
                "answer": answer,
                "model_used": model_used,
                "sources": sources,
                "context_used": len(relevant_contexts),
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error querying RAG pipeline: {e}")
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _intelligent_context_fusion(self, contexts: List[str], metadatas: List[Dict], distances: List[float]) -> tuple:
        """
        Intelligently decide whether to fuse multiple contexts or use only the best one.
        
        Returns:
            tuple: (filtered_contexts, filtered_metadatas, should_fuse)
        """
        if len(contexts) <= 1:
            return contexts, metadatas, False
        
        # Calculate distance statistics
        min_distance = min(distances)
        max_distance = max(distances)
        distance_range = max_distance - min_distance
        
        # Define similarity threshold for fusion
        # If distances are close together (small range), fuse them
        # If distances are far apart (large range), use only the closest
        fusion_threshold = 0.15  # Adjust this value based on your needs
        
        logger.info(f"🔍 Context fusion analysis:")
        logger.info(f"  📏 Distance range: {min_distance:.4f} to {max_distance:.4f} (spread: {distance_range:.4f})")
        logger.info(f"  🎯 Fusion threshold: {fusion_threshold}")
        
        if distance_range <= fusion_threshold:
            # Distances are close together - fuse all contexts
            logger.info(f"  ✅ FUSING {len(contexts)} contexts (distances are similar)")
            return contexts, metadatas, True
        else:
            # Distances are far apart - use only the closest context
            best_idx = distances.index(min_distance)
            logger.info(f"  🎯 Using ONLY closest context (distance {min_distance:.4f}) - others too far")
            return [contexts[best_idx]], [metadatas[best_idx]], False
    
    async def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using Groq API"""
        try:
            if not self.groq_api_key:
                return "Error: GROQ_API_KEY not configured"
            
            prompt = f"""You are an expert medical assistant. You MUST respond ONLY in English language using English script/alphabet.

CRITICAL LANGUAGE REQUIREMENT:
- Your response MUST be in English language only
- Use English alphabet/script only (no Hindi, Bengali, or other scripts)
- Even if the question is in another language, respond in English

INSTRUCTIONS:
1. Carefully examine the medical context provided
2. If the context contains relevant information, provide a clear, medically accurate answer in English
3. If the context lacks sufficient information, clearly state this in English
4. Be direct and concise - do not include reasoning steps in your response

MEDICAL CONTEXT:
{context}

QUESTION: {question}

ENGLISH ANSWER:"""
            
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.response_model,  # Use English-optimized model for response generation
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,  # Lower temperature for more consistent medical responses
                "max_tokens": 512  # Reduced to enforce WhatsApp length limit
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                logger.error(f"Groq API error: {response.status_code} - {response.text}")
                return f"Error generating response: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    async def _generate_answer_with_citations(self, question: str, context: str, metadatas: List[Dict], should_fuse: bool = False) -> tuple:
        """Generate answer with citations using Groq API
        Returns: (answer, model_used)"""
        try:
            if not self.groq_api_key:
                return "Error: GROQ_API_KEY not configured", "error"
            
            # Create citation mapping
            citation_text = self._format_citation_context(context, metadatas)
            
            if should_fuse:
                # Multiple similar contexts - synthesize and cite all
                prompt = f"""You are an expert medical assistant. You MUST respond ONLY in English language using English script/alphabet. Your task is to analyze multiple related medical contexts and synthesize them into a comprehensive answer with multiple citations.

🚨 CRITICAL LENGTH REQUIREMENT: Your ENTIRE response including citations MUST BE UNDER 1500 CHARACTERS. Count carefully! If your draft is too long, shorten it while keeping medical accuracy. 🚨

CRITICAL LANGUAGE REQUIREMENT:
- Your response MUST be in English language only


FUSION INSTRUCTIONS:
1. Carefully analyze ALL provided medical contexts below
2. Synthesize information from multiple sources to provide a comprehensive answer
3. Extract relevant information even if the question language differs from the context language
4. Combine complementary information from different sources
5. Provide a well-structured, medically accurate answer that integrates insights from all relevant contexts
6. Always conclude with citations to ALL sources used
7. KEEP TOTAL RESPONSE UNDER 1500 CHARACTERS

CITATION REQUIREMENTS:
- End your response with: {{retrieved from: [docname_pg{{pagenum}}], [docname_pg{{pagenum}}] etc.}}
- Cite ALL sources that contributed to your answer
- Use the SHORT format: docname_pg{{pagenum}} (e.g., palliative_care_pg5)

MEDICAL CONTEXTS (MULTIPLE SOURCES):
{citation_text}

QUESTION: {question}

SYNTHESIZED ANSWER (UNDER 1500 CHARS):"""
            else:
                # Single closest context - focus on most relevant source
                prompt = f"""You are an expert medical assistant. You MUST respond ONLY in English language using English script/alphabet. Your task is to analyze the most relevant medical document and provide an accurate, focused answer with proper citation.

🚨 CRITICAL LENGTH REQUIREMENT: Your ENTIRE response including citations MUST BE UNDER 1500 CHARACTERS. Count carefully! If your draft is too long, shorten it while keeping medical accuracy. 🚨

CRITICAL LANGUAGE REQUIREMENT:
- Your response MUST be in English language only
- Use English alphabet/script only (no Hindi, Bengali, or other scripts)
- Even if the question is in another language, respond in English

FOCUSED INSTRUCTIONS:
1. Carefully analyze the provided medical context (most relevant to your question)
2. Extract relevant information even if the question language differs from the context language
3. Provide a clear, medically accurate answer based on this specific context
4. Focus on being helpful - if there's any relevant medical information, use it to provide a useful response
5. Always conclude with a proper citation to this specific source
6. KEEP TOTAL RESPONSE UNDER 1500 CHARACTERS

CITATION REQUIREMENTS:
- ONLY cite at the END of your response - NO inline citations in the text
- End your response with: [ Sources : doc_name: pg 1,2,3 ; other_doc: pg 4,5 ]
- Multiple pages from same document: separate with commas
- Multiple documents: separate with semicolons
- Always use this exact format


MEDICAL CONTEXT (MOST RELEVANT):
{citation_text}

QUESTION: {question}

FOCUSED ANSWER (UNDER 1500 CHARS):"""
            
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            # Use MedGemma if selected, otherwise use Groq
            if self.response_model == "medgemma":
                answer, model_used = await self._call_medgemma_endpoint(prompt, citation_text, question)
                if answer.startswith("Error"):
                    return answer
            else:
                payload = {
                    "model": self.response_model,  # Use English-optimized model for response generation
                    "messages": [
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    "temperature": 0.2,  # Very low temperature for consistent medical responses
                    "max_tokens": 512  # Reduced to enforce WhatsApp length limit
                }
                
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result["choices"][0]["message"]["content"].strip()
                    model_used = "gemma"
                else:
                    logger.error(f"Groq API error: {response.status_code} - {response.text}")
                    return f"Error generating response: {response.status_code}"
            
            # Debug: Log what the LLM generated
            logger.info(f"🤖 LLM GENERATED ANSWER: '{answer}'")
            logger.info(f"🔍 Is no-answer response: {self._is_no_answer_response(answer)}")
            
            # Always clean inline citations and ensure proper end citation
            if not self._is_no_answer_response(answer):
                # Clean any inline citations first
                answer = self._clean_inline_citations(answer)
                
                # Add proper citation at the end if not present
                if not self._has_citation(answer):
                    answer = self._add_automatic_citation(answer, metadatas)
            
            return answer, model_used
                
        except Exception as e:
            logger.error(f"Error generating answer with citations: {e}")
            return f"Error generating answer: {str(e)}", "error"
    
    async def _call_medgemma_endpoint(self, prompt: str, citation_text: str, question: str) -> tuple:
        """Call MedGemma endpoint with HuggingFace native format, fallback to Groq if unavailable
        Returns: (response_text, model_used)"""
        try:
            logger.info("🩺 Using MedGemma for response generation...")
            logger.info(f"🔍 Debug: citation_text = {citation_text[:100]}...")
            logger.info(f"🔍 Debug: question = {question[:50]}...")
            
            # Create enhanced prompt for MedGemma with examples and medical structure
            citation_format = "[ Sources : doc_name: pg 1,2,3 ; other_doc: pg 4,5 ]"
            logger.info(f"🔍 Debug: citation_format = {citation_format}")
            
            try:
                # Build prompt using string concatenation to avoid f-string issues with { Sources }
                prompt_parts = [
                    "You are a medical expert providing evidence-based palliative care guidance. Analyze the provided medical literature and give structured, actionable advice.",
                    "",
                    "CITATION REQUIREMENTS:",
                    "- ONLY cite at the END of your response - NO inline citations in the text",
                    "- End your response with: " + citation_format,
                    "- Multiple pages from same document: separate with commas", 
                    "- Multiple documents: separate with semicolons",
                    "",
                    "EXAMPLE FORMAT:",
                    "Question: How to manage pain in bedridden patients?",
                    "Medical Context: [Document: pain management guide, Page 23] Pain assessment should be done every 4 hours using standardized scales. Repositioning every 2 hours prevents pressure sores.",
                    "",
                    "Response:",
                    "*Pain Assessment:*",
                    "• Use 0-10 pain scale every 4 hours",
                    "• Document pain triggers and relief patterns",
                    "",
                    "*Positioning Care:*",
                    "• Reposition patient every 2 hours",
                    "• Use pressure-relieving mattress",
                    "• Check skin integrity at pressure points",
                    "",
                    "*Medication Protocol:*",
                    "• Start with paracetamol 500mg every 6 hours",
                    "• Add weak opioids if pain >4/10",
                    "",
                    "[ Sources : pain_management_guide: pg 23 ]",
                    "",
                    "EXAMPLE 2:",
                    "Question: How to provide tracheostomy care?",
                    "Medical Context: [Document: nursing handbook, Page 67] Suction tracheostomy when secretions accumulate. Clean around stoma twice daily.",
                    "",
                    "Response:",
                    "*Suctioning Technique:*",
                    "• Suction when secretions visible or audible",
                    "• Use sterile technique, limit to 15 seconds",
                    "• Pre-oxygenate before suctioning",
                    "",
                    "*Daily Stoma Care:*",
                    "• Clean around stoma twice daily with saline",
                    "• Change tracheostomy ties when soiled",
                    "• Monitor for signs of infection",
                    "",
                    "[ Sources : nursing_handbook: pg 67 ]",
                    "",
                    "NOW ANSWER THIS QUESTION:",
                    "",
                    "MEDICAL LITERATURE:",
                    citation_text,
                    "",
                    "QUESTION: " + question,
                    "",
                    "STRUCTURED MEDICAL RESPONSE (UNDER 1500 CHARS):"
                ]
                
                english_enforced_prompt = "\n".join(prompt_parts)
                logger.info("🔍 Debug: Prompt creation successful")
                
            except Exception as prompt_error:
                logger.error(f"🔍 Debug: Prompt creation failed: {prompt_error}")
                return f"Error creating prompt: {prompt_error}", "error"
            
            # Format prompt for MedGemma (using the same format as test script)
            try:
                formatted_prompt = f"<start_of_turn>user\n{english_enforced_prompt}<end_of_turn>\n<start_of_turn>model\n"
                logger.info("🔍 Debug: Formatted prompt creation successful")
            except Exception as format_error:
                logger.error(f"🔍 Debug: Formatted prompt creation failed: {format_error}")
                return f"Error formatting prompt: {format_error}", "error"
            
            # Debug: Log prompt details to identify 422 cause
            logger.info(f"🔍 MedGemma prompt length: {len(formatted_prompt)} chars")
            logger.info(f"🔍 MedGemma prompt preview: {formatted_prompt[:200]}...")
            
            # Truncate if too long to prevent 422 errors
            MAX_PROMPT_LENGTH = 8000  # Higher limit for better quality responses
            if len(formatted_prompt) > MAX_PROMPT_LENGTH:
                logger.warning(f"⚠️ Truncating long prompt from {len(formatted_prompt)} to {MAX_PROMPT_LENGTH} chars")
                # Truncate but preserve the format
                truncated_content = english_enforced_prompt[:MAX_PROMPT_LENGTH-200]  # Leave room for formatting
                formatted_prompt = f"<start_of_turn>user\n{truncated_content}<end_of_turn>\n<start_of_turn>model\n"
            
            headers = {
                "Content-Type": "application/json"
            }
            
            # Use HuggingFace native format (matching the working test script)
            payload = {
                "inputs": formatted_prompt,
                "parameters": {
                    "max_new_tokens": 512,
                    "temperature": 0.2,  # Match closer to test script values
                    "top_p": 0.9,        # Match test script
                    "do_sample": True,
                    "return_full_text": False,
                    "stop": ["<end_of_turn>"],  # Keep only the basic stop token like test script
                    "repetition_penalty": 1.1
                },
                "options": {
                    "use_cache": False,
                    "wait_for_model": True,
                    "use_gpu": True
                },
                "stream": False
            }
            
            medgemma_url = "https://izqynpy6ktzegc74.us-east4.gcp.endpoints.huggingface.cloud"
            
            # Try MedGemma endpoint first
            logger.info(f"🩺 Calling MedGemma endpoint: {medgemma_url}")
            response = requests.post(
                medgemma_url,
                headers=headers,
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ MedGemma response received successfully")
                
                if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
                    generated_text = result[0]["generated_text"].strip()
                    
                    # Check for Hindi response and fallback if detected
                    if self._contains_non_english_script(generated_text):
                        logger.warning("⚠️ MedGemma responded in Hindi, falling back to Groq...")
                        return await self._fallback_to_groq(prompt)
                    
                    logger.info(f"🩺 MedGemma generated response: {generated_text[:100]}...")
                    return generated_text, "medgemma"
                else:
                    logger.error(f"❌ Unexpected MedGemma response format: {result}")
                    return await self._fallback_to_groq(prompt)
            else:
                logger.error(f"❌ MedGemma API error: {response.status_code} - {response.text}")
                logger.info("🔄 Falling back to Groq API...")
                return await self._fallback_to_groq(prompt)
                
        except Exception as e:
            logger.error(f"❌ MedGemma endpoint error: {e}")
            logger.info("🔄 Falling back to Groq API...")
            return await self._fallback_to_groq(prompt)
    
    def _contains_non_english_script(self, text: str) -> bool:
        """Check if text contains non-English scripts (Hindi, etc.)"""
        try:
            import unicodedata
            for char in text:
                script = unicodedata.name(char, '').split(' ')[0] if unicodedata.name(char, '') else ''
                if any(non_eng in script for non_eng in ['DEVANAGARI', 'BENGALI', 'TAMIL', 'GUJARATI']):
                    return True
            return False
        except:
            # Fallback: simple check for common Hindi characters
            hindi_chars = ['ह', 'न', 'म', 'स', 'त', 'े', 'ा', 'ि', 'ी', 'ु', 'ू', 'ं', 'ः']
            return any(char in text for char in hindi_chars)
    
    async def _fallback_to_groq(self, prompt: str) -> tuple:
        """Fallback to Groq API when MedGemma fails"""
        try:
            logger.info("🔄 Using Groq fallback...")
            
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gemma2-9b-it",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 512
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"].strip()
                logger.info("✅ Groq fallback successful")
                return answer, "gemma"
            else:
                logger.error(f"Groq fallback error: {response.status_code} - {response.text}")
                return f"Error generating response: {response.status_code}", "error"
                
        except Exception as e:
            logger.error(f"Groq fallback error: {e}")
            return f"Error calling fallback API: {str(e)}", "error"
    
    def _format_citation_context(self, context: str, metadatas: List[Dict]) -> str:
        """Format context with clear source indicators for citation"""
        sections = []
        short_citations = []
        for i, meta in enumerate(metadatas):
            filename = meta['filename'].replace('.pdf', '').replace('_', ' ').replace('-', ' ')
            chunk_num = meta['chunk_index'] + 1
            total_chunks = meta['total_chunks']
            page_count = meta.get('page_count', 1)
            
            # Calculate accurate page number based on actual page count
            estimated_page = max(1, int((chunk_num / total_chunks) * page_count))
            
            # Create short citation format for the LLM to use
            short_filename = meta['filename'].replace('.pdf', '').replace(' ', '_').replace('-', '_').lower()
            short_citation = f"{short_filename}_pg{estimated_page}"
            
            sections.append(f"[Document: {filename}, Chunk {chunk_num}, Page {estimated_page}]")
            short_citations.append(short_citation)
        
        citation_examples = ", ".join(short_citations)
        return context + f"\n\nAvailable sources: " + "; ".join(sections) + f"\n\nFor citations, use these formats: {citation_examples}"
    
    def _has_citation(self, answer: str) -> bool:
        """Check if answer already has a citation in square or curly braces"""
        return (('{' in answer and '}' in answer) or ('[' in answer and ']' in answer)) and \
               ('retrieved from' in answer.lower() or 'sources' in answer.lower())
    
    def _is_no_answer_response(self, answer: str) -> bool:
        """Check if the response indicates insufficient information"""
        no_answer_indicators = [
            "INSUFFICIENT_INFORMATION",
            "don't have enough information",
            "context doesn't contain",
            "cannot answer",
            "not enough information",
            "insufficient context",
            "unable to answer"
        ]
        return any(indicator.lower() in answer.lower() for indicator in no_answer_indicators)
    
    def _clean_inline_citations(self, answer: str) -> str:
        """Remove inline citations and clean up the text"""
        import re
        
        # Remove inline citations like (document_name_pg42)
        answer = re.sub(r'\([^)]*_pg\d+[^)]*\)', '', answer)
        
        # Remove citations in the format: doc_name_pg42, doc_name_pg43
        answer = re.sub(r'[a-zA-Z0-9_]+_pg\d+(?:,\s*[a-zA-Z0-9_]+_pg\d+)*', '', answer)
        
        # Remove any citation blocks including [ Sources : ... ], { Sources : ... } and { retrieved from: ... }
        answer = re.sub(r'[\[\{]\s*[Ss]ources?\s*:.*?[\]\}]', '', answer, flags=re.DOTALL)
        answer = re.sub(r'\{\s*retrieved\s+from:.*?\}', '', answer, flags=re.DOTALL)
        
        # Remove any standalone citations in curly braces
        lines = answer.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip lines that are just citations or mostly citations
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append(line)
                continue
                
            if ((stripped.startswith('{') or stripped.startswith('[')) and 
                ('retrieved from' in stripped.lower() or 'sources' in stripped.lower())):
                continue
                
            # Skip lines that are just document references
            if re.match(r'^[a-zA-Z0-9_]+_pg\d+(?:,\s*[a-zA-Z0-9_]+_pg\d+)*\s*$', stripped):
                continue
                
            cleaned_lines.append(line)
        
        # Clean up extra whitespace
        answer = '\n'.join(cleaned_lines)
        answer = re.sub(r'\n\s*\n\s*\n', '\n\n', answer)  # Remove excessive newlines
        answer = re.sub(r'\s+$', '', answer, flags=re.MULTILINE)  # Remove trailing spaces
        
        return answer.strip()

    def _add_automatic_citation(self, answer: str, metadatas: List[Dict]) -> str:
        """Add automatic citation if the model didn't include one
        Format: [ Sources : doc_name: pg 1,2,3 ; other_doc: pg 4,5 ]"""
        if not metadatas:
            return answer
        
        # Group pages by document
        doc_pages = {}
        for meta in metadatas:
            chunk_num = meta['chunk_index'] + 1
            total_chunks = meta['total_chunks']
            page_count = meta.get('page_count', 1)
            
            # Calculate accurate page number based on actual page count
            estimated_page = max(1, int((chunk_num / total_chunks) * page_count))
            
            # Clean document name
            doc_name = meta['filename'].replace('.pdf', '').replace(' ', '_').replace('-', '_').lower()
            
            if doc_name not in doc_pages:
                doc_pages[doc_name] = []
            if estimated_page not in doc_pages[doc_name]:
                doc_pages[doc_name].append(estimated_page)
        
        # Format citation according to specifications
        # Format: [ Sources : doc_name: pg 1,2,3 ; other_doc: pg 4,5 ]
        citation_parts = []
        for doc_name, pages in doc_pages.items():
            pages.sort()  # Sort pages numerically
            pages_str = ','.join(map(str, pages))
            citation_parts.append(f"{doc_name}: pg {pages_str}")
        
        citation = "\n\n[ Sources : " + " ; ".join(citation_parts) + " ]"
        
        return answer + citation
    
    async def translate_query_to_english(self, query: str, source_language: str) -> Dict[str, Any]:
        """Translate query from local language to English for better embedding matching"""
        try:
            if not self.groq_api_key:
                return {"status": "error", "error": "GROQ_API_KEY not configured"}
            
            # Language mapping for better translation prompts
            language_names = {
                "hi": "Hindi",
                "bn": "Bengali", 
                "ta": "Tamil",
                "gu": "Gujarati",
                "en": "English"
            }
            
            if source_language not in language_names:
                return {"status": "error", "error": f"Unsupported language: {source_language}"}
            
            if source_language == "en":
                # Already in English, no translation needed
                return {"status": "success", "translated_query": query, "original_query": query}
            
            source_lang_name = language_names[source_language]
            
            prompt = f"""You are an expert medical translator. Translate the following {source_lang_name} medical query to English.

CRITICAL REQUIREMENTS:
- Maintain the exact medical meaning and intent
- Preserve medical terminology accuracy
- Keep the translation concise and clear
- Focus on capturing the medical question or concern

{source_lang_name} Query: {query}

Respond with ONLY the English translation, no explanations or additional text."""

            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.translation_model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 500
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post("https://api.groq.com/openai/v1/chat/completions", 
                                       headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        translated_query = result['choices'][0]['message']['content'].strip()
                        logger.info(f"Query translation: {source_lang_name} '{query}' -> English '{translated_query}'")
                        return {
                            "status": "success", 
                            "translated_query": translated_query,
                            "original_query": query
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"Query translation API error {response.status}: {error_text}")
                        return {"status": "error", "error": f"Translation API error: {response.status}"}
                        
        except Exception as e:
            logger.error(f"Query translation error: {str(e)}")
            # Fallback: use original query if translation fails
            return {"status": "fallback", "translated_query": query, "original_query": query}

    async def translate_text(self, text: str, target_language: str) -> Dict[str, Any]:
        """Translate text to target language using Groq API"""
        try:
            if not self.groq_api_key:
                return {"status": "error", "error": "GROQ_API_KEY not configured"}
            
            # Language mapping for better translation prompts
            language_names = {
                "hi": "Hindi",
                "bn": "Bengali",
                "ta": "Tamil",
                "gu": "Gujarati", 
                "en": "English"
            }
            
            if target_language not in language_names:
                return {"status": "error", "error": f"Unsupported language: {target_language}"}
            
            if target_language == "en":
                # No translation needed
                return {"status": "success", "translated_text": text, "target_language": target_language}
            
            target_lang_name = language_names[target_language]
            
            prompt = f"""You are an expert medical translator. Translate the following English medical text to {target_lang_name}. 

CRITICAL REQUIREMENTS:
🚨 Your translated response MUST BE UNDER 1500 CHARACTERS including any citations! 🚨
- Maintain medical accuracy and terminology
- Preserve any citations in their original format
- Use natural, fluent {target_lang_name}
- Keep the same structure and meaning

English Text to Translate:
{text}

{target_lang_name} Translation (UNDER 1500 CHARS):"""

            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.translation_model,  # Good multilingual model
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,  # Very low temperature for consistent translations
                "max_tokens": 512  # Keep within WhatsApp limits
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                translated_text = result["choices"][0]["message"]["content"].strip()
                
                logger.info(f"🌐 TRANSLATION: {target_language}")
                logger.info(f"  📏 Original length: {len(text)}")
                logger.info(f"  📏 Translated length: {len(translated_text)}")
                
                return {
                    "status": "success",
                    "translated_text": translated_text,
                    "target_language": target_language,
                    "original_text": text
                }
            else:
                logger.error(f"Translation API error: {response.status_code} - {response.text}")
                return {
                    "status": "error",
                    "error": f"Translation failed: {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index"""
        try:
            collection_count = self.vector_db.count()
            
            stats = {
                "total_chunks": collection_count,
                "total_documents": len(self.document_metadata),
                "conversations": len(self.conversations),
                "documents": []
            }
            
            for doc_id, metadata in self.document_metadata.items():
                stats["documents"].append({
                    "filename": metadata["filename"],
                    "chunks": metadata["chunk_count"],
                    "indexed_at": metadata["indexed_at"]
                })
            
            return {
                "status": "success",
                "stats": stats
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def check_database_health(self) -> Dict[str, Any]:
        """Check vector database health and detect corruption"""
        try:
            return self.health_manager.detect_corruption()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "is_corrupted": True,
                "corruption_score": 100,
                "issues": [f"Health check failed: {str(e)}"],
                "severity": "critical",
                "timestamp": datetime.now().isoformat()
            }
    
    async def auto_rebuild_database(self) -> Dict[str, Any]:
        """Automatically rebuild database if corruption detected"""
        try:
            return await self.rebuild_manager.auto_rebuild_if_needed()
        except Exception as e:
            logger.error(f"Auto-rebuild failed: {e}")
            return {
                "status": "error",
                "message": f"Auto-rebuild failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def query_with_auto_recovery(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """Query with automatic corruption detection and recovery"""
        try:
            # First, try normal query
            result = await self.query(question, top_k)
            
            # Check if query failed in a way that suggests corruption
            if (result.get("status") == "error" or 
                "could not find the answer" in result.get("answer", "").lower()):
                
                logger.info("Query failed, checking database health...")
                health_status = self.check_database_health()
                
                if health_status["is_corrupted"]:
                    logger.warning("Corruption detected, attempting auto-rebuild...")
                    
                    rebuild_result = await self.auto_rebuild_database()
                    
                    if rebuild_result.get("status") == "success":
                        logger.info("Database rebuilt, retrying query...")
                        # Retry the query after successful rebuild
                        result = await self.query(question, top_k)
                        
                        # Add rebuild info to result
                        result["rebuild_performed"] = True
                        result["rebuild_stats"] = rebuild_result.get("rebuild_stats")
                    else:
                        result["rebuild_attempted"] = True
                        result["rebuild_error"] = rebuild_result.get("message")
                else:
                    result["health_check_performed"] = True
                    result["database_healthy"] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Query with auto-recovery failed: {e}")
            return {
                "status": "error",
                "answer": f"Query failed: {str(e)}",
                "error": str(e)
            }


class SimpleAdminUI:
    """Simplified web-based Admin UI"""
    
    def __init__(self, rag_pipeline: SimpleRAGPipeline):
        self.rag_pipeline = rag_pipeline
        self.upload_dir = Path("uploads")
        self.upload_dir.mkdir(exist_ok=True)
    
    def create_gradio_interface(self):
        """Create Gradio interface for admin UI"""
        
        with gr.Blocks(title="Simple RAG Admin UI", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# Simple RAG Pipeline Admin Interface")
            gr.Markdown("*No database required - file-based storage only*")
            
            # Initialize documents_table as None, will be created later
            documents_table = None
            
            with gr.Tabs() as tabs:
                # File Upload Tab
                with gr.TabItem("📁 Upload Documents"):
                    gr.Markdown("## Upload documents to the RAG corpus")
                    
                    with gr.Row():
                        with gr.Column():
                            file_upload = gr.File(
                                label="Select files to upload",
                                file_count="multiple",
                                file_types=[".pdf", ".txt", ".docx", ".md"]
                            )
                            
                            metadata_json = gr.Textbox(
                                label="Metadata (JSON format) - Optional",
                                placeholder='{"category": "medical", "language": "en"}',
                                lines=3
                            )
                            
                            with gr.Row():
                                upload_btn = gr.Button("Upload & Index", variant="primary")
                                clear_btn = gr.Button("Clear Form", variant="secondary")
                        
                        with gr.Column():
                            upload_status = gr.Textbox(
                                label="Upload Status",
                                lines=10,
                                interactive=False
                            )
                    
                    upload_btn.click(
                        fn=self._handle_file_upload,
                        inputs=[file_upload, metadata_json],
                        outputs=[upload_status, file_upload, metadata_json]
                    )
                    
                    clear_btn.click(
                        fn=lambda: ("", None, ""),
                        inputs=[],
                        outputs=[upload_status, file_upload, metadata_json]
                    )
                
                # Query Test Tab
                with gr.TabItem("💬 Test Queries"):
                    gr.Markdown("## Test RAG pipeline with queries")
                    
                    with gr.Row():
                        with gr.Column():
                            query_input = gr.Textbox(
                                label="Enter your query",
                                placeholder="What is palliative care?",
                                lines=3
                            )
                            
                            with gr.Row():
                                query_btn = gr.Button("Submit Query", variant="primary")
                                clear_query_btn = gr.Button("Clear", variant="secondary")
                        
                        with gr.Column():
                            query_response = gr.Textbox(
                                label="Response",
                                lines=10,
                                interactive=False
                            )
                            
                            sources_output = gr.JSON(label="Sources & Citations")
                    
                    query_btn.click(
                        fn=self._handle_query,
                        inputs=[query_input],
                        outputs=[query_response, sources_output]
                    )
                    
                    clear_query_btn.click(
                        fn=lambda: ("", "", {}),
                        inputs=[],
                        outputs=[query_input, query_response, sources_output]
                    )
                
                # Document Management Tab
                with gr.TabItem("📋 Manage Documents"):
                    gr.Markdown("## View and manage documents in the corpus")
                    gr.Markdown("💡 **Tip:** Select a document from the dropdown, then click Remove Selected")
                    
                    with gr.Row():
                        with gr.Column(scale=4):
                            documents_table = gr.Dataframe(
                                label="Documents in Corpus",
                                headers=["Filename", "Chunks", "Pages", "Indexed At"],
                                interactive=False,
                                wrap=True
                            )
                        
                        with gr.Column(scale=1):
                            document_dropdown = gr.Dropdown(
                                label="Select Document to Remove",
                                choices=[],
                                value=None,
                                interactive=True
                            )
                            
                            selected_doc_info = gr.Textbox(
                                label="Selected Document Info",
                                placeholder="No document selected",
                                lines=3,
                                interactive=False
                            )
                            
                            with gr.Row():
                                remove_btn = gr.Button("🗑️ Remove Selected", variant="stop")
                                refresh_docs_btn = gr.Button("🔄 Refresh", variant="secondary")
                            
                            removal_status = gr.Textbox(
                                label="Status",
                                lines=4,
                                interactive=False
                            )
                    
                    # Hidden component to store selected document ID
                    selected_doc_id = gr.Textbox(visible=False)
                    
                    document_dropdown.change(
                        fn=self._handle_document_selection,
                        inputs=[document_dropdown],
                        outputs=[selected_doc_info, selected_doc_id]
                    )
                    
                    refresh_docs_btn.click(
                        fn=self._refresh_documents,
                        inputs=[],
                        outputs=[documents_table, document_dropdown]
                    )
                    
                    remove_btn.click(
                        fn=self._handle_document_removal,
                        inputs=[selected_doc_id],
                        outputs=[removal_status, documents_table, document_dropdown, selected_doc_info, selected_doc_id]
                    )
                
                # Index Stats Tab
                with gr.TabItem("📊 Index Statistics"):
                    gr.Markdown("## View corpus statistics")
                    
                    refresh_btn = gr.Button("Refresh Stats", variant="secondary")
                    stats_output = gr.JSON(label="Index Statistics")
                    
                    refresh_btn.click(
                        fn=self._get_stats,
                        inputs=[],
                        outputs=[stats_output]
                    )
                
                # Database Health Tab
                with gr.TabItem("🏥 Database Health"):
                    gr.Markdown("## Monitor and repair vector database health")
                    gr.Markdown("*Detects corruption and performs automatic rebuilds when needed*")
                    
                    with gr.Row():
                        with gr.Column():
                            health_check_btn = gr.Button("🔍 Check Health", variant="primary")
                            auto_rebuild_btn = gr.Button("🔧 Auto Rebuild", variant="secondary")
                            manual_rebuild_btn = gr.Button("⚡ Force Rebuild", variant="stop")
                        
                        with gr.Column():
                            health_status = gr.JSON(
                                label="Health Status",
                                value={"status": "Click 'Check Health' to scan database"}
                            )
                    
                    with gr.Row():
                        rebuild_log = gr.Textbox(
                            label="Rebuild Log",
                            lines=8,
                            interactive=False,
                            placeholder="Rebuild operations will be logged here..."
                        )
                    
                    # Health check button
                    health_check_btn.click(
                        fn=self._check_health_status,
                        inputs=[],
                        outputs=[health_status]
                    )
                    
                    # Auto rebuild button
                    auto_rebuild_btn.click(
                        fn=self._handle_auto_rebuild,
                        inputs=[],
                        outputs=[rebuild_log, health_status]
                    )
                    
                    # Manual rebuild button
                    manual_rebuild_btn.click(
                        fn=self._handle_manual_rebuild,
                        inputs=[],
                        outputs=[rebuild_log, health_status]
                    )
        
            # Refresh documents when the management tab is selected
            tabs.select(
                fn=self._handle_tab_change,
                inputs=[],
                outputs=[documents_table, document_dropdown]
            )
            
            # Load initial data when the interface starts
            demo.load(
                fn=self._refresh_documents,
                inputs=[],
                outputs=[documents_table, document_dropdown]
            )
        
        return demo
    
    def _handle_file_upload(self, files, metadata_str):
        """Handle file upload and indexing"""
        try:
            if not files:
                return "No files uploaded", None, ""
            
            # Parse metadata
            metadata = {}
            if metadata_str and metadata_str.strip():
                try:
                    metadata = json.loads(metadata_str)
                except json.JSONDecodeError:
                    return "Invalid JSON metadata format", files, metadata_str
            
            # Process uploaded files
            file_paths = []
            import shutil
            
            for file in files:
                try:
                    # Debug: Log file information
                    logger.info(f"Processing file: {file} (type: {type(file)})")
                    
                    # Gradio provides files as temporary file paths (strings)
                    if isinstance(file, str):
                        source_path = Path(file)
                        if source_path.exists():
                            logger.info(f"Source file exists: {source_path}, size: {source_path.stat().st_size} bytes")
                            file_paths.append(str(source_path))
                            logger.info(f"Using file directly: {source_path}")
                        else:
                            return f"File not found: {file}", files, metadata_str, current_table
                    
                    # Handle file-like objects (alternative Gradio format)
                    elif hasattr(file, 'name'):
                        # Extract filename
                        if hasattr(file, 'orig_name'):
                            filename = file.orig_name
                        elif hasattr(file, 'name'):
                            filename = Path(file.name).name
                        else:
                            filename = "uploaded_file"
                        
                        logger.info(f"Processing file object with name: {filename}")
                        
                        # Use the file directly if it's a path
                        if hasattr(file, 'name') and Path(file.name).exists():
                            file_paths.append(str(file.name))
                            logger.info(f"Using file object directly: {file.name}")
                    
                    else:
                        return f"Unsupported file type: {type(file)} - {file}", files, metadata_str
                        
                except Exception as file_error:
                    logger.error(f"Error processing file {file}: {str(file_error)}")
                    return f"Error processing file {file}: {str(file_error)}\n{traceback.format_exc()}", files, metadata_str
            
            # Index files
            result = asyncio.run(self.rag_pipeline.add_documents(file_paths, metadata))
            
            # Format success message and refresh table
            if result["status"] == "success":
                success_msg = f"✅ Successfully uploaded {result['successful']}/{result['total_files']} files\n\n"
                for res in result["results"]:
                    if res["status"] == "success":
                        success_msg += f"📄 {Path(res['file_path']).name}: {res['chunks']} chunks indexed\n"
                    else:
                        success_msg += f"❌ {Path(res['file_path']).name}: {res['error']}\n"
                success_msg += "\n🔄 Ready for next upload!\n💡 Visit the 'Manage Documents' tab to see the new documents."
                
                # Get updated documents table
                updated_table = self._get_documents_table()
                
                return success_msg, None, ""
            else:
                return f"❌ Upload failed: {result['error']}", files, metadata_str
            
        except Exception as e:
            return f"Error: {str(e)}\n{traceback.format_exc()}", files, metadata_str
    
    def _handle_query(self, query):
        """Handle query testing"""
        try:
            if not query.strip():
                return "Please enter a query", {}
            
            result = asyncio.run(self.rag_pipeline.query(query))
            
            if result["status"] == "success":
                # Enhanced response formatting
                answer = result["answer"]
                sources_info = {
                    "sources": result.get("sources", []),
                    "context_used": result.get("context_used", 0),
                    "timestamp": result.get("timestamp", "")
                }
                return answer, sources_info
            else:
                return f"❌ Query failed: {result['error']}", {}
                
        except Exception as e:
            return f"❌ Error: {str(e)}", {}
    
    def _get_documents_table(self):
        """Get documents as a table for the management interface"""
        try:
            docs_data = []
            
            for doc_id, metadata in self.rag_pipeline.document_metadata.items():
                docs_data.append([
                    metadata["filename"],
                    metadata["chunk_count"],
                    metadata.get("page_count", "N/A"),
                    metadata["indexed_at"][:19].replace("T", " ")  # Format datetime
                ])
            
            return docs_data
        except Exception as e:
            logger.error(f"Error getting documents table: {e}")
            return [["Error loading documents", "", "", ""]]
    
    def _get_document_dropdown_choices(self):
        """Get dropdown choices for document selection"""
        try:
            choices = []
            for doc_id, metadata in self.rag_pipeline.document_metadata.items():
                display_name = f"{metadata['filename']} ({metadata['chunk_count']} chunks)"
                choices.append((display_name, doc_id))
            return choices
        except Exception as e:
            logger.error(f"Error getting dropdown choices: {e}")
            return []
    
    def _refresh_documents(self):
        """Refresh both table and dropdown"""
        table_data = self._get_documents_table()
        dropdown_choices = self._get_document_dropdown_choices()
        return table_data, gr.Dropdown(choices=dropdown_choices, value=None)
    
    def _handle_tab_change(self):
        """Handle tab change - refresh documents data"""
        return self._refresh_documents()
    
    def _handle_document_selection(self, selected_doc_id):
        """Handle document selection from dropdown"""
        try:
            if not selected_doc_id:
                return "No document selected", ""
            
            # Get document info for display
            if selected_doc_id in self.rag_pipeline.document_metadata:
                metadata = self.rag_pipeline.document_metadata[selected_doc_id]
                selected_info = f"📄 {metadata['filename']}\n🆔 {selected_doc_id[:16]}...\n📊 {metadata['chunk_count']} chunks"
                return selected_info, selected_doc_id
            
            return "Document not found", ""
        except Exception as e:
            logger.error(f"Selection error: {e}")
            return "Selection error", ""
    
    def _handle_document_removal(self, doc_id):
        """Handle document removal"""
        try:
            if not doc_id or not doc_id.strip():
                table_data = self._get_documents_table()
                dropdown_choices = self._get_document_dropdown_choices()
                return "❌ Please select a document to remove", table_data, gr.Dropdown(choices=dropdown_choices, value=None), "No document selected", ""
            
            result = asyncio.run(self.rag_pipeline.remove_document(doc_id.strip()))
            
            # Refresh both table and dropdown after removal
            table_data = self._get_documents_table()
            dropdown_choices = self._get_document_dropdown_choices()
            
            if result["status"] == "success":
                success_msg = f"✅ {result['message']}\n🗑️ Removed {result['chunks_removed']} chunks"
                return success_msg, table_data, gr.Dropdown(choices=dropdown_choices, value=None), "No document selected", ""
            else:
                return f"❌ Removal failed: {result['error']}", table_data, gr.Dropdown(choices=dropdown_choices, value=None), "No document selected", ""
                
        except Exception as e:
            table_data = self._get_documents_table()
            dropdown_choices = self._get_document_dropdown_choices()
            return f"❌ Error: {str(e)}", table_data, gr.Dropdown(choices=dropdown_choices, value=None), "No document selected", ""
    
    def _get_stats(self):
        """Get index statistics"""
        try:
            return self.rag_pipeline.get_index_stats()
        except Exception as e:
            return {"error": str(e)}
    
    def _check_health_status(self):
        """Check database health status"""
        try:
            health_status = self.rag_pipeline.check_database_health()
            return health_status
        except Exception as e:
            return {
                "error": f"Health check failed: {str(e)}",
                "is_corrupted": True,
                "severity": "critical"
            }
    
    def _handle_auto_rebuild(self):
        """Handle automatic rebuild based on corruption detection"""
        try:
            # Run the auto rebuild in a thread to prevent UI blocking
            import asyncio
            
            # Create event loop if none exists
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the auto rebuild
            result = loop.run_until_complete(self.rag_pipeline.auto_rebuild_database())
            
            # Format log output
            if result["status"] == "success":
                log_output = f"""🔧 AUTO REBUILD COMPLETED SUCCESSFULLY
                
⏱️  Duration: {result.get('rebuild_stats', {}).get('duration_seconds', 'N/A')}s
📄 Documents processed: {result.get('rebuild_stats', {}).get('documents_processed', 'N/A')}
🧩 Chunks created: {result.get('rebuild_stats', {}).get('chunks_created', 'N/A')}

✅ {result['message']}"""
            elif result["status"] == "healthy":
                log_output = f"✅ DATABASE IS HEALTHY\n\n{result['message']}"
            elif result["status"] == "already_rebuilding":
                log_output = f"⏳ REBUILD IN PROGRESS\n\n{result['message']}"
            else:
                log_output = f"❌ AUTO REBUILD FAILED\n\n{result['message']}"
            
            # Get updated health status
            health_status = self.rag_pipeline.check_database_health()
            
            return log_output, health_status
            
        except Exception as e:
            error_log = f"❌ AUTO REBUILD ERROR\n\nException: {str(e)}"
            error_health = {
                "error": f"Auto rebuild failed: {str(e)}",
                "is_corrupted": True,
                "severity": "critical"
            }
            return error_log, error_health
    
    def _handle_manual_rebuild(self):
        """Handle manual/forced rebuild"""
        try:
            import asyncio
            
            # Create event loop if none exists
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Force rebuild by directly calling the rebuild manager
            result = loop.run_until_complete(
                self.rag_pipeline.rebuild_manager._perform_auto_rebuild({
                    "is_corrupted": True,
                    "corruption_score": 100,
                    "issues": ["Manual rebuild requested"],
                    "severity": "manual"
                })
            )
            
            # Format log output
            if result["status"] == "success":
                log_output = f"""⚡ MANUAL REBUILD COMPLETED
                
⏱️  Duration: {result.get('rebuild_stats', {}).get('duration_seconds', 'N/A')}s
📄 Documents processed: {result.get('rebuild_stats', {}).get('documents_processed', 'N/A')}
🧩 Chunks created: {result.get('rebuild_stats', {}).get('chunks_created', 'N/A')}

✅ {result['message']}"""
            else:
                log_output = f"❌ MANUAL REBUILD FAILED\n\n{result['message']}"
            
            # Get updated health status
            health_status = self.rag_pipeline.check_database_health()
            
            return log_output, health_status
            
        except Exception as e:
            error_log = f"❌ MANUAL REBUILD ERROR\n\nException: {str(e)}"
            error_health = {
                "error": f"Manual rebuild failed: {str(e)}",
                "is_corrupted": True,
                "severity": "critical"
            }
            return error_log, error_health


class NgrokManager:
    """Manage ngrok tunnel for external access"""
    
    def __init__(self):
        self.ngrok_process = None
        self.ngrok_url = None
    
    def start_ngrok(self, port: int = 8000) -> Optional[str]:
        """Start ngrok tunnel"""
        try:
            # Check if ngrok is installed
            result = subprocess.run(["ngrok", "version"], capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning("⚠️ ngrok not found. WhatsApp webhooks will only work locally.")
                return None
            
            logger.info("🌐 Starting ngrok tunnel...")
            
            # Start ngrok
            self.ngrok_process = subprocess.Popen(
                ["ngrok", "http", str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for ngrok to start
            time.sleep(3)
            
            try:
                # Get ngrok tunnels
                import requests
                response = requests.get("http://127.0.0.1:4040/api/tunnels", timeout=5)
                if response.status_code == 200:
                    tunnels = response.json().get("tunnels", [])
                    for tunnel in tunnels:
                        if tunnel.get("proto") == "https":
                            self.ngrok_url = tunnel.get("public_url")
                            logger.info(f"🌐 ngrok tunnel started: {self.ngrok_url}")
                            return self.ngrok_url
            except Exception as e:
                logger.warning(f"Could not get ngrok URL: {e}")
            
            logger.warning("⚠️ Could not get ngrok URL. Check ngrok status manually.")
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to start ngrok: {e}")
            return None
    
    def stop_ngrok(self):
        """Stop ngrok tunnel"""
        if self.ngrok_process:
            logger.info("🛑 Stopping ngrok tunnel...")
            self.ngrok_process.terminate()
            self.ngrok_process = None
            self.ngrok_url = None


def main():
    """Main application entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple RAG Server (No Database)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--no-ngrok", action="store_true", help="Disable ngrok tunnel")
    
    args = parser.parse_args()
    
    # Initialize components
    rag_pipeline = SimpleRAGPipeline()
    admin_ui = SimpleAdminUI(rag_pipeline)
    
    # Initialize ngrok manager
    ngrok_manager = NgrokManager()
    
    # Check if WhatsApp integration is configured
    twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
    twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
    
    # Debug: Print what we found
    logger.info(f"🔍 Checking Twilio configuration:")
    logger.info(f"   TWILIO_ACCOUNT_SID: {'✅ Set' if twilio_sid else '❌ Not set'} ({twilio_sid[:10]}... if set)")
    logger.info(f"   TWILIO_AUTH_TOKEN: {'✅ Set' if twilio_token else '❌ Not set'} ({'*' * 10 if twilio_token else 'None'})")
    
    twilio_configured = (
        twilio_sid and 
        twilio_token and
        twilio_sid != "your_twilio_account_sid_here"
    )
    
    # Initialize WhatsApp bot if configured
    whatsapp_bot = None
    if twilio_configured:
        try:
            from whatsapp_bot import EnhancedWhatsAppBot, EnhancedSTTService, EnhancedTTSService
            stt_service = EnhancedSTTService()
            tts_service = EnhancedTTSService()
            whatsapp_bot = EnhancedWhatsAppBot(rag_pipeline, stt_service, tts_service)
            logger.info("📱 WhatsApp bot initialized")
        except ImportError as e:
            logger.warning(f"⚠️ WhatsApp bot dependencies missing: {e}")
            whatsapp_bot = None
    
    # Create Gradio interface
    gradio_app = admin_ui.create_gradio_interface()
    
    # Create FastAPI app
    app = FastAPI(title="Simple RAG Server with WhatsApp Bot")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Gemini Live service globals
    gemini_service = None
    gemini_session_manager = None
    gemini_live_enabled = False

    @app.on_event("startup")
    async def startup_gemini_live():
        """Initialize Gemini Live service on startup."""
        nonlocal gemini_service, gemini_session_manager, gemini_live_enabled

        if not GEMINI_LIVE_AVAILABLE:
            logger.info("Gemini Live module not available - voice WebSocket disabled")
            return

        try:
            gemini_config = get_gemini_config()

            if not gemini_config.enabled:
                logger.info("Gemini Live disabled in config - voice WebSocket disabled")
                return

            # Initialize Gemini Live service with RAG pipeline
            gemini_service = GeminiLiveService(rag_pipeline=rag_pipeline)

            if not gemini_service.is_available():
                logger.warning(
                    "Gemini Live service not available (check credentials) - "
                    "voice WebSocket disabled"
                )
                return

            # Initialize session manager
            gemini_session_manager = SessionManager(gemini_service)
            await gemini_session_manager.start()

            gemini_live_enabled = True
            logger.info(
                "Gemini Live initialized - WebSocket endpoint /ws/voice available"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Gemini Live: {e}")
            gemini_live_enabled = False

    @app.on_event("shutdown")
    async def shutdown_gemini_live():
        """Cleanup Gemini Live service on shutdown."""
        nonlocal gemini_session_manager

        if gemini_session_manager:
            try:
                await gemini_session_manager.stop()
                logger.info("Gemini Live session manager stopped")
            except Exception as e:
                logger.error(f"Error stopping Gemini session manager: {e}")

    async def stream_gemini_responses(websocket: WebSocket, session: "GeminiLiveSession", user_id: str):
        """
        Stream audio responses from Gemini Live back to the WebSocket client.

        Args:
            websocket: The WebSocket connection
            session: The active Gemini Live session
            user_id: User identifier for logging
        """
        try:
            logger.info(f"Starting response stream for user {user_id}")

            async for response in session.receive_audio():
                # Check if it's a special marker
                if isinstance(response, bytes):
                    if response == b"__TURN_COMPLETE__":
                        # Turn complete - send transcriptions
                        await websocket.send_json({"type": "turn_complete"})

                        # Send assistant response transcription
                        response_text = session.get_response_transcription(clear=True)
                        if response_text:
                            await websocket.send_json({
                                "type": "transcription",
                                "role": "assistant",
                                "text": response_text
                            })

                        # Send user input transcription
                        user_text = session.get_transcription(clear=True)
                        if user_text:
                            await websocket.send_json({
                                "type": "transcription",
                                "role": "user",
                                "text": user_text
                            })

                    elif response == b"__INTERRUPTED__":
                        # User interrupted - notify client
                        await websocket.send_json({"type": "interrupted"})

                    else:
                        # Regular audio chunk - send as binary
                        await websocket.send_bytes(response)

                elif isinstance(response, dict):
                    # Handle dict responses (transcriptions, etc.)
                    if "text" in response:
                        await websocket.send_json({
                            "type": "transcription",
                            "role": response.get("role", "assistant"),
                            "text": response["text"]
                        })

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected during response stream for {user_id}")
        except Exception as e:
            logger.error(f"Error in response stream for {user_id}: {e}")
            try:
                await websocket.send_json({
                    "type": "error",
                    "error": str(e)
                })
            except:
                pass

    @app.websocket("/ws/voice")
    async def voice_websocket(websocket: WebSocket):
        """
        WebSocket endpoint for Gemini Live voice conversations.

        Protocol:
        - Client sends JSON config message first: {"type": "config", "language": "en-IN"}
        - Client sends {"type": "start_audio"} to begin recording
        - Client sends binary audio chunks (Int16 PCM, 16kHz, mono)
        - Client sends {"type": "stop_audio"} to end recording
        - Server sends binary audio responses (Int16 PCM, 24kHz, mono)
        - Server sends JSON transcription/status messages
        - Client can send {"type": "set_language", "language": "hi-IN"} to switch language
        """
        await websocket.accept()

        # Check if Gemini Live is available
        if not gemini_live_enabled:
            await websocket.send_json({
                "type": "error",
                "error": "Voice conversations not available. Gemini Live is disabled."
            })
            await websocket.close(code=1008, reason="Service unavailable")
            return

        session = None
        language = "en-IN"
        user_id = f"web_{datetime.now().timestamp()}_{id(websocket)}"
        response_task = None

        logger.info(f"Voice WebSocket connected: {user_id}")

        try:
            while True:
                data = await websocket.receive()

                # Handle text messages (JSON commands)
                if "text" in data:
                    try:
                        message = json.loads(data["text"])
                        msg_type = message.get("type", "")

                        if msg_type == "config":
                            # Initial configuration
                            language = message.get("language", "en-IN")
                            user_id = message.get("user_id", user_id)
                            logger.info(f"Config received: user={user_id}, language={language}")

                            await websocket.send_json({
                                "type": "config_ack",
                                "user_id": user_id,
                                "language": language
                            })

                        elif msg_type == "start_audio":
                            # Create or get session for this user
                            logger.info(f"Starting audio session for {user_id}")

                            try:
                                session = await gemini_session_manager.get_or_create_session(
                                    user_id=user_id,
                                    language=language,
                                    voice="Aoede"  # Warm, empathetic voice for healthcare
                                )

                                await websocket.send_json({
                                    "type": "session_created",
                                    "session_id": session.session_id if hasattr(session, 'session_id') else user_id
                                })

                                # Start background task to stream responses
                                if response_task is None or response_task.done():
                                    response_task = asyncio.create_task(
                                        stream_gemini_responses(websocket, session, user_id)
                                    )

                            except Exception as e:
                                logger.error(f"Failed to create session for {user_id}: {e}")
                                await websocket.send_json({
                                    "type": "error",
                                    "error": f"Failed to create voice session: {str(e)}"
                                })

                        elif msg_type == "stop_audio":
                            # Audio stream ended
                            logger.info(f"Audio stream stopped for {user_id}")
                            await websocket.send_json({"type": "audio_stopped"})

                        elif msg_type == "set_language":
                            # Switch language - requires new session
                            new_language = message.get("language", language)
                            if new_language != language:
                                logger.info(f"Switching language from {language} to {new_language} for {user_id}")
                                language = new_language

                                # Close existing session
                                if session:
                                    await gemini_session_manager.close_session(user_id)
                                    session = None

                                await websocket.send_json({
                                    "type": "language_changed",
                                    "language": language
                                })

                        elif msg_type == "text":
                            # Text message (for hybrid mode)
                            text = message.get("text", "")
                            if session and text:
                                await session.send_text(text)

                        elif msg_type == "ping":
                            # Keep-alive ping
                            await websocket.send_json({"type": "pong"})

                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON from {user_id}: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "error": "Invalid JSON message"
                        })

                # Handle binary messages (audio data)
                elif "bytes" in data:
                    if session:
                        try:
                            await session.send_audio(data["bytes"])
                        except Exception as e:
                            logger.error(f"Error sending audio for {user_id}: {e}")

        except WebSocketDisconnect:
            logger.info(f"Voice WebSocket disconnected: {user_id}")

        except Exception as e:
            logger.error(f"Voice WebSocket error for {user_id}: {e}")
            try:
                await websocket.send_json({
                    "type": "error",
                    "error": str(e)
                })
            except:
                pass

        finally:
            # Cleanup
            if response_task and not response_task.done():
                response_task.cancel()
                try:
                    await response_task
                except asyncio.CancelledError:
                    pass

            if session and gemini_session_manager:
                try:
                    await gemini_session_manager.close_session(user_id)
                    logger.info(f"Session closed for {user_id}")
                except Exception as e:
                    logger.error(f"Error closing session for {user_id}: {e}")

    @app.get("/")
    async def root():
        return {"message": "Simple RAG Server - No Database Required!"}
    
    @app.get("/health")
    async def health():
        stats = rag_pipeline.get_index_stats()

        # Get Gemini Live status
        gemini_status = "disabled"
        if gemini_live_enabled and gemini_session_manager:
            gemini_status = gemini_session_manager.get_status()
        elif GEMINI_LIVE_AVAILABLE:
            gemini_status = "available but not enabled"

        return {
            "status": "healthy",
            "database": "file-based (no SQL database)",
            "whatsapp_bot": "configured" if whatsapp_bot else "not configured",
            "gemini_live": gemini_status,
            "ngrok_url": ngrok_manager.ngrok_url,
            "stats": stats.get("stats", {})
        }
    
    @app.post("/api/query")
    async def api_query(query: str = Form(...)):
        """Direct API endpoint for queries"""
        try:
            result = await rag_pipeline.query(query)
            return result
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    @app.get("/api/debug")
    async def api_debug():
        """Debug endpoint to check vector database state"""
        try:
            debug_info = {
                "metadata_docs": len(rag_pipeline.document_metadata),
                "document_ids": list(rag_pipeline.document_metadata.keys()),
                "vector_db_count": 0,
                "sample_chunks": []
            }
            
            # Try to get vector DB count
            try:
                debug_info["vector_db_count"] = rag_pipeline.vector_db.count()
            except Exception as e:
                debug_info["vector_db_error"] = str(e)
            
            # Try to get a few sample chunks
            try:
                if debug_info["document_ids"]:
                    first_doc_id = debug_info["document_ids"][0]
                    chunk_ids = [f"{first_doc_id}_chunk_0", f"{first_doc_id}_chunk_1"]
                    sample_result = rag_pipeline.vector_db.get(ids=chunk_ids[:1])
                    if sample_result and sample_result.get('documents'):
                        debug_info["sample_chunks"] = sample_result['documents'][:2]
            except Exception as e:
                debug_info["sample_error"] = str(e)
            
            return debug_info
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    @app.post("/api/upload")
    async def api_upload(file: UploadFile = File(...), metadata: str = Form("{}")):
        """API endpoint for file uploads"""
        try:
            # Save uploaded file
            upload_dir = Path("uploads")
            upload_dir.mkdir(exist_ok=True)
            
            file_path = upload_dir / file.filename
            with open(file_path, "wb") as f:
                f.write(await file.read())
            
            # Parse metadata
            metadata_dict = json.loads(metadata) if metadata else {}
            
            # Index file
            result = await rag_pipeline.add_documents([str(file_path)], metadata_dict)
            
            return result
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    # Add WhatsApp webhook routes if bot is configured
    if whatsapp_bot:
        from fastapi.responses import Response
        
        @app.post("/webhook")
        async def whatsapp_webhook(
            From: str = Form(...),
            To: str = Form(...),
            Body: str = Form(None),
            MediaUrl0: str = Form(None),
            MediaContentType0: str = Form(None),
            NumMedia: str = Form("0")
        ):
            """Handle Twilio WhatsApp webhook"""
            try:
                from twilio.twiml.messaging_response import MessagingResponse
                
                # Detailed logging of webhook data
                logger.info("🔔 WEBHOOK RECEIVED:")
                logger.info(f"  📱 From: {From}")
                logger.info(f"  📍 To: {To}")
                logger.info(f"  💬 Body: '{Body}' (length: {len(Body) if Body else 0})")
                logger.info(f"  🎵 MediaUrl0: {MediaUrl0}")
                logger.info(f"  📋 MediaContentType0: {MediaContentType0}")
                logger.info(f"  🔢 NumMedia: {NumMedia}")
                logger.info(f"  🎯 Has Media: {bool(MediaUrl0)}")
                logger.info(f"  🎯 Is Audio: {MediaContentType0 and MediaContentType0.startswith('audio/') if MediaContentType0 else False}")
                
                # Process the message
                await whatsapp_bot._process_twilio_message(From, To, Body, MediaUrl0, MediaContentType0, NumMedia)
                
                # Return empty TwiML response
                resp = MessagingResponse()
                return Response(content=str(resp), media_type="application/xml")
                
            except Exception as e:
                logger.error(f"❌ Webhook error: {e}", exc_info=True)
                from twilio.twiml.messaging_response import MessagingResponse
                resp = MessagingResponse()
                resp.message("Sorry, I'm experiencing technical difficulties.")
                return Response(content=str(resp), media_type="application/xml")
        
        @app.get("/media/{filename}")
        async def serve_media(filename: str):
            """Serve audio files for WhatsApp"""
            try:
                if hasattr(whatsapp_bot, 'media_files') and filename in whatsapp_bot.media_files:
                    file_path = whatsapp_bot.media_files[filename]
                    if Path(file_path).exists():
                        with open(file_path, 'rb') as f:
                            content = f.read()
                        
                        # Determine content type
                        if filename.endswith('.mp3'):
                            media_type = "audio/mpeg"
                        elif filename.endswith('.ogg'):
                            media_type = "audio/ogg"
                        elif filename.endswith('.wav'):
                            media_type = "audio/wav"
                        else:
                            media_type = "audio/mpeg"
                        
                        return Response(content=content, media_type=media_type)
                
                raise HTTPException(status_code=404, detail="Media not found")
                
            except Exception as e:
                logger.error(f"Media serve error: {e}")
                raise HTTPException(status_code=500, detail="Media error")
    
    # Mount Gradio app
    app = gr.mount_gradio_app(app, gradio_app, path="/admin")

    # Mount web client static files for Gemini Live voice interface
    web_client_path = Path("web_client")
    if web_client_path.exists():
        app.mount("/voice", StaticFiles(directory="web_client", html=True), name="voice")
        logger.info("🎤 Voice interface available at /voice")

    # Start ngrok if not disabled and WhatsApp is configured
    ngrok_url = None
    if not args.no_ngrok and whatsapp_bot:
        ngrok_url = ngrok_manager.start_ngrok(args.port)
        if ngrok_url:
            # Update environment variables for WhatsApp bot
            os.environ["PUBLIC_BASE_URL"] = ngrok_url
            os.environ["NGROK_URL"] = ngrok_url
            logger.info(f"🌐 Environment variables updated with ngrok URL: {ngrok_url}")
    
    # Setup signal handlers for cleanup
    def signal_handler(signum, frame):
        logger.info("🛑 Received shutdown signal. Cleaning up...")
        ngrok_manager.stop_ngrok()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Print startup information
    print("🚀 Starting Simple RAG Server...")
    print("📊 Admin UI: http://localhost:{}/admin".format(args.port))
    print("🔗 API Docs: http://localhost:{}/docs".format(args.port))
    print("💚 Health Check: http://localhost:{}/health".format(args.port))
    if web_client_path.exists():
        print("🎤 Voice UI: http://localhost:{}/voice".format(args.port))
    if GEMINI_LIVE_AVAILABLE:
        print("🔊 Voice WebSocket: ws://localhost:{}/ws/voice".format(args.port))
    print("🗄️ Storage: File-based (no database required)")
    
    if whatsapp_bot:
        if ngrok_url:
            print(f"🌐 Public URL: {ngrok_url}")
            print(f"📱 WhatsApp Webhook: {ngrok_url}/webhook")
            print("=" * 80)
            print("🔧 REQUIRED: Set up Twilio WhatsApp Webhook")
            print("=" * 80)
            print("1. Go to: https://console.twilio.com/us1/develop/sms/try-it-out/whatsapp-learn")
            print("2. In 'Sandbox Configuration' section:")
            print(f"3. Set 'When a message comes in' to: {ngrok_url}/webhook")
            print("4. Set HTTP method to: POST")
            print("5. Click 'Save Configuration'")
            print("")
            print("📱 Then send messages to: whatsapp:+14155238886")
            print("💡 First send: 'join [your-sandbox-code]' to join sandbox")
            print("=" * 80)
        else:
            print("📱 WhatsApp: Configured but no external URL (webhook will be local only)")
            print("   Install ngrok for external access: brew install ngrok")
    else:
        if not twilio_configured:
            print("📱 WhatsApp: Not configured (add TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN to .env)")
        else:
            print("📱 WhatsApp: Configuration error")
    
    print("=" * 80)
    
    # Start server
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info"
        )
    finally:
        ngrok_manager.stop_ngrok()


if __name__ == "__main__":
    main()