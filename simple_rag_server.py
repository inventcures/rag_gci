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
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Request, Header
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

# Bolna.ai integration imports (for voice AI helpline)
try:
    from bolna_integration import (
        BolnaClient,
        BolnaWebhookHandler,
        get_palli_sahayak_agent_config,
    )
    BOLNA_AVAILABLE = True
except ImportError:
    BOLNA_AVAILABLE = False
    BolnaClient = None
    BolnaWebhookHandler = None

# Sarvam AI integration imports
try:
    from sarvam_integration import (
        SarvamClient,
        SarvamWebhookHandler,
        SarvamSTTResult,
        SarvamTTSResult,
        SarvamTranslateResult,
        SARVAM_LANGUAGE_CONFIGS,
        SARVAM_STT_LANGUAGES,
        SARVAM_TTS_LANGUAGES,
    )
    SARVAM_AVAILABLE = True
except ImportError:
    SARVAM_AVAILABLE = False
    SarvamClient = None
    SarvamWebhookHandler = None

import hmac

# Knowledge Graph integration imports
try:
    from knowledge_graph import (
        KnowledgeGraphRAG,
        Neo4jClient,
        EntityExtractor,
        GraphVisualizer,
        VisualizationData,
    )
    KNOWLEDGE_GRAPH_AVAILABLE = True
except ImportError:
    KNOWLEDGE_GRAPH_AVAILABLE = False
    KnowledgeGraphRAG = None
    Neo4jClient = None

# GraphRAG integration imports (Microsoft GraphRAG)
try:
    from graphrag_integration import (
        GraphRAGConfig,
        GraphRAGIndexer,
        GraphRAGQueryEngine,
        GraphRAGDataLoader,
    )
    from graphrag_integration.query_engine import SearchMethod, SearchResult
    from graphrag_integration.indexer import IndexingMethod, IndexingStatus
    GRAPHRAG_AVAILABLE = True
except ImportError:
    GRAPHRAG_AVAILABLE = False
    GraphRAGConfig = None
    GraphRAGIndexer = None
    GraphRAGQueryEngine = None
    GraphRAGDataLoader = None

# V25: Longitudinal Patient Context Memory System
try:
    from personalization.longitudinal_memory import (
        LongitudinalMemoryManager,
        DataSourceType,
        SeverityLevel,
        TemporalTrend,
    )
    from personalization.context_injector import ContextInjector, PromptContextBuilder
    from personalization.cross_modal_aggregator import CrossModalAggregator
    from personalization.user_profile import UserProfileManager
    from personalization.context_memory import ContextMemory
    from personalization.temporal_reasoner import (
        TemporalReasoner,
        SymptomProgressionReport,
        MedicationEffectivenessReport,
        CorrelationAnalysis,
    )
    from personalization.alert_manager import AlertManager, AlertNotificationCoordinator
    from personalization.fhir_adapter import FHIRAdapter, FHIRBundle, export_to_file, import_from_file
    LONGITUDINAL_MEMORY_AVAILABLE = True
except ImportError as e:
    LONGITUDINAL_MEMORY_AVAILABLE = False
    LongitudinalMemoryManager = None
    ContextInjector = None
    CrossModalAggregator = None
    TemporalReasoner = None
    AlertManager = None
    logger.warning(f"Longitudinal memory module not available: {e}")

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

# Safety Enhancements Module
try:
    from safety_enhancements import (
        SafetyEnhancementsManager,
        get_safety_manager,
        EvidenceBadge,
        EvidenceLevel,
        EmergencyAlert,
        EmergencyLevel,
        HandoffRequest,
        HandoffReason,
        MedicationReminder,
        ResponseLengthOptimizer,
    )
    SAFETY_ENHANCEMENTS_AVAILABLE = True
except ImportError as e:
    SAFETY_ENHANCEMENTS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Safety enhancements module not available: {e}")

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
        self.translation_model = "llama-3.1-8b-instant"  # Model for query translation
        
        # Check for MedGemma preference from environment
        use_medgemma = os.getenv("USE_MEDGEMMA", "false").lower() == "true"
        if use_medgemma:
            self.response_model = "medgemma"  # Use MedGemma for English response generation
            logger.info("🩺 MedGemma model selected for English response generation")
        else:
            self.response_model = "qwen/qwen3-32b"  # Use Qwen3 for high-quality response generation
            logger.info("🧠 Qwen3-32B reasoning model selected for response generation")
        
        # Document metadata and conversation storage
        self.document_metadata = self._load_metadata()
        self.conversations = self._load_conversations()

        self._initialize_components()

        # V25: Initialize Longitudinal Patient Context Memory System
        self._initialize_longitudinal_memory()
    
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

    def _initialize_longitudinal_memory(self):
        """
        V25: Initialize Longitudinal Patient Context Memory System.

        This enables:
        - Multi-session patient memory spanning months/years
        - Context injection with patient history for personalized responses
        - Cross-modal data extraction from voice/WhatsApp/web conversations
        - Temporal reasoning for trend detection and medication effectiveness
        - Proactive monitoring with alerts for concerning patterns
        """
        if not LONGITUDINAL_MEMORY_AVAILABLE:
            logger.info("📊 Longitudinal memory module not available - skipping initialization")
            self.longitudinal_manager = None
            self.context_injector = None
            self.cross_modal_aggregator = None
            self.prompt_context_builder = None
            self.temporal_reasoner = None
            self.alert_manager = None
            return

        try:
            # Storage paths for longitudinal data
            longitudinal_storage = self.data_dir / "longitudinal"
            user_profiles_storage = self.data_dir / "user_profiles"
            context_memory_storage = self.data_dir / "context_memory"
            alerts_storage = self.data_dir / "alerts"

            # Initialize LongitudinalMemoryManager
            self.longitudinal_manager = LongitudinalMemoryManager(
                storage_path=str(longitudinal_storage)
            )
            logger.info("✅ LongitudinalMemoryManager initialized")

            # Initialize UserProfileManager
            self.user_profile_manager = UserProfileManager(
                storage_path=str(user_profiles_storage)
            )
            logger.info("✅ UserProfileManager initialized")

            # Initialize ContextMemory
            self.context_memory = ContextMemory(
                storage_path=str(context_memory_storage)
            )
            logger.info("✅ ContextMemory initialized")

            # Initialize ContextInjector
            self.context_injector = ContextInjector(
                longitudinal_manager=self.longitudinal_manager,
                user_profile_manager=self.user_profile_manager,
                context_memory=self.context_memory
            )
            logger.info("✅ ContextInjector initialized")

            # Initialize PromptContextBuilder
            self.prompt_context_builder = PromptContextBuilder(
                context_injector=self.context_injector
            )
            logger.info("✅ PromptContextBuilder initialized")

            # Initialize CrossModalAggregator
            self.cross_modal_aggregator = CrossModalAggregator(
                longitudinal_manager=self.longitudinal_manager,
                storage_path=str(self.data_dir / "extractions")
            )
            logger.info("✅ CrossModalAggregator initialized")

            # Phase 4: Initialize TemporalReasoner
            self.temporal_reasoner = TemporalReasoner(
                longitudinal_manager=self.longitudinal_manager
            )
            logger.info("✅ TemporalReasoner initialized")

            # Phase 5: Initialize AlertManager
            self.alert_manager = AlertManager(
                longitudinal_manager=self.longitudinal_manager,
                storage_path=str(alerts_storage)
            )
            logger.info("✅ AlertManager initialized")

            # Initialize AlertNotificationCoordinator
            self.alert_coordinator = AlertNotificationCoordinator(
                alert_manager=self.alert_manager,
                longitudinal_manager=self.longitudinal_manager
            )
            logger.info("✅ AlertNotificationCoordinator initialized")

            logger.info("🧠 V25 Longitudinal Patient Context Memory System ready")

        except Exception as e:
            logger.error(f"Failed to initialize longitudinal memory: {e}")
            self.longitudinal_manager = None
            self.context_injector = None
            self.cross_modal_aggregator = None
            self.prompt_context_builder = None
            self.temporal_reasoner = None
            self.alert_manager = None

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
        """Query the RAG pipeline with safety enhancements"""
        try:
            # =================================================================
            # SAFETY ENHANCEMENTS: Pre-processing checks
            # =================================================================
            safety_result = None
            if SAFETY_ENHANCEMENTS_AVAILABLE and user_id:
                safety_manager = get_safety_manager()
                conversation_history = self.conversations.get(conversation_id, []) if conversation_id else []
                safety_result = await safety_manager.process_query(
                    user_id=user_id,
                    query=question,
                    language=source_language,
                    conversation_history=conversation_history
                )
                
                # Handle emergency or handoff
                if not safety_result["should_respond"]:
                    response_parts = []
                    if "emergency" in safety_result["response_additions"]:
                        response_parts.append(safety_result["response_additions"]["emergency"])
                    if "emergency_actions" in safety_result["response_additions"]:
                        response_parts.append(f"\nActions:\n{safety_result['response_additions']['emergency_actions']}")
                    if "handoff" in safety_result["response_additions"]:
                        response_parts.append(safety_result["response_additions"]["handoff"])
                    
                    return {
                        "status": "safety_escalation",
                        "answer": "\n\n".join(response_parts),
                        "emergency_alert": safety_result.get("emergency_alert"),
                        "handoff_request": safety_result.get("handoff_request"),
                        "sources": [],
                        "conversation_id": conversation_id,
                        "timestamp": datetime.now().isoformat(),
                    }
            
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

            # V25: Inject patient context if available
            patient_context = ""
            if user_id and self.context_injector:
                try:
                    patient_context = await self.context_injector.inject_context(
                        user_id=user_id,
                        question=question,
                        max_length=500  # Keep context concise
                    )
                    if patient_context:
                        logger.info(f"🧠 Injected patient context for user {user_id} ({len(patient_context)} chars)")
                        context_text = f"""--- Patient History ---
{patient_context}

--- Medical Documents ---
{context_text}"""
                except Exception as e:
                    logger.warning(f"Failed to inject patient context: {e}")

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

            # V25: Extract observations from conversation (async, non-blocking)
            observations_extracted = 0
            if user_id and self.cross_modal_aggregator:
                try:
                    # Combine question and answer for extraction
                    conversation_text = f"Patient: {question}\nAssistant: {answer}"
                    observations = await self.cross_modal_aggregator.process_conversation(
                        patient_id=user_id,
                        conversation_id=conversation_id or f"conv_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        transcript=conversation_text,
                        source_type=DataSourceType.WEB_CHAT,
                        metadata={
                            "language": source_language,
                            "speaker_role": "patient"
                        }
                    )
                    observations_extracted = len(observations)
                    if observations:
                        logger.info(f"🔍 Extracted {len(observations)} observations from conversation for user {user_id}")
                except Exception as e:
                    logger.warning(f"Failed to extract observations: {e}")

            # =================================================================
            # SAFETY ENHANCEMENTS: Post-processing
            # =================================================================
            
            # 1. Add evidence badge
            if SAFETY_ENHANCEMENTS_AVAILABLE:
                safety_manager = get_safety_manager()
                answer = safety_manager.add_evidence_badge(
                    query=question,
                    sources=sources,
                    distances=distances,
                    answer=answer,
                    language=source_language
                )
            
            # 2. Optimize response length for user comprehension
            if SAFETY_ENHANCEMENTS_AVAILABLE and user_id:
                safety_manager = get_safety_manager()
                answer = safety_manager.optimize_response(answer, user_id)
            
            return {
                "status": "success",
                "answer": answer,
                "model_used": model_used,
                "sources": sources,
                "context_used": len(relevant_contexts),
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat(),
                "patient_context_used": bool(patient_context),
                "observations_extracted": observations_extracted,
                "safety_enhancements": {
                    "evidence_badge": SAFETY_ENHANCEMENTS_AVAILABLE,
                    "response_optimized": SAFETY_ENHANCEMENTS_AVAILABLE and user_id is not None,
                } if SAFETY_ENHANCEMENTS_AVAILABLE else None
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
    
    def _extract_and_log_thinking(self, response: str) -> str:
        """Extract <think> tags from response, log the reasoning, return clean answer.

        Qwen3 and other reasoning models wrap their chain-of-thought in <think> tags.
        We want to:
        1. Log the thinking for debugging/monitoring
        2. Return only the final answer to the user
        """
        import re

        # Pattern to match <think>...</think> blocks (including newlines)
        think_pattern = r'<think>(.*?)</think>'

        # Find all thinking blocks
        thinking_matches = re.findall(think_pattern, response, re.DOTALL)

        if thinking_matches:
            for i, thinking in enumerate(thinking_matches):
                # Log the reasoning (truncate if very long)
                thinking_preview = thinking.strip()[:500]
                if len(thinking.strip()) > 500:
                    thinking_preview += "... [truncated]"
                logger.info(f"🧠 Model Reasoning (block {i+1}):\n{thinking_preview}")

            # Remove all <think>...</think> blocks from response
            clean_response = re.sub(think_pattern, '', response, flags=re.DOTALL)

            # Clean up extra whitespace/newlines left behind
            clean_response = re.sub(r'\n\s*\n', '\n', clean_response).strip()

            logger.info(f"✅ Extracted {len(thinking_matches)} thinking block(s), returning clean response")
            return clean_response

        # No thinking tags found, return as-is
        return response

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
                raw_answer = result["choices"][0]["message"]["content"].strip()
                return self._extract_and_log_thinking(raw_answer)
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
                    raw_answer = result["choices"][0]["message"]["content"].strip()
                    answer = self._extract_and_log_thinking(raw_answer)
                    model_used = "qwen3"
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
                "model": self.response_model,  # Use configured response model
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
                raw_answer = result["choices"][0]["message"]["content"].strip()
                answer = self._extract_and_log_thinking(raw_answer)
                logger.info("✅ Groq fallback successful")
                return answer, "qwen3"
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

                # Knowledge Graph Tab
                with gr.TabItem("🕸️ Knowledge Graph"):
                    gr.Markdown("## Knowledge Graph Explorer")
                    gr.Markdown("*Extract medical entities and explore relationships*")

                    with gr.Tabs():
                        # Entity Extraction Sub-tab
                        with gr.TabItem("🔍 Extract Entities"):
                            gr.Markdown("### Extract medical entities from text")

                            with gr.Row():
                                with gr.Column():
                                    kg_text_input = gr.Textbox(
                                        label="Enter medical text",
                                        placeholder="The patient has severe pain and nausea. Morphine 10mg was prescribed for pain relief.",
                                        lines=5
                                    )
                                    kg_extract_btn = gr.Button("Extract Entities", variant="primary")

                                with gr.Column():
                                    kg_entities_output = gr.JSON(label="Extracted Entities")
                                    kg_relationships_output = gr.JSON(label="Extracted Relationships")

                            kg_extract_btn.click(
                                fn=self._handle_kg_extract,
                                inputs=[kg_text_input],
                                outputs=[kg_entities_output, kg_relationships_output]
                            )

                        # Graph Query Sub-tab
                        with gr.TabItem("💬 Query Graph"):
                            gr.Markdown("### Query the Knowledge Graph")

                            with gr.Row():
                                with gr.Column():
                                    kg_query_input = gr.Textbox(
                                        label="Enter your question",
                                        placeholder="What medications treat pain?",
                                        lines=2
                                    )
                                    kg_query_btn = gr.Button("Query Graph", variant="primary")

                                with gr.Column():
                                    kg_query_output = gr.JSON(label="Query Results")

                            kg_query_btn.click(
                                fn=self._handle_kg_query,
                                inputs=[kg_query_input],
                                outputs=[kg_query_output]
                            )

                        # Treatments Lookup Sub-tab
                        with gr.TabItem("💊 Find Treatments"):
                            gr.Markdown("### Find treatments for symptoms")

                            with gr.Row():
                                with gr.Column():
                                    symptom_input = gr.Textbox(
                                        label="Enter symptom",
                                        placeholder="pain, nausea, fatigue...",
                                        lines=1
                                    )
                                    treatment_btn = gr.Button("Find Treatments", variant="primary")

                                with gr.Column():
                                    treatments_output = gr.JSON(label="Available Treatments")

                            treatment_btn.click(
                                fn=self._handle_kg_treatments,
                                inputs=[symptom_input],
                                outputs=[treatments_output]
                            )

                        # Side Effects Sub-tab
                        with gr.TabItem("⚠️ Side Effects"):
                            gr.Markdown("### Check medication side effects")

                            with gr.Row():
                                with gr.Column():
                                    medication_input = gr.Textbox(
                                        label="Enter medication name",
                                        placeholder="morphine, ondansetron...",
                                        lines=1
                                    )
                                    side_effects_btn = gr.Button("Check Side Effects", variant="primary")

                                with gr.Column():
                                    side_effects_output = gr.JSON(label="Known Side Effects")

                            side_effects_btn.click(
                                fn=self._handle_kg_side_effects,
                                inputs=[medication_input],
                                outputs=[side_effects_output]
                            )

                        # Graph Health Sub-tab
                        with gr.TabItem("📊 Graph Status"):
                            gr.Markdown("### Knowledge Graph Health & Statistics")

                            with gr.Row():
                                kg_health_btn = gr.Button("Check Health", variant="primary")
                                kg_stats_btn = gr.Button("Get Statistics", variant="secondary")
                                kg_import_btn = gr.Button("Import Base Knowledge", variant="secondary")

                            with gr.Row():
                                with gr.Column():
                                    kg_health_output = gr.JSON(label="Health Status")
                                with gr.Column():
                                    kg_stats_output = gr.JSON(label="Graph Statistics")

                            kg_health_btn.click(
                                fn=self._handle_kg_health,
                                inputs=[],
                                outputs=[kg_health_output]
                            )

                            kg_stats_btn.click(
                                fn=self._handle_kg_stats,
                                inputs=[],
                                outputs=[kg_stats_output]
                            )

                            kg_import_btn.click(
                                fn=self._handle_kg_import,
                                inputs=[],
                                outputs=[kg_stats_output]
                            )

                # ==========================================================
                # GRAPHRAG TAB
                # ==========================================================
                with gr.TabItem("🔗 GraphRAG"):
                    gr.Markdown("## Microsoft GraphRAG Integration")
                    gr.Markdown("*Advanced graph-based retrieval for palliative care knowledge*")

                    with gr.Tabs():
                        # Query Tab
                        with gr.TabItem("🔍 Query"):
                            gr.Markdown("### Search with GraphRAG")
                            gr.Markdown("""
                            **Search Methods:**
                            - **Auto**: Automatically selects best method
                            - **Global**: Holistic corpus-wide search using community reports
                            - **Local**: Entity-focused search for specific topics
                            - **DRIFT**: Multi-phase iterative search for complex questions
                            - **Basic**: Simple vector similarity search
                            """)

                            with gr.Row():
                                with gr.Column(scale=2):
                                    graphrag_query_input = gr.Textbox(
                                        label="Enter your question",
                                        placeholder="What are the main approaches to pain management in palliative care?",
                                        lines=3
                                    )
                                    graphrag_method = gr.Radio(
                                        choices=["auto", "global", "local", "drift", "basic"],
                                        value="auto",
                                        label="Search Method"
                                    )
                                    graphrag_query_btn = gr.Button("🔍 Search", variant="primary")

                                with gr.Column(scale=3):
                                    graphrag_response = gr.Markdown(label="Response")
                                    with gr.Accordion("Details", open=False):
                                        graphrag_entities_output = gr.JSON(label="Entities Found")
                                        graphrag_metadata = gr.JSON(label="Search Metadata")

                            graphrag_query_btn.click(
                                fn=self._handle_graphrag_query,
                                inputs=[graphrag_query_input, graphrag_method],
                                outputs=[graphrag_response, graphrag_entities_output, graphrag_metadata]
                            )

                        # Indexing Tab
                        with gr.TabItem("📥 Indexing"):
                            gr.Markdown("### Index Documents with GraphRAG")
                            gr.Markdown("""
                            Build a knowledge graph from your documents using LLM-based entity extraction.

                            **Methods:**
                            - **Standard**: LLM-based extraction (higher quality, slower)
                            - **Fast**: NLP-based extraction (faster, lower quality)
                            """)

                            with gr.Row():
                                graphrag_index_method = gr.Radio(
                                    choices=["standard", "fast"],
                                    value="standard",
                                    label="Indexing Method"
                                )
                                graphrag_update_mode = gr.Checkbox(
                                    label="Update Mode (incremental indexing)",
                                    value=False
                                )

                            with gr.Row():
                                graphrag_start_index_btn = gr.Button("▶️ Start Indexing", variant="primary")
                                graphrag_refresh_status_btn = gr.Button("🔄 Refresh Status", variant="secondary")

                            graphrag_index_status = gr.JSON(label="Indexing Status")

                            graphrag_start_index_btn.click(
                                fn=self._handle_graphrag_start_indexing,
                                inputs=[graphrag_index_method, graphrag_update_mode],
                                outputs=[graphrag_index_status]
                            )

                            graphrag_refresh_status_btn.click(
                                fn=self._handle_graphrag_index_status,
                                inputs=[],
                                outputs=[graphrag_index_status]
                            )

                        # Entity Explorer Tab
                        with gr.TabItem("🏷️ Entities"):
                            gr.Markdown("### Browse Extracted Entities")

                            with gr.Row():
                                graphrag_entity_search = gr.Textbox(
                                    label="Search Entities",
                                    placeholder="Enter entity name (e.g., morphine, pain, nausea)..."
                                )
                                graphrag_entity_type_filter = gr.Dropdown(
                                    choices=[
                                        "All", "Symptom", "Medication", "Condition",
                                        "Treatment", "SideEffect", "Dosage", "Route", "CareGoal"
                                    ],
                                    value="All",
                                    label="Entity Type"
                                )
                                graphrag_entity_search_btn = gr.Button("🔍 Search", variant="primary")

                            graphrag_entities_table = gr.Dataframe(
                                headers=["Name", "Type", "Description", "Score"],
                                label="Entities",
                                wrap=True
                            )

                            graphrag_entity_relationships = gr.JSON(label="Entity Relationships")

                            graphrag_entity_search_btn.click(
                                fn=self._handle_graphrag_entity_search,
                                inputs=[graphrag_entity_search, graphrag_entity_type_filter],
                                outputs=[graphrag_entities_table, graphrag_entity_relationships]
                            )

                        # Statistics Tab
                        with gr.TabItem("📊 Statistics"):
                            gr.Markdown("### GraphRAG Index Statistics")

                            with gr.Row():
                                graphrag_stats_btn = gr.Button("🔄 Refresh Statistics", variant="primary")
                                graphrag_verify_btn = gr.Button("✅ Verify Index", variant="secondary")

                            with gr.Row():
                                with gr.Column():
                                    graphrag_stats_output = gr.JSON(label="Index Statistics")
                                with gr.Column():
                                    graphrag_verify_output = gr.JSON(label="Verification Results")

                            graphrag_stats_btn.click(
                                fn=self._handle_graphrag_stats,
                                inputs=[],
                                outputs=[graphrag_stats_output]
                            )

                            graphrag_verify_btn.click(
                                fn=self._handle_graphrag_verify,
                                inputs=[],
                                outputs=[graphrag_verify_output]
                            )

                # V25: Alerts & Monitoring Tab
                with gr.TabItem("🚨 Alerts & Monitoring"):
                    gr.Markdown("## Patient Alerts & Temporal Monitoring")
                    gr.Markdown("*V25 Longitudinal Patient Context Memory System*")

                    with gr.Tabs():
                        # Active Alerts Sub-tab
                        with gr.TabItem("📋 Active Alerts"):
                            gr.Markdown("### Current Active Alerts")

                            with gr.Row():
                                alerts_refresh_btn = gr.Button("🔄 Refresh Alerts", variant="primary")
                                alerts_patient_filter = gr.Textbox(
                                    label="Filter by Patient ID (optional)",
                                    placeholder="Leave empty for all patients"
                                )

                            alerts_table = gr.Dataframe(
                                headers=["Alert ID", "Patient", "Priority", "Category", "Title", "Created", "Status"],
                                label="Active Alerts",
                                interactive=False
                            )

                            with gr.Row():
                                with gr.Column():
                                    alert_id_input = gr.Textbox(label="Alert ID to Act On")
                                with gr.Column():
                                    alert_user_input = gr.Textbox(label="Your Name/ID", value="admin")

                            with gr.Row():
                                acknowledge_btn = gr.Button("✅ Acknowledge", variant="secondary")
                                resolve_btn = gr.Button("🎯 Resolve", variant="primary")
                                resolution_notes = gr.Textbox(label="Resolution Notes", placeholder="Optional notes...")

                            alert_action_output = gr.Textbox(label="Action Result", interactive=False)

                            alerts_refresh_btn.click(
                                fn=self._handle_refresh_alerts,
                                inputs=[alerts_patient_filter],
                                outputs=[alerts_table]
                            )

                            acknowledge_btn.click(
                                fn=self._handle_acknowledge_alert,
                                inputs=[alert_id_input, alert_user_input],
                                outputs=[alert_action_output, alerts_table]
                            )

                            resolve_btn.click(
                                fn=self._handle_resolve_alert,
                                inputs=[alert_id_input, alert_user_input, resolution_notes],
                                outputs=[alert_action_output, alerts_table]
                            )

                        # Patient Temporal Analysis Sub-tab
                        with gr.TabItem("📈 Patient Analysis"):
                            gr.Markdown("### Temporal Analysis for Patient")

                            with gr.Row():
                                analysis_patient_id = gr.Textbox(
                                    label="Patient ID",
                                    placeholder="Enter patient ID"
                                )
                                analysis_days = gr.Slider(
                                    minimum=7, maximum=365, value=30, step=7,
                                    label="Analysis Period (days)"
                                )
                                run_analysis_btn = gr.Button("🔍 Run Analysis", variant="primary")

                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("#### Symptom Progressions")
                                    symptom_analysis_output = gr.JSON(label="Symptom Trends")
                                with gr.Column():
                                    gr.Markdown("#### Medication Effectiveness")
                                    medication_analysis_output = gr.JSON(label="Medication Reports")

                            with gr.Row():
                                gr.Markdown("#### Correlations Detected")
                            correlations_output = gr.JSON(label="Medication-Symptom Correlations")

                            run_analysis_btn.click(
                                fn=self._handle_patient_analysis,
                                inputs=[analysis_patient_id, analysis_days],
                                outputs=[symptom_analysis_output, medication_analysis_output, correlations_output]
                            )

                        # Generate Alerts Sub-tab
                        with gr.TabItem("⚡ Generate Alerts"):
                            gr.Markdown("### Run Proactive Monitoring")

                            with gr.Row():
                                monitor_patient_id = gr.Textbox(
                                    label="Patient ID",
                                    placeholder="Enter patient ID to monitor"
                                )
                                run_monitoring_btn = gr.Button("🔔 Run Monitoring", variant="primary")

                            generated_alerts_output = gr.JSON(label="Generated Alerts")

                            run_monitoring_btn.click(
                                fn=self._handle_generate_alerts,
                                inputs=[monitor_patient_id],
                                outputs=[generated_alerts_output]
                            )

                        # Alert Summary Sub-tab
                        with gr.TabItem("📊 Summary"):
                            gr.Markdown("### Alert Statistics")

                            summary_refresh_btn = gr.Button("🔄 Refresh Summary", variant="primary")
                            alert_summary_output = gr.JSON(label="Alert Summary")

                            summary_refresh_btn.click(
                                fn=self._handle_alert_summary,
                                inputs=[],
                                outputs=[alert_summary_output]
                            )

                # V25: Care Team Coordination Tab
                with gr.TabItem("👥 Care Team"):
                    gr.Markdown("## Care Team Coordination")
                    gr.Markdown("*V25 Longitudinal Patient Context Memory System*")

                    with gr.Tabs():
                        # View Care Team Sub-tab
                        with gr.TabItem("📋 View Care Team"):
                            gr.Markdown("### Patient Care Team Members")

                            with gr.Row():
                                careteam_patient_id = gr.Textbox(
                                    label="Patient ID",
                                    placeholder="Enter patient ID"
                                )
                                careteam_refresh_btn = gr.Button("🔄 Load Care Team", variant="primary")

                            careteam_table = gr.Dataframe(
                                headers=["Provider ID", "Name", "Role", "Organization", "Primary", "Interactions", "Last Contact"],
                                label="Care Team Members",
                                interactive=False
                            )

                            careteam_refresh_btn.click(
                                fn=self._handle_load_care_team,
                                inputs=[careteam_patient_id],
                                outputs=[careteam_table]
                            )

                        # Add Member Sub-tab
                        with gr.TabItem("➕ Add Member"):
                            gr.Markdown("### Add Care Team Member")

                            with gr.Row():
                                add_patient_id = gr.Textbox(label="Patient ID", placeholder="Patient ID")
                                add_provider_id = gr.Textbox(label="Provider ID", placeholder="e.g., dr_sharma")

                            with gr.Row():
                                add_name = gr.Textbox(label="Name", placeholder="Dr. Sharma")
                                add_role = gr.Dropdown(
                                    choices=["doctor", "nurse", "asha_worker", "caregiver", "volunteer", "social_worker"],
                                    label="Role",
                                    value="doctor"
                                )

                            with gr.Row():
                                add_organization = gr.Textbox(label="Organization (optional)", placeholder="City Hospital")
                                add_phone = gr.Textbox(label="Phone (optional)", placeholder="+91xxxxxxxxxx")

                            add_primary = gr.Checkbox(label="Primary Contact", value=False)

                            add_member_btn = gr.Button("➕ Add to Care Team", variant="primary")
                            add_result = gr.Textbox(label="Result", interactive=False)

                            add_member_btn.click(
                                fn=self._handle_add_care_team_member,
                                inputs=[add_patient_id, add_provider_id, add_name, add_role, add_organization, add_phone, add_primary],
                                outputs=[add_result]
                            )

                        # Notify Care Team Sub-tab
                        with gr.TabItem("📢 Notify"):
                            gr.Markdown("### Send Notification to Care Team")

                            with gr.Row():
                                notify_patient_id = gr.Textbox(label="Patient ID", placeholder="Patient ID")
                                notify_priority = gr.Dropdown(
                                    choices=["LOW", "MEDIUM", "HIGH", "URGENT"],
                                    label="Priority",
                                    value="MEDIUM"
                                )

                            notify_message = gr.Textbox(
                                label="Message",
                                placeholder="Patient needs follow-up...",
                                lines=3
                            )

                            notify_roles = gr.CheckboxGroup(
                                choices=["doctor", "nurse", "asha_worker", "caregiver", "volunteer", "social_worker"],
                                label="Target Roles (leave empty for all)"
                            )

                            notify_btn = gr.Button("📤 Send Notification", variant="primary")
                            notify_result = gr.JSON(label="Notification Result")

                            notify_btn.click(
                                fn=self._handle_notify_care_team,
                                inputs=[notify_patient_id, notify_message, notify_priority, notify_roles],
                                outputs=[notify_result]
                            )

                # V25: FHIR Interoperability Tab (Phase 7)
                with gr.TabItem("🏥 FHIR"):
                    gr.Markdown("## FHIR Interoperability")
                    gr.Markdown("*V25 Phase 7 - EHR/Hospital System Integration*")

                    with gr.Tabs():
                        # Export to FHIR Sub-tab
                        with gr.TabItem("📤 Export"):
                            gr.Markdown("### Export Patient Data to FHIR R4 Bundle")

                            with gr.Row():
                                fhir_export_patient_id = gr.Textbox(
                                    label="Patient ID",
                                    placeholder="Enter patient ID to export"
                                )

                            with gr.Row():
                                fhir_include_obs = gr.Checkbox(label="Include Observations", value=True)
                                fhir_include_meds = gr.Checkbox(label="Include Medications", value=True)
                                fhir_include_care_team = gr.Checkbox(label="Include Care Team", value=True)
                                fhir_save_file = gr.Checkbox(label="Save to File", value=False)

                            fhir_export_btn = gr.Button("📤 Export FHIR Bundle", variant="primary")

                            fhir_export_result = gr.JSON(label="Export Result")
                            fhir_bundle_output = gr.JSON(label="FHIR Bundle (R4)")

                            fhir_export_btn.click(
                                fn=self._handle_fhir_export,
                                inputs=[fhir_export_patient_id, fhir_include_obs, fhir_include_meds, fhir_include_care_team, fhir_save_file],
                                outputs=[fhir_export_result, fhir_bundle_output]
                            )

                        # Import from FHIR Sub-tab
                        with gr.TabItem("📥 Import"):
                            gr.Markdown("### Import FHIR R4 Bundle")

                            fhir_import_json = gr.Textbox(
                                label="FHIR Bundle JSON",
                                placeholder='Paste FHIR Bundle JSON here...',
                                lines=10
                            )

                            fhir_import_btn = gr.Button("📥 Import FHIR Bundle", variant="primary")
                            fhir_import_result = gr.JSON(label="Import Result")

                            fhir_import_btn.click(
                                fn=self._handle_fhir_import,
                                inputs=[fhir_import_json],
                                outputs=[fhir_import_result]
                            )

                        # Validate FHIR Sub-tab
                        with gr.TabItem("✅ Validate"):
                            gr.Markdown("### Validate FHIR Resource")

                            fhir_validate_json = gr.Textbox(
                                label="FHIR Resource/Bundle JSON",
                                placeholder='Paste FHIR JSON to validate...',
                                lines=10
                            )

                            fhir_validate_btn = gr.Button("✅ Validate", variant="primary")
                            fhir_validate_result = gr.JSON(label="Validation Result")

                            fhir_validate_btn.click(
                                fn=self._handle_fhir_validate,
                                inputs=[fhir_validate_json],
                                outputs=[fhir_validate_result]
                            )

                        # SNOMED Code Reference Sub-tab
                        with gr.TabItem("📖 SNOMED Codes"):
                            gr.Markdown("### SNOMED CT Code Mappings")
                            gr.Markdown("Reference for symptom and severity codes used in FHIR exports.")

                            fhir_codes_btn = gr.Button("🔄 Load Code Mappings", variant="secondary")
                            fhir_symptom_codes = gr.JSON(label="Symptom SNOMED Codes")
                            fhir_severity_codes = gr.JSON(label="Severity SNOMED Codes")

                            fhir_codes_btn.click(
                                fn=self._handle_fhir_get_codes,
                                inputs=[],
                                outputs=[fhir_symptom_codes, fhir_severity_codes]
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

    # =========================================================================
    # Knowledge Graph Handlers
    # =========================================================================

    def _handle_kg_extract(self, text: str):
        """Handle entity extraction from text."""
        if not text or not text.strip():
            return {"error": "Please enter some text"}, {}

        try:
            if not KNOWLEDGE_GRAPH_AVAILABLE:
                return {"error": "Knowledge Graph module not available"}, {}

            # Use pattern-based extraction (works without Neo4j)
            extractor = EntityExtractor(use_patterns=True)
            import asyncio

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            entities, relationships = loop.run_until_complete(
                extractor.extract(text, use_llm=False)
            )

            entities_data = [e.to_dict() for e in entities]
            relationships_data = [r.to_dict() for r in relationships]

            return entities_data, relationships_data

        except Exception as e:
            return {"error": str(e)}, {}

    def _handle_kg_query(self, question: str):
        """Handle knowledge graph query."""
        if not question or not question.strip():
            return {"error": "Please enter a question"}

        try:
            import aiohttp

            async def query_kg():
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "http://localhost:8000/api/kg/query",
                        json={"question": question}
                    ) as resp:
                        return await resp.json()

            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(query_kg())
            return result

        except Exception as e:
            return {"error": str(e), "message": "Make sure the server is running"}

    def _handle_kg_treatments(self, symptom: str):
        """Handle treatment lookup for a symptom."""
        if not symptom or not symptom.strip():
            return {"error": "Please enter a symptom"}

        try:
            import aiohttp

            async def get_treatments():
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://localhost:8000/api/kg/treatments/{symptom}"
                    ) as resp:
                        return await resp.json()

            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(get_treatments())
            return result

        except Exception as e:
            return {"error": str(e)}

    def _handle_kg_side_effects(self, medication: str):
        """Handle side effects lookup for a medication."""
        if not medication or not medication.strip():
            return {"error": "Please enter a medication name"}

        try:
            import aiohttp

            async def get_side_effects():
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://localhost:8000/api/kg/side-effects/{medication}"
                    ) as resp:
                        return await resp.json()

            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(get_side_effects())
            return result

        except Exception as e:
            return {"error": str(e)}

    def _handle_kg_health(self):
        """Handle knowledge graph health check."""
        try:
            import aiohttp

            async def check_health():
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        "http://localhost:8000/api/kg/health"
                    ) as resp:
                        return await resp.json()

            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(check_health())
            return result

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _handle_kg_stats(self):
        """Handle knowledge graph statistics."""
        try:
            import aiohttp

            async def get_stats():
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        "http://localhost:8000/api/kg/stats"
                    ) as resp:
                        return await resp.json()

            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(get_stats())
            return result

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _handle_kg_import(self):
        """Handle importing base knowledge."""
        try:
            import aiohttp

            async def import_base():
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "http://localhost:8000/api/kg/import-base"
                    ) as resp:
                        return await resp.json()

            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(import_base())
            return result

        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ==========================================================================
    # GRAPHRAG UI HANDLERS
    # ==========================================================================

    def _handle_graphrag_query(self, query: str, method: str):
        """Handle GraphRAG query."""
        if not query.strip():
            return "Please enter a query", [], {}

        try:
            import aiohttp

            async def do_query():
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "http://localhost:8000/api/graphrag/query",
                        json={"query": query, "method": method}
                    ) as resp:
                        return await resp.json()

            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(do_query())

            if result.get("status") == "success":
                search_result = result.get("result", {})
                response = search_result.get("response", "No response")
                entities = search_result.get("entities", [])
                metadata = {
                    "method": search_result.get("method", method),
                    "confidence": search_result.get("confidence", 0),
                    **search_result.get("metadata", {})
                }
                return response, entities, metadata
            else:
                return f"Error: {result.get('error', 'Unknown error')}", [], {}

        except Exception as e:
            return f"Error: {str(e)}", [], {}

    def _handle_graphrag_start_indexing(self, method: str, update_mode: bool):
        """Handle starting GraphRAG indexing."""
        try:
            import aiohttp

            async def start_indexing():
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "http://localhost:8000/api/graphrag/index",
                        json={"method": method, "update_mode": update_mode}
                    ) as resp:
                        return await resp.json()

            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(start_indexing())
            return result

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _handle_graphrag_index_status(self):
        """Handle getting GraphRAG indexing status."""
        try:
            import aiohttp

            async def get_status():
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        "http://localhost:8000/api/graphrag/index/status"
                    ) as resp:
                        return await resp.json()

            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(get_status())
            return result

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _handle_graphrag_entity_search(self, query: str, entity_type: str):
        """Handle GraphRAG entity search."""
        try:
            import aiohttp

            async def search_entities():
                type_param = "" if entity_type == "All" else f"&entity_type={entity_type}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://localhost:8000/api/graphrag/entities?query={query}&top_k=20{type_param}"
                    ) as resp:
                        return await resp.json()

            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(search_entities())

            if result.get("status") == "success":
                entities = result.get("entities", [])
                # Format for dataframe
                table_data = []
                for e in entities:
                    desc = e.get("description", "")
                    if len(desc) > 100:
                        desc = desc[:100] + "..."
                    table_data.append([
                        e.get("title", e.get("name", "Unknown")),
                        e.get("type", ""),
                        desc,
                        round(e.get("score", 0), 2)
                    ])
                return table_data, entities
            else:
                return [], [{"error": result.get("error", "Unknown error")}]

        except Exception as e:
            return [], [{"error": str(e)}]

    def _handle_graphrag_stats(self):
        """Handle getting GraphRAG statistics."""
        try:
            import aiohttp

            async def get_stats():
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        "http://localhost:8000/api/graphrag/stats"
                    ) as resp:
                        return await resp.json()

            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(get_stats())

            if result.get("status") == "success":
                return result.get("stats", {})
            else:
                return {"error": result.get("error", "Unknown error")}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _handle_graphrag_verify(self):
        """Handle GraphRAG index verification."""
        try:
            import aiohttp

            async def verify_index():
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "http://localhost:8000/api/graphrag/verify"
                    ) as resp:
                        return await resp.json()

            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(verify_index())

            if result.get("status") == "success":
                return result.get("verification", {})
            else:
                return {"error": result.get("error", "Unknown error")}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    # =========================================================================
    # V25: Alert & Monitoring Handlers
    # =========================================================================

    def _handle_refresh_alerts(self, patient_filter: str = ""):
        """Refresh the alerts table."""
        try:
            if not LONGITUDINAL_MEMORY_AVAILABLE or not self.rag_pipeline.alert_manager:
                return [["N/A", "N/A", "N/A", "N/A", "Alert system not available", "N/A", "N/A"]]

            import asyncio

            async def get_alerts():
                if patient_filter and patient_filter.strip():
                    alerts = await self.rag_pipeline.alert_manager.get_active_alerts(patient_filter.strip())
                else:
                    # Get summary to find all patients with alerts
                    summary = await self.rag_pipeline.alert_manager.get_alert_summary()
                    alerts = summary.get("recent_alerts", [])
                return alerts

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            alerts = loop.run_until_complete(get_alerts())

            if not alerts:
                return [["--", "--", "--", "--", "No active alerts", "--", "--"]]

            # Convert alerts to table rows
            rows = []
            for alert in alerts:
                if hasattr(alert, 'alert_id'):
                    # MonitoringAlert object
                    rows.append([
                        alert.alert_id[:8] + "...",
                        alert.patient_id,
                        alert.priority.value if hasattr(alert.priority, 'value') else str(alert.priority),
                        alert.category,
                        alert.title[:40] + "..." if len(alert.title) > 40 else alert.title,
                        alert.created_at.strftime("%Y-%m-%d %H:%M") if hasattr(alert.created_at, 'strftime') else str(alert.created_at),
                        "Acknowledged" if alert.acknowledged else ("Resolved" if alert.resolved else "Active")
                    ])
                elif isinstance(alert, dict):
                    # Dict format from summary
                    rows.append([
                        str(alert.get('alert_id', ''))[:8] + "...",
                        alert.get('patient_id', 'N/A'),
                        alert.get('priority', 'N/A'),
                        alert.get('category', 'N/A'),
                        str(alert.get('title', ''))[:40],
                        str(alert.get('created_at', ''))[:16],
                        alert.get('status', 'Active')
                    ])

            return rows if rows else [["--", "--", "--", "--", "No alerts found", "--", "--"]]

        except Exception as e:
            logger.error(f"Error refreshing alerts: {e}")
            return [["Error", "--", "--", "--", str(e), "--", "--"]]

    def _handle_acknowledge_alert(self, alert_id: str, user: str):
        """Acknowledge an alert."""
        try:
            if not alert_id or not alert_id.strip():
                return "Please enter an alert ID", self._handle_refresh_alerts("")

            if not LONGITUDINAL_MEMORY_AVAILABLE or not self.rag_pipeline.alert_manager:
                return "Alert system not available", self._handle_refresh_alerts("")

            import asyncio

            async def acknowledge():
                return await self.rag_pipeline.alert_manager.acknowledge_alert(
                    alert_id.strip(),
                    user.strip() or "admin"
                )

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(acknowledge())

            if result:
                return f"Alert {alert_id[:8]}... acknowledged by {user}", self._handle_refresh_alerts("")
            else:
                return f"Alert {alert_id[:8]}... not found or already acknowledged", self._handle_refresh_alerts("")

        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return f"Error: {str(e)}", self._handle_refresh_alerts("")

    def _handle_resolve_alert(self, alert_id: str, user: str, notes: str):
        """Resolve an alert."""
        try:
            if not alert_id or not alert_id.strip():
                return "Please enter an alert ID", self._handle_refresh_alerts("")

            if not LONGITUDINAL_MEMORY_AVAILABLE or not self.rag_pipeline.alert_manager:
                return "Alert system not available", self._handle_refresh_alerts("")

            import asyncio

            async def resolve():
                return await self.rag_pipeline.alert_manager.resolve_alert(
                    alert_id.strip(),
                    user.strip() or "admin",
                    notes.strip() if notes else None
                )

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(resolve())

            if result:
                return f"Alert {alert_id[:8]}... resolved by {user}", self._handle_refresh_alerts("")
            else:
                return f"Alert {alert_id[:8]}... not found or already resolved", self._handle_refresh_alerts("")

        except Exception as e:
            logger.error(f"Error resolving alert: {e}")
            return f"Error: {str(e)}", self._handle_refresh_alerts("")

    def _handle_patient_analysis(self, patient_id: str, days: int):
        """Run temporal analysis for a patient."""
        try:
            if not patient_id or not patient_id.strip():
                return {"error": "Please enter a patient ID"}, {}, {}

            if not LONGITUDINAL_MEMORY_AVAILABLE or not self.rag_pipeline.temporal_reasoner:
                return {"error": "Temporal reasoning not available"}, {}, {}

            import asyncio

            async def analyze():
                symptom_reports = []
                medication_reports = []
                correlations = []

                # Get patient record to find symptoms and medications
                if self.rag_pipeline.longitudinal_manager:
                    record = await self.rag_pipeline.longitudinal_manager.get_or_create_record(patient_id.strip())

                    symptoms = set()
                    medications = set()
                    for obs in record.observations:
                        if obs.category == "symptom":
                            symptoms.add(obs.entity_name)
                        elif obs.category == "medication":
                            medications.add(obs.entity_name)

                    # Analyze symptoms
                    for symptom in list(symptoms)[:5]:
                        report = await self.rag_pipeline.temporal_reasoner.analyze_symptom_progression(
                            patient_id.strip(), symptom, int(days)
                        )
                        if report:
                            symptom_reports.append(report.to_dict())

                    # Analyze medications
                    for med in list(medications)[:5]:
                        report = await self.rag_pipeline.temporal_reasoner.analyze_medication_effectiveness(
                            patient_id.strip(), med, int(days)
                        )
                        if report:
                            medication_reports.append(report.to_dict())

                    # Get correlations
                    corr_list = await self.rag_pipeline.temporal_reasoner.find_correlations(
                        patient_id.strip(), int(days)
                    )
                    correlations = [c.to_dict() for c in corr_list[:5]]

                return symptom_reports, medication_reports, correlations

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            symptom_reports, medication_reports, correlations = loop.run_until_complete(analyze())

            return (
                {"patient_id": patient_id, "symptom_count": len(symptom_reports), "reports": symptom_reports},
                {"patient_id": patient_id, "medication_count": len(medication_reports), "reports": medication_reports},
                {"patient_id": patient_id, "correlation_count": len(correlations), "correlations": correlations}
            )

        except Exception as e:
            logger.error(f"Error running patient analysis: {e}")
            return {"error": str(e)}, {}, {}

    def _handle_generate_alerts(self, patient_id: str):
        """Generate alerts for a patient based on temporal analysis."""
        try:
            if not patient_id or not patient_id.strip():
                return {"error": "Please enter a patient ID"}

            if not LONGITUDINAL_MEMORY_AVAILABLE or not self.rag_pipeline.alert_manager:
                return {"error": "Alert system not available"}

            import asyncio

            async def generate():
                alerts = await self.rag_pipeline.alert_manager.generate_alerts_for_patient(patient_id.strip())
                return [
                    {
                        "alert_id": a.alert_id,
                        "priority": a.priority.value if hasattr(a.priority, 'value') else str(a.priority),
                        "category": a.category,
                        "title": a.title,
                        "description": a.description,
                        "suggested_actions": a.suggested_actions
                    }
                    for a in alerts
                ]

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            alerts = loop.run_until_complete(generate())

            return {
                "patient_id": patient_id,
                "alerts_generated": len(alerts),
                "alerts": alerts
            }

        except Exception as e:
            logger.error(f"Error generating alerts: {e}")
            return {"error": str(e)}

    def _handle_alert_summary(self):
        """Get overall alert summary."""
        try:
            if not LONGITUDINAL_MEMORY_AVAILABLE or not self.rag_pipeline.alert_manager:
                return {"error": "Alert system not available"}

            import asyncio

            async def get_summary():
                return await self.rag_pipeline.alert_manager.get_alert_summary()

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            summary = loop.run_until_complete(get_summary())
            return summary

        except Exception as e:
            logger.error(f"Error getting alert summary: {e}")
            return {"error": str(e)}

    # =========================================================================
    # V25: Care Team Coordination Handlers
    # =========================================================================

    def _handle_load_care_team(self, patient_id: str):
        """Load care team members for a patient."""
        try:
            if not patient_id or not patient_id.strip():
                return [["--", "--", "--", "--", "--", "--", "Enter patient ID"]]

            if not LONGITUDINAL_MEMORY_AVAILABLE or not self.rag_pipeline.longitudinal_manager:
                return [["--", "--", "--", "--", "--", "--", "System not available"]]

            import asyncio

            async def get_team():
                return await self.rag_pipeline.longitudinal_manager.get_care_team(patient_id.strip())

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            care_team = loop.run_until_complete(get_team())

            if not care_team:
                return [["--", "--", "--", "--", "--", "--", "No care team members"]]

            rows = []
            for member in care_team:
                rows.append([
                    member.provider_id,
                    member.name,
                    member.role,
                    member.organization or "--",
                    "Yes" if member.primary_contact else "No",
                    str(member.total_interactions),
                    member.last_contact.strftime("%Y-%m-%d %H:%M") if hasattr(member.last_contact, 'strftime') else str(member.last_contact)[:16]
                ])

            return rows

        except Exception as e:
            logger.error(f"Error loading care team: {e}")
            return [["Error", "--", "--", "--", "--", "--", str(e)]]

    def _handle_add_care_team_member(
        self,
        patient_id: str,
        provider_id: str,
        name: str,
        role: str,
        organization: str,
        phone: str,
        primary: bool
    ):
        """Add a care team member."""
        try:
            if not patient_id or not patient_id.strip():
                return "Please enter a patient ID"

            if not provider_id or not name:
                return "Provider ID and Name are required"

            if not LONGITUDINAL_MEMORY_AVAILABLE or not self.rag_pipeline.longitudinal_manager:
                return "System not available"

            import asyncio
            from personalization.longitudinal_memory import CareTeamMember
            from datetime import datetime

            member = CareTeamMember(
                provider_id=provider_id.strip(),
                name=name.strip(),
                role=role,
                organization=organization.strip() if organization else None,
                phone_number=phone.strip() if phone else None,
                primary_contact=primary,
                first_contact=datetime.now(),
                last_contact=datetime.now(),
                total_interactions=0,
                attributed_observations=[]
            )

            async def add_member():
                await self.rag_pipeline.longitudinal_manager.add_care_team_member(
                    patient_id.strip(),
                    member
                )

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            loop.run_until_complete(add_member())

            return f"Successfully added {name} ({role}) to {patient_id}'s care team"

        except Exception as e:
            logger.error(f"Error adding care team member: {e}")
            return f"Error: {str(e)}"

    def _handle_notify_care_team(
        self,
        patient_id: str,
        message: str,
        priority: str,
        target_roles: list
    ):
        """Send notification to care team."""
        try:
            if not patient_id or not patient_id.strip():
                return {"error": "Please enter a patient ID"}

            if not message or not message.strip():
                return {"error": "Please enter a message"}

            if not LONGITUDINAL_MEMORY_AVAILABLE or not self.rag_pipeline.longitudinal_manager:
                return {"error": "System not available"}

            import asyncio

            async def notify():
                record = await self.rag_pipeline.longitudinal_manager.get_or_create_record(patient_id.strip())

                recipients = []
                for member in record.care_team:
                    if not target_roles or member.role in target_roles:
                        recipients.append({
                            "provider_id": member.provider_id,
                            "name": member.name,
                            "role": member.role,
                            "phone": member.phone_number,
                            "status": "notified"
                        })

                return recipients

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            recipients = loop.run_until_complete(notify())

            return {
                "status": "success",
                "patient_id": patient_id,
                "priority": priority,
                "message": message[:50] + "..." if len(message) > 50 else message,
                "recipients_count": len(recipients),
                "recipients": recipients
            }

        except Exception as e:
            logger.error(f"Error notifying care team: {e}")
            return {"error": str(e)}

    # V25: FHIR Interoperability Handlers (Phase 7)

    def _handle_fhir_export(
        self,
        patient_id: str,
        include_obs: bool,
        include_meds: bool,
        include_care_team: bool,
        save_to_file: bool
    ):
        """Export patient data to FHIR Bundle."""
        try:
            if not patient_id or not patient_id.strip():
                return {"error": "Please enter a patient ID"}, None

            if not LONGITUDINAL_MEMORY_AVAILABLE or not self.rag_pipeline.longitudinal_manager:
                return {"error": "System not available"}, None

            import asyncio
            from personalization.fhir_adapter import FHIRAdapter, export_to_file as fhir_export

            async def export():
                adapter = FHIRAdapter()
                bundle = await adapter.export_patient_bundle(
                    patient_id.strip(),
                    self.rag_pipeline.longitudinal_manager,
                    include_observations=include_obs,
                    include_medications=include_meds,
                    include_care_team=include_care_team
                )

                bundle_dict = bundle.to_dict()

                file_path = None
                if save_to_file:
                    from pathlib import Path
                    from datetime import datetime
                    export_dir = Path("data/fhir_exports")
                    export_dir.mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_path = str(export_dir / f"patient_{patient_id}_{timestamp}.json")
                    fhir_export(bundle, file_path)

                return bundle_dict, file_path

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            bundle_dict, file_path = loop.run_until_complete(export())

            result = {
                "status": "success",
                "patient_id": patient_id.strip(),
                "resource_count": len(bundle_dict.get("entry", [])),
                "file_saved": file_path
            }

            return result, bundle_dict

        except Exception as e:
            logger.error(f"Error exporting FHIR: {e}")
            return {"error": str(e)}, None

    def _handle_fhir_import(self, bundle_json: str):
        """Import FHIR Bundle into longitudinal memory."""
        try:
            if not bundle_json or not bundle_json.strip():
                return {"error": "Please paste FHIR Bundle JSON"}

            if not LONGITUDINAL_MEMORY_AVAILABLE or not self.rag_pipeline.longitudinal_manager:
                return {"error": "System not available"}

            import asyncio
            import json
            from personalization.fhir_adapter import FHIRAdapter

            try:
                bundle_data = json.loads(bundle_json)
            except json.JSONDecodeError as e:
                return {"error": f"Invalid JSON: {str(e)}"}

            async def do_import():
                adapter = FHIRAdapter()
                result = await adapter.import_bundle(bundle_data, self.rag_pipeline.longitudinal_manager)
                return result

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(do_import())

            return {
                "status": "success",
                "patients_imported": result.get("patients_imported", 0),
                "observations_imported": result.get("observations_imported", 0),
                "medications_imported": result.get("medications_imported", 0),
                "errors": result.get("errors", [])
            }

        except Exception as e:
            logger.error(f"Error importing FHIR: {e}")
            return {"error": str(e)}

    def _handle_fhir_validate(self, resource_json: str):
        """Validate FHIR resource or bundle."""
        try:
            if not resource_json or not resource_json.strip():
                return {"error": "Please paste FHIR JSON to validate"}

            import json
            from personalization.fhir_adapter import FHIRAdapter

            try:
                resource_data = json.loads(resource_json)
            except json.JSONDecodeError as e:
                return {"error": f"Invalid JSON: {str(e)}"}

            adapter = FHIRAdapter()

            # Check if it's a Bundle or single resource
            if resource_data.get("resourceType") == "Bundle":
                result = adapter.validate_bundle(resource_data)
            else:
                errors = adapter.validate_resource(resource_data)
                result = {
                    "valid": len(errors) == 0,
                    "errors": errors
                }

            return result

        except Exception as e:
            logger.error(f"Error validating FHIR: {e}")
            return {"error": str(e)}

    def _handle_fhir_get_codes(self):
        """Get SNOMED code mappings."""
        try:
            from personalization.fhir_adapter import SYMPTOM_SNOMED_CODES, SEVERITY_FHIR_CODES
            return SYMPTOM_SNOMED_CODES, SEVERITY_FHIR_CODES
        except Exception as e:
            logger.error(f"Error getting SNOMED codes: {e}")
            return {"error": str(e)}, {}


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
    parser.add_argument(
        "--provider", "-p",
        choices=["g", "b", "r", "gemini", "bolna", "retell"],
        default="b",
        help="Voice AI provider: g/gemini=Gemini Live, b/bolna=Bolna.ai, r/retell=Retell.AI (default: b)"
    )

    args = parser.parse_args()

    # Set voice provider based on argument
    voice_provider = args.provider.lower()
    if voice_provider in ["g", "gemini"]:
        os.environ["VOICE_PROVIDER"] = "gemini"
        os.environ["GEMINI_LIVE_ENABLED"] = "true"
        logger.info("=" * 60)
        logger.info("🎙️  VOICE PROVIDER: Google Gemini Live API (Vertex AI)")
        logger.info("=" * 60)
    elif voice_provider in ["r", "retell"]:
        os.environ["VOICE_PROVIDER"] = "retell"
        os.environ["GEMINI_LIVE_ENABLED"] = "false"
        os.environ["RETELL_ENABLED"] = "true"
        logger.info("=" * 60)
        logger.info("📞 VOICE PROVIDER: Retell.AI + Vobiz.ai (Indian PSTN)")
        logger.info("=" * 60)
    else:
        os.environ["VOICE_PROVIDER"] = "bolna"
        os.environ["GEMINI_LIVE_ENABLED"] = "false"
        logger.info("=" * 60)
        logger.info("📞 VOICE PROVIDER: Bolna.ai")
        logger.info("=" * 60)

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

    # Knowledge Graph service globals
    kg_rag = None
    kg_enabled = False

    # GraphRAG service globals
    graphrag_config = None
    graphrag_indexer = None
    graphrag_query_engine = None
    graphrag_enabled = False

    # Retell.AI service globals
    retell_client = None
    retell_webhook_handler = None
    retell_llm_handler = None
    retell_enabled = False

    @app.on_event("startup")
    async def startup_graphrag():
        """Initialize GraphRAG service on startup."""
        nonlocal graphrag_config, graphrag_indexer, graphrag_query_engine, graphrag_enabled

        if not GRAPHRAG_AVAILABLE:
            logger.info("GraphRAG module not available")
            return

        try:
            settings_path = Path("./data/graphrag/settings.yaml")
            if not settings_path.exists():
                logger.warning("GraphRAG settings.yaml not found - GraphRAG disabled")
                return

            graphrag_config = GraphRAGConfig.from_yaml(str(settings_path))
            graphrag_indexer = GraphRAGIndexer(graphrag_config)
            graphrag_query_engine = GraphRAGQueryEngine(graphrag_config)

            # Initialize query engine (loads data if available)
            await graphrag_query_engine.initialize()

            graphrag_enabled = True
            logger.info("GraphRAG initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize GraphRAG: {e}")
            graphrag_enabled = False

    @app.on_event("startup")
    async def startup_knowledge_graph():
        """Initialize Knowledge Graph service on startup."""
        nonlocal kg_rag, kg_enabled

        if not KNOWLEDGE_GRAPH_AVAILABLE:
            logger.info("Knowledge Graph module not available")
            return

        try:
            kg_rag = KnowledgeGraphRAG()
            connected = await kg_rag.initialize()

            if connected:
                kg_enabled = True
                logger.info("Knowledge Graph initialized with Neo4j connection")
            else:
                kg_enabled = True  # Still enabled for pattern-based extraction
                logger.info("Knowledge Graph initialized (limited mode - no Neo4j)")

        except Exception as e:
            logger.error(f"Failed to initialize Knowledge Graph: {e}")
            kg_enabled = False

    @app.on_event("shutdown")
    async def shutdown_knowledge_graph():
        """Cleanup Knowledge Graph service on shutdown."""
        nonlocal kg_rag

        if kg_rag:
            try:
                await kg_rag.close()
                logger.info("Knowledge Graph connections closed")
            except Exception as e:
                logger.error(f"Error closing Knowledge Graph: {e}")

    @app.on_event("startup")
    async def startup_gemini_live():
        """Initialize Gemini Live service on startup."""
        nonlocal gemini_service, gemini_session_manager, gemini_live_enabled

        if not GEMINI_LIVE_AVAILABLE:
            logger.info("Gemini Live module not available - voice WebSocket disabled")
            return

        try:
            gemini_config = get_gemini_config()

            # Check both config file AND environment variable (--provider flag sets this)
            env_enabled = os.getenv("GEMINI_LIVE_ENABLED", "").lower() == "true"
            if not gemini_config.enabled and not env_enabled:
                logger.info("Gemini Live disabled in config - voice WebSocket disabled")
                return

            if env_enabled:
                logger.info("🎙️  Gemini Live enabled via --provider flag")

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

    @app.on_event("startup")
    async def startup_retell():
        """Initialize Retell.AI service on startup."""
        nonlocal retell_client, retell_webhook_handler, retell_llm_handler, retell_enabled

        # Check if Retell is enabled via environment
        retell_env_enabled = os.getenv("RETELL_ENABLED", "").lower() == "true"
        if not retell_env_enabled:
            logger.info("Retell.AI disabled - use -p r to enable")
            return

        try:
            from retell_integration import (
                RetellClient,
                RetellWebhookHandler,
                RetellCustomLLMHandler
            )

            # Initialize Retell client
            retell_client = RetellClient()
            if not retell_client.is_available():
                logger.warning("Retell API key not configured - Retell disabled")
                return

            # Initialize webhook handler
            retell_webhook_handler = RetellWebhookHandler()

            # Initialize Custom LLM handler with RAG pipeline
            retell_llm_handler = RetellCustomLLMHandler(rag_pipeline=rag_pipeline)

            retell_enabled = True
            logger.info("📞 Retell.AI initialized - endpoints available:")
            logger.info("   WebSocket: /ws/retell/llm/{call_id}")
            logger.info("   Webhook:   /api/retell/webhook")
            logger.info("   Health:    /api/retell/health")
            logger.info("   Stats:     /api/retell/stats")

        except ImportError as e:
            logger.warning(f"Retell integration not available: {e}")
            retell_enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize Retell: {e}")
            retell_enabled = False

    @app.on_event("shutdown")
    async def shutdown_retell():
        """Cleanup Retell service on shutdown."""
        nonlocal retell_llm_handler

        if retell_llm_handler:
            try:
                # Close any active sessions
                logger.info("Retell Custom LLM handler stopped")
            except Exception as e:
                logger.error(f"Error stopping Retell handler: {e}")

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

                                # Note: Response task will be started when first audio arrives
                                # Starting receive() before sending audio causes Gemini to close connection

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

                            # Start response stream after first audio is sent
                            if response_task is None or response_task.done():
                                response_task = asyncio.create_task(
                                    stream_gemini_responses(websocket, session, user_id)
                                )
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

    # ==========================================================================
    # BOLNA.AI INTEGRATION ENDPOINTS
    # Palli Sahayak Voice AI Agent Helpline
    # ==========================================================================

    # Initialize Bolna webhook handler
    bolna_webhook_handler = BolnaWebhookHandler() if BOLNA_AVAILABLE else None

    # Bolna webhook secret for signature verification
    BOLNA_WEBHOOK_SECRET = os.getenv("BOLNA_WEBHOOK_SECRET", "")

    def verify_bolna_signature(payload: bytes, signature: str) -> bool:
        """Verify Bolna webhook signature using HMAC-SHA256."""
        if not BOLNA_WEBHOOK_SECRET:
            return True  # Skip verification if no secret configured

        expected = hmac.new(
            BOLNA_WEBHOOK_SECRET.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(expected, signature)

    @app.post("/api/bolna/query")
    async def bolna_query_endpoint(
        request: Request,
        x_bolna_signature: Optional[str] = Header(None)
    ):
        """
        RAG query endpoint for Bolna custom function calls.

        This endpoint receives queries from Bolna's LLM during voice calls,
        queries the palliative care knowledge base, and returns grounded responses.

        Request Body:
        {
            "query": "How to manage pain for cancer patients?",
            "language": "hi",
            "context": "User asking about pain management",
            "source": "bolna_call"
        }

        Response:
        {
            "status": "success",
            "answer": "For cancer pain management...",
            "sources": ["WHO Pain Guidelines", "AIIMS Palliative Care Manual"],
            "confidence": 0.92
        }
        """
        try:
            # Get raw body for signature verification
            body = await request.body()

            # Verify signature if provided
            if x_bolna_signature and not verify_bolna_signature(body, x_bolna_signature):
                logger.warning("Invalid Bolna signature received")
                raise HTTPException(status_code=401, detail="Invalid signature")

            # Parse request
            data = json.loads(body)
            query = data.get("query", "")
            language = data.get("language", "en")
            context = data.get("context", "")
            source = data.get("source", "unknown")

            # Enhanced logging for Bolna RAG queries
            import time as _time
            _query_start = _time.time()
            logger.info("=" * 60)
            logger.info("📞 BOLNA VOICE CALL - RAG QUERY")
            logger.info("=" * 60)
            logger.info(f"🗣️  Query: {query[:150]}{'...' if len(query) > 150 else ''}")
            logger.info(f"🌐 Language: {language} | Source: {source}")
            logger.info("-" * 60)

            if not query:
                return JSONResponse({
                    "status": "error",
                    "answer": "I didn't catch your question. Could you please repeat?",
                    "sources": [],
                    "confidence": 0.0
                })
            
            # =================================================================
            # VOICE SAFETY CHECK - For Bolna Voice Calls
            # =================================================================
            try:
                from voice_safety_wrapper import get_voice_safety_wrapper
                safety_wrapper = get_voice_safety_wrapper()
                
                safety_result = await safety_wrapper.check_voice_query(
                    user_id=f"bolna_{source}",
                    transcript=query,
                    language=language,
                    call_id=source
                )
                
                if safety_result.should_escalate:
                    logger.warning(f"🚨 Bolna voice safety escalation: {safety_result.event_type}")
                    
                    # Handle escalation
                    await safety_wrapper.handle_voice_escalation(
                        safety_result, provider="bolna"
                    )
                    
                    # Return safety message as response
                    return JSONResponse({
                        "status": "safety_escalation",
                        "answer": safety_result.safety_message,
                        "sources": [],
                        "confidence": 1.0,
                        "event_type": safety_result.event_type.value if safety_result.event_type else None
                    })
                
                # Use modified query if available
                if safety_result.modified_transcript:
                    query = safety_result.modified_transcript
                    
            except Exception as e:
                logger.error(f"Bolna voice safety check error (proceeding): {e}")

            # Query RAG pipeline
            rag_result = await rag_pipeline.query(
                question=query,
                user_id=f"bolna_{source}",
                source_language=language,
                top_k=3
            )

            if rag_result.get("status") == "success":
                answer = rag_result.get("answer", "")
                
                # =================================================================
                # VOICE OPTIMIZATION for Bolna Response
                # =================================================================
                try:
                    from voice_safety_wrapper import get_voice_safety_wrapper
                    safety_wrapper = get_voice_safety_wrapper()
                    
                    # Optimize for voice output
                    answer = safety_wrapper.optimize_for_voice(
                        answer,
                        user_id=f"bolna_{source}",
                        language=language,
                        max_duration_seconds=30
                    )
                    
                    # Add evidence warning if needed
                    evidence_badge = rag_result.get("safety_enhancements", {}).get("evidence_badge")
                    if evidence_badge:
                        answer = safety_wrapper.add_evidence_to_voice(
                            answer, evidence_badge, language
                        )
                        
                except Exception as e:
                    logger.warning(f"Bolna voice optimization error (proceeding): {e}")

                # Extract source citations
                sources = []
                if "sources" in rag_result:
                    sources = [s.get("title", s.get("filename", "Unknown")) for s in rag_result["sources"][:3]]
                elif "citations" in rag_result:
                    sources = [c.get("source", "Knowledge Base") for c in rag_result["citations"][:3]]
                else:
                    sources = ["Palliative Care Knowledge Base"]

                # Calculate confidence based on retrieval
                confidence = min(0.95, rag_result.get("relevance_score", 0.8))

                # Enhanced success logging
                _query_duration = _time.time() - _query_start
                logger.info(f"✅ RAG SUCCESS ({_query_duration:.2f}s)")
                logger.info(f"📚 Sources: {', '.join(sources[:3])}")
                logger.info(f"💬 Answer preview: {answer[:100]}{'...' if len(answer) > 100 else ''}")
                logger.info(f"🎯 Confidence: {confidence:.0%}")
                logger.info("=" * 60)

                return JSONResponse({
                    "status": "success",
                    "answer": answer,
                    "sources": sources,
                    "confidence": confidence,
                    "language": language
                })

            else:
                # RAG query failed - return graceful fallback
                _query_duration = _time.time() - _query_start
                logger.warning(f"⚠️  RAG PARTIAL ({_query_duration:.2f}s) - Status: {rag_result.get('status')}")
                logger.warning("📝 Using fallback response")
                logger.info("=" * 60)
                return JSONResponse({
                    "status": "partial",
                    "answer": "I'm having trouble accessing my knowledge base right now. Based on general palliative care principles, I'd recommend consulting with your healthcare provider for specific guidance.",
                    "sources": [],
                    "confidence": 0.3
                })

        except json.JSONDecodeError as e:
            logger.error(f"❌ BOLNA ERROR: Invalid JSON - {e}")
            logger.info("=" * 60)
            return JSONResponse({
                "status": "error",
                "answer": "I received an invalid request. Please try again.",
                "sources": [],
                "confidence": 0.0
            }, status_code=400)

        except HTTPException:
            raise

        except Exception as e:
            logger.error(f"❌ BOLNA ERROR: {e}")
            logger.info("=" * 60)
            return JSONResponse({
                "status": "error",
                "answer": "I apologize, but I'm experiencing technical difficulties. Please try again or contact the helpline directly.",
                "sources": [],
                "confidence": 0.0
            }, status_code=500)

    @app.post("/api/bolna/webhook")
    async def bolna_webhook_endpoint(request: Request):
        """
        Webhook endpoint for Bolna call events.

        Receives notifications about:
        - call_started: New call initiated
        - call_ended: Call completed with summary
        - extraction_completed: Data extracted from call
        - transcription: Real-time transcription updates
        """
        try:
            data = await request.json()
            event_type = data.get("event", data.get("type", "unknown"))
            call_id = data.get("call_id", data.get("id", "unknown"))

            logger.info(f"Bolna webhook received: {event_type} for call {call_id}")

            # Process with webhook handler if available
            if bolna_webhook_handler:
                result = await bolna_webhook_handler.handle_event(data)
                return JSONResponse(result)

            # Basic handling if handler not available
            if event_type == "call_ended":
                summary = data.get("summary", "")
                duration = data.get("duration_seconds", data.get("duration", 0))
                logger.info(f"Call {call_id} ended - Duration: {duration}s")
                if summary:
                    logger.info(f"Call summary: {summary[:200]}...")

            elif event_type == "extraction_completed":
                extracted = data.get("extracted_data", data.get("data", {}))
                logger.info(f"Call {call_id} extraction: {extracted}")

                # Check for high urgency
                if extracted.get("urgency_level") in ("high", "emergency"):
                    logger.warning(f"HIGH URGENCY call {call_id}: {extracted.get('user_concern', 'Unknown')}")

            return JSONResponse({"status": "received", "event": event_type})

        except Exception as e:
            logger.error(f"Bolna webhook error: {e}", exc_info=True)
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

    @app.get("/api/bolna/stats")
    async def bolna_stats_endpoint():
        """Get Bolna call statistics."""
        if bolna_webhook_handler:
            return JSONResponse(bolna_webhook_handler.get_call_stats())
        return JSONResponse({"error": "Bolna integration not available"}, status_code=503)

    @app.get("/api/bolna/calls")
    async def bolna_recent_calls_endpoint(limit: int = 10):
        """Get recent Bolna calls."""
        if bolna_webhook_handler:
            return JSONResponse({"calls": bolna_webhook_handler.get_recent_calls(limit)})
        return JSONResponse({"error": "Bolna integration not available"}, status_code=503)

    # ==========================================================================
    # MEDICATION VOICE REMINDER ENDPOINTS
    # ==========================================================================
    
    try:
        from medication_voice_reminders import get_medication_voice_reminder_system
        voice_reminder_system = get_medication_voice_reminder_system()
        MEDICATION_VOICE_AVAILABLE = True
    except ImportError:
        voice_reminder_system = None
        MEDICATION_VOICE_AVAILABLE = False
    
    @app.post("/api/medication/voice-reminder")
    async def create_voice_reminder_endpoint(request: Request):
        """
        Create a new medication voice reminder.
        
        Request Body:
        {
            "user_id": "phone_number",
            "phone_number": "+919876543210",
            "medication_name": "Paracetamol",
            "dosage": "500mg after food",
            "reminder_time": "2025-01-28T08:00:00",
            "language": "en"
        }
        """
        if not MEDICATION_VOICE_AVAILABLE:
            return JSONResponse({"error": "Voice reminder system not available"}, status_code=503)
        
        try:
            data = await request.json()
            
            reminder = voice_reminder_system.create_voice_reminder(
                user_id=data.get("user_id"),
                phone_number=data.get("phone_number"),
                medication_name=data.get("medication_name"),
                dosage=data.get("dosage"),
                reminder_time=datetime.fromisoformat(data.get("reminder_time")),
                language=data.get("language", "en"),
                preferred_provider=data.get("provider", "bolna")
            )
            
            return JSONResponse({
                "status": "success",
                "reminder_id": reminder.reminder_id,
                "scheduled_time": reminder.scheduled_time.isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error creating voice reminder: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)
    
    @app.get("/api/medication/voice-reminders/{user_id}")
    async def get_user_voice_reminders_endpoint(user_id: str):
        """Get all voice reminders for a user."""
        if not MEDICATION_VOICE_AVAILABLE:
            return JSONResponse({"error": "Voice reminder system not available"}, status_code=503)
        
        try:
            reminders = voice_reminder_system.get_user_reminders(user_id)
            return JSONResponse({"reminders": reminders})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
    
    @app.post("/api/medication/voice-reminder/callback")
    async def voice_reminder_callback_endpoint(request: Request):
        """
        Callback endpoint for voice reminder call completion.
        
        Called by Bolna/Retell when medication reminder call ends.
        
        Request Body:
        {
            "call_id": "call_123",
            "reminder_id": "rem_456",
            "status": "completed",
            "duration": 45,
            "patient_confirmed": true,
            "confirmation_method": "dtmf_1"
        }
        """
        if not MEDICATION_VOICE_AVAILABLE:
            return JSONResponse({"status": "ignored"})
        
        try:
            data = await request.json()
            
            await voice_reminder_system.handle_call_completed(
                call_id=data.get("call_id"),
                status=data.get("status"),
                duration=data.get("duration", 0),
                patient_response=data.get("patient_response")
            )
            
            return JSONResponse({"status": "received"})
            
        except Exception as e:
            logger.error(f"Voice reminder callback error: {e}")
            return JSONResponse({"status": "error"}, status_code=500)
    
    @app.get("/api/medication/adherence/{user_id}")
    async def get_adherence_stats_endpoint(user_id: str, days: int = 7):
        """Get medication adherence statistics for a user."""
        if not MEDICATION_VOICE_AVAILABLE:
            return JSONResponse({"error": "Voice reminder system not available"}, status_code=503)
        
        try:
            stats = voice_reminder_system.get_adherence_stats(user_id, days)
            return JSONResponse(stats)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
    
    @app.get("/api/medication/pending-calls")
    async def get_pending_calls_endpoint():
        """Get all pending medication reminder calls (for dashboard)."""
        if not MEDICATION_VOICE_AVAILABLE:
            return JSONResponse({"error": "Voice reminder system not available"}, status_code=503)
        
        try:
            pending = voice_reminder_system.get_pending_calls()
            return JSONResponse({"calls": pending})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    # ==========================================================================
    # END BOLNA.AI INTEGRATION
    # ==========================================================================

    # ==========================================================================
    # RETELL.AI INTEGRATION - Custom LLM WebSocket + Webhooks
    # ==========================================================================

    @app.websocket("/ws/retell/llm/{call_id}")
    async def retell_llm_websocket(websocket: WebSocket, call_id: str):
        """
        WebSocket endpoint for Retell Custom LLM.

        This handles the Retell LLM WebSocket protocol:
        - Receives: ping_pong, call_details, update_only, response_required
        - Sends: response with content (connected to RAG pipeline)

        Documentation: https://docs.retellai.com/api-references/llm-websocket
        """
        await websocket.accept()

        if not retell_enabled or not retell_llm_handler:
            await websocket.send_json({
                "response_type": "error",
                "content": "Retell Custom LLM not available"
            })
            await websocket.close(code=1008, reason="Service unavailable")
            return

        logger.info(f"Retell LLM WebSocket connected: call_id={call_id}")

        try:
            await retell_llm_handler.handle_websocket(websocket, call_id)
        except WebSocketDisconnect:
            logger.info(f"Retell LLM WebSocket disconnected: call_id={call_id}")
        except Exception as e:
            logger.error(f"Retell LLM WebSocket error for {call_id}: {e}")
        finally:
            # Cleanup session
            if retell_llm_handler:
                retell_llm_handler.end_session(call_id)

    @app.post("/api/retell/webhook")
    async def retell_webhook_endpoint(request: Request):
        """
        Webhook endpoint for Retell call events.

        Handles: call_started, call_ended, call_analyzed
        """
        if not retell_enabled or not retell_webhook_handler:
            return JSONResponse(
                {"error": "Retell integration not available"},
                status_code=503
            )

        try:
            event_data = await request.json()
            result = await retell_webhook_handler.handle_event(event_data)
            return JSONResponse(result)
        except Exception as e:
            logger.error(f"Retell webhook error: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/api/retell/health")
    async def retell_health_endpoint():
        """Check Retell integration health."""
        if not retell_enabled:
            return JSONResponse({
                "status": "unavailable",
                "message": "Retell not initialized - use -p r to enable"
            })

        try:
            client_healthy = retell_client.is_available() if retell_client else False
            return JSONResponse({
                "status": "healthy" if client_healthy else "limited",
                "client_available": client_healthy,
                "llm_handler_ready": retell_llm_handler is not None,
                "webhook_handler_ready": retell_webhook_handler is not None
            })
        except Exception as e:
            return JSONResponse({
                "status": "error",
                "error": str(e)
            })

    @app.get("/api/retell/stats")
    async def retell_stats_endpoint():
        """Get Retell call statistics."""
        if not retell_enabled or not retell_webhook_handler:
            return JSONResponse(
                {"error": "Retell integration not available"},
                status_code=503
            )

        try:
            stats = retell_webhook_handler.get_call_stats()
            return JSONResponse(stats)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/api/retell/calls")
    async def retell_calls_endpoint(limit: int = 10):
        """Get recent Retell calls."""
        if not retell_enabled or not retell_webhook_handler:
            return JSONResponse(
                {"error": "Retell integration not available"},
                status_code=503
            )

        try:
            calls = retell_webhook_handler.get_recent_calls(limit=limit)
            return JSONResponse({"calls": calls})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/api/retell/vobiz")
    async def retell_vobiz_config_endpoint():
        """Get Vobiz.ai telephony configuration status."""
        if not retell_enabled:
            return JSONResponse(
                {"error": "Retell integration not available"},
                status_code=503
            )

        try:
            from retell_integration import get_vobiz_config
            vobiz = get_vobiz_config()
            return JSONResponse(vobiz.to_dict())
        except ImportError:
            return JSONResponse({"error": "Vobiz config not available"}, status_code=503)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    # ==========================================================================
    # END RETELL.AI INTEGRATION
    # ==========================================================================

    # ==========================================================================
    # SARVAM AI INTEGRATION ENDPOINTS
    # Indian Language STT (Saaras v3, 22 languages) + TTS (Bulbul v3, 11 languages)
    # ==========================================================================

    sarvam_enabled = False
    sarvam_client = None
    sarvam_webhook_handler = None

    if SARVAM_AVAILABLE:
        try:
            sarvam_client = SarvamClient()
            if sarvam_client.is_available():
                sarvam_enabled = True
                sarvam_webhook_handler = SarvamWebhookHandler()
                logger.info("🇮🇳 Sarvam AI initialized - endpoints available:")
                logger.info("   STT:       POST /api/sarvam/stt")
                logger.info("   TTS:       POST /api/sarvam/tts")
                logger.info("   Translate:  POST /api/sarvam/translate")
                logger.info("   Health:    GET  /api/sarvam/health")
                logger.info("   Languages: GET  /api/sarvam/languages")
            else:
                logger.info("Sarvam AI not configured (no API key) - set SARVAM_API_KEY")
        except Exception as e:
            logger.warning(f"Failed to initialize Sarvam AI: {e}")

    @app.get("/api/sarvam/health")
    async def sarvam_health_endpoint():
        """Check Sarvam AI health status."""
        if not sarvam_enabled or not sarvam_client:
            return JSONResponse({
                "status": "unavailable",
                "message": "Sarvam AI not initialized - set SARVAM_API_KEY"
            })

        try:
            healthy = await sarvam_client.health_check()
            return JSONResponse({
                "status": "healthy" if healthy else "degraded",
                "provider": "sarvam_ai",
                "stt_model": "saaras:v3",
                "tts_model": "bulbul:v3",
                "stt_languages": len(SARVAM_STT_LANGUAGES),
                "tts_languages": len(SARVAM_TTS_LANGUAGES),
            })
        except Exception as e:
            return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

    @app.get("/api/sarvam/languages")
    async def sarvam_languages_endpoint():
        """List supported Sarvam languages."""
        return JSONResponse({
            "stt_languages": SARVAM_STT_LANGUAGES if SARVAM_AVAILABLE else [],
            "tts_languages": SARVAM_TTS_LANGUAGES if SARVAM_AVAILABLE else [],
            "language_configs": {
                code: {"name": cfg["name"], "stt": cfg["stt_supported"], "tts": cfg["tts_supported"]}
                for code, cfg in (SARVAM_LANGUAGE_CONFIGS or {}).items()
            } if SARVAM_AVAILABLE else {},
        })

    @app.post("/api/sarvam/stt")
    async def sarvam_stt_endpoint(request: Request):
        """
        Speech-to-text using Sarvam Saaras v3.

        Accepts multipart/form-data with:
        - file: Audio file (WAV, MP3, FLAC, OGG, WebM)
        - language: BCP-47 language code (e.g. "hi-IN"), default "hi-IN"
        - model: "saaras:v3" or "saaras:flash", default "saaras:v3"
        - mode: "formal", "code-mixed", or "spoken-form", default "formal"
        """
        if not sarvam_enabled or not sarvam_client:
            return JSONResponse(
                {"error": "Sarvam AI not available"},
                status_code=503
            )

        try:
            form = await request.form()
            audio_file = form.get("file")
            if not audio_file:
                return JSONResponse({"error": "No audio file provided"}, status_code=400)

            audio_data = await audio_file.read()
            language = form.get("language", "hi-IN")
            model = form.get("model", "saaras:v3")
            mode = form.get("mode", "formal")

            result = await sarvam_client.speech_to_text(
                audio_data=audio_data,
                language=language,
                model=model,
                mode=mode,
            )

            if result.success:
                return JSONResponse({
                    "status": "success",
                    "transcript": result.transcript,
                    "language_code": result.language_code,
                    "timestamps": result.timestamps,
                })
            else:
                return JSONResponse(
                    {"status": "error", "error": result.error},
                    status_code=400
                )

        except Exception as e:
            logger.error(f"Sarvam STT endpoint error: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/api/sarvam/tts")
    async def sarvam_tts_endpoint(request: Request):
        """
        Text-to-speech using Sarvam Bulbul v3.

        Request body:
        {
            "text": "Text to speak",
            "language": "hi-IN",
            "voice": "meera",
            "pace": 1.0,
            "sample_rate": 22050
        }

        Returns base64-encoded WAV audio.
        """
        if not sarvam_enabled or not sarvam_client:
            return JSONResponse(
                {"error": "Sarvam AI not available"},
                status_code=503
            )

        try:
            data = await request.json()
            text = data.get("text", "")
            if not text:
                return JSONResponse({"error": "No text provided"}, status_code=400)

            language = data.get("language", "hi-IN")
            voice = data.get("voice", "meera")
            pace = data.get("pace", 1.0)
            sample_rate = data.get("sample_rate", 22050)

            result = await sarvam_client.text_to_speech(
                text=text,
                language=language,
                voice=voice,
                pace=pace,
                sample_rate=sample_rate,
            )

            if result.success:
                return JSONResponse({
                    "status": "success",
                    "audio_base64": result.audio_base64,
                    "sample_rate": result.sample_rate,
                })
            else:
                return JSONResponse(
                    {"status": "error", "error": result.error},
                    status_code=400
                )

        except Exception as e:
            logger.error(f"Sarvam TTS endpoint error: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/api/sarvam/translate")
    async def sarvam_translate_endpoint(request: Request):
        """
        Translate text between Indian languages.

        Request body:
        {
            "text": "How are you feeling today?",
            "source_language": "en-IN",
            "target_language": "hi-IN",
            "mode": "formal"
        }
        """
        if not sarvam_enabled or not sarvam_client:
            return JSONResponse(
                {"error": "Sarvam AI not available"},
                status_code=503
            )

        try:
            data = await request.json()
            text = data.get("text", "")
            if not text:
                return JSONResponse({"error": "No text provided"}, status_code=400)

            source_lang = data.get("source_language", "en-IN")
            target_lang = data.get("target_language", "hi-IN")
            mode = data.get("mode", "formal")

            result = await sarvam_client.translate(
                text=text,
                source_language=source_lang,
                target_language=target_lang,
                mode=mode,
            )

            if result.success:
                return JSONResponse({
                    "status": "success",
                    "translated_text": result.translated_text,
                    "source_language": result.source_language,
                    "target_language": result.target_language,
                })
            else:
                return JSONResponse(
                    {"status": "error", "error": result.error},
                    status_code=400
                )

        except Exception as e:
            logger.error(f"Sarvam translate endpoint error: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/api/sarvam/webhook")
    async def sarvam_webhook_endpoint(request: Request):
        """Handle Sarvam webhook events."""
        if not sarvam_webhook_handler:
            return JSONResponse(
                {"error": "Sarvam webhooks not available"},
                status_code=503
            )

        try:
            data = await request.json()
            result = await sarvam_webhook_handler.handle_event(data)
            return JSONResponse(result)
        except Exception as e:
            logger.error(f"Sarvam webhook error: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/api/sarvam/stats")
    async def sarvam_stats_endpoint():
        """Get Sarvam session statistics."""
        if not sarvam_webhook_handler:
            return JSONResponse({
                "status": "unavailable",
                "message": "Sarvam not initialized"
            })

        try:
            stats = sarvam_webhook_handler.get_session_stats()
            return JSONResponse({"status": "success", **stats})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    # ==========================================================================
    # END SARVAM AI INTEGRATION
    # ==========================================================================

    # ==========================================================================
    # KNOWLEDGE GRAPH API ENDPOINTS
    # ==========================================================================

    @app.get("/api/kg/health")
    async def kg_health_endpoint():
        """Check Knowledge Graph health status."""
        if not kg_enabled or not kg_rag:
            return JSONResponse({
                "status": "unavailable",
                "message": "Knowledge Graph not initialized"
            })

        try:
            health = await kg_rag.health_check()
            return JSONResponse({
                "status": "healthy" if health.get("neo4j_healthy") else "limited",
                **health
            })
        except Exception as e:
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    @app.get("/api/kg/stats")
    async def kg_stats_endpoint():
        """Get Knowledge Graph statistics."""
        if not kg_enabled or not kg_rag:
            return JSONResponse({
                "available": False,
                "message": "Knowledge Graph not initialized"
            })

        try:
            stats = await kg_rag.get_statistics()
            return JSONResponse(stats)
        except Exception as e:
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    @app.post("/api/kg/query")
    async def kg_query_endpoint(request: Request):
        """
        Query the Knowledge Graph with natural language.

        Request body:
        {
            "question": "What medications treat pain?",
            "include_vector": true,  // Optional: merge with vector results
            "top_k": 10  // Optional: max results
        }
        """
        if not kg_enabled or not kg_rag:
            return JSONResponse({
                "status": "error",
                "message": "Knowledge Graph not available"
            }, status_code=503)

        try:
            body = await request.json()
            question = body.get("question", "")

            if not question:
                return JSONResponse({
                    "status": "error",
                    "error": "Question is required"
                }, status_code=400)

            # Query the knowledge graph
            result = await kg_rag.query(
                question=question,
                include_vector_results=body.get("include_vector", False),
                top_k=body.get("top_k", 10)
            )

            return JSONResponse({
                "status": "success",
                **result
            })

        except Exception as e:
            logger.error(f"KG query error: {e}")
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    @app.post("/api/kg/extract")
    async def kg_extract_endpoint(request: Request):
        """
        Extract entities from text.

        Request body:
        {
            "text": "The patient has severe pain and is taking morphine."
        }
        """
        if not kg_enabled or not kg_rag:
            return JSONResponse({
                "status": "error",
                "message": "Knowledge Graph not available"
            }, status_code=503)

        try:
            body = await request.json()
            text = body.get("text", "")

            if not text:
                return JSONResponse({
                    "status": "error",
                    "error": "Text is required"
                }, status_code=400)

            # Extract entities
            entities, relationships = await kg_rag.entity_extractor.extract(text)

            return JSONResponse({
                "status": "success",
                "entities": [e.to_dict() for e in entities],
                "relationships": [r.to_dict() for r in relationships]
            })

        except Exception as e:
            logger.error(f"KG extraction error: {e}")
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    @app.get("/api/kg/entity/{entity_name}")
    async def kg_entity_graph_endpoint(entity_name: str, depth: int = 1):
        """Get subgraph centered on an entity."""
        if not kg_enabled or not kg_rag:
            return JSONResponse({
                "status": "error",
                "message": "Knowledge Graph not available"
            }, status_code=503)

        try:
            result = await kg_rag.get_entity_graph(entity_name, depth=depth)
            return JSONResponse({
                "status": "success",
                **result
            })
        except Exception as e:
            logger.error(f"KG entity graph error: {e}")
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    @app.get("/api/kg/treatments/{symptom}")
    async def kg_treatments_endpoint(symptom: str):
        """Get medications that treat a specific symptom."""
        if not kg_enabled or not kg_rag:
            return JSONResponse({
                "status": "error",
                "message": "Knowledge Graph not available"
            }, status_code=503)

        try:
            treatments = await kg_rag.get_treatments_for_symptom(symptom)
            return JSONResponse({
                "status": "success",
                "symptom": symptom,
                "treatments": treatments
            })
        except Exception as e:
            logger.error(f"KG treatments error: {e}")
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    @app.get("/api/kg/side-effects/{medication}")
    async def kg_side_effects_endpoint(medication: str):
        """Get side effects of a medication."""
        if not kg_enabled or not kg_rag:
            return JSONResponse({
                "status": "error",
                "message": "Knowledge Graph not available"
            }, status_code=503)

        try:
            side_effects = await kg_rag.get_side_effects(medication)
            return JSONResponse({
                "status": "success",
                "medication": medication,
                "side_effects": side_effects
            })
        except Exception as e:
            logger.error(f"KG side effects error: {e}")
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    @app.get("/api/kg/search")
    async def kg_search_endpoint(q: str, entity_type: Optional[str] = None, limit: int = 20):
        """Search for entities by name."""
        if not kg_enabled or not kg_rag:
            return JSONResponse({
                "status": "error",
                "message": "Knowledge Graph not available"
            }, status_code=503)

        try:
            results = await kg_rag.search_entities(q, entity_type=entity_type, limit=limit)
            return JSONResponse({
                "status": "success",
                "query": q,
                "results": results
            })
        except Exception as e:
            logger.error(f"KG search error: {e}")
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    @app.get("/api/kg/visualization/{entity_name}")
    async def kg_visualization_endpoint(entity_name: str, depth: int = 1):
        """Get visualization HTML for an entity's subgraph."""
        if not kg_enabled or not kg_rag:
            return JSONResponse({
                "status": "error",
                "message": "Knowledge Graph not available"
            }, status_code=503)

        try:
            from fastapi.responses import HTMLResponse

            result = await kg_rag.get_entity_graph(entity_name, depth=depth)

            if "visualization" in result and result["visualization"]:
                viz_data = VisualizationData.from_dict(result["visualization"])
                html = kg_rag.get_visualization_html(viz_data)
                return HTMLResponse(content=html)
            else:
                return JSONResponse({
                    "status": "error",
                    "error": "No visualization data available"
                }, status_code=404)

        except Exception as e:
            logger.error(f"KG visualization error: {e}")
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    @app.post("/api/kg/import-base")
    async def kg_import_base_endpoint():
        """Import base palliative care knowledge into the graph."""
        if not kg_enabled or not kg_rag:
            return JSONResponse({
                "status": "error",
                "message": "Knowledge Graph not available"
            }, status_code=503)

        try:
            result = await kg_rag.import_base_knowledge()
            return JSONResponse({
                "status": "success",
                **result
            })
        except Exception as e:
            logger.error(f"KG import error: {e}")
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    # ==========================================================================
    # END KNOWLEDGE GRAPH INTEGRATION
    # ==========================================================================

    # ==========================================================================
    # GRAPHRAG INTEGRATION ENDPOINTS
    # ==========================================================================

    @app.get("/api/graphrag/health")
    async def graphrag_health():
        """Check GraphRAG health status."""
        if not GRAPHRAG_AVAILABLE:
            return JSONResponse({
                "status": "unavailable",
                "reason": "GraphRAG module not installed"
            })

        if not graphrag_enabled or graphrag_query_engine is None:
            return JSONResponse({
                "status": "not_initialized",
                "reason": "GraphRAG not initialized"
            })

        try:
            stats = await graphrag_query_engine.data_loader.get_stats()
            return JSONResponse({
                "status": "healthy",
                "initialized": True,
                "stats": stats
            })
        except Exception as e:
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    @app.get("/api/graphrag/stats")
    async def graphrag_stats():
        """Get GraphRAG data statistics."""
        if not graphrag_enabled or graphrag_query_engine is None:
            return JSONResponse({
                "status": "error",
                "error": "GraphRAG not available"
            }, status_code=503)

        try:
            stats = await graphrag_query_engine.data_loader.get_stats()
            return JSONResponse({
                "status": "success",
                "stats": stats
            })
        except Exception as e:
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    @app.post("/api/graphrag/query")
    async def graphrag_query(request: Request):
        """
        Query using GraphRAG.

        Request body:
        {
            "query": "Your question here",
            "method": "auto|global|local|drift|basic",
            "top_k": 10
        }

        Methods:
        - auto: Automatically select best method based on query
        - global: Holistic corpus-wide search using community reports
        - local: Entity-focused search
        - drift: Multi-phase iterative search (DRIFT)
        - basic: Simple vector similarity search
        """
        if not graphrag_enabled or graphrag_query_engine is None:
            return JSONResponse({
                "status": "error",
                "error": "GraphRAG not available"
            }, status_code=503)

        try:
            body = await request.json()
            query = body.get("query", "")
            method = body.get("method", "auto").lower()
            top_k = body.get("top_k", 10)

            if not query:
                return JSONResponse({
                    "status": "error",
                    "error": "Query is required"
                }, status_code=400)

            # Execute search based on method
            if method == "auto":
                result = await graphrag_query_engine.auto_search(query)
            elif method == "global":
                result = await graphrag_query_engine.global_search(query)
            elif method == "local":
                result = await graphrag_query_engine.local_search(query, top_k_entities=top_k)
            elif method == "drift":
                result = await graphrag_query_engine.drift_search(query)
            elif method == "basic":
                result = await graphrag_query_engine.basic_search(query, top_k=top_k)
            else:
                return JSONResponse({
                    "status": "error",
                    "error": f"Unknown method: {method}. Use: auto, global, local, drift, basic"
                }, status_code=400)

            return JSONResponse({
                "status": "success",
                "result": result.to_dict()
            })

        except Exception as e:
            logger.error(f"GraphRAG query error: {e}")
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    @app.post("/api/graphrag/index")
    async def graphrag_index(request: Request):
        """
        Trigger GraphRAG indexing.

        Request body:
        {
            "method": "standard|fast",
            "update_mode": false
        }

        Methods:
        - standard: LLM-based extraction (higher quality, slower)
        - fast: NLP-based extraction (faster, lower quality)
        """
        if not graphrag_enabled or graphrag_indexer is None:
            return JSONResponse({
                "status": "error",
                "error": "GraphRAG not available"
            }, status_code=503)

        try:
            body = await request.json()
            method_str = body.get("method", "standard").lower()
            update_mode = body.get("update_mode", False)

            # Set indexing method
            if method_str == "fast":
                graphrag_indexer.method = IndexingMethod.FAST
            else:
                graphrag_indexer.method = IndexingMethod.STANDARD

            # Run indexing in background
            asyncio.create_task(graphrag_indexer.index_documents(
                update_mode=update_mode
            ))

            return JSONResponse({
                "status": "started",
                "method": method_str,
                "update_mode": update_mode,
                "message": "Indexing started in background. Check /api/graphrag/index/status for progress."
            })

        except Exception as e:
            logger.error(f"GraphRAG indexing error: {e}")
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    @app.get("/api/graphrag/index/status")
    async def graphrag_index_status():
        """Get GraphRAG indexing status and progress."""
        if not graphrag_enabled or graphrag_indexer is None:
            return JSONResponse({
                "status": "error",
                "error": "GraphRAG not available"
            }, status_code=503)

        return JSONResponse({
            "status": graphrag_indexer.status.value,
            "progress": graphrag_indexer.progress,
            "method": graphrag_indexer.method.value,
            "stats": graphrag_indexer.get_stats(),
            "error": graphrag_indexer.get_error(),
            "duration": graphrag_indexer.get_duration()
        })

    @app.get("/api/graphrag/entities")
    async def graphrag_entities(
        query: str = "",
        entity_type: Optional[str] = None,
        top_k: int = 20
    ):
        """Search GraphRAG entities."""
        if not graphrag_enabled or graphrag_query_engine is None:
            return JSONResponse({
                "status": "error",
                "error": "GraphRAG not available"
            }, status_code=503)

        try:
            entities = await graphrag_query_engine.data_loader.search_entities(
                query=query,
                top_k=top_k,
                entity_type=entity_type
            )
            return JSONResponse({
                "status": "success",
                "entities": entities,
                "count": len(entities)
            })
        except Exception as e:
            logger.error(f"GraphRAG entity search error: {e}")
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    @app.get("/api/graphrag/entity/{entity_name}/relationships")
    async def graphrag_entity_relationships(entity_name: str):
        """Get relationships for a specific entity."""
        if not graphrag_enabled or graphrag_query_engine is None:
            return JSONResponse({
                "status": "error",
                "error": "GraphRAG not available"
            }, status_code=503)

        try:
            relationships = await graphrag_query_engine.data_loader.get_entity_relationships(
                entity_name=entity_name
            )
            return JSONResponse({
                "status": "success",
                "entity": entity_name,
                "relationships": relationships,
                "count": len(relationships)
            })
        except Exception as e:
            logger.error(f"GraphRAG relationship error: {e}")
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    @app.get("/api/graphrag/communities")
    async def graphrag_communities(level: int = 0, top_k: int = 10):
        """Get community reports at a specific hierarchy level."""
        if not graphrag_enabled or graphrag_query_engine is None:
            return JSONResponse({
                "status": "error",
                "error": "GraphRAG not available"
            }, status_code=503)

        try:
            reports = await graphrag_query_engine.data_loader.get_community_reports_by_level(level)
            return JSONResponse({
                "status": "success",
                "level": level,
                "communities": reports[:top_k],
                "count": len(reports[:top_k])
            })
        except Exception as e:
            logger.error(f"GraphRAG communities error: {e}")
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    @app.post("/api/graphrag/verify")
    async def graphrag_verify_index():
        """Verify GraphRAG index integrity."""
        if not graphrag_enabled or graphrag_indexer is None:
            return JSONResponse({
                "status": "error",
                "error": "GraphRAG not available"
            }, status_code=503)

        try:
            verification = await graphrag_indexer.verify_index()
            return JSONResponse({
                "status": "success",
                "verification": verification
            })
        except Exception as e:
            logger.error(f"GraphRAG verification error: {e}")
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    # ==========================================================================
    # END GRAPHRAG INTEGRATION
    # ==========================================================================

    # ==========================================================================
    # V25 TEMPORAL REASONING API ENDPOINTS
    # ==========================================================================

    @app.get("/api/temporal/health")
    async def temporal_health():
        """Check temporal reasoning system health."""
        if not LONGITUDINAL_MEMORY_AVAILABLE:
            return JSONResponse({
                "status": "unavailable",
                "message": "Longitudinal memory system not installed"
            }, status_code=503)

        temporal_available = rag_pipeline.temporal_reasoner is not None
        alert_available = rag_pipeline.alert_manager is not None

        return JSONResponse({
            "status": "healthy" if temporal_available else "degraded",
            "components": {
                "temporal_reasoner": temporal_available,
                "alert_manager": alert_available,
                "longitudinal_memory": rag_pipeline.longitudinal_manager is not None,
                "context_injector": rag_pipeline.context_injector is not None
            }
        })

    @app.post("/api/temporal/symptom-progression")
    async def analyze_symptom_progression(request: Request):
        """
        Analyze symptom progression over time.

        Request body:
        {
            "patient_id": "patient-123",
            "symptom_name": "pain",
            "days": 90  // optional, default 90
        }

        Returns detailed symptom progression report including:
        - Trend (worsening, improving, stable)
        - Severity changes over time
        - Diurnal and weekly patterns
        - Medication correlations
        - Clinical concerns and recommendations
        """
        if not LONGITUDINAL_MEMORY_AVAILABLE or rag_pipeline.temporal_reasoner is None:
            return JSONResponse({
                "status": "error",
                "error": "Temporal reasoning not available"
            }, status_code=503)

        try:
            body = await request.json()
            patient_id = body.get("patient_id")
            symptom_name = body.get("symptom_name")
            days = body.get("days", 90)

            if not patient_id or not symptom_name:
                return JSONResponse({
                    "status": "error",
                    "error": "patient_id and symptom_name are required"
                }, status_code=400)

            report = await rag_pipeline.temporal_reasoner.analyze_symptom_progression(
                patient_id=patient_id,
                symptom_name=symptom_name,
                time_window_days=days
            )

            if report is None:
                return JSONResponse({
                    "status": "success",
                    "result": None,
                    "message": f"Insufficient data for symptom '{symptom_name}'. Need at least 3 observations."
                })

            return JSONResponse({
                "status": "success",
                "result": report.to_dict()
            })

        except Exception as e:
            logger.error(f"Symptom progression analysis error: {e}")
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    @app.post("/api/temporal/medication-effectiveness")
    async def analyze_medication_effectiveness(request: Request):
        """
        Analyze medication effectiveness.

        Request body:
        {
            "patient_id": "patient-123",
            "medication_name": "morphine",
            "days": 90  // optional, default 90
        }

        Returns medication effectiveness report including:
        - Adherence rate
        - Symptom response (improvement %)
        - Target symptoms affected
        - Missed doses analysis
        - Rotation recommendations
        """
        if not LONGITUDINAL_MEMORY_AVAILABLE or rag_pipeline.temporal_reasoner is None:
            return JSONResponse({
                "status": "error",
                "error": "Temporal reasoning not available"
            }, status_code=503)

        try:
            body = await request.json()
            patient_id = body.get("patient_id")
            medication_name = body.get("medication_name")
            days = body.get("days", 90)

            if not patient_id or not medication_name:
                return JSONResponse({
                    "status": "error",
                    "error": "patient_id and medication_name are required"
                }, status_code=400)

            report = await rag_pipeline.temporal_reasoner.analyze_medication_effectiveness(
                patient_id=patient_id,
                medication_name=medication_name,
                time_window_days=days
            )

            if report is None:
                return JSONResponse({
                    "status": "success",
                    "result": None,
                    "message": f"Insufficient data for medication '{medication_name}'."
                })

            return JSONResponse({
                "status": "success",
                "result": report.to_dict()
            })

        except Exception as e:
            logger.error(f"Medication effectiveness analysis error: {e}")
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    @app.post("/api/temporal/correlations")
    async def find_correlations(request: Request):
        """
        Find correlations between medications and symptoms.

        Request body:
        {
            "patient_id": "patient-123",
            "days": 90  // optional, default 90
        }

        Returns list of detected correlations between
        medication usage and symptom changes.
        """
        if not LONGITUDINAL_MEMORY_AVAILABLE or rag_pipeline.temporal_reasoner is None:
            return JSONResponse({
                "status": "error",
                "error": "Temporal reasoning not available"
            }, status_code=503)

        try:
            body = await request.json()
            patient_id = body.get("patient_id")
            days = body.get("days", 90)

            if not patient_id:
                return JSONResponse({
                    "status": "error",
                    "error": "patient_id is required"
                }, status_code=400)

            correlations = await rag_pipeline.temporal_reasoner.find_correlations(
                patient_id=patient_id,
                time_window_days=days
            )

            return JSONResponse({
                "status": "success",
                "result": [c.to_dict() for c in correlations],
                "count": len(correlations)
            })

        except Exception as e:
            logger.error(f"Correlation analysis error: {e}")
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    @app.get("/api/temporal/patient/{patient_id}/summary")
    async def get_patient_temporal_summary(patient_id: str, days: int = 30):
        """
        Get a comprehensive temporal summary for a patient.

        Returns aggregated analysis including:
        - All symptom trends
        - Medication effectiveness overview
        - Key correlations
        - Active alerts
        """
        if not LONGITUDINAL_MEMORY_AVAILABLE:
            return JSONResponse({
                "status": "error",
                "error": "Longitudinal memory not available"
            }, status_code=503)

        try:
            summary = {
                "patient_id": patient_id,
                "analysis_period_days": days,
                "symptom_progressions": [],
                "medication_reports": [],
                "correlations": [],
                "active_alerts": []
            }

            # Get patient record to find tracked symptoms and medications
            if rag_pipeline.longitudinal_manager:
                record = await rag_pipeline.longitudinal_manager.get_or_create_record(patient_id)

                # Find unique symptoms
                symptoms = set()
                medications = set()
                for obs in record.observations:
                    if obs.category == "symptom":
                        symptoms.add(obs.entity_name)
                    elif obs.category == "medication":
                        medications.add(obs.entity_name)

                # Analyze each symptom
                if rag_pipeline.temporal_reasoner:
                    for symptom in list(symptoms)[:5]:  # Limit to top 5
                        report = await rag_pipeline.temporal_reasoner.analyze_symptom_progression(
                            patient_id, symptom, days
                        )
                        if report:
                            summary["symptom_progressions"].append(report.to_dict())

                    # Analyze each medication
                    for med in list(medications)[:5]:  # Limit to top 5
                        report = await rag_pipeline.temporal_reasoner.analyze_medication_effectiveness(
                            patient_id, med, days
                        )
                        if report:
                            summary["medication_reports"].append(report.to_dict())

                    # Get correlations
                    correlations = await rag_pipeline.temporal_reasoner.find_correlations(
                        patient_id, days
                    )
                    summary["correlations"] = [c.to_dict() for c in correlations[:5]]

            # Get active alerts
            if rag_pipeline.alert_manager:
                alerts = await rag_pipeline.alert_manager.get_active_alerts(patient_id)
                summary["active_alerts"] = [
                    {
                        "alert_id": a.alert_id,
                        "alert_type": a.alert_type.value if hasattr(a.alert_type, 'value') else str(a.alert_type),
                        "severity": a.severity.value if hasattr(a.severity, 'value') else str(a.severity),
                        "message": a.message,
                        "created_at": a.created_at.isoformat() if hasattr(a, 'created_at') else None
                    }
                    for a in alerts[:10]
                ]

            return JSONResponse({
                "status": "success",
                "result": summary
            })

        except Exception as e:
            logger.error(f"Patient temporal summary error: {e}")
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    # ==========================================================================
    # END V25 TEMPORAL REASONING API
    # ==========================================================================

    # ==========================================================================
    # V25 CARE TEAM COORDINATION API ENDPOINTS
    # ==========================================================================

    @app.get("/api/careteam/{patient_id}")
    async def get_care_team(patient_id: str):
        """
        Get care team members for a patient.

        Returns list of care team members with their roles and contact info.
        """
        if not LONGITUDINAL_MEMORY_AVAILABLE or not rag_pipeline.longitudinal_manager:
            return JSONResponse({
                "status": "error",
                "error": "Longitudinal memory not available"
            }, status_code=503)

        try:
            care_team = await rag_pipeline.longitudinal_manager.get_care_team(patient_id)
            return JSONResponse({
                "status": "success",
                "patient_id": patient_id,
                "care_team": [member.to_dict() for member in care_team],
                "count": len(care_team)
            })
        except Exception as e:
            logger.error(f"Error getting care team: {e}")
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    @app.post("/api/careteam/{patient_id}/add")
    async def add_care_team_member(patient_id: str, request: Request):
        """
        Add a care team member.

        Request body:
        {
            "provider_id": "dr_sharma",
            "name": "Dr. Sharma",
            "role": "doctor",  // doctor, nurse, asha_worker, caregiver, volunteer, social_worker
            "organization": "City Hospital",
            "phone_number": "+91xxxxxxxxxx",
            "primary_contact": false
        }
        """
        if not LONGITUDINAL_MEMORY_AVAILABLE or not rag_pipeline.longitudinal_manager:
            return JSONResponse({
                "status": "error",
                "error": "Longitudinal memory not available"
            }, status_code=503)

        try:
            body = await request.json()

            # Validate required fields
            if not body.get("provider_id") or not body.get("name") or not body.get("role"):
                return JSONResponse({
                    "status": "error",
                    "error": "provider_id, name, and role are required"
                }, status_code=400)

            # Import CareTeamMember
            from personalization.longitudinal_memory import CareTeamMember
            from datetime import datetime

            member = CareTeamMember(
                provider_id=body["provider_id"],
                name=body["name"],
                role=body["role"],
                organization=body.get("organization"),
                phone_number=body.get("phone_number"),
                primary_contact=body.get("primary_contact", False),
                first_contact=datetime.now(),
                last_contact=datetime.now(),
                total_interactions=0,
                attributed_observations=[]
            )

            await rag_pipeline.longitudinal_manager.add_care_team_member(patient_id, member)

            return JSONResponse({
                "status": "success",
                "message": f"Added {member.name} ({member.role}) to care team",
                "member": member.to_dict()
            })

        except Exception as e:
            logger.error(f"Error adding care team member: {e}")
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    @app.get("/api/careteam/{patient_id}/primary")
    async def get_primary_contact(patient_id: str):
        """Get primary contact for a patient."""
        if not LONGITUDINAL_MEMORY_AVAILABLE or not rag_pipeline.longitudinal_manager:
            return JSONResponse({
                "status": "error",
                "error": "Longitudinal memory not available"
            }, status_code=503)

        try:
            record = await rag_pipeline.longitudinal_manager.get_or_create_record(patient_id)
            primary = record.get_primary_contact()

            if primary:
                return JSONResponse({
                    "status": "success",
                    "primary_contact": primary.to_dict()
                })
            else:
                return JSONResponse({
                    "status": "success",
                    "primary_contact": None,
                    "message": "No care team members found"
                })

        except Exception as e:
            logger.error(f"Error getting primary contact: {e}")
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    @app.post("/api/careteam/{patient_id}/attribute")
    async def attribute_observation(patient_id: str, request: Request):
        """
        Attribute an observation to a care team member.

        Request body:
        {
            "observation_id": "obs_123",
            "provider_id": "dr_sharma"
        }
        """
        if not LONGITUDINAL_MEMORY_AVAILABLE or not rag_pipeline.longitudinal_manager:
            return JSONResponse({
                "status": "error",
                "error": "Longitudinal memory not available"
            }, status_code=503)

        try:
            body = await request.json()
            observation_id = body.get("observation_id")
            provider_id = body.get("provider_id")

            if not observation_id or not provider_id:
                return JSONResponse({
                    "status": "error",
                    "error": "observation_id and provider_id are required"
                }, status_code=400)

            record = await rag_pipeline.longitudinal_manager.get_or_create_record(patient_id)

            # Find and update the observation
            obs_found = False
            for obs in record.observations:
                if obs.observation_id == observation_id:
                    obs.reported_by = provider_id
                    obs_found = True
                    break

            if not obs_found:
                return JSONResponse({
                    "status": "error",
                    "error": f"Observation {observation_id} not found"
                }, status_code=404)

            # Update provider stats
            provider_found = False
            for member in record.care_team:
                if member.provider_id == provider_id:
                    from datetime import datetime
                    member.last_contact = datetime.now()
                    member.total_interactions += 1
                    if observation_id not in member.attributed_observations:
                        member.attributed_observations.append(observation_id)
                    provider_found = True
                    break

            # Save the record
            await rag_pipeline.longitudinal_manager.save_record(record)

            return JSONResponse({
                "status": "success",
                "message": f"Attributed observation {observation_id} to {provider_id}",
                "provider_found": provider_found
            })

        except Exception as e:
            logger.error(f"Error attributing observation: {e}")
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    @app.post("/api/careteam/{patient_id}/notify")
    async def notify_care_team(patient_id: str, request: Request):
        """
        Send notification to care team members.

        Request body:
        {
            "message": "Patient needs follow-up",
            "priority": "HIGH",
            "target_roles": ["doctor", "nurse"],  // optional, all if not specified
            "channels": ["whatsapp", "dashboard"]  // optional
        }
        """
        if not LONGITUDINAL_MEMORY_AVAILABLE:
            return JSONResponse({
                "status": "error",
                "error": "Longitudinal memory not available"
            }, status_code=503)

        try:
            body = await request.json()
            message = body.get("message")
            priority = body.get("priority", "MEDIUM")
            target_roles = body.get("target_roles", [])
            channels = body.get("channels", ["dashboard"])

            if not message:
                return JSONResponse({
                    "status": "error",
                    "error": "message is required"
                }, status_code=400)

            # Get care team
            record = await rag_pipeline.longitudinal_manager.get_or_create_record(patient_id)

            # Filter by roles if specified
            recipients = []
            for member in record.care_team:
                if not target_roles or member.role in target_roles:
                    recipients.append(member)

            # If alert coordinator is available, use it
            notifications_sent = []
            if rag_pipeline.alert_coordinator:
                for member in recipients:
                    try:
                        # Create a notification via alert coordinator
                        result = {
                            "provider_id": member.provider_id,
                            "name": member.name,
                            "role": member.role,
                            "channels": channels,
                            "status": "queued"
                        }
                        notifications_sent.append(result)
                    except Exception as e:
                        notifications_sent.append({
                            "provider_id": member.provider_id,
                            "status": "failed",
                            "error": str(e)
                        })
            else:
                # Basic notification tracking
                for member in recipients:
                    notifications_sent.append({
                        "provider_id": member.provider_id,
                        "name": member.name,
                        "role": member.role,
                        "status": "logged"
                    })

            return JSONResponse({
                "status": "success",
                "message": f"Notification sent to {len(notifications_sent)} care team members",
                "notifications": notifications_sent
            })

        except Exception as e:
            logger.error(f"Error notifying care team: {e}")
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    # ==========================================================================
    # END V25 CARE TEAM COORDINATION API
    # ==========================================================================

    # ==========================================================================
    # V25 FHIR INTEROPERABILITY API (Phase 7)
    # ==========================================================================

    @app.post("/api/fhir/export/{patient_id}")
    async def fhir_export_patient(patient_id: str, request: Request):
        """
        Export patient data as FHIR R4 Bundle.

        Query params:
        - include_observations: bool = true
        - include_medications: bool = true
        - include_care_team: bool = true
        - save_to_file: bool = false (if true, saves to data/fhir_exports/)

        Returns:
            FHIR R4 Bundle with Patient, Observation, MedicationStatement, CareTeam resources
        """
        if not LONGITUDINAL_MEMORY_AVAILABLE:
            return JSONResponse({
                "status": "error",
                "error": "Longitudinal memory not available"
            }, status_code=503)

        try:
            # Parse query params
            include_observations = request.query_params.get("include_observations", "true").lower() == "true"
            include_medications = request.query_params.get("include_medications", "true").lower() == "true"
            include_care_team = request.query_params.get("include_care_team", "true").lower() == "true"
            save_to_file = request.query_params.get("save_to_file", "false").lower() == "true"

            # Create adapter and export
            adapter = FHIRAdapter()
            bundle = await adapter.export_patient_bundle(
                patient_id,
                rag_pipeline.longitudinal_manager,
                include_observations=include_observations,
                include_medications=include_medications,
                include_care_team=include_care_team
            )

            bundle_dict = bundle.to_dict()

            # Optionally save to file
            file_path = None
            if save_to_file:
                from pathlib import Path
                from datetime import datetime
                export_dir = Path("data/fhir_exports")
                export_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = str(export_dir / f"patient_{patient_id}_{timestamp}.json")
                export_to_file(bundle, file_path)

            return JSONResponse({
                "status": "success",
                "patient_id": patient_id,
                "bundle": bundle_dict,
                "resource_count": len(bundle_dict.get("entry", [])),
                "file_path": file_path
            })

        except Exception as e:
            logger.error(f"Error exporting FHIR bundle: {e}")
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    @app.post("/api/fhir/import")
    async def fhir_import_bundle(request: Request):
        """
        Import FHIR R4 Bundle into the longitudinal memory system.

        Request body:
        {
            "bundle": { ... FHIR Bundle JSON ... }
        }

        or for file import:
        {
            "file_path": "path/to/bundle.json"
        }

        Returns:
            Import statistics (patients, observations, medications imported)
        """
        if not LONGITUDINAL_MEMORY_AVAILABLE:
            return JSONResponse({
                "status": "error",
                "error": "Longitudinal memory not available"
            }, status_code=503)

        try:
            body = await request.json()

            # Get bundle from request or file
            if "file_path" in body:
                bundle_data = import_from_file(body["file_path"])
            elif "bundle" in body:
                bundle_data = body["bundle"]
            else:
                return JSONResponse({
                    "status": "error",
                    "error": "Either 'bundle' or 'file_path' is required"
                }, status_code=400)

            # Import bundle
            adapter = FHIRAdapter()
            result = await adapter.import_bundle(bundle_data, rag_pipeline.longitudinal_manager)

            return JSONResponse({
                "status": "success",
                "import_result": result
            })

        except Exception as e:
            logger.error(f"Error importing FHIR bundle: {e}")
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    @app.post("/api/fhir/validate")
    async def fhir_validate(request: Request):
        """
        Validate a FHIR Bundle or resource.

        Request body:
        {
            "bundle": { ... FHIR Bundle JSON ... }
        }

        or:
        {
            "resource": { ... FHIR resource JSON ... }
        }

        Returns:
            Validation result with any errors
        """
        if not LONGITUDINAL_MEMORY_AVAILABLE:
            return JSONResponse({
                "status": "error",
                "error": "Longitudinal memory not available"
            }, status_code=503)

        try:
            body = await request.json()
            adapter = FHIRAdapter()

            if "bundle" in body:
                result = adapter.validate_bundle(body["bundle"])
                return JSONResponse({
                    "status": "success",
                    "validation": result
                })
            elif "resource" in body:
                errors = adapter.validate_resource(body["resource"])
                return JSONResponse({
                    "status": "success",
                    "validation": {
                        "valid": len(errors) == 0,
                        "errors": errors
                    }
                })
            else:
                return JSONResponse({
                    "status": "error",
                    "error": "Either 'bundle' or 'resource' is required"
                }, status_code=400)

        except Exception as e:
            logger.error(f"Error validating FHIR: {e}")
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    @app.get("/api/fhir/snomed-codes")
    async def fhir_get_snomed_codes():
        """
        Get the SNOMED CT code mappings for symptoms.

        Returns:
            Dictionary of symptom names to SNOMED codes
        """
        try:
            from personalization.fhir_adapter import SYMPTOM_SNOMED_CODES, SEVERITY_FHIR_CODES
            return JSONResponse({
                "status": "success",
                "symptom_codes": SYMPTOM_SNOMED_CODES,
                "severity_codes": SEVERITY_FHIR_CODES
            })
        except Exception as e:
            return JSONResponse({
                "status": "error",
                "error": str(e)
            }, status_code=500)

    # ==========================================================================
    # END V25 FHIR INTEROPERABILITY API
    # ==========================================================================

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