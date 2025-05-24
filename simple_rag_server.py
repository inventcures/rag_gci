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

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Core RAG components
import chromadb
from chromadb.utils import embedding_functions
import requests
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
            
            text = self._extract_text(file_path)
            
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
                "chunk_count": len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _extract_text(self, file_path: Path) -> str:
        """Extract text from file based on extension"""
        
        if file_path.suffix.lower() in {'.txt', '.md'}:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif file_path.suffix.lower() == '.pdf':
            text = ""
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        
        elif file_path.suffix.lower() == '.docx':
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        
        else:
            return ""
    
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


class SimpleRAGPipeline:
    """Simplified RAG Pipeline without database dependencies"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Storage paths
        self.vector_db_path = self.data_dir / "chroma_db"
        self.metadata_file = self.data_dir / "document_metadata.json"
        self.conversation_file = self.data_dir / "conversations.json"
        
        # Initialize components
        self.document_processor = SimpleDocumentProcessor()
        self.embedding_model = None
        self.vector_db = None
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        # Document metadata and conversation storage
        self.document_metadata = self._load_metadata()
        self.conversations = self._load_conversations()
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize embedding model and vector database"""
        try:
            # Initialize embedding model
            logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
            
            # Initialize ChromaDB
            logger.info("Initializing vector database...")
            chroma_client = chromadb.PersistentClient(path=str(self.vector_db_path))
            
            # Create or get collection
            self.vector_db = chroma_client.get_or_create_collection(
                name="documents",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="BAAI/bge-small-en-v1.5"
                )
            )
            
            logger.info("RAG Pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise
    
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
    
    async def add_documents(self, file_paths: List[str], metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Add documents to the RAG index"""
        try:
            results = []
            
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                # Process document
                doc_result = self.document_processor.process_file(file_path)
                
                if doc_result["status"] != "success":
                    results.append({
                        "file_path": file_path,
                        "status": "error",
                        "error": doc_result["error"]
                    })
                    continue
                
                # Generate document ID
                doc_id = hashlib.md5(file_path.encode()).hexdigest()
                
                # Prepare chunks for indexing
                chunks = doc_result["chunks"]
                chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
                
                # Create metadata for each chunk
                chunk_metadata = []
                for i, chunk in enumerate(chunks):
                    meta = {
                        "file_path": file_path,
                        "filename": Path(file_path).name,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "doc_id": doc_id,
                        **(metadata or {})
                    }
                    chunk_metadata.append(meta)
                
                # Add to vector database
                self.vector_db.add(
                    documents=chunks,
                    metadatas=chunk_metadata,
                    ids=chunk_ids
                )
                
                # Store document metadata
                self.document_metadata[doc_id] = {
                    "file_path": file_path,
                    "filename": Path(file_path).name,
                    "chunk_count": len(chunks),
                    "metadata": metadata or {},
                    "indexed_at": datetime.now().isoformat()
                }
                
                results.append({
                    "file_path": file_path,
                    "status": "success",
                    "doc_id": doc_id,
                    "chunks": len(chunks)
                })
                
                logger.info(f"Successfully indexed: {file_path} ({len(chunks)} chunks)")
            
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
                   user_id: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
        """Query the RAG pipeline"""
        try:
            # Retrieve relevant documents
            search_results = self.vector_db.query(
                query_texts=[question],
                n_results=top_k
            )
            
            if not search_results['documents'] or not search_results['documents'][0]:
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
                if distance <= relevance_threshold:
                    relevant_contexts.append(context)
                    relevant_metadatas.append(meta)
            
            # If no relevant contexts found
            if not relevant_contexts:
                return {
                    "status": "success",
                    "answer": "We are afraid, we could not find the answer to your query in our medical corpus. Please consult a qualified medical doctor or visit your nearest hospital, with your query.",
                    "sources": [],
                    "conversation_id": conversation_id
                }
            
            context_text = "\n\n".join([
                f"Source: {meta['filename']} (chunk {meta['chunk_index']+1})\n{doc}"
                for doc, meta in zip(relevant_contexts, relevant_metadatas)
            ])
            
            # Generate answer using Groq with citation instructions
            answer = await self._generate_answer_with_citations(question, context_text, relevant_metadatas)
            
            # Check if the answer indicates it's not in the corpus
            if self._is_no_answer_response(answer):
                return {
                    "status": "success",
                    "answer": "We are afraid, we could not find the answer to your query in our medical corpus. Please consult a qualified medical doctor or visit your nearest hospital, with your query.",
                    "sources": [],
                    "conversation_id": conversation_id
                }
            
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
    
    async def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using Groq API"""
        try:
            if not self.groq_api_key:
                return "Error: GROQ_API_KEY not configured"
            
            prompt = f"""You are an expert medical assistant with strong analytical reasoning. Analyze the provided context and answer the question with careful reasoning.

INSTRUCTIONS:
1. Carefully examine the medical context provided
2. If the context contains relevant information, reason through it step-by-step
3. Provide a clear, medically accurate answer
4. If the context lacks sufficient information, clearly state this

MEDICAL CONTEXT:
{context}

QUESTION: {question}

REASONING AND ANSWER:
Let me analyze this step-by-step:"""
            
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "qwen-qwq-32b",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,  # Lower temperature for more consistent medical responses
                "max_tokens": 2048   # Increased for Qwen's detailed reasoning
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
    
    async def _generate_answer_with_citations(self, question: str, context: str, metadatas: List[Dict]) -> str:
        """Generate answer with citations using Groq API"""
        try:
            if not self.groq_api_key:
                return "Error: GROQ_API_KEY not configured"
            
            # Create citation mapping
            citation_text = self._format_citation_context(context, metadatas)
            
            prompt = f"""You are an expert medical assistant with strong reasoning capabilities. Your task is to analyze medical documents and provide accurate, well-reasoned answers.

REASONING PROCESS:
1. First, carefully analyze the provided medical context
2. Determine if the context contains sufficient information to answer the question
3. If insufficient, respond with: "INSUFFICIENT_INFORMATION"
4. If sufficient, reason through the medical concepts step-by-step
5. Provide a comprehensive but concise medical explanation
6. Always conclude with a proper citation

CITATION REQUIREMENTS:
- End your response with: {{retrieved from: [Document Name], [Medical Section], page [number]}}
- Use the source information provided below

MEDICAL CONTEXT:
{citation_text}

QUESTION: {question}

REASONING AND ANSWER:
Let me analyze the medical context and provide a reasoned response:"""
            
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "qwen-qwq-32b",
                "messages": [
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.2,  # Very low temperature for consistent medical responses
                "max_tokens": 2048   # Increased for Qwen's detailed reasoning and citations
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
                
                # If no citation was added by the model, add one automatically
                if not self._has_citation(answer) and not self._is_no_answer_response(answer):
                    answer = self._add_automatic_citation(answer, metadatas)
                
                return answer
            else:
                logger.error(f"Groq API error: {response.status_code} - {response.text}")
                return f"Error generating response: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error generating answer with citations: {e}")
            return f"Error generating answer: {str(e)}"
    
    def _format_citation_context(self, context: str, metadatas: List[Dict]) -> str:
        """Format context with clear source indicators for citation"""
        sections = []
        for i, meta in enumerate(metadatas):
            filename = meta['filename'].replace('.pdf', '').replace('_', ' ').replace('-', ' ')
            chunk_num = meta['chunk_index'] + 1
            total_chunks = meta['total_chunks']
            
            # Estimate page number (rough calculation)
            estimated_page = max(1, int((chunk_num / total_chunks) * 100))  # Assume ~100 page document
            
            sections.append(f"[Document: {filename}, Chunk {chunk_num}, ~Page {estimated_page}]")
        
        return context + "\n\nAvailable sources: " + "; ".join(sections)
    
    def _has_citation(self, answer: str) -> bool:
        """Check if answer already has a citation in curly braces"""
        return '{' in answer and '}' in answer and 'retrieved from' in answer.lower()
    
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
    
    def _add_automatic_citation(self, answer: str, metadatas: List[Dict]) -> str:
        """Add automatic citation if the model didn't include one"""
        if not metadatas:
            return answer
        
        # Use the first metadata for citation
        meta = metadatas[0]
        filename = meta['filename'].replace('.pdf', '').replace('_', ' ').replace('-', ' ')
        chunk_num = meta['chunk_index'] + 1
        total_chunks = meta['total_chunks']
        
        # Estimate page number
        estimated_page = max(1, int((chunk_num / total_chunks) * 100))
        
        # Try to infer section from content or use generic term
        section = "Medical Information"  # Generic fallback
        
        citation = f" {{retrieved from: {filename}, {section}, page {estimated_page}}}"
        
        return answer + citation
    
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
            
            with gr.Tabs():
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
                            
                            upload_btn = gr.Button("Upload & Index", variant="primary")
                        
                        with gr.Column():
                            upload_status = gr.Textbox(
                                label="Upload Status",
                                lines=10,
                                interactive=False
                            )
                    
                    upload_btn.click(
                        fn=self._handle_file_upload,
                        inputs=[file_upload, metadata_json],
                        outputs=[upload_status]
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
                            
                            query_btn = gr.Button("Submit Query", variant="primary")
                        
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
        
        return demo
    
    def _handle_file_upload(self, files, metadata_str):
        """Handle file upload and indexing"""
        try:
            if not files:
                return "No files uploaded"
            
            # Parse metadata
            metadata = {}
            if metadata_str and metadata_str.strip():
                try:
                    metadata = json.loads(metadata_str)
                except json.JSONDecodeError:
                    return "Invalid JSON metadata format"
            
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
                            
                            # Option 1: Use file directly (fastest, no copy needed)
                            # This works fine since we're only reading the file
                            file_paths.append(str(source_path))
                            logger.info(f"Using file directly: {source_path}")
                            
                            # Option 2: Copy to uploads directory (uncomment if you want permanent storage)
                            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            # unique_id = str(uuid.uuid4())[:8]
                            # dest_filename = f"{timestamp}_{unique_id}_{source_path.name}"
                            # dest_path = self.upload_dir / dest_filename
                            # 
                            # logger.info(f"Copying {source_path} to {dest_path}")
                            # shutil.copy2(source_path, dest_path)
                            # logger.info(f"Copy successful, file size: {dest_path.stat().st_size} bytes")
                            # file_paths.append(str(dest_path))
                            
                        else:
                            return f"File not found: {file}"
                    
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
                        
                        # Create permanent copy if needed (currently commented out)
                        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        # unique_id = str(uuid.uuid4())[:8]
                        # dest_filename = f"{timestamp}_{unique_id}_{filename}"
                        # dest_path = self.upload_dir / dest_filename
                        # 
                        # if hasattr(file, 'name') and Path(file.name).exists():
                        #     shutil.copy2(file.name, dest_path)
                        # elif hasattr(file, 'read'):
                        #     with open(dest_path, 'wb') as f:
                        #         f.write(file.read())
                        # else:
                        #     return f"Cannot read file: {file}"
                        # 
                        # file_paths.append(str(dest_path))
                    
                    else:
                        return f"Unsupported file type: {type(file)} - {file}"
                        
                except Exception as file_error:
                    logger.error(f"Error processing file {file}: {str(file_error)}")
                    return f"Error processing file {file}: {str(file_error)}\n{traceback.format_exc()}"
            
            # Index files
            result = asyncio.run(self.rag_pipeline.add_documents(file_paths, metadata))
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error: {str(e)}\n{traceback.format_exc()}"
    
    def _handle_query(self, query):
        """Handle query testing"""
        try:
            if not query.strip():
                return "Please enter a query", {}
            
            result = asyncio.run(self.rag_pipeline.query(query))
            
            if result["status"] == "success":
                return result["answer"], {
                    "sources": result.get("sources", []),
                    "context_used": result.get("context_used", 0)
                }
            else:
                return f"Error: {result['error']}", {}
                
        except Exception as e:
            return f"Error: {str(e)}", {}
    
    def _get_stats(self):
        """Get index statistics"""
        try:
            return self.rag_pipeline.get_index_stats()
        except Exception as e:
            return {"error": str(e)}


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
    
    @app.get("/")
    async def root():
        return {"message": "Simple RAG Server - No Database Required!"}
    
    @app.get("/health")
    async def health():
        stats = rag_pipeline.get_index_stats()
        return {
            "status": "healthy",
            "database": "file-based (no SQL database)",
            "whatsapp_bot": "configured" if whatsapp_bot else "not configured",
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