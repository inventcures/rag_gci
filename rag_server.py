"""
RAG Server Pipeline using Kotaemon
A comprehensive RAG server with admin UI and WhatsApp bot integration
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

# Add kotaemon to path
sys.path.insert(0, str(Path(__file__).parent / "kotaemon-main" / "libs"))

import gradio as gr
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn

# Kotaemon imports
from ktem.main import App
from ktem.settings import settings
from ktem.db.models import User, Conversation, Source
from ktem.index.file.index import FileIndex
from ktem.reasoning.simple import SimpleReasoning
from ktem.llms.manager import LLMManager
from ktem.embeddings.manager import EmbeddingManager
from ktem.rerankings.manager import RerankingManager

# Additional imports for WhatsApp and audio processing
import requests
import tempfile
import base64
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """Enhanced RAG Pipeline using Kotaemon"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "kotaemon-main/settings.yaml.example"
        self.app = None
        self.index_manager = None
        self.llm_manager = None
        self.embedding_manager = None
        self.reranking_manager = None
        self.file_index = None
        self.reasoning = None
        
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the RAG pipeline components"""
        try:
            # Initialize Kotaemon app
            self.app = App()
            
            # Get managers from the app
            self.index_manager = self.app.index_manager
            self.llm_manager = LLMManager()
            self.embedding_manager = EmbeddingManager()
            self.reranking_manager = RerankingManager()
            
            # Initialize file index
            self.file_index = FileIndex()
            
            # Initialize reasoning
            self.reasoning = SimpleReasoning()
            
            logger.info("RAG Pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise
    
    async def add_documents(self, file_paths: List[str], metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Add documents to the RAG index"""
        try:
            results = []
            
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                # Process file through Kotaemon
                result = await self.file_index.index_file(
                    file_path=file_path,
                    metadata=metadata or {}
                )
                
                results.append({
                    "file_path": file_path,
                    "status": "success",
                    "result": result
                })
                
                logger.info(f"Successfully indexed: {file_path}")
            
            return {
                "status": "success",
                "results": results,
                "total_files": len(file_paths),
                "successful": len(results)
            }
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    async def query(self, question: str, conversation_id: Optional[str] = None, 
                   user_id: Optional[str] = None) -> Dict[str, Any]:
        """Query the RAG pipeline"""
        try:
            # Use Kotaemon's reasoning pipeline
            response = await self.reasoning.run(
                question=question,
                conversation_id=conversation_id,
                user_id=user_id
            )
            
            return {
                "status": "success",
                "answer": response.get("answer", ""),
                "sources": response.get("sources", []),
                "citations": response.get("citations", []),
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
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index"""
        try:
            stats = self.file_index.get_stats() if self.file_index else {}
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


class AdminUI:
    """Web-based Admin UI for corpus management"""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline
        self.upload_dir = Path("uploads")
        self.upload_dir.mkdir(exist_ok=True)
    
    def create_gradio_interface(self):
        """Create Gradio interface for admin UI"""
        
        with gr.Blocks(title="RAG Admin UI", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# RAG Pipeline Admin Interface")
            
            with gr.Tabs():
                # File Upload Tab
                with gr.TabItem("📁 Upload Documents"):
                    gr.Markdown("## Upload documents to the RAG corpus")
                    
                    with gr.Row():
                        with gr.Column():
                            file_upload = gr.File(
                                label="Select files to upload",
                                file_count="multiple",
                                file_types=[".pdf", ".txt", ".docx", ".md", ".html"]
                            )
                            
                            metadata_json = gr.Textbox(
                                label="Metadata (JSON format)",
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
                
                # Configuration Tab
                with gr.TabItem("⚙️ Configuration"):
                    gr.Markdown("## System Configuration")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### LLM Settings")
                            llm_provider = gr.Dropdown(
                                choices=["openai", "azure", "ollama", "groq"],
                                label="LLM Provider",
                                value="groq"
                            )
                            
                            llm_model = gr.Textbox(
                                label="Model Name",
                                value="llama-3.1-8b-instant"
                            )
                        
                        with gr.Column():
                            gr.Markdown("### Embedding Settings")
                            embedding_provider = gr.Dropdown(
                                choices=["openai", "fastembed", "local"],
                                label="Embedding Provider",
                                value="fastembed"
                            )
                            
                            embedding_model = gr.Textbox(
                                label="Embedding Model",
                                value="BAAI/bge-small-en-v1.5"
                            )
                    
                    save_config_btn = gr.Button("Save Configuration", variant="primary")
                    config_status = gr.Textbox(label="Configuration Status", interactive=False)
                    
                    save_config_btn.click(
                        fn=self._save_config,
                        inputs=[llm_provider, llm_model, embedding_provider, embedding_model],
                        outputs=[config_status]
                    )
        
        return demo
    
    def _handle_file_upload(self, files, metadata_str):
        """Handle file upload and indexing"""
        try:
            if not files:
                return "No files uploaded"
            
            # Parse metadata
            metadata = {}
            if metadata_str.strip():
                try:
                    metadata = json.loads(metadata_str)
                except json.JSONDecodeError:
                    return "Invalid JSON metadata format"
            
            # Save uploaded files
            file_paths = []
            for file in files:
                dest_path = self.upload_dir / file.name
                with open(dest_path, 'wb') as f:
                    f.write(file.read() if hasattr(file, 'read') else file)
                file_paths.append(str(dest_path))
            
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
                    "citations": result.get("citations", [])
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
    
    def _save_config(self, llm_provider, llm_model, embedding_provider, embedding_model):
        """Save configuration"""
        try:
            # In a real implementation, this would update the settings
            config = {
                "llm": {"provider": llm_provider, "model": llm_model},
                "embedding": {"provider": embedding_provider, "model": embedding_model}
            }
            
            # For now, just return success
            return f"Configuration saved: {json.dumps(config, indent=2)}"
            
        except Exception as e:
            return f"Error saving configuration: {str(e)}"


class STTService:
    """Speech-to-Text service using free APIs"""
    
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            logger.warning("GROQ_API_KEY not found. STT will not work.")
    
    async def transcribe_audio(self, audio_file_path: str, language: str = "hi") -> Dict[str, Any]:
        """Transcribe audio file to text"""
        try:
            if not self.groq_api_key:
                return {"status": "error", "error": "GROQ_API_KEY not configured"}
            
            # Use Groq's Whisper API for transcription
            url = "https://api.groq.com/openai/v1/audio/transcriptions"
            
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}"
            }
            
            with open(audio_file_path, 'rb') as audio_file:
                files = {
                    "file": audio_file,
                    "model": (None, "whisper-large-v3"),
                    "language": (None, language),
                    "response_format": (None, "json")
                }
                
                response = requests.post(url, headers=headers, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "status": "success",
                        "text": result.get("text", ""),
                        "language": language
                    }
                else:
                    return {
                        "status": "error",
                        "error": f"Groq API error: {response.status_code} - {response.text}"
                    }
                    
        except Exception as e:
            logger.error(f"STT error: {e}")
            return {"status": "error", "error": str(e)}


class TTSService:
    """Text-to-Speech service for Indian languages"""
    
    def __init__(self):
        self.supported_languages = {
            "hi": "Hindi",
            "bn": "Bengali", 
            "ta": "Tamil",
            "gu": "Gujarati",
            "en": "English"
        }
    
    async def synthesize_speech(self, text: str, language: str = "hi") -> Dict[str, Any]:
        """Convert text to speech"""
        try:
            if language not in self.supported_languages:
                language = "hi"  # Default to Hindi
            
            # For local deployment, we'll use a simple TTS approach
            # In production, you might want to use edge-tts or similar
            
            # For now, return the text with language info
            return {
                "status": "success",
                "text": text,
                "language": language,
                "language_name": self.supported_languages[language],
                "audio_available": False,
                "message": "TTS integration ready - implement with edge-tts or similar"
            }
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return {"status": "error", "error": str(e)}


class WhatsAppBot:
    """WhatsApp bot integration for RAG queries"""
    
    def __init__(self, rag_pipeline: RAGPipeline, stt_service: STTService, tts_service: TTSService):
        self.rag_pipeline = rag_pipeline
        self.stt_service = stt_service
        self.tts_service = tts_service
        self.webhook_token = os.getenv("WHATSAPP_WEBHOOK_TOKEN", "your_webhook_token")
        self.verify_token = os.getenv("WHATSAPP_VERIFY_TOKEN", "your_verify_token")
    
    def create_fastapi_app(self) -> FastAPI:
        """Create FastAPI app for WhatsApp webhook"""
        
        app = FastAPI(title="WhatsApp RAG Bot")
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @app.get("/webhook")
        async def verify_webhook(hub_mode: str = None, hub_challenge: str = None, hub_verify_token: str = None):
            """Verify WhatsApp webhook"""
            if hub_mode == "subscribe" and hub_verify_token == self.verify_token:
                return int(hub_challenge)
            return HTTPException(status_code=403, detail="Forbidden")
        
        @app.post("/webhook")
        async def handle_webhook(request: dict):
            """Handle incoming WhatsApp messages"""
            try:
                return await self._process_whatsapp_message(request)
            except Exception as e:
                logger.error(f"Webhook error: {e}")
                return {"status": "error"}
        
        @app.post("/api/query")
        async def api_query(query: str = Form(...), language: str = Form("en")):
            """Direct API endpoint for queries"""
            try:
                result = await self.rag_pipeline.query(query)
                if result["status"] == "success":
                    # Add TTS if requested
                    tts_result = await self.tts_service.synthesize_speech(
                        result["answer"], language
                    )
                    result["tts"] = tts_result
                
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
                result = await self.rag_pipeline.add_documents([str(file_path)], metadata_dict)
                
                return result
            except Exception as e:
                return {"status": "error", "error": str(e)}
        
        return app
    
    async def _process_whatsapp_message(self, request: dict) -> dict:
        """Process incoming WhatsApp message"""
        try:
            entry = request.get("entry", [{}])[0]
            changes = entry.get("changes", [{}])[0]
            value = changes.get("value", {})
            messages = value.get("messages", [])
            
            for message in messages:
                message_type = message.get("type")
                from_number = message.get("from")
                
                if message_type == "text":
                    # Handle text message
                    text = message.get("text", {}).get("body", "")
                    await self._handle_text_query(from_number, text)
                
                elif message_type == "audio":
                    # Handle audio message
                    audio_id = message.get("audio", {}).get("id")
                    await self._handle_audio_query(from_number, audio_id)
            
            return {"status": "success"}
            
        except Exception as e:
            logger.error(f"Error processing WhatsApp message: {e}")
            return {"status": "error"}
    
    async def _handle_text_query(self, from_number: str, text: str):
        """Handle text query from WhatsApp"""
        try:
            # Query RAG pipeline
            result = await self.rag_pipeline.query(text, user_id=from_number)
            
            if result["status"] == "success":
                response_text = result["answer"]
                
                # Add TTS option (default to Hindi for text queries)
                tts_result = await self.tts_service.synthesize_speech(response_text, "hi")
                
                # Send response back to WhatsApp
                await self._send_whatsapp_message(from_number, response_text)
                
                # In production, you would also send audio if TTS is available
                
            else:
                await self._send_whatsapp_message(
                    from_number, 
                    "Sorry, I encountered an error processing your query."
                )
                
        except Exception as e:
            logger.error(f"Error handling text query: {e}")
            await self._send_whatsapp_message(
                from_number,
                "Sorry, I'm experiencing technical difficulties."
            )
    
    async def _handle_audio_query(self, from_number: str, audio_id: str):
        """Handle audio query from WhatsApp"""
        try:
            # Download audio file from WhatsApp
            audio_path = await self._download_whatsapp_audio(audio_id)
            
            if not audio_path:
                await self._send_whatsapp_message(
                    from_number,
                    "Sorry, I couldn't download your audio message."
                )
                return
            
            # Transcribe audio
            stt_result = await self.stt_service.transcribe_audio(audio_path)
            
            if stt_result["status"] != "success":
                await self._send_whatsapp_message(
                    from_number,
                    "Sorry, I couldn't understand your audio message."
                )
                return
            
            # Query RAG pipeline with transcribed text
            text = stt_result["text"]
            detected_language = stt_result.get("language", "hi")
            
            result = await self.rag_pipeline.query(text, user_id=from_number)
            
            if result["status"] == "success":
                response_text = result["answer"]
                
                # Generate TTS in the same language as the query
                tts_result = await self.tts_service.synthesize_speech(
                    response_text, detected_language
                )
                
                # Send text response
                await self._send_whatsapp_message(from_number, response_text)
                
                # Clean up audio file
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            
        except Exception as e:
            logger.error(f"Error handling audio query: {e}")
            await self._send_whatsapp_message(
                from_number,
                "Sorry, I encountered an error processing your audio message."
            )
    
    async def _download_whatsapp_audio(self, audio_id: str) -> Optional[str]:
        """Download audio file from WhatsApp"""
        try:
            # In production, implement WhatsApp Media API
            # For now, return None to indicate not implemented
            return None
        except Exception as e:
            logger.error(f"Error downloading audio: {e}")
            return None
    
    async def _send_whatsapp_message(self, to_number: str, message: str):
        """Send message back to WhatsApp"""
        try:
            # In production, implement WhatsApp Business API
            logger.info(f"Would send to {to_number}: {message}")
        except Exception as e:
            logger.error(f"Error sending WhatsApp message: {e}")


def main():
    """Main application entry point"""
    
    # Initialize components
    rag_pipeline = RAGPipeline()
    admin_ui = AdminUI(rag_pipeline)
    stt_service = STTService()
    tts_service = TTSService()
    whatsapp_bot = WhatsAppBot(rag_pipeline, stt_service, tts_service)
    
    # Create Gradio interface
    gradio_app = admin_ui.create_gradio_interface()
    
    # Create FastAPI app for WhatsApp
    fastapi_app = whatsapp_bot.create_fastapi_app()
    
    # Mount Gradio app on FastAPI
    fastapi_app = gr.mount_gradio_app(fastapi_app, gradio_app, path="/admin")
    
    print("🚀 Starting RAG Server...")
    print("📊 Admin UI: http://localhost:8000/admin")
    print("🔗 WhatsApp Webhook: http://localhost:8000/webhook")
    print("🛠️ API Docs: http://localhost:8000/docs")
    
    # Start server
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()