#!/usr/bin/env python3
"""
Main Application - Complete RAG Server with Admin UI and WhatsApp Bot
Integrates all components for local deployment with ngrok support
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional
import signal
import subprocess
import time

# Setup environment
sys.path.insert(0, str(Path(__file__).parent / "kotaemon-main" / "libs"))

import uvicorn
import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import requests

# Import our components
from rag_server import RAGPipeline, AdminUI
from whatsapp_bot import EnhancedWhatsAppBot, EnhancedSTTService, EnhancedTTSService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/main.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RAGApplication:
    """Main application orchestrator"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.rag_pipeline = None
        self.admin_ui = None
        self.whatsapp_bot = None
        self.app = None
        self.ngrok_process = None
        self.ngrok_url = None
        
        # Services
        self.stt_service = None
        self.tts_service = None
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            "data", "uploads", "logs", "cache", 
            "cache/tts", "cache/downloads"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """Initialize all components"""
        try:
            logger.info("🚀 Initializing RAG Application...")
            
            # Initialize core services
            logger.info("📊 Initializing RAG Pipeline...")
            self.rag_pipeline = RAGPipeline(self.config_path)
            
            logger.info("🎤 Initializing STT Service...")
            self.stt_service = EnhancedSTTService()
            
            logger.info("🔊 Initializing TTS Service...")
            self.tts_service = EnhancedTTSService()
            
            logger.info("🖥️ Initializing Admin UI...")
            self.admin_ui = AdminUI(self.rag_pipeline)
            
            logger.info("📱 Initializing WhatsApp Bot...")
            self.whatsapp_bot = EnhancedWhatsAppBot(
                self.rag_pipeline,
                self.stt_service,
                self.tts_service
            )
            
            # Create main FastAPI app
            self.app = self._create_main_app()
            
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise
    
    def _create_main_app(self) -> FastAPI:
        """Create main FastAPI application"""
        
        # Create main app
        app = FastAPI(
            title="RAG Server with WhatsApp Bot",
            description="Complete RAG pipeline with admin UI and WhatsApp integration",
            version="1.0.0"
        )
        
        # Add CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Mount static files
        if Path("static").exists():
            app.mount("/static", StaticFiles(directory="static"), name="static")
        
        # Add main routes
        @app.get("/")
        async def root():
            """Redirect to admin UI"""
            return RedirectResponse(url="/admin")
        
        @app.get("/health")
        async def health():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "services": {
                    "rag_pipeline": "running",
                    "admin_ui": "running",
                    "whatsapp_bot": "running",
                    "stt_service": "running",
                    "tts_service": "running"
                },
                "ngrok_url": self.ngrok_url
            }
        
        @app.get("/api/info")
        async def api_info():
            """API information"""
            return {
                "name": "RAG Server API",
                "version": "1.0.0",
                "endpoints": {
                    "admin_ui": "/admin",
                    "whatsapp_webhook": "/webhook",
                    "health": "/health",
                    "api_docs": "/docs"
                },
                "supported_languages": {
                    "stt": list(self.stt_service.supported_languages.keys()),
                    "tts": list(self.tts_service.supported_voices.keys())
                }
            }
        
        # Mount WhatsApp webhook routes
        whatsapp_app = self.whatsapp_bot.create_webhook_app()
        
        # Copy WhatsApp routes to main app
        for route in whatsapp_app.routes:
            if hasattr(route, 'path') and route.path.startswith('/'):
                app.router.routes.append(route)
        
        # Mount Gradio admin UI
        gradio_app = self.admin_ui.create_gradio_interface()
        app = gr.mount_gradio_app(app, gradio_app, path="/admin")
        
        return app
    
    def start_ngrok(self, port: int = 8000) -> Optional[str]:
        """Start ngrok tunnel for external access"""
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
            
            # Wait for ngrok to start and get URL
            time.sleep(3)
            
            try:
                # Get ngrok tunnels
                response = requests.get("http://127.0.0.1:4040/api/tunnels")
                if response.status_code == 200:
                    tunnels = response.json().get("tunnels", [])
                    for tunnel in tunnels:
                        if tunnel.get("proto") == "https":
                            self.ngrok_url = tunnel.get("public_url")
                            logger.info(f"🌐 ngrok tunnel started: {self.ngrok_url}")
                            logger.info(f"📱 WhatsApp webhook URL: {self.ngrok_url}/webhook")
                            return self.ngrok_url
            except:
                pass
            
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
    
    async def run(self, host: str = "0.0.0.0", port: int = 8000, start_ngrok: bool = True):
        """Run the application"""
        try:
            # Initialize components
            await self.initialize()
            
            # Start ngrok if requested
            if start_ngrok:
                self.start_ngrok(port)
            
            # Print startup information
            self._print_startup_info(host, port)
            
            # Setup signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Start server
            config = uvicorn.Config(
                app=self.app,
                host=host,
                port=port,
                log_level="info",
                access_log=True
            )
            
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            logger.error(f"❌ Application failed: {e}")
            raise
        finally:
            self.cleanup()
    
    def _print_startup_info(self, host: str, port: int):
        """Print startup information"""
        print("\n" + "="*60)
        print("🚀 RAG SERVER WITH WHATSAPP BOT")
        print("="*60)
        print(f"📊 Admin UI: http://localhost:{port}/admin")
        print(f"🔗 API Docs: http://localhost:{port}/docs")
        print(f"💚 Health Check: http://localhost:{port}/health")
        print(f"📱 WhatsApp Webhook: http://localhost:{port}/webhook")
        
        if self.ngrok_url:
            print(f"🌐 Public URL: {self.ngrok_url}")
            print(f"📱 Public Webhook: {self.ngrok_url}/webhook")
        
        print("\n📋 Setup Instructions:")
        print("1. Set API keys in .env file")
        print("2. Upload documents via Admin UI")
        print("3. Configure WhatsApp webhook (if using)")
        print("4. Test queries via Admin UI or WhatsApp")
        
        print("\n🔧 Supported Features:")
        print("• Document upload and indexing")
        print("• Text and voice queries")
        print("• Multi-language support (Hindi, Bengali, Tamil, Gujarati)")
        print("• WhatsApp bot integration")
        print("• Real-time speech synthesis")
        
        print("\n🌍 Supported Languages:")
        print("• Hindi (हिंदी)")
        print("• Bengali (বাংলা)")
        print("• Tamil (தமிழ்)")
        print("• Gujarati (ગુજરાતી)")
        print("• English")
        
        print("\n" + "="*60)
        print("✅ Server ready! Press Ctrl+C to stop")
        print("="*60 + "\n")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("🛑 Received shutdown signal. Cleaning up...")
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up resources...")
        
        # Stop ngrok
        self.stop_ngrok()
        
        # Clean up cache files
        try:
            cache_dir = Path("cache")
            if cache_dir.exists():
                for file in cache_dir.rglob("*"):
                    if file.is_file() and file.stat().st_size > 0:
                        # Keep recent files, remove old ones
                        if time.time() - file.stat().st_mtime > 3600:  # 1 hour
                            file.unlink()
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")
        
        logger.info("✅ Cleanup completed")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Server with WhatsApp Bot")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--no-ngrok", action="store_true", help="Disable ngrok tunnel")
    parser.add_argument("--config", help="Path to config file")
    
    args = parser.parse_args()
    
    # Create application
    app = RAGApplication(args.config)
    
    # Run application
    try:
        asyncio.run(app.run(
            host=args.host,
            port=args.port,
            start_ngrok=not args.no_ngrok
        ))
    except KeyboardInterrupt:
        logger.info("👋 Goodbye!")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()