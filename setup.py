#!/usr/bin/env python3
"""
Setup script for RAG Server
Handles environment setup, dependencies installation, and initial configuration
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import yaml


class RAGServerSetup:
    """Setup handler for RAG Server"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.kotaemon_path = self.project_root / "kotaemon-main"
        self.data_dir = self.project_root / "data"
        self.uploads_dir = self.project_root / "uploads"
        self.logs_dir = self.project_root / "logs"
        
    def run_setup(self):
        """Run complete setup process"""
        print("🚀 Setting up RAG Server...")
        
        try:
            self.create_directories()
            self.check_kotaemon()
            self.install_dependencies()
            self.setup_environment()
            self.setup_kotaemon_config()
            self.create_env_file()
            
            print("\n✅ Setup completed successfully!")
            print("\n📋 Next steps:")
            print("1. Set your API keys in .env file")
            print("2. Run: python rag_server.py")
            print("3. Access admin UI at: http://localhost:8000/admin")
            
        except Exception as e:
            print(f"\n❌ Setup failed: {e}")
            sys.exit(1)
    
    def create_directories(self):
        """Create necessary directories"""
        print("📁 Creating directories...")
        
        directories = [
            self.data_dir,
            self.data_dir / "chroma_db",
            self.data_dir / "documents", 
            self.uploads_dir,
            self.logs_dir,
            self.project_root / "cache"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"   Created: {directory}")
    
    def check_kotaemon(self):
        """Check if Kotaemon is available"""
        print("🔍 Checking Kotaemon...")
        
        if not self.kotaemon_path.exists():
            print("❌ Kotaemon not found. Please ensure kotaemon-main folder exists.")
            raise FileNotFoundError("Kotaemon directory not found")
        
        print(f"   Found Kotaemon at: {self.kotaemon_path}")
    
    def install_dependencies(self):
        """Install Python dependencies"""
        print("📦 Installing dependencies...")
        
        # Install main requirements
        self._run_command([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        
        # Install Kotaemon packages
        kotaemon_lib = self.kotaemon_path / "libs" / "kotaemon"
        ktem_lib = self.kotaemon_path / "libs" / "ktem"
        
        if kotaemon_lib.exists():
            print("   Installing kotaemon...")
            self._run_command([
                sys.executable, "-m", "pip", "install", "-e", str(kotaemon_lib)
            ])
        
        if ktem_lib.exists():
            print("   Installing ktem...")
            self._run_command([
                sys.executable, "-m", "pip", "install", "-e", str(ktem_lib)
            ])
    
    def setup_environment(self):
        """Setup environment variables and paths"""
        print("🔧 Setting up environment...")
        
        # Add kotaemon to Python path
        sys.path.insert(0, str(self.kotaemon_path / "libs"))
        
        print("   Environment configured")
    
    def setup_kotaemon_config(self):
        """Setup Kotaemon configuration"""
        print("⚙️ Setting up Kotaemon configuration...")
        
        # Copy example settings if settings.yaml doesn't exist
        settings_example = self.kotaemon_path / "settings.yaml.example"
        settings_file = self.kotaemon_path / "settings.yaml"
        
        if settings_example.exists() and not settings_file.exists():
            shutil.copy2(settings_example, settings_file)
            print(f"   Copied settings from {settings_example}")
        
        # Update settings for local development
        if settings_file.exists():
            self._update_kotaemon_settings(settings_file)
    
    def _update_kotaemon_settings(self, settings_file: Path):
        """Update Kotaemon settings for local development"""
        try:
            with open(settings_file, 'r') as f:
                settings = yaml.safe_load(f) or {}
            
            # Update key settings for local development
            updates = {
                'data_dir': str(self.data_dir),
                'upload_dir': str(self.uploads_dir),
                'llm': {
                    'provider': 'groq',
                    'model': 'llama-3.1-8b-instant'
                },
                'embedding': {
                    'provider': 'fastembed',
                    'model': 'BAAI/bge-small-en-v1.5'
                },
                'vectorstore': {
                    'provider': 'chroma',
                    'persist_directory': str(self.data_dir / 'chroma_db')
                }
            }
            
            # Deep merge updates
            settings.update(updates)
            
            with open(settings_file, 'w') as f:
                yaml.dump(settings, f, default_flow_style=False)
            
            print("   Updated Kotaemon settings")
            
        except Exception as e:
            print(f"   Warning: Could not update settings: {e}")
    
    def create_env_file(self):
        """Create .env file template"""
        print("🔐 Creating environment file...")
        
        env_file = self.project_root / ".env"
        
        if not env_file.exists():
            env_content = """# ===================================
# FULL RAG SERVER CONFIGURATION
# ===================================

# GROQ API KEY (Required)
# Get from: https://console.groq.com/
# Sign up free, go to API Keys, create new key
GROQ_API_KEY=gsk_your_actual_groq_api_key_here_it_starts_with_gsk

# TWILIO CREDENTIALS (Optional - for WhatsApp bot)
# Get from: https://www.twilio.com/try-twilio
# Sign up free, go to Console Dashboard
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_32_character_auth_token_here
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
PUBLIC_BASE_URL=https://your-ngrok-url.ngrok.io

# DATABASE CONFIGURATION
DATABASE_URL=sqlite:///./data/rag_server.db

# KOTAEMON SETTINGS
KOTAEMON_DATA_DIR=./data
KOTAEMON_UPLOAD_DIR=./uploads

# APPLICATION SETTINGS
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# ADVANCED FEATURES (Optional)
ENABLE_USER_MANAGEMENT=false
ENABLE_ANALYTICS=false
MAX_FILE_SIZE_MB=50
MAX_CHUNKS_PER_DOC=100
"""
            
            with open(env_file, 'w') as f:
                f.write(env_content)
            
            print(f"   Created .env file at: {env_file}")
            print("   📝 Please update with your actual API keys:")
            print("   1. Get Groq API key from: https://console.groq.com/")
            print("   2. Replace 'gsk_your_actual_groq_api_key_here_it_starts_with_gsk' with your key")
            print("   3. (Optional) Add Twilio credentials for WhatsApp bot")
        else:
            print("   .env file already exists")
    
    def _run_command(self, cmd: list):
        """Run shell command"""
        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True,
                cwd=self.project_root
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {' '.join(cmd)}")
            print(f"Error: {e.stderr}")
            raise


def main():
    """Main setup function"""
    setup = RAGServerSetup()
    setup.run_setup()


if __name__ == "__main__":
    main()