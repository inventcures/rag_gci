#!/usr/bin/env python3
"""
Quick launcher for Gemini Live Web UI demo
"""

import os
import sys
import webbrowser
import time
from pathlib import Path

# Check environment
print("="*70)
print("Palli Sahayak - Gemini Live Demo Launcher")
print("="*70)
print()

# Check if .env exists
env_file = Path(".env")
if env_file.exists():
    print("✓ .env file found")
    # Load it manually
    with open(env_file) as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value
else:
    print("⚠ .env file not found")

# Check GEMINI_API_KEY
gemini_key = os.getenv("GEMINI_API_KEY", "")
if gemini_key:
    print(f"✓ GEMINI_API_KEY: {'*'*10}{gemini_key[-4:]}")
else:
    print("✗ GEMINI_API_KEY not set!")
    print("  Get key from: https://makersuite.google.com/app/apikey")
    sys.exit(1)

print()
print("Starting FastAPI server...")
print("Once started, open: http://localhost:8000/voice")
print()

# Start server
os.system("python3 simple_rag_server.py")
