#!/bin/bash

echo "🚀 Starting RAG Server with Enhanced Logging"
echo "============================================="
echo "📋 Logs will appear below in real-time"
echo "📱 Send your WhatsApp messages now!"
echo "⏹️  Press Ctrl+C to stop the server"
echo ""

cd /Users/tp53/Documents/tp53_AA/llms4palliative_gci/demo_feb2025/rag_gci

# Install dependencies if needed
./run_simple.sh --no-ngrok --port 8002 2>&1

echo ""
echo "🔚 Server stopped"