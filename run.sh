#!/bin/bash

# RAG Server Startup Script for macOS
# Handles environment setup and server startup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting RAG Server with WhatsApp Bot${NC}"
echo "=================================================="

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${YELLOW}⚠️ This script is optimized for macOS${NC}"
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python
if ! command_exists python3; then
    echo -e "${RED}❌ Python3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python3 found${NC}"

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}❌ Python $PYTHON_VERSION found, but $REQUIRED_VERSION+ required${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python $PYTHON_VERSION is compatible${NC}"

# Check and install dependencies
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ requirements.txt not found${NC}"
    exit 1
fi

echo -e "${BLUE}📦 Checking dependencies...${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}🏗️ Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${BLUE}🔧 Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install/update dependencies
echo -e "${BLUE}📦 Installing dependencies...${NC}"
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️ .env file not found. Creating template...${NC}"
    python setup.py
fi

# Check for API keys
if [ -f ".env" ]; then
    source .env
    
    if [ -z "$GOOGLE_CLOUD_PROJECT" ] && { [ -z "$GEMINI_API_KEY" ] || [ "$GEMINI_API_KEY" = "your_gemini_api_key_here" ]; }; then
        echo -e "${YELLOW}⚠️ Gemini not configured in .env file${NC}"
        echo "Set GOOGLE_CLOUD_PROJECT (Vertex AI) or GEMINI_API_KEY (https://aistudio.google.com/apikey)"
        echo "Then update .env file"
        echo ""
    fi
fi

# Check if ngrok is installed (optional)
if command_exists ngrok; then
    echo -e "${GREEN}✅ ngrok found - WhatsApp webhooks will work${NC}"
    NGROK_ENABLED=true
else
    echo -e "${YELLOW}⚠️ ngrok not found - WhatsApp webhooks will be local only${NC}"
    echo "Install ngrok for external access: brew install ngrok"
    NGROK_ENABLED=false
fi

# Create necessary directories
echo -e "${BLUE}📁 Creating directories...${NC}"
mkdir -p data uploads logs cache

# Check if Kotaemon is available
if [ ! -d "kotaemon-main" ]; then
    echo -e "${RED}❌ kotaemon-main directory not found${NC}"
    echo "Please ensure the Kotaemon codebase is available"
    exit 1
fi

echo -e "${GREEN}✅ Kotaemon found${NC}"

# Parse command line arguments
HOST="0.0.0.0"
PORT="8000"
NO_NGROK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --no-ngrok)
            NO_NGROK=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --host HOST     Host to bind to (default: 0.0.0.0)"
            echo "  --port PORT     Port to bind to (default: 8000)"
            echo "  --no-ngrok      Disable ngrok tunnel"
            echo "  --help, -h      Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Start the server
echo -e "${GREEN}🚀 Starting RAG Server...${NC}"
echo "Host: $HOST"
echo "Port: $PORT"
echo "ngrok: $([ "$NO_NGROK" = true ] && echo "disabled" || echo "enabled")"
echo ""

# Construct Python command
PYTHON_CMD="python main.py --host $HOST --port $PORT"

if [ "$NO_NGROK" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --no-ngrok"
fi

# Set environment variables for better performance on macOS
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Start the server with error handling
echo -e "${BLUE}🌟 All checks passed! Starting server...${NC}"
echo "=================================================="

if ! $PYTHON_CMD; then
    echo -e "${RED}❌ Server failed to start${NC}"
    echo "Check the logs above for error details"
    exit 1
fi