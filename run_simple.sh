#!/bin/bash

# Simple RAG Server Startup Script - No Database Required
# Lightweight version with minimal dependencies

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Simple RAG Server (No Database)${NC}"
echo "=================================================="

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

# Check and install dependencies
echo -e "${BLUE}📦 Checking dependencies...${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}🏗️ Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${BLUE}🔧 Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip only if needed
if ! python -m pip list | grep -q "pip.*$(python -m pip --version | cut -d' ' -f2)" 2>/dev/null; then
    echo -e "${BLUE}⬆️ Upgrading pip...${NC}"
    python -m pip install --upgrade pip
fi

# Check if dependencies are installed
echo -e "${BLUE}📦 Checking dependencies...${NC}"

# Function to check if a package is installed
check_package() {
    python -c "import $1" 2>/dev/null
    return $?
}

# Critical packages that must be present
CRITICAL_PACKAGES=("fastapi" "gradio" "chromadb" "sentence_transformers" "requests" "twilio")
MISSING_PACKAGES=()

# Check each critical package (unless force install is requested)
if [ "$FORCE_INSTALL" = true ]; then
    echo -e "${YELLOW}🔧 Force install requested - reinstalling all dependencies${NC}"
    pip install -r requirements_simple.txt
else
    for package in "${CRITICAL_PACKAGES[@]}"; do
        if ! check_package "$package"; then
            MISSING_PACKAGES+=("$package")
        fi
    done
    
    # Only install if packages are missing
    if [ ${#MISSING_PACKAGES[@]} -eq 0 ]; then
        echo -e "${GREEN}✅ All dependencies are already installed${NC}"
    else
        echo -e "${YELLOW}⚙️ Installing missing packages: ${MISSING_PACKAGES[*]}${NC}"
        pip install -r requirements_simple.txt
        
        # Verify installation
        echo -e "${BLUE}🔍 Verifying installation...${NC}"
        STILL_MISSING=()
        for package in "${MISSING_PACKAGES[@]}"; do
            if ! check_package "$package"; then
                STILL_MISSING+=("$package")
            fi
        done
        
        if [ ${#STILL_MISSING[@]} -eq 0 ]; then
            echo -e "${GREEN}✅ All packages installed successfully${NC}"
        else
            echo -e "${RED}❌ Failed to install: ${STILL_MISSING[*]}${NC}"
            exit 1
        fi
    fi
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️ .env file not found. Creating template...${NC}"
    cat > .env << 'EOF'
# ===================================
# SIMPLE RAG SERVER CONFIGURATION
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
PUBLIC_BASE_URL=http://localhost:8000

# APPLICATION SETTINGS
ENVIRONMENT=development
DEBUG=true
EOF
    
    echo -e "${YELLOW}📝 Created .env template. Please update with your actual API keys:${NC}"
    echo "   1. Get Groq API key from: https://console.groq.com/"
    echo "   2. Replace 'gsk_your_actual_groq_api_key_here_it_starts_with_gsk' with your key"
    echo "   3. (Optional) Add Twilio credentials for WhatsApp bot"
    echo ""
fi

# Check for API keys
if [ -f ".env" ]; then
    source .env
    
    if [ -z "$GROQ_API_KEY" ] || [ "$GROQ_API_KEY" = "your_groq_api_key_here" ]; then
        echo -e "${YELLOW}⚠️ GROQ_API_KEY not set in .env file${NC}"
        echo "Get your free API key from: https://groq.com/"
        echo "Then update .env file with your key"
        echo ""
    fi
fi

# Create necessary directories
echo -e "${BLUE}📁 Creating directories...${NC}"
mkdir -p data uploads cache

# Parse command line arguments
HOST="0.0.0.0"
PORT="8000"
FORCE_INSTALL=false
SCRIPT_ARGS=""

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
        --force-install)
            FORCE_INSTALL=true
            shift
            ;;
        --no-ngrok)
            SCRIPT_ARGS="$SCRIPT_ARGS --no-ngrok"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --host HOST        Host to bind to (default: 0.0.0.0)"
            echo "  --port PORT        Port to bind to (default: 8000)"
            echo "  --force-install    Force reinstall all dependencies"
            echo "  --no-ngrok         Disable ngrok tunnel (WhatsApp won't work externally)"
            echo "  --help, -h         Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Ask for model preference
echo -e "${BLUE}📋 Model Selection${NC}"
echo "Do you want to use MedGemma for English response generation? (Y/y for MedGemma, Enter for default Gemma)"
read -r USE_MEDGEMMA

if [[ "$USE_MEDGEMMA" =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}✅ Using MedGemma model for English responses${NC}"
    export USE_MEDGEMMA=true
else
    echo -e "${GREEN}✅ Using default Gemma model for responses${NC}"
    export USE_MEDGEMMA=false
fi

# Start the simple server
echo -e "${GREEN}🚀 Starting Simple RAG Server...${NC}"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Database: File-based (no SQL required)"
echo "Model: $([ "$USE_MEDGEMMA" = true ] && echo "MedGemma for English responses" || echo "Gemma for all responses")"
echo ""

# Set environment variables for better performance on macOS
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Start the server
echo -e "${BLUE}🌟 All checks passed! Starting simple server...${NC}"
echo "=================================================="

python simple_rag_server.py --host $HOST --port $PORT $SCRIPT_ARGS || {
    echo -e "${RED}❌ Server failed to start${NC}"
    echo "Check the logs above for error details"
    exit 1
}