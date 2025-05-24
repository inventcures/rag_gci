#!/bin/bash

# Smart RAG Server Startup Script
# Automatically detects and starts the appropriate version

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}🚀 Smart RAG Server Startup${NC}"
echo "=================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Parse command line arguments
VERSION=""
VALIDATE=false
FORCE_INSTALL=false
HOST="0.0.0.0"
PORT="8000"

while [[ $# -gt 0 ]]; do
    case $1 in
        --simple)
            VERSION="simple"
            shift
            ;;
        --full)
            VERSION="full"
            shift
            ;;
        --validate)
            VALIDATE=true
            shift
            ;;
        --force-install)
            FORCE_INSTALL=true
            shift
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Version Selection:"
            echo "  --simple           Use simple version (no database)"
            echo "  --full             Use full version (with database)"
            echo ""
            echo "Options:"
            echo "  --validate         Run setup validation first"
            echo "  --force-install    Force reinstall dependencies"
            echo "  --host HOST        Host to bind to (default: 0.0.0.0)"
            echo "  --port PORT        Port to bind to (default: 8000)"
            echo "  --help, -h         Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --simple                    # Start simple version"
            echo "  $0 --full --validate          # Validate then start full version"
            echo "  $0 --simple --force-install   # Reinstall deps and start simple"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Auto-detect version if not specified
if [ -z "$VERSION" ]; then
    echo -e "${BLUE}🔍 Auto-detecting version to use...${NC}"
    
    # Check if user has complex needs
    if [ -f "kotaemon-main/settings.yaml" ] && [ -d "kotaemon-main/libs" ]; then
        echo -e "${YELLOW}📊 Detected Kotaemon setup - recommending full version${NC}"
        VERSION="full"
    else
        echo -e "${GREEN}⚡ Using simple version (faster, no database)${NC}"
        VERSION="simple"
    fi
    
    echo -e "${BLUE}💡 Tip: Use --simple or --full to choose explicitly${NC}"
    echo ""
fi

# Run validation if requested
if [ "$VALIDATE" = true ]; then
    echo -e "${BLUE}🔍 Running setup validation...${NC}"
    if [ -f "validate_setup.py" ]; then
        python validate_setup.py
        validation_result=$?
        
        if [ $validation_result -ne 0 ]; then
            echo -e "${RED}❌ Validation failed. Please fix issues before starting.${NC}"
            exit 1
        fi
        echo -e "${GREEN}✅ Validation passed!${NC}"
        echo ""
    else
        echo -e "${YELLOW}⚠️ Validation script not found, skipping...${NC}"
    fi
fi

# Prepare arguments for the chosen version
SCRIPT_ARGS=""
if [ "$FORCE_INSTALL" = true ]; then
    SCRIPT_ARGS="$SCRIPT_ARGS --force-install"
fi

SCRIPT_ARGS="$SCRIPT_ARGS --host $HOST --port $PORT"

# Start the appropriate version
if [ "$VERSION" = "simple" ]; then
    echo -e "${GREEN}🚀 Starting Simple RAG Server${NC}"
    echo "   • No database required"
    echo "   • File-based storage"
    echo "   • Faster startup"
    echo ""
    
    if [ ! -f "run_simple.sh" ]; then
        echo -e "${RED}❌ run_simple.sh not found${NC}"
        exit 1
    fi
    
    chmod +x run_simple.sh
    exec ./run_simple.sh $SCRIPT_ARGS
    
elif [ "$VERSION" = "full" ]; then
    echo -e "${BLUE}🏢 Starting Full RAG Server${NC}"
    echo "   • Database-backed"
    echo "   • Advanced features"
    echo "   • Full Kotaemon integration"
    echo ""
    
    if [ ! -f "run.sh" ]; then
        echo -e "${RED}❌ run.sh not found${NC}"
        exit 1
    fi
    
    # Check if setup is needed
    if [ ! -f ".env" ] || [ ! -d "venv" ]; then
        echo -e "${YELLOW}🔧 Running initial setup...${NC}"
        python setup.py
    fi
    
    chmod +x run.sh
    exec ./run.sh $SCRIPT_ARGS
    
else
    echo -e "${RED}❌ Invalid version: $VERSION${NC}"
    exit 1
fi