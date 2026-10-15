#!/bin/bash

# Study server startup script
# Optionally validates the setup, then hands over to run_simple.sh, which starts
# simple_rag_server.py: the study server. The Kotaemon "full" server was removed
# (it had no Dose Boundary and no study release), so there is no other version.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}🚀 Study Server Startup${NC}"
echo "=================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Parse command line arguments
VALIDATE=false
FORCE_INSTALL=false
HOST="0.0.0.0"
PORT="8000"

while [[ $# -gt 0 ]]; do
    case $1 in
        --simple)
            # Kept so existing invocations still work; it is the only version.
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
            echo "Options:"
            echo "  --validate         Run setup validation first"
            echo "  --force-install    Force reinstall dependencies"
            echo "  --host HOST        Host to bind to (default: 0.0.0.0)"
            echo "  --port PORT        Port to bind to (default: 8000)"
            echo "  --help, -h         Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                     # Start the study server"
            echo "  $0 --validate          # Validate, then start the study server"
            echo "  $0 --force-install     # Reinstall deps, then start"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

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

# Prepare arguments for run_simple.sh
SCRIPT_ARGS=""
if [ "$FORCE_INSTALL" = true ]; then
    SCRIPT_ARGS="$SCRIPT_ARGS --force-install"
fi

SCRIPT_ARGS="$SCRIPT_ARGS --host $HOST --port $PORT"

# Start the study server
echo -e "${GREEN}🚀 Starting the study server (simple_rag_server.py)${NC}"
echo ""

if [ ! -f "run_simple.sh" ]; then
    echo -e "${RED}❌ run_simple.sh not found${NC}"
    exit 1
fi

chmod +x run_simple.sh
exec ./run_simple.sh $SCRIPT_ARGS
