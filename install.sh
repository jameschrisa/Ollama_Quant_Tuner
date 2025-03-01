#!/bin/bash
# Installation script for Ollama Manager

set -e

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Ollama Manager Installer ===${NC}"
echo

# Check Python version
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}Error: Python 3 is required but not found.${NC}"
    echo "Please install Python 3.7 or higher and try again."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version | cut -d " " -f 2)
echo -e "${GREEN}Found Python version ${PYTHON_VERSION}${NC}"

# Check if pip is installed
if ! command -v $PYTHON_CMD -m pip &>/dev/null; then
    echo -e "${RED}Error: pip is required but not found.${NC}"
    echo "Please install pip and try again."
    exit 1
fi

# Check if Ollama is installed
if ! command -v ollama &>/dev/null; then
    echo -e "${YELLOW}Warning: Ollama is not found in your PATH.${NC}"
    echo "You'll need to install Ollama before using Ollama Manager."
    echo "Visit https://ollama.com/download for installation instructions."
    echo
    
    read -p "Continue with installation anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Installation aborted.${NC}"
        exit 1
    fi
fi

# Create virtual environment
echo -e "${BLUE}Creating virtual environment...${NC}"
$PYTHON_CMD -m venv venv

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Unix-like
    source venv/bin/activate
fi

# Install dependencies
echo -e "${BLUE}Installing dependencies...${NC}"
pip install -r requirements.txt

# Make scripts executable
chmod +x ollama_manager.py
chmod +x main.py

# Create desktop entry on Linux
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo -e "${BLUE}Creating desktop entry...${NC}"
    mkdir -p ~/.local/share/applications
    
    cat > ~/.local/share/applications/ollama-manager.desktop << EOF
[Desktop Entry]
Type=Application
Name=Ollama Manager
Comment=Terminal UI for managing Ollama models
Exec=$(pwd)/venv/bin/python $(pwd)/main.py
Icon=terminal
Terminal=true
Categories=Utility;Development;
EOF

    echo -e "${GREEN}Desktop entry created.${NC}"
fi

echo -e "${GREEN}Installation complete!${NC}"
echo
echo -e "To start Ollama Manager, run: ${YELLOW}./main.py${NC}"
echo -e "Or activate the virtual environment and run: ${YELLOW}python main.py${NC}"
echo
echo -e "${BLUE}Happy modeling!${NC}"
