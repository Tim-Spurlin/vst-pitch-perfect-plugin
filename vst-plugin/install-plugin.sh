#!/bin/bash
# Install script for VST Pitch Perfect Plugin

# Text color variables
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Set source directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SOURCE_DIR="${SCRIPT_DIR}/../build"

# Function to display usage
function show_usage {
    echo -e "${BLUE}VST Pitch Perfect Plugin - Installer${NC}"
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -s, --source DIR  Source directory (default: $SOURCE_DIR)"
    echo "  -h, --help        Show this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -s|--source)
        SOURCE_DIR="$2"
        shift 2
        ;;
        -h|--help)
        show_usage
        exit 0
        ;;
        *)
        echo -e "${RED}Unknown option: $1${NC}"
        show_usage
        exit 1
        ;;
    esac
done

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo -e "${RED}Error: Source directory $SOURCE_DIR does not exist.${NC}"
    echo -e "${YELLOW}Have you built the plugin using './VocalTransformVST/build.sh'?${NC}"
    exit 1
fi

# Determine OS and set installation paths
if [ "$(uname)" == "Darwin" ]; then
    # macOS
    VST3_DIR="$HOME/Library/Audio/Plug-Ins/VST3"
    AU_DIR="$HOME/Library/Audio/Plug-Ins/Components"
    STANDALONE_DIR="$HOME/Applications"
    
    echo -e "${YELLOW}Installing for macOS...${NC}"
    
    # Create directories if they don't exist
    mkdir -p "$VST3_DIR"
    mkdir -p "$AU_DIR"
    mkdir -p "$STANDALONE_DIR"
    
    # Copy VST3
    if [ -d "$SOURCE_DIR/VST3/VST Pitch Perfect.vst3" ]; then
        echo -e "Installing VST3 plugin..."
        cp -R "$SOURCE_DIR/VST3/VST Pitch Perfect.vst3" "$VST3_DIR/"
        echo -e "${GREEN}VST3 plugin installed to $VST3_DIR${NC}"
    fi
    
    # Copy AU
    if [ -d "$SOURCE_DIR/AU/VST Pitch Perfect.component" ]; then
        echo -e "Installing AU plugin..."
        cp -R "$SOURCE_DIR/AU/VST Pitch Perfect.component" "$AU_DIR/"
        echo -e "${GREEN}AU plugin installed to $AU_DIR${NC}"
    fi
    
    # Copy Standalone
    if [ -d "$SOURCE_DIR/Standalone/VST Pitch Perfect.app" ]; then
        echo -e "Installing standalone application..."
        cp -R "$SOURCE_DIR/Standalone/VST Pitch Perfect.app" "$STANDALONE_DIR/"
        echo -e "${GREEN}Standalone application installed to $STANDALONE_DIR${NC}"
    fi
    
elif [ "$(uname)" == "Linux" ]; then
    # Linux
    VST3_DIR="$HOME/.vst3"
    STANDALONE_DIR="$HOME/.local/bin"
    
    echo -e "${YELLOW}Installing for Linux...${NC}"
    
    # Create directories if they don't exist
    mkdir -p "$VST3_DIR"
    mkdir -p "$STANDALONE_DIR"
    
    # Check if we're on Kali Linux
    if grep -q "Kali" /etc/os-release 2>/dev/null; then
        echo -e "${BLUE}Kali Linux detected.${NC}"
    fi
    
    # Copy VST3
    if [ -d "$SOURCE_DIR/VST3/VST Pitch Perfect.vst3" ]; then
        echo -e "Installing VST3 plugin..."
        cp -R "$SOURCE_DIR/VST3/VST Pitch Perfect.vst3" "$VST3_DIR/"
        echo -e "${GREEN}VST3 plugin installed to $VST3_DIR${NC}"
    fi
    
    # Copy Standalone
    if [ -f "$SOURCE_DIR/Standalone/VST Pitch Perfect" ]; then
        echo -e "Installing standalone application..."
        cp "$SOURCE_DIR/Standalone/VST Pitch Perfect" "$STANDALONE_DIR/"
        chmod +x "$STANDALONE_DIR/VST Pitch Perfect"
        echo -e "${GREEN}Standalone application installed to $STANDALONE_DIR${NC}"
    fi
    
else
    # Windows
    VST3_DIR="$USERPROFILE\Documents\VST3"
    STANDALONE_DIR="$USERPROFILE\Documents\VST Pitch Perfect"
    
    echo -e "${YELLOW}Installing for Windows...${NC}"
    
    # Create directories if they don't exist
    mkdir -p "$VST3_DIR"
    mkdir -p "$STANDALONE_DIR"
    
    # Copy VST3
    if [ -d "$SOURCE_DIR/VST3/VST Pitch Perfect.vst3" ]; then
        echo -e "Installing VST3 plugin..."
        cp -R "$SOURCE_DIR/VST3/VST Pitch Perfect.vst3" "$VST3_DIR/"
        echo -e "${GREEN}VST3 plugin installed to $VST3_DIR${NC}"
    fi
    
    # Copy Standalone
    if [ -f "$SOURCE_DIR/Standalone/VST Pitch Perfect.exe" ]; then
        echo -e "Installing standalone application..."
        cp "$SOURCE_DIR/Standalone/VST Pitch Perfect.exe" "$STANDALONE_DIR/"
        echo -e "${GREEN}Standalone application installed to $STANDALONE_DIR${NC}"
    fi
fi

echo -e "\n${GREEN}Installation complete!${NC}"
echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Restart your DAW to detect the newly installed plugin"
echo "2. Make sure the VST Pitch Perfect server is running"
echo "3. Configure the plugin to connect to your server"
echo -e "\nEnjoy your revolutionary vocal transformation plugin!"