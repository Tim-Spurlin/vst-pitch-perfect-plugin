#!/bin/bash
# Build script for VST Pitch Perfect Plugin

# Text color variables
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Set build directory
BUILD_DIR="build"
RELEASE_DIR="../../../build"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Ensure we're in the right directory
cd "$SCRIPT_DIR"

# Function to check if a command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}Error: $1 is required but not installed.${NC}"
        return 1
    fi
    return 0
}

# Check requirements
echo -e "${YELLOW}Checking build requirements...${NC}"
check_command cmake || { echo -e "${RED}Please install CMake from https://cmake.org${NC}"; exit 1; }
check_command git || { echo -e "${RED}Please install Git.${NC}"; exit 1; }

# Check for JUCE
if [ ! -d "JUCE" ]; then
    echo -e "${YELLOW}JUCE framework not found. Cloning from GitHub...${NC}"
    git clone --depth 1 https://github.com/juce-framework/JUCE.git
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to clone JUCE.${NC}"
        exit 1
    fi
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo -e "${YELLOW}Configuring build with CMake...${NC}"
cmake .. -DCMAKE_BUILD_TYPE=Release

if [ $? -ne 0 ]; then
    echo -e "${RED}CMake configuration failed.${NC}"
    exit 1
fi

# Determine number of CPU cores for parallel build
if [ "$(uname)" == "Darwin" ]; then
    # macOS
    CORES=$(sysctl -n hw.logicalcpu)
elif [ "$(uname)" == "Linux" ]; then
    # Linux
    CORES=$(nproc)
else
    # Default to 2 cores
    CORES=2
fi

# Build
echo -e "${YELLOW}Building VST Pitch Perfect Plugin using $CORES cores...${NC}"
cmake --build . --config Release -- -j$CORES

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed.${NC}"
    exit 1
fi

# Create release directories
mkdir -p "$RELEASE_DIR/VST3"
mkdir -p "$RELEASE_DIR/AU"
mkdir -p "$RELEASE_DIR/Standalone"

# Copy built plugins to release directory
echo -e "${YELLOW}Copying built plugins to release directory...${NC}"

# Determine platform and copy accordingly
if [ "$(uname)" == "Darwin" ]; then
    # macOS
    cp -R VocalTransformVST_artefacts/Release/VST3/VST\ Pitch\ Perfect.vst3 "$RELEASE_DIR/VST3/" 2>/dev/null || :
    cp -R VocalTransformVST_artefacts/Release/AU/VST\ Pitch\ Perfect.component "$RELEASE_DIR/AU/" 2>/dev/null || :
    cp -R VocalTransformVST_artefacts/Release/Standalone/VST\ Pitch\ Perfect.app "$RELEASE_DIR/Standalone/" 2>/dev/null || :
elif [ "$(uname)" == "Linux" ]; then
    # Linux
    cp -R VocalTransformVST_artefacts/VST3/VST\ Pitch\ Perfect.vst3 "$RELEASE_DIR/VST3/" 2>/dev/null || :
    cp -R VocalTransformVST_artefacts/Standalone/VST\ Pitch\ Perfect "$RELEASE_DIR/Standalone/" 2>/dev/null || :
else
    # Windows (assuming running in Git Bash or similar)
    cp -R VocalTransformVST_artefacts/Release/VST3/"VST Pitch Perfect.vst3" "$RELEASE_DIR/VST3/" 2>/dev/null || :
    cp -R VocalTransformVST_artefacts/Release/Standalone/"VST Pitch Perfect.exe" "$RELEASE_DIR/Standalone/" 2>/dev/null || :
fi

# Check if any files were copied
if [ -z "$(ls -A "$RELEASE_DIR/VST3" 2>/dev/null)" ] && 
   [ -z "$(ls -A "$RELEASE_DIR/AU" 2>/dev/null)" ] &&
   [ -z "$(ls -A "$RELEASE_DIR/Standalone" 2>/dev/null)" ]; then
    echo -e "${YELLOW}Warning: No plugin files were copied to the release directory.${NC}"
    echo -e "${YELLOW}Check the build artifacts in: ${SCRIPT_DIR}/${BUILD_DIR}/VocalTransformVST_artefacts/${NC}"
else
    echo -e "${GREEN}Plugin files copied to: ${RELEASE_DIR}${NC}"
fi

echo -e "${GREEN}Build completed successfully!${NC}"
cd "$SCRIPT_DIR"

# Print summary
echo -e "\n${GREEN}=== VST Pitch Perfect Plugin Build Summary ===${NC}"
echo -e "Source directory: ${YELLOW}${SCRIPT_DIR}${NC}"
echo -e "Build directory:  ${YELLOW}${SCRIPT_DIR}/${BUILD_DIR}${NC}"
echo -e "Release directory:${YELLOW}${RELEASE_DIR}${NC}"
echo -e "\nThe plugin is now ready to use in your DAW."
echo -e "\n${YELLOW}Note: To use the plugin with the cloud processing backend, make sure the server is running.${NC}"