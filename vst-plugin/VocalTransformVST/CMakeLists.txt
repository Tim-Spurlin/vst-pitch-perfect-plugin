cmake_minimum_required(VERSION 3.15)

project(VocalTransformVST VERSION 1.0.0)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find JUCE
find_package(JUCE CONFIG REQUIRED)

# Define the VST plugin
juce_add_plugin(VocalTransformVST
    PLUGIN_MANUFACTURER_CODE VPrf
    PLUGIN_CODE VTPf
    FORMATS VST3 AU Standalone
    PRODUCT_NAME "VST Pitch Perfect"
    COMPANY_NAME "Tim Spurlin"
    BUNDLE_ID "com.timspurlin.vst-pitch-perfect"
    HARDENED_RUNTIME_ENABLED TRUE # For macOS
    APP_SANDBOX_ENABLED TRUE # For macOS
    VERSION "1.0.0"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS TRUE
    COPY_PLUGIN_AFTER_BUILD TRUE
    VST3_COPY_DIR "${CMAKE_SOURCE_DIR}/../../build/VST3"
    AU_COPY_DIR "${CMAKE_SOURCE_DIR}/../../build/AU"
    PLUGIN_MANUFACTURER_CODE Vspp
    PLUGIN_CODE PfPl
)

# Add source files
target_sources(VocalTransformVST PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
    Source/WebSocketClient.cpp
)

# Add public header files
target_include_directories(VocalTransformVST PUBLIC
    Source
)

# Define preprocessor definitions
target_compile_definitions(VocalTransformVST PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=1
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_DISPLAY_SPLASH_SCREEN=0
    JUCE_REPORT_APP_USAGE=0
    JUCE_STRICT_REFCOUNTEDPOINTER=1
)

# Link against JUCE modules
target_link_libraries(VocalTransformVST PRIVATE
    juce::juce_audio_utils
    juce::juce_audio_processors
    juce::juce_cryptography
    juce::juce_core
    juce::juce_data_structures
    juce::juce_events
    juce::juce_graphics
    juce::juce_gui_basics
    juce::juce_gui_extra
    juce::juce_dsp
    juce::juce_osc
)

# Enable compiler warnings
if(MSVC)
    target_compile_options(VocalTransformVST PRIVATE /W4)
else()
    target_compile_options(VocalTransformVST PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Set optimization flags for Release build
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        target_compile_options(VocalTransformVST PRIVATE /O2)
    else()
        target_compile_options(VocalTransformVST PRIVATE -O3)
    endif()
endif()

# Generate JuceHeader.h
juce_generate_juce_header(VocalTransformVST)

# Add version info
set_target_properties(VocalTransformVST PROPERTIES
    JUCE_COMPANY_COPYRIGHT "Copyright (c) 2025 Tim Spurlin. All rights reserved."
)