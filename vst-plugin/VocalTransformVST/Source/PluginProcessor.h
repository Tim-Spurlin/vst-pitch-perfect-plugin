#pragma once

#include <JuceHeader.h>
#include "WebSocketClient.h"

// Parameter constants for more readable code
namespace ParamIDs {
    const juce::String DryWet = "dryWet";
    const juce::String PitchCorrection = "pitchCorrection";
    const juce::String Timbre = "timbre";
    const juce::String Enhancement = "enhancement";
    const juce::String Latency = "latency";
    const juce::String ServerURL = "serverURL";
}

//==============================================================================
/**
    Main audio processor for the VST Pitch Perfect plugin
    Handles real-time audio processing with cloud-based vocal transformation
*/
class VocalTransformAudioProcessor  : public juce::AudioProcessor,
                                      private juce::Timer
{
public:
    //==============================================================================
    VocalTransformAudioProcessor();
    ~VocalTransformAudioProcessor() override;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;

    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;

    //==============================================================================
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    //==============================================================================
    // Custom methods for VST Pitch Perfect plugin
    
    // Connect to server
    void connectToServer(const juce::String& url);
    
    // Check connection status
    bool isConnected() const;
    
    // Access parameter tree
    juce::AudioProcessorValueTreeState& getParameters() { return parameters; }
    
    // Set bypass status
    void setBypass(bool shouldBypass);
    
    // Get latency information
    float getLatencyMs() const;
    
    // Get processing statistics
    struct ProcessingStats {
        float cpuUsage;
        int buffersProcessed;
        int serverLatencyMs;
        bool isConnected;
        bool isProcessing;
    };
    
    ProcessingStats getStats() const;

private:
    //==============================================================================
    // Timer callback for monitoring performance
    void timerCallback() override;
    
    // Audio processing utilities
    void processAudioBlock(juce::AudioBuffer<float>& buffer);
    void processWithServer(juce::AudioBuffer<float>& buffer);
    void applyDryWetMix(juce::AudioBuffer<float>& dryBuffer, juce::AudioBuffer<float>& wetBuffer);
    
    // Buffer management
    void resizeBuffers(int numSamples);
    
    // Parameter handling
    void initializeParameters();
    void updateParametersFromState();
    
    // Update server settings
    void updateServerSettings();
    
    //==============================================================================
    // WebSocket client for cloud communication
    std::unique_ptr<WebSocketClient> webSocketClient;
    
    // Parameter state
    juce::AudioProcessorValueTreeState parameters;
    
    // Atomic parameters for thread-safe access
    std::atomic<float>* dryWetParam = nullptr;
    std::atomic<float>* pitchCorrectionParam = nullptr;
    std::atomic<float>* timbreParam = nullptr;
    std::atomic<float>* enhancementParam = nullptr;
    std::atomic<float>* latencyParam = nullptr;
    
    // Server URL parameter
    std::atomic<juce::String*> serverURLParam;
    
    // Internal buffers for processing
    juce::AudioBuffer<float> inputBuffer;
    juce::AudioBuffer<float> outputBuffer;
    juce::AudioBuffer<float> dryBuffer;
    
    // Ring buffer for input/output
    juce::AudioSampleBuffer ringBuffer;
    int ringBufferWritePosition = 0;
    int ringBufferReadPosition = 0;
    
    // Processing state
    std::atomic<bool> bypass;
    std::atomic<bool> connected;
    std::atomic<bool> processing;
    std::atomic<float> lastLatencyMs;
    std::atomic<int> buffersProcessed;
    std::atomic<float> cpuUsage;
    
    // Synchronization
    juce::CriticalSection processLock;
    
    // Constants
    const int maxBufferSize = 8192;
    const int defaultLatencyMs = 20;
    const juce::String defaultServerURL = "ws://localhost:8000/ws";
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (VocalTransformAudioProcessor)
};