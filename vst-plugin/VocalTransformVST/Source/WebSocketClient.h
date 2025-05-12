#pragma once

#include <JuceHeader.h>

/**
 * WebSocket client for real-time communication with the vocal transformation server
 * Optimized for low-latency audio streaming
 */
class WebSocketClient : public juce::WebSocketClient,
                       private juce::Timer
{
public:
    //==============================================================================
    WebSocketClient();
    ~WebSocketClient() override;
    
    //==============================================================================
    // Connection management
    
    /** Connect to the server */
    bool connect(const juce::String& url);
    
    /** Disconnect from the server */
    void disconnect();
    
    /** Check if connected to the server */
    bool isConnected() const;
    
    //==============================================================================
    // Audio data transmission
    
    /** Send audio data to the server for processing */
    bool sendAudioData(const juce::MemoryBlock& audioData);
    
    /** Send audio data to the server for processing (from AudioBuffer) */
    bool sendAudioBuffer(const juce::AudioBuffer<float>& buffer, int numSamples, double sampleRate);
    
    /** Get processed audio data from the server */
    bool getProcessedAudio(juce::MemoryBlock& processedData);
    
    /** Get processed audio data as AudioBuffer */
    bool getProcessedAudioBuffer(juce::AudioBuffer<float>& buffer, int numSamples, double sampleRate);
    
    //==============================================================================
    // Server settings
    
    /** Send effect settings to the server */
    bool sendEffectSettings(float pitchCorrection, float timbre, float enhancement);
    
    /** Get server statistics */
    struct ServerStats {
        int latencyMs;
        bool isProcessing;
        int queueLength;
    };
    
    ServerStats getServerStats() const;
    
    //==============================================================================
    // Callback registration
    
    /** Set a callback function for when connection status changes */
    using ConnectionCallback = std::function<void(bool connected)>;
    void setConnectionCallback(ConnectionCallback callback);
    
    /** Set a callback function for when audio processing is complete */
    using AudioProcessedCallback = std::function<void(const juce::MemoryBlock& audioData)>;
    void setAudioProcessedCallback(AudioProcessedCallback callback);
    
private:
    //==============================================================================
    // WebSocketClient overrides
    void connectionOpened() override;
    void connectionClosed() override;
    void connectionError(const juce::String& errorMessage) override;
    void messageReceived(const juce::String& message) override;
    void dataReceived(const juce::MemoryBlock& data) override;
    
    //==============================================================================
    // Timer callback for monitoring connection
    void timerCallback() override;
    
    //==============================================================================
    // Internal utilities
    
    /** Convert AudioBuffer to WAV format MemoryBlock */
    juce::MemoryBlock audioBufferToWav(const juce::AudioBuffer<float>& buffer, int numSamples, double sampleRate);
    
    /** Convert WAV format MemoryBlock to AudioBuffer */
    bool wavToAudioBuffer(const juce::MemoryBlock& wavData, juce::AudioBuffer<float>& buffer, double sampleRate);
    
    /** Send a ping to measure server latency */
    void sendPing();
    
    //==============================================================================
    // State variables
    std::atomic<bool> connected { false };
    std::atomic<bool> reconnecting { false };
    std::atomic<int> latencyMs { 0 };
    std::atomic<int> queueLength { 0 };
    
    // Data storage
    juce::MemoryBlock receivedData;
    std::atomic<bool> newDataAvailable { false };
    
    // Thread synchronization
    juce::CriticalSection dataMutex;
    juce::CriticalSection connectionMutex;
    
    // Callbacks
    ConnectionCallback connectionCallback;
    AudioProcessedCallback audioProcessedCallback;
    
    // Server URL
    juce::String serverUrl;
    
    // Ping measurement
    juce::Time lastPingTime;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(WebSocketClient)
};