#!/usr/bin/env python3
"""
Audio processor for real-time vocal transformation
Optimized for low-latency, high-quality processing
"""

import os
import io
import time
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
import requests
import logging
from scipy import signal

# Configure logging
logger = logging.getLogger(__name__)

class AudioProcessor:
    """Audio processor for real-time vocal transformation"""
    
    def __init__(self, model_endpoint=None, local_model=None):
        """Initialize audio processor with model connection"""
        self.model_endpoint = model_endpoint
        self.local_model = local_model
        
        # Default configuration
        self.sample_rate = 44100
        self.n_fft = 1024
        self.hop_length = 256
        self.n_mels = 80
        self.f0_min = 65
        self.f0_max = 1000
        
        # Optimization settings
        self.use_overlap_processing = True
        self.overlap_ratio = 0.25  # 25% overlap for smoother transitions
        self.buffer_size = 4096  # Buffer size for processing
        
        # Performance tracking
        self.processing_times = []
        self.feature_extraction_times = []
        self.model_inference_times = []
        self.audio_synthesis_times = []
        
        # Audio effects
        self.effects = {
            'pitch_correction': 0.8,  # 0.0 to 1.0, strength of pitch correction
            'formant_preservation': 0.5,  # 0.0 to 1.0, how much to preserve original formants
            'enhancement': 0.7,  # 0.0 to 1.0, voice enhancement strength
        }
        
        # Initialize audio buffer for overlap processing
        self.prev_buffer = None
        
        logger.info("Audio processor initialized")
    
    async def check_model_connection(self):
        """Check if the model is accessible"""
        if self.local_model is not None:
            return True
        
        try:
            response = requests.get(
                f"{self.model_endpoint.split('/v1')[0]}/v1/models/vocal_transformation_model",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error connecting to model: {e}")
            return False
    
    def update_effects(self, effects_dict):
        """Update audio effects settings"""
        for effect, value in effects_dict.items():
            if effect in self.effects:
                self.effects[effect] = max(0.0, min(1.0, float(value)))
        
        logger.info(f"Updated effects: {self.effects}")
    
    def _extract_features(self, audio):
        """Extract mel spectrogram and other features from audio"""
        start_time = time.time()
        
        # Calculate mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=50,
            fmax=self.sample_rate/2-1000  # Keep within human vocal range
        )
        mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Extract fundamental frequency (for pitch information)
        # Use more efficient algorithm for real-time
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, 
            fmin=self.f0_min,
            fmax=self.f0_max,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            frame_length=self.n_fft,
            fill_na=0.0  # Fill NaN values with 0
        )
        
        # Record timing
        self.feature_extraction_times.append(time.time() - start_time)
        
        return mel_db, f0, voiced_flag
    
    def _call_model(self, mel_input):
        """Call the model for inference"""
        start_time = time.time()
        
        if self.local_model is not None:
            # Use local model
            outputs = self.local_model(mel_input)
            pitch_output = outputs['pitch_output'].numpy()
            mel_output = outputs['mel_output'].numpy()
        else:
            # Call remote model via REST API
            payload = {
                "instances": mel_input.tolist()
            }
            headers = {"content-type": "application/json"}
            
            try:
                response = requests.post(self.model_endpoint, json=payload, headers=headers)
                response.raise_for_status()
                
                prediction = response.json()
                pitch_output = np.array(prediction['predictions'][0])
                mel_output = np.array(prediction['predictions'][1])
            except Exception as e:
                logger.error(f"Error calling model API: {e}")
                # Fallback - return input mel spectrogram
                pitch_output = np.zeros((mel_input.shape[0], mel_input.shape[1], 1))
                mel_output = mel_input
        
        # Record timing
        self.model_inference_times.append(time.time() - start_time)
        
        return pitch_output, mel_output
    
    def _synthesize_audio(self, mel_output, original_f0, voiced_flag, original_audio=None):
        """Synthesize audio from model outputs"""
        start_time = time.time()
        
        # Convert from mel spectrogram to audio
        mel_db = mel_output.T  # Convert to (features, time)
        mel_spec = librosa.db_to_power(mel_db)
        
        # Generate raw audio from mel spectrogram
        y_synth = librosa.feature.inverse.mel_to_audio(
            mel_spec,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=2.0
        )
        
        # Apply pitch correction if available and original audio exists
        pitch_strength = self.effects['pitch_correction']
        if pitch_strength > 0 and original_audio is not None:
            # Extract pitch information from synthesized audio
            synth_f0, synth_voiced, _ = librosa.pyin(
                y_synth,
                fmin=self.f0_min,
                fmax=self.f0_max,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            # Get voiced sections
            voiced_indices = np.where(voiced_flag)[0]
            if len(voiced_indices) > 0:
                # Apply formant preservation
                if self.effects['formant_preservation'] > 0:
                    # Extract formants from original audio
                    # (simplified approach for real-time processing)
                    y_synth = self._apply_formant_preservation(
                        y_synth, original_audio, 
                        strength=self.effects['formant_preservation']
                    )
        
        # Apply enhancement if needed
        if self.effects['enhancement'] > 0:
            y_synth = self._enhance_voice(y_synth, strength=self.effects['enhancement'])
        
        # Record timing
        self.audio_synthesis_times.append(time.time() - start_time)
        
        return y_synth
    
    def _apply_formant_preservation(self, synth_audio, original_audio, strength=0.5):
        """Apply formant preservation to maintain voice characteristics"""
        # Use a simple spectral envelope approach for real-time processing
        
        # Get minimum length
        min_length = min(len(synth_audio), len(original_audio))
        synth_audio = synth_audio[:min_length]
        original_audio = original_audio[:min_length]
        
        # Calculate spectral envelopes
        n_fft = 2048  # Larger FFT size for better frequency resolution
        
        # Original spectral envelope
        S_original = np.abs(librosa.stft(original_audio, n_fft=n_fft))
        envelope_original = np.mean(S_original, axis=1)
        
        # Synthesized spectral envelope
        S_synth = np.abs(librosa.stft(synth_audio, n_fft=n_fft))
        envelope_synth = np.mean(S_synth, axis=1)
        
        # Create transfer function
        transfer = envelope_original / (envelope_synth + 1e-8)
        
        # Apply transfer function with strength factor
        S_corrected = S_synth.copy()
        for i in range(S_synth.shape[1]):
            S_corrected[:, i] = S_synth[:, i] * (transfer ** strength)
        
        # Inverse STFT with original phase
        phase = np.angle(librosa.stft(synth_audio, n_fft=n_fft))
        y_corrected = librosa.istft(S_corrected * np.exp(1j * phase))
        
        # Ensure same length as input
        y_corrected = librosa.util.fix_length(y_corrected, min_length)
        
        # Blend with original synthesized audio based on strength
        y_out = (1 - strength) * synth_audio + strength * y_corrected
        
        return y_out
    
    def _enhance_voice(self, audio, strength=0.7):
        """Apply voice enhancement"""
        # Simple enhancement for real-time processing
        
        # Apply subtle compression
        y_compressed = self._apply_compression(audio, threshold=-20, ratio=4.0)
        
        # Apply subtle EQ - boost presence (3-5 kHz) for clarity
        y_eq = self._apply_eq(y_compressed)
        
        # De-ess if needed (reduce harsh S sounds)
        y_deessed = self._apply_deess(y_eq)
        
        # Blend with original audio based on strength
        y_out = (1 - strength) * audio + strength * y_deessed
        
        return y_out
    
    def _apply_compression(self, audio, threshold=-20, ratio=4.0):
        """Apply simple compression for more consistent levels"""
        # Simple peak-based compression
        gain_reduction = np.zeros_like(audio)
        
        # Calculate gain reduction
        above_thresh = audio > (10 ** (threshold / 20))
        gain_reduction[above_thresh] = (audio[above_thresh] - (10 ** (threshold / 20))) * (1 - 1/ratio)
        
        # Apply gain reduction
        compressed = audio - gain_reduction
        
        # Normalize to original level
        max_original = np.max(np.abs(audio))
        max_compressed = np.max(np.abs(compressed))
        if max_compressed > 0:
            compressed = compressed * (max_original / max_compressed)
        
        return compressed
    
    def _apply_eq(self, audio):
        """Apply EQ for voice enhancement"""
        # Design a simple EQ - boost presence region (3-5 kHz)
        b, a = signal.butter(2, [3000/(self.sample_rate/2), 5000/(self.sample_rate/2)], 'bandpass')
        
        # Apply filter
        presence = signal.lfilter(b, a, audio)
        
        # Boost presence by 3dB
        eq_audio = audio + presence * 0.4
        
        # Normalize to original level
        max_original = np.max(np.abs(audio))
        max_eq = np.max(np.abs(eq_audio))
        if max_eq > 0:
            eq_audio = eq_audio * (max_original / max_eq)
        
        return eq_audio
    
    def _apply_deess(self, audio):
        """Apply de-essing to reduce harsh sibilants"""
        # Design a simple de-esser - reduce high frequencies when they're loud
        b, a = signal.butter(2, 7000/(self.sample_rate/2), 'highpass')
        
        # Extract high frequencies
        high_freq = signal.lfilter(b, a, audio)
        
        # Detect excessive high frequency energy
        high_env = np.abs(high_freq)
        threshold = np.mean(high_env) * 2.0
        
        # Create gain reduction
        reduction = np.ones_like(audio)
        mask = high_env > threshold
        reduction[mask] = threshold / (high_env[mask] + 1e-8)
        
        # Apply reduction as dynamic filter
        deessed = audio.copy()
        deessed[mask] = deessed[mask] * reduction[mask]
        
        return deessed
    
    def process_audio_chunk(self, audio_data):
        """Process an audio chunk for real-time vocal transformation"""
        start_time = time.time()
        
        try:
            # Convert audio data to numpy array
            audio_buffer = io.BytesIO(audio_data)
            audio, sr = sf.read(audio_buffer)
            
            # Ensure mono audio
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample if needed
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                sr = self.sample_rate
            
            # Handle overlap processing for smoother transitions
            if self.use_overlap_processing and self.prev_buffer is not None:
                # Create overlapped audio by concatenating previous buffer tail with current chunk
                overlap_samples = int(len(self.prev_buffer) * self.overlap_ratio)
                full_audio = np.concatenate([self.prev_buffer[-overlap_samples:], audio])
                
                # Save current chunk for next iteration
                self.prev_buffer = audio.copy()
            else:
                full_audio = audio.copy()
                self.prev_buffer = audio.copy()
            
            # Extract features
            mel_db, f0, voiced_flag = self._extract_features(full_audio)
            
            # Prepare model input
            mel_input = mel_db.T  # (time, features)
            mel_input = np.expand_dims(mel_input, axis=0)  # Add batch dimension
            
            # Call model for inference
            pitch_output, mel_output = self._call_model(mel_input)
            
            # Remove batch dimension
            pitch_output = pitch_output[0]
            mel_output = mel_output[0]
            
            # Synthesize audio
            y_synth = self._synthesize_audio(mel_output, f0, voiced_flag, full_audio)
            
            # If we used overlap processing, extract only the current chunk part
            if self.use_overlap_processing and len(y_synth) > len(audio):
                overlap_samples = int(len(self.prev_buffer) * self.overlap_ratio)
                y_synth = y_synth[overlap_samples:]
            
            # Ensure output has same length as input
            if len(y_synth) > len(audio):
                y_synth = y_synth[:len(audio)]
            elif len(y_synth) < len(audio):
                y_synth = np.pad(y_synth, (0, len(audio) - len(y_synth)), mode='constant')
            
            # Convert back to audio data
            output_buffer = io.BytesIO()
            sf.write(output_buffer, y_synth, sr, format='WAV')
            output_buffer.seek(0)
            output_data = output_buffer.read()
            
            # Record processing time
            total_time = time.time() - start_time
            self.processing_times.append(total_time)
            
            # Log latency periodically
            if len(self.processing_times) % 50 == 0:
                avg_time = sum(self.processing_times[-50:]) / 50
                logger.debug(f"Average processing time (last 50): {avg_time:.4f}s, "
                           f"Features: {sum(self.feature_extraction_times[-50:])/50:.4f}s, "
                           f"Inference: {sum(self.model_inference_times[-50:])/50:.4f}s, "
                           f"Synthesis: {sum(self.audio_synthesis_times[-50:])/50:.4f}s")
            
            return output_data
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            # Return original audio data in case of error
            return audio_data

if __name__ == "__main__":
    # Test with dummy data
    test_processor = AudioProcessor()
    
    # Generate dummy audio
    sample_rate = 44100
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Generate a simple sine wave
    frequency = 440  # Hz (A4 note)
    test_audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Convert to bytes
    test_buffer = io.BytesIO()
    sf.write(test_buffer, test_audio, sample_rate, format='WAV')
    test_buffer.seek(0)
    test_data = test_buffer.read()
    
    # Process the audio
    processed_data = test_processor.process_audio_chunk(test_data)
    
    print(f"Input size: {len(test_data)} bytes")
    print(f"Output size: {len(processed_data)} bytes")
    print(f"Processing time: {test_processor.processing_times[0]:.4f}s")