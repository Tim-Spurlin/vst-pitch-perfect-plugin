#!/usr/bin/env python3
"""
Evaluate vocal transformation model performance
with metrics for audio quality and latency
"""

import os
import time
import argparse
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import glob
from model import VocalModelConfig

def load_model(model_path):
    """Load trained model from file"""
    print(f"Loading model from {model_path}...")
    return tf.keras.models.load_model(model_path)

def extract_features(audio_path, config):
    """Extract mel spectrogram and pitch features from audio file"""
    print(f"Extracting features from {audio_path}...")
    y, sr = librosa.load(audio_path, sr=config.sample_rate)
    
    # Extract mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=config.n_fft, 
        hop_length=config.hop_length, n_mels=config.n_mels
    )
    mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Extract fundamental frequency (f0)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=config.f0_min, fmax=config.f0_max,
        sr=sr, hop_length=config.hop_length
    )
    
    # Handle NaN values in f0
    f0 = np.nan_to_num(f0)
    
    # Make sure lengths match
    min_length = min(mel_db.shape[1], len(f0))
    mel_db = mel_db[:, :min_length]
    f0 = f0[:min_length]
    
    return {
        'audio': y,
        'sr': sr,
        'mel_spectrogram': mel_db,
        'f0': f0
    }

def evaluate_model(model, features, config):
    """Evaluate model on audio features"""
    # Prepare input data
    mel_input = features['mel_spectrogram'].T  # (time, features)
    mel_input = np.expand_dims(mel_input, axis=0)  # Add batch dimension
    
    # Measure inference time
    start_time = time.time()
    outputs = model.predict(mel_input)
    inference_time = time.time() - start_time
    
    # Extract outputs
    if isinstance(outputs, list):
        pitch_output = outputs[0][0]  # Remove batch dimension
        mel_output = outputs[1][0]  # Remove batch dimension
    else:
        # Handle single output models
        mel_output = outputs[0]
        pitch_output = None
    
    # Calculate metrics
    metrics = {
        'inference_time': inference_time,
        'inference_time_per_second': inference_time / (features['audio'].shape[0] / config.sample_rate)
    }
    
    if pitch_output is not None:
        # Pitch metrics
        pitch_mse = mean_squared_error(features['f0'], pitch_output[:len(features['f0'])])
        metrics['pitch_mse'] = pitch_mse
    
    # Calculate spectral metrics
    if mel_output.shape[0] == features['mel_spectrogram'].shape[1]:
        mel_mse = mean_squared_error(features['mel_spectrogram'].T, mel_output)
        metrics['mel_mse'] = mel_mse
    
    return metrics, pitch_output, mel_output

def synthesize_audio(mel_output, f0_output, config):
    """Synthesize audio from model outputs"""
    # Convert from mel spectrogram to audio
    mel_db = mel_output.T  # Convert back to (features, time)
    mel_spec = librosa.db_to_power(mel_db)
    
    # Generate audio
    y_synth = librosa.feature.inverse.mel_to_audio(
        mel_spec, sr=config.sample_rate, 
        n_fft=config.n_fft, hop_length=config.hop_length
    )
    
    # Apply pitch correction if available
    if f0_output is not None:
        # This is a simplified approach; more sophisticated techniques could be used
        return y_synth
    
    return y_synth

def evaluate_batch(model, audio_dir, output_dir, config):
    """Evaluate model on a batch of audio files"""
    os.makedirs(output_dir, exist_ok=True)
    
    audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return
    
    all_metrics = []
    
    for audio_file in audio_files:
        filename = os.path.basename(audio_file)
        print(f"Processing {filename}...")
        
        # Extract features
        features = extract_features(audio_file, config)
        
        # Evaluate model
        metrics, pitch_output, mel_output = evaluate_model(model, features, config)
        all_metrics.append(metrics)
        
        # Synthesize audio
        y_synth = synthesize_audio(mel_output, pitch_output, config)
        
        # Save synthesized audio
        output_path = os.path.join(output_dir, f"synth_{filename}")
        sf.write(output_path, y_synth, config.sample_rate)
        
        print(f"Metrics for {filename}:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    
    # Calculate average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    print("\nAverage metrics:")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value}")
    
    # Plot latency histogram
    plt.figure(figsize=(10, 6))
    plt.hist([m['inference_time_per_second'] for m in all_metrics], bins=10)
    plt.title('Inference Time per Second of Audio')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'latency_histogram.png'))
    
    return avg_metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate vocal transformation model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model")
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Directory containing audio files for evaluation")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = VocalModelConfig()
    
    # Load model
    model = load_model(args.model_path)
    
    # Evaluate batch
    evaluate_batch(model, args.audio_dir, args.output_dir, config)

if __name__ == "__main__":
    main()