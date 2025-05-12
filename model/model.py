#!/usr/bin/env python3
"""
Core model architecture for VST Pitch Perfect Plugin
This defines the neural network models used for vocal transformation
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Bidirectional, Conv1D, 
    Dropout, Activation, BatchNormalization, 
    Concatenate, Reshape, LeakyReLU
)
from tensorflow.keras.models import Model
import numpy as np

class VocalModelConfig:
    """Configuration for vocal models"""
    def __init__(self):
        self.sample_rate = 22050
        self.n_fft = 1024
        self.hop_length = 256
        self.n_mels = 80
        self.f0_min = 80  # Hz
        self.f0_max = 800  # Hz
        self.voiced_threshold = 0.3

def create_voice_transformation_model(config=None):
    """
    Creates a neural network model for voice transformation
    
    This model takes a mel spectrogram as input and outputs:
    1. Enhanced/corrected pitch contour
    2. Transformed mel spectrogram for improved timbre
    """
    if config is None:
        config = VocalModelConfig()
    
    # Input layer - mel spectrogram
    input_mel = Input(shape=(None, config.n_mels), name='mel_input')
    
    # Initial convolutional layers
    x = Conv1D(128, 3, padding='same')(input_mel)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv1D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Bidirectional LSTM layers for sequential processing
    lstm_1 = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Dropout(0.2)(lstm_1)
    lstm_2 = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Dropout(0.2)(lstm_2)
    
    # Skip connections
    x = Concatenate()([x, lstm_1])
    
    # Pitch processing branch
    pitch_branch = Conv1D(256, 3, padding='same')(x)
    pitch_branch = BatchNormalization()(pitch_branch)
    pitch_branch = LeakyReLU(alpha=0.1)(pitch_branch)
    pitch_branch = Conv1D(128, 3, padding='same')(pitch_branch)
    pitch_branch = BatchNormalization()(pitch_branch)
    pitch_branch = LeakyReLU(alpha=0.1)(pitch_branch)
    pitch_branch = Dense(1, name='pitch_output')(pitch_branch)
    
    # Timbre processing branch - This will enhance the voice quality
    timbre_branch = Conv1D(256, 3, padding='same')(x)
    timbre_branch = BatchNormalization()(timbre_branch)
    timbre_branch = LeakyReLU(alpha=0.1)(timbre_branch)
    timbre_branch = Conv1D(128, 3, padding='same')(timbre_branch)
    timbre_branch = BatchNormalization()(timbre_branch)
    timbre_branch = LeakyReLU(alpha=0.1)(timbre_branch)
    timbre_branch = Dense(config.n_mels, name='mel_output')(timbre_branch)
    
    # Create model with multiple outputs
    model = Model(inputs=input_mel, outputs=[pitch_branch, timbre_branch])
    
    return model

def create_voice_enhancement_model(config=None):
    """
    Creates a model specifically for voice enhancement
    
    This model focuses on making the voice sound clearer and more professional,
    without significantly altering its character.
    """
    if config is None:
        config = VocalModelConfig()
    
    # Input layer - raw audio chunks
    input_audio = Input(shape=(None, 1), name='audio_input')
    
    # Processing layers
    x = Conv1D(64, 5, padding='same')(input_audio)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    x = Conv1D(128, 5, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    x = Conv1D(256, 5, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Apply enhancement
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Output enhanced audio
    output = Dense(1, name='enhanced_output')(x)
    
    model = Model(inputs=input_audio, outputs=output)
    
    return model

def export_model_for_inference(model, export_dir):
    """Exports the model in a format suitable for TensorFlow Serving"""
    tf.saved_model.save(model, export_dir)
    print(f"Model exported to: {export_dir}")

def convert_to_tflite(model, output_path):
    """Converts the model to TFLite format for edge deployment"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to: {output_path}")

if __name__ == "__main__":
    # Test model creation
    config = VocalModelConfig()
    
    # Create and inspect the voice transformation model
    transformation_model = create_voice_transformation_model(config)
    transformation_model.summary()
    
    # Create and inspect the voice enhancement model
    enhancement_model = create_voice_enhancement_model(config)
    enhancement_model.summary()
    
    print("Models created successfully!")