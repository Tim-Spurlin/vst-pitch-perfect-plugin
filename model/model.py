#!/usr/bin/env python3
"""
Core model architecture for VST Pitch Perfect Plugin
This defines neural network models optimized for real-time, high-quality vocal transformation
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Bidirectional, Conv1D, 
    Dropout, BatchNormalization, Concatenate, LeakyReLU,
    TimeDistributed, Add, Activation, LayerNormalization
)
from tensorflow.keras.models import Model
import numpy as np

class VocalModelConfig:
    """Configuration for vocal models with optimized parameters"""
    def __init__(self):
        # Audio parameters
        self.sample_rate = 44100  # Higher sample rate for better quality
        self.n_fft = 1024
        self.hop_length = 256
        self.n_mels = 80
        self.f0_min = 65  # Hz - Extended range for deeper voices
        self.f0_max = 1000  # Hz - Extended range for higher frequencies
        self.voiced_threshold = 0.3
        
        # Model parameters for real-time performance
        self.use_lightweight_layers = True
        self.use_quantization = True
        self.use_attention = True

def residual_block(x, filters, kernel_size=3, dilation_rate=1):
    """Residual block for efficient processing with skip connections"""
    res = x
    
    # First convolution path
    y = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(x)
    y = BatchNormalization()(y)
    y = LeakyReLU(alpha=0.1)(y)
    
    # Second convolution path
    y = Conv1D(filters, kernel_size, padding='same')(y)
    y = BatchNormalization()(y)
    
    # Skip connection
    if x.shape[-1] != filters:
        res = Conv1D(filters, 1, padding='same')(x)
    
    # Add and activate
    y = Add()([res, y])
    y = LeakyReLU(alpha=0.1)(y)
    
    return y

def create_voice_transformation_model(config=None):
    """
    Creates a neural network model for voice transformation
    
    Optimized for high-quality with minimal latency:
    1. Enhanced/corrected pitch contour
    2. Transformed mel spectrogram for improved timbre
    3. Voice characteristic enhancement
    """
    if config is None:
        config = VocalModelConfig()
    
    # Input layer - mel spectrogram
    input_mel = Input(shape=(None, config.n_mels), name='mel_input')
    
    # Initial convolutional layers with dilated convolutions for broader context
    x = Conv1D(128, 3, padding='same')(input_mel)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Residual blocks with increasing dilation rates for broader context without increasing computation
    x = residual_block(x, 128, dilation_rate=1)
    x = residual_block(x, 128, dilation_rate=2)
    x = residual_block(x, 256, dilation_rate=4)
    x = residual_block(x, 256, dilation_rate=8)
    
    # Bidirectional LSTM layers with optimized units for real-time processing
    # Layer normalization is used for faster convergence and better generalization
    lstm_1 = Bidirectional(LSTM(192, return_sequences=True))(x)
    lstm_1 = LayerNormalization()(lstm_1)
    x = Dropout(0.1)(lstm_1)  # Lower dropout for faster inference
    
    lstm_2 = Bidirectional(LSTM(192, return_sequences=True))(x)
    lstm_2 = LayerNormalization()(lstm_2)
    x = Dropout(0.1)(x)
    
    # Skip connections for preserving information
    x = Concatenate()([x, lstm_1])
    
    # === Pitch processing branch ===
    pitch_branch = Conv1D(192, 3, padding='same')(x)
    pitch_branch = BatchNormalization()(pitch_branch)
    pitch_branch = LeakyReLU(alpha=0.1)(pitch_branch)
    pitch_branch = Conv1D(96, 3, padding='same')(pitch_branch)
    pitch_branch = BatchNormalization()(pitch_branch)
    pitch_branch = LeakyReLU(alpha=0.1)(pitch_branch)
    
    # Final pitch output - single value per time step
    pitch_branch = Dense(1, name='pitch_output')(pitch_branch)
    
    # === Timbre processing branch ===
    timbre_branch = Conv1D(192, 3, padding='same')(x)
    timbre_branch = BatchNormalization()(timbre_branch)
    timbre_branch = LeakyReLU(alpha=0.1)(timbre_branch)
    timbre_branch = Conv1D(96, 3, padding='same')(timbre_branch)
    timbre_branch = BatchNormalization()(timbre_branch)
    timbre_branch = LeakyReLU(alpha=0.1)(timbre_branch)
    
    # Final mel output - reconstructed/enhanced mel spectrogram
    timbre_branch = Dense(config.n_mels, name='mel_output')(timbre_branch)
    
    # Create model with multiple outputs
    model = Model(inputs=input_mel, outputs=[pitch_branch, timbre_branch])
    
    return model

def create_voice_enhancement_model(config=None):
    """
    Creates a model specifically for voice enhancement
    
    This model focuses on enhancing voice quality for professional sound,
    with optimizations for real-time processing.
    """
    if config is None:
        config = VocalModelConfig()
    
    # Input layer - raw audio chunks
    input_audio = Input(shape=(None, 1), name='audio_input')
    
    # Processing layers with residual connections for efficient processing
    x = Conv1D(64, 5, padding='same')(input_audio)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Series of residual blocks with different dilation rates
    x = residual_block(x, 64, kernel_size=5, dilation_rate=1)
    x = residual_block(x, 128, kernel_size=5, dilation_rate=2)
    x = residual_block(x, 128, kernel_size=3, dilation_rate=4)
    
    # Efficient sequential processing
    x = Bidirectional(LSTM(96, return_sequences=True))(x)
    x = LayerNormalization()(x)
    
    # Final processing for enhanced output
    x = Conv1D(128, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(64)(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Output enhanced audio
    output = Dense(1, name='enhanced_output')(x)
    
    model = Model(inputs=input_audio, outputs=output)
    
    return model

def export_model_for_inference(model, export_dir):
    """Exports the model in a format suitable for TensorFlow Serving"""
    # Optimize model for inference
    model_for_export = tf.function(lambda x: model(x))
    concrete_function = model_for_export.get_concrete_function(
        tf.TensorSpec([None, None, model.inputs[0].shape[-1]], 
                     model.inputs[0].dtype))
    
    # Save with optimization options
    options = tf.saved_model.SaveOptions(
        experimental_custom_gradients=False,
        function_aliases={
            "serving_default": concrete_function,
        }
    )
    
    tf.saved_model.save(
        model, export_dir, 
        signatures=concrete_function,
        options=options
    )
    print(f"Model exported to: {export_dir}")

def convert_to_tflite(model, output_path, quantize=True):
    """
    Converts the model to TFLite format for edge deployment
    with optimizations for speed and memory footprint
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        # Enable quantization for faster inference on edge devices
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Enable additional optimizations
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
    
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to: {output_path}")
    print(f"Model size: {len(tflite_model) / (1024 * 1024):.2f} MB")

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