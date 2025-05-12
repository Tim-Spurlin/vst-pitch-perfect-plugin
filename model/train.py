#!/usr/bin/env python3
"""
Training script for VST Pitch Perfect voice transformation model.
This script handles loading data, training, and saving models.
"""

import os
import numpy as np
import tensorflow as tf
import glob
import json
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from model import create_voice_transformation_model, VocalModelConfig, export_model_for_inference

class DataPreprocessor:
    def __init__(self, config=None):
        if config is None:
            self.config = VocalModelConfig()
        else:
            self.config = config
    
    def extract_features(self, audio_path):
        """Extract mel spectrograms and pitch features from audio file"""
        # Load audio file
        y, sr = librosa.load(audio_path, sr=self.config.sample_rate)
        
        # Extract mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_fft=self.config.n_fft, 
            hop_length=self.config.hop_length, 
            n_mels=self.config.n_mels
        )
        mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Extract fundamental frequency (f0)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=self.config.f0_min,
            fmax=self.config.f0_max,
            sr=sr,
            hop_length=self.config.hop_length
        )
        
        # Handle NaN values in f0
        f0 = np.nan_to_num(f0)
        
        # Make sure lengths match
        min_length = min(mel_db.shape[1], len(f0))
        mel_db = mel_db[:, :min_length]
        f0 = f0[:min_length]
        
        return {
            'mel_spectrogram': mel_db.T,  # Transpose to (time, features)
            'f0': f0,
            'audio_path': audio_path
        }
    
    def process_dataset(self, audio_dir, features_dir=None, limit=None):
        """Process an entire dataset of audio files"""
        if features_dir:
            os.makedirs(features_dir, exist_ok=True)
        
        audio_files = glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True)
        
        if limit:
            audio_files = audio_files[:limit]
        
        features_list = []
        
        print(f"Processing {len(audio_files)} audio files...")
        for i, audio_path in enumerate(audio_files):
            if i % 10 == 0:
                print(f"Processing file {i+1}/{len(audio_files)}")
            
            try:
                features = self.extract_features(audio_path)
                features_list.append(features)
                
                if features_dir:
                    # Save features to disk
                    basename = os.path.splitext(os.path.basename(audio_path))[0]
                    feature_path = os.path.join(features_dir, f"{basename}.npz")
                    np.savez(
                        feature_path,
                        mel_spectrogram=features['mel_spectrogram'],
                        f0=features['f0'],
                        audio_path=audio_path
                    )
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
        
        return features_list

def load_features(features_dir):
    """Load preprocessed features from disk"""
    feature_files = glob.glob(os.path.join(features_dir, "*.npz"))
    features_list = []
    
    print(f"Loading {len(feature_files)} feature files...")
    for feature_file in feature_files:
        try:
            data = np.load(feature_file, allow_pickle=True)
            features = {
                'mel_spectrogram': data['mel_spectrogram'],
                'f0': data['f0'],
                'audio_path': str(data['audio_path'])
            }
            features_list.append(features)
        except Exception as e:
            print(f"Error loading {feature_file}: {e}")
    
    return features_list

def prepare_training_data(features_list):
    """Prepare data for model training"""
    # Get max sequence length for padding
    max_length = max(f['mel_spectrogram'].shape[0] for f in features_list)
    
    # Initialize arrays
    X = np.zeros((len(features_list), max_length, features_list[0]['mel_spectrogram'].shape[1]))
    y_pitch = np.zeros((len(features_list), max_length, 1))
    
    # Fill arrays with padded data
    for i, features in enumerate(features_list):
        mel = features['mel_spectrogram']
        f0 = features['f0']
        
        seq_length = mel.shape[0]
        X[i, :seq_length, :] = mel
        y_pitch[i, :seq_length, 0] = f0
    
    return X, y_pitch

def create_batch_generator(X, y_pitch, batch_size=32, chunk_size=128):
    """Create a generator that yields chunks of sequences for training"""
    num_samples = X.shape[0]
    
    while True:
        # Shuffle indices
        indices = np.random.permutation(num_samples)
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            
            # Get full sequences
            batch_X = X[batch_indices]
            batch_y_pitch = y_pitch[batch_indices]
            
            # For each sequence, extract random chunks
            X_chunks = []
            y_pitch_chunks = []
            
            for j in range(len(batch_indices)):
                seq_length = np.sum(np.any(batch_X[j] != 0, axis=1))
                
                if seq_length <= chunk_size:
                    # If sequence is shorter than chunk_size, use the whole sequence
                    X_chunks.append(batch_X[j, :seq_length])
                    y_pitch_chunks.append(batch_y_pitch[j, :seq_length])
                else:
                    # Extract a random chunk
                    start = np.random.randint(0, seq_length - chunk_size)
                    X_chunks.append(batch_X[j, start:start+chunk_size])
                    y_pitch_chunks.append(batch_y_pitch[j, start:start+chunk_size])
            
            # Pad sequences to the same length
            X_chunks = tf.keras.preprocessing.sequence.pad_sequences(
                X_chunks, padding='post', dtype='float32')
            y_pitch_chunks = tf.keras.preprocessing.sequence.pad_sequences(
                y_pitch_chunks, padding='post', dtype='float32')
            
            yield X_chunks, {'pitch_output': y_pitch_chunks, 'mel_output': X_chunks}

def train_model(args):
    """Main training function"""
    # Initialize configuration
    config = VocalModelConfig()
    
    # Set up preprocessing
    preprocessor = DataPreprocessor(config)
    
    # Check if features directory exists
    features_dir = args.features_dir
    if features_dir and os.path.exists(features_dir) and len(os.listdir(features_dir)) > 0:
        print(f"Loading preprocessed features from {features_dir}")
        features_list = load_features(features_dir)
    else:
        print(f"Processing audio files from {args.audio_dir}")
        if features_dir:
            print(f"Saving features to {features_dir}")
        features_list = preprocessor.process_dataset(args.audio_dir, features_dir, limit=args.limit)
    
    if not features_list:
        raise ValueError("No features could be extracted from the dataset")
    
    # Prepare training data
    print("Preparing training data...")
    X, y_pitch = prepare_training_data(features_list)
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_pitch, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # Create model
    print("Creating model...")
    model = create_voice_transformation_model(config)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss={
            'pitch_output': 'mse',
            'mel_output': 'mse'
        },
        loss_weights={
            'pitch_output': args.pitch_weight,
            'mel_output': args.mel_weight
        }
    )
    
    # Create data generators
    train_generator = create_batch_generator(
        X_train, y_train, batch_size=args.batch_size, chunk_size=args.chunk_size)
    val_generator = create_batch_generator(
        X_val, y_val, batch_size=args.batch_size, chunk_size=args.chunk_size)
    
    # Create callbacks
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "model_{epoch:02d}.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, 
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=10, 
            monitor='val_loss',
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, 
            patience=5, 
            monitor='val_loss',
            min_lr=1e-6
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(args.output_dir, "logs"),
            histogram_freq=1
        )
    ]
    
    # Calculate steps per epoch and validation steps
    steps_per_epoch = len(X_train) // args.batch_size
    validation_steps = len(X_val) // args.batch_size
    
    # Train model
    print(f"Training model for {args.epochs} epochs...")
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.h5")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Export model for inference
    export_path = os.path.join(args.output_dir, "saved_model")
    export_model_for_inference(model, export_path)
    print(f"Model exported for inference to {export_path}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['pitch_output_loss'])
    plt.plot(history.history['mel_output_loss'])
    plt.title('Component Losses')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Pitch', 'Mel'], loc='upper right')
    
    plot_path = os.path.join(args.output_dir, "training_history.png")
    plt.savefig(plot_path)
    print(f"Training history plot saved to {plot_path}")
    
    return model, history

def main():
    parser = argparse.ArgumentParser(description="Train VST Pitch Perfect voice transformation model")
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Directory containing audio files for training")
    parser.add_argument("--features_dir", type=str, default=None,
                        help="Directory to save/load preprocessed features")
    parser.add_argument("--output_dir", type=str, default="./trained_models",
                        help="Directory to save trained models and outputs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--chunk_size", type=int, default=128,
                        help="Size of audio chunks for training")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Initial learning rate")
    parser.add_argument("--pitch_weight", type=float, default=1.0,
                        help="Weight for pitch loss")
    parser.add_argument("--mel_weight", type=float, default=0.5,
                        help="Weight for mel spectrogram loss")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of audio files to process (for testing)")
    
    args = parser.parse_args()
    train_model(args)

if __name__ == "__main__":
    main()