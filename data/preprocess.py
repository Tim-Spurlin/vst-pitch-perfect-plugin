#!/usr/bin/env python3
"""
Preprocess audio datasets for vocal transformation model training
Optimized for high-quality feature extraction and efficient training
"""

import os
import numpy as np
import librosa
import soundfile as sf
import glob
import pandas as pd
import argparse
import json
import multiprocessing
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.model import VocalModelConfig

class VocalPreprocessor:
    def __init__(self, config=None):
        if config is None:
            self.config = VocalModelConfig()
        else:
            self.config = config
        
        # Ensure sane defaults
        self.min_duration = 1.0  # Minimum audio duration in seconds
        self.max_duration = 30.0  # Maximum audio duration in seconds
        self.augmentation_enabled = True
    
    def extract_features(self, audio_path, output_dir=None, augment=True):
        """Extract features from a single audio file"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.config.sample_rate)
            
            # Filter by duration
            duration = len(y) / sr
            if duration < self.min_duration:
                return None
            if duration > self.max_duration:
                # Truncate to max duration
                y = y[:int(self.max_duration * sr)]
                duration = self.max_duration
            
            # Extract mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=self.config.n_fft, 
                hop_length=self.config.hop_length, n_mels=self.config.n_mels,
                fmin=50, fmax=sr/2-1000  # Adjusted for better vocal representation
            )
            mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            
            # Extract fundamental frequency (f0)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, fmin=self.config.f0_min, fmax=self.config.f0_max,
                sr=sr, hop_length=self.config.hop_length,
                frame_length=self.config.n_fft
            )
            
            # Handle NaN values in f0
            f0 = np.nan_to_num(f0)
            
            # Extract harmonic and percussive components
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # Extract additional features for better voice modeling
            spectral_centroid = librosa.feature.spectral_centroid(
                y=y, sr=sr, n_fft=self.config.n_fft, hop_length=self.config.hop_length)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=y, sr=sr, n_fft=self.config.n_fft, hop_length=self.config.hop_length)
            
            # Make sure all time-dimension features have the same length
            min_length = min(mel_db.shape[1], len(f0), 
                            spectral_centroid.shape[1], spectral_bandwidth.shape[1])
            
            mel_db = mel_db[:, :min_length]
            f0 = f0[:min_length]
            voiced_flag = voiced_flag[:min_length]
            spectral_centroid = spectral_centroid[:, :min_length]
            spectral_bandwidth = spectral_bandwidth[:, :min_length]
            
            # Create feature dictionary
            features = {
                'mel_spectrogram': mel_db,
                'f0': f0,
                'voiced_flag': voiced_flag,
                'spectral_centroid': spectral_centroid,
                'spectral_bandwidth': spectral_bandwidth,
                'audio_path': audio_path,
                'duration': duration,
                'sample_rate': sr
            }
            
            # Save features if output directory is provided
            if output_dir:
                basename = os.path.splitext(os.path.basename(audio_path))[0]
                feature_path = os.path.join(output_dir, f"{basename}.npz")
                np.savez(
                    feature_path,
                    mel_spectrogram=mel_db,
                    f0=f0,
                    voiced_flag=voiced_flag,
                    spectral_centroid=spectral_centroid,
                    spectral_bandwidth=spectral_bandwidth,
                    audio_path=audio_path,
                    duration=duration,
                    sample_rate=sr
                )
                
                # Create audio samples for the extracted features
                audio_output_dir = os.path.join(output_dir, 'audio_samples')
                os.makedirs(audio_output_dir, exist_ok=True)
                
                # Save a short sample of the original audio
                sample_length = min(int(3 * sr), len(y))  # 3 seconds or full audio
                sf.write(os.path.join(audio_output_dir, f"{basename}_original.wav"), 
                         y[:sample_length], sr)
                
                # Save harmonic component
                sf.write(os.path.join(audio_output_dir, f"{basename}_harmonic.wav"), 
                         y_harmonic[:sample_length], sr)
            
            # Apply data augmentation if needed
            if augment and self.augmentation_enabled:
                augmented_features = self._augment_features(features, output_dir)
                return [features] + augmented_features
            
            return [features]
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def _augment_features(self, features, output_dir=None):
        """Apply data augmentation to features"""
        augmented_features = []
        
        # Load original audio
        audio_path = features['audio_path']
        y, sr = librosa.load(audio_path, sr=features['sample_rate'])
        
        # Pitch shift augmentation
        for pitch_shift in [-2, -1, 1, 2]:  # Semitones
            try:
                # Shift pitch
                y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)
                
                # Extract features from shifted audio
                mel_spectrogram = librosa.feature.melspectrogram(
                    y=y_shifted, sr=sr, n_fft=self.config.n_fft, 
                    hop_length=self.config.hop_length, n_mels=self.config.n_mels
                )
                mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
                
                f0, voiced_flag, _ = librosa.pyin(
                    y_shifted, fmin=self.config.f0_min, fmax=self.config.f0_max,
                    sr=sr, hop_length=self.config.hop_length
                )
                f0 = np.nan_to_num(f0)
                
                # Make sure lengths match
                min_length = min(mel_db.shape[1], len(f0))
                mel_db = mel_db[:, :min_length]
                f0 = f0[:min_length]
                
                # Create augmented feature dictionary
                aug_features = {
                    'mel_spectrogram': mel_db,
                    'f0': f0,
                    'voiced_flag': voiced_flag[:min_length],
                    'audio_path': audio_path,
                    'duration': len(y_shifted) / sr,
                    'sample_rate': sr,
                    'augmentation': f'pitch_shift_{pitch_shift}'
                }
                
                augmented_features.append(aug_features)
                
                # Save augmented features if output directory is provided
                if output_dir:
                    basename = os.path.splitext(os.path.basename(audio_path))[0]
                    feature_path = os.path.join(output_dir, f"{basename}_pitch{pitch_shift}.npz")
                    np.savez(
                        feature_path,
                        mel_spectrogram=mel_db,
                        f0=f0,
                        voiced_flag=voiced_flag[:min_length],
                        audio_path=audio_path,
                        duration=len(y_shifted) / sr,
                        sample_rate=sr,
                        augmentation=f'pitch_shift_{pitch_shift}'
                    )
                    
                    # Save augmented audio sample
                    audio_output_dir = os.path.join(output_dir, 'audio_samples')
                    os.makedirs(audio_output_dir, exist_ok=True)
                    
                    # Save a short sample
                    sample_length = min(int(3 * sr), len(y_shifted))  # 3 seconds or full audio
                    sf.write(os.path.join(audio_output_dir, f"{basename}_pitch{pitch_shift}.wav"), 
                             y_shifted[:sample_length], sr)
            except Exception as e:
                print(f"Error in pitch shift augmentation {pitch_shift} for {audio_path}: {e}")
        
        # Time stretch augmentation
        for speed_factor in [0.9, 1.1]:
            try:
                # Time stretch
                y_stretched = librosa.effects.time_stretch(y, rate=speed_factor)
                
                # Extract features from stretched audio
                mel_spectrogram = librosa.feature.melspectrogram(
                    y=y_stretched, sr=sr, n_fft=self.config.n_fft, 
                    hop_length=self.config.hop_length, n_mels=self.config.n_mels
                )
                mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
                
                f0, voiced_flag, _ = librosa.pyin(
                    y_stretched, fmin=self.config.f0_min, fmax=self.config.f0_max,
                    sr=sr, hop_length=self.config.hop_length
                )
                f0 = np.nan_to_num(f0)
                
                # Make sure lengths match
                min_length = min(mel_db.shape[1], len(f0))
                mel_db = mel_db[:, :min_length]
                f0 = f0[:min_length]
                
                # Create augmented feature dictionary
                aug_features = {
                    'mel_spectrogram': mel_db,
                    'f0': f0,
                    'voiced_flag': voiced_flag[:min_length],
                    'audio_path': audio_path,
                    'duration': len(y_stretched) / sr,
                    'sample_rate': sr,
                    'augmentation': f'time_stretch_{speed_factor}'
                }
                
                augmented_features.append(aug_features)
                
                # Save augmented features if output directory is provided
                if output_dir:
                    basename = os.path.splitext(os.path.basename(audio_path))[0]
                    feature_path = os.path.join(output_dir, f"{basename}_speed{speed_factor}.npz")
                    np.savez(
                        feature_path,
                        mel_spectrogram=mel_db,
                        f0=f0,
                        voiced_flag=voiced_flag[:min_length],
                        audio_path=audio_path,
                        duration=len(y_stretched) / sr,
                        sample_rate=sr,
                        augmentation=f'time_stretch_{speed_factor}'
                    )
                    
                    # Save augmented audio sample
                    audio_output_dir = os.path.join(output_dir, 'audio_samples')
                    os.makedirs(audio_output_dir, exist_ok=True)
                    
                    # Save a short sample
                    sample_length = min(int(3 * sr), len(y_stretched))  # 3 seconds or full audio
                    sf.write(os.path.join(audio_output_dir, f"{basename}_speed{speed_factor}.wav"), 
                             y_stretched[:sample_length], sr)
            except Exception as e:
                print(f"Error in time stretch augmentation {speed_factor} for {audio_path}: {e}")
        
        return augmented_features
    
    def process_audio_file(self, args):
        """Process a single audio file (for multiprocessing)"""
        audio_path, output_dir, augment = args
        return self.extract_features(audio_path, output_dir, augment)
    
    def process_dataset(self, dataset_dir, output_dir, file_pattern="**/*.wav", 
                        num_workers=None, limit=None, augment=True, max_per_subdirectory=100):
        """Process an entire dataset of audio files with multiprocessing"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all audio files
        audio_files = glob.glob(os.path.join(dataset_dir, file_pattern), recursive=True)
        
        if not audio_files:
            print(f"No audio files found in {dataset_dir} matching pattern {file_pattern}")
            return None
        
        # Check if we should limit the total number of files
        if limit and limit < len(audio_files):
            audio_files = np.random.choice(audio_files, limit, replace=False)
        
        # Group by subdirectory and apply max_per_subdirectory limit
        if max_per_subdirectory:
            files_by_subdir = {}
            for file_path in audio_files:
                subdir = os.path.dirname(file_path)
                if subdir not in files_by_subdir:
                    files_by_subdir[subdir] = []
                files_by_subdir[subdir].append(file_path)
            
            # Limit files per subdirectory
            limited_files = []
            for subdir, files in files_by_subdir.items():
                if len(files) > max_per_subdirectory:
                    files = np.random.choice(files, max_per_subdirectory, replace=False)
                limited_files.extend(files)
            
            audio_files = limited_files
        
        print(f"Processing {len(audio_files)} audio files...")
        
        # Prepare arguments for multiprocessing
        process_args = [(audio_path, output_dir, augment) for audio_path in audio_files]
        
        # Use multiprocessing to speed up preprocessing
        if num_workers is None:
            num_workers = max(1, multiprocessing.cpu_count() - 1)
        
        features_list = []
        
        if num_workers > 1:
            with multiprocessing.Pool(num_workers) as pool:
                results = list(tqdm(pool.imap(self.process_audio_file, process_args), 
                                   total=len(process_args)))
                
                # Flatten results and remove None values
                for result in results:
                    if result:
                        features_list.extend(result)
        else:
            # Sequential processing
            for args in tqdm(process_args):
                result = self.process_audio_file(args)
                if result:
                    features_list.extend(result)
        
        # Create metadata file
        metadata = {
            'num_original_files': len(audio_files),
            'num_processed_features': len(features_list),
            'config': vars(self.config),
            'augmentation_enabled': self.augmentation_enabled
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Preprocessing complete! {len(features_list)} feature files created.")
        
        return features_list

def main():
    parser = argparse.ArgumentParser(description="Preprocess audio datasets for vocal transformation model")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Directory containing audio files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save preprocessed features")
    parser.add_argument("--file_pattern", type=str, default="**/*.wav",
                        help="Pattern to match audio files (e.g., '**/*.wav' for recursive)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of worker processes (default: CPU count - 1)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of files to process")
    parser.add_argument("--no_augment", action="store_true",
                        help="Disable data augmentation")
    parser.add_argument("--max_per_subdir", type=int, default=100,
                        help="Maximum files to process per subdirectory")
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = VocalPreprocessor()
    
    # Process dataset
    preprocessor.process_dataset(
        args.dataset_dir,
        args.output_dir,
        file_pattern=args.file_pattern,
        num_workers=args.num_workers,
        limit=args.limit,
        augment=not args.no_augment,
        max_per_subdirectory=args.max_per_subdir
    )

if __name__ == "__main__":
    main()