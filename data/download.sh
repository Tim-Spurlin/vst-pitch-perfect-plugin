#!/bin/bash
# Download script for voice datasets used in VST Pitch Perfect Plugin

# Create data directory
mkdir -p datasets
cd datasets

# Download NUS-48E dataset (Singing voice dataset)
echo "Downloading NUS-48E dataset..."
mkdir -p nus48e
cd nus48e

# Use curl with progress bar
curl -L -o nus48e.zip "https://drive.google.com/uc?export=download&id=12wgJmMY4aDv7-YexCNxGBK1J_WsTnR9J" || {
    echo "Failed to download NUS-48E dataset"
    exit 1
}

# Extract dataset
echo "Extracting NUS-48E dataset..."
unzip -q nus48e.zip
rm nus48e.zip
cd ..

# Download VCTK dataset (Speech dataset for voice characteristics)
echo "Downloading VCTK dataset..."
mkdir -p vctk
cd vctk

# Use curl with progress bar
curl -L -o vctk.zip "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip" || {
    echo "Failed to download VCTK dataset"
    exit 1
}

# Extract dataset
echo "Extracting VCTK dataset..."
unzip -q vctk.zip
rm vctk.zip
cd ..

# Download LibriTTS dataset (High-quality TTS dataset)
echo "Downloading LibriTTS dev-clean subset..."
mkdir -p libritts
cd libritts

# Download dev-clean subset (smaller size for testing)
curl -L -o dev-clean.tar.gz "https://www.openslr.org/resources/60/dev-clean.tar.gz" || {
    echo "Failed to download LibriTTS dataset"
    exit 1
}

# Extract dataset
echo "Extracting LibriTTS dataset..."
tar -xzf dev-clean.tar.gz
rm dev-clean.tar.gz
cd ..

# Download NSynth dataset (Instrument sounds, useful for enhancing voice quality understanding)
echo "Downloading NSynth dataset (small subset)..."
mkdir -p nsynth
cd nsynth

curl -L -o nsynth-test.jsonwav.tar.gz "https://storage.googleapis.com/magentadata/datasets/nsynth/nsynth-test.jsonwav.tar.gz" || {
    echo "Failed to download NSynth dataset"
    exit 1
}

# Extract dataset
echo "Extracting NSynth dataset..."
tar -xzf nsynth-test.jsonwav.tar.gz
rm nsynth-test.jsonwav.tar.gz
cd ..

echo "Dataset downloads complete!"
echo "Datasets available in: $(pwd)"
echo "Total dataset size: $(du -sh . | cut -f1)"

# Create a simple index file
echo "Creating dataset index..."
{
  echo "Dataset,Location,Description"
  echo "NUS-48E,./nus48e,Singing voice dataset with 48 English songs from 12 subjects"
  echo "VCTK,./vctk,Speech dataset with 109 native English speakers with various accents"
  echo "LibriTTS,./libritts,High-quality dataset derived from LibriVox audiobooks"
  echo "NSynth,./nsynth,Instrument sounds for enhancing voice quality understanding"
} > dataset_index.csv

echo "Download script completed successfully!"