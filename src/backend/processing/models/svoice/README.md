# SVoice Model

This package provides an implementation of the SVoice model for voice separation, including model architecture, loading, and inference capabilities.

## Overview

SVoice (Speaker-attributed Voice Separation) is a neural network model designed to separate mixed audio signals into individual speaker sources. It uses a combination of spectral processing and deep learning techniques to achieve high-quality voice separation.

## Features

- Separate mixed audio into individual speaker sources
- Support for variable numbers of speakers
- PyTorch-based implementation for GPU acceleration
- Integration with the model registry system
- Command-line interface for easy usage
- Utility functions for audio processing and model management

## Usage

### Basic Usage

```python
import torch
import soundfile as sf
from processing.models.svoice import SVoiceModel, load_svoice_model

# Load audio
audio, sample_rate = sf.read("mixed_audio.wav")

# Option 1: Create a model from scratch
model = SVoiceModel(n_speakers=2)
sources = model.separate(audio)

# Option 2: Load a pretrained model
model = load_svoice_model("path/to/model.pth")
sources = model.separate(audio)

# Save separated sources
for i, source in enumerate(sources):
    sf.write(f"source_{i+1}.wav", source, sample_rate)
```

### Using the Command-Line Interface

```bash
# Download a pretrained model
python -m processing.models.svoice.cli download --output-dir models --model-name svoice_base

# Separate audio sources
python -m processing.models.svoice.cli separate --model-path models/svoice_base.pth --audio-path mixed_audio.wav --output-dir separated

# Get model information
python -m processing.models.svoice.cli info --model-path models/svoice_base.pth
```

### Integration with Model Registry

```python
from processing.registry.model_registry import ModelRegistry
from processing.registry.model_adapters import get_model_loader

# Create a model registry
registry = ModelRegistry("model_registry")

# Register the SVoice model loader
svoice_loader = get_model_loader("svoice")
registry.register_model_loader("svoice", svoice_loader)

# Add a model to the registry
model_id, version_id = registry.add_model(
    name="SVoice",
    description="Voice separation model",
    model_type="svoice",
    model_path="path/to/model.pth",
    version_description="Initial version",
    parameters={"n_speakers": 2}
)

# Load and use the model
process_func = registry.load_model(model_id)
sources = process_func(audio)
```

## Model Architecture

The SVoice model architecture consists of:

1. **Encoder**: Transforms the input spectrogram into a latent representation
2. **Separator**: LSTM-based network that separates the mixed features
3. **Mask Estimator**: Generates masks for each speaker
4. **Decoder**: Transforms the masked features back to the time domain

## Testing

A test script is provided to verify the SVoice model integration:

```bash
# Run all tests
python -m processing.models.svoice.test_svoice

# Run specific test
python -m processing.models.svoice.test_svoice --test-type model
```

## Requirements

- PyTorch >= 1.7.0
- NumPy
- SoundFile
- Librosa (optional, for advanced audio processing)