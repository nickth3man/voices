# Voice Separation Models

This directory contains implementations of various voice separation models and a unified abstraction layer for using them interchangeably.

## Directory Structure

- `abstraction.py`: Voice separation model abstraction layer
- `test_abstraction.py`: Tests for the abstraction layer
- `examples/`: Example scripts demonstrating usage
- `svoice/`: SVoice model implementation
  - `model.py`: Core SVoice model architecture
  - `utils.py`: Utility functions for SVoice
  - `cli.py`: Command-line interface
  - `test_svoice.py`: Test script
  - `README.md`: Documentation

## Voice Separation Model Abstraction Layer

The abstraction layer provides a consistent interface for different voice separation technologies, allowing them to be used interchangeably. It includes adapters for both SVoice and Demucs, and intelligent model selection based on audio characteristics.

### Key Features

- **Unified Interface**: Common API for all voice separation models
- **Intelligent Model Selection**: Automatically selects the best model based on audio characteristics
- **Model Registry Integration**: Works with the model registry for versioning and management
- **Flexible Input Handling**: Supports both NumPy arrays and PyTorch tensors
- **Configurable**: Easily switch between different model types

### Usage

#### Basic Usage

```python
from models.abstraction import create_separation_manager

# Create a manager with default settings
manager = create_separation_manager()

# Separate voices in an audio file
sources = manager.separate(audio, num_speakers=2)
```

#### Automatic Model Selection

```python
from models.abstraction import create_separation_manager, AudioCharacteristics, ModelType

# Create a manager with automatic model selection
manager = create_separation_manager(default_model_type="auto")

# Extract audio characteristics
characteristics = AudioCharacteristics.from_audio(audio, sample_rate)
characteristics.num_speakers = 3  # Set if known

# Separate using the automatically selected model
sources = manager.separate(audio, num_speakers=characteristics.num_speakers)
```

#### Using a Specific Model Type

```python
from models.abstraction import create_separation_manager, ModelType

# Create a manager
manager = create_separation_manager()

# Separate using SVoice
sources = manager.separate(audio, model_type=ModelType.SVOICE)

# Separate using Demucs
sources = manager.separate(audio, model_type=ModelType.DEMUCS)
```

#### Using with Model Registry

```python
from models.abstraction import create_separation_manager

# Create a manager with a model registry
manager = create_separation_manager(registry_dir="/path/to/registry")

# Get information about available models
models = manager.get_available_models()

# Separate using a specific model from the registry
sources = manager.separate(audio, model_id="svoice_model_123")
```

### Command-line Example

The abstraction layer can be used from the command line using the example script:

```bash
python -m models.examples.abstraction_example \
  --input input.wav \
  --output output_dir \
  --speakers 2 \
  --model auto
```

## Model Types

### SVoice

SVoice is a voice separation model designed for separating multiple speakers (3+ speakers) from a single audio mixture. It uses a deep learning approach with STFT-based features and LSTM-based separation.

Key features:
- Support for variable numbers of speakers
- GPU acceleration with PyTorch
- CPU fallback for systems without GPU
- Configurable model parameters

### Demucs

Demucs is an alternative voice separation model that may perform better in certain scenarios, such as noisy environments or when separating music from speech.

Key features:
- Potentially better performance in noisy conditions
- Different architecture approach from SVoice
- Complementary strengths for different audio characteristics

## Development

### Adding a New Model Type

To add a new model type to the abstraction layer:

1. Create a new adapter class that implements the `VoiceSeparationModel` interface
2. Add the new model type to the `ModelType` enum in `abstraction.py`
3. Update the model selection logic in `select_model_type` method
4. Register the model loader with the model registry

### Running Tests

To run the tests for the abstraction layer:

```bash
python -m unittest models.test_abstraction