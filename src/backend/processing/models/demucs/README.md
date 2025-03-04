# Demucs Voice Separation Model

This directory contains the implementation of the Demucs model for voice separation, adapted for the Voices application.

## Overview

Demucs (Deep Extractor for Music Sources) is a waveform-based source separation model originally designed for music source separation. In this implementation, it has been adapted for voice separation tasks. Demucs uses a U-Net architecture with LSTM layers and operates directly on the waveform, making it particularly effective for handling noisy environments.

## Key Features

- **Waveform-based processing**: Works directly on audio waveforms without requiring spectrograms
- **U-Net architecture**: Uses encoder-decoder structure with skip connections
- **LSTM layers**: Captures temporal dependencies in audio
- **Effective for noisy environments**: Particularly good at separating voices in challenging acoustic conditions
- **Variable speaker support**: Can be configured for different numbers of speakers

## Model Architecture

The Demucs model architecture consists of:

1. **Encoder**: A series of convolutional layers that progressively reduce the temporal resolution while increasing the feature dimensions
2. **LSTM**: Bidirectional LSTM layers that capture temporal dependencies
3. **Decoder**: A series of transposed convolutional layers with skip connections from the encoder
4. **Output layer**: Final convolutional layer that produces separated sources

## Usage

### From Python

```python
from src.backend.processing.models.demucs import load_demucs_model, separate_sources

# Load model
model = load_demucs_model("path/to/model")

# Separate sources
sources = model.separate(audio, num_speakers=2)

# Or use the utility function
result = separate_sources(model, "path/to/audio.wav", output_dir="output")
```

### From Command Line

```bash
# Separate voices in an audio file
python -m src.backend.processing.models.demucs.cli separate input.wav --output-dir output --num-speakers 2

# Download a pretrained model
python -m src.backend.processing.models.demucs.cli download --output-dir models

# Display model information
python -m src.backend.processing.models.demucs.cli info --model models/demucs_base.pth
```

## Integration with Voice Separation Abstraction Layer

The Demucs model is integrated with the Voice Separation Abstraction Layer, allowing it to be used interchangeably with other voice separation models like SVoice. The abstraction layer automatically selects the most appropriate model based on audio characteristics.

```python
from src.backend.processing.models.abstraction import create_separation_manager

# Create separation manager
manager = create_separation_manager()

# Separate using auto-selection
sources = manager.separate(audio)

# Force using Demucs
sources = manager.separate(audio, model_type="demucs")
```

## Comparison with SVoice

Compared to SVoice, Demucs:

- Works better in noisy environments
- Operates directly on waveforms rather than spectrograms
- May be less effective for separating more than 2 speakers
- Has different computational characteristics (may be faster or slower depending on the hardware)

## References

- Original Demucs paper: "Music Source Separation in the Waveform Domain"
- Project adapted from: [https://github.com/facebookresearch/demucs](https://github.com/facebookresearch/demucs)