# Audio File I/O and Management

This module provides comprehensive functionality for audio file handling, including:

- Loading audio files with format detection and validation
- Extracting metadata from audio files
- Generating waveform data for visualization
- Converting between different audio formats
- Saving audio files with various format options

## Components

### AudioFileIO

The main class for audio file I/O operations, providing methods for:

- Loading audio files with optional metadata extraction
- Saving audio data to files in various formats
- Getting file information without loading the audio data
- Generating waveform data for visualization

### AudioFormatDetector

Utility class for detecting and validating audio file formats:

- Detecting audio format based on file extension
- Validating if a file has a supported format
- Getting supported formats and extensions

### AudioMetadataExtractor

Utility class for extracting metadata from audio files:

- Basic file metadata (filename, path, format, size, etc.)
- Audio properties (sample rate, channels, duration, etc.)
- Audio characteristics (RMS energy, spectral centroid, etc.)

### WaveformGenerator

Utility class for generating waveform data for visualization:

- Generating waveform data with a specified number of points
- Creating waveform images for visualization

## Usage Examples

### Loading an Audio File

```python
from src.backend.processing.audio.io import load_audio

# Load an audio file with default settings (16kHz, mono)
result = load_audio("path/to/audio.wav")

# Access the audio data and sample rate
audio = result["audio"]
sample_rate = result["sample_rate"]

# Access metadata if extracted
if "metadata" in result:
    metadata = result["metadata"]
    duration = metadata["duration"]
    channels = metadata["channels"]
```

### Saving an Audio File

```python
from src.backend.processing.audio.io import save_audio

# Save audio data to a file
output_path = save_audio(audio, "path/to/output.wav", sample_rate)

# Save with a specific format
output_path = save_audio(audio, "path/to/output", sample_rate, format="flac")
```

### Extracting Metadata

```python
from src.backend.processing.audio.io import extract_audio_metadata

# Extract metadata from an audio file
metadata = extract_audio_metadata("path/to/audio.wav")

# Access metadata fields
duration = metadata["duration"]
sample_rate = metadata["sample_rate"]
channels = metadata["channels"]
```

### Generating Waveform Data

```python
from src.backend.processing.audio.io import generate_waveform

# Generate waveform data with 1000 points
waveform = generate_waveform(audio, n_points=1000)
```

## Command-Line Interface

The module includes a command-line interface for common audio operations:

### Getting File Information

```bash
python -m src.backend.processing.audio.cli info path/to/audio.wav
```

### Extracting Metadata

```bash
python -m src.backend.processing.audio.cli metadata path/to/audio.wav --output metadata.json
```

### Converting Audio Formats

```bash
python -m src.backend.processing.audio.cli convert path/to/input.wav path/to/output.flac --format flac --sample-rate 44100
```

### Generating Waveform Visualization

```bash
python -m src.backend.processing.audio.cli waveform path/to/audio.wav --output waveform.png --width 800 --height 200
```

### Listing Supported Formats

```bash
python -m src.backend.processing.audio.cli formats
```

## Supported Audio Formats

- WAV (.wav, .wave)
- FLAC (.flac)
- MP3 (.mp3)
- OGG (.ogg)
- AAC (.aac, .m4a)
- AIFF (.aiff, .aif)