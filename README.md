# Voices: Multi-Modal Audio Processing Application

![Voices Logo](public/logo.png)

## Overview

Voices is a desktop application designed for processing audio and video files to isolate individual voices, identify speakers, and organize the processed content in a structured and searchable manner. It combines powerful machine learning models with an intuitive user interface to provide professional-grade voice separation capabilities.

## Key Features

- **Voice Isolation**: Extract clear voice tracks from mixed audio sources, even in challenging acoustic environments
- **Speaker Identification**: Identify and track speakers across multiple recordings
- **Content Organization**: Organize and retrieve processed audio by speaker, date, or content
- **Batch Processing**: Efficiently process multiple files with configurable settings
- **Model Comparison**: Compare different voice separation models on the same audio input
- **Enhanced Audio Visualization**: Multi-track display with waveform coloring by speaker
- **Processing Configuration**: Customize model selection and processing parameters
- **User Feedback Collection**: Provide feedback on separation quality and feature suggestions

## Technology Stack

### Frontend
- **Electron**: Cross-platform desktop application framework
- **React**: Component-based UI library
- **Wavesurfer.js**: Audio visualization

### Backend
- **Python**: Core audio processing functionality
- **PyTorch**: Machine learning framework for voice separation models
- **SQLite**: Embedded database for metadata and speaker information

### Voice Separation Models
- **SVoice**: Optimized for multi-speaker scenarios (3+ speakers)
- **Demucs**: Better for noisy environments
- **Model Abstraction Layer**: Unified interface with intelligent model selection

## Installation

### Prerequisites
- Node.js (v14+)
- Python 3.7+
- PyTorch 2.0+
- CUDA-compatible GPU (optional, but recommended for faster processing)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/voices2.git
   cd voices2
   ```

2. Install JavaScript dependencies:
   ```bash
   npm install
   ```

3. Set up Python environment:
   ```bash
   python setup_env.py
   ```
   This script will:
   - Create a virtual environment
   - Install required Python packages
   - Configure PyTorch for your hardware
   - Set up the application database

4. Start the application:
   ```bash
   npm start
   ```

## Project Structure

```
voices2/
├── src/
│   ├── frontend/                # React components and controllers
│   │   ├── components/          # UI components
│   │   │   ├── audio/           # Audio visualization components
│   │   │   ├── feedback/        # User feedback components
│   │   │   ├── layout/          # Layout components (Header, Footer)
│   │   │   ├── metadata/        # Metadata management components
│   │   │   ├── models/          # Model comparison components
│   │   │   ├── pages/           # Page components
│   │   │   ├── processing/      # Processing configuration components
│   │   │   └── testing/         # Integration testing components
│   │   ├── controllers/         # Frontend controllers
│   │   └── styles/              # CSS styles
│   ├── backend/                 # Python backend
│   │   ├── core/                # Core functionality
│   │   │   ├── communication/   # IPC between Electron and Python
│   │   │   ├── config/          # Configuration system
│   │   │   ├── logging/         # Logging framework
│   │   │   └── queue/           # Processing queue
│   │   ├── processing/          # Audio processing
│   │   │   ├── audio/           # Audio I/O and management
│   │   │   ├── experiment/      # ML experimentation framework
│   │   │   ├── models/          # Voice separation models
│   │   │   │   ├── svoice/      # SVoice model implementation
│   │   │   │   └── demucs/      # Demucs model implementation
│   │   │   ├── pipeline/        # Processing pipeline
│   │   │   └── registry/        # Model registry
│   │   └── storage/             # Data storage
│   │       ├── database/        # Database management
│   │       ├── files/           # File management
│   │       └── metadata/        # Metadata management
│   ├── main.js                  # Electron main process
│   └── preload.js               # Electron preload script
├── public/                      # Static assets
│   ├── styles/                  # CSS styles
│   └── index.html               # Main HTML file
├── memory-bank/                 # Project memory bank (documentation)
├── requirements.txt             # Python dependencies
├── setup_env.py                 # Environment setup script
├── webpack.config.js            # Webpack configuration
└── package.json                 # npm configuration
```

## Usage Guide

### Basic Usage

1. **Launch the application**: Run `npm start` from the project directory
2. **Import audio files**: Use the dashboard to import audio files for processing
3. **Configure processing**: Select the appropriate model and settings
4. **Process audio**: Start the processing job
5. **View results**: Explore the separated audio tracks in the visualization view
6. **Export results**: Save the separated tracks to your desired location

### Advanced Features

#### Model Comparison

Compare different voice separation models on the same audio input:

1. Navigate to the Model Comparison page
2. Upload an audio file or select from previously processed files
3. Select the models to compare
4. View the separation results side by side with waveform visualization
5. Compare objective metrics and listen to the results

#### Audio Visualization

Visualize and analyze separated audio tracks:

1. Navigate to the Audio Visualization page
2. Load a processed audio file
3. Use the multi-track display to view all separated voices
4. Zoom and navigate through the audio waveforms
5. Identify speaker transitions with segment markers
6. Play individual speaker tracks or the original mixture

#### Processing Configuration

Customize the voice separation process:

1. Navigate to the Processing Configuration page
2. Select the voice separation model (SVoice, Demucs, or Auto)
3. Configure model-specific parameters
4. Set output format options
5. Configure batch processing settings
6. Save your configuration for future use

#### Integration Testing

Test the integration between different components:

1. Navigate to the Integration Tester page
2. Upload a test audio file
3. Configure test parameters
4. Run individual tests or the complete test suite
5. View detailed test results and logs

## Development

### Building from Source

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   python setup_env.py
   ```
3. Run in development mode:
   ```bash
   npm run dev
   ```
4. Build for production:
   ```bash
   npm run build
   ```

### Adding a New Voice Separation Model

1. Create a new model implementation in `src/backend/processing/models/`
2. Implement the required adapter interface
3. Register the model with the model registry
4. Update the model selection logic in the abstraction layer
5. Add tests for the new model

### Running Tests

Run integration tests:
```bash
cd src/backend/processing/registry && python run_integration_tests.py --verbose
```

Run unit tests for specific components:
```bash
python -m unittest src/backend/processing/models/test_abstraction.py
```

## Configuration

### Environment Variables

- `VOICES_CONFIG_DIR`: Directory for configuration files (default: `~/.voices`)
- `VOICES_DATA_DIR`: Directory for application data (default: `~/.voices/data`)
- `VOICES_LOG_LEVEL`: Logging level (default: `INFO`)
- `VOICES_GPU_ENABLED`: Enable GPU acceleration (default: `true` if available)

### Configuration Files

- `config.yaml`: Main configuration file
- `models.yaml`: Model registry configuration
- `processing.yaml`: Processing pipeline configuration

## Troubleshooting

### Common Issues

#### Python Dependencies

If you encounter issues with Python dependencies, try:
```bash
python setup_env.py --force
```

#### GPU Acceleration

If GPU acceleration is not working:
1. Ensure you have a CUDA-compatible GPU
2. Install the appropriate CUDA toolkit
3. Verify PyTorch is installed with CUDA support:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

#### Integration Test Failures

If integration tests fail:
1. Check the log file at `src/backend/processing/registry/integration_tests.log`
2. Ensure all dependencies are installed
3. Verify the test audio file exists and is valid

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- SVoice and Demucs voice separation technologies
- PyTorch for machine learning framework
- Electron and React for the frontend framework