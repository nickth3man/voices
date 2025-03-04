# Configuration System

This module provides a comprehensive configuration management system for the application, with a focus on ML model settings and hardware-specific configurations.

## Features

- **Dynamic Hardware Detection**: Automatically detects hardware capabilities (GPU, CPU, memory) and adapts configuration accordingly
- **ML Model Settings**: Specialized support for ML model configuration with hardware compatibility checks
- **Schema Validation**: Validates configuration against a defined schema to ensure correctness
- **Default Configurations**: Provides sensible defaults for all settings
- **Environment Overrides**: Supports overriding configuration via environment variables
- **Command-Line Interface**: Includes a CLI for viewing, editing, and validating configuration
- **Configuration File Management**: Handles loading, saving, and merging configuration files

## Components

### ConfigManager

The main class for managing configuration settings:

```python
from src.backend.core.config import ConfigManager

# Initialize with a configuration file
config_manager = ConfigManager("config.json")

# Get a configuration value
log_level = config_manager.get("app.log_level", "INFO")

# Set a configuration value
config_manager.set("app.log_level", "DEBUG")

# Save the configuration
config_manager.save_config()

# Get model settings
model_settings = config_manager.get_model_settings("svoice")

# Check model compatibility with hardware
is_compatible, issues = config_manager.is_model_compatible("svoice")

# Get optimized model settings for current hardware
optimized_settings = config_manager.get_optimal_model_settings("svoice")
```

### Hardware Detection

Detects hardware capabilities for dynamic configuration:

```python
from src.backend.core.config.hardware_detection import detect_hardware_capabilities

# Get hardware capabilities
capabilities = detect_hardware_capabilities()

# Check if GPU is available
if capabilities["gpu_enabled"]:
    print(f"GPU available: {capabilities['gpu_devices'][0]['name']}")
    print(f"GPU memory: {capabilities['gpu_memory']} MB")
```

### Default Configuration

Provides default configuration settings:

```python
from src.backend.core.config.defaults import get_default_config

# Get default configuration
config = get_default_config()

# Get environment-specific overrides
from src.backend.core.config.defaults import get_environment_specific_config
env_config = get_environment_specific_config()
```

### Schema Validation

Validates configuration against a defined schema:

```python
from src.backend.core.config.schema import validate_config, ConfigSchema

# Get the schema
schema = ConfigSchema.get_schema()

# Validate a configuration
errors = validate_config(config)
if errors:
    print("Configuration validation errors:")
    for error in errors:
        print(f"  - {error}")
```

## Command-Line Interface

The configuration system includes a command-line interface for common operations:

### Viewing Configuration

```bash
python -m src.backend.core.config.cli view --config config.json
python -m src.backend.core.config.cli view --config config.json --section app
```

### Creating Default Configuration

```bash
python -m src.backend.core.config.cli create config.json
```

### Validating Configuration

```bash
python -m src.backend.core.config.cli validate config.json
```

### Editing Configuration

```bash
python -m src.backend.core.config.cli edit --config config.json --key app.log_level --value DEBUG
python -m src.backend.core.config.cli edit --config config.json --key models.default_model --value demucs
```

### Detecting Hardware

```bash
python -m src.backend.core.config.cli hardware
python -m src.backend.core.config.cli hardware --output hardware.json
```

### Managing ML Model Settings

```bash
# List available models
python -m src.backend.core.config.cli model list --config config.json

# View model settings
python -m src.backend.core.config.cli model view --config config.json --model svoice

# Check model compatibility
python -m src.backend.core.config.cli model compatibility --config config.json --model svoice

# Optimize model settings for hardware
python -m src.backend.core.config.cli model optimize --config config.json --model svoice --apply
```

## Configuration Structure

The configuration is structured as follows:

```json
{
  "app": {
    "name": "Voices",
    "version": "0.1.0",
    "log_level": "INFO",
    "data_dir": "/path/to/data",
    "temp_dir": "/path/to/temp",
    "max_threads": 4
  },
  "hardware": {
    "gpu_enabled": true,
    "gpu_memory": 8192,
    "gpu_compute_capability": 7.5,
    "cpu_cores": 8,
    "available_memory": 16384,
    "use_gpu": true,
    "use_mps": false,
    "precision": "float16"
  },
  "models": {
    "default_model": "svoice",
    "model_dir": "/path/to/models",
    "auto_download": true,
    "cache_models": true,
    "max_cache_size": 5120,
    "model_settings": {
      "svoice": {
        "enabled": true,
        "priority": 1,
        "parameters": {
          "n_speakers": 2,
          "sample_rate": 16000,
          "n_fft": 512,
          "hop_length": 128,
          "hidden_size": 128,
          "num_layers": 6,
          "bidirectional": true,
          "device": "cuda",
          "precision": "float16"
        },
        "hardware_requirements": {
          "min_gpu_memory": 2048,
          "min_compute_capability": 3.5,
          "min_cpu_cores": 2,
          "min_memory": 4096
        }
      },
      "demucs": {
        "enabled": true,
        "priority": 2,
        "parameters": {
          "n_speakers": 2,
          "sample_rate": 16000,
          "channels": 1,
          "hidden_size": 64,
          "depth": 6,
          "kernel_size": 8,
          "stride": 4,
          "lstm_layers": 2,
          "device": "cuda",
          "precision": "float16"
        },
        "hardware_requirements": {
          "min_gpu_memory": 4096,
          "min_compute_capability": 3.5,
          "min_cpu_cores": 4,
          "min_memory": 8192
        }
      }
    }
  },
  "processing": {
    "pipeline": {
      "loader": {
        "sample_rate": 16000,
        "mono": true
      },
      "preprocessor": {
        "chunk_size": 8192,
        "overlap": 0.25,
        "apply_vad": true
      },
      "separator": {
        "model_type": "auto",
        "num_speakers": 2
      },
      "postprocessor": {
        "apply_denoising": true,
        "apply_normalization": true
      },
      "formatter": {
        "output_dir": "/path/to/output",
        "format": "wav"
      }
    },
    "batch_size": 4,
    "num_workers": 2,
    "chunk_size": 8192,
    "overlap": 0.25,
    "sample_rate": 16000,
    "output_format": "wav",
    "apply_denoising": true,
    "apply_normalization": true
  },
  "experiment": {
    "results_dir": "/path/to/experiments",
    "save_metrics": true,
    "save_audio": true,
    "save_visualizations": true,
    "metrics": ["si_snri", "sdri", "pesq", "stoi"]
  },
  "ui": {
    "theme": "system",
    "language": "en",
    "show_advanced_options": false,
    "auto_update": true,
    "waveform_colors": {
      "mixture": "#3498db",
      "voice1": "#2ecc71",
      "voice2": "#e74c3c",
      "voice3": "#f39c12",
      "voice4": "#9b59b6"
    }
  }
}
```

## Environment Variables

The following environment variables can be used to override configuration settings:

- `VOICES_LOG_LEVEL`: Override the application log level
- `VOICES_DEV_MODE`: Set to "1" to enable development mode (sets log level to DEBUG)
- `VOICES_DATA_DIR`: Override the data directory
- `VOICES_DISABLE_GPU`: Set to "1" to disable GPU usage