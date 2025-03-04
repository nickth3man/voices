"""
Default configuration settings.

This module provides default configuration settings for the application,
including ML model settings and hardware-specific configurations.
"""

import os
import logging
from typing import Dict, Any
from pathlib import Path

from .hardware_detection import detect_hardware_capabilities

logger = logging.getLogger(__name__)


def get_default_config() -> Dict[str, Any]:
    """Get default configuration settings.

    Returns:
        Dict[str, Any]: Default configuration settings
    """
    # Detect hardware capabilities
    hardware_capabilities = detect_hardware_capabilities()
    
    # Get base directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    data_dir = os.path.join(base_dir, 'data')
    
    # Create default configuration
    config = {
        "app": {
            "name": "Voices",
            "version": "0.1.0",
            "log_level": "INFO",
            "data_dir": data_dir,
            "temp_dir": os.path.join(data_dir, 'temp'),
            "max_threads": max(1, os.cpu_count() or 1)
        },
        "hardware": {
            "gpu_enabled": hardware_capabilities["gpu_enabled"],
            "gpu_memory": hardware_capabilities["gpu_memory"],
            "gpu_compute_capability": hardware_capabilities["gpu_compute_capability"],
            "cpu_cores": hardware_capabilities["cpu_cores"],
            "available_memory": hardware_capabilities["available_memory"],
            "use_gpu": hardware_capabilities["use_gpu"],
            "use_mps": hardware_capabilities["use_mps"],
            "precision": hardware_capabilities["recommended_precision"]
        },
        "models": {
            "default_model": "svoice",
            "model_dir": os.path.join(data_dir, 'models'),
            "auto_download": True,
            "cache_models": True,
            "max_cache_size": 5120,  # 5GB
            "model_settings": get_default_model_settings(hardware_capabilities)
        },
        "processing": {
            "pipeline": {
                "loader": {
                    "sample_rate": 16000,
                    "mono": True
                },
                "preprocessor": {
                    "chunk_size": 8192,
                    "overlap": 0.25,
                    "apply_vad": True
                },
                "separator": {
                    "model_type": "auto",
                    "num_speakers": 2
                },
                "postprocessor": {
                    "apply_denoising": True,
                    "apply_normalization": True
                },
                "formatter": {
                    "output_dir": os.path.join(data_dir, 'output'),
                    "format": "wav"
                }
            },
            "batch_size": hardware_capabilities["recommended_batch_size"],
            "num_workers": hardware_capabilities["recommended_num_workers"],
            "chunk_size": 8192,
            "overlap": 0.25,
            "sample_rate": 16000,
            "output_format": "wav",
            "apply_denoising": True,
            "apply_normalization": True
        },
        "experiment": {
            "results_dir": os.path.join(data_dir, 'experiments'),
            "save_metrics": True,
            "save_audio": True,
            "save_visualizations": True,
            "metrics": ["si_snri", "sdri", "pesq", "stoi"]
        },
        "ui": {
            "theme": "system",
            "language": "en",
            "show_advanced_options": False,
            "auto_update": True,
            "waveform_colors": {
                "mixture": "#3498db",
                "voice1": "#2ecc71",
                "voice2": "#e74c3c",
                "voice3": "#f39c12",
                "voice4": "#9b59b6"
            }
        }
    }
    
    return config


def get_default_model_settings(hardware_capabilities: Dict[str, Any]) -> Dict[str, Any]:
    """Get default settings for ML models based on hardware capabilities.

    Args:
        hardware_capabilities: Dictionary containing hardware capabilities

    Returns:
        Dict[str, Any]: Default model settings
    """
    # Base settings for all models
    use_gpu = hardware_capabilities.get("use_gpu", False)
    use_mps = hardware_capabilities.get("use_mps", False)
    precision = hardware_capabilities.get("recommended_precision", "float32")
    
    # Device selection
    device = "cpu"
    if use_gpu:
        device = "cuda"
    elif use_mps:
        device = "mps"
    
    # Default settings
    model_settings = {
        "svoice": {
            "enabled": True,
            "priority": 1,
            "parameters": {
                "n_speakers": 2,
                "sample_rate": 16000,
                "n_fft": 512,
                "hop_length": 128,
                "hidden_size": 128,
                "num_layers": 6,
                "bidirectional": True,
                "device": device,
                "precision": precision
            },
            "hardware_requirements": {
                "min_gpu_memory": 2048,  # 2GB
                "min_compute_capability": 3.5,
                "min_cpu_cores": 2,
                "min_memory": 4096  # 4GB
            }
        },
        "demucs": {
            "enabled": True,
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
                "device": device,
                "precision": precision
            },
            "hardware_requirements": {
                "min_gpu_memory": 4096,  # 4GB
                "min_compute_capability": 3.5,
                "min_cpu_cores": 4,
                "min_memory": 8192  # 8GB
            }
        }
    }
    
    # Optimize settings based on hardware
    if use_gpu:
        # GPU optimizations
        gpu_memory = hardware_capabilities.get("gpu_memory", 0)
        
        # SVoice optimizations
        if gpu_memory >= 8192:  # 8GB+
            model_settings["svoice"]["parameters"]["hidden_size"] = 256
            model_settings["svoice"]["parameters"]["num_layers"] = 8
        
        # Demucs optimizations
        if gpu_memory >= 8192:  # 8GB+
            model_settings["demucs"]["parameters"]["hidden_size"] = 128
            model_settings["demucs"]["parameters"]["depth"] = 8
    
    return model_settings


def get_environment_specific_config() -> Dict[str, Any]:
    """Get environment-specific configuration overrides.

    Returns:
        Dict[str, Any]: Environment-specific configuration
    """
    env_config = {}
    
    # Check for environment variables
    log_level = os.environ.get("VOICES_LOG_LEVEL")
    if log_level:
        env_config["app"] = {"log_level": log_level}
    
    # Check for development mode
    if os.environ.get("VOICES_DEV_MODE") == "1":
        env_config["app"] = env_config.get("app", {})
        env_config["app"]["log_level"] = "DEBUG"
    
    # Check for data directory override
    data_dir = os.environ.get("VOICES_DATA_DIR")
    if data_dir:
        env_config["app"] = env_config.get("app", {})
        env_config["app"]["data_dir"] = data_dir
    
    # Check for GPU disable flag
    if os.environ.get("VOICES_DISABLE_GPU") == "1":
        env_config["hardware"] = {"use_gpu": False, "use_mps": False}
    
    return env_config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries.

    Args:
        base_config: Base configuration
        override_config: Configuration to override base

    Returns:
        Dict[str, Any]: Merged configuration
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            # Recursively merge nested dictionaries
            result[key] = merge_configs(result[key], value)
        else:
            # Override or add value
            result[key] = value
    
    return result