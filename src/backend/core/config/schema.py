"""
Configuration schema definition and validation.

This module defines the schema for configuration files and provides
validation functions to ensure configuration correctness.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import jsonschema

logger = logging.getLogger(__name__)


class ConfigSchema:
    """Configuration schema definitions and utilities."""

    # Base schema for all configurations
    BASE_SCHEMA = {
        "type": "object",
        "properties": {
            "app": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "version": {"type": "string"},
                    "log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
                    "data_dir": {"type": "string"},
                    "temp_dir": {"type": "string"},
                    "max_threads": {"type": "integer", "minimum": 1},
                },
                "required": ["name", "version", "log_level"]
            },
            "hardware": {
                "type": "object",
                "properties": {
                    "gpu_enabled": {"type": "boolean"},
                    "gpu_memory": {"type": "integer", "minimum": 0},
                    "gpu_compute_capability": {"type": "number", "minimum": 0},
                    "cpu_cores": {"type": "integer", "minimum": 1},
                    "available_memory": {"type": "integer", "minimum": 0},
                    "use_gpu": {"type": "boolean"},
                    "use_mps": {"type": "boolean"},  # For Apple Silicon
                    "precision": {"type": "string", "enum": ["float32", "float16", "bfloat16", "int8"]}
                },
                "required": ["gpu_enabled", "use_gpu"]
            },
            "models": {
                "type": "object",
                "properties": {
                    "default_model": {"type": "string"},
                    "model_dir": {"type": "string"},
                    "auto_download": {"type": "boolean"},
                    "cache_models": {"type": "boolean"},
                    "max_cache_size": {"type": "integer", "minimum": 0},
                    "model_settings": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "priority": {"type": "integer", "minimum": 0},
                                "parameters": {"type": "object"},
                                "hardware_requirements": {
                                    "type": "object",
                                    "properties": {
                                        "min_gpu_memory": {"type": "integer", "minimum": 0},
                                        "min_compute_capability": {"type": "number", "minimum": 0},
                                        "min_cpu_cores": {"type": "integer", "minimum": 1},
                                        "min_memory": {"type": "integer", "minimum": 0}
                                    }
                                }
                            },
                            "required": ["enabled"]
                        }
                    }
                },
                "required": ["default_model", "model_dir"]
            },
            "processing": {
                "type": "object",
                "properties": {
                    "pipeline": {
                        "type": "object",
                        "properties": {
                            "loader": {"type": "object"},
                            "preprocessor": {"type": "object"},
                            "separator": {"type": "object"},
                            "postprocessor": {"type": "object"},
                            "formatter": {"type": "object"}
                        }
                    },
                    "batch_size": {"type": "integer", "minimum": 1},
                    "num_workers": {"type": "integer", "minimum": 0},
                    "chunk_size": {"type": "integer", "minimum": 0},
                    "overlap": {"type": "number", "minimum": 0, "maximum": 1},
                    "sample_rate": {"type": "integer", "enum": [8000, 16000, 22050, 44100, 48000]},
                    "output_format": {"type": "string", "enum": ["wav", "mp3", "flac", "ogg"]},
                    "apply_denoising": {"type": "boolean"},
                    "apply_normalization": {"type": "boolean"}
                }
            },
            "experiment": {
                "type": "object",
                "properties": {
                    "results_dir": {"type": "string"},
                    "save_metrics": {"type": "boolean"},
                    "save_audio": {"type": "boolean"},
                    "save_visualizations": {"type": "boolean"},
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            },
            "ui": {
                "type": "object",
                "properties": {
                    "theme": {"type": "string", "enum": ["light", "dark", "system"]},
                    "language": {"type": "string"},
                    "show_advanced_options": {"type": "boolean"},
                    "auto_update": {"type": "boolean"},
                    "waveform_colors": {
                        "type": "object",
                        "additionalProperties": {"type": "string"}
                    }
                }
            }
        },
        "required": ["app", "hardware", "models", "processing"]
    }

    # Schema specifically for ML model settings
    ML_MODEL_SCHEMA = {
        "type": "object",
        "properties": {
            "svoice": {
                "type": "object",
                "properties": {
                    "n_speakers": {"type": "integer", "minimum": 1, "maximum": 10},
                    "sample_rate": {"type": "integer", "enum": [8000, 16000, 22050, 44100, 48000]},
                    "n_fft": {"type": "integer", "minimum": 64},
                    "hop_length": {"type": "integer", "minimum": 16},
                    "hidden_size": {"type": "integer", "minimum": 32},
                    "num_layers": {"type": "integer", "minimum": 1},
                    "bidirectional": {"type": "boolean"},
                    "checkpoint": {"type": "string"},
                    "precision": {"type": "string", "enum": ["float32", "float16", "bfloat16", "int8"]},
                    "hardware_requirements": {
                        "type": "object",
                        "properties": {
                            "min_gpu_memory": {"type": "integer", "default": 2048},
                            "min_compute_capability": {"type": "number", "default": 3.5},
                            "min_cpu_cores": {"type": "integer", "default": 2},
                            "min_memory": {"type": "integer", "default": 4096}
                        }
                    }
                }
            },
            "demucs": {
                "type": "object",
                "properties": {
                    "n_speakers": {"type": "integer", "minimum": 1, "maximum": 10},
                    "sample_rate": {"type": "integer", "enum": [8000, 16000, 22050, 44100, 48000]},
                    "channels": {"type": "integer", "minimum": 1, "maximum": 2},
                    "hidden_size": {"type": "integer", "minimum": 32},
                    "depth": {"type": "integer", "minimum": 1},
                    "kernel_size": {"type": "integer", "minimum": 1},
                    "stride": {"type": "integer", "minimum": 1},
                    "lstm_layers": {"type": "integer", "minimum": 0},
                    "checkpoint": {"type": "string"},
                    "precision": {"type": "string", "enum": ["float32", "float16", "bfloat16", "int8"]},
                    "hardware_requirements": {
                        "type": "object",
                        "properties": {
                            "min_gpu_memory": {"type": "integer", "default": 4096},
                            "min_compute_capability": {"type": "number", "default": 3.5},
                            "min_cpu_cores": {"type": "integer", "default": 4},
                            "min_memory": {"type": "integer", "default": 8192}
                        }
                    }
                }
            }
        }
    }

    @classmethod
    def get_schema(cls) -> Dict[str, Any]:
        """Get the complete configuration schema.

        Returns:
            Dict[str, Any]: The complete configuration schema
        """
        return cls.BASE_SCHEMA

    @classmethod
    def get_ml_model_schema(cls) -> Dict[str, Any]:
        """Get the ML model configuration schema.

        Returns:
            Dict[str, Any]: The ML model configuration schema
        """
        return cls.ML_MODEL_SCHEMA


def validate_config(config: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> List[str]:
    """Validate a configuration against the schema.

    Args:
        config: The configuration to validate
        schema: Optional schema to validate against (uses default if None)

    Returns:
        List[str]: List of validation errors (empty if valid)
    """
    if schema is None:
        schema = ConfigSchema.get_schema()

    validator = jsonschema.Draft7Validator(schema)
    errors = list(validator.iter_errors(config))
    
    error_messages = []
    for error in errors:
        path = ".".join(str(p) for p in error.path) if error.path else "root"
        error_messages.append(f"Error at {path}: {error.message}")
    
    return error_messages


def load_schema_from_file(schema_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a schema from a JSON file.

    Args:
        schema_path: Path to the schema file

    Returns:
        Dict[str, Any]: The loaded schema
    """
    with open(schema_path, 'r') as f:
        return json.load(f)