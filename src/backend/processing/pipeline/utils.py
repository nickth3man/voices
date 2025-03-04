"""
Utility functions for the audio processing pipeline.

This module provides utility functions for working with the audio processing pipeline,
including pipeline creation, configuration, and execution.
"""

import os
import logging
import json
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path

from .components import (
    AudioLoader,
    AudioPreprocessor,
    VoiceSeparator,
    AudioPostprocessor,
    AudioOutputFormatter,
    AudioProcessingPipeline,
    PipelineComponent
)

logger = logging.getLogger(__name__)


def create_pipeline_from_config(config: Dict[str, Any]) -> AudioProcessingPipeline:
    """
    Create a pipeline from a configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured audio processing pipeline
    """
    logger.info("Creating pipeline from configuration")
    
    # Extract component configurations
    loader_config = config.get("loader", {})
    preprocessor_config = config.get("preprocessor", {})
    separator_config = config.get("separator", {})
    postprocessor_config = config.get("postprocessor", {})
    formatter_config = config.get("formatter", {})
    
    # Create components
    components = [
        AudioLoader(**loader_config),
        AudioPreprocessor(**preprocessor_config),
        VoiceSeparator(**separator_config),
        AudioPostprocessor(**postprocessor_config),
        AudioOutputFormatter(**formatter_config)
    ]
    
    # Create pipeline
    pipeline = AudioProcessingPipeline(components=components)
    
    return pipeline


def load_pipeline_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load pipeline configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    logger.info(f"Loading pipeline configuration from {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def save_pipeline_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save pipeline configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration file
    """
    logger.info(f"Saving pipeline configuration to {config_path}")
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def create_default_pipeline_config() -> Dict[str, Any]:
    """
    Create a default pipeline configuration.
    
    Returns:
        Default configuration dictionary
    """
    logger.info("Creating default pipeline configuration")
    
    config = {
        "loader": {
            "sample_rate": 16000,
            "mono": True,
            "normalize": True
        },
        "preprocessor": {
            "chunk_size": None,
            "overlap": 0,
            "apply_vad": False
        },
        "separator": {
            "model_type": "auto",
            "model_id": None,
            "version_id": None,
            "num_speakers": None,
            "registry_dir": None
        },
        "postprocessor": {
            "apply_denoising": False,
            "apply_normalization": True
        },
        "formatter": {
            "output_dir": None,
            "output_format": "wav",
            "naming_pattern": "{base_name}_speaker{speaker_idx}.{ext}"
        }
    }
    
    return config


def process_directory(
    pipeline: AudioProcessingPipeline,
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    file_pattern: str = "*.wav",
    **kwargs
) -> Dict[str, Any]:
    """
    Process all audio files in a directory.
    
    Args:
        pipeline: Audio processing pipeline
        input_dir: Directory containing audio files
        output_dir: Directory to save processed files (if None, uses input_dir/processed)
        file_pattern: Pattern to match audio files
        **kwargs: Additional parameters passed to the pipeline
        
    Returns:
        Dictionary with results for each file
    """
    import glob
    
    logger.info(f"Processing audio files in {input_dir}")
    
    # Resolve paths
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir / "processed"
    else:
        output_dir = Path(output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find audio files
    file_pattern_path = str(input_dir / file_pattern)
    audio_files = glob.glob(file_pattern_path)
    
    logger.info(f"Found {len(audio_files)} audio files matching pattern {file_pattern}")
    
    # Process each file
    results = {}
    for audio_file in audio_files:
        file_name = os.path.basename(audio_file)
        logger.info(f"Processing {file_name}")
        
        try:
            # Set output directory for this file
            file_kwargs = kwargs.copy()
            file_kwargs["output_dir"] = str(output_dir)
            
            # Process file
            result = pipeline.process_file(audio_file, **file_kwargs)
            
            # Store result
            results[file_name] = {
                "status": "success",
                "output_paths": result.get("output_paths", {}),
                "metadata": result.get("metadata", {})
            }
            
        except Exception as e:
            logger.error(f"Error processing {file_name}: {str(e)}")
            results[file_name] = {
                "status": "error",
                "error": str(e)
            }
    
    return results


def evaluate_separation_quality(
    original_sources: Dict[str, np.ndarray],
    separated_sources: Dict[str, np.ndarray],
    sample_rate: int = 16000
) -> Dict[str, float]:
    """
    Evaluate the quality of voice separation.
    
    Args:
        original_sources: Dictionary mapping source names to original source audio
        separated_sources: Dictionary mapping source names to separated source audio
        sample_rate: Sample rate of the audio
        
    Returns:
        Dictionary with quality metrics
    """
    logger.info("Evaluating separation quality")
    
    # In a real implementation, we would use proper metrics like SI-SNR, SDR, etc.
    # For now, we'll just compute a simple SNR
    
    metrics = {}
    
    # Ensure we have the same sources
    common_sources = set(original_sources.keys()) & set(separated_sources.keys())
    
    for source in common_sources:
        original = original_sources[source]
        separated = separated_sources[source]
        
        # Ensure same length
        min_length = min(len(original), len(separated))
        original = original[:min_length]
        separated = separated[:min_length]
        
        # Compute error
        error = original - separated
        
        # Compute SNR
        signal_power = np.mean(original ** 2)
        noise_power = np.mean(error ** 2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = float('inf')
        
        metrics[f"{source}_snr"] = snr
    
    # Compute average SNR
    if common_sources:
        metrics["average_snr"] = sum(metrics[f"{source}_snr"] for source in common_sources) / len(common_sources)
    
    return metrics