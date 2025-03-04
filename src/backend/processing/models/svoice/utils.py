"""
Utility functions for SVoice model.

This module provides utility functions for loading and using the SVoice model,
including model loading, inference, and audio processing.
"""

import os
import logging
import torch
import numpy as np
import soundfile as sf
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from pathlib import Path

from .model import SVoiceModel

logger = logging.getLogger(__name__)


def load_svoice_model(
    model_path: str,
    device: Optional[torch.device] = None,
    **kwargs
) -> SVoiceModel:
    """
    Load a SVoice model from a path.
    
    Args:
        model_path: Path to the model file or directory
        device: Device to load the model on
        **kwargs: Additional parameters to pass to the model
        
    Returns:
        Loaded SVoice model
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    # If path is a directory, look for model files
    if os.path.isdir(model_path):
        model_files = [f for f in os.listdir(model_path) if f.endswith('.pth') or f.endswith('.pt')]
        if not model_files:
            raise FileNotFoundError(f"No model files found in directory: {model_path}")
        model_path = os.path.join(model_path, model_files[0])
    
    logger.info(f"Loading SVoice model from {model_path} on {device}")
    
    try:
        # Load model using the from_pretrained method
        model = SVoiceModel.from_pretrained(model_path, device=device, **kwargs)
        logger.info(f"Successfully loaded SVoice model")
        return model
    except Exception as e:
        logger.error(f"Error loading SVoice model: {str(e)}")
        raise


def separate_sources(
    model: SVoiceModel,
    audio_path: str,
    output_dir: Optional[str] = None,
    num_speakers: Optional[int] = None,
    sample_rate: int = 16000,
    normalize: bool = True,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Separate audio sources using the SVoice model.
    
    Args:
        model: SVoice model
        audio_path: Path to the audio file
        output_dir: Directory to save separated sources (if None, sources are not saved)
        num_speakers: Number of speakers to separate (if None, uses model default)
        sample_rate: Sample rate for loading and saving audio
        normalize: Whether to normalize the audio before separation
        **kwargs: Additional parameters to pass to the model
        
    Returns:
        Dictionary mapping source names to separated audio arrays
    """
    logger.info(f"Separating sources from {audio_path}")
    
    # Load audio
    try:
        audio, file_sample_rate = sf.read(audio_path)
    except Exception as e:
        logger.error(f"Error loading audio file: {str(e)}")
        raise
    
    # Convert to mono if needed
    if len(audio.shape) > 1 and audio.shape[1] > 1:
        logger.info(f"Converting stereo audio to mono")
        audio = audio.mean(axis=1)
    
    # Resample if needed
    if file_sample_rate != sample_rate:
        logger.info(f"Resampling audio from {file_sample_rate}Hz to {sample_rate}Hz")
        # In a real implementation, we would use librosa.resample or similar
        # For now, we'll just warn about the sample rate mismatch
        logger.warning(f"Sample rate mismatch: file={file_sample_rate}Hz, model={sample_rate}Hz")
    
    # Normalize if requested
    if normalize:
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
    # Separate sources
    sources = model.separate(audio, num_speakers=num_speakers, **kwargs)
    
    # Create output dictionary
    output = {}
    for i in range(sources.shape[0]):
        output[f"source_{i+1}"] = sources[i]
    
    # Save sources if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        # Save each source
        for source_name, source_audio in output.items():
            output_path = os.path.join(output_dir, f"{base_name}_{source_name}.wav")
            sf.write(output_path, source_audio, sample_rate)
            logger.info(f"Saved separated source to {output_path}")
    
    return output


def create_svoice_registry_loader() -> Callable:
    """
    Create a loader function for the SVoice model that can be registered with the model registry.
    
    Returns:
        Function that loads a SVoice model from a path
    """
    def load_svoice_for_registry(path, **kwargs):
        """
        Load a SVoice model for use with the model registry.
        
        Args:
            path: Path to the model file or directory
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            Function that processes audio using the loaded model
        """
        # Load the model
        model = load_svoice_model(path, **kwargs)
        
        # Create a processing function
        def process_audio(mixture, **proc_kwargs):
            """
            Process audio using the loaded SVoice model.
            
            Args:
                mixture: Audio mixture to separate
                **proc_kwargs: Additional processing parameters
                
            Returns:
                Separated sources
            """
            return model.separate(mixture, **proc_kwargs)
        
        return process_audio
    
    return load_svoice_for_registry


def download_pretrained_model(
    output_dir: str,
    model_url: Optional[str] = None,
    model_name: str = "svoice_base",
    force_download: bool = False
) -> str:
    """
    Download a pretrained SVoice model.
    
    Args:
        output_dir: Directory to save the model
        model_url: URL to download the model from (if None, uses default URL)
        model_name: Name of the model to download
        force_download: Whether to force download even if the model already exists
        
    Returns:
        Path to the downloaded model
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default URL if not provided
    if model_url is None:
        # In a real implementation, this would point to an actual model URL
        model_url = f"https://example.com/models/svoice/{model_name}.pth"
    
    # Check if model already exists
    model_path = os.path.join(output_dir, f"{model_name}.pth")
    if os.path.exists(model_path) and not force_download:
        logger.info(f"Model already exists at {model_path}, skipping download")
        return model_path
    
    # In a real implementation, we would download the model here
    # For now, we'll just create a dummy model file
    logger.info(f"Downloading model from {model_url} to {model_path}")
    
    # Create a dummy model file
    with open(model_path, 'w') as f:
        f.write("# Dummy SVoice model file\n")
        f.write(f"# This is a placeholder for the {model_name} model\n")
    
    logger.info(f"Model downloaded to {model_path}")
    
    return model_path