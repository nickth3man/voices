"""
Test script for SVoice model integration.

This script provides a simple way to test the SVoice model integration
by creating a dummy audio file and running it through the model.
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
import soundfile as sf
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from processing.models.svoice.model import SVoiceModel
from processing.models.svoice.utils import load_svoice_model, separate_sources
from processing.registry.model_adapters import SVoiceAdapter, get_model_loader
from processing.registry.model_registry import ModelRegistry


def setup_logger():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('svoice_test.log')
        ]
    )
    return logging.getLogger(__name__)


def create_test_audio(output_path, duration=3.0, sample_rate=16000):
    """
    Create a test audio file with mixed sources.
    
    Args:
        output_path: Path to save the audio file
        duration: Duration of the audio in seconds
        sample_rate: Sample rate of the audio
        
    Returns:
        Path to the created audio file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate sample audio (mixture of sine waves)
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    
    # Create two "speakers" with different frequencies
    source1 = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz
    source2 = 0.5 * np.sin(2 * np.pi * 880 * t)  # 880 Hz
    
    # Mix the sources
    mixture = source1 + source2
    
    # Normalize
    mixture = mixture / np.max(np.abs(mixture))
    
    # Save to file
    sf.write(output_path, mixture, sample_rate)
    
    return output_path


def test_svoice_model():
    """Test the SVoice model directly."""
    logger = setup_logger()
    logger.info("Testing SVoice model directly")
    
    # Create a test audio file
    test_audio_path = "test_audio.wav"
    create_test_audio(test_audio_path)
    
    # Create a model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SVoiceModel(n_speakers=2, device=device)
    
    # Load audio
    audio, sample_rate = sf.read(test_audio_path)
    
    # Separate sources
    logger.info("Separating sources using SVoice model")
    sources = model.separate(audio)
    
    # Save separated sources
    os.makedirs("separated", exist_ok=True)
    for i, source in enumerate(sources):
        output_path = f"separated/source_{i+1}.wav"
        sf.write(output_path, source, sample_rate)
        logger.info(f"Saved separated source to {output_path}")
    
    logger.info("SVoice model test completed successfully")


def test_svoice_adapter():
    """Test the SVoice adapter."""
    logger = setup_logger()
    logger.info("Testing SVoice adapter")
    
    # Create a test audio file
    test_audio_path = "test_audio.wav"
    create_test_audio(test_audio_path)
    
    # Create an adapter
    adapter = SVoiceAdapter("dummy_model_path")
    
    # Load audio
    audio, sample_rate = sf.read(test_audio_path)
    
    # Separate sources
    logger.info("Separating sources using SVoice adapter")
    sources = adapter.separate(audio)
    
    # Save separated sources
    os.makedirs("separated_adapter", exist_ok=True)
    for i, source in enumerate(sources):
        output_path = f"separated_adapter/source_{i+1}.wav"
        sf.write(output_path, source, sample_rate)
        logger.info(f"Saved separated source to {output_path}")
    
    # Get model info
    model_info = adapter.get_model_info()
    logger.info(f"Model info: {model_info}")
    
    logger.info("SVoice adapter test completed successfully")


def test_model_registry():
    """Test the model registry with SVoice model."""
    logger = setup_logger()
    logger.info("Testing model registry with SVoice model")
    
    # Create a test audio file
    test_audio_path = "test_audio.wav"
    create_test_audio(test_audio_path)
    
    # Create a model registry
    registry_dir = "model_registry"
    os.makedirs(registry_dir, exist_ok=True)
    registry = ModelRegistry(registry_dir)
    
    # Register the SVoice model loader
    svoice_loader = get_model_loader("svoice")
    registry.register_model_loader("svoice", svoice_loader)
    
    # Add a dummy model to the registry
    model_id, version_id = registry.add_model(
        name="SVoice Test",
        description="Test SVoice model",
        model_type="svoice",
        model_path="dummy_model_path",
        version_description="Initial version",
        parameters={"n_speakers": 2},
        metadata={"test": True},
        tags=["test"]
    )
    
    # Load the model
    logger.info(f"Loading model {model_id} version {version_id}")
    process_func = registry.load_model(model_id)
    
    # Load audio
    audio, sample_rate = sf.read(test_audio_path)
    
    # Separate sources
    logger.info("Separating sources using model from registry")
    sources = process_func(audio)
    
    # Save separated sources
    os.makedirs("separated_registry", exist_ok=True)
    for i, source in enumerate(sources):
        output_path = f"separated_registry/source_{i+1}.wav"
        sf.write(output_path, source, sample_rate)
        logger.info(f"Saved separated source to {output_path}")
    
    logger.info("Model registry test completed successfully")


def main():
    """Main entry point for the test script."""
    parser = argparse.ArgumentParser(description='Test SVoice model integration')
    parser.add_argument('--test-type', type=str, default='all', choices=['model', 'adapter', 'registry', 'all'],
                        help='Type of test to run')
    args = parser.parse_args()
    
    if args.test_type == 'model' or args.test_type == 'all':
        test_svoice_model()
    
    if args.test_type == 'adapter' or args.test_type == 'all':
        test_svoice_adapter()
    
    if args.test_type == 'registry' or args.test_type == 'all':
        test_model_registry()


if __name__ == '__main__':
    main()