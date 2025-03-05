"""
Test script for the Demucs model.

This module provides tests for the Demucs model implementation,
including model loading, inference, and integration with the abstraction layer.
"""

import os
import sys
import logging
import numpy as np
import torch
import unittest
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to allow importing from parent modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from src.backend.processing.models.demucs.model import DemucsModel
from src.backend.processing.models.demucs.utils import load_demucs_model, separate_sources
from src.backend.processing.models.abstraction import VoiceSeparationManager, ModelType


class TestDemucsModel(unittest.TestCase):
    """Test cases for the Demucs model."""
    
    def setUp(self):
        """Set up test environment."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Running tests on {self.device}")
        
        # Create a dummy audio signal
        self.sample_rate = 16000
        self.duration = 3  # seconds
        self.audio = np.random.randn(self.sample_rate * self.duration).astype(np.float32)
        
        # Create a model for testing
        self.model = DemucsModel(
            n_speakers=2,
            sample_rate=self.sample_rate,
            channels=1,
            hidden_size=32,  # Smaller for faster tests
            depth=4,         # Smaller for faster tests
            device=self.device
        )
    
    def test_model_creation(self):
        """Test model creation."""
        self.assertIsInstance(self.model, DemucsModel)
        self.assertEqual(self.model.n_speakers, 2)
        self.assertEqual(self.model.sample_rate, self.sample_rate)
    
    def test_model_forward(self):
        """Test model forward pass."""
        # Convert audio to tensor
        audio_tensor = torch.from_numpy(self.audio).float().to(self.device)
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        # Run forward pass
        with torch.no_grad():
            output = self.model(audio_tensor)
        
        # Check output shape
        self.assertEqual(output.shape[0], 1)  # Batch size
        self.assertEqual(output.shape[1], 2)  # Number of speakers
        self.assertEqual(output.shape[2], 1)  # Channels
        self.assertEqual(output.shape[3], self.audio.shape[0])  # Samples
    
    def test_model_separate(self):
        """Test model separate method."""
        # Separate audio
        sources = self.model.separate(self.audio)
        
        # Check output shape
        self.assertEqual(sources.shape[0], 2)  # Number of speakers
        self.assertEqual(sources.shape[1], self.audio.shape[0])  # Samples
        
        # Test with tensor input
        audio_tensor = torch.from_numpy(self.audio).float().to(self.device)
        sources_tensor = self.model.separate(audio_tensor)
        
        # Check output shape
        self.assertEqual(sources_tensor.shape[0], 2)  # Number of speakers
        self.assertEqual(sources_tensor.shape[1], self.audio.shape[0])  # Samples
    
    def test_model_info(self):
        """Test model info method."""
        info = self.model.get_model_info()
        
        # Check info contents
        self.assertEqual(info["type"], "demucs")
        self.assertEqual(info["name"], "Demucs")
        self.assertEqual(info["n_speakers"], 2)
        self.assertEqual(info["sample_rate"], self.sample_rate)
    
    def test_integration_with_abstraction_layer(self):
        """Test integration with the abstraction layer."""
        # Create a separation manager
        manager = VoiceSeparationManager(
            default_model_type=ModelType.DEMUCS,
            device=self.device
        )
        
        # Separate audio
        sources = manager.separate(self.audio, model_type=ModelType.DEMUCS)
        
        # Check output shape
        self.assertEqual(sources.shape[0], 2)  # Number of speakers
        self.assertEqual(sources.shape[1], self.audio.shape[0])  # Samples


class TestDemucsUtils(unittest.TestCase):
    """Test cases for the Demucs utility functions."""
    
    def setUp(self):
        """Set up test environment."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Running tests on {self.device}")
        
        # Create a dummy audio signal
        self.sample_rate = 16000
        self.duration = 3  # seconds
        self.audio = np.random.randn(self.sample_rate * self.duration).astype(np.float32)
        
        # Create a temporary directory for test files
        self.test_dir = os.path.join(os.path.dirname(__file__), "test_output")
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create a model for testing
        self.model = DemucsModel(
            n_speakers=2,
            sample_rate=self.sample_rate,
            channels=1,
            hidden_size=32,  # Smaller for faster tests
            depth=4,         # Smaller for faster tests
            device=self.device
        )
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test files
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_separate_sources(self):
        """Test separate_sources function."""
        # Create a temporary audio file
        import soundfile as sf
        audio_path = os.path.join(self.test_dir, "test_audio.wav")
        sf.write(audio_path, self.audio, self.sample_rate)
        
        # Separate sources
        output_dir = os.path.join(self.test_dir, "output")
        result = separate_sources(
            self.model,
            audio_path,
            output_dir=output_dir,
            num_speakers=2,
            sample_rate=self.sample_rate
        )
        
        # Check result
        self.assertEqual(len(result), 2)  # Two sources
        self.assertIn("source_1", result)
        self.assertIn("source_2", result)
        
        # Check output files
        self.assertTrue(os.path.exists(output_dir))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "test_audio_source_1.wav")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "test_audio_source_2.wav")))


if __name__ == "__main__":
    unittest.main()