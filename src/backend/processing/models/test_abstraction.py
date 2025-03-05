"""
Tests for the Voice Separation Model Abstraction Layer.

This module provides tests for the abstraction layer to ensure it works correctly
with different voice separation technologies.
"""

import os
import unittest
import numpy as np
import torch
from pathlib import Path

from .abstraction import (
    VoiceSeparationManager,
    AudioCharacteristics,
    ModelType,
    create_separation_manager
)


class TestVoiceSeparationAbstraction(unittest.TestCase):
    """Test cases for the voice separation abstraction layer."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a manager without a registry for basic tests
        self.manager = VoiceSeparationManager()
        
        # Create test audio (random noise)
        self.test_audio = np.random.randn(16000)  # 1 second at 16kHz
        self.test_audio_tensor = torch.from_numpy(self.test_audio).float()
    
    def test_model_selection(self):
        """Test model type selection based on audio characteristics."""
        # Test selection with many speakers (should prefer SVoice)
        chars = AudioCharacteristics(num_speakers=4)
        model_type = self.manager.select_model_type(chars)
        self.assertEqual(model_type, ModelType.SVOICE)
        
        # Test selection with noisy audio (should prefer Demucs)
        chars = AudioCharacteristics(is_noisy=True)
        model_type = self.manager.select_model_type(chars)
        self.assertEqual(model_type, ModelType.DEMUCS)
        
        # Test default selection (should be SVoice)
        chars = AudioCharacteristics()
        model_type = self.manager.select_model_type(chars)
        self.assertEqual(model_type, ModelType.SVOICE)
    
    def test_audio_characteristics_extraction(self):
        """Test extraction of audio characteristics."""
        chars = AudioCharacteristics.from_audio(self.test_audio, 16000)
        self.assertEqual(chars.duration, 1.0)  # 1 second
        self.assertEqual(chars.sample_rate, 16000)
    
    def test_separation_with_numpy(self):
        """Test separation with numpy array input."""
        # This will use the default SVoice model
        sources = self.manager.separate(self.test_audio, num_speakers=2)
        
        # Check output shape (2 speakers, same length as input)
        self.assertEqual(sources.shape, (2, len(self.test_audio)))
    
    def test_separation_with_tensor(self):
        """Test separation with torch tensor input."""
        # This will use the default SVoice model
        sources = self.manager.separate(self.test_audio_tensor, num_speakers=2)
        
        # Check output shape (2 speakers, same length as input)
        self.assertEqual(sources.shape, (2, len(self.test_audio)))
    
    def test_separation_with_model_type(self):
        """Test separation with specific model type."""
        # Test with SVoice
        sources = self.manager.separate(
            self.test_audio,
            num_speakers=2,
            model_type=ModelType.SVOICE
        )
        self.assertEqual(sources.shape, (2, len(self.test_audio)))
        
        # Test with Demucs
        sources = self.manager.separate(
            self.test_audio,
            num_speakers=2,
            model_type=ModelType.DEMUCS
        )
        self.assertEqual(sources.shape, (2, len(self.test_audio)))
    
    def test_factory_function(self):
        """Test the factory function for creating a manager."""
        # Create a manager with default settings
        manager = create_separation_manager()
        self.assertEqual(manager.default_model_type, ModelType.AUTO)
        
        # Create a manager with specific model type
        manager = create_separation_manager(default_model_type="svoice")
        self.assertEqual(manager.default_model_type, ModelType.SVOICE)
        
        # Create a manager with invalid model type (should default to AUTO)
        manager = create_separation_manager(default_model_type="invalid")
        self.assertEqual(manager.default_model_type, ModelType.AUTO)


if __name__ == "__main__":
    unittest.main()