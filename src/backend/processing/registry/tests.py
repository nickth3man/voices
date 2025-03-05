"""
Tests for the Model Registry System.

This module provides tests for the Model Registry System to ensure
it works correctly and handles edge cases properly.
"""

import os
import shutil
import unittest
import tempfile
import logging
import numpy as np
import torch
from pathlib import Path

from .model_registry import ModelRegistry
from .model_adapters import create_model_adapter, get_model_loader
from .model_selector import ModelSelector
from .integration import RegistryExperimentIntegration, create_integration


class TestModelRegistry(unittest.TestCase):
    """Tests for the ModelRegistry class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for registry
        self.temp_dir = tempfile.mkdtemp()
        self.registry_dir = os.path.join(self.temp_dir, "registry")
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("test_model_registry")
        
        # Create registry
        self.registry = ModelRegistry(self.registry_dir, self.logger)
        
        # Register model loaders
        self.registry.register_model_loader("svoice", get_model_loader("svoice"))
        self.registry.register_model_loader("demucs", get_model_loader("demucs"))
        
        # Create test model directories
        self.svoice_model_path = os.path.join(self.temp_dir, "svoice_model")
        self.demucs_model_path = os.path.join(self.temp_dir, "demucs_model")
        os.makedirs(self.svoice_model_path, exist_ok=True)
        os.makedirs(self.demucs_model_path, exist_ok=True)
        
        # Create dummy model files
        with open(os.path.join(self.svoice_model_path, "model.pt"), "w") as f:
            f.write("Dummy SVoice model file")
        
        with open(os.path.join(self.demucs_model_path, "model.pt"), "w") as f:
            f.write("Dummy Demucs model file")
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_add_model(self):
        """Test adding a model to the registry."""
        # Add SVoice model
        model_id, version_id = self.registry.add_model(
            name="Test SVoice",
            description="Test SVoice model",
            model_type="svoice",
            model_path=self.svoice_model_path,
            version_description="Test version",
            parameters={"num_speakers": 3},
            metadata={"test": True},
            tags=["test", "svoice"]
        )
        
        # Check if model was added
        self.assertIn(model_id, self.registry.models)
        
        # Check model properties
        model = self.registry.get_model(model_id)
        self.assertEqual(model.name, "Test SVoice")
        self.assertEqual(model.model_type, "svoice")
        self.assertEqual(model.default_version, version_id)
        
        # Check version properties
        version = model.versions[version_id]
        self.assertEqual(version.description, "Test version")
        self.assertEqual(version.parameters, {"num_speakers": 3})
        self.assertEqual(version.metadata, {"test": True})
        self.assertEqual(version.tags, ["test", "svoice"])
        self.assertTrue(version.is_default)
    
    def test_add_model_version(self):
        """Test adding a new version to an existing model."""
        # Add model
        model_id, version_id = self.registry.add_model(
            name="Test Model",
            description="Test model",
            model_type="svoice",
            model_path=self.svoice_model_path
        )
        
        # Add new version
        new_version_id = self.registry.add_model_version(
            model_id=model_id,
            model_path=self.svoice_model_path,
            version_description="New version",
            parameters={"improved": True},
            tags=["v2"],
            set_as_default=True
        )
        
        # Check if version was added
        model = self.registry.get_model(model_id)
        self.assertIn(new_version_id, model.versions)
        
        # Check if set as default
        self.assertEqual(model.default_version, new_version_id)
        
        # Check version properties
        version = model.versions[new_version_id]
        self.assertEqual(version.description, "New version")
        self.assertEqual(version.parameters, {"improved": True})
        self.assertEqual(version.tags, ["v2"])
        self.assertTrue(version.is_default)
        
        # Check that old version is no longer default
        old_version = model.versions[version_id]
        self.assertFalse(old_version.is_default)
    
    def test_list_models(self):
        """Test listing models in the registry."""
        # Add models
        svoice_id, _ = self.registry.add_model(
            name="SVoice Test",
            description="SVoice test model",
            model_type="svoice",
            model_path=self.svoice_model_path,
            tags=["test", "svoice"]
        )
        
        demucs_id, _ = self.registry.add_model(
            name="Demucs Test",
            description="Demucs test model",
            model_type="demucs",
            model_path=self.demucs_model_path,
            tags=["test", "demucs"]
        )
        
        # List all models
        all_models = self.registry.list_models()
        self.assertEqual(len(all_models), 2)
        self.assertIn(self.registry.get_model(svoice_id), all_models)
        self.assertIn(self.registry.get_model(demucs_id), all_models)
        
        # List by model type
        svoice_models = self.registry.list_models(model_type="svoice")
        self.assertEqual(len(svoice_models), 1)
        self.assertEqual(svoice_models[0].model_id, svoice_id)
        
        # List by tags
        demucs_tag_models = self.registry.list_models(tags=["demucs"])
        self.assertEqual(len(demucs_tag_models), 1)
        self.assertEqual(demucs_tag_models[0].model_id, demucs_id)
        
        # List by common tag
        test_tag_models = self.registry.list_models(tags=["test"])
        self.assertEqual(len(test_tag_models), 2)
    
    def test_delete_model(self):
        """Test deleting a model from the registry."""
        # Add model
        model_id, _ = self.registry.add_model(
            name="Test Model",
            description="Test model",
            model_type="svoice",
            model_path=self.svoice_model_path
        )
        
        # Delete model
        success = self.registry.delete_model(model_id)
        self.assertTrue(success)
        
        # Check if model was deleted
        self.assertNotIn(model_id, self.registry.models)
        self.assertIsNone(self.registry.get_model(model_id))
    
    def test_delete_model_version(self):
        """Test deleting a version from a model."""
        # Add model with initial version
        model_id, version_id1 = self.registry.add_model(
            name="Test Model",
            description="Test model",
            model_type="svoice",
            model_path=self.svoice_model_path
        )
        
        # Add second version
        version_id2 = self.registry.add_model_version(
            model_id=model_id,
            model_path=self.svoice_model_path,
            version_description="Second version"
        )
        
        # Delete first version
        success = self.registry.delete_model_version(model_id, version_id1)
        self.assertTrue(success)
        
        # Check if version was deleted
        model = self.registry.get_model(model_id)
        self.assertNotIn(version_id1, model.versions)
        self.assertIn(version_id2, model.versions)
        
        # Check if default version was updated
        self.assertEqual(model.default_version, version_id2)
    
    def test_load_model(self):
        """Test loading a model from the registry."""
        # Add model
        model_id, version_id = self.registry.add_model(
            name="Test Model",
            description="Test model",
            model_type="svoice",
            model_path=self.svoice_model_path
        )
        
        # Load model
        model_fn = self.registry.get_model_function(model_id)
        self.assertIsNotNone(model_fn)
        
        # Test model function with dummy audio
        audio = np.random.randn(16000)  # 1 second of audio
        output = model_fn(audio)
        
        # Check output shape (should be 2D: num_speakers x samples)
        self.assertEqual(len(output.shape), 2)
    
    def test_update_metrics(self):
        """Test updating model metrics from experiment results."""
        # Add model
        model_id, version_id = self.registry.add_model(
            name="Test Model",
            description="Test model",
            model_type="svoice",
            model_path=self.svoice_model_path
        )
        
        # Create mock experiment result
        from ..experiment.framework import ExperimentResult, Experiment
        
        class MockExperiment:
            def __init__(self):
                self.id = "test_experiment"
                self.name = "Test Experiment"
        
        experiment_result = ExperimentResult(
            experiment=MockExperiment(),
            model_results={
                model_id: {
                    "status": "completed",
                    "metrics": {
                        "si_snri_mean": 10.5,
                        "sdri_mean": 8.7
                    }
                }
            }
        )
        
        # Update metrics
        success = self.registry.update_metrics_from_experiment(
            model_id=model_id,
            version_id=version_id,
            experiment_result=experiment_result
        )
        
        self.assertTrue(success)
        
        # Check if metrics were updated
        version = self.registry.get_model_version(model_id, version_id)
        self.assertEqual(version.performance_metrics["si_snri_mean"], 10.5)
        self.assertEqual(version.performance_metrics["sdri_mean"], 8.7)
        
        # Check if experiment metadata was added
        self.assertIn("experiment_metadata", version.metadata)
        self.assertEqual(len(version.metadata["experiment_metadata"]), 1)
        self.assertEqual(
            version.metadata["experiment_metadata"][0]["experiment_id"],
            "test_experiment"
        )


class TestModelAdapters(unittest.TestCase):
    """Tests for the model adapters."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test model directories
        self.svoice_model_path = os.path.join(self.temp_dir, "svoice_model")
        self.demucs_model_path = os.path.join(self.temp_dir, "demucs_model")
        os.makedirs(self.svoice_model_path, exist_ok=True)
        os.makedirs(self.demucs_model_path, exist_ok=True)
        
        # Create dummy model files
        with open(os.path.join(self.svoice_model_path, "model.pt"), "w") as f:
            f.write("Dummy SVoice model file")
        
        with open(os.path.join(self.demucs_model_path, "model.pt"), "w") as f:
            f.write("Dummy Demucs model file")
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_create_adapters(self):
        """Test creating model adapters."""
        # Create SVoice adapter
        svoice_adapter = create_model_adapter(
            model_type="svoice",
            model_path=self.svoice_model_path
        )
        
        # Create Demucs adapter
        demucs_adapter = create_model_adapter(
            model_type="demucs",
            model_path=self.demucs_model_path
        )
        
        # Check adapter types
        from .model_adapters import SVoiceAdapter, DemucsAdapter
        self.assertIsInstance(svoice_adapter, SVoiceAdapter)
        self.assertIsInstance(demucs_adapter, DemucsAdapter)
    
    def test_adapter_separation(self):
        """Test separation using adapters."""
        # Create adapters
        svoice_adapter = create_model_adapter(
            model_type="svoice",
            model_path=self.svoice_model_path
        )
        
        demucs_adapter = create_model_adapter(
            model_type="demucs",
            model_path=self.demucs_model_path
        )
        
        # Create test audio
        audio = np.random.randn(16000)  # 1 second of audio
        
        # Test separation with SVoice
        svoice_output = svoice_adapter.separate(audio, num_speakers=3)
        self.assertEqual(svoice_output.shape[0], 3)  # 3 speakers
        self.assertEqual(svoice_output.shape[1], 16000)  # Same length as input
        
        # Test separation with Demucs
        demucs_output = demucs_adapter.separate(audio, num_speakers=2)
        self.assertEqual(demucs_output.shape[0], 2)  # 2 speakers
        self.assertEqual(demucs_output.shape[1], 16000)  # Same length as input
    
    def test_get_model_info(self):
        """Test getting model info from adapters."""
        # Create adapters
        svoice_adapter = create_model_adapter(
            model_type="svoice",
            model_path=self.svoice_model_path
        )
        
        demucs_adapter = create_model_adapter(
            model_type="demucs",
            model_path=self.demucs_model_path
        )
        
        # Get model info
        svoice_info = svoice_adapter.get_model_info()
        demucs_info = demucs_adapter.get_model_info()
        
        # Check info
        self.assertEqual(svoice_info["type"], "svoice")
        self.assertEqual(demucs_info["type"], "demucs")
        self.assertEqual(svoice_info["path"], self.svoice_model_path)
        self.assertEqual(demucs_info["path"], self.demucs_model_path)


class TestModelSelector(unittest.TestCase):
    """Tests for the ModelSelector class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for registry
        self.temp_dir = tempfile.mkdtemp()
        self.registry_dir = os.path.join(self.temp_dir, "registry")
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("test_model_selector")
        
        # Create registry
        self.registry = ModelRegistry(self.registry_dir, self.logger)
        
        # Register model loaders
        self.registry.register_model_loader("svoice", get_model_loader("svoice"))
        self.registry.register_model_loader("demucs", get_model_loader("demucs"))
        
        # Create test model directories
        self.svoice_model_path = os.path.join(self.temp_dir, "svoice_model")
        self.demucs_model_path = os.path.join(self.temp_dir, "demucs_model")
        os.makedirs(self.svoice_model_path, exist_ok=True)
        os.makedirs(self.demucs_model_path, exist_ok=True)
        
        # Create dummy model files
        with open(os.path.join(self.svoice_model_path, "model.pt"), "w") as f:
            f.write("Dummy SVoice model file")
        
        with open(os.path.join(self.demucs_model_path, "model.pt"), "w") as f:
            f.write("Dummy Demucs model file")
        
        # Add models to registry
        self.svoice_id, _ = self.registry.add_model(
            name="SVoice Test",
            description="SVoice test model",
            model_type="svoice",
            model_path=self.svoice_model_path,
            tags=["test", "svoice"]
        )
        
        self.demucs_id, _ = self.registry.add_model(
            name="Demucs Test",
            description="Demucs test model",
            model_type="demucs",
            model_path=self.demucs_model_path,
            tags=["test", "demucs"]
        )
        
        # Create selector
        self.selector = ModelSelector(self.registry)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_select_model(self):
        """Test selecting a model based on audio characteristics."""
        # Create test audio
        audio = np.random.randn(16000)  # 1 second of audio
        
        # Test selection with different parameters
        
        # Test with 3+ speakers (should prefer SVoice)
        model_id1, _ = self.selector.select_model(
            audio=audio,
            num_speakers=3
        )
        self.assertEqual(model_id1, self.svoice_id)
        
        # Test with 2 speakers (should prefer Demucs)
        model_id2, _ = self.selector.select_model(
            audio=audio,
            num_speakers=2
        )
        self.assertEqual(model_id2, self.demucs_id)
        
        # Test with noisy environment (should prefer SVoice)
        model_id3, _ = self.selector.select_model(
            audio=audio,
            environment="noisy"
        )
        self.assertEqual(model_id3, self.svoice_id)
        
        # Test with reverberant environment (should prefer Demucs)
        model_id4, _ = self.selector.select_model(
            audio=audio,
            environment="reverberant"
        )
        self.assertEqual(model_id4, self.demucs_id)
        
        # Test with explicit model type preference
        model_id5, _ = self.selector.select_model(
            audio=audio,
            model_type_preference="demucs"
        )
        self.assertEqual(model_id5, self.demucs_id)


if __name__ == "__main__":
    unittest.main()