"""
Example usage of the Model Registry System.

This module provides examples of how to use the Model Registry System
for tracking, versioning, and managing voice separation models.
"""

import os
import logging
import numpy as np
import torch
from pathlib import Path

from .model_registry import ModelRegistry
from .model_adapters import create_model_adapter, get_model_loader
from .model_selector import ModelSelector, select_best_model
from .integration import RegistryExperimentIntegration, create_integration
from ..experiment.framework import ExperimentFramework
from ..experiment.datasets import TestDataset, create_test_dataset


def setup_logging(log_file: str = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("model_registry_examples")
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def example_create_registry():
    """Example of creating and initializing a model registry."""
    # Set up logging
    logger = setup_logging()
    
    # Create registry directory
    registry_dir = Path("./model_registry")
    os.makedirs(registry_dir, exist_ok=True)
    
    # Create registry
    registry = ModelRegistry(registry_dir, logger)
    
    # Register model loaders
    registry.register_model_loader("svoice", get_model_loader("svoice"))
    registry.register_model_loader("demucs", get_model_loader("demucs"))
    
    logger.info("Created and initialized model registry")
    
    return registry


def example_add_models(registry: ModelRegistry):
    """Example of adding models to the registry."""
    # Add SVoice model
    svoice_model_path = "./models/svoice_model"
    os.makedirs(svoice_model_path, exist_ok=True)
    
    # Create a dummy model file
    with open(os.path.join(svoice_model_path, "model.pt"), "w") as f:
        f.write("Dummy SVoice model file")
    
    svoice_id, svoice_version = registry.add_model(
        name="SVoice Base",
        description="Base SVoice model for voice separation",
        model_type="svoice",
        model_path=svoice_model_path,
        version_description="Initial version",
        parameters={"num_speakers": 3},
        metadata={"author": "Example User", "date": "2025-03-04"},
        tags=["svoice", "base", "multi-speaker"]
    )
    
    # Add Demucs model
    demucs_model_path = "./models/demucs_model"
    os.makedirs(demucs_model_path, exist_ok=True)
    
    # Create a dummy model file
    with open(os.path.join(demucs_model_path, "model.pt"), "w") as f:
        f.write("Dummy Demucs model file")
    
    demucs_id, demucs_version = registry.add_model(
        name="Demucs v3",
        description="Demucs v3 model for voice separation",
        model_type="demucs",
        model_path=demucs_model_path,
        version_description="Initial version",
        parameters={"num_speakers": 2},
        metadata={"author": "Example User", "date": "2025-03-04"},
        tags=["demucs", "v3", "stereo"]
    )
    
    # Add performance metrics to models
    svoice_version_obj = registry.get_model_version(svoice_id, svoice_version)
    if svoice_version_obj:
        svoice_version_obj.performance_metrics = {
            "si_snri_mean": 12.5,
            "sdri_mean": 10.8
        }
    
    demucs_version_obj = registry.get_model_version(demucs_id, demucs_version)
    if demucs_version_obj:
        demucs_version_obj.performance_metrics = {
            "si_snri_mean": 11.2,
            "sdri_mean": 9.7
        }
    
    # Save registry to persist changes
    registry._save_registry()
    
    return svoice_id, demucs_id


def example_list_models(registry: ModelRegistry):
    """Example of listing models in the registry."""
    # List all models
    all_models = registry.list_models()
    print(f"Found {len(all_models)} models in the registry:")
    
    for model in all_models:
        print(f"- {model.name} ({model.model_id}): {model.description}")
        
        # Get default version
        default_version = model.get_default_version()
        if default_version:
            print(f"  Default version: {default_version.version_id}")
            
            # Show performance metrics if available
            if default_version.performance_metrics:
                print("  Performance metrics:")
                for metric, value in default_version.performance_metrics.items():
                    print(f"    {metric}: {value}")
    
    # List models by type
    svoice_models = registry.list_models(model_type="svoice")
    print(f"\nFound {len(svoice_models)} SVoice models:")
    for model in svoice_models:
        print(f"- {model.name} ({model.model_id})")
    
    # List models by tags
    tagged_models = registry.list_models(tags=["base"])
    print(f"\nFound {len(tagged_models)} models with 'base' tag:")
    for model in tagged_models:
        print(f"- {model.name} ({model.model_id})")


def example_model_selection(registry: ModelRegistry):
    """Example of selecting models based on audio characteristics."""
    # Create model selector
    selector = ModelSelector(registry)
    
    # Create dummy audio data
    sample_rate = 16000
    duration = 5  # seconds
    audio = np.random.randn(sample_rate * duration)
    
    # Select model for different scenarios
    
    # Scenario 1: 2 speakers, clean audio, balanced quality/speed
    model_id1, version_id1 = selector.select_model(
        audio=audio,
        sample_rate=sample_rate,
        num_speakers=2,
        environment="clean",
        quality_preference="balanced",
        speed_preference="balanced"
    )
    
    print(f"For 2 speakers, clean audio, balanced quality/speed:")
    print(f"Selected model: {model_id1}, version: {version_id1}")
    
    # Scenario 2: 4 speakers, noisy audio, highest quality
    model_id2, version_id2 = selector.select_model(
        audio=audio,
        sample_rate=sample_rate,
        num_speakers=4,
        environment="noisy",
        quality_preference="highest",
        speed_preference="balanced"
    )
    
    print(f"\nFor 4 speakers, noisy audio, highest quality:")
    print(f"Selected model: {model_id2}, version: {version_id2}")
    
    # Scenario 3: 2 speakers, reverberant audio, fastest processing
    model_id3, version_id3 = selector.select_model(
        audio=audio,
        sample_rate=sample_rate,
        num_speakers=2,
        environment="reverberant",
        quality_preference="balanced",
        speed_preference="fastest"
    )
    
    print(f"\nFor 2 speakers, reverberant audio, fastest processing:")
    print(f"Selected model: {model_id3}, version: {version_id3}")


def example_model_adapters():
    """Example of using model adapters."""
    # Create SVoice adapter
    svoice_adapter = create_model_adapter(
        model_type="svoice",
        model_path="./models/svoice_model",
        device=torch.device("cpu")
    )
    
    # Create Demucs adapter
    demucs_adapter = create_model_adapter(
        model_type="demucs",
        model_path="./models/demucs_model",
        device=torch.device("cpu")
    )
    
    # Create dummy audio data
    sample_rate = 16000
    duration = 5  # seconds
    audio = np.random.randn(sample_rate * duration)
    
    # Separate using SVoice
    print("Separating with SVoice adapter...")
    svoice_sources = svoice_adapter.separate(audio, num_speakers=3)
    print(f"SVoice output shape: {svoice_sources.shape}")
    
    # Separate using Demucs
    print("\nSeparating with Demucs adapter...")
    demucs_sources = demucs_adapter.separate(audio, num_speakers=2)
    print(f"Demucs output shape: {demucs_sources.shape}")
    
    # Get model info
    print("\nSVoice model info:")
    print(svoice_adapter.get_model_info())
    
    print("\nDemucs model info:")
    print(demucs_adapter.get_model_info())


def example_integration_with_experiment_framework():
    """Example of integrating with the experiment framework."""
    # Set up logging
    logger = setup_logging()
    
    # Create directories
    registry_dir = Path("./model_registry")
    experiment_dir = Path("./experiments")
    os.makedirs(registry_dir, exist_ok=True)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create integration
    integration = create_integration(
        registry_dir=str(registry_dir),
        experiment_dir=str(experiment_dir),
        logger=logger
    )
    
    # Add example models if registry is empty
    if not integration.registry.list_models():
        example_add_models(integration.registry)
    
    # Create a test dataset
    dataset_dir = Path("./datasets/test_dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create a source directory with dummy audio files
    source_dir = Path("./datasets/sources")
    os.makedirs(source_dir, exist_ok=True)
    
    # Create dummy source files
    for i in range(5):
        dummy_audio = np.random.randn(16000)  # 1 second of audio
        import soundfile as sf
        sf.write(source_dir / f"source_{i}.wav", dummy_audio, 16000)
    
    # Create test dataset
    test_dataset = create_test_dataset(
        name="Test Dataset",
        description="Test dataset for model comparison",
        source_dir=str(source_dir),
        output_dir=str(dataset_dir),
        num_speakers_range=(2, 4),
        duration_range=(1.0, 2.0),
        sample_rate=16000,
        num_items_per_category=2,
        difficulties=["easy", "medium"],
        environments=["clean", "noisy"]
    )
    
    # Compare SVoice and Demucs
    print("Comparing SVoice and Demucs models...")
    result = integration.compare_model_types(
        model_types=["svoice", "demucs"],
        dataset=test_dataset,
        output_dir=str(experiment_dir / "comparison"),
        update_metrics=True
    )
    
    # Print results
    print("\nComparison results:")
    for model_id, model_result in result.model_results.items():
        if model_result["status"] == "completed":
            print(f"Model: {model_id}")
            print(f"Metrics: {model_result.get('metrics', {})}")
            print()


def run_all_examples():
    """Run all examples."""
    print("=== Creating Model Registry ===")
    registry = example_create_registry()
    print("\n=== Adding Models ===")
    example_add_models(registry)
    print("\n=== Listing Models ===")
    example_list_models(registry)
    print("\n=== Model Selection ===")
    example_model_selection(registry)
    print("\n=== Model Adapters ===")
    example_model_adapters()
    print("\n=== Integration with Experiment Framework ===")
    example_integration_with_experiment_framework()


if __name__ == "__main__":
    run_all_examples()