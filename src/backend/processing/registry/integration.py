"""
Integration between Model Registry and Experiment Framework.

This module provides functionality for integrating the Model Registry
with the Experiment Framework, allowing for systematic evaluation and
comparison of voice separation models.
"""

import os
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np
import torch

from ..experiment.framework import ExperimentFramework, Experiment, ExperimentResult
from ..experiment.datasets import TestDataset
from .model_registry import ModelRegistry
from .model_adapters import create_model_adapter


class RegistryExperimentIntegration:
    """Integration between Model Registry and Experiment Framework."""
    
    def __init__(
        self,
        registry: ModelRegistry,
        experiment_framework: ExperimentFramework,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the integration.
        
        Args:
            registry: Model registry instance
            experiment_framework: Experiment framework instance
            logger: Logger instance
        """
        self.registry = registry
        self.framework = experiment_framework
        self.logger = logger or logging.getLogger(__name__)
    
    def register_models_with_framework(self) -> None:
        """
        Register all models in the registry with the experiment framework.
        
        This makes all models available for use in experiments.
        """
        models = self.registry.list_models()
        
        for model_info in models:
            model_id = model_info.model_id
            
            # Create a function that loads and uses the model
            def model_fn(mixture, **kwargs):
                return self.registry.get_model_function(model_id)(mixture, **kwargs)
            
            # Register with the framework
            self.framework.register_model(model_id, model_fn)
            
            self.logger.info(f"Registered model {model_id} with experiment framework")
    
    def create_comparison_experiment(
        self,
        name: str,
        description: str,
        dataset: Union[str, TestDataset],
        model_ids: Optional[List[str]] = None,
        model_types: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Experiment:
        """
        Create an experiment to compare models.
        
        Args:
            name: Name of the experiment
            description: Description of the experiment
            dataset: Dataset to use for evaluation
            model_ids: Specific model IDs to include (if None, use model_types)
            model_types: Types of models to include (if model_ids is None)
            metrics: Metrics to calculate (defaults to SI-SNRi and SDRi)
            parameters: Additional parameters for the experiment
        
        Returns:
            Experiment object
        """
        # Determine which models to include
        if model_ids is not None:
            # Use specific models
            models_to_use = model_ids
        elif model_types is not None:
            # Use all models of specified types
            models = self.registry.list_models()
            models_to_use = [
                m.model_id for m in models
                if m.model_type in model_types
            ]
        else:
            # Use all models
            models = self.registry.list_models()
            models_to_use = [m.model_id for m in models]
        
        if not models_to_use:
            raise ValueError("No models selected for experiment")
        
        # Set default metrics if not provided
        if metrics is None:
            metrics = ["si_snri_mean", "sdri_mean"]
        
        # Create experiment
        experiment = self.framework.create_experiment(
            name=name,
            description=description,
            models=models_to_use,
            dataset=dataset,
            metrics=metrics,
            parameters=parameters or {}
        )
        
        self.logger.info(f"Created comparison experiment: {name} with {len(models_to_use)} models")
        
        return experiment
    
    def run_comparison_experiment(
        self,
        experiment: Union[Experiment, str],
        output_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
        update_metrics: bool = True
    ) -> ExperimentResult:
        """
        Run a comparison experiment.
        
        Args:
            experiment: Experiment object or ID
            output_dir: Directory to save results
            device: Device to run on
            update_metrics: Whether to update model metrics in the registry
        
        Returns:
            ExperimentResult object
        """
        # Run the experiment
        result = self.framework.run_experiment(
            experiment=experiment,
            output_dir=output_dir,
            device=device
        )
        
        # Update metrics in registry if requested
        if update_metrics:
            for model_id, model_result in result.model_results.items():
                if model_result["status"] == "completed":
                    self.registry.update_metrics_from_experiment(
                        model_id=model_id,
                        version_id=None,  # Use default version
                        experiment_result=result
                    )
        
        self.logger.info(f"Completed experiment: {result.experiment.name}")
        
        return result
    
    def evaluate_model(
        self,
        model_id: str,
        version_id: Optional[str] = None,
        dataset: Union[str, TestDataset] = None,
        output_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
        update_metrics: bool = True
    ) -> ExperimentResult:
        """
        Evaluate a specific model.
        
        Args:
            model_id: ID of the model to evaluate
            version_id: ID of the version to evaluate (None for default)
            dataset: Dataset to use for evaluation
            output_dir: Directory to save results
            device: Device to run on
            update_metrics: Whether to update model metrics in the registry
        
        Returns:
            ExperimentResult object
        """
        # Get model info
        model_info = self.registry.get_model(model_id)
        if not model_info:
            raise ValueError(f"Model {model_id} not found in registry")
        
        # Get version info
        version = self.registry.get_model_version(model_id, version_id)
        if not version:
            raise ValueError(f"Version {version_id or 'default'} of model {model_id} not found")
        
        # Create experiment name
        experiment_name = f"Evaluation of {model_info.name}"
        if version_id:
            experiment_name += f" (version {version_id})"
        
        # Create experiment
        experiment = self.framework.create_experiment(
            name=experiment_name,
            description=f"Evaluation of {model_info.name} using {dataset}",
            models=[model_id],
            dataset=dataset,
            metrics=["si_snri_mean", "sdri_mean"]
        )
        
        # Run experiment
        result = self.run_comparison_experiment(
            experiment=experiment,
            output_dir=output_dir,
            device=device,
            update_metrics=update_metrics
        )
        
        return result
    
    def compare_model_types(
        self,
        model_types: List[str],
        dataset: Union[str, TestDataset],
        output_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
        update_metrics: bool = True
    ) -> ExperimentResult:
        """
        Compare different types of models (e.g., SVoice vs Demucs).
        
        Args:
            model_types: Types of models to compare
            dataset: Dataset to use for evaluation
            output_dir: Directory to save results
            device: Device to run on
            update_metrics: Whether to update model metrics in the registry
        
        Returns:
            ExperimentResult object
        """
        # Create experiment name
        experiment_name = f"Comparison of {' vs '.join(model_types)}"
        
        # Create experiment
        experiment = self.create_comparison_experiment(
            name=experiment_name,
            description=f"Comparison of different model types: {', '.join(model_types)}",
            dataset=dataset,
            model_types=model_types
        )
        
        # Run experiment
        result = self.run_comparison_experiment(
            experiment=experiment,
            output_dir=output_dir,
            device=device,
            update_metrics=update_metrics
        )
        
        return result
    
    def benchmark_models(
        self,
        model_ids: Optional[List[str]] = None,
        model_types: Optional[List[str]] = None,
        test_dataset: Optional[TestDataset] = None,
        output_dir: Optional[str] = None,
        num_runs: int = 3,
        use_gpu: bool = True
    ) -> Dict[str, Any]:
        """
        Benchmark models for performance.
        
        Args:
            model_ids: Specific model IDs to benchmark
            model_types: Types of models to benchmark
            test_dataset: Dataset to use for benchmarking
            output_dir: Directory to save results
            num_runs: Number of benchmark runs
            use_gpu: Whether to use GPU if available
        
        Returns:
            Dictionary of benchmark results
        """
        # Determine which models to benchmark
        if model_ids is not None:
            # Use specific models
            models_to_benchmark = model_ids
        elif model_types is not None:
            # Use all models of specified types
            models = self.registry.list_models()
            models_to_benchmark = [
                m.model_id for m in models
                if m.model_type in model_types
            ]
        else:
            # Use all models
            models = self.registry.list_models()
            models_to_benchmark = [m.model_id for m in models]
        
        if not models_to_benchmark:
            raise ValueError("No models selected for benchmarking")
        
        # Run benchmarks
        benchmark_results = self.framework.benchmark_models(
            models=models_to_benchmark,
            test_dataset=test_dataset,
            output_dir=output_dir,
            num_runs=num_runs,
            use_gpu=use_gpu
        )
        
        # Update model metadata with benchmark results
        for model_id, result in benchmark_results.items():
            model_version = self.registry.get_model_version(model_id)
            if model_version:
                # Add processing speed to metadata
                model_version.metadata["processing_speed"] = result.processing_speed
                model_version.metadata["benchmark_results"] = {
                    "cpu_utilization": result.cpu_utilization,
                    "memory_usage": result.memory_usage,
                    "gpu_utilization": result.gpu_utilization,
                    "gpu_memory_usage": result.gpu_memory_usage,
                    "hardware_info": result.hardware_info
                }
        
        # Save registry to persist benchmark results
        self.registry._save_registry()
        
        self.logger.info(f"Completed benchmarking of {len(models_to_benchmark)} models")
        
        return benchmark_results


def create_integration(
    registry_dir: str,
    experiment_dir: str,
    logger: Optional[logging.Logger] = None
) -> RegistryExperimentIntegration:
    """
    Create and initialize the integration between registry and experiment framework.
    
    Args:
        registry_dir: Directory for the model registry
        experiment_dir: Directory for the experiment framework
        logger: Logger instance
    
    Returns:
        RegistryExperimentIntegration instance
    """
    # Create logger if not provided
    if logger is None:
        logger = logging.getLogger("registry_experiment_integration")
    
    # Create registry
    registry = ModelRegistry(registry_dir, logger)
    
    # Register model loaders
    from .model_adapters import get_model_loader
    registry.register_model_loader("svoice", get_model_loader("svoice"))
    registry.register_model_loader("demucs", get_model_loader("demucs"))
    
    # Create experiment framework
    framework = ExperimentFramework(experiment_dir, logger=logger)
    
    # Create integration
    integration = RegistryExperimentIntegration(registry, framework, logger)
    
    # Register models with framework
    integration.register_models_with_framework()
    
    return integration