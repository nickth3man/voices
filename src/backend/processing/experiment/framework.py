"""
ML Experimentation Framework for Voice Separation Models.

This module provides the core framework for systematic evaluation of voice
separation models, including experiment definition, execution, and result tracking.
"""

import os
import json
import yaml
import time
import datetime
import logging
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch

from .metrics import calculate_metrics
from .datasets import TestDataset, TestDatasetItem
from .visualization import visualize_results, generate_comparison_chart, create_report
from .benchmarking import benchmark_model, ModelBenchmarkResult, compare_benchmarks


@dataclass
class Experiment:
    """Definition of a voice separation experiment."""
    
    id: str
    name: str
    description: str
    models: List[Dict[str, Any]]
    dataset: Union[str, TestDataset]
    metrics: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    visualization: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "models": self.models,
            "metrics": self.metrics,
            "parameters": self.parameters,
            "visualization": self.visualization,
            "metadata": self.metadata
        }
        
        # Handle dataset
        if isinstance(self.dataset, str):
            data["dataset"] = self.dataset
        else:
            data["dataset"] = f"TestDataset({self.dataset.name}, {len(self.dataset.items)} items)"
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], dataset: Optional[TestDataset] = None) -> 'Experiment':
        """Create from dictionary."""
        # Handle dataset
        if dataset is None and not isinstance(data["dataset"], str):
            raise ValueError("Dataset must be provided if not a string path")
        
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            models=data["models"],
            dataset=dataset if dataset is not None else data["dataset"],
            metrics=data["metrics"],
            parameters=data.get("parameters", {}),
            visualization=data.get("visualization", {}),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def from_yaml(cls, path: str, dataset: Optional[TestDataset] = None) -> 'Experiment':
        """Load experiment definition from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data, dataset)
    
    def save_yaml(self, path: str) -> None:
        """Save experiment definition to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


@dataclass
class ExperimentResult:
    """Results from a voice separation experiment."""
    
    experiment: Experiment
    model_results: Dict[str, Dict[str, Any]]
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime] = None
    status: str = "running"  # "running", "completed", "failed"
    error: Optional[str] = None
    output_dir: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment": self.experiment.to_dict(),
            "model_results": self.model_results,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "error": self.error,
            "output_dir": self.output_dir
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], experiment: Optional[Experiment] = None) -> 'ExperimentResult':
        """Create from dictionary."""
        return cls(
            experiment=experiment if experiment is not None else Experiment.from_dict(data["experiment"]),
            model_results=data["model_results"],
            start_time=datetime.datetime.fromisoformat(data["start_time"]),
            end_time=datetime.datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            status=data["status"],
            error=data.get("error"),
            output_dir=data.get("output_dir")
        )
    
    def save(self, path: str) -> None:
        """Save experiment results to a file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str, experiment: Optional[Experiment] = None) -> 'ExperimentResult':
        """Load experiment results from a file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data, experiment)
    
    def generate_visualizations(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Generate visualizations for the experiment results.
        
        Args:
            output_dir: Directory to save visualizations. If None, uses self.output_dir.
        
        Returns:
            Dictionary mapping visualization names to file paths
        """
        if output_dir is None:
            if self.output_dir is None:
                raise ValueError("No output directory specified")
            output_dir = self.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for visualization
        visualization_data = {
            "model_metrics": {},
            "speaker_count_metrics": {}
        }
        
        # Extract model metrics
        for model_name, results in self.model_results.items():
            if "metrics" in results:
                visualization_data["model_metrics"][model_name] = results["metrics"]
        
        # Extract speaker count metrics if available
        for model_name, results in self.model_results.items():
            if "speaker_count_metrics" in results:
                for count, metrics in results["speaker_count_metrics"].items():
                    if count not in visualization_data["speaker_count_metrics"]:
                        visualization_data["speaker_count_metrics"][count] = {}
                    
                    for metric, value in metrics.items():
                        if metric not in visualization_data["speaker_count_metrics"][count]:
                            visualization_data["speaker_count_metrics"][count][metric] = []
                        visualization_data["speaker_count_metrics"][count][metric].append(value)
        
        # Generate visualizations
        vis_paths = {}
        
        # Overall metrics visualization
        metrics_vis_path = os.path.join(output_dir, "metrics_visualization.png")
        visualize_results(visualization_data, metrics_vis_path, title=self.experiment.name)
        vis_paths["metrics"] = metrics_vis_path
        
        # Model comparison charts
        for metric in self.experiment.metrics:
            if metric in ["si_snri_mean", "sdri_mean"]:
                chart_path = os.path.join(output_dir, f"{metric}_comparison.png")
                
                # Prepare data for comparison chart
                comparison_data = []
                for model_name, results in self.model_results.items():
                    if "metrics" in results and metric in results["metrics"]:
                        comparison_data.append({
                            "model_name": model_name,
                            "metrics": {metric: results["metrics"][metric]}
                        })
                
                generate_comparison_chart(
                    comparison_data,
                    chart_path,
                    metric=metric,
                    title=f"{self.experiment.name}: {metric.upper()} Comparison"
                )
                vis_paths[f"{metric}_comparison"] = chart_path
        
        # Generate HTML report
        report_path = create_report(visualization_data, output_dir, self.experiment.name)
        vis_paths["report"] = report_path
        
        return vis_paths


class ExperimentFramework:
    """Framework for running voice separation experiments."""
    
    def __init__(
        self,
        base_dir: str,
        model_registry: Optional[Dict[str, Callable]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the experiment framework.
        
        Args:
            base_dir: Base directory for experiment data
            model_registry: Dictionary mapping model names to model functions
            logger: Logger instance
        """
        self.base_dir = Path(base_dir)
        self.model_registry = model_registry or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # Create directory structure
        self.experiments_dir = self.base_dir / "experiments"
        self.datasets_dir = self.base_dir / "datasets"
        self.results_dir = self.base_dir / "results"
        self.visualizations_dir = self.base_dir / "visualizations"
        
        os.makedirs(self.experiments_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)
    
    def register_model(self, name: str, model_fn: Callable) -> None:
        """
        Register a model with the framework.
        
        Args:
            name: Name of the model
            model_fn: Function that takes audio input and returns separated sources
        """
        self.model_registry[name] = model_fn
        self.logger.info(f"Registered model: {name}")
    
    def create_experiment(
        self,
        name: str,
        description: str,
        models: List[str],
        dataset: Union[str, TestDataset],
        metrics: List[str] = ["si_snri_mean", "sdri_mean"],
        parameters: Dict[str, Any] = None,
        visualization: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ) -> Experiment:
        """
        Create a new experiment.
        
        Args:
            name: Name of the experiment
            description: Description of the experiment
            models: List of model names to evaluate
            dataset: Dataset to use for evaluation (path or TestDataset)
            metrics: List of metrics to calculate
            parameters: Additional parameters for the experiment
            visualization: Visualization configuration
            metadata: Additional metadata
        
        Returns:
            Experiment object
        """
        # Generate unique ID
        experiment_id = f"exp_{int(time.time())}_{name.lower().replace(' ', '_')}"
        
        # Validate models
        model_configs = []
        for model_name in models:
            if isinstance(model_name, dict):
                # Already a config
                model_configs.append(model_name)
            elif model_name in self.model_registry:
                model_configs.append({"id": model_name, "configuration": {}})
            else:
                self.logger.warning(f"Model {model_name} not found in registry")
                model_configs.append({"id": model_name, "configuration": {}})
        
        # Create experiment
        experiment = Experiment(
            id=experiment_id,
            name=name,
            description=description,
            models=model_configs,
            dataset=dataset,
            metrics=metrics,
            parameters=parameters or {},
            visualization=visualization or {},
            metadata=metadata or {}
        )
        
        # Save experiment definition
        experiment_path = self.experiments_dir / f"{experiment_id}.yaml"
        experiment.save_yaml(str(experiment_path))
        self.logger.info(f"Created experiment: {name} ({experiment_id})")
        
        return experiment
    
    def run_experiment(
        self,
        experiment: Union[Experiment, str],
        output_dir: Optional[str] = None,
        device: Optional[torch.device] = None
    ) -> ExperimentResult:
        """
        Run an experiment.
        
        Args:
            experiment: Experiment object or path to experiment definition
            output_dir: Directory to save results
            device: Device to run on (CPU or GPU)
        
        Returns:
            ExperimentResult object
        """
        # Load experiment if path
        if isinstance(experiment, str):
            experiment_path = Path(experiment)
            if not experiment_path.exists():
                experiment_path = self.experiments_dir / f"{experiment}.yaml"
                if not experiment_path.exists():
                    experiment_path = self.experiments_dir / f"{experiment}"
            
            experiment = Experiment.from_yaml(str(experiment_path))
        
        # Set output directory
        if output_dir is None:
            output_dir = str(self.results_dir / experiment.id)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load dataset if path
        dataset = experiment.dataset
        if isinstance(dataset, str):
            dataset_path = Path(dataset)
            if not dataset_path.exists():
                dataset_path = self.datasets_dir / dataset
            
            dataset = TestDataset.load(str(dataset_path))
        
        # Initialize result
        result = ExperimentResult(
            experiment=experiment,
            model_results={},
            start_time=datetime.datetime.now(),
            output_dir=output_dir
        )
        
        # Run experiment for each model
        for model_config in experiment.models:
            model_id = model_config["id"]
            model_config_dict = model_config.get("configuration", {})
            
            self.logger.info(f"Running experiment for model: {model_id}")
            
            try:
                # Get model function
                if model_id not in self.model_registry:
                    raise ValueError(f"Model {model_id} not found in registry")
                
                model_fn = self.model_registry[model_id]
                
                # Create model output directory
                model_output_dir = os.path.join(output_dir, model_id)
                os.makedirs(model_output_dir, exist_ok=True)
                
                # Initialize model results
                result.model_results[model_id] = {
                    "status": "running",
                    "start_time": datetime.datetime.now().isoformat()
                }
                
                # Process each item in the dataset
                all_metrics = []
                speaker_count_metrics = {}
                
                for item in dataset.items:
                    # Load audio
                    mixture, sources = item.load_audio()
                    
                    # Process with model
                    estimated_sources = model_fn(mixture, **model_config_dict)
                    
                    # Calculate metrics
                    metrics = calculate_metrics(estimated_sources, sources, mixture)
                    all_metrics.append(metrics)
                    
                    # Group by speaker count
                    if str(item.num_speakers) not in speaker_count_metrics:
                        speaker_count_metrics[str(item.num_speakers)] = []
                    
                    speaker_count_metrics[str(item.num_speakers)].append(metrics)
                    
                    # Save outputs
                    item_output_dir = os.path.join(model_output_dir, f"item_{item.id}")
                    os.makedirs(item_output_dir, exist_ok=True)
                    
                    # Convert to numpy if tensor
                    if isinstance(estimated_sources, torch.Tensor):
                        estimated_sources = estimated_sources.detach().cpu().numpy()
                    
                    # Save estimated sources
                    if isinstance(estimated_sources, np.ndarray):
                        if estimated_sources.ndim == 1:
                            estimated_sources = estimated_sources[np.newaxis, :]
                        
                        for i, source in enumerate(estimated_sources):
                            output_path = os.path.join(item_output_dir, f"estimated_source_{i+1}.wav")
                            import soundfile as sf
                            sf.write(output_path, source, item.sample_rate)
                
                # Calculate average metrics
                avg_metrics = {}
                for metric in experiment.metrics:
                    if all(metric in m for m in all_metrics):
                        values = [m[metric] for m in all_metrics]
                        if isinstance(values[0], (int, float)):
                            avg_metrics[metric] = np.mean(values)
                        else:
                            # For list metrics, average each element
                            avg_metrics[metric] = np.mean(values, axis=0).tolist()
                
                # Calculate average metrics by speaker count
                avg_speaker_count_metrics = {}
                for count, metrics_list in speaker_count_metrics.items():
                    avg_speaker_count_metrics[count] = {}
                    for metric in experiment.metrics:
                        if all(metric in m for m in metrics_list):
                            values = [m[metric] for m in metrics_list]
                            if isinstance(values[0], (int, float)):
                                avg_speaker_count_metrics[count][metric] = np.mean(values)
                            else:
                                # For list metrics, average each element
                                avg_speaker_count_metrics[count][metric] = np.mean(values, axis=0).tolist()
                
                # Update model results
                result.model_results[model_id].update({
                    "status": "completed",
                    "end_time": datetime.datetime.now().isoformat(),
                    "metrics": avg_metrics,
                    "speaker_count_metrics": avg_speaker_count_metrics,
                    "output_dir": model_output_dir
                })
                
                self.logger.info(f"Completed experiment for model: {model_id}")
                
            except Exception as e:
                self.logger.error(f"Error running experiment for model {model_id}: {str(e)}")
                result.model_results[model_id] = {
                    "status": "failed",
                    "error": str(e),
                    "end_time": datetime.datetime.now().isoformat()
                }
        
        # Update experiment result
        result.end_time = datetime.datetime.now()
        result.status = "completed"
        
        # Save result
        result_path = os.path.join(output_dir, "experiment_result.json")
        result.save(result_path)
        
        # Generate visualizations
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        result.generate_visualizations(vis_dir)
        
        return result
    
    def benchmark_models(
        self,
        models: List[str],
        test_dataset: TestDataset,
        output_dir: Optional[str] = None,
        num_runs: int = 3,
        warmup_runs: int = 1,
        use_gpu: bool = True
    ) -> Dict[str, ModelBenchmarkResult]:
        """
        Benchmark multiple models.
        
        Args:
            models: List of model names to benchmark
            test_dataset: Dataset to use for benchmarking
            output_dir: Directory to save benchmark results
            num_runs: Number of benchmark runs to average
            warmup_runs: Number of warmup runs before benchmarking
            use_gpu: Whether to use GPU if available
        
        Returns:
            Dictionary mapping model names to benchmark results
        """
        # Set output directory
        if output_dir is None:
            output_dir = str(self.results_dir / f"benchmark_{int(time.time())}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Run benchmarks
        benchmark_results = {}
        
        for model_name in models:
            if model_name not in self.model_registry:
                self.logger.warning(f"Model {model_name} not found in registry, skipping")
                continue
            
            model_fn = self.model_registry[model_name]
            model_output_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_output_dir, exist_ok=True)
            
            self.logger.info(f"Benchmarking model: {model_name}")
            
            try:
                result = benchmark_model(
                    model_fn=model_fn,
                    test_dataset_items=test_dataset.items,
                    model_name=model_name,
                    output_dir=model_output_dir,
                    num_runs=num_runs,
                    warmup_runs=warmup_runs,
                    use_gpu=use_gpu
                )
                
                benchmark_results[model_name] = result
                self.logger.info(f"Completed benchmark for model: {model_name}")
                
            except Exception as e:
                self.logger.error(f"Error benchmarking model {model_name}: {str(e)}")
        
        # Compare benchmarks
        if len(benchmark_results) > 1:
            comparison = compare_benchmarks(
                list(benchmark_results.values()),
                os.path.join(output_dir, "benchmark_comparison.json")
            )
        
        return benchmark_results