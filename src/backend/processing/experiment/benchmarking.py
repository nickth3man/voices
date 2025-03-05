"""
Performance benchmarking system for voice separation models.

This module provides functionality for benchmarking the performance of
voice separation models, including processing speed, memory usage, and
quality metrics across different hardware configurations.
"""

import os
import time
import json
import numpy as np
import torch
import psutil
import platform
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import gc

from .metrics import calculate_metrics


@dataclass
class ModelBenchmarkResult:
    """Results from benchmarking a model."""
    
    model_name: str
    hardware_info: Dict[str, Any]
    processing_speed: float  # Real-time factor (higher is better)
    memory_usage: Dict[str, float]  # In MB
    cpu_utilization: float  # Percentage
    gpu_utilization: Optional[float] = None  # Percentage, None if no GPU
    gpu_memory_usage: Optional[float] = None  # In MB, None if no GPU
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "hardware_info": self.hardware_info,
            "processing_speed": self.processing_speed,
            "memory_usage": self.memory_usage,
            "cpu_utilization": self.cpu_utilization,
            "gpu_utilization": self.gpu_utilization,
            "gpu_memory_usage": self.gpu_memory_usage,
            "quality_metrics": self.quality_metrics,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelBenchmarkResult':
        """Create from dictionary."""
        return cls(
            model_name=data["model_name"],
            hardware_info=data["hardware_info"],
            processing_speed=data["processing_speed"],
            memory_usage=data["memory_usage"],
            cpu_utilization=data["cpu_utilization"],
            gpu_utilization=data.get("gpu_utilization"),
            gpu_memory_usage=data.get("gpu_memory_usage"),
            quality_metrics=data.get("quality_metrics", {}),
            metadata=data.get("metadata", {})
        )
    
    def save(self, path: str) -> None:
        """Save benchmark results to a file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ModelBenchmarkResult':
        """Load benchmark results from a file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


def get_hardware_info() -> Dict[str, Any]:
    """
    Get information about the hardware environment.
    
    Returns:
        Dictionary containing hardware information
    """
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "memory_total": psutil.virtual_memory().total / (1024 * 1024),  # MB
    }
    
    # Check for GPU
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)  # MB
    else:
        info["gpu"] = None
    
    return info


def benchmark_model(
    model_fn: Callable,
    test_dataset_items: List[Any],
    model_name: str,
    output_dir: Optional[str] = None,
    num_runs: int = 3,
    warmup_runs: int = 1,
    use_gpu: bool = True
) -> ModelBenchmarkResult:
    """
    Benchmark a voice separation model's performance.
    
    Args:
        model_fn: Function that takes audio input and returns separated sources
        test_dataset_items: List of test dataset items to use for benchmarking
        model_name: Name of the model being benchmarked
        output_dir: Directory to save benchmark results and outputs
        num_runs: Number of benchmark runs to average
        warmup_runs: Number of warmup runs before benchmarking
        use_gpu: Whether to use GPU if available
    
    Returns:
        ModelBenchmarkResult containing benchmark results
    """
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get hardware info
    hardware_info = get_hardware_info()
    
    # Set device
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    
    # Initialize metrics
    total_duration = 0
    total_processing_time = 0
    peak_memory = 0
    cpu_percent_values = []
    gpu_utilization_values = []
    gpu_memory_values = []
    quality_metrics_list = []
    
    # Warmup runs
    for _ in range(warmup_runs):
        for item in test_dataset_items[:1]:  # Use first item for warmup
            mixture, sources = item.load_audio()
            _ = model_fn(mixture)
    
    # Force garbage collection
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    # Benchmark runs
    for run in range(num_runs):
        for item in test_dataset_items:
            # Load audio
            mixture, sources = item.load_audio()
            audio_duration = len(mixture) / item.sample_rate
            total_duration += audio_duration
            
            # Start monitoring
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Start timing
            start_time = time.time()
            
            # Start CPU monitoring
            process.cpu_percent()  # First call to initialize
            
            # Process audio
            estimated_sources = model_fn(mixture)
            
            # End timing
            end_time = time.time()
            processing_time = end_time - start_time
            total_processing_time += processing_time
            
            # Collect CPU usage
            cpu_percent = process.cpu_percent()
            cpu_percent_values.append(cpu_percent)
            
            # Collect memory usage
            end_memory = process.memory_info().rss / (1024 * 1024)  # MB
            peak_memory = max(peak_memory, end_memory)
            
            # Collect GPU metrics if available
            if device.type == "cuda":
                gpu_utilization = torch.cuda.utilization()
                gpu_utilization_values.append(gpu_utilization)
                
                gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                gpu_memory_values.append(gpu_memory)
            
            # Calculate quality metrics
            if isinstance(estimated_sources, torch.Tensor):
                estimated_sources = estimated_sources.detach().cpu().numpy()
            
            metrics = calculate_metrics(estimated_sources, np.array(sources), mixture)
            quality_metrics_list.append(metrics)
            
            # Save outputs if directory specified
            if output_dir:
                # Create subdirectory for this item
                item_dir = os.path.join(output_dir, f"item_{item.id}")
                os.makedirs(item_dir, exist_ok=True)
                
                # Save estimated sources
                if isinstance(estimated_sources, np.ndarray):
                    if estimated_sources.ndim == 1:
                        estimated_sources = estimated_sources[np.newaxis, :]
                    
                    for i, source in enumerate(estimated_sources):
                        output_path = os.path.join(item_dir, f"estimated_source_{i+1}.wav")
                        import soundfile as sf
                        sf.write(output_path, source, item.sample_rate)
    
    # Calculate average metrics
    processing_speed = total_duration / total_processing_time  # Real-time factor
    avg_cpu_utilization = np.mean(cpu_percent_values)
    
    memory_usage = {
        "peak_memory_mb": peak_memory,
        "memory_increase_mb": peak_memory - start_memory
    }
    
    # Average quality metrics
    avg_quality_metrics = {}
    for metric in quality_metrics_list[0].keys():
        if isinstance(quality_metrics_list[0][metric], (int, float)):
            avg_quality_metrics[metric] = np.mean([m[metric] for m in quality_metrics_list])
        else:
            # For list metrics, average each element
            avg_quality_metrics[metric] = np.mean([m[metric] for m in quality_metrics_list], axis=0).tolist()
    
    # Create benchmark result
    result = ModelBenchmarkResult(
        model_name=model_name,
        hardware_info=hardware_info,
        processing_speed=processing_speed,
        memory_usage=memory_usage,
        cpu_utilization=avg_cpu_utilization,
        quality_metrics=avg_quality_metrics,
        metadata={
            "num_runs": num_runs,
            "total_duration": total_duration,
            "total_processing_time": total_processing_time,
            "device": str(device),
            "test_items": len(test_dataset_items)
        }
    )
    
    # Add GPU metrics if available
    if device.type == "cuda":
        result.gpu_utilization = np.mean(gpu_utilization_values)
        result.gpu_memory_usage = np.mean(gpu_memory_values)
    
    # Save result if directory specified
    if output_dir:
        result_path = os.path.join(output_dir, f"{model_name}_benchmark_result.json")
        result.save(result_path)
    
    return result


def compare_benchmarks(
    benchmark_results: List[ModelBenchmarkResult],
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare multiple benchmark results.
    
    Args:
        benchmark_results: List of benchmark results to compare
        output_path: Path to save comparison results
    
    Returns:
        Dictionary containing comparison results
    """
    if not benchmark_results:
        return {"error": "No benchmark results provided"}
    
    # Extract model names
    model_names = [result.model_name for result in benchmark_results]
    
    # Create comparison data
    comparison = {
        "models": model_names,
        "processing_speed": [result.processing_speed for result in benchmark_results],
        "cpu_utilization": [result.cpu_utilization for result in benchmark_results],
        "memory_usage": [result.memory_usage["peak_memory_mb"] for result in benchmark_results],
        "quality_metrics": {}
    }
    
    # Add GPU metrics if available
    if any(result.gpu_utilization is not None for result in benchmark_results):
        comparison["gpu_utilization"] = [
            result.gpu_utilization if result.gpu_utilization is not None else 0
            for result in benchmark_results
        ]
        comparison["gpu_memory_usage"] = [
            result.gpu_memory_usage if result.gpu_memory_usage is not None else 0
            for result in benchmark_results
        ]
    
    # Add quality metrics
    for metric in benchmark_results[0].quality_metrics.keys():
        if isinstance(benchmark_results[0].quality_metrics[metric], (int, float)):
            comparison["quality_metrics"][metric] = [
                result.quality_metrics.get(metric, 0) for result in benchmark_results
            ]
    
    # Add hardware info
    comparison["hardware_info"] = benchmark_results[0].hardware_info
    
    # Save comparison if path specified
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
    
    return comparison