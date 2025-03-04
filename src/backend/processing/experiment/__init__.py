"""
ML Experimentation Framework for Voice Separation Models.

This package provides tools for systematic evaluation of voice separation models,
including standardized test datasets, objective metrics calculation, performance
benchmarking, and result visualization.
"""

from .framework import ExperimentFramework, Experiment, ExperimentResult
from .metrics import calculate_si_snri, calculate_sdri, calculate_metrics
from .datasets import TestDataset, create_test_dataset
from .visualization import visualize_results, generate_comparison_chart
from .benchmarking import benchmark_model, ModelBenchmarkResult

__all__ = [
    'ExperimentFramework',
    'Experiment',
    'ExperimentResult',
    'calculate_si_snri',
    'calculate_sdri',
    'calculate_metrics',
    'TestDataset',
    'create_test_dataset',
    'visualize_results',
    'generate_comparison_chart',
    'benchmark_model',
    'ModelBenchmarkResult',
]