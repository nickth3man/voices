"""
Visualization tools for experiment results.

This module provides functions for visualizing the results of voice separation
experiments, including comparative charts, performance metrics, and audio waveforms.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from typing import Dict, List, Tuple, Union, Optional
import pandas as pd
import json
from pathlib import Path
import librosa
import librosa.display


def visualize_results(
    results: Dict,
    output_path: str,
    title: str = "Experiment Results",
    include_audio_plots: bool = True
) -> str:
    """
    Create a comprehensive visualization of experiment results.
    
    Args:
        results: Dictionary containing experiment results
        output_path: Path to save the visualization
        title: Title for the visualization
        include_audio_plots: Whether to include audio waveform plots
    
    Returns:
        Path to the saved visualization file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create figure with multiple subplots
    n_plots = 2 + (1 if include_audio_plots else 0)
    fig = plt.figure(figsize=(12, 4 * n_plots))
    fig.suptitle(title, fontsize=16)
    
    # Plot 1: Metrics by model
    ax1 = fig.add_subplot(n_plots, 1, 1)
    _plot_metrics_by_model(results, ax1)
    
    # Plot 2: Metrics by speaker count
    ax2 = fig.add_subplot(n_plots, 1, 2)
    _plot_metrics_by_speaker_count(results, ax2)
    
    # Plot 3: Audio waveforms (if included)
    if include_audio_plots and 'audio_paths' in results:
        ax3 = fig.add_subplot(n_plots, 1, 3)
        _plot_audio_waveforms(results['audio_paths'], ax3)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def _plot_metrics_by_model(results: Dict, ax: plt.Axes) -> None:
    """Plot metrics by model."""
    if 'model_metrics' not in results:
        ax.text(0.5, 0.5, "No model metrics data available", 
                ha='center', va='center', transform=ax.transAxes)
        return
    
    model_metrics = results['model_metrics']
    models = list(model_metrics.keys())
    metrics = ['si_snri_mean', 'sdri_mean']
    
    x = np.arange(len(models))
    width = 0.35
    
    si_snri_values = [model_metrics[model].get('si_snri_mean', 0) for model in models]
    sdri_values = [model_metrics[model].get('sdri_mean', 0) for model in models]
    
    ax.bar(x - width/2, si_snri_values, width, label='SI-SNRi')
    ax.bar(x + width/2, sdri_values, width, label='SDRi')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('dB')
    ax.set_title('Performance Metrics by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    # Add value labels on bars
    for i, v in enumerate(si_snri_values):
        ax.text(i - width/2, v + 0.1, f"{v:.1f}", ha='center')
    
    for i, v in enumerate(sdri_values):
        ax.text(i + width/2, v + 0.1, f"{v:.1f}", ha='center')


def _plot_metrics_by_speaker_count(results: Dict, ax: plt.Axes) -> None:
    """Plot metrics by speaker count."""
    if 'speaker_count_metrics' not in results:
        ax.text(0.5, 0.5, "No speaker count metrics data available", 
                ha='center', va='center', transform=ax.transAxes)
        return
    
    speaker_metrics = results['speaker_count_metrics']
    speaker_counts = sorted([int(k) for k in speaker_metrics.keys()])
    
    si_snri_values = [speaker_metrics[str(count)].get('si_snri_mean', 0) for count in speaker_counts]
    sdri_values = [speaker_metrics[str(count)].get('sdri_mean', 0) for count in speaker_counts]
    
    ax.plot(speaker_counts, si_snri_values, 'o-', label='SI-SNRi')
    ax.plot(speaker_counts, sdri_values, 's-', label='SDRi')
    
    ax.set_xlabel('Number of Speakers')
    ax.set_ylabel('dB')
    ax.set_title('Performance Metrics by Speaker Count')
    ax.set_xticks(speaker_counts)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Add value labels on points
    for i, v in enumerate(si_snri_values):
        ax.text(speaker_counts[i], v + 0.1, f"{v:.1f}", ha='center')
    
    for i, v in enumerate(sdri_values):
        ax.text(speaker_counts[i], v + 0.1, f"{v:.1f}", ha='center')


def _plot_audio_waveforms(audio_paths: Dict, ax: plt.Axes) -> None:
    """Plot audio waveforms for comparison."""
    if not audio_paths or not isinstance(audio_paths, dict):
        ax.text(0.5, 0.5, "No audio data available", 
                ha='center', va='center', transform=ax.transAxes)
        return
    
    # Load audio files
    waveforms = {}
    sr = None
    
    for label, path in audio_paths.items():
        if os.path.exists(path):
            y, sr_temp = librosa.load(path, sr=None)
            waveforms[label] = y
            if sr is None:
                sr = sr_temp
    
    if not waveforms:
        ax.text(0.5, 0.5, "Could not load audio files", 
                ha='center', va='center', transform=ax.transAxes)
        return
    
    # Plot waveforms
    colors = plt.cm.tab10.colors
    n_waveforms = len(waveforms)
    
    for i, (label, y) in enumerate(waveforms.items()):
        color = colors[i % len(colors)]
        offset = i * 2  # Offset for visual separation
        ax.plot(y + offset, color=color, label=label)
    
    ax.set_yticks([i * 2 for i in range(n_waveforms)])
    ax.set_yticklabels(list(waveforms.keys()))
    ax.set_xlabel('Time (samples)')
    ax.set_title('Audio Waveform Comparison')
    ax.set_xlim(0, min(len(y) for y in waveforms.values()))


def generate_comparison_chart(
    experiment_results: List[Dict],
    output_path: str,
    metric: str = 'si_snri_mean',
    title: str = "Model Comparison",
    sort_by_performance: bool = True
) -> str:
    """
    Generate a chart comparing multiple experiment results.
    
    Args:
        experiment_results: List of experiment result dictionaries
        output_path: Path to save the chart
        metric: Metric to use for comparison ('si_snri_mean' or 'sdri_mean')
        title: Title for the chart
        sort_by_performance: Whether to sort models by performance
    
    Returns:
        Path to the saved chart file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract model names and metrics
    models = []
    metric_values = []
    
    for result in experiment_results:
        if 'model_name' in result and 'metrics' in result:
            models.append(result['model_name'])
            metric_values.append(result['metrics'].get(metric, 0))
    
    if not models:
        # Create a simple figure with a message
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No experiment data available", 
                ha='center', va='center', transform=ax.transAxes)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return output_path
    
    # Sort by performance if requested
    if sort_by_performance:
        sorted_indices = np.argsort(metric_values)[::-1]  # Descending order
        models = [models[i] for i in sorted_indices]
        metric_values = [metric_values[i] for i in sorted_indices]
    
    # Create the chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar chart
    bars = ax.bar(models, metric_values, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.1,
            f"{value:.2f}",
            ha='center',
            va='bottom'
        )
    
    # Add labels and title
    metric_label = "SI-SNRi (dB)" if metric == 'si_snri_mean' else "SDRi (dB)"
    ax.set_xlabel('Model')
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels if there are many models
    if len(models) > 5:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def create_report(
    experiment_results: Dict,
    output_dir: str,
    experiment_name: str = "Voice Separation Experiment"
) -> str:
    """
    Create a comprehensive HTML report for experiment results.
    
    Args:
        experiment_results: Dictionary containing experiment results
        output_dir: Directory to save the report and associated files
        experiment_name: Name of the experiment
    
    Returns:
        Path to the generated HTML report
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations
    vis_path = os.path.join(output_dir, "metrics_visualization.png")
    visualize_results(experiment_results, vis_path, title=experiment_name)
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{experiment_name} - Results Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .metrics {{ margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .visualization {{ margin: 30px 0; text-align: center; }}
            .visualization img {{ max-width: 100%; }}
            .footer {{ margin-top: 30px; font-size: 0.8em; color: #7f8c8d; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{experiment_name}</h1>
            <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="metrics">
                <h2>Experiment Summary</h2>
    """
    
    # Add experiment metadata
    if 'metadata' in experiment_results:
        html_content += "<h3>Metadata</h3><table>"
        for key, value in experiment_results['metadata'].items():
            html_content += f"<tr><th>{key}</th><td>{value}</td></tr>"
        html_content += "</table>"
    
    # Add model metrics
    if 'model_metrics' in experiment_results:
        html_content += "<h3>Model Performance</h3><table>"
        html_content += "<tr><th>Model</th><th>SI-SNRi (dB)</th><th>SDRi (dB)</th></tr>"
        
        for model, metrics in experiment_results['model_metrics'].items():
            si_snri = metrics.get('si_snri_mean', 'N/A')
            sdri = metrics.get('sdri_mean', 'N/A')
            html_content += f"<tr><td>{model}</td><td>{si_snri:.2f}</td><td>{sdri:.2f}</td></tr>"
        
        html_content += "</table>"
    
    # Add speaker count metrics
    if 'speaker_count_metrics' in experiment_results:
        html_content += "<h3>Performance by Speaker Count</h3><table>"
        html_content += "<tr><th>Speaker Count</th><th>SI-SNRi (dB)</th><th>SDRi (dB)</th></tr>"
        
        speaker_counts = sorted([int(k) for k in experiment_results['speaker_count_metrics'].keys()])
        for count in speaker_counts:
            metrics = experiment_results['speaker_count_metrics'][str(count)]
            si_snri = metrics.get('si_snri_mean', 'N/A')
            sdri = metrics.get('sdri_mean', 'N/A')
            html_content += f"<tr><td>{count}</td><td>{si_snri:.2f}</td><td>{sdri:.2f}</td></tr>"
        
        html_content += "</table>"
    
    # Add visualization
    html_content += f"""
            <div class="visualization">
                <h2>Visualizations</h2>
                <img src="{os.path.basename(vis_path)}" alt="Metrics Visualization">
            </div>
            
            <div class="footer">
                <p>Generated by Voices ML Experimentation Framework</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    report_path = os.path.join(output_dir, "experiment_report.html")
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    return report_path