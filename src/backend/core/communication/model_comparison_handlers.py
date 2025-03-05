"""
Model comparison handlers for the communication server.

This module provides handlers for model comparison operations
that can be called from the frontend via the Python Bridge.
"""

import os
import logging
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from ...processing.models.abstraction import create_separation_manager, ModelType
from ...processing.experiment.framework import ExperimentFramework
from ...processing.registry.model_registry import ModelRegistry
from ...processing.audio.io import load_audio, save_audio
from ...storage.database.db_manager import DatabaseManager
from ...storage.files.file_manager import FileManager

# Configure logging
logger = logging.getLogger(__name__)


class ModelComparisonHandlers:
    """
    Handlers for model comparison operations.
    
    This class provides methods that can be called from the frontend
    via the Python Bridge to interact with the model comparison system.
    """
    
    def __init__(self):
        """Initialize the model comparison handlers."""
        # Initialize database and file managers
        self.db_manager = DatabaseManager()
        if not self.db_manager.initialize():
            logger.error("Failed to initialize database")
            raise RuntimeError("Failed to initialize database")
        
        self.file_manager = FileManager(self.db_manager)
        
        # Initialize model registry and separation manager
        registry_dir = os.path.join(os.path.dirname(__file__), "../../../data/model_registry")
        os.makedirs(registry_dir, exist_ok=True)
        
        self.model_registry = ModelRegistry(registry_dir)
        self.separation_manager = create_separation_manager(registry_dir)
        
        # Initialize experiment framework
        experiment_dir = os.path.join(os.path.dirname(__file__), "../../../data/experiments")
        os.makedirs(experiment_dir, exist_ok=True)
        
        self.experiment_framework = ExperimentFramework(
            base_dir=experiment_dir,
            model_registry=self.model_registry.model_loaders
        )
    
    def get_available_models(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a list of available voice separation models.
        
        Args:
            params: Dictionary with parameters
                - modelType: Optional filter by model type
                
        Returns:
            Dictionary with response
                - success: Whether the operation was successful
                - data: List of models if successful
                - error: Error message if unsuccessful
        """
        try:
            model_type = params.get('modelType')
            
            # Convert string model type to enum if provided
            model_type_enum = None
            if model_type:
                try:
                    model_type_enum = ModelType(model_type.lower())
                except ValueError:
                    return {'success': False, 'error': f'Invalid model type: {model_type}'}
            
            # Get available models
            models = self.separation_manager.get_available_models(model_type_enum)
            
            return {'success': True, 'data': models}
        
        except Exception as e:
            logger.error(f"Error in get_available_models: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def compare_models(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare multiple voice separation models on the same audio input.
        
        Args:
            params: Dictionary with parameters
                - audioPath: Path to the audio file
                - modelIds: List of model IDs to compare
                - numSpeakers: Optional number of speakers
                
        Returns:
            Dictionary with response
                - success: Whether the operation was successful
                - data: Comparison results if successful
                - error: Error message if unsuccessful
        """
        try:
            audio_path = params.get('audioPath')
            model_ids = params.get('modelIds', [])
            num_speakers = params.get('numSpeakers')
            
            if not audio_path:
                return {'success': False, 'error': 'Audio path is required'}
            
            if not model_ids:
                return {'success': False, 'error': 'At least one model ID is required'}
            
            # Load audio file
            try:
                audio_data, sample_rate = load_audio(audio_path)
            except Exception as e:
                return {'success': False, 'error': f'Failed to load audio file: {str(e)}'}
            
            # Create temporary directory for outputs
            temp_dir = tempfile.mkdtemp()
            
            # Process with each model
            results = {}
            
            for model_id in model_ids:
                try:
                    # Create output directory for this model
                    model_output_dir = os.path.join(temp_dir, model_id)
                    os.makedirs(model_output_dir, exist_ok=True)
                    
                    # Process audio with this model
                    separated_sources = self.separation_manager.separate(
                        audio_data,
                        num_speakers=num_speakers,
                        model_id=model_id
                    )
                    
                    # Save separated sources
                    source_paths = []
                    for i, source in enumerate(separated_sources):
                        source_path = os.path.join(model_output_dir, f"source_{i+1}.wav")
                        save_audio(source_path, source, sample_rate)
                        source_paths.append(source_path)
                    
                    # Get model info
                    model_info = self.model_registry.get_model(model_id)
                    model_name = model_info.name if model_info else model_id
                    model_type = model_info.model_type if model_info else "unknown"
                    
                    # Add to results
                    results[model_id] = {
                        'name': model_name,
                        'type': model_type,
                        'sourcePaths': source_paths,
                        'numSources': len(source_paths)
                    }
                
                except Exception as e:
                    logger.error(f"Error processing with model {model_id}: {str(e)}")
                    results[model_id] = {
                        'error': str(e)
                    }
            
            return {
                'success': True,
                'data': {
                    'results': results,
                    'tempDir': temp_dir,
                    'originalPath': audio_path,
                    'sampleRate': sample_rate
                }
            }
        
        except Exception as e:
            logger.error(f"Error in compare_models: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_model_metrics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get performance metrics for voice separation models.
        
        Args:
            params: Dictionary with parameters
                - modelIds: List of model IDs to get metrics for
                
        Returns:
            Dictionary with response
                - success: Whether the operation was successful
                - data: Metrics if successful
                - error: Error message if unsuccessful
        """
        try:
            model_ids = params.get('modelIds', [])
            
            if not model_ids:
                return {'success': False, 'error': 'At least one model ID is required'}
            
            # Get metrics for each model
            metrics = {}
            
            for model_id in model_ids:
                try:
                    # Get model version
                    model_version = self.model_registry.get_model_version(model_id)
                    
                    if model_version and model_version.performance_metrics:
                        metrics[model_id] = model_version.performance_metrics
                    else:
                        metrics[model_id] = {
                            'si_snri_mean': 'N/A',
                            'sdri_mean': 'N/A'
                        }
                
                except Exception as e:
                    logger.error(f"Error getting metrics for model {model_id}: {str(e)}")
                    metrics[model_id] = {
                        'error': str(e)
                    }
            
            return {'success': True, 'data': metrics}
        
        except Exception as e:
            logger.error(f"Error in get_model_metrics: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def run_objective_comparison(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run an objective comparison between models using the experiment framework.
        
        Args:
            params: Dictionary with parameters
                - audioPath: Path to the audio file
                - referencePath: Path to the reference audio file
                - modelIds: List of model IDs to compare
                
        Returns:
            Dictionary with response
                - success: Whether the operation was successful
                - data: Comparison results if successful
                - error: Error message if unsuccessful
        """
        try:
            audio_path = params.get('audioPath')
            reference_path = params.get('referencePath')
            model_ids = params.get('modelIds', [])
            
            if not audio_path:
                return {'success': False, 'error': 'Audio path is required'}
            
            if not reference_path:
                return {'success': False, 'error': 'Reference path is required'}
            
            if not model_ids:
                return {'success': False, 'error': 'At least one model ID is required'}
            
            # Create temporary directory for experiment
            temp_dir = tempfile.mkdtemp()
            
            # Create experiment
            experiment = self.experiment_framework.create_experiment(
                name="Model Comparison",
                description="Comparison of voice separation models",
                models=model_ids,
                dataset=audio_path,  # This would need to be adapted for a real implementation
                metrics=["si_snri_mean", "sdri_mean"]
            )
            
            # Run experiment
            result = self.experiment_framework.run_experiment(
                experiment,
                output_dir=temp_dir
            )
            
            # Generate visualizations
            vis_paths = result.generate_visualizations()
            
            # Extract results
            model_results = {}
            for model_id, model_result in result.model_results.items():
                if "metrics" in model_result:
                    model_results[model_id] = {
                        'metrics': model_result["metrics"],
                        'status': model_result["status"]
                    }
            
            return {
                'success': True,
                'data': {
                    'experimentId': experiment.id,
                    'modelResults': model_results,
                    'visualizationPaths': vis_paths,
                    'outputDir': temp_dir
                }
            }
        
        except Exception as e:
            logger.error(f"Error in run_objective_comparison: {str(e)}")
            return {'success': False, 'error': str(e)}


# Create a singleton instance
model_comparison_handlers = ModelComparisonHandlers()

# Define handler functions that can be registered with the server

def handle_get_available_models(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for get_available_models."""
    return model_comparison_handlers.get_available_models(params)

def handle_compare_models(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for compare_models."""
    return model_comparison_handlers.compare_models(params)

def handle_get_model_metrics(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for get_model_metrics."""
    return model_comparison_handlers.get_model_metrics(params)

def handle_run_objective_comparison(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for run_objective_comparison."""
    return model_comparison_handlers.run_objective_comparison(params)


# Dictionary of handlers to register with the server
MODEL_COMPARISON_HANDLERS = {
    'get_available_models': handle_get_available_models,
    'compare_models': handle_compare_models,
    'get_model_metrics': handle_get_model_metrics,
    'run_objective_comparison': handle_run_objective_comparison
}