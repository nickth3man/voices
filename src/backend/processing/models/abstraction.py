"""
Voice Separation Model Abstraction Layer.

This module provides a consistent interface for different voice separation technologies,
allowing them to be used interchangeably. It includes adapters for both SVoice and Demucs,
and intelligent model selection based on audio characteristics.
"""

import os
import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from enum import Enum
from pathlib import Path

from ..registry.model_adapters import VoiceSeparationModel, SVoiceAdapter, DemucsAdapter
from ..registry.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Enum for supported model types."""
    SVOICE = "svoice"
    DEMUCS = "demucs"
    AUTO = "auto"  # For automatic selection


class AudioCharacteristics:
    """Class for storing audio characteristics used for model selection."""
    
    def __init__(
        self,
        num_speakers: Optional[int] = None,
        duration: Optional[float] = None,
        sample_rate: Optional[int] = None,
        is_noisy: Optional[bool] = None,
        is_reverberant: Optional[bool] = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Initialize audio characteristics.
        
        Args:
            num_speakers: Number of speakers in the audio
            duration: Duration of the audio in seconds
            sample_rate: Sample rate of the audio
            is_noisy: Whether the audio contains significant background noise
            is_reverberant: Whether the audio contains significant reverberation
            metadata: Additional metadata about the audio
        """
        self.num_speakers = num_speakers
        self.duration = duration
        self.sample_rate = sample_rate
        self.is_noisy = is_noisy
        self.is_reverberant = is_reverberant
        self.metadata = metadata or {}
    
    @classmethod
    def from_audio(cls, audio: np.ndarray, sample_rate: int) -> 'AudioCharacteristics':
        """
        Extract characteristics from audio data.
        
        Args:
            audio: Audio data
            sample_rate: Sample rate of the audio
            
        Returns:
            AudioCharacteristics object
        """
        # Calculate duration
        duration = len(audio) / sample_rate
        
        # In a real implementation, we would analyze the audio to determine:
        # - Number of speakers (using a speaker diarization model)
        # - Noise level (using signal processing techniques)
        # - Reverberation level (using signal processing techniques)
        
        # For now, we'll just create a basic characteristics object
        return cls(
            duration=duration,
            sample_rate=sample_rate,
            # These would be determined by analysis in a real implementation
            is_noisy=None,
            is_reverberant=None
        )


class VoiceSeparationManager:
    """
    Manager for voice separation models that provides a consistent interface
    and intelligent model selection.
    """
    
    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        default_model_type: ModelType = ModelType.AUTO,
        device: Optional[torch.device] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the voice separation manager.
        
        Args:
            registry: Model registry for loading models
            default_model_type: Default model type to use
            device: Device to run models on
            logger: Logger instance
        """
        self.registry = registry
        self.default_model_type = default_model_type
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger or logging.getLogger(__name__)
        
        # Cache for loaded models
        self.models: Dict[str, VoiceSeparationModel] = {}
        
        self.logger.info(f"Initialized VoiceSeparationManager with default model type {default_model_type.value}")
    
    def select_model_type(self, characteristics: AudioCharacteristics) -> ModelType:
        """
        Select the most appropriate model type based on audio characteristics.
        
        Args:
            characteristics: Audio characteristics
            
        Returns:
            Selected model type
        """
        # If number of speakers is known and greater than 2, prefer SVoice
        if characteristics.num_speakers is not None and characteristics.num_speakers > 2:
            self.logger.info(f"Selected SVoice model for {characteristics.num_speakers} speakers")
            return ModelType.SVOICE
        
        # If audio is very noisy, prefer Demucs (hypothetically better for noisy conditions)
        if characteristics.is_noisy:
            self.logger.info("Selected Demucs model for noisy audio")
            return ModelType.DEMUCS
        
        # Default to SVoice for now as it's our primary implementation
        self.logger.info("Selected SVoice model as default")
        return ModelType.SVOICE
    
    def get_model(
        self,
        model_type: Optional[ModelType] = None,
        model_id: Optional[str] = None,
        version_id: Optional[str] = None,
        characteristics: Optional[AudioCharacteristics] = None
    ) -> VoiceSeparationModel:
        """
        Get a voice separation model.
        
        Args:
            model_type: Type of model to use (if None, uses default or auto-selects)
            model_id: Specific model ID to use from registry (if None, uses best available)
            version_id: Specific version ID to use (if None, uses default version)
            characteristics: Audio characteristics for model selection
            
        Returns:
            Voice separation model
            
        Raises:
            ValueError: If model cannot be loaded
        """
        # Determine model type
        if model_type is None:
            model_type = self.default_model_type
        
        # If auto-selection is requested and characteristics are provided, select model type
        if model_type == ModelType.AUTO and characteristics is not None:
            model_type = self.select_model_type(characteristics)
        elif model_type == ModelType.AUTO:
            # Default to SVoice if no characteristics are provided
            model_type = ModelType.SVOICE
        
        # If registry is available and model_id is provided, load from registry
        if self.registry is not None and model_id is not None:
            # Check if model is already loaded
            cache_key = f"{model_id}_{version_id or 'default'}"
            if cache_key in self.models:
                return self.models[cache_key]
            
            # Load model from registry
            try:
                model = self.registry.load_model(model_id, version_id, device=self.device)
                
                # Wrap in appropriate adapter if needed
                if not isinstance(model, VoiceSeparationModel):
                    model_info = self.registry.get_model(model_id)
                    if model_info and model_info.model_type == ModelType.SVOICE.value:
                        model = SVoiceAdapter(model_id, self.device, self.logger)
                    elif model_info and model_info.model_type == ModelType.DEMUCS.value:
                        model = DemucsAdapter(model_id, self.device, self.logger)
                
                # Cache model
                self.models[cache_key] = model
                return model
            except Exception as e:
                self.logger.error(f"Error loading model {model_id} from registry: {str(e)}")
                raise ValueError(f"Could not load model {model_id} from registry: {str(e)}")
        
        # If registry is available but model_id is not provided, find best model of the requested type
        if self.registry is not None:
            try:
                # Get all models of the requested type
                models = self.registry.list_models(model_type=model_type.value)
                
                if models:
                    # Sort by performance metrics (if available)
                    # In a real implementation, we would use more sophisticated selection
                    best_model = models[0]
                    
                    # Get default version
                    version = best_model.get_default_version()
                    if not version:
                        raise ValueError(f"No default version found for model {best_model.model_id}")
                    
                    # Load model
                    return self.get_model(model_type, best_model.model_id, None, characteristics)
            except Exception as e:
                self.logger.error(f"Error finding best model of type {model_type.value}: {str(e)}")
        
        # If we couldn't load from registry, create a new adapter
        if model_type == ModelType.SVOICE:
            self.logger.info("Creating new SVoice adapter")
            # In a real implementation, we would have a default model path
            return SVoiceAdapter("default_model_path", self.device, self.logger)
        elif model_type == ModelType.DEMUCS:
            self.logger.info("Creating new Demucs adapter")
            # In a real implementation, we would have a default model path
            return DemucsAdapter("default_model_path", self.device, self.logger)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def separate(
        self,
        mixture: Union[np.ndarray, torch.Tensor],
        num_speakers: Optional[int] = None,
        model_type: Optional[ModelType] = None,
        model_id: Optional[str] = None,
        version_id: Optional[str] = None,
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Separate a mixture into individual sources.
        
        Args:
            mixture: Audio mixture to separate
            num_speakers: Number of speakers to separate (if known)
            model_type: Type of model to use (if None, uses default or auto-selects)
            model_id: Specific model ID to use from registry (if None, uses best available)
            version_id: Specific version ID to use (if None, uses default version)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Separated sources
        """
        # Extract audio characteristics for model selection
        if isinstance(mixture, np.ndarray):
            sample_rate = kwargs.get("sample_rate", 16000)
            characteristics = AudioCharacteristics.from_audio(mixture, sample_rate)
        else:
            # For torch tensors, we have less information
            characteristics = AudioCharacteristics(num_speakers=num_speakers)
        
        # Override number of speakers if provided
        if num_speakers is not None:
            characteristics.num_speakers = num_speakers
        
        # Get appropriate model
        model = self.get_model(model_type, model_id, version_id, characteristics)
        
        # Separate using the model
        return model.separate(mixture, num_speakers=num_speakers, **kwargs)
    
    def get_available_models(self, model_type: Optional[ModelType] = None) -> List[Dict[str, Any]]:
        """
        Get information about available models.
        
        Args:
            model_type: Filter by model type
            
        Returns:
            List of model information dictionaries
        """
        if self.registry is None:
            return []
        
        # Get models from registry
        models = self.registry.list_models(
            model_type=model_type.value if model_type else None
        )
        
        # Convert to simplified info dictionaries
        result = []
        for model_info in models:
            # Get default version
            default_version = model_info.get_default_version()
            
            # Create info dictionary
            info = {
                "id": model_info.model_id,
                "name": model_info.name,
                "type": model_info.model_type,
                "description": model_info.description,
                "versions": len(model_info.versions),
                "default_version": default_version.version_id if default_version else None
            }
            
            # Add performance metrics if available
            if default_version and default_version.performance_metrics:
                info["performance"] = default_version.performance_metrics
            
            result.append(info)
        
        return result


# Factory function to create a voice separation manager
def create_separation_manager(
    registry_dir: Optional[str] = None,
    default_model_type: str = "auto",
    device: Optional[torch.device] = None
) -> VoiceSeparationManager:
    """
    Create a voice separation manager.
    
    Args:
        registry_dir: Directory for the model registry
        default_model_type: Default model type to use
        device: Device to run models on
        
    Returns:
        Voice separation manager
    """
    # Create logger
    logger = logging.getLogger(__name__)
    
    # Create registry if directory is provided
    registry = None
    if registry_dir:
        from ..registry.model_registry import ModelRegistry
        registry = ModelRegistry(registry_dir, logger)
        
        # Register model loaders
        from ..registry.model_adapters import get_model_loader
        registry.register_model_loader("svoice", get_model_loader("svoice"))
        registry.register_model_loader("demucs", get_model_loader("demucs"))
    
    # Convert string model type to enum
    try:
        model_type = ModelType(default_model_type.lower())
    except ValueError:
        logger.warning(f"Invalid model type: {default_model_type}, using AUTO")
        model_type = ModelType.AUTO
    
    # Create manager
    return VoiceSeparationManager(registry, model_type, device, logger)