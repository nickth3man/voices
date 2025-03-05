"""
Model Adapters for Voice Separation Technologies.

This module provides adapters for different voice separation technologies,
allowing them to be used interchangeably through a consistent interface.
"""

import os
import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from abc import ABC, abstractmethod


class VoiceSeparationModel(ABC):
    """Abstract base class for voice separation models."""
    
    @abstractmethod
    def separate(
        self,
        mixture: Union[np.ndarray, torch.Tensor],
        num_speakers: Optional[int] = None,
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Separate a mixture into individual sources.
        
        Args:
            mixture: Audio mixture to separate, shape (n_samples,)
            num_speakers: Number of speakers to separate (if known)
            **kwargs: Additional model-specific parameters
        
        Returns:
            Separated sources, shape (n_sources, n_samples)
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        pass


class SVoiceAdapter(VoiceSeparationModel):
    """Adapter for SVoice voice separation model."""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the SVoice adapter.
        
        Args:
            model_path: Path to the SVoice model
            device: Device to run the model on
            logger: Logger instance
        """
        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger or logging.getLogger(__name__)
        
        # Load the model
        self.model = self._load_model()
        
        self.logger.info(f"Initialized SVoice adapter with model from {model_path}")
    
    def _load_model(self) -> Any:
        """
        Load the SVoice model.
        
        Returns:
            Loaded model
        """
        try:
            # Import here to avoid circular imports
            from ..models.svoice.utils import load_svoice_model
            
            # Load the model using the SVoice implementation
            self.logger.info(f"Loading SVoice model from {self.model_path}")
            return load_svoice_model(self.model_path, device=self.device)
            
        except ImportError as e:
            self.logger.warning(f"Could not import SVoice model: {str(e)}")
            self.logger.warning("Falling back to dummy implementation")
            
            # Fallback to dummy implementation if SVoice is not available
            class DummySVoiceModel:
                def __init__(self, path, device):
                    self.path = path
                    self.device = device
                    self.name = "SVoice"
                    self.version = "1.0.0"
                
                def to(self, device):
                    return self
                
                def separate(self, mixture, num_speakers=None, **kwargs):
                    # Simulate separation by creating random sources
                    if isinstance(mixture, torch.Tensor):
                        n_speakers = num_speakers or 2
                        return torch.randn(n_speakers, mixture.shape[0], device=self.device)
                    else:
                        n_speakers = num_speakers or 2
                        return np.random.randn(n_speakers, mixture.shape[0])
                
                def get_model_info(self):
                    return {
                        "type": "svoice",
                        "name": "SVoice (Dummy)",
                        "version": "1.0.0",
                        "path": self.path,
                        "device": str(self.device)
                    }
            
            return DummySVoiceModel(self.model_path, self.device)
    
    def separate(
        self,
        mixture: Union[np.ndarray, torch.Tensor],
        num_speakers: Optional[int] = None,
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Separate a mixture using SVoice.
        
        Args:
            mixture: Audio mixture to separate, shape (n_samples,)
            num_speakers: Number of speakers to separate (if known)
            **kwargs: Additional SVoice-specific parameters
        
        Returns:
            Separated sources, shape (n_sources, n_samples)
        """
        self.logger.debug(f"Separating mixture with SVoice, num_speakers={num_speakers}")
        
        # Check if the model has a separate method (our implementation)
        if hasattr(self.model, 'separate'):
            return self.model.separate(mixture, num_speakers=num_speakers, **kwargs)
        
        # Fallback for dummy model
        # Convert numpy array to tensor if needed
        is_numpy = isinstance(mixture, np.ndarray)
        if is_numpy:
            mixture_tensor = torch.from_numpy(mixture).float().to(self.device)
        else:
            mixture_tensor = mixture.to(self.device)
        
        # Ensure mixture is 1D
        if mixture_tensor.dim() > 1:
            mixture_tensor = mixture_tensor.squeeze()
        
        # Separate using SVoice model
        with torch.no_grad():
            separated_sources = self.model(mixture_tensor, num_speakers=num_speakers, **kwargs)
        
        # Convert back to numpy if input was numpy
        if is_numpy:
            separated_sources = separated_sources.cpu().numpy()
        
        return separated_sources
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the SVoice model.
        
        Returns:
            Dictionary containing model information
        """
        # Check if the model has a get_model_info method (our implementation)
        if hasattr(self.model, 'get_model_info'):
            return self.model.get_model_info()
        
        # Fallback for dummy model
        return {
            "type": "svoice",
            "name": getattr(self.model, "name", "SVoice"),
            "version": getattr(self.model, "version", "unknown"),
            "path": self.model_path,
            "device": str(self.device)
        }


class DemucsAdapter(VoiceSeparationModel):
    """Adapter for Demucs voice separation model."""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Demucs adapter.
        
        Args:
            model_path: Path to the Demucs model
            device: Device to run the model on
            logger: Logger instance
        """
        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger or logging.getLogger(__name__)
        
        # Load the model
        self.model = self._load_model()
        
        self.logger.info(f"Initialized Demucs adapter with model from {model_path}")
    
    def _load_model(self) -> Any:
        """
        Load the Demucs model.
        
        Returns:
            Loaded model
        
        Note:
            This is a placeholder implementation. The actual implementation
            would depend on the Demucs API and model format.
        """
        # Placeholder for actual Demucs model loading
        # In a real implementation, this would use the Demucs API to load the model
        self.logger.info(f"Loading Demucs model from {self.model_path}")
        
        # Placeholder model
        class DummyDemucsModel:
            def __init__(self, path, device):
                self.path = path
                self.device = device
                self.name = "Demucs"
                self.version = "3.0.0"
            
            def to(self, device):
                return self
            
            def __call__(self, mixture, num_speakers=None):
                # Simulate separation by creating random sources
                if isinstance(mixture, torch.Tensor):
                    n_speakers = num_speakers or 2
                    return torch.randn(n_speakers, mixture.shape[0], device=self.device)
                else:
                    n_speakers = num_speakers or 2
                    return np.random.randn(n_speakers, mixture.shape[0])
        
        return DummyDemucsModel(self.model_path, self.device)
    
    def separate(
        self,
        mixture: Union[np.ndarray, torch.Tensor],
        num_speakers: Optional[int] = None,
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Separate a mixture using Demucs.
        
        Args:
            mixture: Audio mixture to separate, shape (n_samples,)
            num_speakers: Number of speakers to separate (if known)
            **kwargs: Additional Demucs-specific parameters
        
        Returns:
            Separated sources, shape (n_sources, n_samples)
        """
        self.logger.debug(f"Separating mixture with Demucs, num_speakers={num_speakers}")
        
        # Convert numpy array to tensor if needed
        is_numpy = isinstance(mixture, np.ndarray)
        if is_numpy:
            mixture_tensor = torch.from_numpy(mixture).float().to(self.device)
        else:
            mixture_tensor = mixture.to(self.device)
        
        # Ensure mixture is 1D
        if mixture_tensor.dim() > 1:
            mixture_tensor = mixture_tensor.squeeze()
        
        # Separate using Demucs model
        with torch.no_grad():
            separated_sources = self.model(mixture_tensor, num_speakers=num_speakers, **kwargs)
        
        # Convert back to numpy if input was numpy
        if is_numpy:
            separated_sources = separated_sources.cpu().numpy()
        
        return separated_sources
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the Demucs model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "type": "demucs",
            "name": getattr(self.model, "name", "Demucs"),
            "version": getattr(self.model, "version", "unknown"),
            "path": self.model_path,
            "device": str(self.device)
        }


class HybridTransformerCNNAdapter(VoiceSeparationModel):
    """Adapter for Hybrid Transformer/CNN voice separation model."""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Hybrid Transformer/CNN adapter.
        
        Args:
            model_path: Path to the model
            device: Device to run the model on
            logger: Logger instance
        """
        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger or logging.getLogger(__name__)
        
        # Load the model
        self.model = self._load_model()
        
        self.logger.info(f"Initialized Hybrid Transformer/CNN adapter with model from {model_path}")
    
    def _load_model(self) -> Any:
        """
        Load the Hybrid Transformer/CNN model.
        
        Returns:
            Loaded model
        
        Note:
            This is a placeholder implementation. The actual implementation
            would depend on the specific model architecture and format.
        """
        # Placeholder for actual model loading
        self.logger.info(f"Loading Hybrid Transformer/CNN model from {self.model_path}")
        
        # Placeholder model
        class DummyHybridTransformerCNNModel:
            def __init__(self, path, device):
                self.path = path
                self.device = device
                self.name = "Hybrid Transformer/CNN"
                self.version = "1.0.0"
            
            def to(self, device):
                return self
            
            def __call__(self, mixture, num_speakers=None, **kwargs):
                # Simulate separation by creating random sources
                if isinstance(mixture, torch.Tensor):
                    n_speakers = num_speakers or 2
                    return torch.randn(n_speakers, mixture.shape[0], device=self.device)
                else:
                    n_speakers = num_speakers or 2
                    return np.random.randn(n_speakers, mixture.shape[0])
        
        return DummyHybridTransformerCNNModel(self.model_path, self.device)
    
    def separate(
        self,
        mixture: Union[np.ndarray, torch.Tensor],
        num_speakers: Optional[int] = None,
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Separate a mixture using Hybrid Transformer/CNN.
        
        Args:
            mixture: Audio mixture to separate, shape (n_samples,)
            num_speakers: Number of speakers to separate (if known)
            **kwargs: Additional model-specific parameters
        
        Returns:
            Separated sources, shape (n_sources, n_samples)
        """
        self.logger.debug(f"Separating mixture with Hybrid Transformer/CNN, num_speakers={num_speakers}")
        
        # Convert numpy array to tensor if needed
        is_numpy = isinstance(mixture, np.ndarray)
        if is_numpy:
            mixture_tensor = torch.from_numpy(mixture).float().to(self.device)
        else:
            mixture_tensor = mixture.to(self.device)
        
        # Ensure mixture is 1D
        if mixture_tensor.dim() > 1:
            mixture_tensor = mixture_tensor.squeeze()
        
        # Separate using model
        with torch.no_grad():
            separated_sources = self.model(mixture_tensor, num_speakers=num_speakers, **kwargs)
        
        # Convert back to numpy if input was numpy
        if is_numpy:
            separated_sources = separated_sources.cpu().numpy()
        
        return separated_sources
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the Hybrid Transformer/CNN model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "type": "hybrid_transformer_cnn",
            "name": getattr(self.model, "name", "Hybrid Transformer/CNN"),
            "version": getattr(self.model, "version", "unknown"),
            "path": self.model_path,
            "device": str(self.device)
        }


class SpleeterAdapter(VoiceSeparationModel):
    """Adapter for Spleeter voice separation model."""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Spleeter adapter.
        
        Args:
            model_path: Path to the Spleeter model
            device: Device to run the model on
            logger: Logger instance
        """
        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger or logging.getLogger(__name__)
        
        # Load the model
        self.model = self._load_model()
        
        self.logger.info(f"Initialized Spleeter adapter with model from {model_path}")
    
    def _load_model(self) -> Any:
        """
        Load the Spleeter model.
        
        Returns:
            Loaded model
        
        Note:
            This is a placeholder implementation. The actual implementation
            would depend on the Spleeter API and model format.
        """
        # Placeholder for actual Spleeter model loading
        self.logger.info(f"Loading Spleeter model from {self.model_path}")
        
        # Placeholder model
        class DummySpleeterModel:
            def __init__(self, path, device):
                self.path = path
                self.device = device
                self.name = "Spleeter"
                self.version = "2.1.0"
            
            def to(self, device):
                return self
            
            def __call__(self, mixture, num_speakers=None, **kwargs):
                # Simulate separation by creating random sources
                if isinstance(mixture, torch.Tensor):
                    n_speakers = num_speakers or 2
                    return torch.randn(n_speakers, mixture.shape[0], device=self.device)
                else:
                    n_speakers = num_speakers or 2
                    return np.random.randn(n_speakers, mixture.shape[0])
        
        return DummySpleeterModel(self.model_path, self.device)
    
    def separate(
        self,
        mixture: Union[np.ndarray, torch.Tensor],
        num_speakers: Optional[int] = None,
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Separate a mixture using Spleeter.
        
        Args:
            mixture: Audio mixture to separate, shape (n_samples,)
            num_speakers: Number of speakers to separate (if known)
            **kwargs: Additional Spleeter-specific parameters
        
        Returns:
            Separated sources, shape (n_sources, n_samples)
        """
        self.logger.debug(f"Separating mixture with Spleeter, num_speakers={num_speakers}")
        
        # Convert numpy array to tensor if needed
        is_numpy = isinstance(mixture, np.ndarray)
        if is_numpy:
            mixture_tensor = torch.from_numpy(mixture).float().to(self.device)
        else:
            mixture_tensor = mixture.to(self.device)
        
        # Ensure mixture is 1D
        if mixture_tensor.dim() > 1:
            mixture_tensor = mixture_tensor.squeeze()
        
        # Separate using Spleeter model
        with torch.no_grad():
            separated_sources = self.model(mixture_tensor, num_speakers=num_speakers, **kwargs)
        
        # Convert back to numpy if input was numpy
        if is_numpy:
            separated_sources = separated_sources.cpu().numpy()
        
        return separated_sources
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the Spleeter model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "type": "spleeter",
            "name": getattr(self.model, "name", "Spleeter"),
            "version": getattr(self.model, "version", "unknown"),
            "path": self.model_path,
            "device": str(self.device)
        }


class VoiceSepUnknownSpeakersAdapter(VoiceSeparationModel):
    """Adapter for Voice Separation with Unknown Speakers model."""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Voice Separation with Unknown Speakers adapter.
        
        Args:
            model_path: Path to the model
            device: Device to run the model on
            logger: Logger instance
        """
        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger or logging.getLogger(__name__)
        
        # Load the model
        self.model = self._load_model()
        
        self.logger.info(f"Initialized Voice Separation with Unknown Speakers adapter with model from {model_path}")
    
    def _load_model(self) -> Any:
        """
        Load the Voice Separation with Unknown Speakers model.
        
        Returns:
            Loaded model
        
        Note:
            This is a placeholder implementation. The actual implementation
            would depend on the specific model architecture and format.
        """
        # Placeholder for actual model loading
        self.logger.info(f"Loading Voice Separation with Unknown Speakers model from {self.model_path}")
        
        # Placeholder model
        class DummyVoiceSepUnknownSpeakersModel:
            def __init__(self, path, device):
                self.path = path
                self.device = device
                self.name = "Voice Separation with Unknown Speakers"
                self.version = "1.0.0"
            
            def to(self, device):
                return self
            
            def __call__(self, mixture, **kwargs):
                # For unknown speakers, we dynamically determine the number of speakers
                # In this dummy implementation, we'll just use a random number between 2 and 4
                if isinstance(mixture, torch.Tensor):
                    n_speakers = torch.randint(2, 5, (1,)).item()
                    return torch.randn(n_speakers, mixture.shape[0], device=self.device)
                else:
                    n_speakers = np.random.randint(2, 5)
                    return np.random.randn(n_speakers, mixture.shape[0])
        
        return DummyVoiceSepUnknownSpeakersModel(self.model_path, self.device)
    
    def separate(
        self,
        mixture: Union[np.ndarray, torch.Tensor],
        num_speakers: Optional[int] = None,
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Separate a mixture using Voice Separation with Unknown Speakers.
        
        Args:
            mixture: Audio mixture to separate, shape (n_samples,)
            num_speakers: Number of speakers to separate (if known, but can be None)
            **kwargs: Additional model-specific parameters
        
        Returns:
            Separated sources, shape (n_sources, n_samples)
        """
        self.logger.debug(f"Separating mixture with Voice Separation with Unknown Speakers")
        
        # Convert numpy array to tensor if needed
        is_numpy = isinstance(mixture, np.ndarray)
        if is_numpy:
            mixture_tensor = torch.from_numpy(mixture).float().to(self.device)
        else:
            mixture_tensor = mixture.to(self.device)
        
        # Ensure mixture is 1D
        if mixture_tensor.dim() > 1:
            mixture_tensor = mixture_tensor.squeeze()
        
        # Separate using model
        with torch.no_grad():
            # Note: num_speakers is intentionally not passed as this model determines it automatically
            separated_sources = self.model(mixture_tensor, **kwargs)
        
        # Convert back to numpy if input was numpy
        if is_numpy:
            separated_sources = separated_sources.cpu().numpy()
        
        return separated_sources
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the Voice Separation with Unknown Speakers model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "type": "voicesep_unknown_speakers",
            "name": getattr(self.model, "name", "Voice Separation with Unknown Speakers"),
            "version": getattr(self.model, "version", "unknown"),
            "path": self.model_path,
            "device": str(self.device)
        }


class LookingToListenAdapter(VoiceSeparationModel):
    """Adapter for Looking to Listen audio-visual separation model."""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Looking to Listen adapter.
        
        Args:
            model_path: Path to the model
            device: Device to run the model on
            logger: Logger instance
        """
        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger or logging.getLogger(__name__)
        
        # Load the model
        self.model = self._load_model()
        
        self.logger.info(f"Initialized Looking to Listen adapter with model from {model_path}")
    
    def _load_model(self) -> Any:
        """
        Load the Looking to Listen model.
        
        Returns:
            Loaded model
        
        Note:
            This is a placeholder implementation. The actual implementation
            would depend on the specific model architecture and format.
        """
        # Placeholder for actual model loading
        self.logger.info(f"Loading Looking to Listen model from {self.model_path}")
        
        # Placeholder model
        class DummyLookingToListenModel:
            def __init__(self, path, device):
                self.path = path
                self.device = device
                self.name = "Looking to Listen"
                self.version = "1.0.0"
            
            def to(self, device):
                return self
            
            def __call__(self, mixture, video_frames=None, num_speakers=None, **kwargs):
                # Simulate separation by creating random sources
                if isinstance(mixture, torch.Tensor):
                    n_speakers = num_speakers or 2
                    return torch.randn(n_speakers, mixture.shape[0], device=self.device)
                else:
                    n_speakers = num_speakers or 2
                    return np.random.randn(n_speakers, mixture.shape[0])
        
        return DummyLookingToListenModel(self.model_path, self.device)
    
    def separate(
        self,
        mixture: Union[np.ndarray, torch.Tensor],
        num_speakers: Optional[int] = None,
        video_frames: Optional[Union[np.ndarray, torch.Tensor]] = None,
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Separate a mixture using Looking to Listen.
        
        Args:
            mixture: Audio mixture to separate, shape (n_samples,)
            num_speakers: Number of speakers to separate (if known)
            video_frames: Video frames corresponding to the audio, shape (n_frames, height, width, channels)
            **kwargs: Additional model-specific parameters
        
        Returns:
            Separated sources, shape (n_sources, n_samples)
        """
        self.logger.debug(f"Separating mixture with Looking to Listen, num_speakers={num_speakers}")
        
        if video_frames is None:
            self.logger.warning("No video frames provided for Looking to Listen model, results may be suboptimal")
        
        # Convert numpy array to tensor if needed
        is_numpy = isinstance(mixture, np.ndarray)
        if is_numpy:
            mixture_tensor = torch.from_numpy(mixture).float().to(self.device)
        else:
            mixture_tensor = mixture.to(self.device)
        
        # Ensure mixture is 1D
        if mixture_tensor.dim() > 1:
            mixture_tensor = mixture_tensor.squeeze()
        
        # Convert video frames to tensor if needed
        if video_frames is not None:
            if isinstance(video_frames, np.ndarray):
                video_frames = torch.from_numpy(video_frames).float().to(self.device)
            else:
                video_frames = video_frames.to(self.device)
        
        # Separate using model
        with torch.no_grad():
            separated_sources = self.model(mixture_tensor, video_frames=video_frames, num_speakers=num_speakers, **kwargs)
        
        # Convert back to numpy if input was numpy
        if is_numpy:
            separated_sources = separated_sources.cpu().numpy()
        
        return separated_sources
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the Looking to Listen model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "type": "looking_to_listen",
            "name": getattr(self.model, "name", "Looking to Listen"),
            "version": getattr(self.model, "version", "unknown"),
            "path": self.model_path,
            "device": str(self.device)
        }


def create_model_adapter(
    model_type: str,
    model_path: str,
    device: Optional[torch.device] = None,
    logger: Optional[logging.Logger] = None
) -> VoiceSeparationModel:
    """
    Create a model adapter for the specified model type.
    
    Args:
        model_type: Type of model ("svoice", "demucs", "hybrid_transformer_cnn",
                                  "spleeter", "voicesep_unknown_speakers", "looking_to_listen")
        model_path: Path to the model
        device: Device to run the model on
        logger: Logger instance
    
    Returns:
        Model adapter instance
    
    Raises:
        ValueError: If model_type is not supported
    """
    if model_type.lower() == "svoice":
        return SVoiceAdapter(model_path, device, logger)
    elif model_type.lower() == "demucs":
        return DemucsAdapter(model_path, device, logger)
    elif model_type.lower() == "hybrid_transformer_cnn":
        return HybridTransformerCNNAdapter(model_path, device, logger)
    elif model_type.lower() == "spleeter":
        return SpleeterAdapter(model_path, device, logger)
    elif model_type.lower() == "voicesep_unknown_speakers":
        return VoiceSepUnknownSpeakersAdapter(model_path, device, logger)
    elif model_type.lower() == "looking_to_listen":
        return LookingToListenAdapter(model_path, device, logger)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_model_loader(model_type: str) -> Callable:
    """
    Get a model loader function for the specified model type.
    
    This function returns a loader that can be registered with the model registry.
    
    Args:
        model_type: Type of model ("svoice", "demucs", "hybrid_transformer_cnn",
                                  "spleeter", "voicesep_unknown_speakers", "looking_to_listen")
    
    Returns:
        Function that loads a model from a path
    
    Raises:
        ValueError: If model_type is not supported
    """
    if model_type.lower() == "svoice":
        try:
            # Try to import the SVoice loader from our implementation
            from ..models.svoice.utils import create_svoice_registry_loader
            return create_svoice_registry_loader()
        except ImportError as e:
            # Fallback to adapter-based loader
            def load_svoice(path, **kwargs):
                adapter = SVoiceAdapter(path, **kwargs)
                return lambda mixture, **proc_kwargs: adapter.separate(mixture, **proc_kwargs)
            return load_svoice
    
    elif model_type.lower() == "demucs":
        def load_demucs(path, **kwargs):
            adapter = DemucsAdapter(path, **kwargs)
            return lambda mixture, **proc_kwargs: adapter.separate(mixture, **proc_kwargs)
        return load_demucs
    
    elif model_type.lower() == "hybrid_transformer_cnn":
        def load_hybrid_transformer_cnn(path, **kwargs):
            adapter = HybridTransformerCNNAdapter(path, **kwargs)
            return lambda mixture, **proc_kwargs: adapter.separate(mixture, **proc_kwargs)
        return load_hybrid_transformer_cnn
    
    elif model_type.lower() == "spleeter":
        def load_spleeter(path, **kwargs):
            adapter = SpleeterAdapter(path, **kwargs)
            return lambda mixture, **proc_kwargs: adapter.separate(mixture, **proc_kwargs)
        return load_spleeter
    
    elif model_type.lower() == "voicesep_unknown_speakers":
        def load_voicesep_unknown_speakers(path, **kwargs):
            adapter = VoiceSepUnknownSpeakersAdapter(path, **kwargs)
            return lambda mixture, **proc_kwargs: adapter.separate(mixture, **proc_kwargs)
        return load_voicesep_unknown_speakers
    
    elif model_type.lower() == "looking_to_listen":
        def load_looking_to_listen(path, **kwargs):
            adapter = LookingToListenAdapter(path, **kwargs)
            # Special handling for video frames
            return lambda mixture, **proc_kwargs: adapter.separate(mixture, **proc_kwargs)
        return load_looking_to_listen
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")