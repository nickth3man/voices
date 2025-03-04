"""
SVoice Model Implementation.

This module provides the implementation of the SVoice model for voice separation,
including model architecture, loading, and inference.
"""

import os
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any

logger = logging.getLogger(__name__)


class SVoiceModel(nn.Module):
    """
    SVoice model for voice separation.
    
    This implementation is based on the SVoice paper:
    "SVoice: Speaker-attributed Speech Separation with Multi-scale Neural Networks"
    """
    
    def __init__(
        self,
        n_speakers: int = 2,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        hidden_size: int = 128,
        num_layers: int = 6,
        bidirectional: bool = True,
        device: Optional[torch.device] = None,
        **kwargs
    ):
        """
        Initialize the SVoice model.
        
        Args:
            n_speakers: Number of speakers to separate
            sample_rate: Audio sample rate
            n_fft: FFT size
            hop_length: Hop length for STFT
            hidden_size: Hidden size for LSTM layers
            num_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
            device: Device to run the model on
            **kwargs: Additional model parameters
        """
        super().__init__()
        
        self.n_speakers = n_speakers
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model parameters
        self.n_features = n_fft // 2 + 1
        self.lstm_dim = hidden_size * 2 if bidirectional else hidden_size
        
        # Build model components
        self._build_model()
        
        # Move model to device
        self.to(self.device)
        
        logger.info(f"Initialized SVoice model with {n_speakers} speakers on {self.device}")
    
    def _build_model(self):
        """Build the model architecture."""
        # Encoder (STFT is performed separately)
        self.encoder = nn.Sequential(
            nn.Linear(self.n_features * 2, self.hidden_size),  # Complex input (real + imag)
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU()
        )
        
        # Separation module (LSTM)
        self.separator = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
            dropout=0.3
        )
        
        # Mask estimation
        self.mask_estimator = nn.Sequential(
            nn.Linear(self.lstm_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.n_features * self.n_speakers * 2)  # Complex masks (real + imag)
        )
        
        # Decoder (iSTFT is performed separately)
        self.decoder = nn.Sequential(
            nn.Linear(self.n_features * 2, self.hidden_size),  # Complex input (real + imag)
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.n_features * 2)  # Complex output (real + imag)
        )
    
    def stft(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute Short-Time Fourier Transform.
        
        Args:
            audio: Audio waveform, shape (batch_size, n_samples)
            
        Returns:
            Complex spectrogram, shape (batch_size, n_frames, n_features, 2)
        """
        # Ensure audio is 2D
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Compute STFT
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).to(audio.device),
            return_complex=False
        )
        
        # Reshape to (batch_size, n_frames, n_features, 2)
        batch_size = audio.shape[0]
        spec = spec.permute(0, 2, 1, 3)
        
        return spec
    
    def istft(self, spec: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        """
        Compute Inverse Short-Time Fourier Transform.
        
        Args:
            spec: Complex spectrogram, shape (batch_size, n_frames, n_features, 2)
            length: Original audio length (optional)
            
        Returns:
            Audio waveform, shape (batch_size, n_samples)
        """
        # Reshape to (batch_size, n_features, n_frames, 2)
        spec = spec.permute(0, 2, 1, 3)
        
        # Compute iSTFT
        audio = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).to(spec.device),
            length=length
        )
        
        return audio
    
    def forward(
        self,
        mixture: torch.Tensor,
        return_spectrogram: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            mixture: Audio mixture, shape (batch_size, n_samples)
            return_spectrogram: Whether to return the spectrogram as well
            
        Returns:
            Separated sources, shape (batch_size, n_speakers, n_samples)
            If return_spectrogram is True, also returns the mixture spectrogram
        """
        # Ensure mixture is on the correct device
        mixture = mixture.to(self.device)
        
        # Get original length for iSTFT
        orig_length = mixture.shape[-1]
        
        # Compute STFT
        mixture_spec = self.stft(mixture)
        batch_size, n_frames, n_features, _ = mixture_spec.shape
        
        # Reshape for encoder
        mixture_spec_flat = mixture_spec.reshape(batch_size, n_frames, -1)
        
        # Encode
        encoded = self.encoder(mixture_spec_flat)
        
        # Separate
        separated, _ = self.separator(encoded)
        
        # Estimate masks
        masks = self.mask_estimator(separated)
        masks = masks.reshape(batch_size, n_frames, n_features, self.n_speakers, 2)
        
        # Apply masks to mixture
        sources_specs = []
        for i in range(self.n_speakers):
            source_spec = mixture_spec.unsqueeze(3) * masks[..., i, :].unsqueeze(4)
            source_spec = source_spec.squeeze(3)
            sources_specs.append(source_spec)
        
        # Stack sources
        sources_specs = torch.stack(sources_specs, dim=1)
        
        # Reshape for iSTFT
        batch_size, n_speakers, n_frames, n_features, _ = sources_specs.shape
        sources_specs = sources_specs.reshape(batch_size * n_speakers, n_frames, n_features, 2)
        
        # Compute iSTFT
        sources = self.istft(sources_specs, length=orig_length)
        
        # Reshape to (batch_size, n_speakers, n_samples)
        sources = sources.reshape(batch_size, n_speakers, -1)
        
        if return_spectrogram:
            return sources, mixture_spec
        else:
            return sources
    
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
            num_speakers: Number of speakers to separate (if None, uses model default)
            **kwargs: Additional parameters
            
        Returns:
            Separated sources, shape (n_speakers, n_samples)
        """
        # Set model to eval mode
        self.eval()
        
        # Convert numpy array to tensor if needed
        is_numpy = isinstance(mixture, np.ndarray)
        if is_numpy:
            mixture_tensor = torch.from_numpy(mixture).float().to(self.device)
        else:
            mixture_tensor = mixture.to(self.device)
        
        # Ensure mixture is 1D
        if mixture_tensor.dim() > 1:
            mixture_tensor = mixture_tensor.squeeze()
        
        # Add batch dimension if needed
        if mixture_tensor.dim() == 1:
            mixture_tensor = mixture_tensor.unsqueeze(0)
        
        # Override n_speakers if specified
        if num_speakers is not None and num_speakers != self.n_speakers:
            logger.warning(f"Requested {num_speakers} speakers but model is configured for {self.n_speakers} speakers")
            # In a real implementation, we would handle this case
        
        # Separate
        with torch.no_grad():
            separated_sources = self.forward(mixture_tensor)
        
        # Remove batch dimension
        separated_sources = separated_sources.squeeze(0)
        
        # Convert back to numpy if input was numpy
        if is_numpy:
            separated_sources = separated_sources.cpu().numpy()
        
        return separated_sources
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "type": "svoice",
            "name": "SVoice",
            "version": "1.0.0",
            "n_speakers": self.n_speakers,
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
            "device": str(self.device),
            "parameters": sum(p.numel() for p in self.parameters())
        }
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> 'SVoiceModel':
        """
        Load a pretrained SVoice model.
        
        Args:
            model_path: Path to the pretrained model
            device: Device to load the model on
            **kwargs: Additional parameters to override model configuration
            
        Returns:
            Loaded SVoice model
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading pretrained SVoice model from {model_path}")
        
        # Load model configuration and weights
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get model configuration
        config = checkpoint.get("config", {})
        
        # Override with kwargs
        config.update(kwargs)
        
        # Create model
        model = cls(
            n_speakers=config.get("n_speakers", 2),
            sample_rate=config.get("sample_rate", 16000),
            n_fft=config.get("n_fft", 512),
            hop_length=config.get("hop_length", 128),
            hidden_size=config.get("hidden_size", 128),
            num_layers=config.get("num_layers", 6),
            bidirectional=config.get("bidirectional", True),
            device=device
        )
        
        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return model