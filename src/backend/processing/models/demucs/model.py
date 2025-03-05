"""
Demucs Model Implementation.

This module provides the implementation of the Demucs model for voice separation,
including model architecture, loading, and inference.

Demucs (Deep Extractor for Music Sources) is a waveform-based music source separation
model that has been adapted for voice separation tasks. It uses a U-Net architecture
with LSTM layers and operates directly on the waveform.
"""

import os
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any

logger = logging.getLogger(__name__)


class DemucsModel(nn.Module):
    """
    Demucs model for voice separation.
    
    This implementation is based on the Demucs paper:
    "Music Source Separation in the Waveform Domain"
    """
    
    def __init__(
        self,
        n_speakers: int = 2,
        sample_rate: int = 16000,
        channels: int = 1,
        hidden_size: int = 64,
        depth: int = 6,
        kernel_size: int = 8,
        stride: int = 4,
        lstm_layers: int = 2,
        device: Optional[torch.device] = None,
        **kwargs
    ):
        """
        Initialize the Demucs model.
        
        Args:
            n_speakers: Number of speakers to separate
            sample_rate: Audio sample rate
            channels: Number of input channels (1 for mono, 2 for stereo)
            hidden_size: Base hidden size for the model
            depth: Number of layers in the encoder/decoder
            kernel_size: Kernel size for convolutional layers
            stride: Stride for convolutional layers
            lstm_layers: Number of LSTM layers
            device: Device to run the model on
            **kwargs: Additional model parameters
        """
        super().__init__()
        
        self.n_speakers = n_speakers
        self.sample_rate = sample_rate
        self.channels = channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.lstm_layers = lstm_layers
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build model components
        self._build_model()
        
        # Move model to device
        self.to(self.device)
        
        logger.info(f"Initialized Demucs model with {n_speakers} speakers on {self.device}")
    
    def _build_model(self):
        """Build the model architecture."""
        # Encoder
        self.encoder = nn.ModuleList()
        in_channels = self.channels
        for i in range(self.depth):
            out_channels = min(self.hidden_size * (2 ** i), 1024)
            self.encoder.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, self.kernel_size, self.stride, padding=self.kernel_size//2),
                    nn.ReLU(),
                    nn.Conv1d(out_channels, out_channels, 1),
                    nn.ReLU()
                )
            )
            in_channels = out_channels
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=in_channels,
            num_layers=self.lstm_layers,
            bidirectional=True,
            batch_first=True
        )
        
        # Linear layer after LSTM
        self.lstm_linear = nn.Linear(in_channels * 2, in_channels)
        
        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(self.depth):
            if i == 0:
                out_channels = min(self.hidden_size * (2 ** (self.depth - i - 1)), 1024)
                in_channels = min(self.hidden_size * (2 ** (self.depth - i)), 1024)
            else:
                out_channels = min(self.hidden_size * (2 ** (self.depth - i - 1)), 1024)
                in_channels = min(self.hidden_size * (2 ** (self.depth - i)), 1024) * 2  # *2 for skip connection
            
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose1d(in_channels, out_channels, self.kernel_size, self.stride, padding=self.kernel_size//2),
                    nn.ReLU(),
                    nn.Conv1d(out_channels, out_channels, 1),
                    nn.ReLU()
                )
            )
        
        # Output layer
        self.output = nn.Conv1d(out_channels, self.channels * self.n_speakers, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Audio waveform, shape (batch_size, channels, n_samples)
            
        Returns:
            Separated sources, shape (batch_size, n_speakers, channels, n_samples)
        """
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # Get original shape
        batch_size, channels, n_samples = x.shape
        
        # Encoder
        encoder_outputs = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            encoder_outputs.append(x)
        
        # LSTM
        x = x.transpose(1, 2)  # (batch_size, n_frames, channels)
        x, _ = self.lstm(x)
        x = self.lstm_linear(x)
        x = x.transpose(1, 2)  # (batch_size, channels, n_frames)
        
        # Decoder with skip connections
        for i, layer in enumerate(self.decoder):
            if i > 0:
                skip = encoder_outputs[-(i+1)]
                # Ensure same length
                if x.size(2) != skip.size(2):
                    x = torch.nn.functional.interpolate(x, size=skip.size(2))
                x = torch.cat([x, skip], dim=1)
            x = layer(x)
        
        # Output layer
        x = self.output(x)
        
        # Reshape to (batch_size, n_speakers, channels, n_samples)
        x = x.view(batch_size, self.n_speakers, channels, -1)
        
        # Ensure output has the same length as input
        if x.size(3) != n_samples:
            x = torch.nn.functional.interpolate(x, size=n_samples)
        
        return x
    
    def separate(
        self,
        mixture: Union[np.ndarray, torch.Tensor],
        num_speakers: Optional[int] = None,
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Separate a mixture into individual sources.
        
        Args:
            mixture: Audio mixture to separate, shape (n_samples,) or (channels, n_samples)
            num_speakers: Number of speakers to separate (if None, uses model default)
            **kwargs: Additional parameters
            
        Returns:
            Separated sources, shape (n_speakers, n_samples) or (n_speakers, channels, n_samples)
        """
        # Set model to eval mode
        self.eval()
        
        # Convert numpy array to tensor if needed
        is_numpy = isinstance(mixture, np.ndarray)
        if is_numpy:
            mixture_tensor = torch.from_numpy(mixture).float().to(self.device)
        else:
            mixture_tensor = mixture.to(self.device)
        
        # Ensure mixture has the right shape
        if mixture_tensor.dim() == 1:
            # Single channel, add channel dimension
            mixture_tensor = mixture_tensor.unsqueeze(0)
        
        # Add batch dimension if needed
        if mixture_tensor.dim() == 2:
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
        
        # If input was mono, convert output to (n_speakers, n_samples)
        if mixture_tensor.size(1) == 1:
            separated_sources = separated_sources.squeeze(1)
        
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
            "type": "demucs",
            "name": "Demucs",
            "version": "1.0.0",
            "n_speakers": self.n_speakers,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "hidden_size": self.hidden_size,
            "depth": self.depth,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "lstm_layers": self.lstm_layers,
            "device": str(self.device),
            "parameters": sum(p.numel() for p in self.parameters())
        }
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> 'DemucsModel':
        """
        Load a pretrained Demucs model.
        
        Args:
            model_path: Path to the pretrained model
            device: Device to load the model on
            **kwargs: Additional parameters to override model configuration
            
        Returns:
            Loaded Demucs model
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading pretrained Demucs model from {model_path}")
        
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
            channels=config.get("channels", 1),
            hidden_size=config.get("hidden_size", 64),
            depth=config.get("depth", 6),
            kernel_size=config.get("kernel_size", 8),
            stride=config.get("stride", 4),
            lstm_layers=config.get("lstm_layers", 2),
            device=device
        )
        
        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return model