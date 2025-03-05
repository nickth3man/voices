"""
Intelligent Model Selection for Voice Separation.

This module provides functionality for selecting the most appropriate
voice separation model based on audio characteristics and requirements.
"""

import os
import logging
import numpy as np
import librosa
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from .model_registry import ModelRegistry, ModelInfo, ModelVersion


class ModelSelector:
    """Intelligent model selector for voice separation."""
    
    def __init__(
        self,
        model_registry: ModelRegistry,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the model selector.
        
        Args:
            model_registry: Model registry instance
            logger: Logger instance
        """
        self.registry = model_registry
        self.logger = logger or logging.getLogger(__name__)
    
    def select_model(
        self,
        audio: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
        num_speakers: Optional[int] = None,
        environment: Optional[str] = None,
        duration: Optional[float] = None,
        quality_preference: str = "balanced",
        speed_preference: str = "balanced",
        model_type_preference: Optional[str] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Select the most appropriate model based on audio characteristics and preferences.
        
        Args:
            audio: Audio data (optional, if provided will be analyzed)
            sample_rate: Sample rate of the audio
            num_speakers: Number of speakers in the audio (if known)
            environment: Audio environment ("clean", "noisy", "reverberant")
            duration: Duration of the audio in seconds
            quality_preference: Preference for separation quality ("highest", "balanced", "fastest")
            speed_preference: Preference for processing speed ("fastest", "balanced", "highest_quality")
            model_type_preference: Preference for model type ("svoice", "demucs", None for automatic)
        
        Returns:
            Tuple of (model_id, version_id)
        """
        self.logger.info("Selecting model based on audio characteristics and preferences")
        
        # Get all available models
        all_models = self.registry.list_models()
        if not all_models:
            raise ValueError("No models available in the registry")
        
        # Filter by model type if specified
        if model_type_preference:
            models = [m for m in all_models if m.model_type == model_type_preference]
            if not models:
                self.logger.warning(f"No models of type {model_type_preference} found, using all available models")
                models = all_models
        else:
            models = all_models
        
        # Analyze audio if provided
        audio_features = {}
        if audio is not None:
            audio_features = self._analyze_audio(audio, sample_rate)
            
            # Update num_speakers if not provided but detected
            if num_speakers is None and "estimated_num_speakers" in audio_features:
                num_speakers = audio_features["estimated_num_speakers"]
            
            # Update environment if not provided but detected
            if environment is None and "environment" in audio_features:
                environment = audio_features["environment"]
            
            # Update duration if not provided
            if duration is None:
                duration = len(audio) / sample_rate
        
        # Score each model based on suitability
        model_scores = []
        
        for model in models:
            # Get the default version
            default_version = model.get_default_version()
            if not default_version:
                continue
            
            # Calculate base score
            score = 50  # Start with a neutral score
            
            # Adjust score based on model type
            if num_speakers is not None:
                if num_speakers >= 3 and model.model_type == "svoice":
                    score += 20  # SVoice is better for 3+ speakers
                elif num_speakers <= 2 and model.model_type == "demucs":
                    score += 10  # Demucs is good for 1-2 speakers
            
            # Adjust score based on environment
            if environment:
                if environment == "noisy" and model.model_type == "svoice":
                    score += 10  # SVoice handles noise better
                elif environment == "reverberant" and model.model_type == "demucs":
                    score += 10  # Demucs handles reverb better
            
            # Adjust score based on performance metrics if available
            if default_version.performance_metrics:
                # Quality metrics
                if "si_snri_mean" in default_version.performance_metrics:
                    si_snri = default_version.performance_metrics["si_snri_mean"]
                    # Normalize to 0-20 range (assuming SI-SNRi ranges from 0 to 15 dB)
                    quality_score = min(20, max(0, si_snri * 1.5))
                    
                    # Apply quality preference
                    if quality_preference == "highest":
                        score += quality_score
                    elif quality_preference == "balanced":
                        score += quality_score * 0.7
                    # For "fastest", we don't add quality score
            
            # Adjust score based on processing speed if available
            if "processing_speed" in default_version.metadata:
                speed = default_version.metadata["processing_speed"]
                # Normalize to 0-20 range (assuming speed ranges from 0.1 to 5x real-time)
                speed_score = min(20, max(0, speed * 5))
                
                # Apply speed preference
                if speed_preference == "fastest":
                    score += speed_score
                elif speed_preference == "balanced":
                    score += speed_score * 0.7
                # For "highest_quality", we don't add speed score
            
            # Adjust score based on model type preference
            if model_type_preference and model.model_type == model_type_preference:
                score += 15
            
            model_scores.append((model.model_id, default_version.version_id, score))
        
        if not model_scores:
            raise ValueError("No suitable models found")
        
        # Sort by score (descending)
        model_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Get the best model
        best_model_id, best_version_id, best_score = model_scores[0]
        
        self.logger.info(f"Selected model {best_model_id} version {best_version_id} with score {best_score}")
        
        return best_model_id, best_version_id
    
    def _analyze_audio(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Analyze audio to extract relevant features for model selection.
        
        Args:
            audio: Audio data
            sample_rate: Sample rate of the audio
        
        Returns:
            Dictionary of audio features
        """
        features = {}
        
        # Ensure audio is mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Calculate basic features
        features["duration"] = len(audio) / sample_rate
        features["rms"] = np.sqrt(np.mean(audio**2))
        
        # Estimate number of speakers (placeholder implementation)
        # In a real implementation, this would use a speaker counting model
        features["estimated_num_speakers"] = self._estimate_num_speakers(audio, sample_rate)
        
        # Estimate environment (clean, noisy, reverberant)
        features["environment"] = self._estimate_environment(audio, sample_rate)
        
        return features
    
    def _estimate_num_speakers(self, audio: np.ndarray, sample_rate: int) -> int:
        """
        Estimate the number of speakers in the audio.
        
        Args:
            audio: Audio data
            sample_rate: Sample rate of the audio
        
        Returns:
            Estimated number of speakers
        
        Note:
            This is a placeholder implementation. In a real implementation,
            this would use a speaker counting model or diarization system.
        """
        # Placeholder: simple energy-based heuristic
        # In a real implementation, this would use a more sophisticated approach
        
        # Split audio into segments
        segment_length = int(0.5 * sample_rate)  # 500ms segments
        segments = [audio[i:i+segment_length] for i in range(0, len(audio), segment_length)]
        
        # Calculate energy for each segment
        energies = [np.sum(segment**2) for segment in segments if len(segment) == segment_length]
        
        if not energies:
            return 1
        
        # Normalize energies
        energies = np.array(energies)
        energies = (energies - np.min(energies)) / (np.max(energies) - np.min(energies) + 1e-8)
        
        # Count energy peaks as a very rough proxy for speaker turns
        peaks = 0
        threshold = 0.5
        above_threshold = False
        
        for energy in energies:
            if energy > threshold and not above_threshold:
                peaks += 1
                above_threshold = True
            elif energy <= threshold:
                above_threshold = False
        
        # Map peaks to estimated speakers
        if peaks <= 2:
            return 2
        elif peaks <= 5:
            return 3
        else:
            return 4
    
    def _estimate_environment(self, audio: np.ndarray, sample_rate: int) -> str:
        """
        Estimate the acoustic environment of the audio.
        
        Args:
            audio: Audio data
            sample_rate: Sample rate of the audio
        
        Returns:
            Estimated environment ("clean", "noisy", "reverberant")
        
        Note:
            This is a placeholder implementation. In a real implementation,
            this would use more sophisticated acoustic analysis.
        """
        # Placeholder: simple spectral features
        # In a real implementation, this would use more sophisticated acoustic analysis
        
        # Calculate spectral features
        spec = np.abs(librosa.stft(audio))
        spec_db = librosa.amplitude_to_db(spec, ref=np.max)
        
        # Calculate noise level (high frequency energy)
        high_freq_energy = np.mean(spec_db[-int(spec_db.shape[0]/3):, :])
        
        # Calculate reverberation (decay rate)
        # Simplified: use the variance of the spectral flux as a proxy for reverberation
        spec_flux = np.diff(spec_db, axis=1)
        reverb_measure = np.var(np.mean(spec_flux, axis=0))
        
        # Classify environment
        if high_freq_energy > -30:  # High noise level
            return "noisy"
        elif reverb_measure < 5:  # High reverberation
            return "reverberant"
        else:
            return "clean"


def select_best_model(
    registry: ModelRegistry,
    audio: np.ndarray,
    sample_rate: int,
    num_speakers: Optional[int] = None,
    quality_preference: str = "balanced",
    speed_preference: str = "balanced"
) -> Tuple[str, Optional[str]]:
    """
    Select the best model for the given audio.
    
    This is a convenience function that creates a ModelSelector and uses it
    to select the best model.
    
    Args:
        registry: Model registry instance
        audio: Audio data
        sample_rate: Sample rate of the audio
        num_speakers: Number of speakers (if known)
        quality_preference: Preference for separation quality
        speed_preference: Preference for processing speed
    
    Returns:
        Tuple of (model_id, version_id)
    """
    selector = ModelSelector(registry)
    return selector.select_model(
        audio=audio,
        sample_rate=sample_rate,
        num_speakers=num_speakers,
        quality_preference=quality_preference,
        speed_preference=speed_preference
    )