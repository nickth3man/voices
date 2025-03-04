"""
Audio file I/O and management module.

This module provides functionality for loading, saving, and managing audio files,
including format detection, validation, metadata extraction, and waveform generation.
"""

import os
import logging
import numpy as np
import librosa
import soundfile as sf
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any, BinaryIO
import warnings

# Configure logging
logger = logging.getLogger(__name__)

# Suppress specific librosa warnings
warnings.filterwarnings("ignore", message="PySoundFile failed.*")


class AudioFileError(Exception):
    """Exception raised for audio file errors."""
    pass


class UnsupportedFormatError(AudioFileError):
    """Exception raised for unsupported audio formats."""
    pass


class AudioFormatDetector:
    """Utility class for detecting and validating audio file formats."""
    
    # Supported audio formats and their extensions
    SUPPORTED_FORMATS = {
        'wav': ['wav', 'wave'],
        'flac': ['flac'],
        'mp3': ['mp3'],
        'ogg': ['ogg'],
        'aac': ['aac', 'm4a'],
        'aiff': ['aiff', 'aif']
    }
    
    # Format to extension mapping (for saving)
    FORMAT_TO_EXT = {
        'wav': 'wav',
        'flac': 'flac',
        'mp3': 'mp3',
        'ogg': 'ogg',
        'aac': 'm4a',
        'aiff': 'aiff'
    }
    
    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """
        Get a list of supported audio formats.
        
        Returns:
            List of supported format names
        """
        return list(cls.SUPPORTED_FORMATS.keys())
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """
        Get a list of supported file extensions.
        
        Returns:
            List of supported file extensions
        """
        extensions = []
        for format_exts in cls.SUPPORTED_FORMATS.values():
            extensions.extend(format_exts)
        return extensions
    
    @classmethod
    def detect_format(cls, file_path: Union[str, Path]) -> str:
        """
        Detect the audio format of a file based on its extension.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Detected format name
            
        Raises:
            UnsupportedFormatError: If the file format is not supported
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower().lstrip('.')
        
        for format_name, extensions in cls.SUPPORTED_FORMATS.items():
            if extension in extensions:
                return format_name
        
        raise UnsupportedFormatError(f"Unsupported audio format: {extension}")
    
    @classmethod
    def validate_format(cls, file_path: Union[str, Path]) -> bool:
        """
        Validate if a file has a supported audio format.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            True if the format is supported, False otherwise
        """
        try:
            cls.detect_format(file_path)
            return True
        except UnsupportedFormatError:
            return False
    
    @classmethod
    def get_extension_for_format(cls, format_name: str) -> str:
        """
        Get the default file extension for a format.
        
        Args:
            format_name: Audio format name
            
        Returns:
            Default file extension for the format
            
        Raises:
            UnsupportedFormatError: If the format is not supported
        """
        if format_name in cls.FORMAT_TO_EXT:
            return cls.FORMAT_TO_EXT[format_name]
        
        raise UnsupportedFormatError(f"Unsupported audio format: {format_name}")


class AudioMetadataExtractor:
    """Utility class for extracting metadata from audio files."""
    
    @staticmethod
    def extract_metadata(file_path: Union[str, Path], audio: Optional[np.ndarray] = None, 
                         sr: Optional[int] = None) -> Dict[str, Any]:
        """
        Extract metadata from an audio file.
        
        Args:
            file_path: Path to the audio file
            audio: Optional pre-loaded audio data
            sr: Optional sample rate of pre-loaded audio
            
        Returns:
            Dictionary containing metadata
        """
        file_path = Path(file_path)
        
        # Basic file metadata
        metadata = {
            "filename": file_path.name,
            "path": str(file_path.absolute()),
            "format": AudioFormatDetector.detect_format(file_path),
            "size_bytes": file_path.stat().st_size,
            "last_modified": file_path.stat().st_mtime
        }
        
        # Load audio if not provided
        if audio is None or sr is None:
            try:
                audio, sr = librosa.load(file_path, sr=None, mono=False)
            except Exception as e:
                logger.warning(f"Could not load audio for metadata extraction: {str(e)}")
                return metadata
        
        # Convert to mono for analysis if needed
        if len(audio.shape) > 1:
            mono_audio = librosa.to_mono(audio)
            channels = audio.shape[0]
        else:
            mono_audio = audio
            channels = 1
        
        # Audio properties
        duration = librosa.get_duration(y=mono_audio, sr=sr)
        metadata.update({
            "sample_rate": sr,
            "channels": channels,
            "duration": duration,
            "samples": len(mono_audio),
            "bit_depth": 16  # Assuming 16-bit, actual bit depth may require format-specific handling
        })
        
        # Audio characteristics
        try:
            # RMS energy
            rms = librosa.feature.rms(y=mono_audio)[0]
            metadata["rms_mean"] = float(np.mean(rms))
            metadata["rms_max"] = float(np.max(rms))
            
            # Spectral centroid
            cent = librosa.feature.spectral_centroid(y=mono_audio, sr=sr)[0]
            metadata["spectral_centroid_mean"] = float(np.mean(cent))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(mono_audio)[0]
            metadata["zero_crossing_rate_mean"] = float(np.mean(zcr))
            
            # Estimate tempo
            tempo, _ = librosa.beat.beat_track(y=mono_audio, sr=sr)
            metadata["estimated_tempo"] = float(tempo)
            
            # Detect silence
            non_silent = librosa.effects.split(mono_audio, top_db=30)
            silent_duration = duration - sum(end - start for start, end in non_silent) / sr
            metadata["silence_duration"] = float(silent_duration)
            metadata["silence_percentage"] = float(silent_duration / duration * 100)
            
        except Exception as e:
            logger.warning(f"Error extracting audio characteristics: {str(e)}")
        
        return metadata


class WaveformGenerator:
    """Utility class for generating waveform data for visualization."""
    
    @staticmethod
    def generate_waveform(audio: np.ndarray, n_points: int = 1000) -> np.ndarray:
        """
        Generate waveform data for visualization.
        
        Args:
            audio: Audio data
            n_points: Number of points in the waveform
            
        Returns:
            Numpy array with waveform data
        """
        # Ensure audio is mono
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        
        # Resample to desired number of points
        if len(audio) > n_points:
            # Use mean pooling to reduce the number of points
            points_per_bin = len(audio) // n_points
            waveform = np.array([
                np.mean(audio[i * points_per_bin:(i + 1) * points_per_bin])
                for i in range(n_points)
            ])
        else:
            # If audio is shorter than n_points, use interpolation
            indices = np.linspace(0, len(audio) - 1, n_points)
            waveform = np.interp(indices, np.arange(len(audio)), audio)
        
        return waveform
    
    @staticmethod
    def generate_waveform_image(audio: np.ndarray, width: int = 800, height: int = 200) -> np.ndarray:
        """
        Generate a waveform image for visualization.
        
        Args:
            audio: Audio data
            width: Width of the image in pixels
            height: Height of the image in pixels
            
        Returns:
            Numpy array with image data (height x width)
        """
        # Generate waveform data
        waveform = WaveformGenerator.generate_waveform(audio, width)
        
        # Normalize to [-1, 1]
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
        
        # Create image
        image = np.zeros((height, width))
        
        # Map waveform to image
        for x in range(width):
            amplitude = waveform[x]
            center_y = height // 2
            y_offset = int(amplitude * (height // 2 - 1))
            
            # Draw vertical line from center to amplitude
            start_y = center_y
            end_y = center_y + y_offset
            
            if start_y > end_y:
                start_y, end_y = end_y, start_y
                
            for y in range(start_y, end_y + 1):
                image[y, x] = 1.0
        
        return image


class AudioFileIO:
    """Main class for audio file I/O operations."""
    
    def __init__(self, default_sr: int = 16000, default_mono: bool = True):
        """
        Initialize the AudioFileIO instance.
        
        Args:
            default_sr: Default sample rate for loading audio
            default_mono: Whether to convert audio to mono by default
        """
        self.default_sr = default_sr
        self.default_mono = default_mono
        self.logger = logging.getLogger(__name__)
    
    def load_file(self, file_path: Union[str, Path], sr: Optional[int] = None, 
                  mono: Optional[bool] = None, extract_metadata: bool = True) -> Dict[str, Any]:
        """
        Load an audio file with optional metadata extraction.
        
        Args:
            file_path: Path to the audio file
            sr: Target sample rate (None for original)
            mono: Whether to convert to mono
            extract_metadata: Whether to extract metadata
            
        Returns:
            Dictionary containing audio data, sample rate, and metadata
            
        Raises:
            AudioFileError: If the file cannot be loaded
        """
        file_path = Path(file_path)
        sr = sr if sr is not None else self.default_sr
        mono = mono if mono is not None else self.default_mono
        
        self.logger.info(f"Loading audio file: {file_path}")
        
        # Validate format
        if not AudioFormatDetector.validate_format(file_path):
            raise UnsupportedFormatError(f"Unsupported audio format: {file_path.suffix}")
        
        try:
            # Load audio with librosa
            audio, file_sr = librosa.load(file_path, sr=sr, mono=mono)
            
            result = {
                "audio": audio,
                "sample_rate": file_sr if sr is None else sr
            }
            
            # Extract metadata if requested
            if extract_metadata:
                result["metadata"] = AudioMetadataExtractor.extract_metadata(
                    file_path, audio, result["sample_rate"]
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error loading audio file: {str(e)}")
            raise AudioFileError(f"Could not load audio file: {str(e)}")
    
    def save_file(self, audio: np.ndarray, file_path: Union[str, Path], 
                  sr: int, format: Optional[str] = None, **kwargs) -> str:
        """
        Save audio data to a file.
        
        Args:
            audio: Audio data
            file_path: Path to save the file
            sr: Sample rate
            format: Output format (if None, inferred from file extension)
            **kwargs: Additional format-specific parameters
            
        Returns:
            Path to the saved file
            
        Raises:
            AudioFileError: If the file cannot be saved
        """
        file_path = Path(file_path)
        
        # Determine format
        if format is None:
            try:
                format = AudioFormatDetector.detect_format(file_path)
            except UnsupportedFormatError:
                # Default to WAV if format cannot be determined
                format = 'wav'
                file_path = file_path.with_suffix('.wav')
        else:
            # Ensure file has the correct extension
            extension = AudioFormatDetector.get_extension_for_format(format)
            file_path = file_path.with_suffix(f'.{extension}')
        
        self.logger.info(f"Saving audio to {file_path} in {format} format")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(file_path.parent, exist_ok=True)
            
            # Save with soundfile
            sf.write(file_path, audio, sr, format=format, **kwargs)
            
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Error saving audio file: {str(e)}")
            raise AudioFileError(f"Could not save audio file: {str(e)}")
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about an audio file without loading the audio data.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary containing file information
            
        Raises:
            AudioFileError: If the file information cannot be retrieved
        """
        file_path = Path(file_path)
        
        self.logger.debug(f"Getting info for audio file: {file_path}")
        
        try:
            # Get basic file info with soundfile
            info = sf.info(file_path)
            
            result = {
                "path": str(file_path),
                "format": info.format,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "duration": info.duration,
                "frames": info.frames,
                "size_bytes": file_path.stat().st_size,
                "last_modified": file_path.stat().st_mtime
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting audio file info: {str(e)}")
            raise AudioFileError(f"Could not get audio file info: {str(e)}")
    
    def generate_waveform(self, audio: np.ndarray, n_points: int = 1000) -> np.ndarray:
        """
        Generate waveform data for visualization.
        
        Args:
            audio: Audio data
            n_points: Number of points in the waveform
            
        Returns:
            Numpy array with waveform data
        """
        return WaveformGenerator.generate_waveform(audio, n_points)
    
    def generate_waveform_image(self, audio: np.ndarray, width: int = 800, 
                               height: int = 200) -> np.ndarray:
        """
        Generate a waveform image for visualization.
        
        Args:
            audio: Audio data
            width: Width of the image in pixels
            height: Height of the image in pixels
            
        Returns:
            Numpy array with image data
        """
        return WaveformGenerator.generate_waveform_image(audio, width, height)
    
    def extract_metadata(self, file_path: Union[str, Path], audio: Optional[np.ndarray] = None, 
                         sr: Optional[int] = None) -> Dict[str, Any]:
        """
        Extract metadata from an audio file.
        
        Args:
            file_path: Path to the audio file
            audio: Optional pre-loaded audio data
            sr: Optional sample rate of pre-loaded audio
            
        Returns:
            Dictionary containing metadata
        """
        return AudioMetadataExtractor.extract_metadata(file_path, audio, sr)


# Convenience functions

def load_audio(file_path: Union[str, Path], sr: Optional[int] = 16000, 
               mono: bool = True, extract_metadata: bool = True) -> Dict[str, Any]:
    """
    Load an audio file with optional metadata extraction.
    
    Args:
        file_path: Path to the audio file
        sr: Target sample rate (None for original)
        mono: Whether to convert to mono
        extract_metadata: Whether to extract metadata
        
    Returns:
        Dictionary containing audio data, sample rate, and metadata
    """
    audio_io = AudioFileIO(default_sr=sr, default_mono=mono)
    return audio_io.load_file(file_path, sr, mono, extract_metadata)


def save_audio(audio: np.ndarray, file_path: Union[str, Path], sr: int, 
               format: Optional[str] = None, **kwargs) -> str:
    """
    Save audio data to a file.
    
    Args:
        audio: Audio data
        file_path: Path to save the file
        sr: Sample rate
        format: Output format (if None, inferred from file extension)
        **kwargs: Additional format-specific parameters
        
    Returns:
        Path to the saved file
    """
    audio_io = AudioFileIO()
    return audio_io.save_file(audio, file_path, sr, format, **kwargs)


def get_audio_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about an audio file without loading the audio data.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Dictionary containing file information
    """
    audio_io = AudioFileIO()
    return audio_io.get_file_info(file_path)


def extract_audio_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract metadata from an audio file.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Dictionary containing metadata
    """
    return AudioMetadataExtractor.extract_metadata(file_path)


def generate_waveform(audio: np.ndarray, n_points: int = 1000) -> np.ndarray:
    """
    Generate waveform data for visualization.
    
    Args:
        audio: Audio data
        n_points: Number of points in the waveform
        
    Returns:
        Numpy array with waveform data
    """
    return WaveformGenerator.generate_waveform(audio, n_points)