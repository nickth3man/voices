"""
Audio Processing Pipeline Components.

This module provides the base classes and implementations for the audio processing
pipeline components, including audio loading, preprocessing, voice separation,
post-processing, and output formatting.
"""

import os
import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from abc import ABC, abstractmethod
from pathlib import Path

from ..models.abstraction import VoiceSeparationManager, ModelType, AudioCharacteristics

logger = logging.getLogger(__name__)


class PipelineComponent(ABC):
    """Base class for all pipeline components."""
    
    def __init__(self, name: str = None):
        """
        Initialize the pipeline component.
        
        Args:
            name: Name of the component (defaults to class name)
        """
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        
    @abstractmethod
    def process(self, data: Any, **kwargs) -> Any:
        """
        Process the input data.
        
        Args:
            data: Input data to process
            **kwargs: Additional processing parameters
            
        Returns:
            Processed data
        """
        pass
    
    def __call__(self, data: Any, **kwargs) -> Any:
        """
        Call the component as a function.
        
        Args:
            data: Input data to process
            **kwargs: Additional processing parameters
            
        Returns:
            Processed data
        """
        self.logger.debug(f"Processing data with {self.name}")
        return self.process(data, **kwargs)


class AudioLoader(PipelineComponent):
    """Component for loading audio files."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        mono: bool = True,
        normalize: bool = True,
        **kwargs
    ):
        """
        Initialize the audio loader.
        
        Args:
            sample_rate: Target sample rate for loaded audio
            mono: Whether to convert stereo audio to mono
            normalize: Whether to normalize audio after loading
            **kwargs: Additional parameters
        """
        super().__init__(name="AudioLoader")
        self.sample_rate = sample_rate
        self.mono = mono
        self.normalize = normalize
        
    def process(
        self,
        audio_path: Union[str, Path],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Load an audio file.
        
        Args:
            audio_path: Path to the audio file
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing the loaded audio data and metadata
        """
        import soundfile as sf
        
        self.logger.info(f"Loading audio from {audio_path}")
        
        try:
            # Load audio file
            audio, file_sample_rate = sf.read(audio_path)
            
            # Convert to mono if needed
            if self.mono and len(audio.shape) > 1 and audio.shape[1] > 1:
                self.logger.debug(f"Converting stereo audio to mono")
                audio = audio.mean(axis=1)
            
            # Resample if needed
            if file_sample_rate != self.sample_rate:
                self.logger.debug(f"Resampling from {file_sample_rate}Hz to {self.sample_rate}Hz")
                # In a real implementation, we would use librosa.resample or similar
                # For now, we'll just warn about the sample rate mismatch
                self.logger.warning(f"Sample rate mismatch: file={file_sample_rate}Hz, target={self.sample_rate}Hz")
                # Placeholder for resampling
                # audio = librosa.resample(audio, orig_sr=file_sample_rate, target_sr=self.sample_rate)
            
            # Normalize if requested
            if self.normalize:
                self.logger.debug(f"Normalizing audio")
                audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            # Create metadata
            metadata = {
                "original_path": str(audio_path),
                "original_sample_rate": file_sample_rate,
                "target_sample_rate": self.sample_rate,
                "duration": len(audio) / self.sample_rate,
                "channels": 1 if self.mono or len(audio.shape) == 1 else audio.shape[1],
                "normalized": self.normalize
            }
            
            return {
                "audio": audio,
                "sample_rate": self.sample_rate,
                "metadata": metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error loading audio file: {str(e)}")
            raise


class AudioPreprocessor(PipelineComponent):
    """Component for preprocessing audio before separation."""
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        overlap: int = 0,
        apply_vad: bool = False,
        **kwargs
    ):
        """
        Initialize the audio preprocessor.
        
        Args:
            chunk_size: Size of chunks to split audio into (samples)
            overlap: Overlap between chunks (samples)
            apply_vad: Whether to apply voice activity detection
            **kwargs: Additional parameters
        """
        super().__init__(name="AudioPreprocessor")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.apply_vad = apply_vad
        
    def process(
        self,
        data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Preprocess audio data.
        
        Args:
            data: Dictionary containing audio data and metadata
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing preprocessed audio data and metadata
        """
        audio = data["audio"]
        sample_rate = data["sample_rate"]
        metadata = data.get("metadata", {})
        
        self.logger.debug(f"Preprocessing audio: {len(audio)} samples, {sample_rate}Hz")
        
        # Apply voice activity detection if requested
        if self.apply_vad:
            self.logger.debug(f"Applying voice activity detection")
            # In a real implementation, we would use a VAD model
            # For now, we'll just use a simple energy-based approach
            energy = np.abs(audio)
            threshold = 0.05 * np.max(energy)
            voice_mask = energy > threshold
            
            # Add VAD info to metadata
            metadata["vad_applied"] = True
            metadata["vad_threshold"] = threshold
            metadata["voice_percentage"] = np.mean(voice_mask) * 100
            
            self.logger.debug(f"Voice detected in {metadata['voice_percentage']:.1f}% of samples")
        
        # Split into chunks if requested
        if self.chunk_size is not None:
            self.logger.debug(f"Splitting audio into chunks of {self.chunk_size} samples with {self.overlap} overlap")
            
            # Calculate number of chunks
            audio_len = len(audio)
            step = self.chunk_size - self.overlap
            num_chunks = max(1, (audio_len - self.overlap) // step)
            
            # Create chunks
            chunks = []
            for i in range(num_chunks):
                start = i * step
                end = min(start + self.chunk_size, audio_len)
                chunk = audio[start:end]
                
                # Pad if necessary
                if len(chunk) < self.chunk_size:
                    chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)))
                
                chunks.append(chunk)
            
            # Add chunking info to metadata
            metadata["chunked"] = True
            metadata["chunk_size"] = self.chunk_size
            metadata["overlap"] = self.overlap
            metadata["num_chunks"] = num_chunks
            
            self.logger.debug(f"Split audio into {num_chunks} chunks")
            
            # Update audio with chunks
            audio = np.array(chunks)
        
        # Update data dictionary
        data["audio"] = audio
        data["metadata"] = metadata
        
        return data


class VoiceSeparator(PipelineComponent):
    """Component for separating voices using a voice separation model."""
    
    def __init__(
        self,
        model_type: Union[str, ModelType] = "auto",
        model_id: Optional[str] = None,
        version_id: Optional[str] = None,
        num_speakers: Optional[int] = None,
        device: Optional[torch.device] = None,
        registry_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the voice separator.
        
        Args:
            model_type: Type of model to use ("svoice", "demucs", or "auto")
            model_id: Specific model ID to use
            version_id: Specific version ID to use
            num_speakers: Number of speakers to separate
            device: Device to run the model on
            registry_dir: Directory for the model registry
            **kwargs: Additional parameters
        """
        super().__init__(name="VoiceSeparator")
        
        # Convert string model type to enum if needed
        if isinstance(model_type, str):
            try:
                self.model_type = ModelType(model_type.lower())
            except ValueError:
                self.logger.warning(f"Invalid model type: {model_type}, using AUTO")
                self.model_type = ModelType.AUTO
        else:
            self.model_type = model_type
            
        self.model_id = model_id
        self.version_id = version_id
        self.num_speakers = num_speakers
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create separation manager
        from ..models.abstraction import create_separation_manager
        self.manager = create_separation_manager(
            registry_dir=registry_dir,
            default_model_type=self.model_type.value,
            device=self.device
        )
        
        self.logger.info(f"Initialized voice separator with model type {self.model_type.value}")
        
    def process(
        self,
        data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Separate voices in audio data.
        
        Args:
            data: Dictionary containing audio data and metadata
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing separated audio data and metadata
        """
        audio = data["audio"]
        sample_rate = data["sample_rate"]
        metadata = data.get("metadata", {})
        
        # Get number of speakers (from init, kwargs, or default)
        num_speakers = kwargs.get("num_speakers", self.num_speakers)
        
        # Check if audio is chunked
        is_chunked = metadata.get("chunked", False)
        
        self.logger.debug(f"Separating voices in audio: {audio.shape}, chunked={is_chunked}")
        
        # Extract audio characteristics for model selection
        characteristics = AudioCharacteristics(
            num_speakers=num_speakers,
            duration=metadata.get("duration"),
            sample_rate=sample_rate,
            is_noisy=metadata.get("is_noisy"),
            is_reverberant=metadata.get("is_reverberant")
        )
        
        # Process differently based on chunking
        if is_chunked:
            # Process each chunk separately
            separated_chunks = []
            num_chunks = len(audio)
            
            for i, chunk in enumerate(audio):
                self.logger.debug(f"Processing chunk {i+1}/{num_chunks}")
                separated = self.manager.separate(
                    mixture=chunk,
                    num_speakers=num_speakers,
                    model_type=self.model_type,
                    model_id=self.model_id,
                    version_id=self.version_id,
                    sample_rate=sample_rate,
                    characteristics=characteristics,
                    **kwargs
                )
                separated_chunks.append(separated)
            
            # Combine chunks
            # In a real implementation, we would handle overlapping chunks properly
            # For now, we'll just concatenate them
            if num_speakers is None:
                # If num_speakers wasn't specified, use the number from the first chunk
                num_speakers = separated_chunks[0].shape[0]
            
            # Initialize with the right shape
            chunk_size = metadata["chunk_size"]
            step = chunk_size - metadata["overlap"]
            total_length = (num_chunks - 1) * step + chunk_size
            separated = np.zeros((num_speakers, total_length))
            
            # Add each chunk
            for i, chunk in enumerate(separated_chunks):
                start = i * step
                end = start + chunk_size
                separated[:, start:end] += chunk
            
            # Handle overlapping regions
            if metadata["overlap"] > 0:
                # Apply linear crossfade in overlapping regions
                # This is a simplified approach
                for i in range(1, num_chunks):
                    start = i * step
                    overlap_end = start + metadata["overlap"]
                    
                    # Create linear weights for crossfade
                    weights = np.linspace(0, 1, metadata["overlap"])
                    
                    # Apply crossfade
                    for j in range(num_speakers):
                        separated[j, start:overlap_end] *= weights
        else:
            # Process the entire audio at once
            separated = self.manager.separate(
                mixture=audio,
                num_speakers=num_speakers,
                model_type=self.model_type,
                model_id=self.model_id,
                version_id=self.version_id,
                sample_rate=sample_rate,
                characteristics=characteristics,
                **kwargs
            )
        
        # Get model info
        model = self.manager.get_model(
            model_type=self.model_type,
            model_id=self.model_id,
            version_id=self.version_id,
            characteristics=characteristics
        )
        model_info = model.get_model_info()
        
        # Update metadata
        metadata["separation"] = {
            "model_type": model_info.get("type", "unknown"),
            "model_name": model_info.get("name", "unknown"),
            "model_version": model_info.get("version", "unknown"),
            "num_speakers": separated.shape[0],
            "device": str(self.device)
        }
        
        # Update data dictionary
        data["separated_audio"] = separated
        data["metadata"] = metadata
        
        return data


class AudioPostprocessor(PipelineComponent):
    """Component for postprocessing separated audio."""
    
    def __init__(
        self,
        apply_denoising: bool = False,
        apply_normalization: bool = True,
        **kwargs
    ):
        """
        Initialize the audio postprocessor.
        
        Args:
            apply_denoising: Whether to apply denoising
            apply_normalization: Whether to normalize separated sources
            **kwargs: Additional parameters
        """
        super().__init__(name="AudioPostprocessor")
        self.apply_denoising = apply_denoising
        self.apply_normalization = apply_normalization
        
    def process(
        self,
        data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Postprocess separated audio data.
        
        Args:
            data: Dictionary containing separated audio data and metadata
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing postprocessed audio data and metadata
        """
        separated = data["separated_audio"]
        metadata = data.get("metadata", {})
        
        num_speakers = separated.shape[0]
        self.logger.debug(f"Postprocessing {num_speakers} separated sources")
        
        # Apply denoising if requested
        if self.apply_denoising:
            self.logger.debug(f"Applying denoising")
            # In a real implementation, we would use a proper denoising algorithm
            # For now, we'll just apply a simple threshold
            for i in range(num_speakers):
                # Simple noise gate
                threshold = 0.01 * np.max(np.abs(separated[i]))
                mask = np.abs(separated[i]) < threshold
                separated[i][mask] = 0
            
            metadata["postprocessing"] = metadata.get("postprocessing", {})
            metadata["postprocessing"]["denoising_applied"] = True
        
        # Apply normalization if requested
        if self.apply_normalization:
            self.logger.debug(f"Normalizing separated sources")
            for i in range(num_speakers):
                max_val = np.max(np.abs(separated[i]))
                if max_val > 0:
                    separated[i] = separated[i] / max_val
            
            metadata["postprocessing"] = metadata.get("postprocessing", {})
            metadata["postprocessing"]["normalization_applied"] = True
        
        # Update data dictionary
        data["separated_audio"] = separated
        data["metadata"] = metadata
        
        return data


class AudioOutputFormatter(PipelineComponent):
    """Component for formatting and saving separated audio."""
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        output_format: str = "wav",
        naming_pattern: str = "{base_name}_speaker{speaker_idx}.{ext}",
        **kwargs
    ):
        """
        Initialize the audio output formatter.
        
        Args:
            output_dir: Directory to save output files (if None, files are not saved)
            output_format: Output file format (wav, flac, etc.)
            naming_pattern: Pattern for output filenames
            **kwargs: Additional parameters
        """
        super().__init__(name="AudioOutputFormatter")
        self.output_dir = output_dir
        self.output_format = output_format
        self.naming_pattern = naming_pattern
        
    def process(
        self,
        data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Format and optionally save separated audio data.
        
        Args:
            data: Dictionary containing separated audio data and metadata
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing formatted output data
        """
        separated = data["separated_audio"]
        sample_rate = data["sample_rate"]
        metadata = data.get("metadata", {})
        
        num_speakers = separated.shape[0]
        self.logger.debug(f"Formatting {num_speakers} separated sources")
        
        # Create output dictionary
        output = {
            "sources": {},
            "metadata": metadata
        }
        
        # Add each source to the output
        for i in range(num_speakers):
            source_name = f"speaker_{i+1}"
            output["sources"][source_name] = separated[i]
        
        # Save sources if output_dir is provided
        if self.output_dir:
            import soundfile as sf
            
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Get base filename without extension
            original_path = metadata.get("original_path", "unknown")
            base_name = os.path.splitext(os.path.basename(original_path))[0]
            
            # Save each source
            output_paths = {}
            for i, (source_name, source_audio) in enumerate(output["sources"].items()):
                # Format filename
                filename = self.naming_pattern.format(
                    base_name=base_name,
                    speaker_idx=i+1,
                    source_name=source_name,
                    ext=self.output_format
                )
                output_path = os.path.join(self.output_dir, filename)
                
                # Save file
                sf.write(output_path, source_audio, sample_rate)
                self.logger.info(f"Saved separated source to {output_path}")
                
                # Add path to output
                output_paths[source_name] = output_path
            
            # Add output paths to output
            output["output_paths"] = output_paths
        
        return output


class AudioProcessingPipeline:
    """Complete audio processing pipeline combining multiple components."""
    
    def __init__(
        self,
        components: List[PipelineComponent] = None,
        **kwargs
    ):
        """
        Initialize the audio processing pipeline.
        
        Args:
            components: List of pipeline components (if None, creates default pipeline)
            **kwargs: Additional parameters passed to default components
        """
        self.logger = logging.getLogger(__name__)
        
        # Create default pipeline if no components are provided
        if components is None:
            self.logger.info("Creating default audio processing pipeline")
            self.components = [
                AudioLoader(
                    sample_rate=kwargs.get("sample_rate", 16000),
                    mono=kwargs.get("mono", True),
                    normalize=kwargs.get("normalize", True)
                ),
                AudioPreprocessor(
                    chunk_size=kwargs.get("chunk_size", None),
                    overlap=kwargs.get("overlap", 0),
                    apply_vad=kwargs.get("apply_vad", False)
                ),
                VoiceSeparator(
                    model_type=kwargs.get("model_type", "auto"),
                    model_id=kwargs.get("model_id", None),
                    version_id=kwargs.get("version_id", None),
                    num_speakers=kwargs.get("num_speakers", None),
                    device=kwargs.get("device", None),
                    registry_dir=kwargs.get("registry_dir", None)
                ),
                AudioPostprocessor(
                    apply_denoising=kwargs.get("apply_denoising", False),
                    apply_normalization=kwargs.get("apply_normalization", True)
                ),
                AudioOutputFormatter(
                    output_dir=kwargs.get("output_dir", None),
                    output_format=kwargs.get("output_format", "wav"),
                    naming_pattern=kwargs.get("naming_pattern", "{base_name}_speaker{speaker_idx}.{ext}")
                )
            ]
        else:
            self.components = components
        
        self.logger.info(f"Initialized audio processing pipeline with {len(self.components)} components")
    
    def process(
        self,
        input_data: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process data through the pipeline.
        
        Args:
            input_data: Input data for the first component
            **kwargs: Additional parameters passed to all components
            
        Returns:
            Output from the last component
        """
        self.logger.info(f"Processing data through pipeline with {len(self.components)} components")
        
        # Initialize with input data
        data = input_data
        
        # Process through each component
        for i, component in enumerate(self.components):
            self.logger.debug(f"Running component {i+1}/{len(self.components)}: {component.name}")
            data = component(data, **kwargs)
        
        return data
    
    def process_file(
        self,
        audio_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process an audio file through the pipeline.
        
        Args:
            audio_path: Path to the audio file
            **kwargs: Additional parameters
            
        Returns:
            Processed output
        """
        self.logger.info(f"Processing audio file: {audio_path}")
        return self.process(audio_path, **kwargs)