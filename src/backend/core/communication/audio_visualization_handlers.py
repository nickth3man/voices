"""
Audio visualization handlers for the communication server.

This module provides handlers for audio visualization operations
that can be called from the frontend via the Python Bridge.
"""

import os
import logging
import json
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

from ...processing.audio.io import load_audio, save_audio
from ...processing.models.abstraction import create_separation_manager
from ...storage.database.db_manager import DatabaseManager
from ...storage.files.file_manager import FileManager
from ...storage.metadata.metadata_manager import MetadataManager

# Configure logging
logger = logging.getLogger(__name__)


class AudioVisualizationHandlers:
    """
    Handlers for audio visualization operations.
    
    This class provides methods that can be called from the frontend
    via the Python Bridge to interact with the audio visualization system.
    """
    
    def __init__(self):
        """Initialize the audio visualization handlers."""
        # Initialize database and file managers
        self.db_manager = DatabaseManager()
        if not self.db_manager.initialize():
            logger.error("Failed to initialize database")
            raise RuntimeError("Failed to initialize database")
        
        self.file_manager = FileManager(self.db_manager)
        self.metadata_manager = MetadataManager(self.db_manager)
        
        # Initialize separation manager for speaker separation
        registry_dir = os.path.join(os.path.dirname(__file__), "../../../data/model_registry")
        os.makedirs(registry_dir, exist_ok=True)
        
        self.separation_manager = create_separation_manager(registry_dir)
    
    def get_speaker_segments(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get speaker segments for an audio file.
        
        Args:
            params: Dictionary with parameters
                - audioPath: Path to the audio file
                
        Returns:
            Dictionary with response
                - success: Whether the operation was successful
                - data: Speaker segments if successful
                - error: Error message if unsuccessful
        """
        try:
            audio_path = params.get('audioPath')
            
            if not audio_path:
                return {'success': False, 'error': 'Audio path is required'}
            
            # Load audio file
            try:
                audio_data, sample_rate = load_audio(audio_path)
            except Exception as e:
                return {'success': False, 'error': f'Failed to load audio file: {str(e)}'}
            
            # Get audio duration in seconds
            duration = len(audio_data) / sample_rate
            
            # For now, we'll create simulated speaker segments
            # In a real implementation, this would use a speaker diarization model
            segments, speakers = self._simulate_speaker_segments(duration)
            
            return {
                'success': True,
                'data': {
                    'segments': segments,
                    'speakers': speakers,
                    'duration': duration,
                    'sampleRate': sample_rate
                }
            }
        
        except Exception as e:
            logger.error(f"Error in get_speaker_segments: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def separate_speakers(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Separate speakers in an audio file.
        
        Args:
            params: Dictionary with parameters
                - audioPath: Path to the audio file
                - numSpeakers: Optional number of speakers
                
        Returns:
            Dictionary with response
                - success: Whether the operation was successful
                - data: Separated tracks if successful
                - error: Error message if unsuccessful
        """
        try:
            audio_path = params.get('audioPath')
            num_speakers = params.get('numSpeakers')
            
            if not audio_path:
                return {'success': False, 'error': 'Audio path is required'}
            
            # Load audio file
            try:
                audio_data, sample_rate = load_audio(audio_path)
            except Exception as e:
                return {'success': False, 'error': f'Failed to load audio file: {str(e)}'}
            
            # Create temporary directory for outputs
            temp_dir = tempfile.mkdtemp()
            
            # Get speaker segments
            _, speakers = self._simulate_speaker_segments(len(audio_data) / sample_rate)
            
            # Use the best available model for separation
            try:
                # Get available models
                models = self.separation_manager.get_available_models()
                
                if not models:
                    return {'success': False, 'error': 'No voice separation models available'}
                
                # Use the first available model
                model_id = models[0]['id']
                
                # Separate audio
                separated_sources = self.separation_manager.separate(
                    audio_data,
                    num_speakers=num_speakers or len(speakers),
                    model_id=model_id
                )
                
                # Save separated sources
                tracks = []
                for i, source in enumerate(separated_sources):
                    source_path = os.path.join(temp_dir, f"speaker_{i+1}.wav")
                    save_audio(source_path, source, sample_rate)
                    
                    # Create track info
                    speaker_id = speakers[i]['id'] if i < len(speakers) else f"speaker_{i+1}"
                    tracks.append({
                        'path': source_path,
                        'speaker': speaker_id,
                        'index': i
                    })
                
                return {
                    'success': True,
                    'data': {
                        'tracks': tracks,
                        'tempDir': temp_dir,
                        'originalPath': audio_path,
                        'sampleRate': sample_rate
                    }
                }
            
            except Exception as e:
                logger.error(f"Error separating speakers: {str(e)}")
                return {'success': False, 'error': f'Failed to separate speakers: {str(e)}'}
        
        except Exception as e:
            logger.error(f"Error in separate_speakers: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_audio_metadata(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get metadata for an audio file.
        
        Args:
            params: Dictionary with parameters
                - audioPath: Path to the audio file
                
        Returns:
            Dictionary with response
                - success: Whether the operation was successful
                - data: Audio metadata if successful
                - error: Error message if unsuccessful
        """
        try:
            audio_path = params.get('audioPath')
            
            if not audio_path:
                return {'success': False, 'error': 'Audio path is required'}
            
            # Extract metadata
            try:
                metadata = self.metadata_manager.extract_metadata(audio_path)
                return {'success': True, 'data': metadata}
            except Exception as e:
                return {'success': False, 'error': f'Failed to extract metadata: {str(e)}'}
        
        except Exception as e:
            logger.error(f"Error in get_audio_metadata: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _simulate_speaker_segments(self, duration: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Simulate speaker segments for testing purposes.
        
        In a real implementation, this would use a speaker diarization model.
        
        Args:
            duration: Duration of the audio in seconds
            
        Returns:
            Tuple of (segments, speakers)
                - segments: List of speaker segments
                - speakers: List of speaker information
        """
        # Create 2-4 speakers
        num_speakers = np.random.randint(2, 5)
        speakers = []
        
        for i in range(num_speakers):
            speakers.append({
                'id': f"speaker_{i+1}",
                'name': f"Speaker {i+1}"
            })
        
        # Create segments
        segments = []
        current_time = 0
        
        while current_time < duration:
            # Random speaker
            speaker_idx = np.random.randint(0, num_speakers)
            speaker_id = speakers[speaker_idx]['id']
            
            # Random segment length (1-10 seconds)
            segment_length = np.random.uniform(1, 10)
            
            # Ensure we don't exceed the duration
            end_time = min(current_time + segment_length, duration)
            
            segments.append({
                'start': current_time,
                'end': end_time,
                'speaker': speaker_id
            })
            
            # Move to next segment
            current_time = end_time
        
        return segments, speakers


# Create a singleton instance
audio_visualization_handlers = AudioVisualizationHandlers()

# Define handler functions that can be registered with the server

def handle_get_speaker_segments(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for get_speaker_segments."""
    return audio_visualization_handlers.get_speaker_segments(params)

def handle_separate_speakers(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for separate_speakers."""
    return audio_visualization_handlers.separate_speakers(params)

def handle_get_audio_metadata(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for get_audio_metadata."""
    return audio_visualization_handlers.get_audio_metadata(params)


# Dictionary of handlers to register with the server
AUDIO_VISUALIZATION_HANDLERS = {
    'get_speaker_segments': handle_get_speaker_segments,
    'separate_speakers': handle_separate_speakers,
    'get_audio_metadata': handle_get_audio_metadata
}