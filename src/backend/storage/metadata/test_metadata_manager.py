#!/usr/bin/env python3
"""
Test script for the metadata manager.

This script tests the functionality of the metadata manager, including
metadata extraction, storage, retrieval, and search.
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from backend.storage.database.db_manager import DatabaseManager
from backend.storage.database.models import ProcessedFile, FileMetadata, CustomMetadata
from backend.storage.files.file_manager import FileManager
from backend.storage.metadata.metadata_manager import MetadataManager


class TestMetadataManager(unittest.TestCase):
    """Test case for the metadata manager."""
    
    def setUp(self):
        """Set up the test case."""
        # Create a temporary database
        self.temp_db_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db_file.close()
        
        # Initialize database manager with temporary database
        self.db_manager = DatabaseManager(db_path=self.temp_db_file.name)
        self.db_manager.initialize(create_tables=True)
        
        # Initialize file manager
        self.file_manager = FileManager(self.db_manager)
        
        # Initialize metadata manager
        self.metadata_manager = MetadataManager(self.db_manager, self.file_manager)
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a test audio file
        self.test_audio_path = os.path.join(self.temp_dir.name, 'test_audio.wav')
        self._create_test_audio_file(self.test_audio_path)
        
        # Register the test file in the database
        self.test_file = ProcessedFile(
            filename='test_audio.wav',
            original_path=self.test_audio_path,
            processed_path=self.test_audio_path,
            file_format='wav'
        )
        self.db_manager.add(self.test_file)
        self.db_manager.commit()
    
    def tearDown(self):
        """Tear down the test case."""
        # Close database connection
        self.db_manager.close()
        
        # Remove temporary files
        os.unlink(self.temp_db_file.name)
        self.temp_dir.cleanup()
    
    def _create_test_audio_file(self, file_path):
        """Create a test audio file."""
        try:
            import numpy as np
            import soundfile as sf
            
            # Create a simple sine wave
            sample_rate = 44100
            duration = 2.0  # seconds
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            data = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
            
            # Save as WAV file
            sf.write(file_path, data, sample_rate)
        except ImportError:
            # If soundfile is not available, create an empty file
            with open(file_path, 'wb') as f:
                f.write(b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00')
    
    def test_extract_metadata(self):
        """Test metadata extraction."""
        # Extract metadata
        metadata = self.metadata_manager.extract_metadata(self.test_audio_path)
        
        # Check basic metadata
        self.assertIn('duration', metadata)
        self.assertIn('sample_rate', metadata)
        self.assertIn('channels', metadata)
        self.assertIn('file_format', metadata)
        
        # Check audio characteristics
        if 'error' not in metadata:  # Skip if librosa is not available
            self.assertIn('rms_mean', metadata)
            self.assertIn('spectral_centroid_mean', metadata)
            self.assertIn('spectral_bandwidth_mean', metadata)
            self.assertIn('zero_crossing_rate_mean', metadata)
    
    def test_store_and_get_metadata(self):
        """Test storing and retrieving metadata."""
        # Extract metadata
        metadata = self.metadata_manager.extract_metadata(self.test_audio_path)
        
        # Store metadata
        result = self.metadata_manager.store_metadata(self.test_file.id, metadata)
        self.assertTrue(result)
        
        # Get metadata
        retrieved_metadata = self.metadata_manager.get_metadata(self.test_file.id)
        
        # Check if metadata was retrieved
        self.assertIsNotNone(retrieved_metadata)
        
        # Check if basic metadata matches
        self.assertEqual(retrieved_metadata['file_id'], self.test_file.id)
        self.assertEqual(retrieved_metadata['filename'], self.test_file.filename)
        
        # Check if audio characteristics were stored
        if 'error' not in metadata:  # Skip if librosa is not available
            self.assertIn('rms_mean', retrieved_metadata)
            self.assertIn('spectral_centroid_mean', retrieved_metadata)
            self.assertIn('spectral_bandwidth_mean', retrieved_metadata)
            self.assertIn('zero_crossing_rate_mean', retrieved_metadata)
    
    def test_custom_metadata(self):
        """Test custom metadata fields."""
        # Add custom metadata
        result = self.metadata_manager.add_custom_metadata(
            self.test_file.id,
            'recording_location',
            'Studio A',
            'text'
        )
        self.assertTrue(result)
        
        # Get metadata
        metadata = self.metadata_manager.get_metadata(self.test_file.id)
        
        # Check if custom metadata was stored
        self.assertIn('custom_recording_location', metadata)
        self.assertEqual(metadata['custom_recording_location'], 'Studio A')
        
        # Remove custom metadata
        result = self.metadata_manager.remove_custom_metadata(
            self.test_file.id,
            'recording_location'
        )
        self.assertTrue(result)
        
        # Get metadata again
        metadata = self.metadata_manager.get_metadata(self.test_file.id)
        
        # Check if custom metadata was removed
        self.assertNotIn('custom_recording_location', metadata)
    
    def test_search_by_metadata(self):
        """Test searching by metadata."""
        # Extract and store metadata
        metadata = self.metadata_manager.extract_metadata(self.test_audio_path)
        self.metadata_manager.store_metadata(self.test_file.id, metadata)
        
        # Add custom metadata
        self.metadata_manager.add_custom_metadata(
            self.test_file.id,
            'recording_location',
            'Studio A',
            'text'
        )
        
        # Search by filename
        results = self.metadata_manager.search_by_metadata({'filename': 'test'})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['file_id'], self.test_file.id)
        
        # Search by file format
        results = self.metadata_manager.search_by_metadata({'file_format': 'wav'})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['file_id'], self.test_file.id)
        
        # Search by duration
        if 'duration' in metadata:
            results = self.metadata_manager.search_by_metadata({
                'duration_min': metadata['duration'] - 1,
                'duration_max': metadata['duration'] + 1
            })
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]['file_id'], self.test_file.id)
    
    def test_export_import_metadata(self):
        """Test exporting and importing metadata."""
        # Extract and store metadata
        metadata = self.metadata_manager.extract_metadata(self.test_audio_path)
        self.metadata_manager.store_metadata(self.test_file.id, metadata)
        
        # Add custom metadata
        self.metadata_manager.add_custom_metadata(
            self.test_file.id,
            'recording_location',
            'Studio A',
            'text'
        )
        
        # Export metadata
        export_path = os.path.join(self.temp_dir.name, 'metadata_export.json')
        result = self.metadata_manager.export_metadata(self.test_file.id, export_path)
        self.assertTrue(result)
        
        # Check if export file exists
        self.assertTrue(os.path.exists(export_path))
        
        # Create a new file
        new_file = ProcessedFile(
            filename='new_audio.wav',
            original_path=self.test_audio_path,
            processed_path=self.test_audio_path,
            file_format='wav'
        )
        self.db_manager.add(new_file)
        self.db_manager.commit()
        
        # Import metadata to new file
        result = self.metadata_manager.import_metadata(export_path, new_file.id)
        self.assertTrue(result)
        
        # Get metadata for new file
        new_metadata = self.metadata_manager.get_metadata(new_file.id)
        
        # Check if metadata was imported
        self.assertIsNotNone(new_metadata)
        self.assertEqual(new_metadata['file_id'], new_file.id)
        self.assertEqual(new_metadata['filename'], new_file.filename)
        
        # Check if custom metadata was imported
        self.assertIn('custom_recording_location', new_metadata)
        self.assertEqual(new_metadata['custom_recording_location'], 'Studio A')
    
    def test_batch_extract_metadata(self):
        """Test batch metadata extraction."""
        # Create additional test files
        for i in range(3):
            file_path = os.path.join(self.temp_dir.name, f'test_audio_{i}.wav')
            self._create_test_audio_file(file_path)
        
        # Batch extract metadata
        results = self.metadata_manager.batch_extract_metadata(self.temp_dir.name)
        
        # Check results
        self.assertIn('total_files', results)
        self.assertIn('successful', results)
        self.assertIn('failed', results)
        self.assertIn('files', results)
        
        # Check if all files were processed
        self.assertEqual(results['total_files'], 4)  # Original test file + 3 new files
        
        # Check if files were registered in the database
        files = self.db_manager.query(ProcessedFile).all()
        self.assertEqual(len(files), 4)  # Original test file + 3 new files
        
        # Check if metadata was stored
        metadata_entries = self.db_manager.query(FileMetadata).all()
        self.assertEqual(len(metadata_entries), 4)  # One for each file


if __name__ == '__main__':
    unittest.main()