"""
Tests for the audio file I/O and management module.

This module contains tests for the audio file I/O functionality, including
loading, format detection, metadata extraction, waveform generation, and saving.
"""

import os
import unittest
import tempfile
import numpy as np
from pathlib import Path

from .io import (
    AudioFileIO,
    AudioFormatDetector,
    AudioMetadataExtractor,
    WaveformGenerator,
    load_audio,
    save_audio,
    get_audio_info,
    extract_audio_metadata,
    generate_waveform,
    AudioFileError,
    UnsupportedFormatError
)


class TestAudioFormatDetector(unittest.TestCase):
    """Tests for the AudioFormatDetector class."""
    
    def test_get_supported_formats(self):
        """Test getting supported formats."""
        formats = AudioFormatDetector.get_supported_formats()
        self.assertIsInstance(formats, list)
        self.assertIn('wav', formats)
        self.assertIn('mp3', formats)
    
    def test_get_supported_extensions(self):
        """Test getting supported extensions."""
        extensions = AudioFormatDetector.get_supported_extensions()
        self.assertIsInstance(extensions, list)
        self.assertIn('wav', extensions)
        self.assertIn('mp3', extensions)
    
    def test_detect_format(self):
        """Test detecting format from file extension."""
        self.assertEqual(AudioFormatDetector.detect_format('test.wav'), 'wav')
        self.assertEqual(AudioFormatDetector.detect_format('test.mp3'), 'mp3')
        self.assertEqual(AudioFormatDetector.detect_format('test.flac'), 'flac')
        
        with self.assertRaises(UnsupportedFormatError):
            AudioFormatDetector.detect_format('test.xyz')
    
    def test_validate_format(self):
        """Test validating file format."""
        self.assertTrue(AudioFormatDetector.validate_format('test.wav'))
        self.assertTrue(AudioFormatDetector.validate_format('test.mp3'))
        self.assertFalse(AudioFormatDetector.validate_format('test.xyz'))
    
    def test_get_extension_for_format(self):
        """Test getting extension for format."""
        self.assertEqual(AudioFormatDetector.get_extension_for_format('wav'), 'wav')
        self.assertEqual(AudioFormatDetector.get_extension_for_format('mp3'), 'mp3')
        
        with self.assertRaises(UnsupportedFormatError):
            AudioFormatDetector.get_extension_for_format('xyz')


class TestWaveformGenerator(unittest.TestCase):
    """Tests for the WaveformGenerator class."""
    
    def test_generate_waveform(self):
        """Test generating waveform data."""
        # Create a simple sine wave
        sample_rate = 16000
        duration = 1.0  # seconds
        frequency = 440.0  # Hz
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = np.sin(2 * np.pi * frequency * t)
        
        # Generate waveform
        n_points = 100
        waveform = WaveformGenerator.generate_waveform(audio, n_points)
        
        # Check result
        self.assertEqual(len(waveform), n_points)
        self.assertIsInstance(waveform, np.ndarray)
    
    def test_generate_waveform_image(self):
        """Test generating waveform image."""
        # Create a simple sine wave
        sample_rate = 16000
        duration = 1.0  # seconds
        frequency = 440.0  # Hz
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = np.sin(2 * np.pi * frequency * t)
        
        # Generate waveform image
        width = 200
        height = 100
        image = WaveformGenerator.generate_waveform_image(audio, width, height)
        
        # Check result
        self.assertEqual(image.shape, (height, width))
        self.assertIsInstance(image, np.ndarray)


class TestAudioFileIO(unittest.TestCase):
    """Tests for the AudioFileIO class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create a simple sine wave
        self.sample_rate = 16000
        self.duration = 1.0  # seconds
        self.frequency = 440.0  # Hz
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        self.audio = np.sin(2 * np.pi * self.frequency * t)
        
        # Save test audio file
        self.test_wav_path = self.temp_path / 'test.wav'
        import soundfile as sf
        sf.write(self.test_wav_path, self.audio, self.sample_rate)
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    def test_load_file(self):
        """Test loading an audio file."""
        audio_io = AudioFileIO()
        result = audio_io.load_file(self.test_wav_path)
        
        # Check result
        self.assertIn('audio', result)
        self.assertIn('sample_rate', result)
        self.assertIn('metadata', result)
        
        self.assertEqual(result['sample_rate'], self.sample_rate)
        self.assertEqual(len(result['audio']), int(self.sample_rate * self.duration))
    
    def test_save_file(self):
        """Test saving an audio file."""
        audio_io = AudioFileIO()
        
        # Save in different formats
        formats = ['wav', 'flac']
        for format in formats:
            output_path = self.temp_path / f'output.{format}'
            saved_path = audio_io.save_file(self.audio, output_path, self.sample_rate, format)
            
            # Check that file exists
            self.assertTrue(os.path.exists(saved_path))
            
            # Check that file can be loaded
            result = audio_io.load_file(saved_path)
            self.assertIn('audio', result)
            self.assertEqual(result['sample_rate'], self.sample_rate)
    
    def test_get_file_info(self):
        """Test getting file information."""
        audio_io = AudioFileIO()
        info = audio_io.get_file_info(self.test_wav_path)
        
        # Check result
        self.assertIn('path', info)
        self.assertIn('format', info)
        self.assertIn('sample_rate', info)
        self.assertIn('channels', info)
        self.assertIn('duration', info)
        
        self.assertEqual(info['sample_rate'], self.sample_rate)
        self.assertAlmostEqual(info['duration'], self.duration, delta=0.1)
    
    def test_extract_metadata(self):
        """Test extracting metadata."""
        audio_io = AudioFileIO()
        metadata = audio_io.extract_metadata(self.test_wav_path)
        
        # Check result
        self.assertIn('filename', metadata)
        self.assertIn('path', metadata)
        self.assertIn('format', metadata)
        self.assertIn('sample_rate', metadata)
        self.assertIn('duration', metadata)
        
        self.assertEqual(metadata['sample_rate'], self.sample_rate)
        self.assertAlmostEqual(metadata['duration'], self.duration, delta=0.1)
    
    def test_generate_waveform(self):
        """Test generating waveform."""
        audio_io = AudioFileIO()
        n_points = 100
        waveform = audio_io.generate_waveform(self.audio, n_points)
        
        # Check result
        self.assertEqual(len(waveform), n_points)


class TestConvenienceFunctions(unittest.TestCase):
    """Tests for the convenience functions."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create a simple sine wave
        self.sample_rate = 16000
        self.duration = 1.0  # seconds
        self.frequency = 440.0  # Hz
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        self.audio = np.sin(2 * np.pi * self.frequency * t)
        
        # Save test audio file
        self.test_wav_path = self.temp_path / 'test.wav'
        import soundfile as sf
        sf.write(self.test_wav_path, self.audio, self.sample_rate)
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    def test_load_audio(self):
        """Test load_audio function."""
        result = load_audio(self.test_wav_path)
        
        # Check result
        self.assertIn('audio', result)
        self.assertIn('sample_rate', result)
        self.assertIn('metadata', result)
        
        self.assertEqual(result['sample_rate'], self.sample_rate)
    
    def test_save_audio(self):
        """Test save_audio function."""
        output_path = self.temp_path / 'output.wav'
        saved_path = save_audio(self.audio, output_path, self.sample_rate)
        
        # Check that file exists
        self.assertTrue(os.path.exists(saved_path))
    
    def test_get_audio_info(self):
        """Test get_audio_info function."""
        info = get_audio_info(self.test_wav_path)
        
        # Check result
        self.assertIn('path', info)
        self.assertIn('format', info)
        self.assertIn('sample_rate', info)
        
        self.assertEqual(info['sample_rate'], self.sample_rate)
    
    def test_extract_audio_metadata(self):
        """Test extract_audio_metadata function."""
        metadata = extract_audio_metadata(self.test_wav_path)
        
        # Check result
        self.assertIn('filename', metadata)
        self.assertIn('path', metadata)
        self.assertIn('format', metadata)
        
        self.assertEqual(metadata['format'], 'wav')
    
    def test_generate_waveform(self):
        """Test generate_waveform function."""
        n_points = 100
        waveform = generate_waveform(self.audio, n_points)
        
        # Check result
        self.assertEqual(len(waveform), n_points)


if __name__ == '__main__':
    unittest.main()