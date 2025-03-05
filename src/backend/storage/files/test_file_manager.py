"""
Test module for the file organization system.

This module provides tests for the FileManager class to ensure proper
functionality of storage location management, naming conventions,
automatic file organization, and storage usage monitoring.
"""

import os
import shutil
import unittest
import tempfile
import json
from unittest.mock import MagicMock, patch

from ..database.db_manager import DatabaseManager
from ..database.models import ProcessedFile
from .file_manager import FileManager


class TestFileManager(unittest.TestCase):
    """Test case for the FileManager class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock database manager
        self.db_manager = MagicMock(spec=DatabaseManager)
        
        # Create test file manager
        self.file_manager = FileManager(
            self.db_manager,
            base_dir=self.temp_dir
        )
        
        # Create test files
        self.test_files = []
        for i in range(3):
            fd, path = tempfile.mkstemp(suffix=f"_test{i}.wav")
            os.close(fd)
            self.test_files.append(path)
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove test files
        for path in self.test_files:
            if os.path.exists(path):
                os.remove(path)
        
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_initialize_storage_dirs(self):
        """Test initialization of storage directories."""
        # Check that all default directories are created
        for dir_type, dir_name in FileManager.DEFAULT_DIRS.items():
            dir_path = os.path.join(self.temp_dir, dir_name)
            self.assertTrue(os.path.isdir(dir_path))
            self.assertEqual(self.file_manager.storage_dirs[dir_type], dir_path)
    
    def test_initialize_storage_dirs_with_config(self):
        """Test initialization of storage directories with custom configuration."""
        # Create custom configuration
        config = {
            "directories": {
                "original": "custom_original",
                "processed": "custom_processed"
            }
        }
        
        # Create file manager with custom configuration
        custom_file_manager = FileManager(
            self.db_manager,
            base_dir=self.temp_dir,
            config=config
        )
        
        # Check custom directories
        self.assertEqual(
            custom_file_manager.storage_dirs["original"],
            os.path.join(self.temp_dir, "custom_original")
        )
        self.assertEqual(
            custom_file_manager.storage_dirs["processed"],
            os.path.join(self.temp_dir, "custom_processed")
        )
        
        # Check default directories for non-customized types
        self.assertEqual(
            custom_file_manager.storage_dirs["temp"],
            os.path.join(self.temp_dir, "temp")
        )
    
    def test_get_storage_path(self):
        """Test getting storage path for a directory type."""
        # Check valid directory type
        path = self.file_manager.get_storage_path("original")
        self.assertEqual(path, os.path.join(self.temp_dir, "original"))
        
        # Check invalid directory type
        with self.assertRaises(ValueError):
            self.file_manager.get_storage_path("invalid")
    
    def test_generate_filename(self):
        """Test filename generation based on naming conventions."""
        # Test with all parameters
        filename = self.file_manager.generate_filename(
            "test file.wav",
            "John Doe",
            "processed"
        )
        
        # Check format: speaker_basename_processing_timestamp.ext
        parts = os.path.splitext(filename)[0].split("_")
        self.assertEqual(parts[0], "john")
        self.assertEqual(parts[1], "doe")
        self.assertEqual(parts[2], "test")
        self.assertEqual(parts[3], "file")
        self.assertEqual(parts[4], "processed")
        self.assertTrue(len(parts) >= 6)  # At least one timestamp part
        self.assertTrue(filename.endswith(".wav"))
        
        # Test without speaker
        filename = self.file_manager.generate_filename(
            "test file.wav",
            processing_type="processed"
        )
        parts = os.path.splitext(filename)[0].split("_")
        self.assertEqual(parts[0], "test")
        self.assertEqual(parts[1], "file")
        self.assertEqual(parts[2], "processed")
        
        # Test without processing type
        filename = self.file_manager.generate_filename(
            "test file.wav",
            speaker_name="John Doe"
        )
        parts = os.path.splitext(filename)[0].split("_")
        self.assertEqual(parts[0], "john")
        self.assertEqual(parts[1], "doe")
        self.assertEqual(parts[2], "test")
        self.assertEqual(parts[3], "file")
        
        # Test without timestamp
        filename = self.file_manager.generate_filename(
            "test file.wav",
            speaker_name="John Doe",
            processing_type="processed",
            timestamp=False
        )
        parts = os.path.splitext(filename)[0].split("_")
        self.assertEqual(parts[0], "john")
        self.assertEqual(parts[1], "doe")
        self.assertEqual(parts[2], "test")
        self.assertEqual(parts[3], "file")
        self.assertEqual(parts[4], "processed")
        self.assertEqual(len(parts), 5)  # No timestamp parts
    
    def test_store_file(self):
        """Test storing a file."""
        # Store test file
        dest_path = self.file_manager.store_file(
            self.test_files[0],
            "original",
            speaker_name="Test Speaker",
            processing_type="raw"
        )
        
        # Check that file was stored
        self.assertTrue(os.path.isfile(dest_path))
        self.assertTrue(dest_path.startswith(self.file_manager.get_storage_path("original")))
        
        # Check with invalid source path
        invalid_path = self.file_manager.store_file(
            "nonexistent_file.wav",
            "original"
        )
        self.assertIsNone(invalid_path)
    
    def test_move_file(self):
        """Test moving a file."""
        # Move test file
        dest_path = self.file_manager.move_file(
            self.test_files[0],
            "processed",
            speaker_name="Test Speaker",
            processing_type="processed"
        )
        
        # Check that file was moved
        self.assertTrue(os.path.isfile(dest_path))
        self.assertFalse(os.path.isfile(self.test_files[0]))
        self.assertTrue(dest_path.startswith(self.file_manager.get_storage_path("processed")))
        
        # Check with invalid source path
        invalid_path = self.file_manager.move_file(
            "nonexistent_file.wav",
            "processed"
        )
        self.assertIsNone(invalid_path)
    
    def test_delete_file(self):
        """Test deleting a file."""
        # Delete test file
        success = self.file_manager.delete_file(self.test_files[1])
        
        # Check that file was deleted
        self.assertTrue(success)
        self.assertFalse(os.path.isfile(self.test_files[1]))
        
        # Check with invalid path
        invalid_success = self.file_manager.delete_file("nonexistent_file.wav")
        self.assertFalse(invalid_success)
    
    def test_archive_file(self):
        """Test archiving a file."""
        # Archive test file
        dest_path = self.file_manager.archive_file(
            self.test_files[2],
            speaker_name="Test Speaker"
        )
        
        # Check that file was archived
        self.assertTrue(os.path.isfile(dest_path))
        self.assertFalse(os.path.isfile(self.test_files[2]))
        self.assertTrue(dest_path.startswith(self.file_manager.get_storage_path("archive")))
        self.assertTrue("archived" in os.path.basename(dest_path))
    
    @patch('os.path.getsize')
    def test_get_storage_usage(self, mock_getsize):
        """Test getting storage usage information."""
        # Mock file sizes
        mock_getsize.return_value = 1024 * 1024  # 1 MB
        
        # Create test files in storage directories
        for dir_type in self.file_manager.storage_dirs:
            dir_path = self.file_manager.get_storage_path(dir_type)
            for i in range(2):
                with open(os.path.join(dir_path, f"test{i}.wav"), "w") as f:
                    f.write("test")
        
        # Get storage usage
        usage_info = self.file_manager.get_storage_usage()
        
        # Check usage information
        self.assertEqual(usage_info["file_count"], 10)  # 2 files in each of 5 directories
        self.assertEqual(usage_info["total_size"], 10 * 1024 * 1024)  # 10 MB
        self.assertTrue("disk_total" in usage_info)
        self.assertTrue("disk_used" in usage_info)
        self.assertTrue("disk_free" in usage_info)
        
        # Check specific directory
        usage_info = self.file_manager.get_storage_usage("original")
        self.assertEqual(usage_info["file_count"], 2)
        self.assertEqual(usage_info["total_size"], 2 * 1024 * 1024)  # 2 MB
    
    def test_find_files(self):
        """Test finding files matching a pattern."""
        # Create test files in original directory
        original_dir = self.file_manager.get_storage_path("original")
        for ext in ["wav", "mp3", "flac"]:
            with open(os.path.join(original_dir, f"test.{ext}"), "w") as f:
                f.write("test")
        
        # Find files
        wav_files = self.file_manager.find_files("*.wav", "original")
        self.assertEqual(len(wav_files), 1)
        self.assertTrue(wav_files[0].endswith("test.wav"))
        
        # Find all audio files
        audio_files = self.file_manager.find_files("*.{wav,mp3,flac}", "original")
        self.assertEqual(len(audio_files), 3)
    
    @patch('os.path.getsize')
    def test_register_file(self, mock_getsize):
        """Test registering a file in the database."""
        # Mock file size
        mock_getsize.return_value = 1024 * 1024  # 1 MB
        
        # Create test file
        test_file = os.path.join(self.temp_dir, "test_register.wav")
        with open(test_file, "w") as f:
            f.write("test")
        
        # Mock database operations
        processed_file = ProcessedFile(id=1)
        self.db_manager.add.return_value = True
        
        # Register file
        with patch('builtins.open', unittest.mock.mock_open(read_data='test')):
            file_id = self.file_manager.register_file(
                test_file,
                "original_path.wav",
                speaker_id=1,
                metadata={"key": "value"}
            )
        
        # Check registration
        self.assertIsNotNone(file_id)
        self.db_manager.add.assert_called_once()
        
        # Check with invalid file
        invalid_id = self.file_manager.register_file("nonexistent_file.wav")
        self.assertIsNone(invalid_id)
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        # Save configuration
        config_path = os.path.join(self.temp_dir, "test_config.json")
        success = self.file_manager.save_config(config_path)
        
        # Check that configuration was saved
        self.assertTrue(success)
        self.assertTrue(os.path.isfile(config_path))
        
        # Load configuration
        config = FileManager.load_config(config_path)
        
        # Check configuration
        self.assertEqual(config["base_dir"], self.temp_dir)
        self.assertEqual(
            config["directories"]["original"],
            os.path.basename(self.file_manager.storage_dirs["original"])
        )


if __name__ == "__main__":
    unittest.main()