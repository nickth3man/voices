"""
Example usage of the file organization system.

This example demonstrates how to use the FileManager class to manage
storage locations, implement naming conventions, handle automatic file
organization, and monitor storage usage.
"""

import os
import sys
import logging
import tempfile
import json

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from src.backend.storage.database.db_manager import DatabaseManager
from src.backend.storage.files.file_manager import FileManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def basic_usage_example():
    """Demonstrate basic usage of the FileManager."""
    print("\n=== Basic Usage Example ===\n")
    
    # Initialize database manager
    db_manager = DatabaseManager()
    if not db_manager.initialize():
        logger.error("Failed to initialize database")
        return
    
    # Create a temporary directory for the example
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory: {temp_dir}")
        
        # Initialize file manager
        file_manager = FileManager(db_manager, base_dir=temp_dir)
        print(f"Initialized file manager with base directory: {temp_dir}")
        
        # Create a temporary file for the example
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(b"This is a test audio file")
            temp_file_path = temp_file.name
        
        print(f"Created temporary file: {temp_file_path}")
        
        # Store the file
        dest_path = file_manager.store_file(
            temp_file_path,
            "original",
            speaker_name="John Doe",
            processing_type="raw"
        )
        
        print(f"Stored file: {temp_file_path} -> {dest_path}")
        
        # Get storage usage information
        usage_info = file_manager.get_storage_usage()
        print("\nStorage Usage Information:")
        print(f"Total Size: {usage_info['total_size_formatted']}")
        print(f"Total Files: {usage_info['file_count']}")
        print(f"Disk Space: {usage_info['disk_used_formatted']} / {usage_info['disk_total_formatted']} ({usage_info['disk_percent']}% used)")
        print(f"Free Space: {usage_info['disk_free_formatted']}")
        
        # Clean up
        os.remove(temp_file_path)
        print(f"\nCleaned up temporary file: {temp_file_path}")


def file_organization_example():
    """Demonstrate file organization capabilities."""
    print("\n=== File Organization Example ===\n")
    
    # Initialize database manager
    db_manager = DatabaseManager()
    if not db_manager.initialize():
        logger.error("Failed to initialize database")
        return
    
    # Create a temporary directory for the example
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory: {temp_dir}")
        
        # Initialize file manager
        file_manager = FileManager(db_manager, base_dir=temp_dir)
        
        # Create temporary files for the example
        temp_files = []
        for i in range(5):
            with tempfile.NamedTemporaryFile(suffix=f"_test{i}.wav", delete=False) as temp_file:
                temp_file.write(f"This is test audio file {i}".encode())
                temp_files.append(temp_file.name)
        
        print(f"Created {len(temp_files)} temporary files")
        
        # Store files in different directories
        stored_files = []
        
        # Store in original directory
        dest_path = file_manager.store_file(
            temp_files[0],
            "original",
            speaker_name="John Doe"
        )
        stored_files.append(dest_path)
        print(f"Stored file in original directory: {dest_path}")
        
        # Store in processed directory
        dest_path = file_manager.store_file(
            temp_files[1],
            "processed",
            speaker_name="John Doe",
            processing_type="processed"
        )
        stored_files.append(dest_path)
        print(f"Stored file in processed directory: {dest_path}")
        
        # Store in temp directory
        dest_path = file_manager.store_file(
            temp_files[2],
            "temp"
        )
        stored_files.append(dest_path)
        print(f"Stored file in temp directory: {dest_path}")
        
        # Archive a file
        dest_path = file_manager.archive_file(
            temp_files[3],
            speaker_name="John Doe"
        )
        stored_files.append(dest_path)
        print(f"Archived file: {dest_path}")
        
        # Export a file
        dest_path = file_manager.store_file(
            temp_files[4],
            "exports",
            new_filename="exported_file.wav"
        )
        stored_files.append(dest_path)
        print(f"Exported file: {dest_path}")
        
        # Organize files
        print("\nOrganizing files...")
        file_manager.organize_files()
        
        # Find organized files
        print("\nFinding organized files...")
        for dir_type in file_manager.storage_dirs:
            files = file_manager.find_files("*.wav", dir_type)
            print(f"Found {len(files)} files in {dir_type} directory:")
            for file_path in files:
                print(f"  {file_path}")
        
        # Clean up
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        print(f"\nCleaned up {len(temp_files)} temporary files")


def database_integration_example():
    """Demonstrate integration with the database."""
    print("\n=== Database Integration Example ===\n")
    
    # Initialize database manager
    db_manager = DatabaseManager()
    if not db_manager.initialize():
        logger.error("Failed to initialize database")
        return
    
    # Create a temporary directory for the example
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory: {temp_dir}")
        
        # Initialize file manager
        file_manager = FileManager(db_manager, base_dir=temp_dir)
        
        # Create a temporary file for the example
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(b"This is a test audio file")
            temp_file_path = temp_file.name
        
        print(f"Created temporary file: {temp_file_path}")
        
        # Store the file
        dest_path = file_manager.store_file(
            temp_file_path,
            "original",
            speaker_name="John Doe",
            processing_type="raw"
        )
        
        print(f"Stored file: {temp_file_path} -> {dest_path}")
        
        # Register the file in the database
        metadata = {
            "duration": 60.0,
            "sample_rate": 44100,
            "channels": 2,
            "bit_depth": 16,
            "recording_date": "2025-03-04",
            "notes": "Example recording for demonstration"
        }
        
        file_id = file_manager.register_file(
            dest_path,
            temp_file_path,
            speaker_id=None,  # No speaker ID for this example
            metadata=metadata
        )
        
        if file_id:
            print(f"Registered file with ID: {file_id}")
            
            # Get file information
            file_info = file_manager.get_file_info(file_id)
            
            if file_info:
                print("\nFile Information:")
                print(f"  ID: {file_info['id']}")
                print(f"  Filename: {file_info['filename']}")
                print(f"  Format: {file_info['file_format']}")
                print(f"  Size: {file_info['file_size_formatted']}")
                print(f"  Original Path: {file_info['original_path']}")
                print(f"  Processed Path: {file_info['processed_path']}")
                print(f"  Created At: {file_info['created_at']}")
                
                if file_info['metadata']:
                    print("\n  Metadata:")
                    for key, value in file_info['metadata'].items():
                        print(f"    {key}: {value}")
        
        # Clean up
        os.remove(temp_file_path)
        print(f"\nCleaned up temporary file: {temp_file_path}")


def configuration_example():
    """Demonstrate configuration capabilities."""
    print("\n=== Configuration Example ===\n")
    
    # Initialize database manager
    db_manager = DatabaseManager()
    if not db_manager.initialize():
        logger.error("Failed to initialize database")
        return
    
    # Create a temporary directory for the example
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory: {temp_dir}")
        
        # Create custom configuration
        config = {
            "directories": {
                "original": "custom_original",
                "processed": "custom_processed",
                "temp": "custom_temp",
                "archive": "custom_archive",
                "exports": "custom_exports"
            },
            "organization": {
                "original": {
                    "by_date": True,
                    "by_speaker": False
                },
                "processed": {
                    "by_date": True,
                    "by_speaker": True
                }
            }
        }
        
        # Initialize file manager with custom configuration
        file_manager = FileManager(db_manager, base_dir=temp_dir, config=config)
        
        print("Initialized file manager with custom configuration:")
        print(json.dumps(config, indent=2))
        
        # Save configuration to file
        config_path = os.path.join(temp_dir, "file_manager_config.json")
        file_manager.save_config(config_path)
        
        print(f"\nSaved configuration to: {config_path}")
        
        # Load configuration from file
        loaded_config = FileManager.load_config(config_path)
        
        print("\nLoaded configuration:")
        print(json.dumps(loaded_config, indent=2))
        
        # Check storage directories
        print("\nStorage Directories:")
        for dir_type, dir_path in file_manager.storage_dirs.items():
            print(f"  {dir_type}: {dir_path}")


def main():
    """Run all examples."""
    print("File Organization System Examples")
    print("=================================")
    
    basic_usage_example()
    file_organization_example()
    database_integration_example()
    configuration_example()
    
    print("\nAll examples completed successfully!")


if __name__ == "__main__":
    main()