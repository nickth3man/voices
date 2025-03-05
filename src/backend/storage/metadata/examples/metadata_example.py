"""
Example script demonstrating the use of the metadata management system.

This script shows how to:
1. Extract metadata from an audio file
2. Store metadata in the database
3. Add custom metadata
4. Search for files based on metadata
5. Generate metadata statistics
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.backend.storage.database.db_manager import DatabaseManager
from src.backend.storage.files.file_manager import FileManager
from src.backend.storage.metadata.metadata_manager import MetadataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_managers():
    """Set up and return database, file, and metadata managers."""
    # Initialize database manager
    db_manager = DatabaseManager()
    if not db_manager.initialize():
        logger.error("Failed to initialize database")
        sys.exit(1)
    
    # Initialize file manager
    file_manager = FileManager(db_manager)
    
    # Initialize metadata manager
    metadata_manager = MetadataManager(db_manager, file_manager)
    
    return db_manager, file_manager, metadata_manager


def extract_metadata_example(metadata_manager, file_path):
    """Example of extracting metadata from an audio file."""
    print("\n=== Extracting Metadata ===")
    
    try:
        # Extract metadata
        metadata = metadata_manager.extract_metadata(file_path)
        
        # Print metadata
        print(f"Metadata for {file_path}:")
        print(json.dumps(metadata, indent=2))
        
        return metadata
    
    except Exception as e:
        logger.error(f"Error extracting metadata: {str(e)}")
        return None


def store_metadata_example(metadata_manager, file_manager, file_path, metadata=None):
    """Example of storing metadata in the database."""
    print("\n=== Storing Metadata ===")
    
    try:
        # Extract and store metadata
        file_id = metadata_manager.extract_and_store_metadata(file_path)
        
        if file_id is not None:
            print(f"Metadata stored for file ID: {file_id}")
            return file_id
        else:
            logger.error("Failed to store metadata")
            return None
    
    except Exception as e:
        logger.error(f"Error storing metadata: {str(e)}")
        return None


def add_custom_metadata_example(metadata_manager, file_id):
    """Example of adding custom metadata to a file."""
    print("\n=== Adding Custom Metadata ===")
    
    try:
        # Add custom metadata fields
        custom_fields = [
            ("recording_location", "Studio A", "text"),
            ("recording_date", "2025-03-01", "date"),
            ("is_final_version", True, "boolean"),
            ("take_number", 3, "number"),
            ("tags", ["vocals", "clean", "processed"], "tags")
        ]
        
        for field_name, field_value, field_type in custom_fields:
            result = metadata_manager.add_custom_metadata(
                file_id, field_name, field_value, field_type
            )
            
            if result:
                print(f"Added custom field '{field_name}' ({field_type}) with value: {field_value}")
            else:
                logger.error(f"Failed to add custom field '{field_name}'")
        
        # Get updated metadata
        updated_metadata = metadata_manager.get_metadata(file_id)
        print("\nUpdated metadata with custom fields:")
        print(json.dumps(updated_metadata.get("custom_fields", {}), indent=2))
    
    except Exception as e:
        logger.error(f"Error adding custom metadata: {str(e)}")


def search_metadata_example(metadata_manager):
    """Example of searching for files based on metadata."""
    print("\n=== Searching by Metadata ===")
    
    try:
        # Search criteria examples
        search_examples = [
            {"file_format": "wav", "criteria_name": "WAV files"},
            {"duration_min": 60, "criteria_name": "Files longer than 60 seconds"},
            {"metadata_field": "rms_mean", "metadata_value": "0.1", "criteria_name": "Files with RMS mean around 0.1"},
            {"custom_field": "recording_location", "custom_value": "Studio", "criteria_name": "Files recorded in a studio"}
        ]
        
        for criteria in search_examples:
            criteria_name = criteria.pop("criteria_name")
            
            print(f"\nSearching for {criteria_name}:")
            results = metadata_manager.search_by_metadata(criteria)
            
            print(f"Found {len(results)} matching files:")
            for result in results:
                print(f"- ID: {result['id']}, Filename: {result['filename']}")
    
    except Exception as e:
        logger.error(f"Error searching metadata: {str(e)}")


def metadata_statistics_example(metadata_manager):
    """Example of generating metadata statistics."""
    print("\n=== Metadata Statistics ===")
    
    try:
        # Get metadata statistics
        statistics = metadata_manager.get_metadata_statistics()
        
        # Print statistics
        print("Metadata Statistics:")
        print(f"Total files: {statistics['total_files']}")
        print(f"Files with metadata: {statistics['files_with_metadata']}")
        print(f"Files with custom fields: {statistics['files_with_custom_fields']}")
        print(f"Average duration: {statistics['average_duration']:.2f} seconds")
        print(f"Total duration: {statistics['total_duration']:.2f} seconds")
        
        print("\nFormat distribution:")
        for format_name, count in statistics.get('format_distribution', {}).items():
            print(f"  - {format_name}: {count}")
        
        print("\nSample rate distribution:")
        for sample_rate, count in statistics.get('sample_rate_distribution', {}).items():
            print(f"  - {sample_rate} Hz: {count}")
        
        print("\nChannels distribution:")
        for channels, count in statistics.get('channels_distribution', {}).items():
            print(f"  - {channels} channel(s): {count}")
        
        if statistics.get('custom_field_types'):
            print("\nCustom field types:")
            for field_type, count in statistics['custom_field_types'].items():
                print(f"  - {field_type}: {count}")
    
    except Exception as e:
        logger.error(f"Error generating metadata statistics: {str(e)}")


def batch_processing_example(metadata_manager, directory_path):
    """Example of batch processing audio files."""
    print("\n=== Batch Processing ===")
    
    try:
        # Check if directory exists
        if not os.path.exists(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return
        
        # Batch extract metadata
        print(f"Processing audio files in {directory_path}...")
        results = metadata_manager.batch_extract_metadata(directory_path, recursive=True)
        
        # Print results
        print(f"Processed {results['total_files']} files:")
        print(f"  - Successfully processed: {results['processed_files']}")
        print(f"  - Failed: {results['failed_files']}")
        print(f"  - Skipped: {results['skipped_files']}")
    
    except Exception as e:
        logger.error(f"Error batch processing files: {str(e)}")


def export_import_example(metadata_manager, file_id, export_path):
    """Example of exporting and importing metadata."""
    print("\n=== Export/Import Metadata ===")
    
    try:
        # Export metadata
        print(f"Exporting metadata for file ID {file_id} to {export_path}...")
        export_result = metadata_manager.export_metadata(file_id, export_path)
        
        if export_result:
            print(f"Metadata exported successfully to {export_path}")
            
            # Modify the exported metadata
            with open(export_path, 'r') as f:
                metadata = json.load(f)
            
            # Add a new custom field
            if "metadata" in metadata and "custom_fields" in metadata["metadata"]:
                metadata["metadata"]["custom_fields"]["exported_and_modified"] = {
                    "type": "boolean",
                    "value": True
                }
            
            # Save the modified metadata
            with open(export_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print("Modified the exported metadata")
            
            # Import the modified metadata
            print(f"Importing modified metadata back to file ID {file_id}...")
            import_result = metadata_manager.import_metadata(export_path, file_id)
            
            if import_result:
                print("Metadata imported successfully")
                
                # Get the updated metadata
                updated_metadata = metadata_manager.get_metadata(file_id)
                if "custom_fields" in updated_metadata and "exported_and_modified" in updated_metadata["custom_fields"]:
                    print("Verified that the imported metadata contains the new custom field")
            else:
                logger.error("Failed to import metadata")
        else:
            logger.error(f"Failed to export metadata for file ID {file_id}")
    
    except Exception as e:
        logger.error(f"Error in export/import example: {str(e)}")


def main():
    """Main function demonstrating metadata management functionality."""
    print("=== Metadata Management Example ===")
    
    # Set up managers
    db_manager, file_manager, metadata_manager = setup_managers()
    
    # Example audio file path (replace with an actual audio file path)
    audio_file_path = input("Enter the path to an audio file: ")
    
    if not os.path.exists(audio_file_path):
        logger.error(f"File not found: {audio_file_path}")
        sys.exit(1)
    
    # Extract metadata example
    metadata = extract_metadata_example(metadata_manager, audio_file_path)
    
    # Store metadata example
    file_id = store_metadata_example(metadata_manager, file_manager, audio_file_path, metadata)
    
    if file_id is not None:
        # Add custom metadata example
        add_custom_metadata_example(metadata_manager, file_id)
        
        # Export/import example
        export_path = os.path.join(os.path.dirname(audio_file_path), "metadata_export.json")
        export_import_example(metadata_manager, file_id, export_path)
    
    # Search metadata example
    search_metadata_example(metadata_manager)
    
    # Metadata statistics example
    metadata_statistics_example(metadata_manager)
    
    # Batch processing example
    directory_path = os.path.dirname(audio_file_path)
    batch_processing_example(metadata_manager, directory_path)
    
    # Clean up
    db_manager.close_session()
    print("\n=== Example Completed ===")


if __name__ == "__main__":
    main()