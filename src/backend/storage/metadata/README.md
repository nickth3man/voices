# Metadata Management Module

This module provides functionality for extracting, storing, retrieving, and searching metadata for audio files in the Voices application.

## Overview

The Metadata Management module is responsible for:

1. Extracting metadata from audio files
2. Storing metadata in the database
3. Retrieving metadata for specific files
4. Searching for files based on metadata criteria
5. Managing custom metadata fields
6. Exporting and importing metadata
7. Batch processing of metadata extraction
8. Providing statistics about metadata in the database

## Components

### MetadataManager

The `MetadataManager` class is the main component of this module. It provides methods for all metadata operations:

- `extract_metadata(file_path, extract_audio_characteristics=True)`: Extract metadata from an audio file
- `store_metadata(file_id, metadata)`: Store metadata for a file
- `get_metadata(file_id)`: Get metadata for a file
- `add_custom_metadata(file_id, field_name, field_value, field_type='text')`: Add custom metadata for a file
- `remove_custom_metadata(file_id, field_name)`: Remove custom metadata for a file
- `search_by_metadata(criteria, limit=100, offset=0)`: Search for files based on metadata criteria
- `export_metadata(file_id, export_path=None)`: Export metadata for a file to a JSON file
- `import_metadata(import_path, file_id)`: Import metadata for a file from a JSON file
- `batch_extract_metadata(directory, recursive=False)`: Extract metadata for all audio files in a directory
- `get_metadata_statistics()`: Get statistics about metadata in the database

### CLI Interface

The module includes a command-line interface (`cli.py`) that provides access to all metadata operations:

```
python -m backend.storage.metadata.cli extract <file_path> [--basic] [--output OUTPUT]
python -m backend.storage.metadata.cli store <file_id> <metadata_file>
python -m backend.storage.metadata.cli get <file_id> [--output OUTPUT]
python -m backend.storage.metadata.cli add-custom <file_id> <field_name> <field_value> [--field-type {text,number,boolean,date}]
python -m backend.storage.metadata.cli remove-custom <file_id> <field_name>
python -m backend.storage.metadata.cli search [--filename FILENAME] [--file-format FILE_FORMAT] [--duration-min DURATION_MIN] [--duration-max DURATION_MAX] [--created-after CREATED_AFTER] [--created-before CREATED_BEFORE] [--limit LIMIT] [--offset OFFSET] [--output OUTPUT]
python -m backend.storage.metadata.cli export <file_id> <output>
python -m backend.storage.metadata.cli import <file_id> <input>
python -m backend.storage.metadata.cli batch-extract <directory> [--recursive] [--output OUTPUT]
python -m backend.storage.metadata.cli statistics [--output OUTPUT]
python -m backend.storage.metadata.cli list-files [--filename FILENAME] [--file-format FILE_FORMAT] [--output OUTPUT]
```

## Database Schema

The module uses two tables in the database:

### FileMetadata

Stores metadata for audio files:

- `id`: Primary key
- `file_id`: Foreign key to ProcessedFile
- `duration`: Duration in seconds
- `sample_rate`: Sample rate in Hz
- `channels`: Number of audio channels
- `file_format`: File format (wav, mp3, etc.)
- `rms_mean`: Root Mean Square energy (mean)
- `rms_std`: Root Mean Square energy (std)
- `spectral_centroid_mean`: Spectral centroid (mean)
- `spectral_centroid_std`: Spectral centroid (std)
- `spectral_bandwidth_mean`: Spectral bandwidth (mean)
- `spectral_bandwidth_std`: Spectral bandwidth (std)
- `zero_crossing_rate_mean`: Zero crossing rate (mean)
- `zero_crossing_rate_std`: Zero crossing rate (std)
- `mfcc_means`: MFCC means (JSON)
- `mfcc_stds`: MFCC standard deviations (JSON)
- `extracted_at`: When metadata was extracted
- `created_at`: When the record was created
- `updated_at`: When the record was last updated

### CustomMetadata

Stores custom metadata fields for audio files:

- `id`: Primary key
- `file_id`: Foreign key to ProcessedFile
- `field_name`: Name of the custom field
- `field_value`: Value of the custom field
- `field_type`: Type of the custom field (text, number, boolean, date)
- `created_at`: When the record was created
- `updated_at`: When the record was last updated

## Frontend Integration

The module integrates with the frontend through the `MetadataBridge` controller, which provides methods for all metadata operations. The frontend also includes components for metadata management:

- `MetadataManagement`: Main component that integrates both editor and search
- `MetadataEditor`: Component for viewing and editing metadata
- `MetadataSearch`: Component for searching files based on metadata criteria

## Dependencies

- `librosa`: For audio analysis and metadata extraction
- `numpy`: For numerical operations
- `sqlalchemy`: For database operations
- `backend.storage.database`: For database access
- `backend.storage.files`: For file management

## Usage Examples

### Extracting Metadata

```python
from backend.storage.database.db_manager import DatabaseManager
from backend.storage.files.file_manager import FileManager
from backend.storage.metadata.metadata_manager import MetadataManager

# Initialize managers
db_manager = DatabaseManager()
file_manager = FileManager(db_manager)
metadata_manager = MetadataManager(db_manager, file_manager)

# Extract metadata
metadata = metadata_manager.extract_metadata('path/to/audio.wav')
print(metadata)
```

### Searching by Metadata

```python
# Search for files with specific criteria
criteria = {
    'filename': 'voice',
    'file_format': 'wav',
    'duration_min': 10.0,
    'duration_max': 60.0
}

results = metadata_manager.search_by_metadata(criteria)
print(results)
```

### Adding Custom Metadata

```python
# Add custom metadata
metadata_manager.add_custom_metadata(
    file_id=1,
    field_name='recording_location',
    field_value='Studio A',
    field_type='text'
)
```

### Batch Processing

```python
# Extract metadata for all audio files in a directory
results = metadata_manager.batch_extract_metadata(
    directory='path/to/audio/files',
    recursive=True
)
print(results)