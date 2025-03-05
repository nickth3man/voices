# File Organization System

The File Organization System manages storage locations, implements naming conventions, handles automatic file organization, and provides monitoring of storage usage for the Voices application.

## Features

- **Storage Location Management**: Configurable directory structure for different file types (original, processed, temporary, archive, exports)
- **Naming Conventions**: Consistent file naming with support for speaker names, processing types, and timestamps
- **Automatic File Organization**: Organize files based on date or custom rules
- **Storage Usage Monitoring**: Track disk usage and file statistics
- **Database Integration**: Register files in the database for tracking and metadata
- **CLI Interface**: Command-line tools for managing files

## Directory Structure

The File Organization System creates the following directory structure:

```
base_dir/
├── original/     # Original audio files
├── processed/    # Processed audio files
├── temp/         # Temporary files
├── archive/      # Archived files
├── exports/      # Exported files
```

## Usage

### Python API

```python
from src.backend.storage.database.db_manager import DatabaseManager
from src.backend.storage.files.file_manager import FileManager

# Initialize database manager
db_manager = DatabaseManager()
db_manager.initialize()

# Initialize file manager
file_manager = FileManager(db_manager)

# Store a file
dest_path = file_manager.store_file(
    source_path="path/to/audio.wav",
    dir_type="original",
    speaker_name="John Doe",
    processing_type="raw"
)

# Get storage usage information
usage_info = file_manager.get_storage_usage()
print(f"Total size: {usage_info['total_size_formatted']}")
```

### Command-Line Interface

The File Organization System provides a command-line interface for common operations:

```bash
# Initialize the file organization system
python -m src.backend.storage.files.cli init

# Store a file
python -m src.backend.storage.files.cli store path/to/audio.wav --dir-type original --speaker "John Doe"

# Get storage usage information
python -m src.backend.storage.files.cli usage

# Find files matching a pattern
python -m src.backend.storage.files.cli find "*.wav"

# Organize files by date
python -m src.backend.storage.files.cli organize
```

## Configuration

The File Organization System can be configured through a JSON configuration file:

```json
{
  "base_dir": "/path/to/storage",
  "directories": {
    "original": "original_files",
    "processed": "processed_files",
    "temp": "temporary",
    "archive": "archived_files",
    "exports": "exported_files"
  },
  "organization": {
    "original": {
      "by_date": true,
      "by_speaker": false
    },
    "processed": {
      "by_date": true,
      "by_speaker": true
    }
  }
}
```

Load a configuration file:

```python
config = FileManager.load_config("path/to/config.json")
file_manager = FileManager(db_manager, config=config)
```

Or via CLI:

```bash
python -m src.backend.storage.files.cli init --config path/to/config.json
```

## API Reference

### FileManager

The `FileManager` class provides methods for managing files:

- `store_file(source_path, dir_type, new_filename=None, speaker_name=None, processing_type=None)`: Store a file in the appropriate directory
- `move_file(source_path, dir_type, new_filename=None, speaker_name=None, processing_type=None)`: Move a file to the appropriate directory
- `delete_file(file_path)`: Delete a file
- `archive_file(source_path, new_filename=None, speaker_name=None)`: Archive a file
- `organize_files(dir_type=None)`: Organize files based on configuration
- `get_storage_usage(dir_type=None)`: Get storage usage information
- `find_files(pattern, dir_type=None)`: Find files matching a pattern
- `register_file(file_path, original_path=None, speaker_id=None, metadata=None)`: Register a file in the database
- `get_file_info(file_id)`: Get information about a registered file
- `get_all_files()`: Get information about all registered files
- `export_file(file_id, export_format=None, export_path=None)`: Export a file to a specific format and location
- `cleanup_temp_files(max_age_days=7)`: Clean up temporary files older than a specified age
- `save_config(config_path=None)`: Save the current configuration to a file
- `load_config(config_path)`: Load configuration from a file

## Integration with Other Components

The File Organization System integrates with:

- **Database System**: Files can be registered in the database for tracking and metadata
- **Speaker Management System**: Files can be linked to speakers
- **Audio Processing Pipeline**: Processed files can be stored and organized