# Speaker Management System

The Speaker Management System provides comprehensive functionality for managing speaker profiles in the Voices application, including CRUD operations, tagging system, metadata storage, and linking to processed audio files.

## Features

- **Speaker Profile Management**: Create, read, update, and delete speaker profiles
- **Tagging System**: Add, remove, and search by tags
- **Metadata Storage**: Store and retrieve speaker metadata
- **Audio File Linking**: Link processed audio files to speakers
- **Search Capabilities**: Search speakers by name, description, or tags
- **CLI Interface**: Command-line interface for all operations

## Components

The Speaker Management System consists of the following components:

1. **SpeakerManager Class**: Core class providing all speaker management functionality
2. **CLI Interface**: Command-line interface for speaker management
3. **Test Script**: Comprehensive test script for verifying functionality

## Usage

### Using the SpeakerManager Class

```python
from src.backend.storage.database.db_manager import DatabaseManager
from src.backend.storage.database.speaker_manager import SpeakerManager

# Initialize database manager and speaker manager
db_manager = DatabaseManager()
speaker_manager = SpeakerManager(db_manager)

# Create a speaker
speaker = speaker_manager.create_speaker(
    name="John Doe",
    description="A male voice actor",
    tags=["male", "deep", "english"]
)

# Get a speaker by ID
speaker = speaker_manager.get_speaker(speaker_id)

# Update a speaker
success = speaker_manager.update_speaker(
    speaker_id=speaker_id,
    name="Updated Name",
    description="Updated description"
)

# Delete a speaker
success = speaker_manager.delete_speaker(speaker_id)

# Add tags to a speaker
success = speaker_manager.add_tags(speaker_id, ["new", "tags"])

# Remove tags from a speaker
success = speaker_manager.remove_tags(speaker_id, ["old", "tags"])

# Get all tags for a speaker
tags = speaker_manager.get_tags(speaker_id)

# Get all unique tags across all speakers
all_tags = speaker_manager.get_all_tags()

# Find speakers by tags
speakers = speaker_manager.find_speakers_by_tags(["english", "male"], match_all=True)

# Search speakers by name or description
speakers = speaker_manager.search_speakers("voice actor")

# Link a processed file to a speaker
success = speaker_manager.link_processed_file(speaker_id, file_id)

# Unlink a processed file from its speaker
success = speaker_manager.unlink_processed_file(file_id)

# Get all processed files for a speaker
files = speaker_manager.get_processed_files(speaker_id)
```

### Using the CLI Interface

The Speaker Management System provides a comprehensive command-line interface for all operations:

```bash
# Create a speaker
python -m src.backend.storage.database.speaker_cli create "John Doe" --description "A male voice actor" --tags male deep english

# Get a speaker
python -m src.backend.storage.database.speaker_cli get 1

# List all speakers
python -m src.backend.storage.database.speaker_cli list

# Update a speaker
python -m src.backend.storage.database.speaker_cli update 1 --name "Updated Name" --description "Updated description"

# Delete a speaker
python -m src.backend.storage.database.speaker_cli delete 1

# Add tags to a speaker
python -m src.backend.storage.database.speaker_cli add-tags 1 new tags

# Remove tags from a speaker
python -m src.backend.storage.database.speaker_cli remove-tags 1 old tags

# Get all tags for a speaker
python -m src.backend.storage.database.speaker_cli get-tags 1

# List all unique tags
python -m src.backend.storage.database.speaker_cli list-tags

# Find speakers by tags
python -m src.backend.storage.database.speaker_cli find-by-tags english male --match-all

# Search speakers
python -m src.backend.storage.database.speaker_cli search "voice actor"

# Link a processed file to a speaker
python -m src.backend.storage.database.speaker_cli link-file 1 2

# Unlink a processed file
python -m src.backend.storage.database.speaker_cli unlink-file 2

# Get all processed files for a speaker
python -m src.backend.storage.database.speaker_cli get-files 1
```

## Testing

The Speaker Management System includes a comprehensive test script that verifies all functionality:

```bash
python -m src.backend.storage.database.test_speaker_manager
```

The test script performs the following tests:

1. Speaker creation
2. Speaker retrieval
3. Speaker update
4. Tag operations (add, remove, get)
5. Processed file operations (link, unlink, get)
6. Speaker search
7. Speaker deletion

## Integration with Other Components

The Speaker Management System integrates with the following components:

- **Database Manager**: For database operations
- **Processed Files**: For linking speakers to processed audio files
- **Audio Processing Pipeline**: For associating speakers with processed audio
- **User Interface**: For displaying and managing speaker profiles

## Error Handling

The Speaker Management System includes comprehensive error handling:

- All operations return appropriate success/failure indicators
- Detailed error messages are logged for debugging
- Robust validation of input parameters
- Graceful handling of missing speakers or files

## Future Enhancements

Potential future enhancements for the Speaker Management System:

1. Speaker verification using voice prints
2. Automatic speaker identification in audio files
3. Speaker similarity matching
4. Enhanced metadata for speaker characteristics
5. Integration with external speaker databases