# Database Module

This module provides database functionality for the Voices application, including schema definition, database management, migrations, and a command-line interface.

## Overview

The database module is responsible for storing and retrieving data related to:
- Speakers (voice profiles)
- Processed audio files
- Processing history
- ML model performance metrics

It uses SQLite as the database engine with SQLAlchemy as the ORM layer.

## Components

### Models (`models.py`)

Defines the SQLAlchemy ORM models for the database tables:

- **Speaker**: Information about speakers in audio files
  - Properties: name, description, tags
  - Relationships: processed_files

- **ProcessedFile**: Information about processed audio files
  - Properties: filename, paths, format, duration, sample rate, channels, metadata
  - Relationships: speaker, processing_history

- **ProcessingHistory**: History of processing operations
  - Properties: operation type, model type, parameters, timestamps, success status
  - Relationships: processed_file, model_performance

- **ModelPerformance**: Performance metrics for ML models
  - Properties: model type, version, metrics (SI-SNRi, SDRi), resource usage
  - Relationships: processing_history

### Database Manager (`db_manager.py`)

Provides a high-level API for interacting with the database:

- Connection management
- Table creation
- CRUD operations
- Query interface
- Database backup

### Migrations (`migrations.py`)

Handles database schema migrations:

- Migration tracking
- Migration file management
- Migration application
- Initial schema creation

### CLI (`cli.py`)

Command-line interface for database operations:

- Database initialization
- Migration management
- Data listing and querying
- Database backup

## Usage

### Initialization

```python
from src.backend.storage.database import DatabaseManager

# Create a database manager
db_manager = DatabaseManager()

# Initialize the database (connect, create tables, run migrations)
db_manager.initialize()
```

### Adding Data

```python
from src.backend.storage.database import DatabaseManager, Speaker, ProcessedFile

# Create a database manager
db_manager = DatabaseManager()
db_manager.connect()

# Create a speaker
speaker = Speaker(name="John Doe", description="Male speaker")
db_manager.add(speaker)

# Create a processed file
processed_file = ProcessedFile(
    filename="recording.wav",
    original_path="/path/to/original.wav",
    processed_path="/path/to/processed.wav",
    file_format="wav",
    duration=120.5,
    sample_rate=44100,
    channels=2,
    speaker_id=speaker.id
)
db_manager.add(processed_file)
```

### Querying Data

```python
from src.backend.storage.database import DatabaseManager, Speaker, ProcessedFile

# Create a database manager
db_manager = DatabaseManager()
db_manager.connect()

# Get all speakers
speakers = db_manager.get_all(Speaker)

# Get a specific speaker
speaker = db_manager.get_by_id(Speaker, 1)

# Query processed files for a speaker
files = db_manager.session.query(ProcessedFile).filter_by(speaker_id=1).all()
```

### Using the CLI

The database module provides a command-line interface for common operations:

```bash
# Initialize the database
python -m src.backend.storage.database.cli init

# Create a new migration
python -m src.backend.storage.database.cli create-migration add_new_field

# Run pending migrations
python -m src.backend.storage.database.cli run-migrations

# List speakers
python -m src.backend.storage.database.cli list-speakers

# List processed files
python -m src.backend.storage.database.cli list-files

# List model performance metrics
python -m src.backend.storage.database.cli list-performance

# Backup the database
python -m src.backend.storage.database.cli backup --backup-path /path/to/backup.db
```

## Migration System

The database module includes a migration system for managing schema changes over time:

1. Migrations are stored in the `migrations` directory as SQL or JSON files
2. Each migration has a version number and description
3. Migrations are tracked in the `schema_migrations` table
4. Only pending migrations are applied when running migrations

To create a new migration:

```bash
python -m src.backend.storage.database.cli create-migration add_new_field
```

This creates a new migration file that you can edit to add your schema changes.

## Dependencies

- SQLAlchemy: ORM and database toolkit
- SQLite: Database engine