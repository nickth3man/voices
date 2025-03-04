"""
Command-line interface for database operations.

This module provides a CLI for managing the database, including initialization,
migrations, and basic operations.
"""

import os
import sys
import argparse
import logging
import json
from typing import List, Optional

from .db_manager import DatabaseManager
from .speaker_manager import SpeakerManager
from .migrations import create_migration, run_migrations
from .models import Speaker, ProcessedFile, ProcessingHistory, ModelPerformance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def init_db(args: argparse.Namespace) -> bool:
    """
    Initialize the database.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        bool: True if initialization successful, False otherwise.
    """
    db_path = args.db_path
    db_manager = DatabaseManager(db_path)
    
    logger.info(f"Initializing database at {db_path}")
    if db_manager.initialize():
        logger.info("Database initialized successfully")
        return True
    else:
        logger.error("Failed to initialize database")
        return False


def create_migration_cmd(args: argparse.Namespace) -> bool:
    """
    Create a new migration.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        bool: True if migration created successfully, False otherwise.
    """
    name = args.name
    
    try:
        path = create_migration(name)
        logger.info(f"Created migration at {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create migration: {str(e)}")
        return False


def run_migrations_cmd(args: argparse.Namespace) -> bool:
    """
    Run pending migrations.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        bool: True if migrations ran successfully, False otherwise.
    """
    db_path = args.db_path
    db_manager = DatabaseManager(db_path)
    
    if not db_manager.connect():
        logger.error("Failed to connect to database")
        return False
    
    logger.info(f"Running migrations on database at {db_path}")
    if db_manager.run_migrations():
        logger.info("Migrations completed successfully")
        return True
    else:
        logger.error("Failed to run migrations")
        return False


def list_speakers(args: argparse.Namespace) -> bool:
    """
    List all speakers in the database.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        bool: True if listing successful, False otherwise.
    """
    db_path = args.db_path
    db_manager = DatabaseManager(db_path)
    
    if not db_manager.connect():
        logger.error("Failed to connect to database")
        return False
    
    try:
        speaker_manager = SpeakerManager(db_manager)
        speakers = speaker_manager.get_all_speakers()
        
        if not speakers:
            print("No speakers found in the database.")
            return True
        
        print(f"Found {len(speakers)} speakers:")
        for speaker in speakers:
            tags = ", ".join(speaker.tags) if speaker.tags else "None"
            print(f"  ID: {speaker.id}, Name: {speaker.name}")
            print(f"    Description: {speaker.description or 'None'}")
            print(f"    Tags: {tags}")
            print(f"    Created: {speaker.created_at}")
            print(f"    Updated: {speaker.updated_at}")
            print()
        
        return True
    except Exception as e:
        logger.error(f"Error listing speakers: {str(e)}")
        return False


def create_speaker(args: argparse.Namespace) -> bool:
    """
    Create a new speaker.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        bool: True if creation successful, False otherwise.
    """
    db_path = args.db_path
    db_manager = DatabaseManager(db_path)
    
    if not db_manager.connect():
        logger.error("Failed to connect to database")
        return False
    
    try:
        name = args.name
        description = args.description
        tags = args.tags.split(",") if args.tags else []
        
        # Strip whitespace from tags
        tags = [tag.strip() for tag in tags if tag.strip()]
        
        speaker_manager = SpeakerManager(db_manager)
        speaker = speaker_manager.create_speaker(name, description, tags)
        
        if speaker:
            print(f"Created speaker: ID={speaker.id}, Name='{speaker.name}'")
            return True
        else:
            print("Failed to create speaker.")
            return False
    except Exception as e:
        logger.error(f"Error creating speaker: {str(e)}")
        return False


def update_speaker(args: argparse.Namespace) -> bool:
    """
    Update a speaker.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        bool: True if update successful, False otherwise.
    """
    db_path = args.db_path
    db_manager = DatabaseManager(db_path)
    
    if not db_manager.connect():
        logger.error("Failed to connect to database")
        return False
    
    try:
        speaker_id = args.id
        name = args.name
        description = args.description
        
        speaker_manager = SpeakerManager(db_manager)
        success = speaker_manager.update_speaker(speaker_id, name, description)
        
        if success:
            print(f"Updated speaker: ID={speaker_id}")
            return True
        else:
            print(f"Failed to update speaker: ID={speaker_id}")
            return False
    except Exception as e:
        logger.error(f"Error updating speaker: {str(e)}")
        return False


def delete_speaker(args: argparse.Namespace) -> bool:
    """
    Delete a speaker.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        bool: True if deletion successful, False otherwise.
    """
    db_path = args.db_path
    db_manager = DatabaseManager(db_path)
    
    if not db_manager.connect():
        logger.error("Failed to connect to database")
        return False
    
    try:
        speaker_id = args.id
        
        speaker_manager = SpeakerManager(db_manager)
        success = speaker_manager.delete_speaker(speaker_id)
        
        if success:
            print(f"Deleted speaker: ID={speaker_id}")
            return True
        else:
            print(f"Failed to delete speaker: ID={speaker_id}")
            return False
    except Exception as e:
        logger.error(f"Error deleting speaker: {str(e)}")
        return False


def add_tags(args: argparse.Namespace) -> bool:
    """
    Add tags to a speaker.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        bool: True if tags added successfully, False otherwise.
    """
    db_path = args.db_path
    db_manager = DatabaseManager(db_path)
    
    if not db_manager.connect():
        logger.error("Failed to connect to database")
        return False
    
    try:
        speaker_id = args.id
        tags = args.tags.split(",")
        
        # Strip whitespace from tags
        tags = [tag.strip() for tag in tags if tag.strip()]
        
        if not tags:
            print("No valid tags provided.")
            return False
        
        speaker_manager = SpeakerManager(db_manager)
        success = speaker_manager.add_tags(speaker_id, tags)
        
        if success:
            print(f"Added tags to speaker: ID={speaker_id}, Tags={', '.join(tags)}")
            return True
        else:
            print(f"Failed to add tags to speaker: ID={speaker_id}")
            return False
    except Exception as e:
        logger.error(f"Error adding tags to speaker: {str(e)}")
        return False


def remove_tags(args: argparse.Namespace) -> bool:
    """
    Remove tags from a speaker.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        bool: True if tags removed successfully, False otherwise.
    """
    db_path = args.db_path
    db_manager = DatabaseManager(db_path)
    
    if not db_manager.connect():
        logger.error("Failed to connect to database")
        return False
    
    try:
        speaker_id = args.id
        tags = args.tags.split(",")
        
        # Strip whitespace from tags
        tags = [tag.strip() for tag in tags if tag.strip()]
        
        if not tags:
            print("No valid tags provided.")
            return False
        
        speaker_manager = SpeakerManager(db_manager)
        success = speaker_manager.remove_tags(speaker_id, tags)
        
        if success:
            print(f"Removed tags from speaker: ID={speaker_id}, Tags={', '.join(tags)}")
            return True
        else:
            print(f"Failed to remove tags from speaker: ID={speaker_id}")
            return False
    except Exception as e:
        logger.error(f"Error removing tags from speaker: {str(e)}")
        return False


def list_speaker_tags(args: argparse.Namespace) -> bool:
    """
    List all tags for a speaker.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        bool: True if listing successful, False otherwise.
    """
    db_path = args.db_path
    db_manager = DatabaseManager(db_path)
    
    if not db_manager.connect():
        logger.error("Failed to connect to database")
        return False
    
    try:
        speaker_id = args.id
        
        speaker_manager = SpeakerManager(db_manager)
        tags = speaker_manager.get_tags(speaker_id)
        
        if not tags:
            print(f"No tags found for speaker: ID={speaker_id}")
            return True
        
        print(f"Tags for speaker ID={speaker_id}:")
        for tag in tags:
            print(f"  {tag}")
        
        return True
    except Exception as e:
        logger.error(f"Error listing tags for speaker: {str(e)}")
        return False


def list_all_tags(args: argparse.Namespace) -> bool:
    """
    List all unique tags used across all speakers.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        bool: True if listing successful, False otherwise.
    """
    db_path = args.db_path
    db_manager = DatabaseManager(db_path)
    
    if not db_manager.connect():
        logger.error("Failed to connect to database")
        return False
    
    try:
        speaker_manager = SpeakerManager(db_manager)
        tags = speaker_manager.get_all_tags()
        
        if not tags:
            print("No tags found in the database.")
            return True
        
        print(f"Found {len(tags)} unique tags:")
        for tag in sorted(tags):
            print(f"  {tag}")
        
        return True
    except Exception as e:
        logger.error(f"Error listing all tags: {str(e)}")
        return False


def find_speakers_by_tags(args: argparse.Namespace) -> bool:
    """
    Find speakers by tags.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        bool: True if search successful, False otherwise.
    """
    db_path = args.db_path
    db_manager = DatabaseManager(db_path)
    
    if not db_manager.connect():
        logger.error("Failed to connect to database")
        return False
    
    try:
        tags = args.tags.split(",")
        match_all = args.match_all
        
        # Strip whitespace from tags
        tags = [tag.strip() for tag in tags if tag.strip()]
        
        if not tags:
            print("No valid tags provided.")
            return False
        
        speaker_manager = SpeakerManager(db_manager)
        speakers = speaker_manager.find_speakers_by_tags(tags, match_all)
        
        if not speakers:
            match_type = "all" if match_all else "any"
            print(f"No speakers found with {match_type} of the tags: {', '.join(tags)}")
            return True
        
        match_type = "all" if match_all else "any"
        print(f"Found {len(speakers)} speakers with {match_type} of the tags: {', '.join(tags)}")
        for speaker in speakers:
            speaker_tags = ", ".join(speaker.tags) if speaker.tags else "None"
            print(f"  ID: {speaker.id}, Name: {speaker.name}")
            print(f"    Tags: {speaker_tags}")
            print()
        
        return True
    except Exception as e:
        logger.error(f"Error finding speakers by tags: {str(e)}")
        return False


def search_speakers(args: argparse.Namespace) -> bool:
    """
    Search for speakers by name or description.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        bool: True if search successful, False otherwise.
    """
    db_path = args.db_path
    db_manager = DatabaseManager(db_path)
    
    if not db_manager.connect():
        logger.error("Failed to connect to database")
        return False
    
    try:
        query = args.query
        
        speaker_manager = SpeakerManager(db_manager)
        speakers = speaker_manager.search_speakers(query)
        
        if not speakers:
            print(f"No speakers found matching query: '{query}'")
            return True
        
        print(f"Found {len(speakers)} speakers matching query: '{query}'")
        for speaker in speakers:
            tags = ", ".join(speaker.tags) if speaker.tags else "None"
            print(f"  ID: {speaker.id}, Name: {speaker.name}")
            print(f"    Description: {speaker.description or 'None'}")
            print(f"    Tags: {tags}")
            print()
        
        return True
    except Exception as e:
        logger.error(f"Error searching speakers: {str(e)}")
        return False


def link_file_to_speaker(args: argparse.Namespace) -> bool:
    """
    Link a processed file to a speaker.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        bool: True if linking successful, False otherwise.
    """
    db_path = args.db_path
    db_manager = DatabaseManager(db_path)
    
    if not db_manager.connect():
        logger.error("Failed to connect to database")
        return False
    
    try:
        speaker_id = args.speaker_id
        file_id = args.file_id
        
        speaker_manager = SpeakerManager(db_manager)
        success = speaker_manager.link_processed_file(speaker_id, file_id)
        
        if success:
            print(f"Linked file ID={file_id} to speaker ID={speaker_id}")
            return True
        else:
            print(f"Failed to link file ID={file_id} to speaker ID={speaker_id}")
            return False
    except Exception as e:
        logger.error(f"Error linking file to speaker: {str(e)}")
        return False


def unlink_file_from_speaker(args: argparse.Namespace) -> bool:
    """
    Unlink a processed file from its speaker.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        bool: True if unlinking successful, False otherwise.
    """
    db_path = args.db_path
    db_manager = DatabaseManager(db_path)
    
    if not db_manager.connect():
        logger.error("Failed to connect to database")
        return False
    
    try:
        file_id = args.file_id
        
        speaker_manager = SpeakerManager(db_manager)
        success = speaker_manager.unlink_processed_file(file_id)
        
        if success:
            print(f"Unlinked file ID={file_id} from speaker")
            return True
        else:
            print(f"Failed to unlink file ID={file_id}")
            return False
    except Exception as e:
        logger.error(f"Error unlinking file from speaker: {str(e)}")
        return False


def list_speaker_files(args: argparse.Namespace) -> bool:
    """
    List all processed files for a speaker.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        bool: True if listing successful, False otherwise.
    """
    db_path = args.db_path
    db_manager = DatabaseManager(db_path)
    
    if not db_manager.connect():
        logger.error("Failed to connect to database")
        return False
    
    try:
        speaker_id = args.id
        
        speaker_manager = SpeakerManager(db_manager)
        files = speaker_manager.get_processed_files(speaker_id)
        
        if not files:
            print(f"No processed files found for speaker: ID={speaker_id}")
            return True
        
        print(f"Found {len(files)} processed files for speaker ID={speaker_id}:")
        for file in files:
            print(f"  ID: {file.id}, Filename: {file.filename}")
            print(f"    Format: {file.file_format}, Duration: {file.duration}s")
            print(f"    Original Path: {file.original_path}")
            print(f"    Processed Path: {file.processed_path}")
            print()
        
        return True
    except Exception as e:
        logger.error(f"Error listing files for speaker: {str(e)}")
        return False


def list_processed_files(args: argparse.Namespace) -> bool:
    """
    List all processed files in the database.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        bool: True if listing successful, False otherwise.
    """
    db_path = args.db_path
    db_manager = DatabaseManager(db_path)
    
    if not db_manager.connect():
        logger.error("Failed to connect to database")
        return False
    
    try:
        files = db_manager.get_all(ProcessedFile)
        
        if not files:
            print("No processed files found in the database.")
            return True
        
        print(f"Found {len(files)} processed files:")
        for file in files:
            speaker_name = file.speaker.name if file.speaker else "Unknown"
            print(f"  ID: {file.id}, Filename: {file.filename}, Speaker: {speaker_name}")
        
        return True
    except Exception as e:
        logger.error(f"Error listing processed files: {str(e)}")
        return False


def list_model_performance(args: argparse.Namespace) -> bool:
    """
    List model performance metrics in the database.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        bool: True if listing successful, False otherwise.
    """
    db_path = args.db_path
    db_manager = DatabaseManager(db_path)
    
    if not db_manager.connect():
        logger.error("Failed to connect to database")
        return False
    
    try:
        metrics = db_manager.get_all(ModelPerformance)
        
        if not metrics:
            print("No model performance metrics found in the database.")
            return True
        
        print(f"Found {len(metrics)} model performance records:")
        for metric in metrics:
            print(f"  ID: {metric.id}, Model: {metric.model_type.value}, Version: {metric.model_version}")
            print(f"    SI-SNRi: {metric.si_snri}, SDRi: {metric.sdri}")
            print(f"    Processing Time: {metric.processing_time}s, Real-time Factor: {metric.real_time_factor}")
        
        return True
    except Exception as e:
        logger.error(f"Error listing model performance metrics: {str(e)}")
        return False


def backup_database_cmd(args: argparse.Namespace) -> bool:
    """
    Backup the database.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        bool: True if backup successful, False otherwise.
    """
    db_path = args.db_path
    backup_path = args.backup_path
    db_manager = DatabaseManager(db_path)
    
    if not db_manager.connect():
        logger.error("Failed to connect to database")
        return False
    
    logger.info(f"Backing up database from {db_path} to {backup_path}")
    if db_manager.backup_database(backup_path):
        logger.info("Database backup completed successfully")
        return True
    else:
        logger.error("Failed to backup database")
        return False


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    parser = argparse.ArgumentParser(description="Database management CLI for Voices application")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Default database path
    default_db_path = os.path.join(os.path.expanduser("~"), ".voices", "voices.db")
    
    # init command
    init_parser = subparsers.add_parser("init", help="Initialize the database")
    init_parser.add_argument("--db-path", default=default_db_path, help="Path to the database file")
    
    # create-migration command
    create_migration_parser = subparsers.add_parser("create-migration", help="Create a new migration")
    create_migration_parser.add_argument("name", help="Name of the migration")
    
    # run-migrations command
    run_migrations_parser = subparsers.add_parser("run-migrations", help="Run pending migrations")
    run_migrations_parser.add_argument("--db-path", default=default_db_path, help="Path to the database file")
    
    # list-speakers command
    list_speakers_parser = subparsers.add_parser("list-speakers", help="List all speakers in the database")
    list_speakers_parser.add_argument("--db-path", default=default_db_path, help="Path to the database file")
    
    # create-speaker command
    create_speaker_parser = subparsers.add_parser("create-speaker", help="Create a new speaker")
    create_speaker_parser.add_argument("--db-path", default=default_db_path, help="Path to the database file")
    create_speaker_parser.add_argument("--name", required=True, help="Name of the speaker")
    create_speaker_parser.add_argument("--description", help="Description of the speaker")
    create_speaker_parser.add_argument("--tags", help="Comma-separated list of tags")
    
    # update-speaker command
    update_speaker_parser = subparsers.add_parser("update-speaker", help="Update a speaker")
    update_speaker_parser.add_argument("--db-path", default=default_db_path, help="Path to the database file")
    update_speaker_parser.add_argument("--id", required=True, type=int, help="ID of the speaker")
    update_speaker_parser.add_argument("--name", help="New name for the speaker")
    update_speaker_parser.add_argument("--description", help="New description for the speaker")
    
    # delete-speaker command
    delete_speaker_parser = subparsers.add_parser("delete-speaker", help="Delete a speaker")
    delete_speaker_parser.add_argument("--db-path", default=default_db_path, help="Path to the database file")
    delete_speaker_parser.add_argument("--id", required=True, type=int, help="ID of the speaker")
    
    # add-tags command
    add_tags_parser = subparsers.add_parser("add-tags", help="Add tags to a speaker")
    add_tags_parser.add_argument("--db-path", default=default_db_path, help="Path to the database file")
    add_tags_parser.add_argument("--id", required=True, type=int, help="ID of the speaker")
    add_tags_parser.add_argument("--tags", required=True, help="Comma-separated list of tags to add")
    
    # remove-tags command
    remove_tags_parser = subparsers.add_parser("remove-tags", help="Remove tags from a speaker")
    remove_tags_parser.add_argument("--db-path", default=default_db_path, help="Path to the database file")
    remove_tags_parser.add_argument("--id", required=True, type=int, help="ID of the speaker")
    remove_tags_parser.add_argument("--tags", required=True, help="Comma-separated list of tags to remove")
    
    # list-speaker-tags command
    list_speaker_tags_parser = subparsers.add_parser("list-speaker-tags", help="List all tags for a speaker")
    list_speaker_tags_parser.add_argument("--db-path", default=default_db_path, help="Path to the database file")
    list_speaker_tags_parser.add_argument("--id", required=True, type=int, help="ID of the speaker")
    
    # list-all-tags command
    list_all_tags_parser = subparsers.add_parser("list-all-tags", help="List all unique tags used across all speakers")
    list_all_tags_parser.add_argument("--db-path", default=default_db_path, help="Path to the database file")
    
    # find-speakers-by-tags command
    find_speakers_parser = subparsers.add_parser("find-speakers-by-tags", help="Find speakers by tags")
    find_speakers_parser.add_argument("--db-path", default=default_db_path, help="Path to the database file")
    find_speakers_parser.add_argument("--tags", required=True, help="Comma-separated list of tags to search for")
    find_speakers_parser.add_argument("--match-all", action="store_true", help="Match all tags (default: match any)")
    
    # search-speakers command
    search_speakers_parser = subparsers.add_parser("search-speakers", help="Search for speakers by name or description")
    search_speakers_parser.add_argument("--db-path", default=default_db_path, help="Path to the database file")
    search_speakers_parser.add_argument("--query", required=True, help="Search query string")
    
    # link-file-to-speaker command
    link_file_parser = subparsers.add_parser("link-file-to-speaker", help="Link a processed file to a speaker")
    link_file_parser.add_argument("--db-path", default=default_db_path, help="Path to the database file")
    link_file_parser.add_argument("--speaker-id", required=True, type=int, help="ID of the speaker")
    link_file_parser.add_argument("--file-id", required=True, type=int, help="ID of the processed file")
    
    # unlink-file-from-speaker command
    unlink_file_parser = subparsers.add_parser("unlink-file-from-speaker", help="Unlink a processed file from its speaker")
    unlink_file_parser.add_argument("--db-path", default=default_db_path, help="Path to the database file")
    unlink_file_parser.add_argument("--file-id", required=True, type=int, help="ID of the processed file")
    
    # list-speaker-files command
    list_speaker_files_parser = subparsers.add_parser("list-speaker-files", help="List all processed files for a speaker")
    list_speaker_files_parser.add_argument("--db-path", default=default_db_path, help="Path to the database file")
    list_speaker_files_parser.add_argument("--id", required=True, type=int, help="ID of the speaker")
    
    # list-files command
    list_files_parser = subparsers.add_parser("list-files", help="List all processed files in the database")
    list_files_parser.add_argument("--db-path", default=default_db_path, help="Path to the database file")
    
    # list-performance command
    list_performance_parser = subparsers.add_parser("list-performance", help="List model performance metrics")
    list_performance_parser.add_argument("--db-path", default=default_db_path, help="Path to the database file")
    
    # backup command
    backup_parser = subparsers.add_parser("backup", help="Backup the database")
    backup_parser.add_argument("--db-path", default=default_db_path, help="Path to the database file")
    backup_parser.add_argument("--backup-path", required=True, help="Path for the backup file")
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    if not parsed_args.command:
        parser.print_help()
        return 1
    
    # Execute command
    success = False
    if parsed_args.command == "init":
        success = init_db(parsed_args)
    elif parsed_args.command == "create-migration":
        success = create_migration_cmd(parsed_args)
    elif parsed_args.command == "run-migrations":
        success = run_migrations_cmd(parsed_args)
    elif parsed_args.command == "list-speakers":
        success = list_speakers(parsed_args)
    elif parsed_args.command == "create-speaker":
        success = create_speaker(parsed_args)
    elif parsed_args.command == "update-speaker":
        success = update_speaker(parsed_args)
    elif parsed_args.command == "delete-speaker":
        success = delete_speaker(parsed_args)
    elif parsed_args.command == "add-tags":
        success = add_tags(parsed_args)
    elif parsed_args.command == "remove-tags":
        success = remove_tags(parsed_args)
    elif parsed_args.command == "list-speaker-tags":
        success = list_speaker_tags(parsed_args)
    elif parsed_args.command == "list-all-tags":
        success = list_all_tags(parsed_args)
    elif parsed_args.command == "find-speakers-by-tags":
        success = find_speakers_by_tags(parsed_args)
    elif parsed_args.command == "search-speakers":
        success = search_speakers(parsed_args)
    elif parsed_args.command == "link-file-to-speaker":
        success = link_file_to_speaker(parsed_args)
    elif parsed_args.command == "unlink-file-from-speaker":
        success = unlink_file_from_speaker(parsed_args)
    elif parsed_args.command == "list-speaker-files":
        success = list_speaker_files(parsed_args)
    elif parsed_args.command == "list-files":
        success = list_processed_files(parsed_args)
    elif parsed_args.command == "list-performance":
        success = list_model_performance(parsed_args)
    elif parsed_args.command == "backup":
        success = backup_database_cmd(parsed_args)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())