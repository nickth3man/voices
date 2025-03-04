#!/usr/bin/env python3
"""
Command-line interface for the metadata manager.

This module provides a command-line interface for extracting, storing,
retrieving, and searching metadata for audio files.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from backend.storage.database.db_manager import DatabaseManager
from backend.storage.files.file_manager import FileManager
from backend.storage.metadata.metadata_manager import MetadataManager
from backend.storage.database.models import ProcessedFile


def extract_metadata(args: argparse.Namespace) -> None:
    """
    Extract metadata from an audio file.
    
    Args:
        args: Command-line arguments
    """
    # Initialize managers
    db_manager = DatabaseManager()
    file_manager = FileManager(db_manager)
    metadata_manager = MetadataManager(db_manager, file_manager)
    
    # Extract metadata
    metadata = metadata_manager.extract_metadata(
        args.file_path,
        extract_audio_characteristics=not args.basic
    )
    
    # Print metadata
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {args.output}")
    else:
        print(json.dumps(metadata, indent=2))


def store_metadata(args: argparse.Namespace) -> None:
    """
    Store metadata for a file.
    
    Args:
        args: Command-line arguments
    """
    # Initialize managers
    db_manager = DatabaseManager()
    file_manager = FileManager(db_manager)
    metadata_manager = MetadataManager(db_manager, file_manager)
    
    # Get file ID
    file_id = args.file_id
    
    # Load metadata from file
    with open(args.metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Store metadata
    result = metadata_manager.store_metadata(file_id, metadata)
    
    # Print result
    if result:
        print(f"Metadata stored successfully for file ID {file_id}")
    else:
        print(f"Failed to store metadata for file ID {file_id}")


def get_metadata(args: argparse.Namespace) -> None:
    """
    Get metadata for a file.
    
    Args:
        args: Command-line arguments
    """
    # Initialize managers
    db_manager = DatabaseManager()
    file_manager = FileManager(db_manager)
    metadata_manager = MetadataManager(db_manager, file_manager)
    
    # Get file ID
    file_id = args.file_id
    
    # Get metadata
    metadata = metadata_manager.get_metadata(file_id)
    
    # Print metadata
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {args.output}")
    else:
        print(json.dumps(metadata, indent=2))


def add_custom_metadata(args: argparse.Namespace) -> None:
    """
    Add custom metadata for a file.
    
    Args:
        args: Command-line arguments
    """
    # Initialize managers
    db_manager = DatabaseManager()
    file_manager = FileManager(db_manager)
    metadata_manager = MetadataManager(db_manager, file_manager)
    
    # Get file ID
    file_id = args.file_id
    
    # Add custom metadata
    result = metadata_manager.add_custom_metadata(
        file_id,
        args.field_name,
        args.field_value,
        args.field_type
    )
    
    # Print result
    if result:
        print(f"Custom metadata added successfully for file ID {file_id}")
    else:
        print(f"Failed to add custom metadata for file ID {file_id}")


def remove_custom_metadata(args: argparse.Namespace) -> None:
    """
    Remove custom metadata for a file.
    
    Args:
        args: Command-line arguments
    """
    # Initialize managers
    db_manager = DatabaseManager()
    file_manager = FileManager(db_manager)
    metadata_manager = MetadataManager(db_manager, file_manager)
    
    # Get file ID
    file_id = args.file_id
    
    # Remove custom metadata
    result = metadata_manager.remove_custom_metadata(
        file_id,
        args.field_name
    )
    
    # Print result
    if result:
        print(f"Custom metadata removed successfully for file ID {file_id}")
    else:
        print(f"Failed to remove custom metadata for file ID {file_id}")


def search_by_metadata(args: argparse.Namespace) -> None:
    """
    Search for files based on metadata criteria.
    
    Args:
        args: Command-line arguments
    """
    # Initialize managers
    db_manager = DatabaseManager()
    file_manager = FileManager(db_manager)
    metadata_manager = MetadataManager(db_manager, file_manager)
    
    # Build criteria
    criteria = {}
    
    if args.filename:
        criteria['filename'] = args.filename
    
    if args.file_format:
        criteria['file_format'] = args.file_format
    
    if args.duration_min is not None:
        criteria['duration_min'] = args.duration_min
    
    if args.duration_max is not None:
        criteria['duration_max'] = args.duration_max
    
    if args.created_after:
        criteria['created_after'] = args.created_after
    
    if args.created_before:
        criteria['created_before'] = args.created_before
    
    # Search by metadata
    results = metadata_manager.search_by_metadata(
        criteria,
        limit=args.limit,
        offset=args.offset
    )
    
    # Print results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Search results saved to {args.output}")
    else:
        print(json.dumps(results, indent=2))


def export_metadata(args: argparse.Namespace) -> None:
    """
    Export metadata for a file to a JSON file.
    
    Args:
        args: Command-line arguments
    """
    # Initialize managers
    db_manager = DatabaseManager()
    file_manager = FileManager(db_manager)
    metadata_manager = MetadataManager(db_manager, file_manager)
    
    # Get file ID
    file_id = args.file_id
    
    # Export metadata
    result = metadata_manager.export_metadata(file_id, args.output)
    
    # Print result
    if isinstance(result, dict) and 'error' in result:
        print(f"Failed to export metadata: {result['error']}")
    else:
        print(f"Metadata exported successfully to {args.output}")


def import_metadata(args: argparse.Namespace) -> None:
    """
    Import metadata for a file from a JSON file.
    
    Args:
        args: Command-line arguments
    """
    # Initialize managers
    db_manager = DatabaseManager()
    file_manager = FileManager(db_manager)
    metadata_manager = MetadataManager(db_manager, file_manager)
    
    # Get file ID
    file_id = args.file_id
    
    # Import metadata
    result = metadata_manager.import_metadata(args.input, file_id)
    
    # Print result
    if result:
        print(f"Metadata imported successfully for file ID {file_id}")
    else:
        print(f"Failed to import metadata for file ID {file_id}")


def batch_extract_metadata(args: argparse.Namespace) -> None:
    """
    Extract metadata for all audio files in a directory.
    
    Args:
        args: Command-line arguments
    """
    # Initialize managers
    db_manager = DatabaseManager()
    file_manager = FileManager(db_manager)
    metadata_manager = MetadataManager(db_manager, file_manager)
    
    # Batch extract metadata
    results = metadata_manager.batch_extract_metadata(
        args.directory,
        recursive=args.recursive
    )
    
    # Print results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Batch extraction results saved to {args.output}")
    else:
        print(json.dumps(results, indent=2))


def get_metadata_statistics(args: argparse.Namespace) -> None:
    """
    Get statistics about metadata in the database.
    
    Args:
        args: Command-line arguments
    """
    # Initialize managers
    db_manager = DatabaseManager()
    file_manager = FileManager(db_manager)
    metadata_manager = MetadataManager(db_manager, file_manager)
    
    # Get metadata statistics
    statistics = metadata_manager.get_metadata_statistics()
    
    # Print statistics
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(statistics, f, indent=2)
        print(f"Metadata statistics saved to {args.output}")
    else:
        print(json.dumps(statistics, indent=2))


def list_files(args: argparse.Namespace) -> None:
    """
    List files in the database.
    
    Args:
        args: Command-line arguments
    """
    # Initialize managers
    db_manager = DatabaseManager()
    
    # Get files
    query = db_manager.query(ProcessedFile)
    
    if args.filename:
        query = query.filter(ProcessedFile.filename.like(f"%{args.filename}%"))
    
    if args.file_format:
        query = query.filter(ProcessedFile.file_format == args.file_format)
    
    files = query.all()
    
    # Format results
    results = []
    for file in files:
        results.append({
            'id': file.id,
            'filename': file.filename,
            'file_format': file.file_format,
            'original_path': file.original_path,
            'processed_path': file.processed_path,
            'created_at': file.created_at.isoformat() if file.created_at else None,
            'updated_at': file.updated_at.isoformat() if file.updated_at else None
        })
    
    # Print results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"File list saved to {args.output}")
    else:
        print(json.dumps(results, indent=2))


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description='Metadata Manager CLI')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Extract metadata
    extract_parser = subparsers.add_parser('extract', help='Extract metadata from an audio file')
    extract_parser.add_argument('file_path', help='Path to the audio file')
    extract_parser.add_argument('--basic', action='store_true', help='Extract only basic metadata')
    extract_parser.add_argument('--output', '-o', help='Output file for metadata')
    extract_parser.set_defaults(func=extract_metadata)
    
    # Store metadata
    store_parser = subparsers.add_parser('store', help='Store metadata for a file')
    store_parser.add_argument('file_id', type=int, help='ID of the file')
    store_parser.add_argument('metadata_file', help='Path to the metadata file')
    store_parser.set_defaults(func=store_metadata)
    
    # Get metadata
    get_parser = subparsers.add_parser('get', help='Get metadata for a file')
    get_parser.add_argument('file_id', type=int, help='ID of the file')
    get_parser.add_argument('--output', '-o', help='Output file for metadata')
    get_parser.set_defaults(func=get_metadata)
    
    # Add custom metadata
    add_custom_parser = subparsers.add_parser('add-custom', help='Add custom metadata for a file')
    add_custom_parser.add_argument('file_id', type=int, help='ID of the file')
    add_custom_parser.add_argument('field_name', help='Name of the custom field')
    add_custom_parser.add_argument('field_value', help='Value for the custom field')
    add_custom_parser.add_argument('--field-type', default='text', choices=['text', 'number', 'boolean', 'date'],
                                  help='Type of the custom field')
    add_custom_parser.set_defaults(func=add_custom_metadata)
    
    # Remove custom metadata
    remove_custom_parser = subparsers.add_parser('remove-custom', help='Remove custom metadata for a file')
    remove_custom_parser.add_argument('file_id', type=int, help='ID of the file')
    remove_custom_parser.add_argument('field_name', help='Name of the custom field')
    remove_custom_parser.set_defaults(func=remove_custom_metadata)
    
    # Search by metadata
    search_parser = subparsers.add_parser('search', help='Search for files based on metadata criteria')
    search_parser.add_argument('--filename', help='Filename to search for')
    search_parser.add_argument('--file-format', help='File format to search for')
    search_parser.add_argument('--duration-min', type=float, help='Minimum duration in seconds')
    search_parser.add_argument('--duration-max', type=float, help='Maximum duration in seconds')
    search_parser.add_argument('--created-after', help='Created after date (ISO format)')
    search_parser.add_argument('--created-before', help='Created before date (ISO format)')
    search_parser.add_argument('--limit', type=int, default=100, help='Maximum number of results')
    search_parser.add_argument('--offset', type=int, default=0, help='Offset for pagination')
    search_parser.add_argument('--output', '-o', help='Output file for search results')
    search_parser.set_defaults(func=search_by_metadata)
    
    # Export metadata
    export_parser = subparsers.add_parser('export', help='Export metadata for a file to a JSON file')
    export_parser.add_argument('file_id', type=int, help='ID of the file')
    export_parser.add_argument('output', help='Output file for metadata')
    export_parser.set_defaults(func=export_metadata)
    
    # Import metadata
    import_parser = subparsers.add_parser('import', help='Import metadata for a file from a JSON file')
    import_parser.add_argument('file_id', type=int, help='ID of the file')
    import_parser.add_argument('input', help='Input file for metadata')
    import_parser.set_defaults(func=import_metadata)
    
    # Batch extract metadata
    batch_extract_parser = subparsers.add_parser('batch-extract', help='Extract metadata for all audio files in a directory')
    batch_extract_parser.add_argument('directory', help='Directory to scan for audio files')
    batch_extract_parser.add_argument('--recursive', '-r', action='store_true', help='Scan subdirectories')
    batch_extract_parser.add_argument('--output', '-o', help='Output file for batch extraction results')
    batch_extract_parser.set_defaults(func=batch_extract_metadata)
    
    # Get metadata statistics
    statistics_parser = subparsers.add_parser('statistics', help='Get statistics about metadata in the database')
    statistics_parser.add_argument('--output', '-o', help='Output file for metadata statistics')
    statistics_parser.set_defaults(func=get_metadata_statistics)
    
    # List files
    list_parser = subparsers.add_parser('list-files', help='List files in the database')
    list_parser.add_argument('--filename', help='Filter by filename')
    list_parser.add_argument('--file-format', help='Filter by file format')
    list_parser.add_argument('--output', '-o', help='Output file for file list')
    list_parser.set_defaults(func=list_files)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()