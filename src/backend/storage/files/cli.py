"""
Command-line interface for the file organization system.

This module provides a CLI for managing file organization, including
storage locations, naming conventions, automatic organization, and
storage usage monitoring.
"""

import os
import sys
import argparse
import logging
import json
from typing import List, Dict, Any, Optional

from ..database.db_manager import DatabaseManager
from .file_manager import FileManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_parser() -> argparse.ArgumentParser:
    """
    Set up the argument parser for the CLI.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="File Organization System for Voices Application",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Initialize command
    init_parser = subparsers.add_parser("init", help="Initialize file organization system")
    init_parser.add_argument("--base-dir", help="Base directory for file storage")
    init_parser.add_argument("--config", help="Path to configuration file")
    
    # Store command
    store_parser = subparsers.add_parser("store", help="Store a file")
    store_parser.add_argument("file", help="Path to the file to store")
    store_parser.add_argument("--dir-type", default="original", 
                             choices=["original", "processed", "temp", "archive", "exports"],
                             help="Type of storage directory")
    store_parser.add_argument("--filename", help="New filename (optional)")
    store_parser.add_argument("--speaker", help="Speaker name for filename generation")
    store_parser.add_argument("--processing-type", help="Processing type for filename generation")
    store_parser.add_argument("--register", action="store_true", help="Register file in database")
    
    # Move command
    move_parser = subparsers.add_parser("move", help="Move a file")
    move_parser.add_argument("file", help="Path to the file to move")
    move_parser.add_argument("--dir-type", required=True,
                            choices=["original", "processed", "temp", "archive", "exports"],
                            help="Type of storage directory")
    move_parser.add_argument("--filename", help="New filename (optional)")
    move_parser.add_argument("--speaker", help="Speaker name for filename generation")
    move_parser.add_argument("--processing-type", help="Processing type for filename generation")
    move_parser.add_argument("--register", action="store_true", help="Register file in database")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a file")
    delete_parser.add_argument("file", help="Path to the file to delete")
    
    # Archive command
    archive_parser = subparsers.add_parser("archive", help="Archive a file")
    archive_parser.add_argument("file", help="Path to the file to archive")
    archive_parser.add_argument("--filename", help="New filename (optional)")
    archive_parser.add_argument("--speaker", help="Speaker name for filename generation")
    
    # Organize command
    organize_parser = subparsers.add_parser("organize", help="Organize files")
    organize_parser.add_argument("--dir-type", 
                                choices=["original", "processed", "temp", "archive", "exports"],
                                help="Type of storage directory to organize (optional)")
    
    # Usage command
    usage_parser = subparsers.add_parser("usage", help="Get storage usage information")
    usage_parser.add_argument("--dir-type", 
                             choices=["original", "processed", "temp", "archive", "exports"],
                             help="Type of storage directory to check (optional)")
    usage_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # Find command
    find_parser = subparsers.add_parser("find", help="Find files matching a pattern")
    find_parser.add_argument("pattern", help="Glob pattern to match files")
    find_parser.add_argument("--dir-type", 
                            choices=["original", "processed", "temp", "archive", "exports"],
                            help="Type of storage directory to search (optional)")
    
    # Register command
    register_parser = subparsers.add_parser("register", help="Register a file in the database")
    register_parser.add_argument("file", help="Path to the file to register")
    register_parser.add_argument("--original", help="Path to the original file (optional)")
    register_parser.add_argument("--speaker-id", type=int, help="ID of the associated speaker (optional)")
    register_parser.add_argument("--metadata", help="JSON metadata for the file (optional)")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get information about a registered file")
    info_parser.add_argument("file_id", type=int, help="ID of the file")
    info_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all registered files")
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export a file")
    export_parser.add_argument("file_id", type=int, help="ID of the file to export")
    export_parser.add_argument("--format", help="Format to export to (optional)")
    export_parser.add_argument("--path", help="Path to export to (optional)")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up temporary files")
    cleanup_parser.add_argument("--days", type=int, default=7, help="Maximum age of files in days")
    
    # Config commands
    config_subparsers = subparsers.add_parser("config", help="Configuration commands")
    config_subparsers = config_subparsers.add_subparsers(dest="config_command", help="Configuration command to execute")
    
    # Save config command
    save_config_parser = config_subparsers.add_parser("save", help="Save configuration")
    save_config_parser.add_argument("--path", help="Path to save the configuration to (optional)")
    
    # Load config command
    load_config_parser = config_subparsers.add_parser("load", help="Load configuration")
    load_config_parser.add_argument("path", help="Path to the configuration file")
    
    return parser


def init_command(args: argparse.Namespace, db_manager: DatabaseManager) -> int:
    """
    Initialize the file organization system.
    
    Args:
        args: Command-line arguments.
        db_manager: Database manager instance.
        
    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    try:
        # Load configuration if provided
        config = None
        if args.config:
            config = FileManager.load_config(args.config)
            if not config:
                logger.error(f"Failed to load configuration from {args.config}")
                return 1
        
        # Initialize file manager
        file_manager = FileManager(db_manager, args.base_dir, config)
        
        # Save configuration
        file_manager.save_config()
        
        print(f"File organization system initialized in {file_manager.base_dir}")
        return 0
    except Exception as e:
        logger.error(f"Error initializing file organization system: {str(e)}")
        return 1


def store_command(args: argparse.Namespace, file_manager: FileManager) -> int:
    """
    Store a file.
    
    Args:
        args: Command-line arguments.
        file_manager: File manager instance.
        
    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    try:
        # Store file
        dest_path = file_manager.store_file(
            args.file,
            args.dir_type,
            args.filename,
            args.speaker,
            args.processing_type
        )
        
        if not dest_path:
            logger.error(f"Failed to store file: {args.file}")
            return 1
        
        print(f"Stored file: {args.file} -> {dest_path}")
        
        # Register file if requested
        if args.register:
            file_id = file_manager.register_file(
                dest_path,
                args.file
            )
            
            if file_id:
                print(f"Registered file with ID: {file_id}")
            else:
                logger.error(f"Failed to register file: {dest_path}")
                return 1
        
        return 0
    except Exception as e:
        logger.error(f"Error storing file: {str(e)}")
        return 1


def move_command(args: argparse.Namespace, file_manager: FileManager) -> int:
    """
    Move a file.
    
    Args:
        args: Command-line arguments.
        file_manager: File manager instance.
        
    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    try:
        # Move file
        dest_path = file_manager.move_file(
            args.file,
            args.dir_type,
            args.filename,
            args.speaker,
            args.processing_type
        )
        
        if not dest_path:
            logger.error(f"Failed to move file: {args.file}")
            return 1
        
        print(f"Moved file: {args.file} -> {dest_path}")
        
        # Register file if requested
        if args.register:
            file_id = file_manager.register_file(
                dest_path,
                args.file
            )
            
            if file_id:
                print(f"Registered file with ID: {file_id}")
            else:
                logger.error(f"Failed to register file: {dest_path}")
                return 1
        
        return 0
    except Exception as e:
        logger.error(f"Error moving file: {str(e)}")
        return 1


def delete_command(args: argparse.Namespace, file_manager: FileManager) -> int:
    """
    Delete a file.
    
    Args:
        args: Command-line arguments.
        file_manager: File manager instance.
        
    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    try:
        # Delete file
        success = file_manager.delete_file(args.file)
        
        if not success:
            logger.error(f"Failed to delete file: {args.file}")
            return 1
        
        print(f"Deleted file: {args.file}")
        return 0
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        return 1


def archive_command(args: argparse.Namespace, file_manager: FileManager) -> int:
    """
    Archive a file.
    
    Args:
        args: Command-line arguments.
        file_manager: File manager instance.
        
    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    try:
        # Archive file
        dest_path = file_manager.archive_file(
            args.file,
            args.filename,
            args.speaker
        )
        
        if not dest_path:
            logger.error(f"Failed to archive file: {args.file}")
            return 1
        
        print(f"Archived file: {args.file} -> {dest_path}")
        return 0
    except Exception as e:
        logger.error(f"Error archiving file: {str(e)}")
        return 1


def organize_command(args: argparse.Namespace, file_manager: FileManager) -> int:
    """
    Organize files.
    
    Args:
        args: Command-line arguments.
        file_manager: File manager instance.
        
    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    try:
        # Organize files
        success = file_manager.organize_files(args.dir_type)
        
        if not success:
            logger.error("Failed to organize files")
            return 1
        
        if args.dir_type:
            print(f"Organized files in {args.dir_type} directory")
        else:
            print("Organized files in all directories")
        
        return 0
    except Exception as e:
        logger.error(f"Error organizing files: {str(e)}")
        return 1


def usage_command(args: argparse.Namespace, file_manager: FileManager) -> int:
    """
    Get storage usage information.
    
    Args:
        args: Command-line arguments.
        file_manager: File manager instance.
        
    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    try:
        # Get storage usage
        usage_info = file_manager.get_storage_usage(args.dir_type)
        
        if "error" in usage_info:
            logger.error(f"Failed to get storage usage: {usage_info['error']}")
            return 1
        
        # Output as JSON if requested
        if args.json:
            print(json.dumps(usage_info, indent=2, default=str))
            return 0
        
        # Output as formatted text
        print("Storage Usage Information:")
        print(f"Total Size: {usage_info['total_size_formatted']}")
        print(f"Total Files: {usage_info['file_count']}")
        print(f"Disk Space: {usage_info['disk_used_formatted']} / {usage_info['disk_total_formatted']} ({usage_info['disk_percent']}% used)")
        print(f"Free Space: {usage_info['disk_free_formatted']}")
        print("\nDirectory Breakdown:")
        
        for dir_type, dir_info in usage_info["directories"].items():
            print(f"  {dir_type}: {dir_info['size_formatted']} ({dir_info['file_count']} files)")
        
        return 0
    except Exception as e:
        logger.error(f"Error getting storage usage: {str(e)}")
        return 1


def find_command(args: argparse.Namespace, file_manager: FileManager) -> int:
    """
    Find files matching a pattern.
    
    Args:
        args: Command-line arguments.
        file_manager: File manager instance.
        
    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    try:
        # Find files
        matching_files = file_manager.find_files(args.pattern, args.dir_type)
        
        if not matching_files:
            print(f"No files found matching pattern: {args.pattern}")
            return 0
        
        print(f"Found {len(matching_files)} files matching pattern: {args.pattern}")
        for file_path in matching_files:
            print(f"  {file_path}")
        
        return 0
    except Exception as e:
        logger.error(f"Error finding files: {str(e)}")
        return 1


def register_command(args: argparse.Namespace, file_manager: FileManager) -> int:
    """
    Register a file in the database.
    
    Args:
        args: Command-line arguments.
        file_manager: File manager instance.
        
    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    try:
        # Parse metadata if provided
        metadata = None
        if args.metadata:
            try:
                metadata = json.loads(args.metadata)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON metadata: {args.metadata}")
                return 1
        
        # Register file
        file_id = file_manager.register_file(
            args.file,
            args.original,
            args.speaker_id,
            metadata
        )
        
        if not file_id:
            logger.error(f"Failed to register file: {args.file}")
            return 1
        
        print(f"Registered file with ID: {file_id}")
        return 0
    except Exception as e:
        logger.error(f"Error registering file: {str(e)}")
        return 1


def info_command(args: argparse.Namespace, file_manager: FileManager) -> int:
    """
    Get information about a registered file.
    
    Args:
        args: Command-line arguments.
        file_manager: File manager instance.
        
    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    try:
        # Get file info
        file_info = file_manager.get_file_info(args.file_id)
        
        if not file_info:
            logger.error(f"File not found: ID={args.file_id}")
            return 1
        
        # Output as JSON if requested
        if args.json:
            print(json.dumps(file_info, indent=2, default=str))
            return 0
        
        # Output as formatted text
        print(f"File Information (ID: {file_info['id']}):")
        print(f"  Filename: {file_info['filename']}")
        print(f"  Format: {file_info['file_format']}")
        print(f"  Size: {file_info['file_size_formatted']}")
        print(f"  Duration: {file_info['duration'] or 'Unknown'}")
        print(f"  Sample Rate: {file_info['sample_rate'] or 'Unknown'}")
        print(f"  Channels: {file_info['channels'] or 'Unknown'}")
        print(f"  Created: {file_info['created_at']}")
        print(f"  Updated: {file_info['updated_at']}")
        print(f"  Speaker ID: {file_info['speaker_id'] or 'None'}")
        print(f"  Original Path: {file_info['original_path']}")
        print(f"  Processed Path: {file_info['processed_path']}")
        print(f"  File Exists: {file_info['exists']}")
        
        if file_info['metadata']:
            print("\n  Metadata:")
            for key, value in file_info['metadata'].items():
                print(f"    {key}: {value}")
        
        return 0
    except Exception as e:
        logger.error(f"Error getting file info: {str(e)}")
        return 1


def list_command(args: argparse.Namespace, file_manager: FileManager) -> int:
    """
    List all registered files.
    
    Args:
        args: Command-line arguments.
        file_manager: File manager instance.
        
    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    try:
        # Get all files
        files = file_manager.get_all_files()
        
        if not files:
            print("No registered files found")
            return 0
        
        # Output as JSON if requested
        if args.json:
            print(json.dumps(files, indent=2, default=str))
            return 0
        
        # Output as formatted text
        print(f"Registered Files ({len(files)}):")
        for file_info in files:
            print(f"  ID: {file_info['id']}, Name: {file_info['filename']}, Format: {file_info['file_format']}, Size: {file_info['file_size_formatted']}")
        
        return 0
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        return 1


def export_command(args: argparse.Namespace, file_manager: FileManager) -> int:
    """
    Export a file.
    
    Args:
        args: Command-line arguments.
        file_manager: File manager instance.
        
    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    try:
        # Export file
        export_path = file_manager.export_file(
            args.file_id,
            args.format,
            args.path
        )
        
        if not export_path:
            logger.error(f"Failed to export file: ID={args.file_id}")
            return 1
        
        print(f"Exported file: ID={args.file_id} -> {export_path}")
        return 0
    except Exception as e:
        logger.error(f"Error exporting file: {str(e)}")
        return 1


def cleanup_command(args: argparse.Namespace, file_manager: FileManager) -> int:
    """
    Clean up temporary files.
    
    Args:
        args: Command-line arguments.
        file_manager: File manager instance.
        
    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    try:
        # Clean up temporary files
        deleted_count = file_manager.cleanup_temp_files(args.days)
        
        print(f"Cleaned up {deleted_count} temporary files")
        return 0
    except Exception as e:
        logger.error(f"Error cleaning up temporary files: {str(e)}")
        return 1


def save_config_command(args: argparse.Namespace, file_manager: FileManager) -> int:
    """
    Save configuration.
    
    Args:
        args: Command-line arguments.
        file_manager: File manager instance.
        
    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    try:
        # Save configuration
        success = file_manager.save_config(args.path)
        
        if not success:
            logger.error("Failed to save configuration")
            return 1
        
        print(f"Saved configuration to {args.path or os.path.join(file_manager.base_dir, 'file_manager_config.json')}")
        return 0
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")
        return 1


def load_config_command(args: argparse.Namespace, file_manager: FileManager) -> int:
    """
    Load configuration.
    
    Args:
        args: Command-line arguments.
        file_manager: File manager instance.
        
    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    try:
        # Load configuration
        config = FileManager.load_config(args.path)
        
        if not config:
            logger.error(f"Failed to load configuration from {args.path}")
            return 1
        
        print(f"Loaded configuration from {args.path}")
        print(json.dumps(config, indent=2))
        return 0
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return 1


def main() -> int:
    """
    Main entry point for the CLI.
    
    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager()
        if not db_manager.initialize():
            logger.error("Failed to initialize database")
            return 1
        
        # Handle init command separately
        if args.command == "init":
            return init_command(args, db_manager)
        
        # For all other commands, initialize file manager
        file_manager = FileManager(db_manager)
        
        # Dispatch command
        if args.command == "store":
            return store_command(args, file_manager)
        elif args.command == "move":
            return move_command(args, file_manager)
        elif args.command == "delete":
            return delete_command(args, file_manager)
        elif args.command == "archive":
            return archive_command(args, file_manager)
        elif args.command == "organize":
            return organize_command(args, file_manager)
        elif args.command == "usage":
            return usage_command(args, file_manager)
        elif args.command == "find":
            return find_command(args, file_manager)
        elif args.command == "register":
            return register_command(args, file_manager)
        elif args.command == "info":
            return info_command(args, file_manager)
        elif args.command == "list":
            return list_command(args, file_manager)
        elif args.command == "export":
            return export_command(args, file_manager)
        elif args.command == "cleanup":
            return cleanup_command(args, file_manager)
        elif args.command == "config":
            if args.config_command == "save":
                return save_config_command(args, file_manager)
            elif args.config_command == "load":
                return load_config_command(args, file_manager)
            else:
                logger.error(f"Unknown config command: {args.config_command}")
                return 1
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
    except Exception as e:
        logger.error(f"Error executing command: {str(e)}")
        return 1
    finally:
        # Clean up
        if 'db_manager' in locals():
            db_manager.close_session()


if __name__ == "__main__":
    sys.exit(main())