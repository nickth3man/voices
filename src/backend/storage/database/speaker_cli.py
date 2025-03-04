"""
Command-line interface for the Speaker Management System.

This module provides a CLI for managing speaker profiles, including CRUD operations,
tagging system, and linking to processed audio files.
"""

import argparse
import sys
import json
from typing import List, Optional

from .db_manager import DatabaseManager
from .speaker_manager import SpeakerManager


def setup_parser() -> argparse.ArgumentParser:
    """Set up the argument parser for the speaker CLI."""
    parser = argparse.ArgumentParser(
        description="Speaker Management System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create speaker
    create_parser = subparsers.add_parser("create", help="Create a new speaker profile")
    create_parser.add_argument("name", help="Name of the speaker")
    create_parser.add_argument("--description", "-d", help="Description of the speaker")
    create_parser.add_argument("--tags", "-t", nargs="+", help="Tags for the speaker")
    
    # Get speaker
    get_parser = subparsers.add_parser("get", help="Get a speaker profile")
    get_parser.add_argument("id", type=int, help="ID of the speaker")
    
    # List speakers
    list_parser = subparsers.add_parser("list", help="List all speaker profiles")
    
    # Update speaker
    update_parser = subparsers.add_parser("update", help="Update a speaker profile")
    update_parser.add_argument("id", type=int, help="ID of the speaker")
    update_parser.add_argument("--name", "-n", help="New name for the speaker")
    update_parser.add_argument("--description", "-d", help="New description for the speaker")
    
    # Delete speaker
    delete_parser = subparsers.add_parser("delete", help="Delete a speaker profile")
    delete_parser.add_argument("id", type=int, help="ID of the speaker")
    
    # Add tags
    add_tags_parser = subparsers.add_parser("add-tags", help="Add tags to a speaker")
    add_tags_parser.add_argument("id", type=int, help="ID of the speaker")
    add_tags_parser.add_argument("tags", nargs="+", help="Tags to add")
    
    # Remove tags
    remove_tags_parser = subparsers.add_parser("remove-tags", help="Remove tags from a speaker")
    remove_tags_parser.add_argument("id", type=int, help="ID of the speaker")
    remove_tags_parser.add_argument("tags", nargs="+", help="Tags to remove")
    
    # Get tags
    get_tags_parser = subparsers.add_parser("get-tags", help="Get all tags for a speaker")
    get_tags_parser.add_argument("id", type=int, help="ID of the speaker")
    
    # List all tags
    list_tags_parser = subparsers.add_parser("list-tags", help="List all unique tags used across all speakers")
    
    # Find speakers by tags
    find_by_tags_parser = subparsers.add_parser("find-by-tags", help="Find speakers by tags")
    find_by_tags_parser.add_argument("tags", nargs="+", help="Tags to search for")
    find_by_tags_parser.add_argument("--match-all", "-a", action="store_true", 
                                    help="If set, speakers must have all tags. Otherwise, speakers must have any of the tags.")
    
    # Search speakers
    search_parser = subparsers.add_parser("search", help="Search for speakers by name or description")
    search_parser.add_argument("query", help="Search query string")
    
    # Link processed file
    link_file_parser = subparsers.add_parser("link-file", help="Link a processed file to a speaker")
    link_file_parser.add_argument("speaker_id", type=int, help="ID of the speaker")
    link_file_parser.add_argument("file_id", type=int, help="ID of the processed file")
    
    # Unlink processed file
    unlink_file_parser = subparsers.add_parser("unlink-file", help="Unlink a processed file from its speaker")
    unlink_file_parser.add_argument("file_id", type=int, help="ID of the processed file")
    
    # Get processed files
    get_files_parser = subparsers.add_parser("get-files", help="Get all processed files for a speaker")
    get_files_parser.add_argument("id", type=int, help="ID of the speaker")
    
    return parser


def format_speaker(speaker) -> str:
    """Format a speaker object as a string."""
    if not speaker:
        return "Speaker not found"
    
    return (
        f"ID: {speaker.id}\n"
        f"Name: {speaker.name}\n"
        f"Description: {speaker.description or 'N/A'}\n"
        f"Tags: {', '.join(speaker.tags) if speaker.tags else 'None'}\n"
        f"Created: {speaker.created_at}\n"
        f"Updated: {speaker.updated_at}"
    )


def format_processed_file(file) -> str:
    """Format a processed file object as a string."""
    if not file:
        return "File not found"
    
    return (
        f"ID: {file.id}\n"
        f"Filename: {file.filename}\n"
        f"Original Path: {file.original_path}\n"
        f"Processed Path: {file.processed_path}\n"
        f"Format: {file.file_format}\n"
        f"Duration: {file.duration or 'N/A'} seconds\n"
        f"Sample Rate: {file.sample_rate or 'N/A'} Hz\n"
        f"Channels: {file.channels or 'N/A'}\n"
        f"Speaker ID: {file.speaker_id or 'None'}\n"
        f"Created: {file.created_at}\n"
        f"Updated: {file.updated_at}"
    )


def main():
    """Main entry point for the speaker CLI."""
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize database manager and speaker manager
    db_manager = DatabaseManager()
    speaker_manager = SpeakerManager(db_manager)
    
    # Execute command
    if args.command == "create":
        speaker = speaker_manager.create_speaker(
            name=args.name,
            description=args.description,
            tags=args.tags
        )
        if speaker:
            print(f"Speaker created successfully:")
            print(format_speaker(speaker))
        else:
            print("Failed to create speaker")
    
    elif args.command == "get":
        speaker = speaker_manager.get_speaker(args.id)
        if speaker:
            print(format_speaker(speaker))
        else:
            print(f"Speaker with ID {args.id} not found")
    
    elif args.command == "list":
        speakers = speaker_manager.get_all_speakers()
        if speakers:
            print(f"Found {len(speakers)} speakers:")
            for speaker in speakers:
                print("\n" + format_speaker(speaker))
        else:
            print("No speakers found")
    
    elif args.command == "update":
        if not (args.name or args.description):
            print("Error: At least one of --name or --description must be provided")
            return
        
        success = speaker_manager.update_speaker(
            speaker_id=args.id,
            name=args.name,
            description=args.description
        )
        if success:
            print(f"Speaker with ID {args.id} updated successfully")
            speaker = speaker_manager.get_speaker(args.id)
            print(format_speaker(speaker))
        else:
            print(f"Failed to update speaker with ID {args.id}")
    
    elif args.command == "delete":
        success = speaker_manager.delete_speaker(args.id)
        if success:
            print(f"Speaker with ID {args.id} deleted successfully")
        else:
            print(f"Failed to delete speaker with ID {args.id}")
    
    elif args.command == "add-tags":
        success = speaker_manager.add_tags(args.id, args.tags)
        if success:
            print(f"Tags added to speaker with ID {args.id}")
            tags = speaker_manager.get_tags(args.id)
            print(f"Current tags: {', '.join(tags) if tags else 'None'}")
        else:
            print(f"Failed to add tags to speaker with ID {args.id}")
    
    elif args.command == "remove-tags":
        success = speaker_manager.remove_tags(args.id, args.tags)
        if success:
            print(f"Tags removed from speaker with ID {args.id}")
            tags = speaker_manager.get_tags(args.id)
            print(f"Current tags: {', '.join(tags) if tags else 'None'}")
        else:
            print(f"Failed to remove tags from speaker with ID {args.id}")
    
    elif args.command == "get-tags":
        tags = speaker_manager.get_tags(args.id)
        if tags:
            print(f"Tags for speaker with ID {args.id}:")
            print(', '.join(tags))
        else:
            print(f"No tags found for speaker with ID {args.id}")
    
    elif args.command == "list-tags":
        tags = speaker_manager.get_all_tags()
        if tags:
            print(f"All unique tags ({len(tags)}):")
            print(', '.join(sorted(tags)))
        else:
            print("No tags found")
    
    elif args.command == "find-by-tags":
        speakers = speaker_manager.find_speakers_by_tags(args.tags, args.match_all)
        if speakers:
            match_type = "all" if args.match_all else "any"
            print(f"Found {len(speakers)} speakers matching {match_type} of the tags {', '.join(args.tags)}:")
            for speaker in speakers:
                print("\n" + format_speaker(speaker))
        else:
            match_type = "all" if args.match_all else "any"
            print(f"No speakers found matching {match_type} of the tags {', '.join(args.tags)}")
    
    elif args.command == "search":
        speakers = speaker_manager.search_speakers(args.query)
        if speakers:
            print(f"Found {len(speakers)} speakers matching '{args.query}':")
            for speaker in speakers:
                print("\n" + format_speaker(speaker))
        else:
            print(f"No speakers found matching '{args.query}'")
    
    elif args.command == "link-file":
        success = speaker_manager.link_processed_file(args.speaker_id, args.file_id)
        if success:
            print(f"File with ID {args.file_id} linked to speaker with ID {args.speaker_id}")
        else:
            print(f"Failed to link file with ID {args.file_id} to speaker with ID {args.speaker_id}")
    
    elif args.command == "unlink-file":
        success = speaker_manager.unlink_processed_file(args.file_id)
        if success:
            print(f"File with ID {args.file_id} unlinked from speaker")
        else:
            print(f"Failed to unlink file with ID {args.file_id}")
    
    elif args.command == "get-files":
        files = speaker_manager.get_processed_files(args.id)
        if files:
            print(f"Found {len(files)} files for speaker with ID {args.id}:")
            for file in files:
                print("\n" + format_processed_file(file))
        else:
            print(f"No files found for speaker with ID {args.id}")


if __name__ == "__main__":
    main()