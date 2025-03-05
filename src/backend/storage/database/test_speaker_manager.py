"""
Test script for the Speaker Management System.

This script tests the basic functionality of the SpeakerManager class,
including CRUD operations, tagging system, and linking to processed files.
"""

import os
import sys
import datetime
from typing import List, Optional

# Add parent directory to path to allow importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.backend.storage.database.db_manager import DatabaseManager
from src.backend.storage.database.speaker_manager import SpeakerManager
from src.backend.storage.database.models import Speaker, ProcessedFile


def test_speaker_creation():
    """Test creating a speaker."""
    print("\n=== Testing Speaker Creation ===")
    
    # Initialize database manager and speaker manager
    db_manager = DatabaseManager(db_path=":memory:")
    db_manager.initialize_database()
    speaker_manager = SpeakerManager(db_manager)
    
    # Create a speaker
    speaker = speaker_manager.create_speaker(
        name="Test Speaker",
        description="A test speaker for testing",
        tags=["test", "demo", "voice"]
    )
    
    # Verify speaker was created
    assert speaker is not None, "Speaker creation failed"
    assert speaker.id is not None, "Speaker ID not assigned"
    assert speaker.name == "Test Speaker", f"Speaker name mismatch: {speaker.name}"
    assert speaker.description == "A test speaker for testing", f"Speaker description mismatch: {speaker.description}"
    assert set(speaker.tags) == {"test", "demo", "voice"}, f"Speaker tags mismatch: {speaker.tags}"
    
    print("✓ Speaker creation successful")
    return speaker, speaker_manager, db_manager


def test_speaker_retrieval(speaker_id: int, speaker_manager: SpeakerManager):
    """Test retrieving a speaker."""
    print("\n=== Testing Speaker Retrieval ===")
    
    # Get the speaker
    speaker = speaker_manager.get_speaker(speaker_id)
    
    # Verify speaker was retrieved
    assert speaker is not None, "Speaker retrieval failed"
    assert speaker.id == speaker_id, f"Speaker ID mismatch: {speaker.id} != {speaker_id}"
    assert speaker.name == "Test Speaker", f"Speaker name mismatch: {speaker.name}"
    
    print("✓ Speaker retrieval successful")
    return speaker


def test_speaker_update(speaker_id: int, speaker_manager: SpeakerManager):
    """Test updating a speaker."""
    print("\n=== Testing Speaker Update ===")
    
    # Update the speaker
    success = speaker_manager.update_speaker(
        speaker_id=speaker_id,
        name="Updated Speaker",
        description="An updated test speaker"
    )
    
    # Verify update was successful
    assert success, "Speaker update failed"
    
    # Get the updated speaker
    speaker = speaker_manager.get_speaker(speaker_id)
    
    # Verify speaker was updated
    assert speaker is not None, "Speaker retrieval after update failed"
    assert speaker.name == "Updated Speaker", f"Speaker name not updated: {speaker.name}"
    assert speaker.description == "An updated test speaker", f"Speaker description not updated: {speaker.description}"
    
    print("✓ Speaker update successful")
    return speaker


def test_tag_operations(speaker_id: int, speaker_manager: SpeakerManager):
    """Test tag operations."""
    print("\n=== Testing Tag Operations ===")
    
    # Add tags
    success = speaker_manager.add_tags(speaker_id, ["new", "tags"])
    assert success, "Adding tags failed"
    
    # Get tags
    tags = speaker_manager.get_tags(speaker_id)
    assert "new" in tags, f"Tag 'new' not added: {tags}"
    assert "tags" in tags, f"Tag 'tags' not added: {tags}"
    
    print("✓ Adding tags successful")
    
    # Remove tags
    success = speaker_manager.remove_tags(speaker_id, ["demo"])
    assert success, "Removing tags failed"
    
    # Get tags again
    tags = speaker_manager.get_tags(speaker_id)
    assert "demo" not in tags, f"Tag 'demo' not removed: {tags}"
    
    print("✓ Removing tags successful")
    
    # Get all tags
    all_tags = speaker_manager.get_all_tags()
    assert len(all_tags) > 0, "No tags found"
    
    print(f"✓ All tags retrieved: {', '.join(all_tags)}")
    
    # Find speakers by tags
    speakers = speaker_manager.find_speakers_by_tags(["test"], match_all=False)
    assert len(speakers) > 0, "No speakers found with tag 'test'"
    
    print(f"✓ Found {len(speakers)} speakers with tag 'test'")
    
    return tags


def test_processed_file_operations(speaker_id: int, speaker_manager: SpeakerManager, db_manager: DatabaseManager):
    """Test processed file operations."""
    print("\n=== Testing Processed File Operations ===")
    
    # Create a processed file
    now = datetime.datetime.now()
    file = ProcessedFile(
        filename="test.wav",
        original_path="/path/to/original/test.wav",
        processed_path="/path/to/processed/test.wav",
        file_format="wav",
        duration=120.5,
        sample_rate=44100,
        channels=2,
        created_at=now,
        updated_at=now,
        metadata={"bitrate": 320, "codec": "PCM"}
    )
    
    # Insert into database
    file_id = db_manager.insert(file)
    assert file_id is not None, "File insertion failed"
    file.id = file_id
    
    print(f"✓ Created processed file with ID {file_id}")
    
    # Link file to speaker
    success = speaker_manager.link_processed_file(speaker_id, file_id)
    assert success, "Linking file to speaker failed"
    
    print(f"✓ Linked file {file_id} to speaker {speaker_id}")
    
    # Get processed files for speaker
    files = speaker_manager.get_processed_files(speaker_id)
    assert len(files) > 0, "No files found for speaker"
    assert files[0].id == file_id, f"File ID mismatch: {files[0].id} != {file_id}"
    
    print(f"✓ Retrieved {len(files)} files for speaker {speaker_id}")
    
    # Unlink file from speaker
    success = speaker_manager.unlink_processed_file(file_id)
    assert success, "Unlinking file from speaker failed"
    
    print(f"✓ Unlinked file {file_id} from speaker")
    
    # Verify file is unlinked
    files = speaker_manager.get_processed_files(speaker_id)
    assert len(files) == 0, f"Files still linked to speaker: {len(files)}"
    
    print("✓ Verified file is unlinked")
    
    return file_id


def test_speaker_search(speaker_manager: SpeakerManager):
    """Test speaker search."""
    print("\n=== Testing Speaker Search ===")
    
    # Create additional speakers for search testing
    speaker1 = speaker_manager.create_speaker(
        name="John Doe",
        description="A male voice actor",
        tags=["male", "deep", "english"]
    )
    
    speaker2 = speaker_manager.create_speaker(
        name="Jane Smith",
        description="A female voice actor",
        tags=["female", "high", "english"]
    )
    
    # Search by name
    speakers = speaker_manager.search_speakers("John")
    assert len(speakers) > 0, "No speakers found with name containing 'John'"
    assert any(s.name == "John Doe" for s in speakers), "John Doe not found in search results"
    
    print(f"✓ Found {len(speakers)} speakers with name containing 'John'")
    
    # Search by description
    speakers = speaker_manager.search_speakers("female")
    assert len(speakers) > 0, "No speakers found with description containing 'female'"
    assert any(s.name == "Jane Smith" for s in speakers), "Jane Smith not found in search results"
    
    print(f"✓ Found {len(speakers)} speakers with description containing 'female'")
    
    # Find by tags
    speakers = speaker_manager.find_speakers_by_tags(["english"], match_all=True)
    assert len(speakers) >= 2, "Not enough speakers found with tag 'english'"
    
    print(f"✓ Found {len(speakers)} speakers with tag 'english'")
    
    # Find by multiple tags (any)
    speakers = speaker_manager.find_speakers_by_tags(["male", "high"], match_all=False)
    assert len(speakers) >= 2, "Not enough speakers found with tags 'male' or 'high'"
    
    print(f"✓ Found {len(speakers)} speakers with tags 'male' or 'high'")
    
    # Find by multiple tags (all)
    speakers = speaker_manager.find_speakers_by_tags(["female", "english"], match_all=True)
    assert len(speakers) >= 1, "No speakers found with both tags 'female' and 'english'"
    assert any(s.name == "Jane Smith" for s in speakers), "Jane Smith not found in search results"
    
    print(f"✓ Found {len(speakers)} speakers with both tags 'female' and 'english'")
    
    return speaker1, speaker2


def test_speaker_deletion(speaker_id: int, speaker_manager: SpeakerManager):
    """Test deleting a speaker."""
    print("\n=== Testing Speaker Deletion ===")
    
    # Delete the speaker
    success = speaker_manager.delete_speaker(speaker_id)
    
    # Verify deletion was successful
    assert success, "Speaker deletion failed"
    
    # Try to get the deleted speaker
    speaker = speaker_manager.get_speaker(speaker_id)
    
    # Verify speaker was deleted
    assert speaker is None, f"Speaker not deleted: {speaker}"
    
    print("✓ Speaker deletion successful")


def main():
    """Run all tests."""
    print("=== Starting Speaker Management System Tests ===")
    
    # Test speaker creation
    speaker, speaker_manager, db_manager = test_speaker_creation()
    
    # Test speaker retrieval
    speaker = test_speaker_retrieval(speaker.id, speaker_manager)
    
    # Test speaker update
    speaker = test_speaker_update(speaker.id, speaker_manager)
    
    # Test tag operations
    tags = test_tag_operations(speaker.id, speaker_manager)
    
    # Test processed file operations
    file_id = test_processed_file_operations(speaker.id, speaker_manager, db_manager)
    
    # Test speaker search
    speaker1, speaker2 = test_speaker_search(speaker_manager)
    
    # Test speaker deletion
    test_speaker_deletion(speaker.id, speaker_manager)
    
    print("\n=== All Speaker Management System Tests Passed ===")


if __name__ == "__main__":
    main()