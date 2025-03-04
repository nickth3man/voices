"""
Speaker management functionality for the Voices application.

This module provides a SpeakerManager class for managing speaker profiles,
including CRUD operations, tagging system, and linking to processed audio files.
"""

import logging
import datetime
from typing import List, Optional, Set, Dict, Any, Union

from .db_manager import DatabaseManager
from .models import Speaker, ProcessedFile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SpeakerManager:
    """
    Manager class for speaker profile operations.
    
    This class provides methods for creating, updating, deleting, and querying
    speaker profiles, as well as managing tags and linking to processed files.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize the SpeakerManager.
        
        Args:
            db_manager: Database manager instance.
        """
        self.db_manager = db_manager
    
    def create_speaker(self, name: str, description: Optional[str] = None, 
                      tags: Optional[List[str]] = None) -> Optional[Speaker]:
        """
        Create a new speaker profile.
        
        Args:
            name: Name of the speaker.
            description: Optional description of the speaker.
            tags: Optional list of tags for the speaker.
            
        Returns:
            Speaker: The created speaker object, or None if creation failed.
        """
        try:
            # Create timestamp
            now = datetime.datetime.now()
            
            # Create speaker object
            speaker = Speaker(
                name=name,
                description=description,
                tags=tags or [],
                created_at=now,
                updated_at=now
            )
            
            # Insert into database
            speaker_id = self.db_manager.insert(speaker)
            if speaker_id:
                speaker.id = speaker_id
                logger.info(f"Created speaker: ID={speaker_id}, Name='{name}'")
                return speaker
            else:
                logger.error(f"Failed to create speaker: Name='{name}'")
                return None
        except Exception as e:
            logger.error(f"Error creating speaker: {str(e)}")
            return None
    
    def update_speaker(self, speaker_id: int, name: Optional[str] = None, 
                      description: Optional[str] = None) -> bool:
        """
        Update a speaker profile.
        
        Args:
            speaker_id: ID of the speaker to update.
            name: New name for the speaker (if None, name is not updated).
            description: New description for the speaker (if None, description is not updated).
            
        Returns:
            bool: True if update successful, False otherwise.
        """
        try:
            # Get the speaker
            speaker = self.db_manager.get_by_id(Speaker, speaker_id)
            if not speaker:
                logger.error(f"Speaker not found: ID={speaker_id}")
                return False
            
            # Update fields if provided
            if name is not None:
                speaker.name = name
            
            if description is not None:
                speaker.description = description
            
            # Update timestamp
            speaker.updated_at = datetime.datetime.now()
            
            # Update in database
            success = self.db_manager.update(speaker)
            if success:
                logger.info(f"Updated speaker: ID={speaker_id}")
                return True
            else:
                logger.error(f"Failed to update speaker: ID={speaker_id}")
                return False
        except Exception as e:
            logger.error(f"Error updating speaker: {str(e)}")
            return False
    
    def delete_speaker(self, speaker_id: int) -> bool:
        """
        Delete a speaker profile.
        
        Args:
            speaker_id: ID of the speaker to delete.
            
        Returns:
            bool: True if deletion successful, False otherwise.
        """
        try:
            # Get the speaker
            speaker = self.db_manager.get_by_id(Speaker, speaker_id)
            if not speaker:
                logger.error(f"Speaker not found: ID={speaker_id}")
                return False
            
            # Delete from database
            success = self.db_manager.delete(speaker)
            if success:
                logger.info(f"Deleted speaker: ID={speaker_id}")
                return True
            else:
                logger.error(f"Failed to delete speaker: ID={speaker_id}")
                return False
        except Exception as e:
            logger.error(f"Error deleting speaker: {str(e)}")
            return False
    
    def get_speaker(self, speaker_id: int) -> Optional[Speaker]:
        """
        Get a speaker by ID.
        
        Args:
            speaker_id: ID of the speaker to retrieve.
            
        Returns:
            Speaker: The speaker object, or None if not found.
        """
        try:
            speaker = self.db_manager.get_by_id(Speaker, speaker_id)
            return speaker
        except Exception as e:
            logger.error(f"Error getting speaker: {str(e)}")
            return None
    
    def get_all_speakers(self) -> List[Speaker]:
        """
        Get all speakers.
        
        Returns:
            List[Speaker]: List of all speaker objects.
        """
        try:
            speakers = self.db_manager.get_all(Speaker)
            return speakers
        except Exception as e:
            logger.error(f"Error getting all speakers: {str(e)}")
            return []
    
    def add_tags(self, speaker_id: int, tags: List[str]) -> bool:
        """
        Add tags to a speaker.
        
        Args:
            speaker_id: ID of the speaker.
            tags: List of tags to add.
            
        Returns:
            bool: True if tags added successfully, False otherwise.
        """
        try:
            # Get the speaker
            speaker = self.db_manager.get_by_id(Speaker, speaker_id)
            if not speaker:
                logger.error(f"Speaker not found: ID={speaker_id}")
                return False
            
            # Add new tags (avoid duplicates)
            current_tags = set(speaker.tags)
            for tag in tags:
                current_tags.add(tag)
            
            speaker.tags = list(current_tags)
            speaker.updated_at = datetime.datetime.now()
            
            # Update in database
            success = self.db_manager.update(speaker)
            if success:
                logger.info(f"Added tags to speaker: ID={speaker_id}, Tags={', '.join(tags)}")
                return True
            else:
                logger.error(f"Failed to add tags to speaker: ID={speaker_id}")
                return False
        except Exception as e:
            logger.error(f"Error adding tags to speaker: {str(e)}")
            return False
    
    def remove_tags(self, speaker_id: int, tags: List[str]) -> bool:
        """
        Remove tags from a speaker.
        
        Args:
            speaker_id: ID of the speaker.
            tags: List of tags to remove.
            
        Returns:
            bool: True if tags removed successfully, False otherwise.
        """
        try:
            # Get the speaker
            speaker = self.db_manager.get_by_id(Speaker, speaker_id)
            if not speaker:
                logger.error(f"Speaker not found: ID={speaker_id}")
                return False
            
            # Remove tags
            current_tags = set(speaker.tags)
            for tag in tags:
                if tag in current_tags:
                    current_tags.remove(tag)
            
            speaker.tags = list(current_tags)
            speaker.updated_at = datetime.datetime.now()
            
            # Update in database
            success = self.db_manager.update(speaker)
            if success:
                logger.info(f"Removed tags from speaker: ID={speaker_id}, Tags={', '.join(tags)}")
                return True
            else:
                logger.error(f"Failed to remove tags from speaker: ID={speaker_id}")
                return False
        except Exception as e:
            logger.error(f"Error removing tags from speaker: {str(e)}")
            return False
    
    def get_tags(self, speaker_id: int) -> List[str]:
        """
        Get all tags for a speaker.
        
        Args:
            speaker_id: ID of the speaker.
            
        Returns:
            List[str]: List of tags for the speaker.
        """
        try:
            speaker = self.db_manager.get_by_id(Speaker, speaker_id)
            if not speaker:
                logger.error(f"Speaker not found: ID={speaker_id}")
                return []
            
            return speaker.tags
        except Exception as e:
            logger.error(f"Error getting tags for speaker: {str(e)}")
            return []
    
    def get_all_tags(self) -> Set[str]:
        """
        Get all unique tags used across all speakers.
        
        Returns:
            Set[str]: Set of all unique tags.
        """
        try:
            speakers = self.db_manager.get_all(Speaker)
            all_tags = set()
            
            for speaker in speakers:
                all_tags.update(speaker.tags)
            
            return all_tags
        except Exception as e:
            logger.error(f"Error getting all tags: {str(e)}")
            return set()
    
    def find_speakers_by_tags(self, tags: List[str], match_all: bool = False) -> List[Speaker]:
        """
        Find speakers by tags.
        
        Args:
            tags: List of tags to search for.
            match_all: If True, speakers must have all tags. If False, speakers must have any of the tags.
            
        Returns:
            List[Speaker]: List of matching speakers.
        """
        try:
            speakers = self.db_manager.get_all(Speaker)
            matching_speakers = []
            
            for speaker in speakers:
                speaker_tags = set(speaker.tags)
                search_tags = set(tags)
                
                if match_all:
                    # All tags must match
                    if search_tags.issubset(speaker_tags):
                        matching_speakers.append(speaker)
                else:
                    # Any tag must match
                    if search_tags.intersection(speaker_tags):
                        matching_speakers.append(speaker)
            
            return matching_speakers
        except Exception as e:
            logger.error(f"Error finding speakers by tags: {str(e)}")
            return []
    
    def search_speakers(self, query: str) -> List[Speaker]:
        """
        Search for speakers by name or description.
        
        Args:
            query: Search query string.
            
        Returns:
            List[Speaker]: List of matching speakers.
        """
        try:
            speakers = self.db_manager.get_all(Speaker)
            matching_speakers = []
            
            query = query.lower()
            for speaker in speakers:
                if (query in speaker.name.lower() or 
                    (speaker.description and query in speaker.description.lower())):
                    matching_speakers.append(speaker)
            
            return matching_speakers
        except Exception as e:
            logger.error(f"Error searching speakers: {str(e)}")
            return []
    
    def link_processed_file(self, speaker_id: int, file_id: int) -> bool:
        """
        Link a processed file to a speaker.
        
        Args:
            speaker_id: ID of the speaker.
            file_id: ID of the processed file.
            
        Returns:
            bool: True if linking successful, False otherwise.
        """
        try:
            # Get the speaker and file
            speaker = self.db_manager.get_by_id(Speaker, speaker_id)
            if not speaker:
                logger.error(f"Speaker not found: ID={speaker_id}")
                return False
            
            file = self.db_manager.get_by_id(ProcessedFile, file_id)
            if not file:
                logger.error(f"Processed file not found: ID={file_id}")
                return False
            
            # Link file to speaker
            file.speaker_id = speaker_id
            
            # Update in database
            success = self.db_manager.update(file)
            if success:
                logger.info(f"Linked file ID={file_id} to speaker ID={speaker_id}")
                return True
            else:
                logger.error(f"Failed to link file ID={file_id} to speaker ID={speaker_id}")
                return False
        except Exception as e:
            logger.error(f"Error linking file to speaker: {str(e)}")
            return False
    
    def unlink_processed_file(self, file_id: int) -> bool:
        """
        Unlink a processed file from its speaker.
        
        Args:
            file_id: ID of the processed file.
            
        Returns:
            bool: True if unlinking successful, False otherwise.
        """
        try:
            # Get the file
            file = self.db_manager.get_by_id(ProcessedFile, file_id)
            if not file:
                logger.error(f"Processed file not found: ID={file_id}")
                return False
            
            # Unlink file from speaker
            file.speaker_id = None
            
            # Update in database
            success = self.db_manager.update(file)
            if success:
                logger.info(f"Unlinked file ID={file_id} from speaker")
                return True
            else:
                logger.error(f"Failed to unlink file ID={file_id}")
                return False
        except Exception as e:
            logger.error(f"Error unlinking file from speaker: {str(e)}")
            return False
    
    def get_processed_files(self, speaker_id: int) -> List[ProcessedFile]:
        """
        Get all processed files for a speaker.
        
        Args:
            speaker_id: ID of the speaker.
            
        Returns:
            List[ProcessedFile]: List of processed files for the speaker.
        """
        try:
            # Get the speaker
            speaker = self.db_manager.get_by_id(Speaker, speaker_id)
            if not speaker:
                logger.error(f"Speaker not found: ID={speaker_id}")
                return []
            
            # Get all files for the speaker
            files = self.db_manager.query(
                ProcessedFile, 
                {"speaker_id": speaker_id}
            )
            
            return files
        except Exception as e:
            logger.error(f"Error getting processed files for speaker: {str(e)}")
            return []