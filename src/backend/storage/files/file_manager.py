"""
File organization system for the Voices application.

This module provides a FileManager class for managing storage locations,
implementing naming conventions, handling automatic file organization,
and providing monitoring of storage usage.
"""

import os
import shutil
import logging
import datetime
import re
import json
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path
import platform
import psutil

from ..database.db_manager import DatabaseManager
from ..database.models import ProcessedFile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FileManager:
    """
    Manager class for file organization operations.
    
    This class provides methods for managing storage locations, implementing
    naming conventions, handling automatic file organization, and monitoring
    storage usage.
    """
    
    # Default directory structure
    DEFAULT_DIRS = {
        "original": "original",
        "processed": "processed",
        "temp": "temp",
        "archive": "archive",
        "exports": "exports"
    }
    
    # Supported audio formats
    SUPPORTED_FORMATS = [
        "wav", "mp3", "flac", "ogg", "m4a", "aac", "aiff"
    ]
    
    def __init__(self, db_manager: DatabaseManager, base_dir: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the FileManager.
        
        Args:
            db_manager: Database manager instance.
            base_dir: Base directory for file storage. If None, uses a default path.
            config: Configuration dictionary for file organization settings.
        """
        self.db_manager = db_manager
        
        # Set base directory
        if base_dir is None:
            # Use default path in user's home directory
            home_dir = os.path.expanduser("~")
            self.base_dir = os.path.join(home_dir, ".voices", "files")
        else:
            self.base_dir = base_dir
        
        # Load configuration or use defaults
        self.config = config or {}
        self.storage_dirs = self._initialize_storage_dirs()
        
        # Create directory structure
        self._create_directory_structure()
    
    def _initialize_storage_dirs(self) -> Dict[str, str]:
        """
        Initialize storage directory paths based on configuration.
        
        Returns:
            Dict[str, str]: Dictionary of storage directory paths.
        """
        storage_dirs = {}
        
        # Get directory names from config or use defaults
        dir_config = self.config.get("directories", {})
        
        for dir_type, default_name in self.DEFAULT_DIRS.items():
            dir_name = dir_config.get(dir_type, default_name)
            storage_dirs[dir_type] = os.path.join(self.base_dir, dir_name)
        
        return storage_dirs
    
    def _create_directory_structure(self) -> bool:
        """
        Create the directory structure for file organization.
        
        Returns:
            bool: True if directories created successfully, False otherwise.
        """
        try:
            # Create base directory if it doesn't exist
            os.makedirs(self.base_dir, exist_ok=True)
            
            # Create subdirectories
            for dir_path in self.storage_dirs.values():
                os.makedirs(dir_path, exist_ok=True)
            
            logger.info(f"Created directory structure in {self.base_dir}")
            return True
        except Exception as e:
            logger.error(f"Error creating directory structure: {str(e)}")
            return False
    
    def get_storage_path(self, dir_type: str) -> str:
        """
        Get the path for a specific storage directory type.
        
        Args:
            dir_type: Type of storage directory (original, processed, temp, etc.).
            
        Returns:
            str: Path to the requested directory.
            
        Raises:
            ValueError: If the directory type is invalid.
        """
        if dir_type not in self.storage_dirs:
            raise ValueError(f"Invalid directory type: {dir_type}")
        
        return self.storage_dirs[dir_type]
    
    def generate_filename(self, original_filename: str, speaker_name: Optional[str] = None,
                         processing_type: Optional[str] = None, timestamp: bool = True) -> str:
        """
        Generate a filename based on naming conventions.
        
        Args:
            original_filename: Original filename.
            speaker_name: Optional speaker name to include in filename.
            processing_type: Optional processing type to include in filename.
            timestamp: Whether to include a timestamp in the filename.
            
        Returns:
            str: Generated filename following naming conventions.
        """
        # Extract base name and extension
        base_name, ext = os.path.splitext(os.path.basename(original_filename))
        
        # Clean base name (remove special characters, replace spaces with underscores)
        base_name = re.sub(r'[^\w\s-]', '', base_name).strip().lower()
        base_name = re.sub(r'[-\s]+', '_', base_name)
        
        # Build filename components
        components = []
        
        if speaker_name:
            # Clean speaker name
            clean_speaker = re.sub(r'[^\w\s-]', '', speaker_name).strip().lower()
            clean_speaker = re.sub(r'[-\s]+', '_', clean_speaker)
            components.append(clean_speaker)
        
        components.append(base_name)
        
        if processing_type:
            components.append(processing_type)
        
        if timestamp:
            time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            components.append(time_str)
        
        # Join components and add extension
        new_filename = "_".join(components) + ext
        
        return new_filename
    
    def store_file(self, source_path: str, dir_type: str, 
                  new_filename: Optional[str] = None,
                  speaker_name: Optional[str] = None,
                  processing_type: Optional[str] = None) -> Optional[str]:
        """
        Store a file in the appropriate directory with proper naming.
        
        Args:
            source_path: Path to the source file.
            dir_type: Type of storage directory (original, processed, etc.).
            new_filename: Optional new filename. If None, generates one.
            speaker_name: Optional speaker name for filename generation.
            processing_type: Optional processing type for filename generation.
            
        Returns:
            Optional[str]: Path to the stored file, or None if storage failed.
        """
        try:
            # Check if source file exists
            if not os.path.isfile(source_path):
                logger.error(f"Source file does not exist: {source_path}")
                return None
            
            # Get destination directory
            dest_dir = self.get_storage_path(dir_type)
            
            # Generate filename if not provided
            if new_filename is None:
                new_filename = self.generate_filename(
                    os.path.basename(source_path),
                    speaker_name,
                    processing_type
                )
            
            # Create destination path
            dest_path = os.path.join(dest_dir, new_filename)
            
            # Copy file
            shutil.copy2(source_path, dest_path)
            logger.info(f"Stored file: {source_path} -> {dest_path}")
            
            return dest_path
        except Exception as e:
            logger.error(f"Error storing file: {str(e)}")
            return None
    
    def move_file(self, source_path: str, dir_type: str,
                 new_filename: Optional[str] = None,
                 speaker_name: Optional[str] = None,
                 processing_type: Optional[str] = None) -> Optional[str]:
        """
        Move a file to the appropriate directory with proper naming.
        
        Args:
            source_path: Path to the source file.
            dir_type: Type of storage directory (original, processed, etc.).
            new_filename: Optional new filename. If None, generates one.
            speaker_name: Optional speaker name for filename generation.
            processing_type: Optional processing type for filename generation.
            
        Returns:
            Optional[str]: Path to the moved file, or None if move failed.
        """
        try:
            # Check if source file exists
            if not os.path.isfile(source_path):
                logger.error(f"Source file does not exist: {source_path}")
                return None
            
            # Get destination directory
            dest_dir = self.get_storage_path(dir_type)
            
            # Generate filename if not provided
            if new_filename is None:
                new_filename = self.generate_filename(
                    os.path.basename(source_path),
                    speaker_name,
                    processing_type
                )
            
            # Create destination path
            dest_path = os.path.join(dest_dir, new_filename)
            
            # Move file
            shutil.move(source_path, dest_path)
            logger.info(f"Moved file: {source_path} -> {dest_path}")
            
            return dest_path
        except Exception as e:
            logger.error(f"Error moving file: {str(e)}")
            return None
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file.
        
        Args:
            file_path: Path to the file to delete.
            
        Returns:
            bool: True if deletion successful, False otherwise.
        """
        try:
            # Check if file exists
            if not os.path.isfile(file_path):
                logger.error(f"File does not exist: {file_path}")
                return False
            
            # Delete file
            os.remove(file_path)
            logger.info(f"Deleted file: {file_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error deleting file: {str(e)}")
            return False
    
    def archive_file(self, source_path: str, 
                    new_filename: Optional[str] = None,
                    speaker_name: Optional[str] = None) -> Optional[str]:
        """
        Archive a file.
        
        Args:
            source_path: Path to the source file.
            new_filename: Optional new filename. If None, generates one.
            speaker_name: Optional speaker name for filename generation.
            
        Returns:
            Optional[str]: Path to the archived file, or None if archiving failed.
        """
        return self.move_file(
            source_path,
            "archive",
            new_filename,
            speaker_name,
            "archived"
        )
    
    def organize_files(self, dir_type: Optional[str] = None) -> bool:
        """
        Organize files in a directory based on naming conventions.
        
        Args:
            dir_type: Type of storage directory to organize. If None, organizes all directories.
            
        Returns:
            bool: True if organization successful, False otherwise.
        """
        try:
            if dir_type is None:
                # Organize all directories
                for dir_type in self.storage_dirs:
                    self._organize_directory(dir_type)
            else:
                # Organize specific directory
                self._organize_directory(dir_type)
            
            return True
        except Exception as e:
            logger.error(f"Error organizing files: {str(e)}")
            return False
    
    def _organize_directory(self, dir_type: str) -> bool:
        """
        Organize files in a specific directory.
        
        Args:
            dir_type: Type of storage directory to organize.
            
        Returns:
            bool: True if organization successful, False otherwise.
        """
        try:
            dir_path = self.get_storage_path(dir_type)
            
            # Get organization rules from config
            org_rules = self.config.get("organization", {}).get(dir_type, {})
            
            # If no specific rules, use default organization
            if not org_rules:
                # Default: organize by date (YYYY/MM/DD)
                for filename in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, filename)
                    
                    # Skip directories
                    if not os.path.isfile(file_path):
                        continue
                    
                    # Get file modification time
                    mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                    year_dir = str(mod_time.year)
                    month_dir = f"{mod_time.month:02d}"
                    day_dir = f"{mod_time.day:02d}"
                    
                    # Create date-based directory structure
                    date_dir = os.path.join(dir_path, year_dir, month_dir, day_dir)
                    os.makedirs(date_dir, exist_ok=True)
                    
                    # Move file to date directory
                    dest_path = os.path.join(date_dir, filename)
                    if file_path != dest_path:  # Avoid moving to same location
                        shutil.move(file_path, dest_path)
            else:
                # Use custom organization rules
                # TODO: Implement custom organization rules
                pass
            
            logger.info(f"Organized files in {dir_path}")
            return True
        except Exception as e:
            logger.error(f"Error organizing directory {dir_type}: {str(e)}")
            return False
    
    def get_storage_usage(self, dir_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get storage usage information.
        
        Args:
            dir_type: Type of storage directory to check. If None, checks all directories.
            
        Returns:
            Dict[str, Any]: Dictionary with storage usage information.
        """
        try:
            usage_info = {
                "total_size": 0,
                "file_count": 0,
                "directories": {}
            }
            
            if dir_type is None:
                # Check all directories
                for dir_type, dir_path in self.storage_dirs.items():
                    dir_size, file_count = self._get_directory_size(dir_path)
                    usage_info["directories"][dir_type] = {
                        "path": dir_path,
                        "size": dir_size,
                        "size_formatted": self._format_size(dir_size),
                        "file_count": file_count
                    }
                    usage_info["total_size"] += dir_size
                    usage_info["file_count"] += file_count
            else:
                # Check specific directory
                dir_path = self.get_storage_path(dir_type)
                dir_size, file_count = self._get_directory_size(dir_path)
                usage_info["directories"][dir_type] = {
                    "path": dir_path,
                    "size": dir_size,
                    "size_formatted": self._format_size(dir_size),
                    "file_count": file_count
                }
                usage_info["total_size"] = dir_size
                usage_info["file_count"] = file_count
            
            # Add formatted total size
            usage_info["total_size_formatted"] = self._format_size(usage_info["total_size"])
            
            # Add disk space information
            disk_usage = self._get_disk_usage(self.base_dir)
            usage_info.update(disk_usage)
            
            return usage_info
        except Exception as e:
            logger.error(f"Error getting storage usage: {str(e)}")
            return {"error": str(e)}
    
    def _get_directory_size(self, dir_path: str) -> Tuple[int, int]:
        """
        Calculate the total size and file count of a directory.
        
        Args:
            dir_path: Path to the directory.
            
        Returns:
            Tuple[int, int]: (total size in bytes, file count)
        """
        total_size = 0
        file_count = 0
        
        for dirpath, dirnames, filenames in os.walk(dir_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.isfile(file_path):
                    total_size += os.path.getsize(file_path)
                    file_count += 1
        
        return total_size, file_count
    
    def _format_size(self, size_bytes: int) -> str:
        """
        Format size in bytes to human-readable format.
        
        Args:
            size_bytes: Size in bytes.
            
        Returns:
            str: Formatted size string.
        """
        if size_bytes == 0:
            return "0 B"
        
        size_names = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        
        return f"{s} {size_names[i]}"
    
    def _get_disk_usage(self, path: str) -> Dict[str, Any]:
        """
        Get disk usage information for the path.
        
        Args:
            path: Path to check disk usage for.
            
        Returns:
            Dict[str, Any]: Dictionary with disk usage information.
        """
        disk_usage = psutil.disk_usage(path)
        
        return {
            "disk_total": disk_usage.total,
            "disk_used": disk_usage.used,
            "disk_free": disk_usage.free,
            "disk_percent": disk_usage.percent,
            "disk_total_formatted": self._format_size(disk_usage.total),
            "disk_used_formatted": self._format_size(disk_usage.used),
            "disk_free_formatted": self._format_size(disk_usage.free)
        }
    
    def find_files(self, pattern: str, dir_type: Optional[str] = None) -> List[str]:
        """
        Find files matching a pattern.
        
        Args:
            pattern: Glob pattern to match files.
            dir_type: Type of storage directory to search. If None, searches all directories.
            
        Returns:
            List[str]: List of matching file paths.
        """
        try:
            matching_files = []
            
            if dir_type is None:
                # Search all directories
                for dir_path in self.storage_dirs.values():
                    for file_path in Path(dir_path).rglob(pattern):
                        if file_path.is_file():
                            matching_files.append(str(file_path))
            else:
                # Search specific directory
                dir_path = self.get_storage_path(dir_type)
                for file_path in Path(dir_path).rglob(pattern):
                    if file_path.is_file():
                        matching_files.append(str(file_path))
            
            return matching_files
        except Exception as e:
            logger.error(f"Error finding files: {str(e)}")
            return []
    
    def register_file(self, file_path: str, original_path: Optional[str] = None,
                     speaker_id: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """
        Register a file in the database.
        
        Args:
            file_path: Path to the file.
            original_path: Optional path to the original file.
            speaker_id: Optional ID of the associated speaker.
            metadata: Optional metadata for the file.
            
        Returns:
            Optional[int]: ID of the registered file, or None if registration failed.
        """
        try:
            # Check if file exists
            if not os.path.isfile(file_path):
                logger.error(f"File does not exist: {file_path}")
                return None
            
            # Extract file information
            filename = os.path.basename(file_path)
            file_format = os.path.splitext(filename)[1].lstrip('.').lower()
            
            # Create ProcessedFile object
            processed_file = ProcessedFile(
                filename=filename,
                original_path=original_path or "",
                processed_path=file_path,
                file_format=file_format,
                speaker_id=speaker_id,
                metadata=metadata or {}
            )
            
            # Add to database
            if self.db_manager.add(processed_file):
                logger.info(f"Registered file in database: {file_path}")
                return processed_file.id
            else:
                logger.error(f"Failed to register file in database: {file_path}")
                return None
        except Exception as e:
            logger.error(f"Error registering file: {str(e)}")
            return None
    
    def get_file_info(self, file_id: int) -> Optional[Dict[str, Any]]:
        """
        Get information about a registered file.
        
        Args:
            file_id: ID of the file.
            
        Returns:
            Optional[Dict[str, Any]]: Dictionary with file information, or None if not found.
        """
        try:
            # Get file from database
            processed_file = self.db_manager.get_by_id(ProcessedFile, file_id)
            if not processed_file:
                logger.error(f"File not found in database: ID={file_id}")
                return None
            
            # Get file size
            file_size = 0
            if os.path.isfile(processed_file.processed_path):
                file_size = os.path.getsize(processed_file.processed_path)
            
            # Build file info dictionary
            file_info = {
                "id": processed_file.id,
                "filename": processed_file.filename,
                "original_path": processed_file.original_path,
                "processed_path": processed_file.processed_path,
                "file_format": processed_file.file_format,
                "duration": processed_file.duration,
                "sample_rate": processed_file.sample_rate,
                "channels": processed_file.channels,
                "created_at": processed_file.created_at,
                "updated_at": processed_file.updated_at,
                "metadata": processed_file.metadata,
                "speaker_id": processed_file.speaker_id,
                "file_size": file_size,
                "file_size_formatted": self._format_size(file_size),
                "exists": os.path.isfile(processed_file.processed_path)
            }
            
            return file_info
        except Exception as e:
            logger.error(f"Error getting file info: {str(e)}")
            return None
    
    def get_all_files(self) -> List[Dict[str, Any]]:
        """
        Get information about all registered files.
        
        Returns:
            List[Dict[str, Any]]: List of dictionaries with file information.
        """
        try:
            # Get all files from database
            processed_files = self.db_manager.get_all(ProcessedFile)
            
            # Build file info list
            file_info_list = []
            for processed_file in processed_files:
                # Get file size
                file_size = 0
                if os.path.isfile(processed_file.processed_path):
                    file_size = os.path.getsize(processed_file.processed_path)
                
                # Build file info dictionary
                file_info = {
                    "id": processed_file.id,
                    "filename": processed_file.filename,
                    "file_format": processed_file.file_format,
                    "duration": processed_file.duration,
                    "created_at": processed_file.created_at,
                    "speaker_id": processed_file.speaker_id,
                    "file_size": file_size,
                    "file_size_formatted": self._format_size(file_size),
                    "exists": os.path.isfile(processed_file.processed_path)
                }
                
                file_info_list.append(file_info)
            
            return file_info_list
        except Exception as e:
            logger.error(f"Error getting all files: {str(e)}")
            return []
    
    def export_file(self, file_id: int, export_format: Optional[str] = None,
                   export_path: Optional[str] = None) -> Optional[str]:
        """
        Export a file to a specific format and location.
        
        Args:
            file_id: ID of the file to export.
            export_format: Format to export to. If None, uses the original format.
            export_path: Path to export to. If None, uses the exports directory.
            
        Returns:
            Optional[str]: Path to the exported file, or None if export failed.
        """
        try:
            # Get file from database
            processed_file = self.db_manager.get_by_id(ProcessedFile, file_id)
            if not processed_file:
                logger.error(f"File not found in database: ID={file_id}")
                return None
            
            # Check if source file exists
            if not os.path.isfile(processed_file.processed_path):
                logger.error(f"Source file does not exist: {processed_file.processed_path}")
                return None
            
            # Determine export format
            if export_format is None:
                export_format = processed_file.file_format
            
            # Check if format is supported
            if export_format.lower() not in self.SUPPORTED_FORMATS:
                logger.error(f"Unsupported export format: {export_format}")
                return None
            
            # Determine export path
            if export_path is None:
                # Use exports directory
                exports_dir = self.get_storage_path("exports")
                export_filename = f"{os.path.splitext(processed_file.filename)[0]}.{export_format}"
                export_path = os.path.join(exports_dir, export_filename)
            
            # If format is the same, just copy the file
            if export_format.lower() == processed_file.file_format.lower():
                shutil.copy2(processed_file.processed_path, export_path)
                logger.info(f"Exported file: {processed_file.processed_path} -> {export_path}")
                return export_path
            
            # Otherwise, convert the file (requires additional libraries)
            # TODO: Implement file format conversion
            logger.error("File format conversion not implemented yet")
            return None
        except Exception as e:
            logger.error(f"Error exporting file: {str(e)}")
            return None
    
    def cleanup_temp_files(self, max_age_days: int = 7) -> int:
        """
        Clean up temporary files older than a specified age.
        
        Args:
            max_age_days: Maximum age of files in days.
            
        Returns:
            int: Number of files deleted.
        """
        try:
            temp_dir = self.get_storage_path("temp")
            deleted_count = 0
            
            # Calculate cutoff time
            cutoff_time = datetime.datetime.now() - datetime.timedelta(days=max_age_days)
            
            # Iterate through files in temp directory
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                
                # Skip directories
                if not os.path.isfile(file_path):
                    continue
                
                # Check file modification time
                mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                
                # Delete if older than cutoff
                if mod_time < cutoff_time:
                    os.remove(file_path)
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} temporary files")
            return deleted_count
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {str(e)}")
            return 0
    
    def save_config(self, config_path: Optional[str] = None) -> bool:
        """
        Save the current configuration to a file.
        
        Args:
            config_path: Path to save the configuration to. If None, uses a default path.
            
        Returns:
            bool: True if save successful, False otherwise.
        """
        try:
            if config_path is None:
                # Use default path
                config_path = os.path.join(self.base_dir, "file_manager_config.json")
            
            # Create config dictionary
            config = {
                "base_dir": self.base_dir,
                "directories": {
                    dir_type: os.path.basename(dir_path)
                    for dir_type, dir_path in self.storage_dirs.items()
                },
                "organization": self.config.get("organization", {})
            }
            
            # Save to file
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            logger.info(f"Saved configuration to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    @classmethod
    def load_config(cls, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to the configuration file.
            
        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return {}