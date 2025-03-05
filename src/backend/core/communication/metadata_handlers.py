"""
Metadata handlers for the communication server.

This module provides handlers for metadata-related operations
that can be called from the frontend via the Python Bridge.
"""

import os
import logging
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from ...storage.database.db_manager import DatabaseManager
from ...storage.files.file_manager import FileManager
from ...storage.metadata.metadata_manager import MetadataManager
from ...storage.database.models import ProcessedFile

# Configure logging
logger = logging.getLogger(__name__)


class MetadataHandlers:
    """
    Handlers for metadata-related operations.
    
    This class provides methods that can be called from the frontend
    via the Python Bridge to interact with the metadata management system.
    """
    
    def __init__(self):
        """Initialize the metadata handlers."""
        self.db_manager = DatabaseManager()
        if not self.db_manager.initialize():
            logger.error("Failed to initialize database")
            raise RuntimeError("Failed to initialize database")
        
        self.file_manager = FileManager(self.db_manager)
        self.metadata_manager = MetadataManager(self.db_manager, self.file_manager)
    
    def metadata_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get metadata for a file.
        
        Args:
            params: Dictionary with parameters
                - fileId: ID of the file
                
        Returns:
            Dictionary with response
                - success: Whether the operation was successful
                - data: Metadata if successful
                - error: Error message if unsuccessful
        """
        try:
            file_id = params.get('fileId')
            if not file_id:
                return {'success': False, 'error': 'File ID is required'}
            
            metadata = self.metadata_manager.get_metadata(file_id)
            
            if metadata is None:
                return {'success': False, 'error': f'No metadata found for file ID: {file_id}'}
            
            return {'success': True, 'data': metadata}
        
        except Exception as e:
            logger.error(f"Error in metadata_get: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def metadata_update(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update metadata for a file.
        
        Args:
            params: Dictionary with parameters
                - fileId: ID of the file
                - metadata: Metadata to update
                
        Returns:
            Dictionary with response
                - success: Whether the operation was successful
                - error: Error message if unsuccessful
        """
        try:
            file_id = params.get('fileId')
            metadata = params.get('metadata')
            
            if not file_id:
                return {'success': False, 'error': 'File ID is required'}
            
            if not metadata:
                return {'success': False, 'error': 'Metadata is required'}
            
            result = self.metadata_manager.store_metadata(file_id, metadata)
            
            if result:
                return {'success': True}
            else:
                return {'success': False, 'error': f'Failed to update metadata for file ID: {file_id}'}
        
        except Exception as e:
            logger.error(f"Error in metadata_update: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def metadata_add_custom(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add custom metadata field.
        
        Args:
            params: Dictionary with parameters
                - fileId: ID of the file
                - fieldName: Name of the custom field
                - fieldValue: Value for the custom field
                - fieldType: Type of the custom field
                
        Returns:
            Dictionary with response
                - success: Whether the operation was successful
                - error: Error message if unsuccessful
        """
        try:
            file_id = params.get('fileId')
            field_name = params.get('fieldName')
            field_value = params.get('fieldValue')
            field_type = params.get('fieldType', 'text')
            
            if not file_id:
                return {'success': False, 'error': 'File ID is required'}
            
            if not field_name:
                return {'success': False, 'error': 'Field name is required'}
            
            result = self.metadata_manager.add_custom_metadata(
                file_id, field_name, field_value, field_type
            )
            
            if result:
                return {'success': True}
            else:
                return {'success': False, 'error': f'Failed to add custom metadata for file ID: {file_id}'}
        
        except Exception as e:
            logger.error(f"Error in metadata_add_custom: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def metadata_remove_custom(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove custom metadata field.
        
        Args:
            params: Dictionary with parameters
                - fileId: ID of the file
                - fieldName: Name of the custom field to remove
                
        Returns:
            Dictionary with response
                - success: Whether the operation was successful
                - error: Error message if unsuccessful
        """
        try:
            file_id = params.get('fileId')
            field_name = params.get('fieldName')
            
            if not file_id:
                return {'success': False, 'error': 'File ID is required'}
            
            if not field_name:
                return {'success': False, 'error': 'Field name is required'}
            
            result = self.metadata_manager.remove_custom_metadata(file_id, field_name)
            
            if result:
                return {'success': True}
            else:
                return {'success': False, 'error': f'Failed to remove custom metadata for file ID: {file_id}'}
        
        except Exception as e:
            logger.error(f"Error in metadata_remove_custom: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def metadata_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for files based on metadata criteria.
        
        Args:
            params: Dictionary with parameters
                - criteria: Search criteria
                - limit: Maximum number of results
                - offset: Offset for pagination
                
        Returns:
            Dictionary with response
                - success: Whether the operation was successful
                - data: Search results if successful
                - error: Error message if unsuccessful
        """
        try:
            criteria = params.get('criteria', {})
            limit = params.get('limit', 100)
            offset = params.get('offset', 0)
            
            results = self.metadata_manager.search_by_metadata(criteria, limit, offset)
            
            return {'success': True, 'data': results}
        
        except Exception as e:
            logger.error(f"Error in metadata_search: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def file_get_details(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get file details.
        
        Args:
            params: Dictionary with parameters
                - fileId: ID of the file
                
        Returns:
            Dictionary with response
                - success: Whether the operation was successful
                - data: File details if successful
                - error: Error message if unsuccessful
        """
        try:
            file_id = params.get('fileId')
            
            if not file_id:
                return {'success': False, 'error': 'File ID is required'}
            
            file = self.db_manager.get_by_id(ProcessedFile, file_id)
            
            if not file:
                return {'success': False, 'error': f'File not found: ID={file_id}'}
            
            file_details = {
                'id': file.id,
                'filename': file.filename,
                'original_path': file.original_path,
                'processed_path': file.processed_path,
                'file_format': file.file_format,
                'duration': file.duration,
                'sample_rate': file.sample_rate,
                'channels': file.channels,
                'created_at': file.created_at.isoformat() if file.created_at else None,
                'updated_at': file.updated_at.isoformat() if file.updated_at else None,
                'speaker_id': file.speaker_id
            }
            
            return {'success': True, 'data': file_details}
        
        except Exception as e:
            logger.error(f"Error in file_get_details: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def metadata_statistics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get metadata statistics.
        
        Args:
            params: Dictionary with parameters (none required)
                
        Returns:
            Dictionary with response
                - success: Whether the operation was successful
                - data: Statistics if successful
                - error: Error message if unsuccessful
        """
        try:
            statistics = self.metadata_manager.get_metadata_statistics()
            
            return {'success': True, 'data': statistics}
        
        except Exception as e:
            logger.error(f"Error in metadata_statistics: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def metadata_export(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Export metadata to a file.
        
        Args:
            params: Dictionary with parameters
                - fileId: ID of the file
                - exportPath: Optional path to export the metadata to
                
        Returns:
            Dictionary with response
                - success: Whether the operation was successful
                - data: Export information if successful
                - error: Error message if unsuccessful
        """
        try:
            file_id = params.get('fileId')
            export_path = params.get('exportPath')
            
            if not file_id:
                return {'success': False, 'error': 'File ID is required'}
            
            # If no export path is provided, create a temporary file
            if not export_path:
                temp_dir = tempfile.gettempdir()
                export_path = os.path.join(temp_dir, f"metadata_export_{file_id}.json")
            
            result = self.metadata_manager.export_metadata(file_id, export_path)
            
            if result:
                return {
                    'success': True,
                    'data': {
                        'path': export_path
                    }
                }
            else:
                return {'success': False, 'error': f'Failed to export metadata for file ID: {file_id}'}
        
        except Exception as e:
            logger.error(f"Error in metadata_export: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def metadata_import(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Import metadata from a file.
        
        Args:
            params: Dictionary with parameters
                - fileId: ID of the file
                - importPath: Path to import the metadata from
                
        Returns:
            Dictionary with response
                - success: Whether the operation was successful
                - error: Error message if unsuccessful
        """
        try:
            file_id = params.get('fileId')
            import_path = params.get('importPath')
            
            if not file_id:
                return {'success': False, 'error': 'File ID is required'}
            
            if not import_path:
                return {'success': False, 'error': 'Import path is required'}
            
            result = self.metadata_manager.import_metadata(import_path, file_id)
            
            if result:
                return {'success': True}
            else:
                return {'success': False, 'error': f'Failed to import metadata for file ID: {file_id}'}
        
        except Exception as e:
            logger.error(f"Error in metadata_import: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def metadata_batch_extract(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Batch extract metadata for files in a directory.
        
        Args:
            params: Dictionary with parameters
                - directory: Directory to scan for audio files
                - recursive: Whether to scan subdirectories
                
        Returns:
            Dictionary with response
                - success: Whether the operation was successful
                - data: Batch results if successful
                - error: Error message if unsuccessful
        """
        try:
            directory = params.get('directory')
            recursive = params.get('recursive', False)
            
            if not directory:
                return {'success': False, 'error': 'Directory is required'}
            
            results = self.metadata_manager.batch_extract_metadata(directory, recursive)
            
            return {'success': True, 'data': results}
        
        except Exception as e:
            logger.error(f"Error in metadata_batch_extract: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def metadata_extract(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from a file.
        
        Args:
            params: Dictionary with parameters
                - filePath: Path to the audio file
                - extractAudioCharacteristics: Whether to extract audio characteristics
                
        Returns:
            Dictionary with response
                - success: Whether the operation was successful
                - data: Metadata if successful
                - error: Error message if unsuccessful
        """
        try:
            file_path = params.get('filePath')
            extract_audio_characteristics = params.get('extractAudioCharacteristics', True)
            
            if not file_path:
                return {'success': False, 'error': 'File path is required'}
            
            metadata = self.metadata_manager.extract_metadata(
                file_path,
                extract_audio_characteristics=extract_audio_characteristics
            )
            
            return {'success': True, 'data': metadata}
        
        except Exception as e:
            logger.error(f"Error in metadata_extract: {str(e)}")
            return {'success': False, 'error': str(e)}


# Create a singleton instance
metadata_handlers = MetadataHandlers()

# Define handler functions that can be registered with the server

def handle_metadata_get(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for metadata_get."""
    return metadata_handlers.metadata_get(params)

def handle_metadata_update(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for metadata_update."""
    return metadata_handlers.metadata_update(params)

def handle_metadata_add_custom(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for metadata_add_custom."""
    return metadata_handlers.metadata_add_custom(params)

def handle_metadata_remove_custom(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for metadata_remove_custom."""
    return metadata_handlers.metadata_remove_custom(params)

def handle_metadata_search(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for metadata_search."""
    return metadata_handlers.metadata_search(params)

def handle_file_get_details(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for file_get_details."""
    return metadata_handlers.file_get_details(params)

def handle_metadata_statistics(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for metadata_statistics."""
    return metadata_handlers.metadata_statistics(params)

def handle_metadata_export(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for metadata_export."""
    return metadata_handlers.metadata_export(params)

def handle_metadata_import(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for metadata_import."""
    return metadata_handlers.metadata_import(params)

def handle_metadata_batch_extract(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for metadata_batch_extract."""
    return metadata_handlers.metadata_batch_extract(params)

def handle_metadata_extract(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for metadata_extract."""
    return metadata_handlers.metadata_extract(params)


# Dictionary of handlers to register with the server
METADATA_HANDLERS = {
    'metadata_get': handle_metadata_get,
    'metadata_update': handle_metadata_update,
    'metadata_add_custom': handle_metadata_add_custom,
    'metadata_remove_custom': handle_metadata_remove_custom,
    'metadata_search': handle_metadata_search,
    'file_get_details': handle_file_get_details,
    'metadata_statistics': handle_metadata_statistics,
    'metadata_export': handle_metadata_export,
    'metadata_import': handle_metadata_import,
    'metadata_batch_extract': handle_metadata_batch_extract,
    'metadata_extract': handle_metadata_extract
}