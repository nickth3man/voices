#!/usr/bin/env python3
"""
Metadata Manager for the Voices application.

This module provides functionality for extracting, storing, retrieving,
and searching metadata for audio files.
"""

import os
import json
import datetime
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from sqlalchemy import or_, and_
from sqlalchemy.exc import SQLAlchemyError

from backend.storage.database.models import ProcessedFile, FileMetadata, CustomMetadata
from backend.storage.database.db_manager import DatabaseManager
from backend.storage.files.file_manager import FileManager


class MetadataManager:
    """
    Metadata Manager for audio files.
    
    This class provides functionality for extracting, storing, retrieving,
    and searching metadata for audio files.
    """
    
    def __init__(self, db_manager: DatabaseManager, file_manager: Optional[FileManager] = None):
        """
        Initialize the metadata manager.
        
        Args:
            db_manager: Database manager instance
            file_manager: File manager instance (optional)
        """
        self.db_manager = db_manager
        self.file_manager = file_manager
        self.logger = logging.getLogger(__name__)
    
    def extract_metadata(self, file_path: str, extract_audio_characteristics: bool = True) -> Dict[str, Any]:
        """
        Extract metadata from an audio file.
        
        Args:
            file_path: Path to the audio file
            extract_audio_characteristics: Whether to extract audio characteristics
            
        Returns:
            Dictionary containing metadata
        """
        metadata = {}
        
        # Check if file exists
        if not os.path.exists(file_path):
            return {'error': f"File not found: {file_path}"}
        
        try:
            # Basic file metadata
            file_stats = os.stat(file_path)
            metadata['file_size'] = file_stats.st_size
            metadata['file_format'] = os.path.splitext(file_path)[1].lower().replace('.', '')
            
            # Extract audio metadata using librosa
            if LIBROSA_AVAILABLE:
                try:
                    # Load audio file
                    y, sr = librosa.load(file_path, sr=None)
                    
                    # Basic audio metadata
                    metadata['duration'] = float(librosa.get_duration(y=y, sr=sr))
                    metadata['sample_rate'] = int(sr)
                    metadata['channels'] = 1 if y.ndim == 1 else y.shape[0]
                    
                    # Extract audio characteristics if requested
                    if extract_audio_characteristics:
                        # RMS energy
                        rms = librosa.feature.rms(y=y)[0]
                        metadata['rms_mean'] = float(np.mean(rms))
                        metadata['rms_std'] = float(np.std(rms))
                        
                        # Spectral centroid
                        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                        metadata['spectral_centroid_mean'] = float(np.mean(centroid))
                        metadata['spectral_centroid_std'] = float(np.std(centroid))
                        
                        # Spectral bandwidth
                        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
                        metadata['spectral_bandwidth_mean'] = float(np.mean(bandwidth))
                        metadata['spectral_bandwidth_std'] = float(np.std(bandwidth))
                        
                        # Zero crossing rate
                        zcr = librosa.feature.zero_crossing_rate(y)[0]
                        metadata['zero_crossing_rate_mean'] = float(np.mean(zcr))
                        metadata['zero_crossing_rate_std'] = float(np.std(zcr))
                        
                        # MFCC
                        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                        metadata['mfcc_means'] = [float(np.mean(mfcc[i])) for i in range(mfcc.shape[0])]
                        metadata['mfcc_stds'] = [float(np.std(mfcc[i])) for i in range(mfcc.shape[0])]
                    
                    metadata['extracted_at'] = datetime.datetime.utcnow().isoformat()
                
                except Exception as e:
                    self.logger.error(f"Error extracting audio metadata: {str(e)}")
                    metadata['error'] = f"Error extracting audio metadata: {str(e)}"
            else:
                metadata['error'] = "Librosa not available. Audio characteristics extraction is disabled."
        
        except Exception as e:
            self.logger.error(f"Error extracting metadata: {str(e)}")
            metadata['error'] = f"Error extracting metadata: {str(e)}"
        
        return metadata
    
    def store_metadata(self, file_id: int, metadata: Dict[str, Any]) -> bool:
        """
        Store metadata for a file.
        
        Args:
            file_id: ID of the file
            metadata: Dictionary containing metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if file exists
            file = self.db_manager.query(ProcessedFile).filter_by(id=file_id).first()
            if not file:
                self.logger.error(f"File not found: {file_id}")
                return False
            
            # Check if metadata already exists
            existing_metadata = self.db_manager.query(FileMetadata).filter_by(file_id=file_id).first()
            
            if existing_metadata:
                # Update existing metadata
                for key, value in metadata.items():
                    if key in ['duration', 'sample_rate', 'channels', 'file_format',
                              'rms_mean', 'rms_std', 'spectral_centroid_mean', 'spectral_centroid_std',
                              'spectral_bandwidth_mean', 'spectral_bandwidth_std',
                              'zero_crossing_rate_mean', 'zero_crossing_rate_std']:
                        setattr(existing_metadata, key, value)
                    elif key == 'mfcc_means':
                        existing_metadata.mfcc_means = json.dumps(value)
                    elif key == 'mfcc_stds':
                        existing_metadata.mfcc_stds = json.dumps(value)
                
                existing_metadata.updated_at = datetime.datetime.utcnow()
                if 'extracted_at' in metadata:
                    existing_metadata.extracted_at = datetime.datetime.fromisoformat(metadata['extracted_at'])
                
                self.db_manager.commit()
            else:
                # Create new metadata
                new_metadata = FileMetadata(
                    file_id=file_id,
                    duration=metadata.get('duration'),
                    sample_rate=metadata.get('sample_rate'),
                    channels=metadata.get('channels'),
                    file_format=metadata.get('file_format'),
                    rms_mean=metadata.get('rms_mean'),
                    rms_std=metadata.get('rms_std'),
                    spectral_centroid_mean=metadata.get('spectral_centroid_mean'),
                    spectral_centroid_std=metadata.get('spectral_centroid_std'),
                    spectral_bandwidth_mean=metadata.get('spectral_bandwidth_mean'),
                    spectral_bandwidth_std=metadata.get('spectral_bandwidth_std'),
                    zero_crossing_rate_mean=metadata.get('zero_crossing_rate_mean'),
                    zero_crossing_rate_std=metadata.get('zero_crossing_rate_std'),
                    mfcc_means=json.dumps(metadata.get('mfcc_means', [])),
                    mfcc_stds=json.dumps(metadata.get('mfcc_stds', [])),
                    extracted_at=datetime.datetime.fromisoformat(metadata['extracted_at']) if 'extracted_at' in metadata else None
                )
                
                self.db_manager.add(new_metadata)
                self.db_manager.commit()
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error storing metadata: {str(e)}")
            self.db_manager.rollback()
            return False
    
    def get_metadata(self, file_id: int) -> Dict[str, Any]:
        """
        Get metadata for a file.
        
        Args:
            file_id: ID of the file
            
        Returns:
            Dictionary containing metadata
        """
        try:
            # Get file
            file = self.db_manager.query(ProcessedFile).filter_by(id=file_id).first()
            if not file:
                return {'error': f"File not found: {file_id}"}
            
            # Get metadata
            metadata_obj = self.db_manager.query(FileMetadata).filter_by(file_id=file_id).first()
            if not metadata_obj:
                return {
                    'file_id': file_id,
                    'filename': file.filename,
                    'file_format': file.file_format,
                    'created_at': file.created_at.isoformat() if file.created_at else None,
                    'updated_at': file.updated_at.isoformat() if file.updated_at else None,
                    'error': 'No metadata available for this file'
                }
            
            # Convert to dictionary
            metadata = {
                'file_id': file_id,
                'filename': file.filename,
                'file_format': file.file_format,
                'duration': metadata_obj.duration,
                'sample_rate': metadata_obj.sample_rate,
                'channels': metadata_obj.channels,
                'rms_mean': metadata_obj.rms_mean,
                'rms_std': metadata_obj.rms_std,
                'spectral_centroid_mean': metadata_obj.spectral_centroid_mean,
                'spectral_centroid_std': metadata_obj.spectral_centroid_std,
                'spectral_bandwidth_mean': metadata_obj.spectral_bandwidth_mean,
                'spectral_bandwidth_std': metadata_obj.spectral_bandwidth_std,
                'zero_crossing_rate_mean': metadata_obj.zero_crossing_rate_mean,
                'zero_crossing_rate_std': metadata_obj.zero_crossing_rate_std,
                'mfcc_means': json.loads(metadata_obj.mfcc_means) if metadata_obj.mfcc_means else [],
                'mfcc_stds': json.loads(metadata_obj.mfcc_stds) if metadata_obj.mfcc_stds else [],
                'extracted_at': metadata_obj.extracted_at.isoformat() if metadata_obj.extracted_at else None,
                'created_at': metadata_obj.created_at.isoformat() if metadata_obj.created_at else None,
                'updated_at': metadata_obj.updated_at.isoformat() if metadata_obj.updated_at else None
            }
            
            # Get custom metadata
            custom_metadata = self.db_manager.query(CustomMetadata).filter_by(file_id=file_id).all()
            for cm in custom_metadata:
                metadata[f'custom_{cm.field_name}'] = cm.field_value
                metadata[f'custom_{cm.field_name}_type'] = cm.field_type
            
            return metadata
        
        except Exception as e:
            self.logger.error(f"Error getting metadata: {str(e)}")
            return {'error': f"Error getting metadata: {str(e)}"}
    
    def add_custom_metadata(self, file_id: int, field_name: str, field_value: str, field_type: str = 'text') -> bool:
        """
        Add custom metadata for a file.
        
        Args:
            file_id: ID of the file
            field_name: Name of the custom field
            field_value: Value for the custom field
            field_type: Type of the custom field
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if file exists
            file = self.db_manager.query(ProcessedFile).filter_by(id=file_id).first()
            if not file:
                self.logger.error(f"File not found: {file_id}")
                return False
            
            # Check if field already exists
            existing_field = self.db_manager.query(CustomMetadata).filter_by(
                file_id=file_id, field_name=field_name).first()
            
            if existing_field:
                # Update existing field
                existing_field.field_value = field_value
                existing_field.field_type = field_type
                existing_field.updated_at = datetime.datetime.utcnow()
            else:
                # Create new field
                new_field = CustomMetadata(
                    file_id=file_id,
                    field_name=field_name,
                    field_value=field_value,
                    field_type=field_type
                )
                self.db_manager.add(new_field)
            
            self.db_manager.commit()
            return True
        
        except Exception as e:
            self.logger.error(f"Error adding custom metadata: {str(e)}")
            self.db_manager.rollback()
            return False
    
    def remove_custom_metadata(self, file_id: int, field_name: str) -> bool:
        """
        Remove custom metadata for a file.
        
        Args:
            file_id: ID of the file
            field_name: Name of the custom field
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if field exists
            field = self.db_manager.query(CustomMetadata).filter_by(
                file_id=file_id, field_name=field_name).first()
            
            if not field:
                self.logger.error(f"Field not found: {field_name} for file {file_id}")
                return False
            
            # Delete field
            self.db_manager.delete(field)
            self.db_manager.commit()
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error removing custom metadata: {str(e)}")
            self.db_manager.rollback()
            return False
    
    def search_by_metadata(self, criteria: Dict[str, Any], limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Search for files based on metadata criteria.
        
        Args:
            criteria: Dictionary containing search criteria
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of dictionaries containing file metadata
        """
        try:
            # Build query
            query = self.db_manager.query(ProcessedFile, FileMetadata).\
                outerjoin(FileMetadata, ProcessedFile.id == FileMetadata.file_id)
            
            # Apply criteria
            filters = []
            
            # Filename
            if 'filename' in criteria and criteria['filename']:
                filters.append(ProcessedFile.filename.like(f"%{criteria['filename']}%"))
            
            # File format
            if 'file_format' in criteria and criteria['file_format']:
                filters.append(ProcessedFile.file_format == criteria['file_format'])
            
            # Duration range
            if 'duration_min' in criteria and criteria['duration_min']:
                filters.append(FileMetadata.duration >= float(criteria['duration_min']))
            if 'duration_max' in criteria and criteria['duration_max']:
                filters.append(FileMetadata.duration <= float(criteria['duration_max']))
            
            # Sample rate
            if 'sample_rate' in criteria and criteria['sample_rate']:
                filters.append(FileMetadata.sample_rate == int(criteria['sample_rate']))
            
            # Channels
            if 'channels' in criteria and criteria['channels']:
                filters.append(FileMetadata.channels == int(criteria['channels']))
            
            # Created date range
            if 'created_after' in criteria and criteria['created_after']:
                created_after = datetime.datetime.fromisoformat(criteria['created_after'])
                filters.append(ProcessedFile.created_at >= created_after)
            if 'created_before' in criteria and criteria['created_before']:
                created_before = datetime.datetime.fromisoformat(criteria['created_before'])
                filters.append(ProcessedFile.created_at <= created_before)
            
            # Apply filters
            if filters:
                query = query.filter(and_(*filters))
            
            # Get total count
            total_count = query.count()
            
            # Apply pagination
            query = query.limit(limit).offset(offset)
            
            # Execute query
            results = query.all()
            
            # Format results
            formatted_results = []
            for file, metadata in results:
                result = {
                    'file_id': file.id,
                    'filename': file.filename,
                    'file_format': file.file_format,
                    'created_at': file.created_at.isoformat() if file.created_at else None,
                    'updated_at': file.updated_at.isoformat() if file.updated_at else None
                }
                
                if metadata:
                    result.update({
                        'duration': metadata.duration,
                        'sample_rate': metadata.sample_rate,
                        'channels': metadata.channels,
                        'rms_mean': metadata.rms_mean,
                        'spectral_centroid_mean': metadata.spectral_centroid_mean,
                        'spectral_bandwidth_mean': metadata.spectral_bandwidth_mean,
                        'zero_crossing_rate_mean': metadata.zero_crossing_rate_mean,
                        'extracted_at': metadata.extracted_at.isoformat() if metadata.extracted_at else None
                    })
                
                formatted_results.append(result)
            
            return {
                'success': True,
                'total_count': total_count,
                'limit': limit,
                'offset': offset,
                'data': formatted_results
            }
        
        except Exception as e:
            self.logger.error(f"Error searching metadata: {str(e)}")
            return {
                'success': False,
                'error': f"Error searching metadata: {str(e)}"
            }
    
    def export_metadata(self, file_id: int, export_path: Optional[str] = None) -> Union[Dict[str, Any], bool]:
        """
        Export metadata for a file to a JSON file.
        
        Args:
            file_id: ID of the file
            export_path: Path to export the metadata to (optional)
            
        Returns:
            Dictionary containing metadata if export_path is None, otherwise True if successful
        """
        try:
            # Get metadata
            metadata = self.get_metadata(file_id)
            
            if 'error' in metadata:
                return {'error': metadata['error']}
            
            # If no export path, return metadata
            if not export_path:
                return metadata
            
            # Export to file
            with open(export_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error exporting metadata: {str(e)}")
            return {'error': f"Error exporting metadata: {str(e)}"}
    
    def import_metadata(self, import_path: str, file_id: int) -> bool:
        """
        Import metadata for a file from a JSON file.
        
        Args:
            import_path: Path to import the metadata from
            file_id: ID of the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if file exists
            file = self.db_manager.query(ProcessedFile).filter_by(id=file_id).first()
            if not file:
                self.logger.error(f"File not found: {file_id}")
                return False
            
            # Import from file
            with open(import_path, 'r') as f:
                metadata = json.load(f)
            
            # Store metadata
            result = self.store_metadata(file_id, metadata)
            if not result:
                return False
            
            # Import custom metadata
            custom_fields = {k: v for k, v in metadata.items() if k.startswith('custom_') and not k.endswith('_type')}
            for field_name, field_value in custom_fields.items():
                # Remove 'custom_' prefix
                field_name = field_name.replace('custom_', '')
                field_type = metadata.get(f'custom_{field_name}_type', 'text')
                
                # Add custom metadata
                self.add_custom_metadata(file_id, field_name, field_value, field_type)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error importing metadata: {str(e)}")
            return False
    
    def batch_extract_metadata(self, directory: str, recursive: bool = False) -> Dict[str, Any]:
        """
        Extract metadata for all audio files in a directory.
        
        Args:
            directory: Directory to scan for audio files
            recursive: Whether to scan subdirectories
            
        Returns:
            Dictionary containing results
        """
        results = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'files': []
        }
        
        try:
            # Check if directory exists
            if not os.path.exists(directory) or not os.path.isdir(directory):
                return {'error': f"Directory not found: {directory}"}
            
            # Get audio files
            audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a', '.aiff', '.aif']
            audio_files = []
            
            if recursive:
                for root, _, files in os.walk(directory):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in audio_extensions):
                            audio_files.append(os.path.join(root, file))
            else:
                for file in os.listdir(directory):
                    if any(file.lower().endswith(ext) for ext in audio_extensions):
                        audio_files.append(os.path.join(directory, file))
            
            results['total_files'] = len(audio_files)
            
            # Process each file
            for file_path in audio_files:
                try:
                    # Extract metadata
                    metadata = self.extract_metadata(file_path)
                    
                    # Register file in database if not already registered
                    filename = os.path.basename(file_path)
                    file_format = os.path.splitext(filename)[1].lower().replace('.', '')
                    
                    file = self.db_manager.query(ProcessedFile).filter_by(
                        filename=filename, original_path=file_path).first()
                    
                    if not file:
                        file = ProcessedFile(
                            filename=filename,
                            original_path=file_path,
                            processed_path=file_path,
                            file_format=file_format
                        )
                        self.db_manager.add(file)
                        self.db_manager.commit()
                    
                    # Store metadata
                    result = self.store_metadata(file.id, metadata)
                    
                    if result:
                        results['successful'] += 1
                        results['files'].append({
                            'file_id': file.id,
                            'filename': filename,
                            'path': file_path,
                            'success': True
                        })
                    else:
                        results['failed'] += 1
                        results['files'].append({
                            'file_id': file.id,
                            'filename': filename,
                            'path': file_path,
                            'success': False,
                            'error': 'Failed to store metadata'
                        })
                
                except Exception as e:
                    results['failed'] += 1
                    results['files'].append({
                        'filename': os.path.basename(file_path),
                        'path': file_path,
                        'success': False,
                        'error': str(e)
                    })
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error in batch metadata extraction: {str(e)}")
            return {'error': f"Error in batch metadata extraction: {str(e)}"}
    
    def get_metadata_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about metadata in the database.
        
        Returns:
            Dictionary containing statistics
        """
        try:
            # Get total counts
            total_files = self.db_manager.query(ProcessedFile).count()
            files_with_metadata = self.db_manager.query(FileMetadata).count()
            files_with_custom_metadata = self.db_manager.query(CustomMetadata.file_id.distinct()).count()
            
            # Get file format distribution
            format_query = self.db_manager.query(
                ProcessedFile.file_format, 
                self.db_manager.func.count(ProcessedFile.id)
            ).group_by(ProcessedFile.file_format).all()
            
            format_distribution = {format_: count for format_, count in format_query}
            
            # Get duration statistics
            duration_stats = {}
            if files_with_metadata > 0:
                duration_query = self.db_manager.query(
                    self.db_manager.func.min(FileMetadata.duration),
                    self.db_manager.func.max(FileMetadata.duration),
                    self.db_manager.func.avg(FileMetadata.duration)
                ).first()
                
                duration_stats = {
                    'min': duration_query[0],
                    'max': duration_query[1],
                    'avg': duration_query[2]
                }
            
            # Get custom field statistics
            custom_field_query = self.db_manager.query(
                CustomMetadata.field_name,
                CustomMetadata.field_type,
                self.db_manager.func.count(CustomMetadata.id)
            ).group_by(CustomMetadata.field_name, CustomMetadata.field_type).all()
            
            custom_fields = [
                {
                    'name': name,
                    'type': type_,
                    'count': count
                }
                for name, type_, count in custom_field_query
            ]
            
            return {
                'success': True,
                'total_files': total_files,
                'files_with_metadata': files_with_metadata,
                'files_with_custom_metadata': files_with_custom_metadata,
                'format_distribution': format_distribution,
                'duration_stats': duration_stats,
                'custom_fields': custom_fields
            }
        
        except Exception as e:
            self.logger.error(f"Error getting metadata statistics: {str(e)}")
            return {
                'success': False,
                'error': f"Error getting metadata statistics: {str(e)}"
            }