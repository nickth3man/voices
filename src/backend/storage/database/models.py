"""
Database models for the Voices application.

This module defines the SQLAlchemy ORM models for the database tables:
- Speaker: Information about speakers in audio files
- ProcessedFile: Information about processed audio files
- ProcessingHistory: History of processing operations
- ModelPerformance: Performance metrics for ML models
- FileMetadata: Metadata for audio files
- CustomMetadata: Custom metadata fields for audio files
"""

import datetime
import enum
import sqlalchemy
from typing import List, Optional

from sqlalchemy import (
    Column, Integer, String, Float, DateTime,
    ForeignKey, Text, Boolean, Enum, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class ModelType(enum.Enum):
    """Enum for different types of voice separation models."""
    SVOICE = "svoice"
    DEMUCS = "demucs"
    AUTO = "auto"


class Speaker(Base):
    """Speaker model for storing information about speakers in audio files."""
    
    __tablename__ = "speakers"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    tags = Column(JSON, nullable=True)  # Store tags as JSON array
    
    # Relationships
    processed_files = relationship("ProcessedFile", back_populates="speaker")
    
    def __repr__(self):
        return f"<Speaker(id={self.id}, name='{self.name}')>"


class ProcessedFile(Base):
    """ProcessedFile model for storing information about processed audio files."""
    
    __tablename__ = "processed_files"
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    original_path = Column(String(255), nullable=False)
    processed_path = Column(String(255), nullable=False)
    file_format = Column(String(10), nullable=False)  # wav, mp3, etc.
    duration = Column(Float, nullable=True)  # Duration in seconds
    sample_rate = Column(Integer, nullable=True)
    channels = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    metadata = Column(JSON, nullable=True)  # Store additional metadata as JSON
    
    # Foreign keys
    speaker_id = Column(Integer, ForeignKey("speakers.id"), nullable=True)
    
    # Relationships
    speaker = relationship("Speaker", back_populates="processed_files")
    processing_history = relationship("ProcessingHistory", back_populates="processed_file")
    
    def __repr__(self):
        return f"<ProcessedFile(id={self.id}, filename='{self.filename}')>"


class ProcessingHistory(Base):
    """ProcessingHistory model for storing history of processing operations."""
    
    __tablename__ = "processing_history"
    
    id = Column(Integer, primary_key=True)
    operation_type = Column(String(50), nullable=False)  # e.g., "voice_separation", "denoising"
    model_type = Column(Enum(ModelType), nullable=True)
    model_version = Column(String(50), nullable=True)
    parameters = Column(JSON, nullable=True)  # Store processing parameters as JSON
    start_time = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    success = Column(Boolean, nullable=False, default=False)
    error_message = Column(Text, nullable=True)
    
    # Foreign keys
    processed_file_id = Column(Integer, ForeignKey("processed_files.id"), nullable=False)
    
    # Relationships
    processed_file = relationship("ProcessedFile", back_populates="processing_history")
    model_performance = relationship("ModelPerformance", back_populates="processing_history", uselist=False)
    
    def __repr__(self):
        return f"<ProcessingHistory(id={self.id}, operation_type='{self.operation_type}', model_type={self.model_type})>"


class ModelPerformance(Base):
    """ModelPerformance model for storing performance metrics for ML models."""
    
    __tablename__ = "model_performance"
    
    id = Column(Integer, primary_key=True)
    model_type = Column(Enum(ModelType), nullable=False)
    model_version = Column(String(50), nullable=False)
    si_snri = Column(Float, nullable=True)  # Scale-Invariant Signal-to-Noise Ratio improvement
    sdri = Column(Float, nullable=True)  # Signal-to-Distortion Ratio improvement
    cpu_usage = Column(Float, nullable=True)  # CPU usage percentage
    memory_usage = Column(Float, nullable=True)  # Memory usage in MB
    processing_time = Column(Float, nullable=True)  # Processing time in seconds
    real_time_factor = Column(Float, nullable=True)  # Ratio of processing time to audio duration
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    additional_metrics = Column(JSON, nullable=True)  # Store additional metrics as JSON
    
    # Foreign keys
    processing_history_id = Column(Integer, ForeignKey("processing_history.id"), nullable=False)
    
    # Relationships
    processing_history = relationship("ProcessingHistory", back_populates="model_performance")
    
    def __repr__(self):
        return f"<ModelPerformance(id={self.id}, model_type={self.model_type}, model_version='{self.model_version}')>"


class FileMetadata(Base):
    """FileMetadata model for storing metadata for audio files."""
    
    __tablename__ = "file_metadata"
    
    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey("processed_files.id"), nullable=False, unique=True)
    
    # Basic audio metadata
    duration = Column(Float, nullable=True)  # Duration in seconds
    sample_rate = Column(Integer, nullable=True)  # Sample rate in Hz
    channels = Column(Integer, nullable=True)  # Number of audio channels
    file_format = Column(String(10), nullable=True)  # wav, mp3, etc.
    
    # Audio characteristics
    rms_mean = Column(Float, nullable=True)  # Root Mean Square energy (mean)
    rms_std = Column(Float, nullable=True)  # Root Mean Square energy (std)
    spectral_centroid_mean = Column(Float, nullable=True)  # Spectral centroid (mean)
    spectral_centroid_std = Column(Float, nullable=True)  # Spectral centroid (std)
    spectral_bandwidth_mean = Column(Float, nullable=True)  # Spectral bandwidth (mean)
    spectral_bandwidth_std = Column(Float, nullable=True)  # Spectral bandwidth (std)
    zero_crossing_rate_mean = Column(Float, nullable=True)  # Zero crossing rate (mean)
    zero_crossing_rate_std = Column(Float, nullable=True)  # Zero crossing rate (std)
    
    # MFCC coefficients (stored as JSON)
    mfcc_means = Column(JSON, nullable=True)  # MFCC means
    mfcc_stds = Column(JSON, nullable=True)  # MFCC standard deviations
    
    # Timestamps
    extracted_at = Column(DateTime, nullable=True)  # When metadata was extracted
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Relationships
    processed_file = relationship("ProcessedFile", backref="metadata")
    custom_fields = relationship("CustomMetadata", back_populates="file_metadata")
    
    def __repr__(self):
        return f"<FileMetadata(id={self.id}, file_id={self.file_id})>"


class CustomMetadata(Base):
    """CustomMetadata model for storing custom metadata fields for audio files."""
    
    __tablename__ = "custom_metadata"
    
    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey("processed_files.id"), nullable=False)
    field_name = Column(String(100), nullable=False)
    field_value = Column(Text, nullable=True)
    field_type = Column(String(20), nullable=False, default="text")  # text, number, boolean, date
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Relationships
    file_metadata = relationship("FileMetadata", back_populates="custom_fields")
    
    # Unique constraint for file_id and field_name
    __table_args__ = (
        sqlalchemy.UniqueConstraint('file_id', 'field_name', name='uix_file_field'),
    )
    
    def __repr__(self):
        return f"<CustomMetadata(id={self.id}, file_id={self.file_id}, field_name='{self.field_name}')>"