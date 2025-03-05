"""
Database module for the Voices application.

This module provides database functionality for storing and retrieving data related to
speakers, processed files, processing history, and ML model performance metrics.
"""

from .db_manager import DatabaseManager
from .models import (
    Speaker,
    ProcessedFile,
    ProcessingHistory,
    ModelPerformance,
    Base
)

__all__ = [
    'DatabaseManager',
    'Speaker',
    'ProcessedFile',
    'ProcessingHistory',
    'ModelPerformance',
    'Base'
]