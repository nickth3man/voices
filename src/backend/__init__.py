"""
Backend package for the Voices application.

This package provides the Python backend functionality for the Voices application,
including audio processing, machine learning models, and data storage.
"""

# Import subpackages to make them available through the backend package
from . import core

__all__ = [
    'core'
]