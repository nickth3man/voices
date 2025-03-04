"""
Core package for the Voices application backend.

This package provides the core functionality for the backend,
including communication, configuration, logging, and processing queue.
"""

# Import subpackages to make them available through the core package
from . import communication

__all__ = [
    'communication'
]