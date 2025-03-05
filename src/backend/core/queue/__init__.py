"""
Processing Queue package for the Voices application.

This package provides functionality for managing processing tasks, including
parallel processing, progress tracking, priority handling, and intelligent
resource allocation based on task type and system capabilities.
"""

from .queue_manager import QueueManager
from .task import Task, TaskStatus
from .priority import Priority
from .resource_manager import ResourceManager
from .worker import Worker

__all__ = [
    'QueueManager',
    'Task',
    'TaskStatus',
    'Priority',
    'ResourceManager',
    'Worker'
]