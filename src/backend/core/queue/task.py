"""
Task definitions for the processing queue.

This module provides the Task class and related functionality for
representing and managing processing tasks in the queue.
"""

import uuid
import time
import logging
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Callable, Set, Union
from dataclasses import dataclass, field

from .priority import Priority, calculate_task_priority

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a processing task."""
    
    PENDING = auto()     # Task is waiting to be processed
    RUNNING = auto()     # Task is currently being processed
    COMPLETED = auto()   # Task has been successfully completed
    FAILED = auto()      # Task has failed
    CANCELLED = auto()   # Task has been cancelled
    PAUSED = auto()      # Task has been paused


@dataclass
class TaskProgress:
    """Progress information for a task."""
    
    current: int = 0
    total: int = 100
    message: str = ""
    
    @property
    def percentage(self) -> float:
        """Calculate the percentage of completion."""
        if self.total == 0:
            return 0.0
        return (self.current / self.total) * 100.0
    
    def update(self, current: int, total: Optional[int] = None, message: Optional[str] = None) -> None:
        """
        Update the progress information.
        
        Args:
            current: Current progress value
            total: Total progress value (if None, uses existing total)
            message: Progress message (if None, uses existing message)
        """
        self.current = current
        if total is not None:
            self.total = total
        if message is not None:
            self.message = message


class Task:
    """Represents a processing task in the queue."""
    
    def __init__(
        self,
        task_id: Optional[str] = None,
        name: str = "Unnamed Task",
        description: str = "",
        function: Optional[Callable] = None,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        priority: Priority = Priority.NORMAL,
        dependencies: Optional[List[str]] = None,
        resource_requirements: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        timeout: Optional[float] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Initialize a task.
        
        Args:
            task_id: Unique identifier for the task (if None, generates a UUID)
            name: Name of the task
            description: Description of the task
            function: Function to execute for the task
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            priority: Priority of the task
            dependencies: List of task IDs that must complete before this task
            resource_requirements: Dictionary of resource requirements
            max_retries: Maximum number of retry attempts
            timeout: Timeout in seconds (None for no timeout)
            tags: List of tags for categorizing the task
        """
        self.task_id = task_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.function = function
        self.args = args or []
        self.kwargs = kwargs or {}
        self.priority = priority
        self.dependencies = set(dependencies or [])
        self.resource_requirements = resource_requirements or {}
        self.max_retries = max_retries
        self.timeout = timeout
        self.tags = set(tags or [])
        
        # Status and tracking
        self.status = TaskStatus.PENDING
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None
        self.progress = TaskProgress()
        self.result = None
        self.error = None
        self.retry_count = 0
        self.worker_id = None
        
        # Dynamic priority factors
        self.age_factor = 0.0
        self.dependency_factor = 0.0
        self.resource_factor = 0.0
        
        logger.debug(f"Created task {self.task_id}: {self.name}")
    
    @property
    def effective_priority(self) -> float:
        """Calculate the effective priority of the task."""
        return calculate_task_priority(
            self.priority,
            self.age_factor,
            self.dependency_factor,
            self.resource_factor
        )
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate the duration of the task in seconds."""
        if self.started_at is None:
            return None
        
        end_time = self.completed_at or time.time()
        return end_time - self.started_at
    
    @property
    def wait_time(self) -> float:
        """Calculate the wait time of the task in seconds."""
        start_time = self.started_at or time.time()
        return start_time - self.created_at
    
    @property
    def is_ready(self) -> bool:
        """Check if the task is ready to be processed (all dependencies satisfied)."""
        return len(self.dependencies) == 0
    
    def update_progress(self, current: int, total: Optional[int] = None, message: Optional[str] = None) -> None:
        """
        Update the progress of the task.
        
        Args:
            current: Current progress value
            total: Total progress value (if None, uses existing total)
            message: Progress message (if None, uses existing message)
        """
        self.progress.update(current, total, message)
        logger.debug(f"Task {self.task_id} progress: {self.progress.percentage:.1f}% - {self.progress.message}")
    
    def mark_running(self, worker_id: str) -> None:
        """
        Mark the task as running.
        
        Args:
            worker_id: ID of the worker processing the task
        """
        self.status = TaskStatus.RUNNING
        self.started_at = time.time()
        self.worker_id = worker_id
        logger.info(f"Task {self.task_id} started by worker {worker_id}")
    
    def mark_completed(self, result: Any = None) -> None:
        """
        Mark the task as completed.
        
        Args:
            result: Result of the task
        """
        self.status = TaskStatus.COMPLETED
        self.completed_at = time.time()
        self.result = result
        self.progress.update(self.progress.total, self.progress.total, "Completed")
        logger.info(f"Task {self.task_id} completed in {self.duration:.2f}s")
    
    def mark_failed(self, error: Exception) -> None:
        """
        Mark the task as failed.
        
        Args:
            error: Error that caused the failure
        """
        self.status = TaskStatus.FAILED
        self.completed_at = time.time()
        self.error = error
        logger.error(f"Task {self.task_id} failed: {str(error)}")
    
    def mark_cancelled(self) -> None:
        """Mark the task as cancelled."""
        self.status = TaskStatus.CANCELLED
        self.completed_at = time.time()
        logger.info(f"Task {self.task_id} cancelled")
    
    def mark_paused(self) -> None:
        """Mark the task as paused."""
        self.status = TaskStatus.PAUSED
        logger.info(f"Task {self.task_id} paused")
    
    def retry(self) -> bool:
        """
        Retry the task if possible.
        
        Returns:
            True if the task can be retried, False otherwise
        """
        if self.retry_count >= self.max_retries:
            logger.warning(f"Task {self.task_id} has reached maximum retry count ({self.max_retries})")
            return False
        
        self.retry_count += 1
        self.status = TaskStatus.PENDING
        self.started_at = None
        self.worker_id = None
        logger.info(f"Task {self.task_id} queued for retry (attempt {self.retry_count}/{self.max_retries})")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the task to a dictionary.
        
        Returns:
            Dictionary representation of the task
        """
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "priority": self.priority.name,
            "status": self.status.name,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress": {
                "current": self.progress.current,
                "total": self.progress.total,
                "percentage": self.progress.percentage,
                "message": self.progress.message
            },
            "dependencies": list(self.dependencies),
            "resource_requirements": self.resource_requirements,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "timeout": self.timeout,
            "tags": list(self.tags),
            "worker_id": self.worker_id,
            "duration": self.duration,
            "wait_time": self.wait_time,
            "effective_priority": self.effective_priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """
        Create a task from a dictionary.
        
        Args:
            data: Dictionary representation of the task
            
        Returns:
            Task instance
        """
        task = cls(
            task_id=data.get("task_id"),
            name=data.get("name", "Unnamed Task"),
            description=data.get("description", ""),
            priority=Priority.from_string(data.get("priority", "NORMAL")),
            dependencies=data.get("dependencies", []),
            resource_requirements=data.get("resource_requirements", {}),
            max_retries=data.get("max_retries", 3),
            timeout=data.get("timeout"),
            tags=data.get("tags", [])
        )
        
        # Set status if provided
        if "status" in data:
            task.status = TaskStatus[data["status"]]
        
        # Set timestamps if provided
        if "created_at" in data:
            task.created_at = data["created_at"]
        if "started_at" in data:
            task.started_at = data["started_at"]
        if "completed_at" in data:
            task.completed_at = data["completed_at"]
        
        # Set progress if provided
        if "progress" in data:
            progress = data["progress"]
            task.progress.update(
                progress.get("current", 0),
                progress.get("total", 100),
                progress.get("message", "")
            )
        
        # Set retry count if provided
        if "retry_count" in data:
            task.retry_count = data["retry_count"]
        
        # Set worker ID if provided
        if "worker_id" in data:
            task.worker_id = data["worker_id"]
        
        return task