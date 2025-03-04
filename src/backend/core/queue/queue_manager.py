"""
Queue manager for processing tasks.

This module provides the QueueManager class for managing the processing queue,
including task submission, scheduling, and execution.
"""

import os
import time
import json
import logging
import threading
import heapq
from typing import Dict, List, Any, Optional, Callable, Set, Tuple, Union
from collections import defaultdict, deque
from dataclasses import dataclass, field

from .task import Task, TaskStatus
from .priority import Priority
from .resource_manager import ResourceManager
from .worker import Worker, WorkerPool

logger = logging.getLogger(__name__)


class QueueManager:
    """
    Manager for the processing queue.
    
    This class is responsible for managing the processing queue, including
    task submission, scheduling, and execution.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        resource_manager: Optional[ResourceManager] = None,
        num_workers: int = 0
    ):
        """
        Initialize the queue manager.
        
        Args:
            config: Configuration dictionary
            resource_manager: Resource manager instance (if None, creates a new one)
            num_workers: Number of worker threads (0 for auto-detection)
        """
        self.config = config or {}
        
        # Create resource manager if not provided
        self.resource_manager = resource_manager or ResourceManager(self.config.get('resources', {}))
        
        # Initialize task queues
        self._tasks = {}  # All tasks by ID
        self._pending_tasks = []  # Priority queue of pending tasks
        self._running_tasks = set()  # Set of running task IDs
        self._completed_tasks = {}  # Completed tasks by ID
        self._failed_tasks = {}  # Failed tasks by ID
        self._cancelled_tasks = {}  # Cancelled tasks by ID
        self._paused_tasks = {}  # Paused tasks by ID
        
        # Task dependencies
        self._dependent_tasks = defaultdict(set)  # Map of task ID to set of dependent task IDs
        
        # Locks
        self._queue_lock = threading.RLock()
        self._scheduler_lock = threading.RLock()
        
        # Create worker pool
        self.worker_pool = WorkerPool(
            num_workers=num_workers,
            resource_manager=self.resource_manager,
            on_task_completed=self._on_task_completed,
            on_task_failed=self._on_task_failed,
            on_task_progress=self._on_task_progress
        )
        
        # Scheduler thread
        self._scheduler_thread = None
        self._scheduler_stop_event = threading.Event()
        self._scheduler_interval = self.config.get('scheduler_interval', 1.0)  # seconds
        
        # Event callbacks
        self._task_callbacks = defaultdict(list)
        
        logger.info(f"Queue manager initialized")
    
    def start(self) -> None:
        """Start the queue manager."""
        # Start worker pool
        self.worker_pool.start()
        
        # Start scheduler thread
        self._scheduler_stop_event.clear()
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="QueueManager-Scheduler"
        )
        self._scheduler_thread.daemon = True
        self._scheduler_thread.start()
        
        logger.info("Queue manager started")
    
    def stop(self) -> None:
        """Stop the queue manager."""
        # Stop scheduler thread
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_stop_event.set()
            self._scheduler_thread.join(timeout=5.0)
        
        # Stop worker pool
        self.worker_pool.stop()
        
        logger.info("Queue manager stopped")
    
    def submit_task(self, task: Task) -> str:
        """
        Submit a task to the queue.
        
        Args:
            task: Task to submit
            
        Returns:
            Task ID
        """
        with self._queue_lock:
            # Add task to tasks dictionary
            self._tasks[task.task_id] = task
            
            # Check if task has dependencies
            if task.dependencies:
                # Check if dependencies exist
                for dep_id in list(task.dependencies):
                    if dep_id not in self._tasks:
                        logger.warning(f"Task {task.task_id} depends on non-existent task {dep_id}")
                        task.dependencies.remove(dep_id)
                    else:
                        dep_task = self._tasks[dep_id]
                        # If dependency is already completed, remove it
                        if dep_task.status == TaskStatus.COMPLETED:
                            task.dependencies.remove(dep_id)
                        else:
                            # Add this task as a dependent of its dependency
                            self._dependent_tasks[dep_id].add(task.task_id)
            
            # If task has no dependencies, add to pending queue
            if not task.dependencies:
                # Add to priority queue
                heapq.heappush(self._pending_tasks, (-task.effective_priority, task.created_at, task.task_id))
            
            logger.info(f"Task {task.task_id} submitted: {task.name}")
            
            # Trigger task submitted event
            self._trigger_event('task_submitted', task)
            
            return task.task_id
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            True if the task was cancelled, False otherwise
        """
        with self._queue_lock:
            if task_id not in self._tasks:
                logger.warning(f"Cannot cancel non-existent task {task_id}")
                return False
            
            task = self._tasks[task_id]
            
            # Check if task is already completed, failed, or cancelled
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                logger.warning(f"Cannot cancel task {task_id} with status {task.status.name}")
                return False
            
            # If task is running, cancel it through the worker pool
            if task.status == TaskStatus.RUNNING:
                # Find the worker running this task
                for worker in self.worker_pool.workers:
                    if worker.current_task and worker.current_task.task_id == task_id:
                        worker.cancel_current_task()
                        break
            
            # If task is pending, remove it from the pending queue
            elif task.status == TaskStatus.PENDING:
                # Remove from pending queue (will be filtered out in _get_next_task)
                task.mark_cancelled()
                self._cancelled_tasks[task_id] = task
                
                # Remove from tasks dictionary
                del self._tasks[task_id]
            
            # If task is paused, move it to cancelled
            elif task.status == TaskStatus.PAUSED:
                task.mark_cancelled()
                self._cancelled_tasks[task_id] = task
                del self._paused_tasks[task_id]
                del self._tasks[task_id]
            
            logger.info(f"Task {task_id} cancelled")
            
            # Trigger task cancelled event
            self._trigger_event('task_cancelled', task)
            
            return True
    
    def pause_task(self, task_id: str) -> bool:
        """
        Pause a pending task.
        
        Args:
            task_id: ID of the task to pause
            
        Returns:
            True if the task was paused, False otherwise
        """
        with self._queue_lock:
            if task_id not in self._tasks:
                logger.warning(f"Cannot pause non-existent task {task_id}")
                return False
            
            task = self._tasks[task_id]
            
            # Can only pause pending tasks
            if task.status != TaskStatus.PENDING:
                logger.warning(f"Cannot pause task {task_id} with status {task.status.name}")
                return False
            
            # Mark task as paused
            task.mark_paused()
            
            # Move to paused tasks
            self._paused_tasks[task_id] = task
            
            logger.info(f"Task {task_id} paused")
            
            # Trigger task paused event
            self._trigger_event('task_paused', task)
            
            return True
    
    def resume_task(self, task_id: str) -> bool:
        """
        Resume a paused task.
        
        Args:
            task_id: ID of the task to resume
            
        Returns:
            True if the task was resumed, False otherwise
        """
        with self._queue_lock:
            if task_id not in self._paused_tasks:
                logger.warning(f"Cannot resume non-paused task {task_id}")
                return False
            
            task = self._paused_tasks[task_id]
            
            # Mark task as pending
            task.status = TaskStatus.PENDING
            
            # Move back to tasks dictionary
            self._tasks[task_id] = task
            del self._paused_tasks[task_id]
            
            # Add to pending queue if no dependencies
            if not task.dependencies:
                heapq.heappush(self._pending_tasks, (-task.effective_priority, task.created_at, task.task_id))
            
            logger.info(f"Task {task_id} resumed")
            
            # Trigger task resumed event
            self._trigger_event('task_resumed', task)
            
            return True
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a task by ID.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task or None if not found
        """
        # Check active tasks
        if task_id in self._tasks:
            return self._tasks[task_id]
        
        # Check completed tasks
        if task_id in self._completed_tasks:
            return self._completed_tasks[task_id]
        
        # Check failed tasks
        if task_id in self._failed_tasks:
            return self._failed_tasks[task_id]
        
        # Check cancelled tasks
        if task_id in self._cancelled_tasks:
            return self._cancelled_tasks[task_id]
        
        # Check paused tasks
        if task_id in self._paused_tasks:
            return self._paused_tasks[task_id]
        
        return None
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        Get the status of a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task status or None if task not found
        """
        task = self.get_task(task_id)
        return task.status if task else None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get the status of the queue.
        
        Returns:
            Dictionary with queue status information
        """
        with self._queue_lock:
            return {
                'pending_tasks': len(self._pending_tasks),
                'running_tasks': len(self._running_tasks),
                'completed_tasks': len(self._completed_tasks),
                'failed_tasks': len(self._failed_tasks),
                'cancelled_tasks': len(self._cancelled_tasks),
                'paused_tasks': len(self._paused_tasks),
                'total_tasks': (
                    len(self._tasks) +
                    len(self._completed_tasks) +
                    len(self._failed_tasks) +
                    len(self._cancelled_tasks) +
                    len(self._paused_tasks)
                ),
                'worker_pool': self.worker_pool.get_status(),
                'resources': self.resource_manager.get_resource_usage()
            }
    
    def clear_completed_tasks(self, max_age: Optional[float] = None) -> int:
        """
        Clear completed tasks from the queue.
        
        Args:
            max_age: Maximum age in seconds (if None, clears all completed tasks)
            
        Returns:
            Number of tasks cleared
        """
        with self._queue_lock:
            if max_age is None:
                count = len(self._completed_tasks)
                self._completed_tasks.clear()
                return count
            
            current_time = time.time()
            to_remove = []
            
            for task_id, task in self._completed_tasks.items():
                if task.completed_at and (current_time - task.completed_at) > max_age:
                    to_remove.append(task_id)
            
            for task_id in to_remove:
                del self._completed_tasks[task_id]
            
            return len(to_remove)
    
    def clear_failed_tasks(self, max_age: Optional[float] = None) -> int:
        """
        Clear failed tasks from the queue.
        
        Args:
            max_age: Maximum age in seconds (if None, clears all failed tasks)
            
        Returns:
            Number of tasks cleared
        """
        with self._queue_lock:
            if max_age is None:
                count = len(self._failed_tasks)
                self._failed_tasks.clear()
                return count
            
            current_time = time.time()
            to_remove = []
            
            for task_id, task in self._failed_tasks.items():
                if task.completed_at and (current_time - task.completed_at) > max_age:
                    to_remove.append(task_id)
            
            for task_id in to_remove:
                del self._failed_tasks[task_id]
            
            return len(to_remove)
    
    def clear_cancelled_tasks(self, max_age: Optional[float] = None) -> int:
        """
        Clear cancelled tasks from the queue.
        
        Args:
            max_age: Maximum age in seconds (if None, clears all cancelled tasks)
            
        Returns:
            Number of tasks cleared
        """
        with self._queue_lock:
            if max_age is None:
                count = len(self._cancelled_tasks)
                self._cancelled_tasks.clear()
                return count
            
            current_time = time.time()
            to_remove = []
            
            for task_id, task in self._cancelled_tasks.items():
                if task.completed_at and (current_time - task.completed_at) > max_age:
                    to_remove.append(task_id)
            
            for task_id in to_remove:
                del self._cancelled_tasks[task_id]
            
            return len(to_remove)
    
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """
        Register a callback for a specific event type.
        
        Args:
            event_type: Type of event to register for
            callback: Callback function
        """
        self._task_callbacks[event_type].append(callback)
    
    def unregister_callback(self, event_type: str, callback: Callable) -> bool:
        """
        Unregister a callback for a specific event type.
        
        Args:
            event_type: Type of event to unregister from
            callback: Callback function
            
        Returns:
            True if the callback was unregistered, False otherwise
        """
        if event_type in self._task_callbacks and callback in self._task_callbacks[event_type]:
            self._task_callbacks[event_type].remove(callback)
            return True
        return False
    
    def _trigger_event(self, event_type: str, task: Task) -> None:
        """
        Trigger an event.
        
        Args:
            event_type: Type of event to trigger
            task: Task associated with the event
        """
        if event_type in self._task_callbacks:
            for callback in self._task_callbacks[event_type]:
                try:
                    callback(task)
                except Exception as e:
                    logger.error(f"Error in {event_type} callback: {str(e)}")
    
    def _scheduler_loop(self) -> None:
        """Scheduler loop for processing tasks."""
        logger.info("Scheduler thread started")
        
        while not self._scheduler_stop_event.is_set():
            try:
                self._schedule_tasks()
            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
            
            # Wait for next scheduling interval
            self._scheduler_stop_event.wait(self._scheduler_interval)
        
        logger.info("Scheduler thread stopped")
    
    def _schedule_tasks(self) -> None:
        """Schedule tasks for execution."""
        with self._scheduler_lock:
            # Update task priorities based on age
            self._update_task_priorities()
            
            # Get idle worker
            idle_worker = self.worker_pool.get_idle_worker()
            if not idle_worker:
                return
            
            # Get next task
            task = self._get_next_task()
            if not task:
                return
            
            # Execute task
            logger.debug(f"Scheduling task {task.task_id} on worker {idle_worker.name}")
            idle_worker.execute_task(task)
            
            # Mark task as running
            with self._queue_lock:
                self._running_tasks.add(task.task_id)
    
    def _get_next_task(self) -> Optional[Task]:
        """
        Get the next task to execute.
        
        Returns:
            Next task or None if no tasks are available
        """
        with self._queue_lock:
            while self._pending_tasks:
                # Get highest priority task
                _, _, task_id = heapq.heappop(self._pending_tasks)
                
                # Check if task still exists and is pending
                if task_id in self._tasks:
                    task = self._tasks[task_id]
                    if task.status == TaskStatus.PENDING and not task.dependencies:
                        # Check if resources are available
                        if self.resource_manager.can_allocate_resources(task.resource_requirements):
                            return task
                        else:
                            # Put task back in queue with lower priority
                            # This allows other tasks that can run with available resources to execute
                            task.resource_factor = 0.0  # Lower resource factor
                            heapq.heappush(self._pending_tasks, (-task.effective_priority, task.created_at, task.task_id))
                            logger.debug(f"Task {task_id} waiting for resources")
            
            return None
    
    def _update_task_priorities(self) -> None:
        """Update task priorities based on age and other factors."""
        with self._queue_lock:
            current_time = time.time()
            
            # Update pending tasks
            for task_id in [t[2] for t in self._pending_tasks]:
                if task_id in self._tasks:
                    task = self._tasks[task_id]
                    
                    # Update age factor (0.0 to 1.0 based on wait time)
                    max_wait_time = 3600.0  # 1 hour
                    wait_time = current_time - task.created_at
                    task.age_factor = min(1.0, wait_time / max_wait_time)
                    
                    # Update dependency factor (0.0 to 1.0 based on number of dependent tasks)
                    if task_id in self._dependent_tasks:
                        num_dependents = len(self._dependent_tasks[task_id])
                        task.dependency_factor = min(1.0, num_dependents / 10.0)
                    
                    # Update resource factor (0.0 to 1.0 based on resource availability)
                    if self.resource_manager.can_allocate_resources(task.resource_requirements):
                        task.resource_factor = 1.0
                    else:
                        task.resource_factor = 0.0
    
    def _on_task_completed(self, task_id: str, result: Any) -> None:
        """
        Callback for when a task is completed.
        
        Args:
            task_id: ID of the completed task
            result: Result of the task
        """
        with self._queue_lock:
            if task_id not in self._tasks:
                logger.warning(f"Completed task {task_id} not found in tasks")
                return
            
            task = self._tasks[task_id]
            
            # Move task to completed tasks
            self._completed_tasks[task_id] = task
            self._running_tasks.remove(task_id)
            del self._tasks[task_id]
            
            logger.info(f"Task {task_id} completed")
            
            # Update dependent tasks
            if task_id in self._dependent_tasks:
                for dependent_id in self._dependent_tasks[task_id]:
                    if dependent_id in self._tasks:
                        dependent_task = self._tasks[dependent_id]
                        if task_id in dependent_task.dependencies:
                            dependent_task.dependencies.remove(task_id)
                            
                            # If no more dependencies, add to pending queue
                            if not dependent_task.dependencies and dependent_task.status == TaskStatus.PENDING:
                                heapq.heappush(
                                    self._pending_tasks,
                                    (-dependent_task.effective_priority, dependent_task.created_at, dependent_id)
                                )
                                logger.debug(f"Task {dependent_id} ready for execution")
                
                # Clear dependent tasks
                del self._dependent_tasks[task_id]
            
            # Trigger task completed event
            self._trigger_event('task_completed', task)
    
    def _on_task_failed(self, task_id: str, error: Exception) -> None:
        """
        Callback for when a task fails.
        
        Args:
            task_id: ID of the failed task
            error: Error that caused the failure
        """
        with self._queue_lock:
            if task_id not in self._tasks:
                logger.warning(f"Failed task {task_id} not found in tasks")
                return
            
            task = self._tasks[task_id]
            
            # Check if task can be retried
            if task.retry():
                # Add back to pending queue
                heapq.heappush(self._pending_tasks, (-task.effective_priority, task.created_at, task_id))
                logger.info(f"Task {task_id} queued for retry (attempt {task.retry_count}/{task.max_retries})")
                
                # Trigger task retry event
                self._trigger_event('task_retry', task)
            else:
                # Move task to failed tasks
                self._failed_tasks[task_id] = task
                self._running_tasks.remove(task_id)
                del self._tasks[task_id]
                
                logger.info(f"Task {task_id} failed: {str(error)}")
                
                # Trigger task failed event
                self._trigger_event('task_failed', task)
    
    def _on_task_progress(self, task_id: str, current: int, total: int, message: str) -> None:
        """
        Callback for task progress updates.
        
        Args:
            task_id: ID of the task
            current: Current progress value
            total: Total progress value
            message: Progress message
        """
        task = self.get_task(task_id)
        if task:
            # Trigger task progress event
            self._trigger_event('task_progress', task)