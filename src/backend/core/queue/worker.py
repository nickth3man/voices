"""
Worker implementation for the processing queue.

This module provides the Worker class for executing tasks from the queue.
"""

import os
import time
import uuid
import logging
import threading
import traceback
from typing import Dict, List, Any, Optional, Callable, Union

from .task import Task, TaskStatus

logger = logging.getLogger(__name__)


class Worker:
    """
    Worker for executing tasks from the queue.
    
    This class is responsible for executing tasks, handling errors,
    and reporting progress and results.
    """
    
    def __init__(
        self,
        worker_id: Optional[str] = None,
        name: Optional[str] = None,
        resource_manager = None,
        on_task_completed: Optional[Callable[[str, Any], None]] = None,
        on_task_failed: Optional[Callable[[str, Exception], None]] = None,
        on_task_progress: Optional[Callable[[str, int, int, str], None]] = None
    ):
        """
        Initialize a worker.
        
        Args:
            worker_id: Unique identifier for the worker (if None, generates a UUID)
            name: Name of the worker
            resource_manager: Resource manager instance
            on_task_completed: Callback for when a task is completed
            on_task_failed: Callback for when a task fails
            on_task_progress: Callback for task progress updates
        """
        self.worker_id = worker_id or str(uuid.uuid4())
        self.name = name or f"Worker-{self.worker_id[:8]}"
        self.resource_manager = resource_manager
        
        # Callbacks
        self.on_task_completed = on_task_completed
        self.on_task_failed = on_task_failed
        self.on_task_progress = on_task_progress
        
        # Worker state
        self.current_task = None
        self.is_running = False
        self.thread = None
        self._stop_event = threading.Event()
        
        logger.info(f"Worker {self.name} initialized")
    
    def start(self) -> None:
        """Start the worker."""
        if self.is_running:
            logger.warning(f"Worker {self.name} is already running")
            return
        
        self.is_running = True
        self._stop_event.clear()
        logger.info(f"Worker {self.name} started")
    
    def stop(self) -> None:
        """Stop the worker."""
        if not self.is_running:
            return
        
        self.is_running = False
        self._stop_event.set()
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        
        logger.info(f"Worker {self.name} stopped")
    
    def execute_task(self, task: Task) -> None:
        """
        Execute a task.
        
        Args:
            task: Task to execute
        """
        if not self.is_running:
            logger.warning(f"Worker {self.name} is not running, cannot execute task {task.task_id}")
            return
        
        if self.current_task:
            logger.warning(f"Worker {self.name} is already executing task {self.current_task.task_id}, "
                          f"cannot execute task {task.task_id}")
            return
        
        self.current_task = task
        
        # Create a thread for task execution
        self.thread = threading.Thread(
            target=self._execute_task_thread,
            args=(task,),
            name=f"Worker-{self.worker_id[:8]}-Task-{task.task_id[:8]}"
        )
        self.thread.daemon = True
        self.thread.start()
    
    def _execute_task_thread(self, task: Task) -> None:
        """
        Thread function for executing a task.
        
        Args:
            task: Task to execute
        """
        logger.info(f"Worker {self.name} executing task {task.task_id}: {task.name}")
        
        # Mark task as running
        task.mark_running(self.worker_id)
        
        # Allocate resources if resource manager is available
        if self.resource_manager and task.resource_requirements:
            self.resource_manager.allocate_resources(task.task_id, task.resource_requirements)
        
        try:
            # Set up progress callback
            def progress_callback(current: int, total: Optional[int] = None, message: Optional[str] = None) -> None:
                task.update_progress(current, total, message)
                if self.on_task_progress:
                    self.on_task_progress(task.task_id, current, task.progress.total, message or "")
            
            # Add progress callback to kwargs if the function accepts it
            kwargs = task.kwargs.copy()
            kwargs['progress_callback'] = progress_callback
            
            # Execute the task function
            start_time = time.time()
            result = task.function(*task.args, **kwargs)
            execution_time = time.time() - start_time
            
            # Check if task was cancelled during execution
            if self._stop_event.is_set() or task.status == TaskStatus.CANCELLED:
                logger.info(f"Task {task.task_id} was cancelled during execution")
                task.mark_cancelled()
                return
            
            # Mark task as completed
            task.mark_completed(result)
            logger.info(f"Task {task.task_id} completed in {execution_time:.2f}s")
            
            # Call completion callback
            if self.on_task_completed:
                self.on_task_completed(task.task_id, result)
        
        except Exception as e:
            # Log the error
            logger.error(f"Error executing task {task.task_id}: {str(e)}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            
            # Mark task as failed
            task.mark_failed(e)
            
            # Call failure callback
            if self.on_task_failed:
                self.on_task_failed(task.task_id, e)
        
        finally:
            # Release resources if resource manager is available
            if self.resource_manager and task.resource_requirements:
                self.resource_manager.release_resources(task.task_id)
            
            # Clear current task
            self.current_task = None
    
    def cancel_current_task(self) -> bool:
        """
        Cancel the current task.
        
        Returns:
            True if a task was cancelled, False otherwise
        """
        if not self.current_task:
            return False
        
        task = self.current_task
        logger.info(f"Cancelling task {task.task_id}")
        
        # Mark task as cancelled
        task.mark_cancelled()
        
        # Stop the worker thread
        self._stop_event.set()
        
        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        
        # Reset worker state
        self._stop_event.clear()
        self.current_task = None
        
        return True
    
    def is_idle(self) -> bool:
        """
        Check if the worker is idle.
        
        Returns:
            True if the worker is idle, False otherwise
        """
        return self.is_running and self.current_task is None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the worker.
        
        Returns:
            Dictionary with worker status information
        """
        status = {
            'worker_id': self.worker_id,
            'name': self.name,
            'is_running': self.is_running,
            'is_idle': self.is_idle(),
            'current_task': None
        }
        
        if self.current_task:
            status['current_task'] = {
                'task_id': self.current_task.task_id,
                'name': self.current_task.name,
                'status': self.current_task.status.name,
                'progress': {
                    'current': self.current_task.progress.current,
                    'total': self.current_task.progress.total,
                    'percentage': self.current_task.progress.percentage,
                    'message': self.current_task.progress.message
                }
            }
        
        return status


class WorkerPool:
    """
    Pool of workers for executing tasks.
    
    This class manages a pool of workers for executing tasks in parallel.
    """
    
    def __init__(
        self,
        num_workers: int = 0,
        resource_manager = None,
        on_task_completed: Optional[Callable[[str, Any], None]] = None,
        on_task_failed: Optional[Callable[[str, Exception], None]] = None,
        on_task_progress: Optional[Callable[[str, int, int, str], None]] = None
    ):
        """
        Initialize a worker pool.
        
        Args:
            num_workers: Number of workers to create (0 for auto-detection)
            resource_manager: Resource manager instance
            on_task_completed: Callback for when a task is completed
            on_task_failed: Callback for when a task fails
            on_task_progress: Callback for task progress updates
        """
        self.resource_manager = resource_manager
        
        # Callbacks
        self.on_task_completed = on_task_completed
        self.on_task_failed = on_task_failed
        self.on_task_progress = on_task_progress
        
        # Determine number of workers
        if num_workers <= 0:
            # Auto-detect based on CPU cores
            import multiprocessing
            num_workers = max(1, multiprocessing.cpu_count() - 1)
        
        # Create workers
        self.workers = []
        for i in range(num_workers):
            worker = Worker(
                name=f"Worker-{i+1}",
                resource_manager=resource_manager,
                on_task_completed=on_task_completed,
                on_task_failed=on_task_failed,
                on_task_progress=on_task_progress
            )
            self.workers.append(worker)
        
        logger.info(f"Worker pool initialized with {len(self.workers)} workers")
    
    def start(self) -> None:
        """Start all workers in the pool."""
        for worker in self.workers:
            worker.start()
        
        logger.info(f"Worker pool started with {len(self.workers)} workers")
    
    def stop(self) -> None:
        """Stop all workers in the pool."""
        for worker in self.workers:
            worker.stop()
        
        logger.info("Worker pool stopped")
    
    def get_idle_worker(self) -> Optional[Worker]:
        """
        Get an idle worker from the pool.
        
        Returns:
            Idle worker or None if no idle workers are available
        """
        for worker in self.workers:
            if worker.is_idle():
                return worker
        
        return None
    
    def execute_task(self, task: Task) -> bool:
        """
        Execute a task using an available worker.
        
        Args:
            task: Task to execute
            
        Returns:
            True if the task was assigned to a worker, False otherwise
        """
        worker = self.get_idle_worker()
        if not worker:
            logger.debug(f"No idle workers available for task {task.task_id}")
            return False
        
        worker.execute_task(task)
        return True
    
    def cancel_all_tasks(self) -> int:
        """
        Cancel all running tasks.
        
        Returns:
            Number of tasks cancelled
        """
        count = 0
        for worker in self.workers:
            if worker.cancel_current_task():
                count += 1
        
        logger.info(f"Cancelled {count} tasks")
        return count
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the worker pool.
        
        Returns:
            Dictionary with worker pool status information
        """
        return {
            'total_workers': len(self.workers),
            'idle_workers': sum(1 for worker in self.workers if worker.is_idle()),
            'workers': [worker.get_status() for worker in self.workers]
        }