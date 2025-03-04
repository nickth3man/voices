"""
Resource management for the processing queue.

This module provides functionality for managing system resources and
allocating them to tasks based on their requirements and priorities.
"""

import os
import logging
import platform
import threading
import psutil
from typing import Dict, List, Any, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    Manages system resources and allocates them to tasks.
    
    This class is responsible for tracking available system resources,
    determining if tasks can be executed based on their resource requirements,
    and allocating resources to tasks.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the resource manager.
        
        Args:
            config: Configuration dictionary for resource management
        """
        self.config = config or {}
        self._lock = threading.RLock()
        
        # Resource limits (0 means no limit)
        self.max_cpu_percent = self.config.get('max_cpu_percent', 90)
        self.max_memory_percent = self.config.get('max_memory_percent', 90)
        self.max_gpu_percent = self.config.get('max_gpu_percent', 90)
        self.max_parallel_tasks = self.config.get('max_parallel_tasks', 0)
        
        # Resource reservations
        self.reserved_cpu_percent = self.config.get('reserved_cpu_percent', 10)
        self.reserved_memory_percent = self.config.get('reserved_memory_percent', 10)
        self.reserved_gpu_percent = self.config.get('reserved_gpu_percent', 10)
        
        # Currently allocated resources
        self.allocated_resources = {
            'cpu_percent': 0,
            'memory_bytes': 0,
            'gpu_percent': 0,
            'task_count': 0
        }
        
        # Resource allocation by task
        self.task_allocations = {}
        
        # Initialize system info
        self.update_system_info()
        
        logger.info(f"Resource manager initialized with {self.cpu_count} CPUs, "
                   f"{self.format_bytes(self.total_memory)} memory")
        
        # Check for GPU availability
        if self.has_gpu:
            logger.info(f"GPU detected: {self.gpu_info}")
    
    def update_system_info(self) -> None:
        """Update system resource information."""
        with self._lock:
            # CPU info
            self.cpu_count = psutil.cpu_count(logical=True)
            self.cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory info
            memory = psutil.virtual_memory()
            self.total_memory = memory.total
            self.available_memory = memory.available
            self.used_memory = memory.used
            self.memory_percent = memory.percent
            
            # GPU info (placeholder - in a real implementation, we would use a library like pynvml)
            self.has_gpu = False
            self.gpu_info = {}
            self.gpu_memory_total = 0
            self.gpu_memory_used = 0
            self.gpu_percent = 0
            
            # Try to detect NVIDIA GPU using a simple approach
            try:
                # This is a placeholder. In a real implementation, we would use a proper GPU detection library
                if platform.system() == 'Windows':
                    import subprocess
                    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', 
                                           '--format=csv,noheader,nounits'], 
                                          capture_output=True, text=True, check=False)
                    if result.returncode == 0 and result.stdout.strip():
                        self.has_gpu = True
                        # Parse the output (this is simplified)
                        parts = result.stdout.strip().split(',')
                        if len(parts) >= 4:
                            self.gpu_info = {
                                'name': parts[0].strip(),
                                'memory_total': int(parts[1].strip()) * 1024 * 1024,  # Convert to bytes
                                'memory_used': int(parts[2].strip()) * 1024 * 1024,   # Convert to bytes
                                'utilization': float(parts[3].strip())
                            }
                            self.gpu_memory_total = self.gpu_info['memory_total']
                            self.gpu_memory_used = self.gpu_info['memory_used']
                            self.gpu_percent = self.gpu_info['utilization']
            except Exception as e:
                logger.warning(f"Error detecting GPU: {str(e)}")
    
    def format_bytes(self, bytes_value: int) -> str:
        """
        Format bytes as a human-readable string.
        
        Args:
            bytes_value: Number of bytes
            
        Returns:
            Human-readable string (e.g., "1.23 GB")
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024 or unit == 'TB':
                return f"{bytes_value:.2f} {unit}"
            bytes_value /= 1024
    
    def get_available_resources(self) -> Dict[str, Any]:
        """
        Get information about available system resources.
        
        Returns:
            Dictionary with resource information
        """
        self.update_system_info()
        
        with self._lock:
            # Calculate available resources
            available_cpu_percent = max(0, self.max_cpu_percent - self.cpu_percent - self.allocated_resources['cpu_percent'])
            available_memory = max(0, self.available_memory - self.allocated_resources['memory_bytes'])
            
            # GPU resources (if available)
            available_gpu_percent = 0
            if self.has_gpu:
                available_gpu_percent = max(0, self.max_gpu_percent - self.gpu_percent - self.allocated_resources['gpu_percent'])
            
            # Task count
            available_task_slots = 0
            if self.max_parallel_tasks > 0:
                available_task_slots = max(0, self.max_parallel_tasks - self.allocated_resources['task_count'])
            else:
                available_task_slots = self.cpu_count - self.allocated_resources['task_count']
            
            return {
                'cpu_percent': available_cpu_percent,
                'memory_bytes': available_memory,
                'memory_formatted': self.format_bytes(available_memory),
                'gpu_percent': available_gpu_percent,
                'task_slots': available_task_slots,
                'has_gpu': self.has_gpu
            }
    
    def can_allocate_resources(self, requirements: Dict[str, Any]) -> bool:
        """
        Check if the required resources can be allocated.
        
        Args:
            requirements: Dictionary of resource requirements
            
        Returns:
            True if resources can be allocated, False otherwise
        """
        available = self.get_available_resources()
        
        # Check CPU requirements
        required_cpu = requirements.get('cpu_percent', 0)
        if required_cpu > available['cpu_percent']:
            logger.debug(f"Cannot allocate CPU: required {required_cpu}%, available {available['cpu_percent']}%")
            return False
        
        # Check memory requirements
        required_memory = requirements.get('memory_bytes', 0)
        if required_memory > available['memory_bytes']:
            logger.debug(f"Cannot allocate memory: required {self.format_bytes(required_memory)}, "
                        f"available {available['memory_formatted']}")
            return False
        
        # Check GPU requirements
        if requirements.get('requires_gpu', False) and not available['has_gpu']:
            logger.debug("Cannot allocate GPU: GPU required but not available")
            return False
        
        required_gpu = requirements.get('gpu_percent', 0)
        if required_gpu > available['gpu_percent']:
            logger.debug(f"Cannot allocate GPU: required {required_gpu}%, available {available['gpu_percent']}%")
            return False
        
        # Check task slot availability
        if available['task_slots'] <= 0:
            logger.debug("Cannot allocate task slot: no slots available")
            return False
        
        return True
    
    def allocate_resources(self, task_id: str, requirements: Dict[str, Any]) -> bool:
        """
        Allocate resources to a task.
        
        Args:
            task_id: ID of the task
            requirements: Dictionary of resource requirements
            
        Returns:
            True if resources were allocated, False otherwise
        """
        if not self.can_allocate_resources(requirements):
            return False
        
        with self._lock:
            # Calculate resource requirements
            cpu_percent = requirements.get('cpu_percent', 0)
            memory_bytes = requirements.get('memory_bytes', 0)
            gpu_percent = requirements.get('gpu_percent', 0)
            
            # Allocate resources
            self.allocated_resources['cpu_percent'] += cpu_percent
            self.allocated_resources['memory_bytes'] += memory_bytes
            self.allocated_resources['gpu_percent'] += gpu_percent
            self.allocated_resources['task_count'] += 1
            
            # Record allocation for this task
            self.task_allocations[task_id] = {
                'cpu_percent': cpu_percent,
                'memory_bytes': memory_bytes,
                'gpu_percent': gpu_percent
            }
            
            logger.debug(f"Allocated resources for task {task_id}: "
                        f"CPU: {cpu_percent}%, Memory: {self.format_bytes(memory_bytes)}, "
                        f"GPU: {gpu_percent}%")
            
            return True
    
    def release_resources(self, task_id: str) -> None:
        """
        Release resources allocated to a task.
        
        Args:
            task_id: ID of the task
        """
        with self._lock:
            if task_id not in self.task_allocations:
                logger.warning(f"Cannot release resources for unknown task {task_id}")
                return
            
            # Get allocated resources
            allocation = self.task_allocations[task_id]
            
            # Release resources
            self.allocated_resources['cpu_percent'] -= allocation['cpu_percent']
            self.allocated_resources['memory_bytes'] -= allocation['memory_bytes']
            self.allocated_resources['gpu_percent'] -= allocation['gpu_percent']
            self.allocated_resources['task_count'] -= 1
            
            # Ensure we don't go below zero
            for key in self.allocated_resources:
                if isinstance(self.allocated_resources[key], (int, float)):
                    self.allocated_resources[key] = max(0, self.allocated_resources[key])
            
            # Remove task allocation
            del self.task_allocations[task_id]
            
            logger.debug(f"Released resources for task {task_id}")
    
    def estimate_resource_requirements(self, task_type: str, input_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Estimate resource requirements for a task based on its type and input size.
        
        Args:
            task_type: Type of the task
            input_size: Size of the input data in bytes
            
        Returns:
            Dictionary of estimated resource requirements
        """
        # Default requirements
        requirements = {
            'cpu_percent': 10,
            'memory_bytes': 100 * 1024 * 1024,  # 100 MB
            'gpu_percent': 0,
            'requires_gpu': False
        }
        
        # Adjust based on task type
        if task_type == 'audio_processing':
            requirements['cpu_percent'] = 50
            requirements['memory_bytes'] = 500 * 1024 * 1024  # 500 MB
        elif task_type == 'voice_separation':
            requirements['cpu_percent'] = 80
            requirements['memory_bytes'] = 2 * 1024 * 1024 * 1024  # 2 GB
            requirements['gpu_percent'] = 50
            requirements['requires_gpu'] = True
        elif task_type == 'batch_processing':
            requirements['cpu_percent'] = 70
            requirements['memory_bytes'] = 1 * 1024 * 1024 * 1024  # 1 GB
        
        # Adjust based on input size if provided
        if input_size is not None:
            # Scale memory requirements based on input size
            # This is a simplified approach - in a real implementation, we would use a more sophisticated model
            if input_size > 100 * 1024 * 1024:  # > 100 MB
                scale_factor = input_size / (100 * 1024 * 1024)
                requirements['memory_bytes'] = int(requirements['memory_bytes'] * min(scale_factor, 10))
        
        return requirements
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """
        Get current resource usage information.
        
        Returns:
            Dictionary with resource usage information
        """
        self.update_system_info()
        
        with self._lock:
            return {
                'cpu': {
                    'total': self.cpu_count,
                    'percent': self.cpu_percent,
                    'allocated': self.allocated_resources['cpu_percent']
                },
                'memory': {
                    'total': self.total_memory,
                    'total_formatted': self.format_bytes(self.total_memory),
                    'used': self.used_memory,
                    'used_formatted': self.format_bytes(self.used_memory),
                    'available': self.available_memory,
                    'available_formatted': self.format_bytes(self.available_memory),
                    'percent': self.memory_percent,
                    'allocated': self.allocated_resources['memory_bytes'],
                    'allocated_formatted': self.format_bytes(self.allocated_resources['memory_bytes'])
                },
                'gpu': {
                    'available': self.has_gpu,
                    'info': self.gpu_info,
                    'percent': self.gpu_percent,
                    'allocated': self.allocated_resources['gpu_percent']
                },
                'tasks': {
                    'count': self.allocated_resources['task_count'],
                    'max': self.max_parallel_tasks if self.max_parallel_tasks > 0 else self.cpu_count
                }
            }