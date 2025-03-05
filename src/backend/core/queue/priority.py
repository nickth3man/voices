"""
Priority definitions for the processing queue.

This module provides the Priority enum and related functionality for
prioritizing tasks in the processing queue.
"""

from enum import Enum, auto
from typing import Dict, Any, Optional


class Priority(Enum):
    """Priority levels for processing tasks."""
    
    LOW = 0
    NORMAL = 50
    HIGH = 100
    CRITICAL = 200
    
    @classmethod
    def from_string(cls, priority_str: str) -> 'Priority':
        """
        Convert a string to a Priority enum value.
        
        Args:
            priority_str: String representation of priority
            
        Returns:
            Priority enum value
            
        Raises:
            ValueError: If the string is not a valid priority
        """
        priority_map = {
            'low': cls.LOW,
            'normal': cls.NORMAL,
            'high': cls.HIGH,
            'critical': cls.CRITICAL
        }
        
        priority_str = priority_str.lower()
        if priority_str not in priority_map:
            valid_priorities = ', '.join(priority_map.keys())
            raise ValueError(f"Invalid priority: {priority_str}. Valid priorities are: {valid_priorities}")
        
        return priority_map[priority_str]
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'Priority':
        """
        Extract priority from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Priority enum value
        """
        priority_str = config.get('priority', 'normal')
        return cls.from_string(priority_str)


def calculate_task_priority(
    base_priority: Priority,
    age_factor: float = 0.0,
    dependency_factor: float = 0.0,
    resource_factor: float = 0.0
) -> float:
    """
    Calculate the effective priority of a task based on various factors.
    
    Args:
        base_priority: Base priority level
        age_factor: Factor based on how long the task has been in the queue (0.0-1.0)
        dependency_factor: Factor based on task dependencies (0.0-1.0)
        resource_factor: Factor based on resource availability (0.0-1.0)
        
    Returns:
        Effective priority value (higher means higher priority)
    """
    # Start with the base priority value
    priority_value = base_priority.value
    
    # Age factor: Tasks get higher priority the longer they wait
    # This prevents starvation of low-priority tasks
    age_boost = 50 * age_factor
    
    # Dependency factor: Tasks with many dependent tasks get higher priority
    dependency_boost = 30 * dependency_factor
    
    # Resource factor: Tasks that can use currently available resources get higher priority
    resource_boost = 20 * resource_factor
    
    # Calculate effective priority
    effective_priority = priority_value + age_boost + dependency_boost + resource_boost
    
    return effective_priority