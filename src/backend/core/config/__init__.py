"""
Configuration management module.

This module provides functionality for loading, saving, and managing configuration
settings for the application, including ML model settings and hardware-specific
configurations.
"""

from .config_manager import ConfigManager
from .schema import ConfigSchema, validate_config
from .defaults import get_default_config
from .hardware_detection import detect_hardware_capabilities

__all__ = [
    'ConfigManager',
    'ConfigSchema',
    'validate_config',
    'get_default_config',
    'detect_hardware_capabilities',
]