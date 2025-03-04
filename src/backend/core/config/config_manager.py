"""
Configuration manager for loading, saving, and accessing configuration settings.

This module provides the ConfigManager class for managing application configuration,
including loading from files, saving to files, and accessing configuration values.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

from .schema import validate_config, ConfigSchema
from .defaults import get_default_config, get_environment_specific_config, merge_configs
from .hardware_detection import detect_hardware_capabilities, get_model_compatibility, get_optimal_model_settings

logger = logging.getLogger(__name__)


class ConfigManager:
    """Configuration manager for the application.
    
    This class handles loading, saving, and accessing configuration settings,
    including ML model settings and hardware-specific configurations.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize the configuration manager.
        
        Args:
            config_path: Optional path to the configuration file
        """
        self._config = None
        self._config_path = config_path
        self._hardware_capabilities = None
        self.load_config(config_path)

    @property
    def config(self) -> Dict[str, Any]:
        """Get the current configuration.
        
        Returns:
            Dict[str, Any]: The current configuration
        """
        return self._config

    @property
    def hardware_capabilities(self) -> Dict[str, Any]:
        """Get the detected hardware capabilities.
        
        Returns:
            Dict[str, Any]: The detected hardware capabilities
        """
        if self._hardware_capabilities is None:
            self._hardware_capabilities = detect_hardware_capabilities()
        return self._hardware_capabilities

    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Load configuration from a file or use defaults.
        
        Args:
            config_path: Optional path to the configuration file
            
        Returns:
            Dict[str, Any]: The loaded configuration
        """
        # Update config path if provided
        if config_path is not None:
            self._config_path = config_path
        
        # Start with default configuration
        config = get_default_config()
        
        # Apply environment-specific overrides
        env_config = get_environment_specific_config()
        if env_config:
            config = merge_configs(config, env_config)
        
        # Load from file if available
        if self._config_path and os.path.exists(self._config_path):
            try:
                with open(self._config_path, 'r') as f:
                    file_config = json.load(f)
                
                # Validate file configuration
                errors = validate_config(file_config)
                if errors:
                    logger.warning(f"Configuration validation errors: {errors}")
                    logger.warning("Using default configuration with partial overrides")
                    
                    # Apply only valid parts of the file configuration
                    config = self._apply_partial_config(config, file_config)
                else:
                    # Apply full file configuration
                    config = merge_configs(config, file_config)
                    
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load configuration from {self._config_path}: {e}")
                logger.info("Using default configuration")
        else:
            logger.info("No configuration file found, using default configuration")
        
        # Update hardware section with latest capabilities
        config["hardware"].update(self.hardware_capabilities)
        
        # Store the configuration
        self._config = config
        
        return config

    def save_config(self, config_path: Optional[Union[str, Path]] = None) -> bool:
        """Save the current configuration to a file.
        
        Args:
            config_path: Optional path to save the configuration file
            
        Returns:
            bool: True if the configuration was saved successfully, False otherwise
        """
        # Update config path if provided
        if config_path is not None:
            self._config_path = config_path
        
        # Check if config path is available
        if not self._config_path:
            logger.error("No configuration path specified")
            return False
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
        
        try:
            with open(self._config_path, 'w') as f:
                json.dump(self._config, f, indent=2)
            logger.info(f"Configuration saved to {self._config_path}")
            return True
        except IOError as e:
            logger.error(f"Failed to save configuration to {self._config_path}: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key.
        
        Args:
            key: The configuration key (dot notation for nested keys)
            default: Default value if the key is not found
            
        Returns:
            Any: The configuration value or default
        """
        if not self._config:
            return default
        
        # Handle nested keys with dot notation
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value by key.
        
        Args:
            key: The configuration key (dot notation for nested keys)
            value: The value to set
        """
        if not self._config:
            self._config = {}
        
        # Handle nested keys with dot notation
        keys = key.split('.')
        config = self._config
        
        # Navigate to the nested dictionary
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value

    def reset_to_defaults(self) -> None:
        """Reset the configuration to default values."""
        self._config = get_default_config()
        
        # Update hardware section with latest capabilities
        self._config["hardware"].update(self.hardware_capabilities)

    def validate(self) -> List[str]:
        """Validate the current configuration.
        
        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        if not self._config:
            return ["No configuration loaded"]
        
        return validate_config(self._config)

    def get_model_settings(self, model_id: str) -> Dict[str, Any]:
        """Get settings for a specific ML model.
        
        Args:
            model_id: The model identifier
            
        Returns:
            Dict[str, Any]: The model settings
        """
        model_settings = self.get(f"models.model_settings.{model_id}")
        if not model_settings:
            logger.warning(f"No settings found for model {model_id}, using defaults")
            # Get default settings for this model
            default_settings = get_default_config().get("models", {}).get("model_settings", {}).get(model_id, {})
            return default_settings
        
        return model_settings

    def is_model_compatible(self, model_id: str) -> tuple[bool, List[str]]:
        """Check if a model is compatible with the current hardware.
        
        Args:
            model_id: The model identifier
            
        Returns:
            tuple[bool, List[str]]: (is_compatible, list of compatibility issues)
        """
        model_settings = self.get_model_settings(model_id)
        if not model_settings:
            return False, [f"Model {model_id} not found in configuration"]
        
        hardware_requirements = model_settings.get("hardware_requirements", {})
        return get_model_compatibility(hardware_requirements, self.hardware_capabilities)

    def get_optimal_model_settings(self, model_id: str) -> Dict[str, Any]:
        """Get optimal settings for a model based on hardware capabilities.
        
        Args:
            model_id: The model identifier
            
        Returns:
            Dict[str, Any]: Optimized model settings
        """
        model_settings = self.get_model_settings(model_id)
        if not model_settings:
            return {}
        
        model_config = model_settings.get("parameters", {})
        return get_optimal_model_settings(model_id, model_config, self.hardware_capabilities)

    def update_model_settings(self, model_id: str, settings: Dict[str, Any]) -> None:
        """Update settings for a specific ML model.
        
        Args:
            model_id: The model identifier
            settings: The settings to update
        """
        current_settings = self.get_model_settings(model_id) or {}
        
        # Update parameters if provided
        if "parameters" in settings and isinstance(settings["parameters"], dict):
            current_parameters = current_settings.get("parameters", {})
            current_parameters.update(settings["parameters"])
            current_settings["parameters"] = current_parameters
            
            # Remove parameters from settings to avoid duplication
            settings_copy = settings.copy()
            del settings_copy["parameters"]
        else:
            settings_copy = settings
        
        # Update other settings
        current_settings.update(settings_copy)
        
        # Set the updated settings
        self.set(f"models.model_settings.{model_id}", current_settings)

    def _apply_partial_config(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply partial configuration, skipping invalid sections.
        
        Args:
            base_config: Base configuration
            override_config: Configuration to override base
            
        Returns:
            Dict[str, Any]: Merged configuration with only valid parts applied
        """
        result = base_config.copy()
        
        # Apply top-level sections that pass validation
        for key, value in override_config.items():
            if key in base_config:
                # Create a test config with just this section
                test_config = base_config.copy()
                test_config[key] = value
                
                # Validate the test config
                errors = validate_config(test_config)
                
                if not errors:
                    # Section is valid, apply it
                    result[key] = value
                else:
                    logger.warning(f"Skipping invalid configuration section '{key}': {errors}")
        
        return result