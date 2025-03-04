"""
Example script demonstrating the usage of the configuration system.

This script shows how to:
1. Load and save configuration
2. Access and modify configuration values
3. Detect hardware capabilities
4. Manage ML model settings
5. Validate configuration
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path to allow importing the config module
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.backend.core.config import (
    ConfigManager,
    get_default_config,
    detect_hardware_capabilities,
    validate_config
)


def main():
    """Main function demonstrating configuration system usage."""
    print("Configuration System Example")
    print("===========================\n")
    
    # Create output directory
    output_dir = Path("config_example_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Create a default configuration file
    print("1. Creating default configuration...")
    config_path = output_dir / "config.json"
    default_config = get_default_config()
    
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    print(f"Default configuration saved to {config_path}\n")
    
    # 2. Load configuration using ConfigManager
    print("2. Loading configuration...")
    config_manager = ConfigManager(config_path)
    print("Configuration loaded successfully\n")
    
    # 3. Access configuration values
    print("3. Accessing configuration values:")
    app_name = config_manager.get("app.name")
    app_version = config_manager.get("app.version")
    log_level = config_manager.get("app.log_level")
    default_model = config_manager.get("models.default_model")
    
    print(f"  - App name: {app_name}")
    print(f"  - App version: {app_version}")
    print(f"  - Log level: {log_level}")
    print(f"  - Default model: {default_model}\n")
    
    # 4. Modify configuration values
    print("4. Modifying configuration values...")
    config_manager.set("app.log_level", "DEBUG")
    config_manager.set("models.default_model", "demucs")
    config_manager.set("ui.theme", "dark")
    
    # Save the modified configuration
    modified_config_path = output_dir / "modified_config.json"
    config_manager.save_config(modified_config_path)
    
    print(f"Modified configuration saved to {modified_config_path}")
    print(f"  - Log level changed to: {config_manager.get('app.log_level')}")
    print(f"  - Default model changed to: {config_manager.get('models.default_model')}")
    print(f"  - UI theme changed to: {config_manager.get('ui.theme')}\n")
    
    # 5. Detect hardware capabilities
    print("5. Detecting hardware capabilities...")
    hardware_capabilities = detect_hardware_capabilities()
    
    print("Hardware capabilities:")
    print(f"  - GPU enabled: {hardware_capabilities['gpu_enabled']}")
    if hardware_capabilities['gpu_enabled'] and hardware_capabilities['gpu_devices']:
        print(f"  - GPU devices: {len(hardware_capabilities['gpu_devices'])}")
        for i, device in enumerate(hardware_capabilities['gpu_devices']):
            print(f"    - Device {i+1}: {device['name']}")
            print(f"      Memory: {device['memory']} MB")
            print(f"      Compute capability: {device['compute_capability']}")
    
    print(f"  - CPU cores: {hardware_capabilities['cpu_cores']}")
    print(f"  - Available memory: {hardware_capabilities['available_memory']} MB")
    print(f"  - Recommended precision: {hardware_capabilities['recommended_precision']}")
    print(f"  - Recommended batch size: {hardware_capabilities['recommended_batch_size']}\n")
    
    # Save hardware capabilities to a file
    hardware_path = output_dir / "hardware_capabilities.json"
    with open(hardware_path, 'w') as f:
        json.dump(hardware_capabilities, f, indent=2)
    
    print(f"Hardware capabilities saved to {hardware_path}\n")
    
    # 6. Work with ML model settings
    print("6. Working with ML model settings...")
    
    # Get settings for a specific model
    svoice_settings = config_manager.get_model_settings("svoice")
    print("SVoice model settings:")
    print(f"  - Enabled: {svoice_settings['enabled']}")
    print(f"  - Priority: {svoice_settings['priority']}")
    print(f"  - Sample rate: {svoice_settings['parameters']['sample_rate']}")
    print(f"  - Hidden size: {svoice_settings['parameters']['hidden_size']}")
    
    # Check model compatibility with hardware
    is_compatible, issues = config_manager.is_model_compatible("svoice")
    print(f"\nSVoice compatibility with current hardware: {is_compatible}")
    if not is_compatible:
        print("Compatibility issues:")
        for issue in issues:
            print(f"  - {issue}")
    
    # Get optimized model settings
    optimized_settings = config_manager.get_optimal_model_settings("svoice")
    print("\nOptimized SVoice settings for current hardware:")
    for key, value in optimized_settings.items():
        print(f"  - {key}: {value}")
    
    # Update model settings
    print("\nUpdating model settings...")
    config_manager.update_model_settings("svoice", {
        "parameters": {
            "hidden_size": 256,
            "num_layers": 8
        }
    })
    
    # Save the updated configuration
    updated_config_path = output_dir / "updated_config.json"
    config_manager.save_config(updated_config_path)
    
    print(f"Updated configuration saved to {updated_config_path}")
    print(f"  - SVoice hidden size changed to: {config_manager.get('models.model_settings.svoice.parameters.hidden_size')}")
    print(f"  - SVoice num layers changed to: {config_manager.get('models.model_settings.svoice.parameters.num_layers')}\n")
    
    # 7. Validate configuration
    print("7. Validating configuration...")
    errors = validate_config(config_manager.config)
    
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration is valid\n")
    
    # 8. Create an invalid configuration for demonstration
    print("8. Creating and validating an invalid configuration...")
    invalid_config = {
        "app": {
            "name": "Voices",
            "version": "0.1.0",
            "log_level": "INVALID_LEVEL"  # Invalid log level
        }
    }
    
    errors = validate_config(invalid_config)
    
    print("Invalid configuration validation errors:")
    for error in errors:
        print(f"  - {error}")
    
    print("\nConfiguration System Example completed successfully!")


if __name__ == "__main__":
    main()