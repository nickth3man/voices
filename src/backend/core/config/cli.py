"""
Command-line interface for the configuration system.

This module provides a command-line interface for managing configuration settings,
including viewing, editing, and validating configuration files.
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .config_manager import ConfigManager
from .schema import validate_config
from .defaults import get_default_config
from .hardware_detection import detect_hardware_capabilities

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Configuration Management CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # View command
    view_parser = subparsers.add_parser("view", help="View configuration")
    view_parser.add_argument("--config", "-c", help="Path to configuration file")
    view_parser.add_argument("--section", "-s", help="Configuration section to view")
    view_parser.add_argument("--format", "-f", choices=["json", "yaml"], default="json", 
                            help="Output format")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a default configuration file")
    create_parser.add_argument("output", help="Output file path")
    create_parser.add_argument("--force", "-f", action="store_true", 
                              help="Overwrite existing file")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a configuration file")
    validate_parser.add_argument("config", help="Path to configuration file")
    
    # Edit command
    edit_parser = subparsers.add_parser("edit", help="Edit a configuration value")
    edit_parser.add_argument("--config", "-c", required=True, help="Path to configuration file")
    edit_parser.add_argument("--key", "-k", required=True, help="Configuration key (dot notation)")
    edit_parser.add_argument("--value", "-v", required=True, help="Value to set")
    edit_parser.add_argument("--type", "-t", choices=["string", "int", "float", "bool", "json"], 
                            default="string", help="Value type")
    
    # Hardware command
    hardware_parser = subparsers.add_parser("hardware", help="Detect hardware capabilities")
    hardware_parser.add_argument("--output", "-o", help="Output file path")
    hardware_parser.add_argument("--format", "-f", choices=["json", "yaml"], default="json", 
                                help="Output format")
    
    # Model command
    model_parser = subparsers.add_parser("model", help="Manage ML model settings")
    model_subparsers = model_parser.add_subparsers(dest="model_command", help="Model command")
    
    # Model list command
    model_list_parser = model_subparsers.add_parser("list", help="List available models")
    model_list_parser.add_argument("--config", "-c", help="Path to configuration file")
    
    # Model view command
    model_view_parser = model_subparsers.add_parser("view", help="View model settings")
    model_view_parser.add_argument("--config", "-c", help="Path to configuration file")
    model_view_parser.add_argument("--model", "-m", required=True, help="Model ID")
    model_view_parser.add_argument("--format", "-f", choices=["json", "yaml"], default="json", 
                                  help="Output format")
    
    # Model optimize command
    model_optimize_parser = model_subparsers.add_parser("optimize", 
                                                      help="Optimize model settings for hardware")
    model_optimize_parser.add_argument("--config", "-c", required=True, 
                                      help="Path to configuration file")
    model_optimize_parser.add_argument("--model", "-m", required=True, help="Model ID")
    model_optimize_parser.add_argument("--apply", "-a", action="store_true", 
                                      help="Apply optimized settings")
    
    # Model compatibility command
    model_compat_parser = model_subparsers.add_parser("compatibility", 
                                                    help="Check model compatibility with hardware")
    model_compat_parser.add_argument("--config", "-c", help="Path to configuration file")
    model_compat_parser.add_argument("--model", "-m", required=True, help="Model ID")
    
    return parser.parse_args()


def view_config(args):
    """View configuration settings."""
    config_manager = ConfigManager(args.config)
    
    if args.section:
        # View specific section
        section = config_manager.get(args.section)
        if section is None:
            logger.error(f"Section '{args.section}' not found in configuration")
            sys.exit(1)
        data = section
    else:
        # View entire configuration
        data = config_manager.config
    
    # Format output
    if args.format == "json":
        output = json.dumps(data, indent=2)
    elif args.format == "yaml":
        try:
            import yaml
            output = yaml.dump(data, default_flow_style=False)
        except ImportError:
            logger.warning("PyYAML not installed, falling back to JSON format")
            output = json.dumps(data, indent=2)
    
    print(output)


def create_config(args):
    """Create a default configuration file."""
    output_path = args.output
    
    # Check if file exists
    if os.path.exists(output_path) and not args.force:
        logger.error(f"File '{output_path}' already exists. Use --force to overwrite.")
        sys.exit(1)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Get default configuration
    config = get_default_config()
    
    # Write configuration to file
    try:
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Default configuration created at '{output_path}'")
    except IOError as e:
        logger.error(f"Failed to write configuration to '{output_path}': {e}")
        sys.exit(1)


def validate_config_file(args):
    """Validate a configuration file."""
    config_path = args.config
    
    # Check if file exists
    if not os.path.exists(config_path):
        logger.error(f"File '{config_path}' does not exist")
        sys.exit(1)
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load configuration from '{config_path}': {e}")
        sys.exit(1)
    
    # Validate configuration
    errors = validate_config(config)
    
    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)
    else:
        logger.info("Configuration validation successful")


def edit_config(args):
    """Edit a configuration value."""
    config_manager = ConfigManager(args.config)
    
    # Parse value based on type
    if args.type == "int":
        try:
            value = int(args.value)
        except ValueError:
            logger.error(f"Invalid integer value: {args.value}")
            sys.exit(1)
    elif args.type == "float":
        try:
            value = float(args.value)
        except ValueError:
            logger.error(f"Invalid float value: {args.value}")
            sys.exit(1)
    elif args.type == "bool":
        value = args.value.lower() in ("true", "yes", "1", "y")
    elif args.type == "json":
        try:
            value = json.loads(args.value)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON value: {args.value}")
            sys.exit(1)
    else:
        value = args.value
    
    # Set the value
    config_manager.set(args.key, value)
    
    # Save the configuration
    if config_manager.save_config():
        logger.info(f"Configuration updated: {args.key} = {value}")
    else:
        logger.error("Failed to save configuration")
        sys.exit(1)


def detect_hardware(args):
    """Detect hardware capabilities."""
    capabilities = detect_hardware_capabilities()
    
    # Format output
    if args.format == "json":
        output = json.dumps(capabilities, indent=2)
    elif args.format == "yaml":
        try:
            import yaml
            output = yaml.dump(capabilities, default_flow_style=False)
        except ImportError:
            logger.warning("PyYAML not installed, falling back to JSON format")
            output = json.dumps(capabilities, indent=2)
    
    # Print or save output
    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write(output)
            logger.info(f"Hardware capabilities saved to '{args.output}'")
        except IOError as e:
            logger.error(f"Failed to write to '{args.output}': {e}")
            sys.exit(1)
    else:
        print(output)


def list_models(args):
    """List available models."""
    config_manager = ConfigManager(args.config)
    
    model_settings = config_manager.get("models.model_settings", {})
    
    if not model_settings:
        logger.info("No models found in configuration")
        return
    
    print("Available models:")
    for model_id, settings in model_settings.items():
        enabled = settings.get("enabled", False)
        priority = settings.get("priority", 0)
        status = "Enabled" if enabled else "Disabled"
        
        # Check compatibility
        is_compatible, issues = config_manager.is_model_compatible(model_id)
        compatibility = "Compatible" if is_compatible else "Incompatible"
        
        print(f"  - {model_id} (Priority: {priority}, Status: {status}, Compatibility: {compatibility})")
        
        if not is_compatible:
            print("    Compatibility issues:")
            for issue in issues:
                print(f"      - {issue}")


def view_model(args):
    """View model settings."""
    config_manager = ConfigManager(args.config)
    
    model_settings = config_manager.get_model_settings(args.model)
    
    if not model_settings:
        logger.error(f"Model '{args.model}' not found in configuration")
        sys.exit(1)
    
    # Format output
    if args.format == "json":
        output = json.dumps(model_settings, indent=2)
    elif args.format == "yaml":
        try:
            import yaml
            output = yaml.dump(model_settings, default_flow_style=False)
        except ImportError:
            logger.warning("PyYAML not installed, falling back to JSON format")
            output = json.dumps(model_settings, indent=2)
    
    print(output)


def optimize_model(args):
    """Optimize model settings for hardware."""
    config_manager = ConfigManager(args.config)
    
    # Check if model exists
    model_settings = config_manager.get_model_settings(args.model)
    if not model_settings:
        logger.error(f"Model '{args.model}' not found in configuration")
        sys.exit(1)
    
    # Get optimized settings
    optimized_settings = config_manager.get_optimal_model_settings(args.model)
    
    if not optimized_settings:
        logger.error(f"Failed to optimize settings for model '{args.model}'")
        sys.exit(1)
    
    # Print optimized settings
    print(f"Optimized settings for model '{args.model}':")
    print(json.dumps(optimized_settings, indent=2))
    
    # Apply optimized settings if requested
    if args.apply:
        config_manager.update_model_settings(args.model, {"parameters": optimized_settings})
        
        if config_manager.save_config():
            logger.info(f"Optimized settings applied to model '{args.model}'")
        else:
            logger.error("Failed to save configuration")
            sys.exit(1)


def check_model_compatibility(args):
    """Check model compatibility with hardware."""
    config_manager = ConfigManager(args.config)
    
    # Check if model exists
    model_settings = config_manager.get_model_settings(args.model)
    if not model_settings:
        logger.error(f"Model '{args.model}' not found in configuration")
        sys.exit(1)
    
    # Check compatibility
    is_compatible, issues = config_manager.is_model_compatible(args.model)
    
    if is_compatible:
        print(f"Model '{args.model}' is compatible with current hardware")
    else:
        print(f"Model '{args.model}' is not compatible with current hardware:")
        for issue in issues:
            print(f"  - {issue}")
        
        # Print hardware capabilities
        print("\nCurrent hardware capabilities:")
        print(json.dumps(config_manager.hardware_capabilities, indent=2))


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    if args.command == "view":
        view_config(args)
    elif args.command == "create":
        create_config(args)
    elif args.command == "validate":
        validate_config_file(args)
    elif args.command == "edit":
        edit_config(args)
    elif args.command == "hardware":
        detect_hardware(args)
    elif args.command == "model":
        if args.model_command == "list":
            list_models(args)
        elif args.model_command == "view":
            view_model(args)
        elif args.model_command == "optimize":
            optimize_model(args)
        elif args.model_command == "compatibility":
            check_model_compatibility(args)
        else:
            logger.error("No model command specified")
            sys.exit(1)
    else:
        logger.error("No command specified")
        sys.exit(1)


if __name__ == "__main__":
    main()