"""
Command-line interface for the Model Registry System.

This module provides a command-line interface for interacting with the
model registry, allowing users to list, add, and manage models.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional

from .model_registry import ModelRegistry
from .model_adapters import create_model_adapter, get_model_loader


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("model_registry_cli")


def create_registry(registry_dir: str, logger: logging.Logger) -> ModelRegistry:
    """Create and initialize the model registry."""
    registry = ModelRegistry(registry_dir, logger)
    
    # Register model loaders
    registry.register_model_loader("svoice", get_model_loader("svoice"))
    registry.register_model_loader("demucs", get_model_loader("demucs"))
    
    return registry


def list_models(registry: ModelRegistry, args: argparse.Namespace) -> None:
    """List models in the registry."""
    models = registry.list_models(model_type=args.type, tags=args.tags)
    
    if not models:
        print("No models found in the registry.")
        return
    
    print(f"Found {len(models)} models in the registry:")
    print("-" * 80)
    
    for model in models:
        default_version = model.get_default_version()
        version_count = len(model.versions)
        
        print(f"ID: {model.model_id}")
        print(f"Name: {model.name}")
        print(f"Type: {model.model_type}")
        print(f"Description: {model.description}")
        print(f"Created: {model.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Versions: {version_count}")
        
        if default_version:
            print(f"Default Version: {default_version.version_id}")
            
            if default_version.performance_metrics:
                print("Performance Metrics:")
                for metric, value in default_version.performance_metrics.items():
                    print(f"  {metric}: {value}")
        
        print("-" * 80)


def show_model(registry: ModelRegistry, args: argparse.Namespace) -> None:
    """Show details of a specific model."""
    model_info = registry.get_model(args.model_id)
    
    if not model_info:
        print(f"Model {args.model_id} not found in the registry.")
        return
    
    print(f"Model: {model_info.name} ({model_info.model_id})")
    print(f"Type: {model_info.model_type}")
    print(f"Description: {model_info.description}")
    print(f"Created: {model_info.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Default Version: {model_info.default_version}")
    print("\nVersions:")
    print("-" * 80)
    
    for version_id, version in model_info.versions.items():
        print(f"Version ID: {version_id}")
        print(f"Description: {version.description}")
        print(f"Created: {version.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Path: {version.path}")
        print(f"Default: {'Yes' if version.is_default else 'No'}")
        
        if version.tags:
            print(f"Tags: {', '.join(version.tags)}")
        
        if version.performance_metrics:
            print("Performance Metrics:")
            for metric, value in version.performance_metrics.items():
                print(f"  {metric}: {value}")
        
        print("-" * 80)


def add_model(registry: ModelRegistry, args: argparse.Namespace) -> None:
    """Add a new model to the registry."""
    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist.")
        return
    
    # Parse tags if provided
    tags = args.tags.split(",") if args.tags else []
    
    # Parse metadata if provided
    metadata = {}
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            print("Error: Metadata must be a valid JSON string.")
            return
    
    # Add model to registry
    try:
        model_id, version_id = registry.add_model(
            name=args.name,
            description=args.description,
            model_type=args.type,
            model_path=args.model_path,
            version_description=args.version_description,
            parameters={},
            metadata=metadata,
            tags=tags
        )
        
        print(f"Successfully added model {args.name} with ID {model_id}")
        print(f"Initial version ID: {version_id}")
    
    except Exception as e:
        print(f"Error adding model: {str(e)}")


def add_version(registry: ModelRegistry, args: argparse.Namespace) -> None:
    """Add a new version to an existing model."""
    # Check if model exists
    model_info = registry.get_model(args.model_id)
    if not model_info:
        print(f"Error: Model {args.model_id} not found in the registry.")
        return
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist.")
        return
    
    # Parse tags if provided
    tags = args.tags.split(",") if args.tags else []
    
    # Parse metadata if provided
    metadata = {}
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            print("Error: Metadata must be a valid JSON string.")
            return
    
    # Add version to registry
    try:
        version_id = registry.add_model_version(
            model_id=args.model_id,
            model_path=args.model_path,
            version_description=args.description,
            parameters={},
            metadata=metadata,
            tags=tags,
            set_as_default=args.set_default
        )
        
        print(f"Successfully added version {version_id} to model {args.model_id}")
        if args.set_default:
            print(f"Set as default version")
    
    except Exception as e:
        print(f"Error adding version: {str(e)}")


def delete_model(registry: ModelRegistry, args: argparse.Namespace) -> None:
    """Delete a model from the registry."""
    # Check if model exists
    model_info = registry.get_model(args.model_id)
    if not model_info:
        print(f"Error: Model {args.model_id} not found in the registry.")
        return
    
    # Confirm deletion
    if not args.force:
        confirm = input(f"Are you sure you want to delete model {model_info.name} ({args.model_id})? [y/N] ")
        if confirm.lower() != "y":
            print("Deletion cancelled.")
            return
    
    # Delete model
    success = registry.delete_model(args.model_id)
    
    if success:
        print(f"Successfully deleted model {args.model_id}")
    else:
        print(f"Error deleting model {args.model_id}")


def delete_version(registry: ModelRegistry, args: argparse.Namespace) -> None:
    """Delete a version from a model."""
    # Check if model exists
    model_info = registry.get_model(args.model_id)
    if not model_info:
        print(f"Error: Model {args.model_id} not found in the registry.")
        return
    
    # Check if version exists
    if args.version_id not in model_info.versions:
        print(f"Error: Version {args.version_id} not found in model {args.model_id}.")
        return
    
    # Confirm deletion
    if not args.force:
        confirm = input(f"Are you sure you want to delete version {args.version_id} of model {model_info.name}? [y/N] ")
        if confirm.lower() != "y":
            print("Deletion cancelled.")
            return
    
    # Delete version
    success = registry.delete_model_version(args.model_id, args.version_id)
    
    if success:
        print(f"Successfully deleted version {args.version_id} of model {args.model_id}")
    else:
        print(f"Error deleting version {args.version_id} of model {args.model_id}")


def export_catalog(registry: ModelRegistry, args: argparse.Namespace) -> None:
    """Export a catalog of all models in the registry."""
    output_path = args.output
    
    if not output_path:
        output_path = "model_catalog.json"
    
    try:
        catalog_path = registry.export_model_catalog(output_path)
        print(f"Successfully exported model catalog to {catalog_path}")
    
    except Exception as e:
        print(f"Error exporting model catalog: {str(e)}")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Model Registry CLI")
    parser.add_argument("--registry-dir", default="./model_registry", help="Directory for the model registry")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List models command
    list_parser = subparsers.add_parser("list", help="List models in the registry")
    list_parser.add_argument("--type", help="Filter by model type")
    list_parser.add_argument("--tags", help="Filter by tags (comma-separated)")
    
    # Show model command
    show_parser = subparsers.add_parser("show", help="Show details of a specific model")
    show_parser.add_argument("model_id", help="ID of the model to show")
    
    # Add model command
    add_parser = subparsers.add_parser("add", help="Add a new model to the registry")
    add_parser.add_argument("--name", required=True, help="Name of the model")
    add_parser.add_argument("--description", required=True, help="Description of the model")
    add_parser.add_argument("--type", required=True, choices=["svoice", "demucs"], help="Type of model")
    add_parser.add_argument("--model-path", required=True, help="Path to the model file or directory")
    add_parser.add_argument("--version-description", default="Initial version", help="Description of the initial version")
    add_parser.add_argument("--tags", help="Tags for the model (comma-separated)")
    add_parser.add_argument("--metadata", help="Additional metadata as JSON string")
    
    # Add version command
    add_version_parser = subparsers.add_parser("add-version", help="Add a new version to an existing model")
    add_version_parser.add_argument("model_id", help="ID of the model")
    add_version_parser.add_argument("--model-path", required=True, help="Path to the model file or directory")
    add_version_parser.add_argument("--description", required=True, help="Description of the version")
    add_version_parser.add_argument("--tags", help="Tags for the version (comma-separated)")
    add_version_parser.add_argument("--metadata", help="Additional metadata as JSON string")
    add_version_parser.add_argument("--set-default", action="store_true", help="Set as default version")
    
    # Delete model command
    delete_parser = subparsers.add_parser("delete", help="Delete a model from the registry")
    delete_parser.add_argument("model_id", help="ID of the model to delete")
    delete_parser.add_argument("--force", "-f", action="store_true", help="Force deletion without confirmation")
    
    # Delete version command
    delete_version_parser = subparsers.add_parser("delete-version", help="Delete a version from a model")
    delete_version_parser.add_argument("model_id", help="ID of the model")
    delete_version_parser.add_argument("version_id", help="ID of the version to delete")
    delete_version_parser.add_argument("--force", "-f", action="store_true", help="Force deletion without confirmation")
    
    # Export catalog command
    export_parser = subparsers.add_parser("export", help="Export a catalog of all models")
    export_parser.add_argument("--output", "-o", help="Output file path")
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.verbose)
    
    # Create registry
    registry_dir = os.path.abspath(args.registry_dir)
    os.makedirs(registry_dir, exist_ok=True)
    registry = create_registry(registry_dir, logger)
    
    # Execute command
    if args.command == "list":
        list_models(registry, args)
    elif args.command == "show":
        show_model(registry, args)
    elif args.command == "add":
        add_model(registry, args)
    elif args.command == "add-version":
        add_version(registry, args)
    elif args.command == "delete":
        delete_model(registry, args)
    elif args.command == "delete-version":
        delete_version(registry, args)
    elif args.command == "export":
        export_catalog(registry, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()