"""
Model Registry System for Voice Separation Models.

This module provides a comprehensive registry system for tracking, versioning,
and managing different voice separation models, including their metadata and
performance metrics.
"""

import os
import json
import yaml
import logging
import datetime
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from dataclasses import dataclass, field
import torch

# Import experiment framework components for integration
from ..experiment.framework import ExperimentResult


@dataclass
class ModelVersion:
    """A specific version of a model in the registry."""
    
    version_id: str
    model_id: str
    created_at: datetime.datetime
    path: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_default: bool = False
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version_id": self.version_id,
            "model_id": self.model_id,
            "created_at": self.created_at.isoformat(),
            "path": self.path,
            "description": self.description,
            "parameters": self.parameters,
            "performance_metrics": self.performance_metrics,
            "metadata": self.metadata,
            "is_default": self.is_default,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create from dictionary."""
        return cls(
            version_id=data["version_id"],
            model_id=data["model_id"],
            created_at=datetime.datetime.fromisoformat(data["created_at"]),
            path=data["path"],
            description=data["description"],
            parameters=data.get("parameters", {}),
            performance_metrics=data.get("performance_metrics", {}),
            metadata=data.get("metadata", {}),
            is_default=data.get("is_default", False),
            tags=data.get("tags", [])
        )


@dataclass
class ModelInfo:
    """Information about a model in the registry."""
    
    model_id: str
    name: str
    description: str
    model_type: str  # "svoice", "demucs", etc.
    created_at: datetime.datetime
    versions: Dict[str, ModelVersion] = field(default_factory=dict)
    default_version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "description": self.description,
            "model_type": self.model_type,
            "created_at": self.created_at.isoformat(),
            "versions": {k: v.to_dict() for k, v in self.versions.items()},
            "default_version": self.default_version,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelInfo':
        """Create from dictionary."""
        model_info = cls(
            model_id=data["model_id"],
            name=data["name"],
            description=data["description"],
            model_type=data["model_type"],
            created_at=datetime.datetime.fromisoformat(data["created_at"]),
            default_version=data.get("default_version"),
            metadata=data.get("metadata", {})
        )
        
        # Load versions
        for version_id, version_data in data.get("versions", {}).items():
            model_info.versions[version_id] = ModelVersion.from_dict(version_data)
        
        return model_info
    
    def get_default_version(self) -> Optional[ModelVersion]:
        """Get the default version of this model."""
        if self.default_version and self.default_version in self.versions:
            return self.versions[self.default_version]
        return None
    
    def add_version(self, version: ModelVersion) -> None:
        """Add a new version to this model."""
        self.versions[version.version_id] = version
        
        # If this is the first version or marked as default, set as default
        if not self.default_version or version.is_default:
            self.default_version = version.version_id
            version.is_default = True


class ModelRegistry:
    """Registry for voice separation models."""
    
    def __init__(
        self,
        registry_dir: str,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the model registry.
        
        Args:
            registry_dir: Directory to store the registry data
            logger: Logger instance
        """
        self.registry_dir = Path(registry_dir)
        self.logger = logger or logging.getLogger(__name__)
        self.models: Dict[str, ModelInfo] = {}
        self.model_loaders: Dict[str, Callable] = {}
        
        # Create directory structure
        self.models_dir = self.registry_dir / "models"
        self.metadata_dir = self.registry_dir / "metadata"
        self.index_path = self.registry_dir / "registry_index.json"
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Load registry if it exists
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load the registry from disk."""
        if not self.index_path.exists():
            self.logger.info("Registry index not found, creating new registry")
            return
        
        try:
            with open(self.index_path, 'r') as f:
                registry_data = json.load(f)
            
            for model_id, model_data in registry_data.items():
                self.models[model_id] = ModelInfo.from_dict(model_data)
            
            self.logger.info(f"Loaded registry with {len(self.models)} models")
        except Exception as e:
            self.logger.error(f"Error loading registry: {str(e)}")
    
    def _save_registry(self) -> None:
        """Save the registry to disk."""
        registry_data = {model_id: model.to_dict() for model_id, model in self.models.items()}
        
        with open(self.index_path, 'w') as f:
            json.dump(registry_data, f, indent=2)
        
        self.logger.info(f"Saved registry with {len(self.models)} models")
    
    def register_model_loader(self, model_type: str, loader_fn: Callable) -> None:
        """
        Register a model loader function for a specific model type.
        
        Args:
            model_type: Type of model (e.g., "svoice", "demucs")
            loader_fn: Function that loads a model from a path
        """
        self.model_loaders[model_type] = loader_fn
        self.logger.info(f"Registered model loader for type: {model_type}")
    
    def add_model(
        self,
        name: str,
        description: str,
        model_type: str,
        model_path: str,
        version_description: str = "Initial version",
        parameters: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
        tags: List[str] = None
    ) -> Tuple[str, str]:
        """
        Add a new model to the registry.
        
        Args:
            name: Name of the model
            description: Description of the model
            model_type: Type of model (e.g., "svoice", "demucs")
            model_path: Path to the model file or directory
            version_description: Description of the initial version
            parameters: Model parameters
            metadata: Additional metadata
            tags: Tags for categorizing the model
        
        Returns:
            Tuple of (model_id, version_id)
        """
        # Generate IDs
        timestamp = datetime.datetime.now()
        model_id = f"{model_type}_{name.lower().replace(' ', '_')}_{int(timestamp.timestamp())}"
        version_id = f"v1_{int(timestamp.timestamp())}"
        
        # Create model directory
        model_dir = self.models_dir / model_id
        version_dir = model_dir / version_id
        os.makedirs(version_dir, exist_ok=True)
        
        # Copy model files
        source_path = Path(model_path)
        if source_path.is_file():
            target_path = version_dir / source_path.name
            shutil.copy2(source_path, target_path)
            relative_path = str(target_path.relative_to(self.registry_dir))
        else:
            # Copy directory contents
            for item in source_path.glob('*'):
                if item.is_file():
                    shutil.copy2(item, version_dir / item.name)
                else:
                    shutil.copytree(item, version_dir / item.name)
            relative_path = str(version_dir.relative_to(self.registry_dir))
        
        # Create model version
        version = ModelVersion(
            version_id=version_id,
            model_id=model_id,
            created_at=timestamp,
            path=relative_path,
            description=version_description,
            parameters=parameters or {},
            metadata=metadata or {},
            is_default=True,
            tags=tags or []
        )
        
        # Create model info
        model_info = ModelInfo(
            model_id=model_id,
            name=name,
            description=description,
            model_type=model_type,
            created_at=timestamp,
            default_version=version_id,
            metadata=metadata or {}
        )
        
        # Add version to model
        model_info.add_version(version)
        
        # Add to registry
        self.models[model_id] = model_info
        
        # Save registry
        self._save_registry()
        
        self.logger.info(f"Added model {name} ({model_id}) with version {version_id}")
        
        return model_id, version_id
    
    def add_model_version(
        self,
        model_id: str,
        model_path: str,
        version_description: str,
        parameters: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
        tags: List[str] = None,
        set_as_default: bool = False
    ) -> str:
        """
        Add a new version to an existing model.
        
        Args:
            model_id: ID of the model
            model_path: Path to the model file or directory
            version_description: Description of the version
            parameters: Model parameters
            metadata: Additional metadata
            tags: Tags for categorizing the model version
            set_as_default: Whether to set this as the default version
        
        Returns:
            Version ID
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model_info = self.models[model_id]
        
        # Generate version ID
        timestamp = datetime.datetime.now()
        version_number = len(model_info.versions) + 1
        version_id = f"v{version_number}_{int(timestamp.timestamp())}"
        
        # Create version directory
        model_dir = self.models_dir / model_id
        version_dir = model_dir / version_id
        os.makedirs(version_dir, exist_ok=True)
        
        # Copy model files
        source_path = Path(model_path)
        if source_path.is_file():
            target_path = version_dir / source_path.name
            shutil.copy2(source_path, target_path)
            relative_path = str(target_path.relative_to(self.registry_dir))
        else:
            # Copy directory contents
            for item in source_path.glob('*'):
                if item.is_file():
                    shutil.copy2(item, version_dir / item.name)
                else:
                    shutil.copytree(item, version_dir / item.name)
            relative_path = str(version_dir.relative_to(self.registry_dir))
        
        # Create model version
        version = ModelVersion(
            version_id=version_id,
            model_id=model_id,
            created_at=timestamp,
            path=relative_path,
            description=version_description,
            parameters=parameters or {},
            metadata=metadata or {},
            is_default=set_as_default,
            tags=tags or []
        )
        
        # Add version to model
        model_info.add_version(version)
        
        # Update default version if requested
        if set_as_default:
            model_info.default_version = version_id
            
            # Update is_default flag for all versions
            for v in model_info.versions.values():
                v.is_default = (v.version_id == version_id)
        
        # Save registry
        self._save_registry()
        
        self.logger.info(f"Added version {version_id} to model {model_id}")
        
        return version_id
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get information about a model.
        
        Args:
            model_id: ID of the model
        
        Returns:
            ModelInfo object or None if not found
        """
        return self.models.get(model_id)
    
    def get_model_version(self, model_id: str, version_id: Optional[str] = None) -> Optional[ModelVersion]:
        """
        Get a specific version of a model.
        
        Args:
            model_id: ID of the model
            version_id: ID of the version, or None for default version
        
        Returns:
            ModelVersion object or None if not found
        """
        model_info = self.get_model(model_id)
        if not model_info:
            return None
        
        if version_id:
            return model_info.versions.get(version_id)
        else:
            return model_info.get_default_version()
    
    def list_models(
        self,
        model_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[ModelInfo]:
        """
        List models in the registry, optionally filtered by type or tags.
        
        Args:
            model_type: Filter by model type
            tags: Filter by tags (models with any of these tags will be included)
        
        Returns:
            List of ModelInfo objects
        """
        models = list(self.models.values())
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        if tags:
            filtered_models = []
            for model in models:
                # Check if any version has any of the specified tags
                has_tag = False
                for version in model.versions.values():
                    if any(tag in version.tags for tag in tags):
                        has_tag = True
                        break
                
                if has_tag:
                    filtered_models.append(model)
            
            models = filtered_models
        
        return models
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model from the registry.
        
        Args:
            model_id: ID of the model
        
        Returns:
            True if successful, False otherwise
        """
        if model_id not in self.models:
            return False
        
        # Remove model directory
        model_dir = self.models_dir / model_id
        if model_dir.exists():
            shutil.rmtree(model_dir)
        
        # Remove from registry
        del self.models[model_id]
        
        # Save registry
        self._save_registry()
        
        self.logger.info(f"Deleted model {model_id}")
        
        return True
    
    def delete_model_version(self, model_id: str, version_id: str) -> bool:
        """
        Delete a specific version of a model.
        
        Args:
            model_id: ID of the model
            version_id: ID of the version
        
        Returns:
            True if successful, False otherwise
        """
        model_info = self.get_model(model_id)
        if not model_info or version_id not in model_info.versions:
            return False
        
        # Check if this is the only version
        if len(model_info.versions) == 1:
            self.logger.warning(f"Cannot delete the only version of model {model_id}")
            return False
        
        # Check if this is the default version
        is_default = model_info.default_version == version_id
        
        # Remove version directory
        version_dir = self.models_dir / model_id / version_id
        if version_dir.exists():
            shutil.rmtree(version_dir)
        
        # Remove from model
        del model_info.versions[version_id]
        
        # Update default version if needed
        if is_default:
            # Set the most recent version as default
            versions = sorted(
                model_info.versions.values(),
                key=lambda v: v.created_at,
                reverse=True
            )
            if versions:
                model_info.default_version = versions[0].version_id
                versions[0].is_default = True
        
        # Save registry
        self._save_registry()
        
        self.logger.info(f"Deleted version {version_id} of model {model_id}")
        
        return True
    
    def load_model(
        self,
        model_id: str,
        version_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Load a model from the registry.
        
        Args:
            model_id: ID of the model
            version_id: ID of the version, or None for default version
            **kwargs: Additional arguments to pass to the model loader
        
        Returns:
            Loaded model
        
        Raises:
            ValueError: If model or version not found, or no loader available
        """
        # Get model version
        version = self.get_model_version(model_id, version_id)
        if not version:
            raise ValueError(f"Model {model_id} version {version_id or 'default'} not found")
        
        # Get model info
        model_info = self.get_model(model_id)
        if not model_info:
            raise ValueError(f"Model {model_id} not found")
        
        # Check if loader is available
        if model_info.model_type not in self.model_loaders:
            raise ValueError(f"No loader registered for model type {model_info.model_type}")
        
        # Get full path
        model_path = self.registry_dir / version.path
        
        # Load model
        loader = self.model_loaders[model_info.model_type]
        model = loader(model_path, **kwargs)
        
        self.logger.info(f"Loaded model {model_id} version {version.version_id}")
        
        return model
    
    def get_model_function(
        self,
        model_id: str,
        version_id: Optional[str] = None,
        **kwargs
    ) -> Callable:
        """
        Get a function that processes audio using the specified model.
        
        This is used to integrate with the experiment framework.
        
        Args:
            model_id: ID of the model
            version_id: ID of the version, or None for default version
            **kwargs: Additional arguments to pass to the model loader
        
        Returns:
            Function that takes audio input and returns separated sources
        
        Raises:
            ValueError: If model or version not found, or no loader available
        """
        # Pre-load the model
        model = self.load_model(model_id, version_id, **kwargs)
        
        # Get model info and version
        model_info = self.get_model(model_id)
        version = self.get_model_version(model_id, version_id)
        
        # Create a function that processes audio using this model
        def process_audio(mixture, **process_kwargs):
            """
            Process audio using the loaded model.
            
            Args:
                mixture: Audio mixture to separate
                **process_kwargs: Additional processing parameters
            
            Returns:
                Separated sources
            """
            # Combine parameters from model version with process-specific parameters
            combined_kwargs = {**version.parameters, **process_kwargs}
            
            # Process audio (implementation depends on model type)
            if model_info.model_type == "svoice":
                # SVoice-specific processing
                return self._process_with_svoice(model, mixture, **combined_kwargs)
            elif model_info.model_type == "demucs":
                # Demucs-specific processing
                return self._process_with_demucs(model, mixture, **combined_kwargs)
            else:
                # Generic processing (assumes model has a __call__ method)
                return model(mixture, **combined_kwargs)
        
        return process_audio
    
    def _process_with_svoice(self, model, mixture, **kwargs):
        """Process audio with SVoice model."""
        # This is a placeholder implementation
        # The actual implementation would depend on the SVoice API
        return model(mixture, **kwargs)
    
    def _process_with_demucs(self, model, mixture, **kwargs):
        """Process audio with Demucs model."""
        # This is a placeholder implementation
        # The actual implementation would depend on the Demucs API
        return model(mixture, **kwargs)
    
    def update_metrics_from_experiment(
        self,
        model_id: str,
        version_id: Optional[str],
        experiment_result: ExperimentResult
    ) -> bool:
        """
        Update model metrics from experiment results.
        
        Args:
            model_id: ID of the model
            version_id: ID of the version, or None for default version
            experiment_result: Results from an experiment
        
        Returns:
            True if successful, False otherwise
        """
        # Get model version
        version = self.get_model_version(model_id, version_id)
        if not version:
            self.logger.error(f"Model {model_id} version {version_id or 'default'} not found")
            return False
        
        # Extract metrics for this model
        model_results = {}
        for result_model_id, result in experiment_result.model_results.items():
            if result_model_id == model_id:
                model_results = result
                break
        
        if not model_results or "metrics" not in model_results:
            self.logger.error(f"No metrics found for model {model_id} in experiment results")
            return False
        
        # Update metrics
        metrics = model_results["metrics"]
        version.performance_metrics.update(metrics)
        
        # Add experiment metadata
        if "experiment_metadata" not in version.metadata:
            version.metadata["experiment_metadata"] = []
        
        version.metadata["experiment_metadata"].append({
            "experiment_id": experiment_result.experiment.id,
            "experiment_name": experiment_result.experiment.name,
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": metrics
        })
        
        # Save registry
        self._save_registry()
        
        self.logger.info(f"Updated metrics for model {model_id} version {version.version_id}")
        
        return True
    
    def export_model_catalog(self, output_path: str) -> str:
        """
        Export a catalog of all models in the registry.
        
        Args:
            output_path: Path to save the catalog
        
        Returns:
            Path to the saved catalog
        """
        catalog = {
            "models": [],
            "generated_at": datetime.datetime.now().isoformat(),
            "total_models": len(self.models),
            "total_versions": sum(len(m.versions) for m in self.models.values())
        }
        
        for model_id, model_info in self.models.items():
            model_entry = {
                "id": model_id,
                "name": model_info.name,
                "description": model_info.description,
                "type": model_info.model_type,
                "created_at": model_info.created_at.isoformat(),
                "versions": [],
                "default_version": model_info.default_version
            }
            
            # Add versions
            for version_id, version in model_info.versions.items():
                version_entry = {
                    "id": version_id,
                    "description": version.description,
                    "created_at": version.created_at.isoformat(),
                    "is_default": version.is_default,
                    "tags": version.tags
                }
                
                # Add performance metrics if available
                if version.performance_metrics:
                    version_entry["performance"] = {
                        "si_snri_mean": version.performance_metrics.get("si_snri_mean", "N/A"),
                        "sdri_mean": version.performance_metrics.get("sdri_mean", "N/A")
                    }
                
                model_entry["versions"].append(version_entry)
            
            catalog["models"].append(model_entry)
        
        # Save catalog
        with open(output_path, 'w') as f:
            json.dump(catalog, f, indent=2)
        
        return output_path