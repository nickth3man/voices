"""
Processing Configuration Handlers

This module provides handlers for processing configuration operations, including
model selection, processing parameters, output format options, and batch processing settings.
"""

import json
import os
import platform
import psutil
import torch
from ..config.config_manager import ConfigManager
from ...processing.models.abstraction import ModelType
from ...processing.registry.model_registry import ModelRegistry

# Initialize dependencies
config_manager = ConfigManager()
model_registry = ModelRegistry()

# Dictionary of handlers for processing configuration operations
PROCESSING_CONFIG_HANDLERS = {
    "get_available_models": handle_get_available_models,
    "get_hardware_info": handle_get_hardware_info,
    "get_optimal_settings": handle_get_optimal_settings,
    "save_configuration": handle_save_configuration,
    "load_configuration": handle_load_configuration
}

def handle_get_available_models(data):
    """
    Get a list of available voice separation models with their details.
    
    Returns:
        dict: Response with available models and their details
    """
    try:
        # Get all registered models
        models = model_registry.list_models()
        
        # Format model details for frontend
        model_details = []
        for model in models:
            # Get model metrics if available
            metrics = model_registry.get_model_metrics(model.id) or {}
            
            model_details.append({
                "id": model.id,
                "name": model.name,
                "type": model.type.value,  # Convert enum to string
                "description": model.description,
                "version": model.version,
                "metrics": {
                    "si_snri": metrics.get("si_snri", 0),
                    "sdri": metrics.get("sdri", 0),
                    "processing_speed": metrics.get("processing_speed", 0),
                    "memory_usage": metrics.get("memory_usage", 0)
                },
                "hardware_requirements": {
                    "min_cpu_cores": model.hardware_requirements.get("min_cpu_cores", 2),
                    "min_ram_gb": model.hardware_requirements.get("min_ram_gb", 4),
                    "gpu_required": model.hardware_requirements.get("gpu_required", False),
                    "min_vram_gb": model.hardware_requirements.get("min_vram_gb", 0)
                },
                "optimal_for": model.optimal_for or []
            })
        
        return {
            "success": True,
            "models": model_details
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def handle_get_hardware_info(data):
    """
    Get information about the system's hardware capabilities.
    
    Returns:
        dict: Response with hardware information
    """
    try:
        # Get CPU information
        cpu_count = psutil.cpu_count(logical=False)
        cpu_logical_count = psutil.cpu_count(logical=True)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Get memory information
        memory = psutil.virtual_memory()
        memory_total_gb = round(memory.total / (1024 ** 3), 2)
        memory_available_gb = round(memory.available / (1024 ** 3), 2)
        
        # Get GPU information if available
        gpu_info = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu = {
                    "name": torch.cuda.get_device_name(i),
                    "index": i,
                    "memory_total_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024 ** 3), 2),
                    "memory_allocated_gb": round(torch.cuda.memory_allocated(i) / (1024 ** 3), 2),
                    "memory_reserved_gb": round(torch.cuda.memory_reserved(i) / (1024 ** 3), 2)
                }
                gpu_info.append(gpu)
        
        # Get disk information
        disk = psutil.disk_usage('/')
        disk_total_gb = round(disk.total / (1024 ** 3), 2)
        disk_free_gb = round(disk.free / (1024 ** 3), 2)
        
        # Get system information
        system_info = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "platform_release": platform.release(),
            "architecture": platform.machine()
        }
        
        return {
            "success": True,
            "hardware": {
                "cpu": {
                    "physical_cores": cpu_count,
                    "logical_cores": cpu_logical_count,
                    "usage_percent": cpu_percent
                },
                "memory": {
                    "total_gb": memory_total_gb,
                    "available_gb": memory_available_gb,
                    "percent_used": memory.percent
                },
                "gpu": gpu_info,
                "disk": {
                    "total_gb": disk_total_gb,
                    "free_gb": disk_free_gb,
                    "percent_used": disk.percent
                },
                "system": system_info
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def handle_get_optimal_settings(data):
    """
    Get optimal processing settings based on hardware capabilities.
    
    Args:
        data (dict): Request data with model_id
        
    Returns:
        dict: Response with optimal settings
    """
    try:
        model_id = data.get("model_id")
        if not model_id:
            return {
                "success": False,
                "error": "Model ID is required"
            }
        
        # Get hardware information
        hardware_info = handle_get_hardware_info({})
        if not hardware_info.get("success"):
            return hardware_info
        
        hardware = hardware_info.get("hardware")
        
        # Get model details
        model = model_registry.get_model(model_id)
        if not model:
            return {
                "success": False,
                "error": f"Model with ID {model_id} not found"
            }
        
        # Determine if hardware meets minimum requirements
        meets_requirements = True
        requirements_issues = []
        
        # Check CPU cores
        min_cpu_cores = model.hardware_requirements.get("min_cpu_cores", 2)
        if hardware["cpu"]["physical_cores"] < min_cpu_cores:
            meets_requirements = False
            requirements_issues.append(f"CPU cores: {hardware['cpu']['physical_cores']} (minimum: {min_cpu_cores})")
        
        # Check RAM
        min_ram_gb = model.hardware_requirements.get("min_ram_gb", 4)
        if hardware["memory"]["total_gb"] < min_ram_gb:
            meets_requirements = False
            requirements_issues.append(f"RAM: {hardware['memory']['total_gb']} GB (minimum: {min_ram_gb} GB)")
        
        # Check GPU if required
        gpu_required = model.hardware_requirements.get("gpu_required", False)
        if gpu_required and not hardware["gpu"]:
            meets_requirements = False
            requirements_issues.append("GPU: Not available (required)")
        
        # Check VRAM if GPU is required
        if gpu_required and hardware["gpu"]:
            min_vram_gb = model.hardware_requirements.get("min_vram_gb", 0)
            if hardware["gpu"][0]["memory_total_gb"] < min_vram_gb:
                meets_requirements = False
                requirements_issues.append(f"VRAM: {hardware['gpu'][0]['memory_total_gb']} GB (minimum: {min_vram_gb} GB)")
        
        # Determine optimal settings based on hardware
        optimal_settings = {}
        
        # Determine batch size based on available memory
        if hardware["gpu"] and model.type != ModelType.CPU_ONLY:
            # GPU-based batch size calculation
            available_vram = hardware["gpu"][0]["memory_total_gb"] - hardware["gpu"][0]["memory_allocated_gb"]
            optimal_settings["batch_size"] = max(1, min(16, int(available_vram * 2)))
        else:
            # CPU-based batch size calculation
            available_ram = hardware["memory"]["available_gb"]
            optimal_settings["batch_size"] = max(1, min(8, int(available_ram / 2)))
        
        # Determine number of workers based on CPU cores
        optimal_settings["num_workers"] = max(1, min(4, hardware["cpu"]["physical_cores"] - 1))
        
        # Determine chunk size based on available memory
        if hardware["gpu"] and model.type != ModelType.CPU_ONLY:
            # GPU-based chunk size calculation
            optimal_settings["chunk_size"] = 32000  # Default for GPU
        else:
            # CPU-based chunk size calculation
            optimal_settings["chunk_size"] = 16000  # Default for CPU
        
        # Determine precision based on GPU capabilities
        if hardware["gpu"] and model.type != ModelType.CPU_ONLY:
            # Check if GPU supports mixed precision
            if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7:
                optimal_settings["precision"] = "mixed"  # Use mixed precision for Volta+ GPUs
            else:
                optimal_settings["precision"] = "single"  # Use single precision for older GPUs
        else:
            optimal_settings["precision"] = "single"  # Use single precision for CPU
        
        return {
            "success": True,
            "meets_requirements": meets_requirements,
            "requirements_issues": requirements_issues,
            "optimal_settings": optimal_settings
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def handle_save_configuration(data):
    """
    Save processing configuration to file or database.
    
    Args:
        data (dict): Configuration data to save
        
    Returns:
        dict: Response indicating success or failure
    """
    try:
        config_name = data.get("name", "default")
        config_data = data.get("config", {})
        
        if not config_data:
            return {
                "success": False,
                "error": "No configuration data provided"
            }
        
        # Create configs directory if it doesn't exist
        configs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "configs")
        os.makedirs(configs_dir, exist_ok=True)
        
        # Save configuration to file
        config_path = os.path.join(configs_dir, f"{config_name}.json")
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        return {
            "success": True,
            "message": f"Configuration saved as {config_name}",
            "path": config_path
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def handle_load_configuration(data):
    """
    Load processing configuration from file or database.
    
    Args:
        data (dict): Request data with configuration name
        
    Returns:
        dict: Response with loaded configuration
    """
    try:
        config_name = data.get("name", "default")
        
        # Get configuration file path
        configs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "configs")
        config_path = os.path.join(configs_dir, f"{config_name}.json")
        
        # Check if configuration file exists
        if not os.path.exists(config_path):
            return {
                "success": False,
                "error": f"Configuration {config_name} not found"
            }
        
        # Load configuration from file
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        return {
            "success": True,
            "config": config_data
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }