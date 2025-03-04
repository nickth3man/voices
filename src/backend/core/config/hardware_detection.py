"""
Hardware capability detection for dynamic configuration.

This module provides functionality to detect hardware capabilities such as
GPU availability, memory, CPU cores, etc., to enable dynamic configuration
based on the available hardware.
"""

import os
import platform
import logging
import subprocess
from typing import Dict, Any, Optional, List, Tuple
import psutil

logger = logging.getLogger(__name__)

# Try to import torch for GPU detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, GPU detection will be limited")


def detect_hardware_capabilities() -> Dict[str, Any]:
    """Detect hardware capabilities of the system.

    Returns:
        Dict[str, Any]: Dictionary containing hardware capabilities
    """
    capabilities = {
        "gpu_enabled": False,
        "gpu_devices": [],
        "gpu_memory": 0,
        "gpu_compute_capability": 0.0,
        "cpu_cores": os.cpu_count() or 1,
        "cpu_threads": psutil.cpu_count(logical=True) or 1,
        "available_memory": get_system_memory(),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "cuda_available": False,
        "mps_available": False,  # Apple Silicon GPU
        "recommended_precision": "float32",
        "recommended_batch_size": 1,
        "recommended_num_workers": 1
    }

    # Detect GPU capabilities
    if TORCH_AVAILABLE:
        capabilities.update(detect_torch_gpu_capabilities())
    else:
        capabilities.update(detect_gpu_capabilities_fallback())

    # Set recommended settings based on detected hardware
    capabilities.update(get_recommended_settings(capabilities))

    return capabilities


def detect_torch_gpu_capabilities() -> Dict[str, Any]:
    """Detect GPU capabilities using PyTorch.

    Returns:
        Dict[str, Any]: Dictionary containing GPU capabilities
    """
    gpu_info = {
        "gpu_enabled": False,
        "gpu_devices": [],
        "gpu_memory": 0,
        "gpu_compute_capability": 0.0,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch, 'mps') and torch.backends.mps.is_available()
    }

    # CUDA GPU detection
    if gpu_info["cuda_available"]:
        gpu_info["gpu_enabled"] = True
        gpu_count = torch.cuda.device_count()
        
        total_memory = 0
        max_compute_capability = 0.0
        
        for i in range(gpu_count):
            device_props = torch.cuda.get_device_properties(i)
            device_info = {
                "name": device_props.name,
                "memory": device_props.total_memory // (1024 * 1024),  # Convert to MB
                "compute_capability": float(f"{device_props.major}.{device_props.minor}"),
                "multi_processor_count": device_props.multi_processor_count
            }
            gpu_info["gpu_devices"].append(device_info)
            
            total_memory += device_info["memory"]
            max_compute_capability = max(max_compute_capability, device_info["compute_capability"])
        
        gpu_info["gpu_memory"] = total_memory
        gpu_info["gpu_compute_capability"] = max_compute_capability
    
    # Apple Silicon MPS detection
    elif gpu_info["mps_available"]:
        gpu_info["gpu_enabled"] = True
        # Apple Silicon doesn't expose memory info through PyTorch
        # Use a reasonable estimate based on system memory
        system_memory = get_system_memory()
        gpu_info["gpu_memory"] = system_memory // 4  # Estimate GPU memory as 1/4 of system memory
        gpu_info["gpu_compute_capability"] = 7.0  # Arbitrary value for Apple Silicon
        gpu_info["gpu_devices"].append({
            "name": "Apple Silicon GPU",
            "memory": gpu_info["gpu_memory"],
            "compute_capability": gpu_info["gpu_compute_capability"],
            "multi_processor_count": os.cpu_count() or 1
        })
    
    return gpu_info


def detect_gpu_capabilities_fallback() -> Dict[str, Any]:
    """Fallback method to detect GPU capabilities without PyTorch.

    Returns:
        Dict[str, Any]: Dictionary containing GPU capabilities
    """
    gpu_info = {
        "gpu_enabled": False,
        "gpu_devices": [],
        "gpu_memory": 0,
        "gpu_compute_capability": 0.0,
        "cuda_available": False,
        "mps_available": False
    }
    
    # Check for NVIDIA GPUs using nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_capability.major,compute_capability.minor", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.returncode == 0:
            gpu_info["gpu_enabled"] = True
            gpu_info["cuda_available"] = True
            
            lines = result.stdout.strip().split('\n')
            total_memory = 0
            max_compute_capability = 0.0
            
            for line in lines:
                if not line.strip():
                    continue
                    
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    name = parts[0]
                    memory = int(parts[1].split()[0])  # Extract number from "XXXX MiB"
                    cc_major = int(parts[2])
                    cc_minor = int(parts[3])
                    compute_capability = float(f"{cc_major}.{cc_minor}")
                    
                    device_info = {
                        "name": name,
                        "memory": memory,
                        "compute_capability": compute_capability,
                        "multi_processor_count": 0  # Not available without CUDA
                    }
                    gpu_info["gpu_devices"].append(device_info)
                    
                    total_memory += memory
                    max_compute_capability = max(max_compute_capability, compute_capability)
            
            gpu_info["gpu_memory"] = total_memory
            gpu_info["gpu_compute_capability"] = max_compute_capability
    except (subprocess.SubprocessError, FileNotFoundError):
        # nvidia-smi not available or failed
        pass
    
    # Check for Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        gpu_info["gpu_enabled"] = True
        gpu_info["mps_available"] = True
        
        # Estimate GPU memory as 1/4 of system memory
        system_memory = get_system_memory()
        gpu_info["gpu_memory"] = system_memory // 4
        gpu_info["gpu_compute_capability"] = 7.0  # Arbitrary value for Apple Silicon
        
        gpu_info["gpu_devices"].append({
            "name": "Apple Silicon GPU",
            "memory": gpu_info["gpu_memory"],
            "compute_capability": gpu_info["gpu_compute_capability"],
            "multi_processor_count": os.cpu_count() or 1
        })
    
    return gpu_info


def get_system_memory() -> int:
    """Get the total system memory in MB.

    Returns:
        int: Total system memory in MB
    """
    try:
        return psutil.virtual_memory().total // (1024 * 1024)  # Convert to MB
    except Exception as e:
        logger.warning(f"Failed to get system memory: {e}")
        return 4096  # Default to 4GB


def get_recommended_settings(capabilities: Dict[str, Any]) -> Dict[str, Any]:
    """Get recommended settings based on hardware capabilities.

    Args:
        capabilities: Dictionary containing hardware capabilities

    Returns:
        Dict[str, Any]: Dictionary containing recommended settings
    """
    recommended = {
        "use_gpu": False,
        "use_mps": False,
        "recommended_precision": "float32",
        "recommended_batch_size": 1,
        "recommended_num_workers": min(4, capabilities["cpu_threads"])
    }
    
    # Determine if GPU should be used
    if capabilities["cuda_available"]:
        recommended["use_gpu"] = True
        
        # Set precision based on compute capability
        cc = capabilities["gpu_compute_capability"]
        if cc >= 7.0:  # Volta and newer
            recommended["recommended_precision"] = "float16"
        
        # Set batch size based on available GPU memory
        memory_gb = capabilities["gpu_memory"] / 1024
        if memory_gb >= 16:
            recommended["recommended_batch_size"] = 8
        elif memory_gb >= 8:
            recommended["recommended_batch_size"] = 4
        elif memory_gb >= 4:
            recommended["recommended_batch_size"] = 2
    
    # For Apple Silicon
    elif capabilities["mps_available"]:
        recommended["use_mps"] = True
        recommended["recommended_precision"] = "float16"
        
        # Set batch size based on estimated GPU memory
        memory_gb = capabilities["gpu_memory"] / 1024
        if memory_gb >= 8:
            recommended["recommended_batch_size"] = 4
        elif memory_gb >= 4:
            recommended["recommended_batch_size"] = 2
    
    return recommended


def get_model_compatibility(
    model_requirements: Dict[str, Any],
    hardware_capabilities: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """Check if a model is compatible with the current hardware.

    Args:
        model_requirements: Dictionary containing model hardware requirements
        hardware_capabilities: Dictionary containing hardware capabilities

    Returns:
        Tuple[bool, List[str]]: (is_compatible, list of compatibility issues)
    """
    issues = []
    
    # Check GPU memory
    if model_requirements.get("min_gpu_memory", 0) > hardware_capabilities.get("gpu_memory", 0):
        issues.append(
            f"Insufficient GPU memory: {hardware_capabilities.get('gpu_memory')}MB available, "
            f"{model_requirements.get('min_gpu_memory')}MB required"
        )
    
    # Check compute capability
    if model_requirements.get("min_compute_capability", 0) > hardware_capabilities.get("gpu_compute_capability", 0):
        issues.append(
            f"Insufficient GPU compute capability: {hardware_capabilities.get('gpu_compute_capability')} available, "
            f"{model_requirements.get('min_compute_capability')} required"
        )
    
    # Check CPU cores
    if model_requirements.get("min_cpu_cores", 0) > hardware_capabilities.get("cpu_cores", 0):
        issues.append(
            f"Insufficient CPU cores: {hardware_capabilities.get('cpu_cores')} available, "
            f"{model_requirements.get('min_cpu_cores')} required"
        )
    
    # Check system memory
    if model_requirements.get("min_memory", 0) > hardware_capabilities.get("available_memory", 0):
        issues.append(
            f"Insufficient system memory: {hardware_capabilities.get('available_memory')}MB available, "
            f"{model_requirements.get('min_memory')}MB required"
        )
    
    return len(issues) == 0, issues


def get_optimal_model_settings(
    model_id: str,
    model_config: Dict[str, Any],
    hardware_capabilities: Dict[str, Any]
) -> Dict[str, Any]:
    """Get optimal model settings based on hardware capabilities.

    Args:
        model_id: Model identifier
        model_config: Model configuration
        hardware_capabilities: Hardware capabilities

    Returns:
        Dict[str, Any]: Optimized model settings
    """
    optimized_config = model_config.copy()
    
    # Set precision based on hardware
    if hardware_capabilities.get("use_gpu", False):
        optimized_config["precision"] = hardware_capabilities.get("recommended_precision", "float32")
    
    # Set device based on hardware
    if hardware_capabilities.get("use_gpu", False):
        optimized_config["device"] = "cuda"
    elif hardware_capabilities.get("use_mps", False):
        optimized_config["device"] = "mps"
    else:
        optimized_config["device"] = "cpu"
    
    # Model-specific optimizations
    if model_id == "svoice":
        # SVoice specific optimizations
        pass
    elif model_id == "demucs":
        # Demucs specific optimizations
        pass
    
    return optimized_config