#!/usr/bin/env python3
"""
Environment Setup Script for Voices Application

This script installs all required Python dependencies for the Voices application,
including those needed for integration tests.
"""

import subprocess
import sys
import os
import platform

def check_python_version():
    """Check if Python version is compatible."""
    required_version = (3, 7)
    current_version = sys.version_info
    
    if current_version < required_version:
        print(f"Error: Python {required_version[0]}.{required_version[1]} or higher is required.")
        print(f"Current version: {current_version[0]}.{current_version[1]}")
        sys.exit(1)
    
    print(f"Python version check passed: {current_version[0]}.{current_version[1]}.{current_version[2]}")

def install_dependencies():
    """Install dependencies from requirements.txt."""
    print("Installing dependencies from requirements.txt...")
    
    try:
        # First try to install everything except numba and llvmlite
        print("Installing main dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "numpy", "torch", "psutil", "pyyaml", "soundfile",
            "matplotlib", "pandas", "cffi", "pycparser"
        ])
        
        # Check if numba and llvmlite are already installed
        try:
            import numba
            import llvmlite
            print(f"Numba {numba.__version__} and llvmlite {llvmlite.__version__} are already installed.")
        except ImportError:
            print("Numba or llvmlite not found. Attempting to install compatible versions...")
            try:
                # Try to install specific versions known to work together
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "llvmlite==0.39.1", "numba==0.56.4"
                ])
                print("Successfully installed numba and llvmlite")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Could not install numba and llvmlite: {e}")
                print("This is not critical for basic functionality but may affect some audio processing features.")
        
        print("Successfully installed core dependencies")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

def check_gpu():
    """Check if GPU is available for PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU is available: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
        else:
            print("GPU is not available. PyTorch will use CPU.")
    except ImportError:
        print("PyTorch not installed yet. Skipping GPU check.")
    except Exception as e:
        print(f"Error checking GPU: {e}")

def create_virtual_env():
    """Create a virtual environment if not already in one."""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("Already running in a virtual environment.")
        return
    
    print("Creating a virtual environment...")
    venv_dir = "venv"
    
    try:
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
        
        # Provide activation instructions
        if platform.system() == "Windows":
            activate_script = os.path.join(venv_dir, "Scripts", "activate")
            print(f"\nVirtual environment created. To activate, run:")
            print(f"    {activate_script}")
        else:
            activate_script = os.path.join(venv_dir, "bin", "activate")
            print(f"\nVirtual environment created. To activate, run:")
            print(f"    source {activate_script}")
            
        print("\nAfter activation, run this script again to install dependencies.")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        sys.exit(1)

def main():
    """Main function."""
    print("Setting up environment for Voices Application...")
    
    # Check Python version
    check_python_version()
    
    # Create virtual environment if not already in one
    create_virtual_env()
    
    # Install dependencies
    install_dependencies()
    
    # Check GPU
    check_gpu()
    
    print("\nEnvironment setup complete!")
    print("\nKnown Issues:")
    print("- The llvmlite/numba installation may require Visual Studio Build Tools")
    print("  This is not critical for basic functionality but may affect some audio processing features")
    print("- Some integration tests may fail due to implementation issues that need to be addressed")
    
    print("\nYou can now run the integration tests with:")
    print("    cd src/backend/processing/registry && python run_integration_tests.py --verbose")
    print("\nNote: Even if some tests fail, this is expected as we're still working on fixing implementation issues.")

if __name__ == "__main__":
    main()