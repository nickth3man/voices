"""
Demucs Model Package.

This package provides the implementation of the Demucs model for voice separation,
including model architecture, loading, and inference.
"""

from .model import DemucsModel
from .utils import load_demucs_model, separate_sources, create_demucs_registry_loader