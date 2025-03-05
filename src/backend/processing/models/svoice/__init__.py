"""
SVoice Model Package.

This package provides the implementation of the SVoice model for voice separation,
including model architecture, loading, and inference.
"""

from .model import SVoiceModel
from .utils import (
    load_svoice_model,
    separate_sources,
    create_svoice_registry_loader,
    download_pretrained_model
)

__all__ = [
    'SVoiceModel',
    'load_svoice_model',
    'separate_sources',
    'create_svoice_registry_loader',
    'download_pretrained_model'
]