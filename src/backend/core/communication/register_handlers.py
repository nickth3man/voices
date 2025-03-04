"""
Register handlers with the IPC server.

This module registers all the command handlers with the IPC server.
"""

import logging
from typing import Dict, Any

from .server import register_command
from .metadata_handlers import METADATA_HANDLERS
from .model_comparison_handlers import MODEL_COMPARISON_HANDLERS
from .audio_visualization_handlers import AUDIO_VISUALIZATION_HANDLERS
from .processing_config_handlers import PROCESSING_CONFIG_HANDLERS
from .feedback_handlers import FEEDBACK_HANDLERS

# Configure logging
logger = logging.getLogger(__name__)
def register_all_handlers() -> None:
    """Register all command handlers with the IPC server."""
    # Register metadata handlers
    register_metadata_handlers()
    
    # Register model comparison handlers
    register_model_comparison_handlers()
    
    # Register audio visualization handlers
    register_audio_visualization_handlers()
    
    # Register processing configuration handlers
    register_processing_config_handlers()
    
    # Register feedback handlers
    register_feedback_handlers()
    
    logger.info("All handlers registered")
    logger.info("All handlers registered")


def register_metadata_handlers() -> None:
    """Register metadata-related command handlers."""
    for command, handler in METADATA_HANDLERS.items():
        register_command(command, handler)
    
    logger.info(f"Registered {len(METADATA_HANDLERS)} metadata handlers")


def register_model_comparison_handlers() -> None:
    """Register model comparison command handlers."""
    for command, handler in MODEL_COMPARISON_HANDLERS.items():
        register_command(command, handler)
    
    logger.info(f"Registered {len(MODEL_COMPARISON_HANDLERS)} model comparison handlers")


def register_audio_visualization_handlers() -> None:
    """Register audio visualization command handlers."""
    for command, handler in AUDIO_VISUALIZATION_HANDLERS.items():
        register_command(command, handler)
    
    logger.info(f"Registered {len(AUDIO_VISUALIZATION_HANDLERS)} audio visualization handlers")


def register_processing_config_handlers() -> None:
    """Register processing configuration command handlers."""
    for command, handler in PROCESSING_CONFIG_HANDLERS.items():
        register_command(command, handler)
    
    logger.info(f"Registered {len(PROCESSING_CONFIG_HANDLERS)} processing configuration handlers")


def register_feedback_handlers() -> None:
    """Register feedback command handlers."""
    for command, handler in FEEDBACK_HANDLERS.items():
        register_command(command, handler)
    
    logger.info(f"Registered {len(FEEDBACK_HANDLERS)} feedback handlers")


# Register handlers when this module is imported
register_all_handlers()