"""
Register handlers with the IPC server.

This module registers all the command handlers with the IPC server.
"""

import logging
from typing import Dict, Any

from .server import register_command
from .metadata_handlers import METADATA_HANDLERS
from .model_comparison_handlers import MODEL_COMPARISON_HANDLERS

# Configure logging
logger = logging.getLogger(__name__)


def register_all_handlers() -> None:
    """Register all command handlers with the IPC server."""
    # Register metadata handlers
    register_metadata_handlers()
    
    # Register model comparison handlers
    register_model_comparison_handlers()
    
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


# Register handlers when this module is imported
register_all_handlers()