"""
Communication package for the Voices application.

This package provides the communication bridge between the Electron frontend
and the Python backend, implementing a JSON-based IPC protocol over stdin/stdout.
"""

from .server import (
    IPCServer,
    register_command,
    emit_event,
    server
)

__all__ = [
    'IPCServer',
    'register_command',
    'emit_event',
    'server'
]