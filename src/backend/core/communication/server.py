#!/usr/bin/env python3
"""
Python server for the Voices application

This module implements the server-side of the IPC bridge between
the Electron frontend and the Python backend. It handles incoming
requests from the frontend, processes them, and sends back responses.

The communication protocol is based on JSON messages over stdin/stdout.
"""

import json
import sys
import threading
import uuid
import logging
import traceback
from typing import Dict, Any, Callable, Optional, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler('python_server.log')
    ]
)
logger = logging.getLogger('voices.server')

# Type definitions
CommandHandler = Callable[[Dict[str, Any]], Dict[str, Any]]
EventCallback = Callable[[str, Dict[str, Any]], None]


class IPCServer:
    """
    IPC Server for handling communication with the Electron frontend.
    
    This class implements a JSON-based protocol for bidirectional
    communication between the Electron frontend and Python backend.
    """
    
    def __init__(self):
        """Initialize the IPC server."""
        self.command_handlers: Dict[str, CommandHandler] = {}
        self.event_listeners: List[EventCallback] = []
        self.running = False
        self.input_thread: Optional[threading.Thread] = None
        
        # Register built-in commands
        self.register_command('ping', self._handle_ping)
        self.register_command('get_status', self._handle_get_status)
        
        logger.info("IPC Server initialized")
    
    def register_command(self, command: str, handler: CommandHandler) -> None:
        """
        Register a command handler.
        
        Args:
            command: The command name
            handler: The function to handle the command
        """
        self.command_handlers[command] = handler
        logger.debug(f"Registered command handler for '{command}'")
    
    def add_event_listener(self, callback: EventCallback) -> None:
        """
        Add an event listener.
        
        Args:
            callback: The callback function for events
        """
        self.event_listeners.append(callback)
    
    def remove_event_listener(self, callback: EventCallback) -> None:
        """
        Remove an event listener.
        
        Args:
            callback: The callback function to remove
        """
        if callback in self.event_listeners:
            self.event_listeners.remove(callback)
    
    def emit_event(self, event: str, data: Dict[str, Any] = None) -> None:
        """
        Emit an event to the frontend.
        
        Args:
            event: The event name
            data: The event data
        """
        if data is None:
            data = {}
        
        # Send the event to the frontend
        self._send_message({
            'type': 'event',
            'event': event,
            'data': data
        })
        
        # Notify event listeners
        for listener in self.event_listeners:
            try:
                listener(event, data)
            except Exception as e:
                logger.error(f"Error in event listener: {e}")
                logger.debug(traceback.format_exc())
    
    def start(self) -> None:
        """Start the IPC server."""
        if self.running:
            return
        
        self.running = True
        self.input_thread = threading.Thread(target=self._input_loop)
        self.input_thread.daemon = True
        self.input_thread.start()
        
        logger.info("IPC Server started")
    
    def stop(self) -> None:
        """Stop the IPC server."""
        self.running = False
        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=1.0)
        
        logger.info("IPC Server stopped")
    
    def _input_loop(self) -> None:
        """Main input loop for reading messages from stdin."""
        try:
            while self.running:
                # Read a line from stdin
                line = sys.stdin.readline()
                if not line:
                    logger.warning("Received EOF on stdin, stopping server")
                    self.running = False
                    break
                
                # Process the message
                try:
                    message = json.loads(line)
                    self._process_message(message)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON received: {line.strip()}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    logger.debug(traceback.format_exc())
        except Exception as e:
            logger.error(f"Error in input loop: {e}")
            logger.debug(traceback.format_exc())
            self.running = False
    
    def _process_message(self, message: Dict[str, Any]) -> None:
        """
        Process an incoming message.
        
        Args:
            message: The message to process
        """
        if not isinstance(message, dict):
            logger.error(f"Invalid message format: {message}")
            return
        
        # Check if this is a command
        if 'command' in message and 'id' in message:
            command = message['command']
            request_id = message['id']
            params = message.get('params', {})
            
            logger.debug(f"Received command '{command}' with ID {request_id}")
            
            # Find the handler for this command
            handler = self.command_handlers.get(command)
            if handler:
                try:
                    # Execute the handler
                    result = handler(params)
                    
                    # Send the response
                    self._send_response(request_id, result)
                except Exception as e:
                    logger.error(f"Error handling command '{command}': {e}")
                    logger.debug(traceback.format_exc())
                    
                    # Send error response
                    self._send_error(request_id, str(e))
            else:
                logger.warning(f"Unknown command: {command}")
                self._send_error(request_id, f"Unknown command: {command}")
        else:
            logger.warning(f"Invalid message format: {message}")
    
    def _send_response(self, request_id: str, data: Dict[str, Any]) -> None:
        """
        Send a response to a request.
        
        Args:
            request_id: The ID of the request
            data: The response data
        """
        self._send_message({
            'type': 'response',
            'id': request_id,
            'data': data
        })
    
    def _send_error(self, request_id: str, error: str) -> None:
        """
        Send an error response to a request.
        
        Args:
            request_id: The ID of the request
            error: The error message
        """
        self._send_message({
            'type': 'response',
            'id': request_id,
            'error': error
        })
    
    def _send_message(self, message: Dict[str, Any]) -> None:
        """
        Send a message to the frontend.
        
        Args:
            message: The message to send
        """
        try:
            # Convert the message to JSON and write to stdout
            json_message = json.dumps(message)
            print(json_message, flush=True)
            logger.debug(f"Sent message: {message['type']}")
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            logger.debug(traceback.format_exc())
    
    # Built-in command handlers
    
    def _handle_ping(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle the 'ping' command.
        
        Args:
            params: The command parameters
        
        Returns:
            The response data
        """
        return {'pong': True, 'timestamp': params.get('timestamp', 0)}
    
    def _handle_get_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle the 'get_status' command.
        
        Args:
            params: The command parameters
        
        Returns:
            The response data
        """
        return {
            'status': 'running',
            'commands': list(self.command_handlers.keys()),
            'uptime': 0  # TODO: Track uptime
        }


# Create a global server instance
server = IPCServer()


def register_command(command: str, handler: CommandHandler) -> None:
    """
    Register a command handler with the global server.
    
    Args:
        command: The command name
        handler: The function to handle the command
    """
    server.register_command(command, handler)


def emit_event(event: str, data: Dict[str, Any] = None) -> None:
    """
    Emit an event to the frontend.
    
    Args:
        event: The event name
        data: The event data
    """
    server.emit_event(event, data)


def main() -> None:
    """Main entry point for the server."""
    try:
        # Start the server
        server.start()
        
        # Send a ready event
        server.emit_event('ready', {'timestamp': 0})
        
        # Keep the main thread alive
        while server.running:
            try:
                # Sleep to avoid busy waiting
                threading.Event().wait(1.0)
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, stopping server")
                break
    except Exception as e:
        logger.error(f"Error in main: {e}")
        logger.debug(traceback.format_exc())
    finally:
        # Stop the server
        server.stop()


if __name__ == '__main__':
    main()