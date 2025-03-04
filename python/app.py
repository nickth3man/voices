#!/usr/bin/env python3
"""
Main application entry point for the Voices Python backend.
Handles communication with the Electron frontend and coordinates audio processing.
"""

import json
import logging
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "app.log")),
    ],
)
logger = logging.getLogger("voices")

# Global thread pool for processing tasks
executor = ThreadPoolExecutor(max_workers=4)
# Dictionary to track active processing tasks
active_tasks = {}


class CommunicationBridge:
    """Handles communication between Electron and Python processes."""

    def __init__(self):
        """Initialize the communication bridge."""
        self.handlers = {
            "ping": self.handle_ping,
            "process_file": self.handle_process_file,
            "get_task_status": self.handle_get_task_status,
            "cancel_task": self.handle_cancel_task,
            "get_system_info": self.handle_get_system_info,
        }

    def start(self):
        """Start the communication bridge, reading from stdin and writing to stdout."""
        logger.info("Starting communication bridge")
        try:
            for line in sys.stdin:
                self.process_message(line.strip())
        except KeyboardInterrupt:
            logger.info("Communication bridge stopped by keyboard interrupt")
        except Exception as e:
            logger.error(f"Error in communication bridge: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            logger.info("Communication bridge stopped")

    def process_message(self, message: str):
        """Process an incoming message from Electron."""
        try:
            request = json.loads(message)
            request_id = request.get("id")
            action = request.get("action")
            params = request.get("params", {})

            if not request_id or not action:
                self.send_error(None, "Invalid request format")
                return

            if action not in self.handlers:
                self.send_error(request_id, f"Unknown action: {action}")
                return

            # Call the appropriate handler
            handler = self.handlers[action]
            response = handler(request_id, params)
            self.send_response(request_id, response)

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON: {message}")
            self.send_error(None, "Invalid JSON")
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            logger.error(traceback.format_exc())
            self.send_error(None, f"Internal error: {str(e)}")

    def send_response(self, request_id: str, data: Dict[str, Any]):
        """Send a response back to Electron."""
        response = {
            "id": request_id,
            "success": True,
            "data": data,
        }
        print(json.dumps(response), flush=True)

    def send_error(self, request_id: Optional[str], error_message: str):
        """Send an error response back to Electron."""
        response = {
            "id": request_id,
            "success": False,
            "error": error_message,
        }
        print(json.dumps(response), flush=True)

    def send_progress(self, request_id: str, progress: float, status: str):
        """Send a progress update back to Electron."""
        response = {
            "id": request_id,
            "success": True,
            "progress": progress,
            "status": status,
        }
        print(json.dumps(response), flush=True)

    # Handler methods

    def handle_ping(self, request_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a ping request."""
        logger.info("Received ping request")
        return {"pong": True, "timestamp": params.get("timestamp", 0)}

    def handle_process_file(self, request_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a request to process an audio file."""
        file_path = params.get("file_path")
        if not file_path:
            return {"error": "No file path provided"}

        options = params.get("options", {})
        logger.info(f"Processing file: {file_path} with options: {options}")

        # Submit the task to the thread pool
        future = executor.submit(
            process_audio_file, request_id, file_path, options, 
            lambda progress, status: self.send_progress(request_id, progress, status)
        )
        active_tasks[request_id] = {
            "future": future,
            "file_path": file_path,
            "options": options,
            "progress": 0,
            "status": "started",
        }

        return {
            "task_id": request_id,
            "status": "started",
            "message": f"Started processing {os.path.basename(file_path)}",
        }

    def handle_get_task_status(self, request_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a request to get the status of a task."""
        task_id = params.get("task_id")
        if not task_id or task_id not in active_tasks:
            return {"error": "Invalid task ID"}

        task = active_tasks[task_id]
        return {
            "task_id": task_id,
            "status": task["status"],
            "progress": task["progress"],
            "file_path": task["file_path"],
        }

    def handle_cancel_task(self, request_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a request to cancel a task."""
        task_id = params.get("task_id")
        if not task_id or task_id not in active_tasks:
            return {"error": "Invalid task ID"}

        task = active_tasks[task_id]
        if not task["future"].done():
            task["future"].cancel()
            task["status"] = "cancelled"
            logger.info(f"Task {task_id} cancelled")

        return {
            "task_id": task_id,
            "status": "cancelled",
            "message": "Task cancelled",
        }

    def handle_get_system_info(self, request_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a request to get system information."""
        import platform
        import psutil

        try:
            import torch
            gpu_available = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count() if gpu_available else 0
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)] if gpu_available else []
        except ImportError:
            gpu_available = False
            gpu_count = 0
            gpu_names = []

        return {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(logical=False),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "gpu_available": gpu_available,
            "gpu_count": gpu_count,
            "gpu_names": gpu_names,
        }


def process_audio_file(
    task_id: str, 
    file_path: str, 
    options: Dict[str, Any], 
    progress_callback
) -> Dict[str, Any]:
    """
    Process an audio file to isolate voices.
    This is a placeholder implementation that will be expanded later.
    """
    try:
        # Update task status
        active_tasks[task_id]["status"] = "processing"
        
        # Simulate processing steps
        total_steps = 5
        
        # Step 1: Load file
        progress_callback(0.1, "Loading audio file")
        active_tasks[task_id]["progress"] = 0.1
        # TODO: Implement actual file loading with librosa
        
        # Step 2: Preprocess
        progress_callback(0.3, "Preprocessing audio")
        active_tasks[task_id]["progress"] = 0.3
        # TODO: Implement preprocessing (normalization, etc.)
        
        # Step 3: Run Demucs
        progress_callback(0.5, "Separating audio sources")
        active_tasks[task_id]["progress"] = 0.5
        # TODO: Implement Demucs processing
        
        # Step 4: Post-process
        progress_callback(0.8, "Post-processing audio")
        active_tasks[task_id]["progress"] = 0.8
        # TODO: Implement post-processing
        
        # Step 5: Save results
        progress_callback(0.9, "Saving results")
        active_tasks[task_id]["progress"] = 0.9
        # TODO: Implement file saving
        
        # Complete
        progress_callback(1.0, "Processing complete")
        active_tasks[task_id]["progress"] = 1.0
        active_tasks[task_id]["status"] = "completed"
        
        # Return results
        return {
            "success": True,
            "message": f"Processed {os.path.basename(file_path)}",
            "output_files": [
                # These will be actual output files in the real implementation
                f"{file_path}_vocals.wav",
                f"{file_path}_accompaniment.wav",
            ],
        }
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        logger.error(traceback.format_exc())
        active_tasks[task_id]["status"] = "error"
        return {
            "success": False,
            "error": str(e),
        }


if __name__ == "__main__":
    logger.info("Starting Voices Python backend")
    bridge = CommunicationBridge()
    bridge.start()