"""
Example usage of the processing queue.

This example demonstrates how to use the processing queue for parallel processing
of audio files with different priorities and resource requirements.
"""

import os
import sys
import time
import logging
import argparse
from typing import Dict, List, Any, Optional

# Add parent directory to path to allow importing the queue package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from src.backend.core.queue.queue_manager import QueueManager
from src.backend.core.queue.task import Task, TaskStatus
from src.backend.core.queue.priority import Priority
from src.backend.core.queue.resource_manager import ResourceManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_audio_file(file_path: str, output_dir: str, progress_callback=None) -> Dict[str, Any]:
    """
    Simulate processing an audio file.
    
    Args:
        file_path: Path to the audio file
        output_dir: Directory to save the processed file
        progress_callback: Callback for progress updates
        
    Returns:
        Dictionary with processing results
    """
    logger.info(f"Processing audio file: {file_path}")
    
    # Simulate file loading
    if progress_callback:
        progress_callback(10, 100, "Loading audio file")
    time.sleep(1)
    
    # Simulate preprocessing
    if progress_callback:
        progress_callback(30, 100, "Preprocessing audio")
    time.sleep(2)
    
    # Simulate voice separation
    if progress_callback:
        progress_callback(50, 100, "Separating voices")
    time.sleep(3)
    
    # Simulate postprocessing
    if progress_callback:
        progress_callback(80, 100, "Postprocessing audio")
    time.sleep(1)
    
    # Simulate saving output
    if progress_callback:
        progress_callback(90, 100, "Saving output files")
    time.sleep(1)
    
    # Simulate completion
    if progress_callback:
        progress_callback(100, 100, "Processing completed")
    
    # Return simulated results
    return {
        "input_file": file_path,
        "output_dir": output_dir,
        "output_files": [
            os.path.join(output_dir, f"speaker1_{os.path.basename(file_path)}"),
            os.path.join(output_dir, f"speaker2_{os.path.basename(file_path)}")
        ],
        "processing_time": 8.0  # seconds
    }


def on_task_completed(task_id: str, result: Any) -> None:
    """Callback for when a task is completed."""
    logger.info(f"Task {task_id} completed")
    logger.info(f"Result: {result}")


def on_task_failed(task_id: str, error: Exception) -> None:
    """Callback for when a task fails."""
    logger.error(f"Task {task_id} failed: {str(error)}")


def on_task_progress(task_id: str, current: int, total: int, message: str) -> None:
    """Callback for task progress updates."""
    logger.info(f"Task {task_id} progress: {current}/{total} - {message}")


def main():
    """Main entry point for the example."""
    parser = argparse.ArgumentParser(description="Processing Queue Example")
    parser.add_argument("--input-dir", "-i", required=True, help="Input directory containing audio files")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory for processed files")
    parser.add_argument("--num-workers", "-n", type=int, default=0, help="Number of worker threads (0 for auto)")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create queue manager
    queue_manager = QueueManager(num_workers=args.num_workers)
    
    # Register callbacks
    queue_manager.register_callback('task_completed', on_task_completed)
    queue_manager.register_callback('task_failed', on_task_failed)
    queue_manager.register_callback('task_progress', on_task_progress)
    
    # Start queue manager
    queue_manager.start()
    
    try:
        # Find audio files in the input directory
        audio_files = []
        for root, _, files in os.walk(args.input_dir):
            for file in files:
                if file.endswith(('.wav', '.mp3', '.flac')):
                    audio_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        # Create tasks for each audio file
        task_ids = []
        for i, file_path in enumerate(audio_files):
            # Determine priority based on file size
            file_size = os.path.getsize(file_path)
            if file_size > 10 * 1024 * 1024:  # > 10 MB
                priority = Priority.LOW
            elif file_size > 5 * 1024 * 1024:  # > 5 MB
                priority = Priority.NORMAL
            else:
                priority = Priority.HIGH
            
            # Estimate resource requirements
            resource_requirements = queue_manager.resource_manager.estimate_resource_requirements(
                'audio_processing',
                file_size
            )
            
            # Create task
            task = Task(
                name=f"Process {os.path.basename(file_path)}",
                description=f"Process audio file: {file_path}",
                function=process_audio_file,
                args=[file_path, args.output_dir],
                kwargs={},
                priority=priority,
                resource_requirements=resource_requirements
            )
            
            # Submit task
            task_id = queue_manager.submit_task(task)
            task_ids.append(task_id)
            
            logger.info(f"Submitted task {task_id} for file {file_path} with priority {priority.name}")
        
        # Wait for all tasks to complete
        logger.info("Waiting for all tasks to complete...")
        
        while True:
            # Check if all tasks are completed
            all_completed = True
            for task_id in task_ids:
                task = queue_manager.get_task(task_id)
                if task and task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                    all_completed = False
                    break
            
            if all_completed:
                break
            
            # Print queue status
            status = queue_manager.get_queue_status()
            logger.info(f"Queue status: {status['pending_tasks']} pending, {status['running_tasks']} running, "
                       f"{status['completed_tasks']} completed, {status['failed_tasks']} failed")
            
            time.sleep(1)
        
        # Print final results
        logger.info("All tasks completed")
        
        completed_count = 0
        failed_count = 0
        
        for task_id in task_ids:
            task = queue_manager.get_task(task_id)
            if task:
                if task.status == TaskStatus.COMPLETED:
                    completed_count += 1
                elif task.status == TaskStatus.FAILED:
                    failed_count += 1
        
        logger.info(f"Results: {completed_count} completed, {failed_count} failed")
    
    finally:
        # Stop queue manager
        queue_manager.stop()


if __name__ == "__main__":
    main()