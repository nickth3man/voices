"""
Command-line interface for the processing queue.

This module provides a command-line interface for interacting with the processing queue,
allowing users to submit, monitor, and manage tasks from the command line.
"""

import os
import sys
import time
import json
import logging
import argparse
from typing import Dict, List, Any, Optional

from .queue_manager import QueueManager
from .task import Task, TaskStatus
from .priority import Priority
from .resource_manager import ResourceManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Processing Queue CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Start queue manager
    start_parser = subparsers.add_parser("start", help="Start the queue manager")
    start_parser.add_argument("--num-workers", "-n", type=int, default=0, help="Number of worker threads (0 for auto)")
    start_parser.add_argument("--config", "-c", help="Configuration file path")
    
    # Stop queue manager
    stop_parser = subparsers.add_parser("stop", help="Stop the queue manager")
    
    # Submit task
    submit_parser = subparsers.add_parser("submit", help="Submit a task to the queue")
    submit_parser.add_argument("--name", "-n", required=True, help="Task name")
    submit_parser.add_argument("--description", "-d", help="Task description")
    submit_parser.add_argument("--command", required=True, help="Command to execute")
    submit_parser.add_argument("--priority", "-p", choices=["low", "normal", "high", "critical"], default="normal", help="Task priority")
    submit_parser.add_argument("--dependencies", help="Comma-separated list of task IDs that must complete before this task")
    submit_parser.add_argument("--cpu-percent", type=float, default=10.0, help="CPU percentage required")
    submit_parser.add_argument("--memory-mb", type=float, default=100.0, help="Memory required in MB")
    submit_parser.add_argument("--gpu-percent", type=float, default=0.0, help="GPU percentage required")
    submit_parser.add_argument("--requires-gpu", action="store_true", help="Whether the task requires a GPU")
    submit_parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retry attempts")
    submit_parser.add_argument("--timeout", type=float, help="Timeout in seconds")
    submit_parser.add_argument("--tags", help="Comma-separated list of tags")
    
    # Cancel task
    cancel_parser = subparsers.add_parser("cancel", help="Cancel a task")
    cancel_parser.add_argument("task_id", help="ID of the task to cancel")
    
    # Pause task
    pause_parser = subparsers.add_parser("pause", help="Pause a task")
    pause_parser.add_argument("task_id", help="ID of the task to pause")
    
    # Resume task
    resume_parser = subparsers.add_parser("resume", help="Resume a paused task")
    resume_parser.add_argument("task_id", help="ID of the task to resume")
    
    # Get task status
    status_parser = subparsers.add_parser("status", help="Get task status")
    status_parser.add_argument("task_id", nargs="?", help="ID of the task (if omitted, shows queue status)")
    
    # List tasks
    list_parser = subparsers.add_parser("list", help="List tasks")
    list_parser.add_argument("--status", choices=["pending", "running", "completed", "failed", "cancelled", "paused", "all"], default="all", help="Filter by status")
    list_parser.add_argument("--limit", "-l", type=int, default=10, help="Maximum number of tasks to show")
    list_parser.add_argument("--format", "-f", choices=["table", "json"], default="table", help="Output format")
    
    # Clear completed tasks
    clear_parser = subparsers.add_parser("clear", help="Clear completed, failed, or cancelled tasks")
    clear_parser.add_argument("--status", choices=["completed", "failed", "cancelled", "all"], default="all", help="Status of tasks to clear")
    clear_parser.add_argument("--max-age", type=float, help="Maximum age in seconds (if omitted, clears all matching tasks)")
    
    # Show resource usage
    resources_parser = subparsers.add_parser("resources", help="Show resource usage")
    
    return parser.parse_args()


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    if not config_path:
        return {}
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return {}


def create_queue_manager(args) -> QueueManager:
    """
    Create a queue manager instance.
    
    Args:
        args: Command-line arguments
        
    Returns:
        QueueManager instance
    """
    # Load configuration
    config = load_config(args.config)
    
    # Create queue manager
    queue_manager = QueueManager(
        config=config,
        num_workers=args.num_workers
    )
    
    return queue_manager


def start_queue_manager(args):
    """Start the queue manager."""
    queue_manager = create_queue_manager(args)
    queue_manager.start()
    
    print(f"Queue manager started with {len(queue_manager.worker_pool.workers)} workers")
    
    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping queue manager...")
        queue_manager.stop()
        print("Queue manager stopped")


def stop_queue_manager(args):
    """Stop the queue manager."""
    # In a real implementation, we would need a way to communicate with a running queue manager
    # For now, we'll just print a message
    print("This command would stop a running queue manager")
    print("In a real implementation, this would communicate with the running process")


def submit_task(args):
    """Submit a task to the queue."""
    # Create queue manager
    queue_manager = create_queue_manager(args)
    queue_manager.start()
    
    try:
        # Parse dependencies
        dependencies = []
        if args.dependencies:
            dependencies = [dep.strip() for dep in args.dependencies.split(",")]
        
        # Parse tags
        tags = []
        if args.tags:
            tags = [tag.strip() for tag in args.tags.split(",")]
        
        # Create resource requirements
        resource_requirements = {
            "cpu_percent": args.cpu_percent,
            "memory_bytes": int(args.memory_mb * 1024 * 1024),  # Convert MB to bytes
            "gpu_percent": args.gpu_percent,
            "requires_gpu": args.requires_gpu
        }
        
        # Create a dummy function that executes the command
        def execute_command(*args, **kwargs):
            import subprocess
            progress_callback = kwargs.get("progress_callback")
            
            # Report starting
            if progress_callback:
                progress_callback(0, 100, "Starting command execution")
            
            # Execute command
            result = subprocess.run(
                args[0],
                shell=True,
                capture_output=True,
                text=True
            )
            
            # Report completion
            if progress_callback:
                progress_callback(100, 100, "Command execution completed")
            
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        
        # Create task
        task = Task(
            name=args.name,
            description=args.description or "",
            function=execute_command,
            args=[args.command],
            kwargs={},
            priority=Priority.from_string(args.priority),
            dependencies=dependencies,
            resource_requirements=resource_requirements,
            max_retries=args.max_retries,
            timeout=args.timeout,
            tags=tags
        )
        
        # Submit task
        task_id = queue_manager.submit_task(task)
        
        print(f"Task submitted with ID: {task_id}")
        
        # Wait for task to complete
        print("Waiting for task to complete...")
        
        while True:
            task = queue_manager.get_task(task_id)
            if not task:
                print("Task not found")
                break
            
            if task.status == TaskStatus.COMPLETED:
                print(f"Task completed with result: {task.result}")
                break
            elif task.status == TaskStatus.FAILED:
                print(f"Task failed with error: {task.error}")
                break
            elif task.status == TaskStatus.CANCELLED:
                print("Task was cancelled")
                break
            
            # Print progress
            if task.status == TaskStatus.RUNNING:
                print(f"Progress: {task.progress.percentage:.1f}% - {task.progress.message}")
            
            time.sleep(1)
    
    finally:
        # Stop queue manager
        queue_manager.stop()


def cancel_task(args):
    """Cancel a task."""
    # Create queue manager
    queue_manager = create_queue_manager(args)
    queue_manager.start()
    
    try:
        # Cancel task
        if queue_manager.cancel_task(args.task_id):
            print(f"Task {args.task_id} cancelled")
        else:
            print(f"Failed to cancel task {args.task_id}")
    
    finally:
        # Stop queue manager
        queue_manager.stop()


def pause_task(args):
    """Pause a task."""
    # Create queue manager
    queue_manager = create_queue_manager(args)
    queue_manager.start()
    
    try:
        # Pause task
        if queue_manager.pause_task(args.task_id):
            print(f"Task {args.task_id} paused")
        else:
            print(f"Failed to pause task {args.task_id}")
    
    finally:
        # Stop queue manager
        queue_manager.stop()


def resume_task(args):
    """Resume a paused task."""
    # Create queue manager
    queue_manager = create_queue_manager(args)
    queue_manager.start()
    
    try:
        # Resume task
        if queue_manager.resume_task(args.task_id):
            print(f"Task {args.task_id} resumed")
        else:
            print(f"Failed to resume task {args.task_id}")
    
    finally:
        # Stop queue manager
        queue_manager.stop()


def get_task_status(args):
    """Get task status."""
    # Create queue manager
    queue_manager = create_queue_manager(args)
    queue_manager.start()
    
    try:
        if args.task_id:
            # Get task status
            task = queue_manager.get_task(args.task_id)
            if task:
                print(f"Task ID: {task.task_id}")
                print(f"Name: {task.name}")
                print(f"Status: {task.status.name}")
                print(f"Priority: {task.priority.name}")
                print(f"Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(task.created_at))}")
                
                if task.started_at:
                    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(task.started_at))}")
                
                if task.completed_at:
                    print(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(task.completed_at))}")
                
                if task.status == TaskStatus.RUNNING:
                    print(f"Progress: {task.progress.percentage:.1f}% - {task.progress.message}")
                
                if task.status == TaskStatus.COMPLETED:
                    print(f"Result: {task.result}")
                
                if task.status == TaskStatus.FAILED:
                    print(f"Error: {task.error}")
                
                if task.dependencies:
                    print(f"Dependencies: {', '.join(task.dependencies)}")
                
                if task.tags:
                    print(f"Tags: {', '.join(task.tags)}")
            else:
                print(f"Task {args.task_id} not found")
        else:
            # Get queue status
            status = queue_manager.get_queue_status()
            
            print("Queue Status:")
            print(f"Pending tasks: {status['pending_tasks']}")
            print(f"Running tasks: {status['running_tasks']}")
            print(f"Completed tasks: {status['completed_tasks']}")
            print(f"Failed tasks: {status['failed_tasks']}")
            print(f"Cancelled tasks: {status['cancelled_tasks']}")
            print(f"Paused tasks: {status['paused_tasks']}")
            print(f"Total tasks: {status['total_tasks']}")
            
            print("\nWorker Pool:")
            worker_pool = status['worker_pool']
            print(f"Total workers: {worker_pool['total_workers']}")
            print(f"Idle workers: {worker_pool['idle_workers']}")
            
            print("\nResource Usage:")
            resources = status['resources']
            print(f"CPU: {resources['cpu']['percent']:.1f}% used, {resources['cpu']['allocated']:.1f}% allocated")
            print(f"Memory: {resources['memory']['percent']:.1f}% used, "
                 f"{resources['memory']['allocated_formatted']} allocated")
            
            if resources['gpu']['available']:
                print(f"GPU: {resources['gpu']['percent']:.1f}% used, {resources['gpu']['allocated']:.1f}% allocated")
            else:
                print("GPU: Not available")
    
    finally:
        # Stop queue manager
        queue_manager.stop()


def list_tasks(args):
    """List tasks."""
    # Create queue manager
    queue_manager = create_queue_manager(args)
    queue_manager.start()
    
    try:
        # Get tasks based on status
        tasks = []
        
        if args.status == "all" or args.status == "pending":
            for task_id in queue_manager._tasks:
                task = queue_manager.get_task(task_id)
                if task and task.status == TaskStatus.PENDING:
                    tasks.append(task)
        
        if args.status == "all" or args.status == "running":
            for task_id in queue_manager._running_tasks:
                task = queue_manager.get_task(task_id)
                if task:
                    tasks.append(task)
        
        if args.status == "all" or args.status == "completed":
            for task_id in queue_manager._completed_tasks:
                task = queue_manager.get_task(task_id)
                if task:
                    tasks.append(task)
        
        if args.status == "all" or args.status == "failed":
            for task_id in queue_manager._failed_tasks:
                task = queue_manager.get_task(task_id)
                if task:
                    tasks.append(task)
        
        if args.status == "all" or args.status == "cancelled":
            for task_id in queue_manager._cancelled_tasks:
                task = queue_manager.get_task(task_id)
                if task:
                    tasks.append(task)
        
        if args.status == "all" or args.status == "paused":
            for task_id in queue_manager._paused_tasks:
                task = queue_manager.get_task(task_id)
                if task:
                    tasks.append(task)
        
        # Sort tasks by creation time (newest first)
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        
        # Limit number of tasks
        if args.limit > 0:
            tasks = tasks[:args.limit]
        
        # Output tasks
        if args.format == "json":
            # JSON output
            output = [task.to_dict() for task in tasks]
            print(json.dumps(output, indent=2))
        else:
            # Table output
            if not tasks:
                print("No tasks found")
                return
            
            # Print header
            print(f"{'ID':<36} {'Name':<20} {'Status':<10} {'Priority':<10} {'Created':<20} {'Progress':<10}")
            print("-" * 110)
            
            # Print tasks
            for task in tasks:
                created = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(task.created_at))
                progress = f"{task.progress.percentage:.1f}%" if task.status == TaskStatus.RUNNING else ""
                
                print(f"{task.task_id:<36} {task.name[:20]:<20} {task.status.name:<10} "
                     f"{task.priority.name:<10} {created:<20} {progress:<10}")
    
    finally:
        # Stop queue manager
        queue_manager.stop()


def clear_tasks(args):
    """Clear completed, failed, or cancelled tasks."""
    # Create queue manager
    queue_manager = create_queue_manager(args)
    queue_manager.start()
    
    try:
        count = 0
        
        if args.status == "all" or args.status == "completed":
            completed_count = queue_manager.clear_completed_tasks(args.max_age)
            count += completed_count
            print(f"Cleared {completed_count} completed tasks")
        
        if args.status == "all" or args.status == "failed":
            failed_count = queue_manager.clear_failed_tasks(args.max_age)
            count += failed_count
            print(f"Cleared {failed_count} failed tasks")
        
        if args.status == "all" or args.status == "cancelled":
            cancelled_count = queue_manager.clear_cancelled_tasks(args.max_age)
            count += cancelled_count
            print(f"Cleared {cancelled_count} cancelled tasks")
        
        print(f"Total: Cleared {count} tasks")
    
    finally:
        # Stop queue manager
        queue_manager.stop()


def show_resources(args):
    """Show resource usage."""
    # Create queue manager
    queue_manager = create_queue_manager(args)
    queue_manager.start()
    
    try:
        # Get resource usage
        resources = queue_manager.resource_manager.get_resource_usage()
        
        print("Resource Usage:")
        
        # CPU
        print("\nCPU:")
        print(f"Total CPUs: {resources['cpu']['total']}")
        print(f"Usage: {resources['cpu']['percent']:.1f}%")
        print(f"Allocated: {resources['cpu']['allocated']:.1f}%")
        
        # Memory
        print("\nMemory:")
        print(f"Total: {resources['memory']['total_formatted']}")
        print(f"Used: {resources['memory']['used_formatted']} ({resources['memory']['percent']:.1f}%)")
        print(f"Available: {resources['memory']['available_formatted']}")
        print(f"Allocated: {resources['memory']['allocated_formatted']}")
        
        # GPU
        print("\nGPU:")
        if resources['gpu']['available']:
            print(f"GPU Info: {resources['gpu']['info'].get('name', 'Unknown')}")
            print(f"Usage: {resources['gpu']['percent']:.1f}%")
            print(f"Allocated: {resources['gpu']['allocated']:.1f}%")
        else:
            print("No GPU detected")
        
        # Tasks
        print("\nTasks:")
        print(f"Running: {resources['tasks']['count']}")
        print(f"Maximum: {resources['tasks']['max']}")
    
    finally:
        # Stop queue manager
        queue_manager.stop()


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    if args.command == "start":
        start_queue_manager(args)
    elif args.command == "stop":
        stop_queue_manager(args)
    elif args.command == "submit":
        submit_task(args)
    elif args.command == "cancel":
        cancel_task(args)
    elif args.command == "pause":
        pause_task(args)
    elif args.command == "resume":
        resume_task(args)
    elif args.command == "status":
        get_task_status(args)
    elif args.command == "list":
        list_tasks(args)
    elif args.command == "clear":
        clear_tasks(args)
    elif args.command == "resources":
        show_resources(args)
    else:
        print("No command specified")
        sys.exit(1)


if __name__ == "__main__":
    main()