# Processing Queue

The processing queue module provides functionality for managing processing tasks, including parallel processing, progress tracking, priority handling, and intelligent resource allocation based on task type and system capabilities.

## Features

- **Parallel Processing**: Execute multiple tasks concurrently based on available system resources
- **Priority Handling**: Process tasks based on priority levels (LOW, NORMAL, HIGH, CRITICAL)
- **Progress Tracking**: Monitor task progress in real-time
- **Resource Allocation**: Intelligently allocate system resources based on task requirements
- **Dependency Management**: Define task dependencies to ensure proper execution order
- **Error Handling**: Automatic retry mechanism for failed tasks
- **Task Cancellation**: Cancel running or pending tasks
- **Task Pausing**: Pause and resume tasks as needed

## Components

- **QueueManager**: Main class for managing the processing queue
- **Task**: Represents a processing task with metadata and status
- **Priority**: Defines priority levels for tasks
- **ResourceManager**: Manages system resources and allocates them to tasks
- **Worker**: Executes tasks and reports progress
- **WorkerPool**: Manages a pool of workers for parallel processing

## Usage

### Basic Usage

```python
from src.backend.core.queue.queue_manager import QueueManager
from src.backend.core.queue.task import Task
from src.backend.core.queue.priority import Priority

# Create a queue manager
queue_manager = QueueManager()

# Start the queue manager
queue_manager.start()

# Create a task
task = Task(
    name="Process Audio",
    description="Process an audio file",
    function=process_audio_file,
    args=["input.wav", "output_dir"],
    priority=Priority.HIGH
)

# Submit the task
task_id = queue_manager.submit_task(task)

# Wait for the task to complete
while True:
    task = queue_manager.get_task(task_id)
    if task.status == TaskStatus.COMPLETED:
        print(f"Task completed with result: {task.result}")
        break
    elif task.status == TaskStatus.FAILED:
        print(f"Task failed with error: {task.error}")
        break
    time.sleep(1)

# Stop the queue manager
queue_manager.stop()
```

### Task Dependencies

```python
# Create tasks with dependencies
task1 = Task(name="Task 1", function=func1)
task2 = Task(name="Task 2", function=func2)
task3 = Task(name="Task 3", function=func3, dependencies=[task1.task_id, task2.task_id])

# Submit tasks
task1_id = queue_manager.submit_task(task1)
task2_id = queue_manager.submit_task(task2)
task3_id = queue_manager.submit_task(task3)  # Will only execute after task1 and task2 complete
```

### Resource Requirements

```python
# Define resource requirements
resource_requirements = {
    "cpu_percent": 50,
    "memory_bytes": 1024 * 1024 * 1024,  # 1 GB
    "gpu_percent": 30,
    "requires_gpu": True
}

# Create task with resource requirements
task = Task(
    name="GPU Processing",
    function=process_with_gpu,
    resource_requirements=resource_requirements
)
```

### Progress Tracking

```python
# Define a function that reports progress
def process_data(data, progress_callback=None):
    total_items = len(data)
    
    for i, item in enumerate(data):
        # Process item
        result = process_item(item)
        
        # Report progress
        if progress_callback:
            progress_callback(i + 1, total_items, f"Processed item {i + 1}/{total_items}")
    
    return results

# Create and submit task
task = Task(name="Process Data", function=process_data, args=[data])
task_id = queue_manager.submit_task(task)

# Register progress callback
def on_progress(task_id, current, total, message):
    print(f"Task {task_id} progress: {current}/{total} - {message}")

queue_manager.register_callback('task_progress', on_progress)
```

## Command-Line Interface

The queue package includes a command-line interface for managing tasks:

```bash
# Start the queue manager
python -m src.backend.core.queue.cli start --num-workers 4

# Submit a task
python -m src.backend.core.queue.cli submit --name "Process Audio" --command "python process_audio.py input.wav output/"

# Check task status
python -m src.backend.core.queue.cli status <task_id>

# List all tasks
python -m src.backend.core.queue.cli list

# Cancel a task
python -m src.backend.core.queue.cli cancel <task_id>
```

## Examples

See the `examples` directory for complete usage examples:

- `queue_example.py`: Demonstrates basic queue usage for processing audio files

## Integration with Pipeline

The processing queue can be integrated with the audio processing pipeline to enable parallel processing of multiple audio files:

```python
from src.backend.core.queue.queue_manager import QueueManager
from src.backend.core.queue.task import Task
from src.backend.processing.pipeline.utils import create_pipeline_from_config

# Create pipeline
pipeline = create_pipeline_from_config(config)

# Create queue manager
queue_manager = QueueManager()
queue_manager.start()

# Process multiple files in parallel
for file_path in audio_files:
    task = Task(
        name=f"Process {os.path.basename(file_path)}",
        function=pipeline.process_file,
        args=[file_path],
        kwargs={"output_dir": output_dir}
    )
    queue_manager.submit_task(task)