import json
from dataclasses import dataclass
from statistics import mean


@dataclass
class TaskMetrics:
    total_runtime: float
    avg_task_completion_time: float
    avg_task_waiting_time: float
    total_tasks: int
    total_instances: int
    avg_instances_per_task: float


def calculate_task_metrics(task_record_path: str) -> TaskMetrics:
    """
    Calculate various metrics from the task record JSON file.

    Args:
        task_record_path: Path to the task record JSON file

    Returns:
        TaskMetrics object containing various calculated metrics
    """
    with open(task_record_path) as f:
        task_records = json.load(f)

    # Initialize metrics
    total_runtime = 0.0
    task_completion_times = []
    task_waiting_times = []
    total_instances = 0

    # Calculate metrics for each task
    for _, task_info in task_records.items():
        # Calculate task runtime
        task_start_time = task_info["task_start_time"]
        task_end_time = task_info["task_end_time"]
        task_submit_time = task_info["task_submit_time"]

        # Add to total runtime
        task_runtime = task_end_time - task_start_time
        total_runtime += task_runtime

        # Calculate completion time and waiting time
        task_completion_time = task_end_time - task_start_time
        task_waiting_time = task_start_time - task_submit_time

        task_completion_times.append(task_completion_time)
        task_waiting_times.append(task_waiting_time)

        # Count instances
        total_instances += task_info["task_meta"]["task_inst_num"]

    # Calculate averages
    total_tasks = len(task_records)
    avg_task_completion_time = mean(task_completion_times) if task_completion_times else 0
    avg_task_waiting_time = mean(task_waiting_times) if task_waiting_times else 0
    avg_instances_per_task = total_instances / total_tasks if total_tasks > 0 else 0

    return TaskMetrics(
        total_runtime=total_runtime,
        avg_task_completion_time=avg_task_completion_time,
        avg_task_waiting_time=avg_task_waiting_time,
        total_tasks=total_tasks,
        total_instances=total_instances,
        avg_instances_per_task=avg_instances_per_task,
    )


def print_task_metrics(metrics: TaskMetrics):
    """
    Print the task metrics in a formatted way.

    Args:
        metrics: TaskMetrics object containing the calculated metrics
    """
    print("\n=== Task Execution Metrics ===")
    print(f"Total Runtime: {metrics.total_runtime:.2f} seconds")
    print(f"Total Tasks: {metrics.total_tasks}")
    print(f"Total Instances: {metrics.total_instances}")
    print(f"Average Instances per Task: {metrics.avg_instances_per_task:.2f}")
    print(f"Average Task Completion Time: {metrics.avg_task_completion_time:.2f} seconds")
    print(f"Average Task Waiting Time: {metrics.avg_task_waiting_time:.2f} seconds")
    print("===========================\n")
