import json
from dataclasses import dataclass
from statistics import mean


@dataclass
class TaskMetrics:
    total_tasks: int
    total_instances: int
    avg_instances_per_task: float
    task_status_counts: dict[str, int]
    completed_tasks_count: int
    completed_tasks_runtime: float
    completed_tasks_avg_completion_time: float
    completed_tasks_avg_waiting_time: float


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
    total_instances = 0
    task_status_counts = {}
    completed_tasks_count = 0
    completed_tasks_runtime = 0.0
    completed_tasks_completion_times = []
    completed_tasks_waiting_times = []
    earliest_start_time = float("inf")
    latest_end_time = float("-inf")

    # Calculate metrics for each task
    for _, task_info in task_records.items():
        # 统计任务状态
        task_status = task_info["task_meta"]["task_status"]
        task_status_counts[task_status] = task_status_counts.get(task_status, 0) + 1

        # Calculate task runtime
        task_start_time = task_info["task_start_time"]
        task_end_time = task_info["task_end_time"]
        task_submit_time = task_info["task_submit_time"]

        # Calculate completion time and waiting time
        task_completion_time = task_end_time - task_submit_time
        task_waiting_time = task_start_time - task_submit_time

        # Count instances
        total_instances += task_info["task_meta"]["task_inst_num"]

        # 统计已完成任务
        if task_status == "finished":
            completed_tasks_count += 1
            completed_tasks_completion_times.append(task_completion_time)
            completed_tasks_waiting_times.append(task_waiting_time)
            # Update earliest start and latest end times
            earliest_start_time = min(earliest_start_time, task_start_time)
            latest_end_time = max(latest_end_time, task_end_time)

    # Calculate completed tasks runtime as the time difference between earliest start and latest end
    if completed_tasks_count > 0:
        completed_tasks_runtime = latest_end_time - earliest_start_time

    # Calculate averages
    total_tasks = len(task_records)
    avg_instances_per_task = total_instances / total_tasks if total_tasks > 0 else 0

    # 计算已完成任务的平均值
    completed_tasks_avg_completion_time = (
        mean(completed_tasks_completion_times) if completed_tasks_completion_times else 0
    )
    completed_tasks_avg_waiting_time = mean(completed_tasks_waiting_times) if completed_tasks_waiting_times else 0

    return TaskMetrics(
        total_tasks=total_tasks,
        total_instances=total_instances,
        avg_instances_per_task=avg_instances_per_task,
        task_status_counts=task_status_counts,
        completed_tasks_count=completed_tasks_count,
        completed_tasks_runtime=completed_tasks_runtime,
        completed_tasks_avg_completion_time=completed_tasks_avg_completion_time,
        completed_tasks_avg_waiting_time=completed_tasks_avg_waiting_time,
    )


def print_task_metrics(metrics: dict):
    """
    Print the task metrics in a formatted way.

    Args:
        metrics: TaskMetrics object containing the calculated metrics
    """
    print("\n=== Task Execution Metrics ===")
    print(f"Total Tasks: {metrics['total_tasks']}")
    print(f"Total Instances: {metrics['total_instances']}")
    print(f"Average Instances per Task: {metrics['avg_instances_per_task']:.2f}")

    print("\n=== Task Status Distribution ===")
    for status, count in metrics["task_status_counts"].items():
        print(f"{status}: {count} tasks")

    print("\n=== Completed Tasks Statistics ===")
    print(f"Completed Tasks Count: {metrics['completed_tasks_count']}")
    print(f"Completed Tasks Total Runtime: {metrics['completed_tasks_runtime']:.2f} seconds")
    print(f"Completed Tasks Average Completion Time: {metrics['completed_tasks_avg_completion_time']:.2f} seconds")
    print(f"Completed Tasks Average Waiting Time: {metrics['completed_tasks_avg_waiting_time']:.2f} seconds")
    print("===========================\n")
