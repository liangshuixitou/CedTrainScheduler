from dataclasses import dataclass


@dataclass
class Metrics:
    scheduler_name: str
    task_count: int
    total_runtime: float
    avg_queue_time: float
    avg_running_time: float
    avg_execution_time: float
    cloud_count: int
    edge_count: int
    terminal_count: int
    complete_count: int
    pending_count: int
