from cedtrainscheduler.scheduler.types.cluster import Cluster
from cedtrainscheduler.scheduler.types.task import TaskWrapRuntimeInfo
from cedtrainscheduler.simulator.executor import GPUExecutor


class SchedulerBase:
    def __init__(self, scheduler_name: str):
        self.scheduler_name = scheduler_name

    def schedule(
        self,
        clusters: dict[str, Cluster],
        gpu_task_queue: dict[str, GPUExecutor],
        task_record: dict[str, TaskWrapRuntimeInfo],
    ) -> list[TaskWrapRuntimeInfo]:
        pass
