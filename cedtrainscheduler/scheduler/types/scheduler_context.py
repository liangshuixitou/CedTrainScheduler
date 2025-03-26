from cedtrainscheduler.scheduler.types.task import TaskMeta
from cedtrainscheduler.scheduler.types.task import TaskWrapRuntimeInfo
from cedtrainscheduler.simulator.fs import FileSystem
from cedtrainscheduler.simulator.manager import ClusterManager


class SchedulerContext:
    def __init__(
        self,
        current_time: float,
        cluster_manager: ClusterManager,
        task_record: dict[str, TaskWrapRuntimeInfo],
        file_system: FileSystem,
        task_queue: list[TaskMeta],
    ):
        self.current_time = current_time
        self.cluster_manager = cluster_manager
        self.task_record = task_record
        self.file_system = file_system
        self.task_queue = task_queue
