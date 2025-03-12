from cedtrainscheduler.scheduler.types.task import TaskMeta
from cedtrainscheduler.scheduler.types.task import TaskWrapRuntimeInfo
from cedtrainscheduler.simulator.fs import FileSystem
from cedtrainscheduler.simulator.manager import ClusterManager
from cedtrainscheduler.simulator.record import Record


class SchedulerBase:
    def __init__(
        self,
        scheduler_name: str,
        cluster_manager: ClusterManager,
        task_record: Record,
        file_system: FileSystem,
    ):
        self.max_task_num = 50
        self.scheduler_name = scheduler_name
        self.task_queue: list[TaskMeta] = []  # submitted task
        self.cluster_manager = cluster_manager
        self.task_record = task_record
        self.file_system = file_system

        self.task_record = self.task_record.task_record
        self.task_data_info = self.file_system.task_data_info
        self.gpu_task_queue = self.cluster_manager.gpu_task_queue
        self.clusters = self.cluster_manager.clusters

    def submit_task(self, task: TaskMeta):
        pass

    def schedule(self, current_time: float) -> tuple[TaskWrapRuntimeInfo, bool]:
        pass

    def is_queue_full(self) -> bool:
        return len(self.task_queue) >= self.max_task_num
