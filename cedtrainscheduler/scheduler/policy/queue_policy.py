from cedtrainscheduler.scheduler.types.cluster import GPUType
from cedtrainscheduler.scheduler.types.task import TaskMeta
from cedtrainscheduler.simulator.fs import FileSystem
from cedtrainscheduler.simulator.manager import ClusterManager
from cedtrainscheduler.simulator.record import Record


class QueuePolicy:
    def __init__(
        self, cluster_manager: ClusterManager, task_record: Record, file_system: FileSystem, task_queue: list[TaskMeta]
    ):
        self.cluster_manager = cluster_manager
        self.task_record = task_record
        self.file_system = file_system
        self.task_queue = task_queue

        self.task_record = self.task_record.task_record
        self.task_data_info = self.file_system.task_data_info
        self.gpu_task_queue = self.cluster_manager.gpu_task_queue
        self.clusters = self.cluster_manager.clusters

    def add_task(self, task_list: list[TaskMeta]):
        self.task_queue.extend(task_list)

    def sort_task_queue(self):
        pass

    def pop_one_task(self, current_time: float) -> TaskMeta:
        pass


class FCFSQueuePolicy(QueuePolicy):
    def __init__(
        self, cluster_manager: ClusterManager, task_record: Record, file_system: FileSystem, task_queue: list[TaskMeta]
    ):
        super().__init__(cluster_manager, task_record, file_system, task_queue)
        self.is_sorted = False

    def add_task(self, task_list: list[TaskMeta]):
        super().add_task(task_list)
        self.is_sorted = False

    def sort_task_queue(self):
        self.task_queue.sort(key=lambda x: x.task_start_time)
        self.is_sorted = True

    def pop_one_task(self, current_time: float) -> TaskMeta:
        if not self.is_sorted:
            self.sort_task_queue()
        return self.task_queue.pop(0)


class SFJQueuePolicy(QueuePolicy):
    def __init__(
        self, cluster_manager: ClusterManager, task_record: Record, file_system: FileSystem, task_queue: list[TaskMeta]
    ):
        super().__init__(cluster_manager, task_record, file_system, task_queue)
        self.is_sorted = False

    def add_task(self, task_list: list[TaskMeta]):
        super().add_task(task_list)
        self.is_sorted = False

    def sort_task_queue(self):
        self.task_queue.sort(key=lambda x: x.task_runtime[GPUType.T4])
        self.is_sorted = True

    def pop_one_task(self, current_time: float) -> TaskMeta:
        if not self.is_sorted:
            self.sort_task_queue()
        return self.task_queue.pop(0)
