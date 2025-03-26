from cedtrainscheduler.scheduler.types.cluster import Cluster
from cedtrainscheduler.scheduler.types.cluster import GPUType
from cedtrainscheduler.scheduler.types.scheduler_context import SchedulerContext
from cedtrainscheduler.scheduler.types.task import TaskMeta
from cedtrainscheduler.scheduler.types.task import TaskWrapRuntimeInfo
from cedtrainscheduler.simulator.executor import GPUExecutor
from cedtrainscheduler.simulator.fs import FileSystem
from cedtrainscheduler.simulator.fs import TaskDataInfo
from cedtrainscheduler.simulator.manager import ClusterManager


class QueuePolicy:
    def __init__(self):
        self.current_time: int = 0

        self.cluster_manager: ClusterManager = None
        self.task_record: dict[str, TaskWrapRuntimeInfo] = {}
        self.file_system: FileSystem = None
        self.task_queue: list[TaskMeta] = []

        self.task_data_info: dict[str, TaskDataInfo] = {}
        self.gpu_task_queue: dict[str, GPUExecutor] = {}
        self.clusters: dict[str, Cluster] = {}

    def set_scheduler_context(self, scheduler_context: SchedulerContext):
        self.current_time = scheduler_context.current_time
        self.cluster_manager = scheduler_context.cluster_manager
        self.file_system = scheduler_context.file_system
        self.task_record = scheduler_context.task_record
        self.task_queue = scheduler_context.task_queue

        self.task_data_info = self.file_system.task_data_info
        self.gpu_task_queue = self.cluster_manager.gpu_task_queue
        self.clusters = self.cluster_manager.clusters

    def add_task(self, scheduler_context: SchedulerContext, task_list: list[TaskMeta]):
        self.set_scheduler_context(scheduler_context)
        self.task_queue.extend(task_list)

    def _sort_task_queue(self):
        pass

    def pop_one_task(self, scheduler_context: SchedulerContext) -> TaskMeta:
        pass


class FCFSQueuePolicy(QueuePolicy):
    def __init__(self):
        super().__init__()
        self.is_sorted = False

    def add_task(self, scheduler_context: SchedulerContext, task_list: list[TaskMeta]):
        super().add_task(scheduler_context, task_list)
        self.is_sorted = False

    def _sort_task_queue(self):
        self.task_queue.sort(key=lambda x: x.task_start_time)
        self.is_sorted = True

    def pop_one_task(self, scheduler_context: SchedulerContext) -> TaskMeta:
        self.set_scheduler_context(scheduler_context)
        if not self.is_sorted:
            self._sort_task_queue()
        return self.task_queue.pop(0)


class SFJQueuePolicy(QueuePolicy):
    def __init__(self):
        super().__init__()
        self.is_sorted = False

    def add_task(self, scheduler_context: SchedulerContext, task_list: list[TaskMeta]):
        super().add_task(scheduler_context, task_list)
        self.is_sorted = False

    def _sort_task_queue(self):
        self.task_queue.sort(key=lambda x: x.task_runtime[GPUType.T4])
        self.is_sorted = True

    def pop_one_task(self, scheduler_context: SchedulerContext) -> TaskMeta:
        self.set_scheduler_context(scheduler_context)
        if not self.is_sorted:
            self._sort_task_queue()
        return self.task_queue.pop(0)
