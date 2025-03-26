from cedtrainscheduler.scheduler.types.cluster import Cluster
from cedtrainscheduler.scheduler.types.scheduler_context import SchedulerContext
from cedtrainscheduler.scheduler.types.task import ScheduleInfo
from cedtrainscheduler.scheduler.types.task import TaskMeta
from cedtrainscheduler.simulator.executor import GPUExecutor
from cedtrainscheduler.simulator.fs import FileSystem
from cedtrainscheduler.simulator.fs import TaskDataInfo
from cedtrainscheduler.simulator.manager import ClusterManager
from cedtrainscheduler.simulator.record import Record


class ClusterPolicy:
    def __init__(self):
        self.cluster_manager: ClusterManager = None
        self.task_record: Record = None
        self.file_system: FileSystem = None
        self.task_queue: list[TaskMeta] = []

        self.task_data_info: dict[str, TaskDataInfo] = {}
        self.gpu_task_queue: dict[str, GPUExecutor] = {}
        self.clusters: dict[str, Cluster] = {}

    def set_scheduler_context(self, scheduler_context: SchedulerContext):
        self.cluster_manager = scheduler_context.cluster_manager
        self.file_system = scheduler_context.file_system
        self.task_record = scheduler_context.task_record
        self.task_queue = scheduler_context.task_queue

        self.task_data_info = self.file_system.task_data_info
        self.gpu_task_queue = self.cluster_manager.gpu_task_queue
        self.clusters = self.cluster_manager.clusters

    def schedule(self, scheduler_context: SchedulerContext, task: TaskMeta, cluster_id: str) -> ScheduleInfo:
        pass


class GreedyPolicy(ClusterPolicy):
    def schedule(self, scheduler_context: SchedulerContext, task: TaskMeta, cluster_id: str) -> ScheduleInfo:
        self.set_scheduler_context(scheduler_context)
        current_time = scheduler_context.current_time

        cluster_gpus = []
        nodes = self.clusters[cluster_id].nodes
        for node in nodes:
            for gpu in node.gpus:
                cluster_gpus.append(gpu.gpu_id)

        cluster_gpus.sort(key=lambda gpu_id: self.gpu_task_queue[gpu_id].queue_time(current_time, self.task_record))

        selected_gpus = cluster_gpus[: task.task_inst_num]

        schedule_infos = {}
        for inst_id, gpu_id in enumerate(selected_gpus):
            schedule_infos[inst_id] = ScheduleInfo(
                inst_id=inst_id,
                gpu_id=gpu_id,
            )
        return schedule_infos
