import random
from collections import defaultdict

from cedtrainscheduler.runtime.utils.logger import setup_logger
from cedtrainscheduler.scheduler.types.cluster import Cluster
from cedtrainscheduler.scheduler.types.scheduler_context import SchedulerContext
from cedtrainscheduler.scheduler.types.task import TaskMeta
from cedtrainscheduler.scheduler.types.task import TaskWrapRuntimeInfo
from cedtrainscheduler.simulator.executor import GPUExecutor
from cedtrainscheduler.simulator.fs import FileSystem
from cedtrainscheduler.simulator.fs import TaskDataInfo
from cedtrainscheduler.simulator.manager import ClusterManager


class CentralPolicy:
    def __init__(self):
        self.cluster_manager: ClusterManager = None
        self.task_record: dict[str, TaskWrapRuntimeInfo] = {}
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

    def schedule(self, scheduler_context: SchedulerContext, task: TaskMeta) -> str:
        # list[cluster_id]
        pass

    def pick_task(self) -> TaskMeta:
        pass


class DataAffinityPolicy(CentralPolicy):
    def schedule(self, scheduler_context: SchedulerContext, task: TaskMeta) -> str:
        self.set_scheduler_context(scheduler_context)

        # 查找所有的数据所在的节点的集群
        nodes = (
            self.task_data_info[task.task_name].dataset.storage_nodes
            + self.task_data_info[task.task_name].model.storage_nodes
        )
        cluster_ids: set[str] = set()
        for node in nodes:
            cluster_ids.add(self.cluster_manager.node_cluster_map[node].cluster_id)

        # 如果集群数量大于1，随机选择一个集群
        return random.choice(list(cluster_ids))


class ResourceAffinityPolicy(CentralPolicy):
    def __init__(self):
        super().__init__()
        self.logger = setup_logger(__name__)

    def schedule(self, scheduler_context: SchedulerContext, task: TaskMeta) -> str:
        self.set_scheduler_context(scheduler_context)

        current_time = scheduler_context.current_time

        cluster_queue_time: dict[str, tuple[int, float]] = defaultdict(lambda: (0, 0))
        for cluster_id in self.clusters.keys():
            for node in self.clusters[cluster_id].nodes:
                for gpu in node.gpus:
                    gpu_time = self.gpu_task_queue[gpu.gpu_id].queue_time(current_time, self.task_record)
                    cluster_queue_time[cluster_id] = (
                        cluster_queue_time[cluster_id][0] + 1,
                        cluster_queue_time[cluster_id][1] + gpu_time,
                    )
        avg_queue_times: dict[str, float] = {}

        for cluster_id, (gpu_count, total_queue_time) in cluster_queue_time.items():
            avg_queue_times[cluster_id] = total_queue_time / gpu_count

        # 按平均队列等待时间排序
        sorted_clusters = sorted(avg_queue_times.items(), key=lambda x: x[1])
        self.logger.info(f"sorted_clusters: {sorted_clusters}")
        # 选择平均队列等待时间最少的前1个集群
        selected_cluster = sorted_clusters[0][0]
        return selected_cluster
