import random
from collections import defaultdict

from cedtrainscheduler.scheduler.types.task import TaskMeta
from cedtrainscheduler.simulator.fs import FileSystem
from cedtrainscheduler.simulator.manager import ClusterManager
from cedtrainscheduler.simulator.record import Record


class CentralPolicy:
    def __init__(self, cluster_manager: ClusterManager, task_record: Record, file_system: FileSystem):
        self.cluster_manager = cluster_manager
        self.task_record = task_record
        self.file_system = file_system

        self.task_record = self.task_record.task_record
        self.task_data_info = self.file_system.task_data_info
        self.gpu_task_queue = self.cluster_manager.gpu_task_queue
        self.clusters = self.cluster_manager.clusters

    def schedule(self, current_time: float, task: TaskMeta) -> str:
        # list[cluster_id]
        pass

    def pick_task(self) -> TaskMeta:
        pass


class DataAffinityPolicy(CentralPolicy):
    def __init__(self, cluster_manager: ClusterManager, task_record: Record, file_system: FileSystem):
        super().__init__(cluster_manager, task_record, file_system)

    def schedule(self, current_time: float, task: TaskMeta) -> str:
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
    def __init__(self, cluster_manager: ClusterManager, task_record: Record, file_system: FileSystem):
        super().__init__(cluster_manager, task_record, file_system)

    def schedule(self, current_time: float, task: TaskMeta) -> str:
        cluster_queue_time: dict[str, tuple[int, float]] = defaultdict(lambda: (0, 0))
        for cluster_id in self.clusters.keys():
            for node in self.clusters[cluster_id].nodes:
                for gpu in node.gpus:
                    gpu_time = self.gpu_task_queue[gpu.gpu_id].queue_time(current_time, self.task_record)
                    cluster_queue_time[cluster_id] = (
                        cluster_queue_time[cluster_id][0] + 1,
                        cluster_queue_time[cluster_id][1] + gpu_time,
                    )
        # 选择gpu_time最少的前5名
        avg_queue_times: dict[str, float] = {}

        for cluster_id, (gpu_count, total_queue_time) in cluster_queue_time.items():
            avg_queue_times[cluster_id] = total_queue_time / gpu_count

        # 按平均队列等待时间排序
        sorted_clusters = sorted(avg_queue_times.items(), key=lambda x: x[1])

        # 选择平均队列等待时间最少的前1个集群
        selected_cluster = sorted_clusters[0][0]
        return selected_cluster
