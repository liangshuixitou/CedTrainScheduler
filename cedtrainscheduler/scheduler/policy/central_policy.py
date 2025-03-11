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

    def schedule(self, current_time: float, task: TaskMeta) -> list[str]:
        # list[cluster_id]
        pass

    def pick_task(self) -> TaskMeta:
        pass


class DataAffinityPolicy(CentralPolicy):
    def __init__(self, cluster_manager: ClusterManager, task_record: Record, file_system: FileSystem):
        super().__init__(cluster_manager, task_record, file_system)

    def schedule(self, current_time: float, task: TaskMeta) -> list[str]:
        # 查找所有的数据所在的节点的集群
        nodes = (
            self.task_data_info[task.task_name].dataset.storage_nodes
            + self.task_data_info[task.task_name].model.storage_nodes
        )
        cluster_ids: set[str] = set()
        for node in nodes:
            cluster_ids.add(self.cluster_manager.node_cluster_map[node].cluster_id)

        return list(cluster_ids)


class ResourceAffinityPolicy(CentralPolicy):
    def __init__(self, cluster_manager: ClusterManager, task_record: Record, file_system: FileSystem):
        super().__init__(cluster_manager, task_record, file_system)

    def schedule(self, current_time: float, task: TaskMeta) -> list[str]:
        cluster_ids = list(self.clusters.keys())

        return cluster_ids
