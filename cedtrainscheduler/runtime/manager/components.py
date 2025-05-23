import json
from asyncio import Lock

from cedtrainscheduler.runtime.components import ComponentInfo
from cedtrainscheduler.runtime.manager.constant import TASK_RECORD_SAVE_PATH
from cedtrainscheduler.runtime.master.api_client import ManagerMasterClient
from cedtrainscheduler.runtime.types.cluster import Cluster
from cedtrainscheduler.runtime.types.cluster import GPU
from cedtrainscheduler.runtime.types.model import TaskWrapRuntimeInfoModel
from cedtrainscheduler.runtime.types.task import TaskInst
from cedtrainscheduler.runtime.types.task import TaskWrapRuntimeInfo
from cedtrainscheduler.simulator.fs import FileSystem


class ClusterManager:
    def __init__(self):
        # dict[master_id, master_info]
        self.master_record: dict[str, ComponentInfo] = {}
        # dict[master_id, master_client]
        self.master_client_record: dict[str, ManagerMasterClient] = {}
        # dict[cluster_id, cluster]
        self.clusters: dict[str, Cluster] = {}
        self.cluster_lock = Lock()

    async def register_master(self, master: ComponentInfo, cluster: Cluster):
        async with self.cluster_lock:
            self.master_record[master.component_id] = master
            self.master_client_record[master.component_id] = ManagerMasterClient(
                master.component_ip, master.component_port
            )
            self.clusters[cluster.cluster_id] = cluster

    async def remove_master(self, master_id: str):
        async with self.cluster_lock:
            del self.master_record[master_id]
            del self.master_client_record[self.master_record[master_id].component_ip]
            del self.clusters[self.master_record[master_id].cluster_id]

    async def get_master(self, master_id: str) -> ComponentInfo:
        async with self.cluster_lock:
            return self.master_record[master_id]

    async def get_master_client_by_cluster_id(self, cluster_id: str) -> ManagerMasterClient:
        async with self.cluster_lock:
            return self.master_client_record[cluster_id]

    async def get_cluster(self, cluster_id: str) -> Cluster:
        async with self.cluster_lock:
            return self.clusters[cluster_id]

    async def get_gpu_by_gpu_id(self, gpu_id: str) -> GPU:
        async with self.cluster_lock:
            for cluster in self.clusters.values():
                for node in cluster.nodes.values():
                    for gpu in node.gpus.values():
                        if gpu.gpu_id == gpu_id:
                            return gpu

    async def get_cluster_by_gpu_id(self, gpu_id: str) -> Cluster:
        async with self.cluster_lock:
            for cluster in self.clusters.values():
                for node in cluster.nodes.values():
                    for gpu in node.gpus.values():
                        if gpu.gpu_id == gpu_id:
                            return cluster
        return None

    async def snapshot(self) -> dict[str, Cluster]:
        async with self.cluster_lock:
            return self.clusters


class TaskManager:
    def __init__(self):
        self.task_infos: dict[str, TaskWrapRuntimeInfo] = {}
        # dict[gpu_id, list[TaskInst]]
        self.task_queue_map: dict[str, list[TaskInst]] = {}
        self.task_lock = Lock()

    async def get_task_info(self, task_id: str) -> TaskWrapRuntimeInfo:
        async with self.task_lock:
            return self.task_infos[task_id]

    async def add_task_info(self, task_info: TaskWrapRuntimeInfo):
        async with self.task_lock:
            self.task_infos[task_info.task_meta.task_id] = task_info

    async def snapshot(self) -> dict[str, TaskWrapRuntimeInfo]:
        async with self.task_lock:
            return self.task_infos

    async def extend_task_queue_map(self, task_queue_map: dict[str, list[TaskInst]]):
        async with self.task_lock:
            self.task_queue_map.update(task_queue_map)

    async def get_task_queue_map(self) -> dict[str, list[TaskInst]]:
        async with self.task_lock:
            return self.task_queue_map

    async def save(self):
        async with self.task_lock:
            with open(TASK_RECORD_SAVE_PATH, "w") as f:
                task_record_dict = {}
                for task_id, task_info in self.task_infos.items():
                    task_record_dict[task_id] = TaskWrapRuntimeInfoModel.from_task_wrap_runtime_info(
                        task_info
                    ).model_dump()
                json.dump(task_record_dict, f, indent=2, ensure_ascii=False)


class FileSystemManager:
    def __init__(self):
        self.file_system: FileSystem = None

    def set_file_system(self, config_path: str, cluster_manager: ClusterManager):
        self.file_system = FileSystem(config_path, cluster_manager)

    def get_file_system(self) -> FileSystem:
        return self.file_system
