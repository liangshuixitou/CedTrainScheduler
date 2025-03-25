import asyncio
import logging
from asyncio import Lock

from cedtrainscheduler.runtime.components import BaseComponent
from cedtrainscheduler.runtime.components import ComponentInfo
from cedtrainscheduler.runtime.manager.api_server import ManagerAPIServer
from cedtrainscheduler.runtime.master.api_client import ManagerMasterClient
from cedtrainscheduler.runtime.types.args import ManagerArgs
from cedtrainscheduler.runtime.types.cluster import Cluster
from cedtrainscheduler.runtime.types.task import TaskWrapRuntimeInfo
from cedtrainscheduler.scheduler.factory import SchedulerFactory


class Manager(BaseComponent):
    """
    Manager组件，负责调度和管理任务
    """

    def __init__(self, manager_args: ManagerArgs):
        super().__init__(manager_args.manager_info.component_id)
        self.ip = manager_args.manager_info.component_ip
        self.port = manager_args.manager_info.component_port

        self.cluster_manager = ClusterManager()
        self.task_manager = TaskManager()

        self.scheduler = SchedulerFactory.create_scheduler(
            manager_args.scheduler_name,
            ...,  # TODO: 需要修改为cluster_manager和task_manager
        )
        self.api_server = ManagerAPIServer(self)

        self.logger = logging.getLogger(__name__)
        self.scheduler_thread = None

    async def start(self):
        """启动Manager服务"""
        await self.api_server.start(host=self.ip, port=self.port)
        await self._start_scheduler_daemon()

    async def stop(self):
        """停止Manager服务"""
        await self.api_server.stop()

    async def _start_scheduler_daemon(self):
        async def scheduler_loop():
            pass

        self.scheduler_thread = asyncio.create_task(scheduler_loop())

    async def handle_task_submit(self, task_info: TaskWrapRuntimeInfo):
        self.task_manager.add_task_info(task_info)

    async def handle_master_register(
        self, cluster: Cluster, task_infos: dict[str, TaskWrapRuntimeInfo], master_info: ComponentInfo
    ):
        self.cluster_manager.register_master(master_info, cluster)
        for task_id, task_info in task_infos.items():
            self.cluster_manager.add_task_info(task_id, task_info)


class ClusterManager:
    def __init__(self):
        # dict[master_id, master_info]
        self.master_record: dict[str, ComponentInfo] = {}
        # dict[master_id, master_client]
        self.master_client_record: dict[str, ManagerMasterClient] = {}
        # dict[cluster_id, cluster]
        self.clusters: dict[str, Cluster] = {}
        self.cluster_lock = Lock()

    def register_master(self, master: ComponentInfo, cluster: Cluster):
        with self.cluster_lock:
            self.master_record[master.component_id] = master
            self.master_client_record[master.component_id] = ManagerMasterClient(
                master.component_ip, master.component_port
            )
            self.clusters[cluster.cluster_id] = cluster

    def remove_master(self, master_id: str):
        with self.cluster_lock:
            del self.master_record[master_id]
            del self.master_client_record[self.master_record[master_id].component_ip]
            del self.clusters[self.master_record[master_id].cluster_id]

    def get_master(self, master_id: str) -> ComponentInfo:
        with self.cluster_lock:
            return self.master_record[master_id]

    def get_master_client_by_cluster_id(self, cluster_id: str) -> ManagerMasterClient:
        with self.cluster_lock:
            return self.master_client_record[self.clusters[cluster_id].master_info.component_ip]

    def get_cluster(self, cluster_id: str) -> Cluster:
        with self.cluster_lock:
            return self.clusters[cluster_id]

    def get_cluster_by_gpu_id(self, gpu_id: str) -> Cluster:
        with self.cluster_lock:
            for cluster in self.clusters.values():
                for node in cluster.nodes.values():
                    for gpu in node.gpus.values():
                        if gpu.gpu_id == gpu_id:
                            return cluster
        return None


class TaskManager:
    def __init__(self, task_infos: dict[str, TaskWrapRuntimeInfo]):
        self.task_infos = task_infos

    def add_task_info(self, task_info: TaskWrapRuntimeInfo):
        self.task_infos[task_info.task_meta.task_id] = task_info
