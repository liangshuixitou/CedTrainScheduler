import asyncio
import time
from asyncio import Lock

from cedtrainscheduler.runtime.components import BaseServer
from cedtrainscheduler.runtime.components import ComponentInfo
from cedtrainscheduler.runtime.components import ComponentType
from cedtrainscheduler.runtime.manager.api_server import ManagerAPIServer
from cedtrainscheduler.runtime.manager.constant import FS_CONFIG_PATH
from cedtrainscheduler.runtime.manager.service import ManagerService
from cedtrainscheduler.runtime.manager.utils import TypeConverter
from cedtrainscheduler.runtime.master.api_client import ManagerMasterClient
from cedtrainscheduler.runtime.types.args import ManagerArgs
from cedtrainscheduler.runtime.types.cluster import Cluster
from cedtrainscheduler.runtime.types.task import TaskMeta
from cedtrainscheduler.runtime.types.task import TaskWrapRuntimeInfo as RuntimeTaskWrapRuntimeInfo
from cedtrainscheduler.scheduler.factory import SchedulerFactory
from cedtrainscheduler.scheduler.types.scheduler_context import SchedulerContext
from cedtrainscheduler.scheduler.types.task import TaskWrapRuntimeInfo as SchedulerTaskWrapRuntimeInfo
from cedtrainscheduler.simulator.fs import FileSystem
from cedtrainscheduler.simulator.manager import ClusterManager as SchedulerClusterManager
from cedtrainscheduler.runtime.utils.logger import setup_logger

MANAGER_SCHEDULER_INTERVAL = 1


class Manager(BaseServer, ManagerService):
    """
    Manager组件，负责调度和管理任务
    """

    def __init__(self, manager_args: ManagerArgs):
        # 初始化Server基类
        super().__init__(manager_args.manager_info)

        self.cluster_manager = ClusterManager()
        self.task_manager = TaskManager()
        self.file_system_manager = FileSystemManager()

        self.scheduler = SchedulerFactory.create_scheduler(manager_args.scheduler_name)
        self.api_server = ManagerAPIServer(self)
        self.logger = setup_logger(__name__)

    async def _start(self):
        """实现启动所有服务"""
        # 启动API服务器并追踪任务
        api_server_task = await self.api_server.start(
            host=self.component_info.component_ip, port=self.component_info.component_port
        )
        self._tasks.append(api_server_task)

        # 启动调度器守护进程并追踪任务
        scheduler_task = asyncio.create_task(self._scheduler_daemon())
        self._tasks.append(scheduler_task)

    async def _stop(self):
        """实现停止所有服务"""
        # 停止API服务器
        await self.api_server.stop()
        # 基类的stop方法会处理所有task的取消和清理

    async def _scheduler_daemon(self):
        """调度器守护进程"""
        try:
            while self._running:
                await self._schedule()
                await asyncio.sleep(MANAGER_SCHEDULER_INTERVAL)
        except asyncio.CancelledError:
            self.logger.info("Scheduler daemon cancelled")
        except Exception as e:
            self.logger.error(f"Scheduler daemon error: {e}")
            raise

    async def _schedule(self):
        if self.scheduler.is_queue_full():
            cluster_manager = await self._build_scheduler_cluster_manager()
            task_record = await self._build_scheduler_task_record()
            task_queue = self.scheduler.task_queue
            self.file_system_manager.set_file_system(FS_CONFIG_PATH, cluster_manager)
            task_wrap_runtime_info, _ = self.scheduler.schedule(
                SchedulerContext(
                    current_time=time.time(),
                    cluster_manager=cluster_manager,
                    task_record=task_record,
                    file_system=self.file_system_manager.get_file_system(),
                    task_queue=task_queue,
                )
            )
            await self.task_manager.add_task_info(task_wrap_runtime_info)
            # gpu_id -> cluster_id
            cluster_id = await self.cluster_manager.get_cluster_by_gpu_id(
                task_wrap_runtime_info.schedule_infos[0].gpu_id
            )
            master_client = await self.cluster_manager.get_master_client_by_cluster_id(cluster_id)
            await master_client.submit_task(task_wrap_runtime_info)
            self.logger.info(
                f"Scheduler scheduled task {task_wrap_runtime_info.task_meta.task_id} to cluster {cluster_id}"
            )

    async def _build_scheduler_cluster_manager(self) -> SchedulerClusterManager:
        scheduler_cluster_manager = SchedulerClusterManager.from_clusters(
            TypeConverter.convert_runtime_cluster_to_scheduler_cluster(cluster)
            for cluster in (await self.cluster_manager.snapshot()).values()
        )

        # TODO: build gpu_executor_map
        return scheduler_cluster_manager

    async def _build_scheduler_task_record(self) -> dict[str, SchedulerTaskWrapRuntimeInfo]:
        task_record = {}
        snapshot = await self.task_manager.snapshot()
        for task_id, task_info in snapshot.items():
            task_record[task_id] = (
                TypeConverter.convert_runtime_task_wrap_runtime_info_to_scheduler_task_wrap_runtime_info(task_info)
            )
        return task_record

    async def handle_task_submit(self, task_meta: TaskMeta):
        self.scheduler.submit_task(task_meta)

    async def handle_master_register(
        self, cluster: Cluster, task_infos: dict[str, RuntimeTaskWrapRuntimeInfo], master_info: ComponentInfo
    ) -> bool:
        self.logger.info(f"handle_master_register: {master_info.component_id}")
        await self.cluster_manager.register_master(master_info, cluster)
        for task_info in task_infos.values():
            await self.task_manager.add_task_info(task_info)


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
            return self.master_client_record[self.clusters[cluster_id].master_info.component_ip]

    async def get_cluster(self, cluster_id: str) -> Cluster:
        async with self.cluster_lock:
            return self.clusters[cluster_id]

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
        self.task_infos: dict[str, RuntimeTaskWrapRuntimeInfo] = {}

    async def add_task_info(self, task_info: RuntimeTaskWrapRuntimeInfo):
        async with self.task_lock:
            self.task_infos[task_info.task_meta.task_id] = task_info

    async def snapshot(self) -> dict[str, RuntimeTaskWrapRuntimeInfo]:
        async with self.task_lock:
            return self.task_infos


class FileSystemManager:
    def __init__(self):
        self.file_system: FileSystem = None

    def set_file_system(self, config_path: str, cluster_manager: ClusterManager):
        self.file_system = FileSystem(config_path, cluster_manager)

    def get_file_system(self) -> FileSystem:
        return self.file_system
