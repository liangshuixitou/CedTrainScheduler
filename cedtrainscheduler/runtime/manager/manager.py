import asyncio
import logging
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
        self.logger = logging.getLogger(__name__)
        self._scheduler_task = None

    async def _serve(self):
        """实现Server基类的抽象方法，运行后台任务"""
        try:
            # 启动API服务器
            await self.api_server.start(host=self.component_info.component_ip, port=self.component_info.component_port)

            # 启动调度器守护进程
            self._scheduler_task = asyncio.create_task(self._scheduler_daemon())
            await self._scheduler_task
        except asyncio.CancelledError:
            self.logger.info("调度器服务被取消")
        except Exception as e:
            self.logger.error(f"调度器服务出错: {e}")
            raise

    async def _scheduler_daemon(self):
        """调度器守护进程"""
        try:
            while self._running:
                await self._schedule()
                await asyncio.sleep(MANAGER_SCHEDULER_INTERVAL)
        except asyncio.CancelledError:
            self.logger.info("调度器循环被取消")
        except Exception as e:
            self.logger.error(f"调度器循环出错: {e}")
            raise

    async def _schedule(self):
        if self.scheduler.is_queue_full():
            cluster_manager = self._build_scheduler_cluster_manager()
            task_record = self._build_scheduler_task_record()
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
            self.task_manager.add_task_info(task_wrap_runtime_info)
            # gpu_id -> cluster_id
            cluster_id = self.cluster_manager.get_cluster_by_gpu_id(task_wrap_runtime_info.schedule_infos[0].gpu_id)
            master_client = self.cluster_manager.get_master_client_by_cluster_id(cluster_id)
            await master_client.submit_task(task_wrap_runtime_info)
            self.logger.info(
                f"Scheduler scheduled task {task_wrap_runtime_info.task_meta.task_id} to cluster {cluster_id}"
            )

    async def stop(self):
        """重写Server基类的stop方法，添加Manager特定的清理逻辑"""
        # 取消调度器任务
        if self._scheduler_task and not self._scheduler_task.done():
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        # 停止API服务器
        await self.api_server.stop()

        # 调用基类的stop方法
        await super().stop()

    def _build_scheduler_cluster_manager(self) -> SchedulerClusterManager:
        return SchedulerClusterManager.from_clusters(
            TypeConverter.convert_runtime_cluster_to_scheduler_cluster(cluster)
            for cluster in self.cluster_manager.snapshot().values()
        )

    def _build_scheduler_task_record(self) -> dict[str, SchedulerTaskWrapRuntimeInfo]:
        task_record = {}
        snapshot = self.task_manager.snapshot()
        for task_id, task_info in snapshot.items():
            task_record[task_id] = (
                TypeConverter.convert_runtime_task_wrap_runtime_info_to_scheduler_task_wrap_runtime_info(task_info)
            )
        return task_record

    async def handle_task_submit(self, task_meta: TaskMeta):
        self.scheduler.submit_task(task_meta)

    async def handle_master_register(
        self, cluster: Cluster, task_infos: dict[str, RuntimeTaskWrapRuntimeInfo], master_info: ComponentInfo
    ):
        self.cluster_manager.register_master(master_info, cluster)
        for task_info in task_infos.values():
            self.task_manager.add_task_info(task_info)


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

    def snapshot(self) -> dict[str, Cluster]:
        with self.cluster_lock:
            return self.clusters


class TaskManager:
    def __init__(self):
        self.task_infos: dict[str, RuntimeTaskWrapRuntimeInfo] = {}
        self.task_lock = Lock()

    def add_task_info(self, task_info: RuntimeTaskWrapRuntimeInfo):
        with self.task_lock:
            self.task_infos[task_info.task_meta.task_id] = task_info

    def snapshot(self) -> dict[str, RuntimeTaskWrapRuntimeInfo]:
        with self.task_lock:
            return self.task_infos


class FileSystemManager:
    def __init__(self):
        self.file_system: FileSystem = None

    def set_file_system(self, config_path: str, cluster_manager: ClusterManager):
        self.file_system = FileSystem(config_path, cluster_manager)

    def get_file_system(self) -> FileSystem:
        return self.file_system


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Start the Manager service")
    parser.add_argument("--id", default="manager", help="Manager component ID")
    parser.add_argument("--ip", default="127.0.0.1", help="Manager IP address")
    parser.add_argument("--port", type=int, default=5001, help="Manager port")
    parser.add_argument("--scheduler-name", default="scheduler", help="Scheduler name")
    parser.add_argument("--cluster-name", default="cluster", help="Cluster name")

    args = parser.parse_args()

    manager = Manager(
        ManagerArgs(
            manager_info=ComponentInfo(
                component_id=args.id,
                component_ip=args.ip,
                component_port=args.port,
                component_type=ComponentType.MANAGER,
            ),
            scheduler_name=args.scheduler_name,
        )
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    await manager.run()


if __name__ == "__main__":
    asyncio.run(main())
