import asyncio
import time

from cedtrainscheduler.runtime.components import BaseServer
from cedtrainscheduler.runtime.components import ComponentInfo
from cedtrainscheduler.runtime.manager.api_server import ManagerAPIServer
from cedtrainscheduler.runtime.manager.components import ClusterManager
from cedtrainscheduler.runtime.manager.components import FileSystemManager
from cedtrainscheduler.runtime.manager.components import TaskManager
from cedtrainscheduler.runtime.manager.constant import FS_CONFIG_PATH
from cedtrainscheduler.runtime.manager.constant import TASK_RECORD_SAVE_PATH
from cedtrainscheduler.runtime.manager.service import ManagerService
from cedtrainscheduler.runtime.manager.utils import SchedulerUtils
from cedtrainscheduler.runtime.types.args import ManagerArgs
from cedtrainscheduler.runtime.types.cluster import Cluster
from cedtrainscheduler.runtime.types.task import TaskInst
from cedtrainscheduler.runtime.types.task import TaskMeta
from cedtrainscheduler.runtime.types.task import TaskWrapRuntimeInfo
from cedtrainscheduler.runtime.utils.logger import setup_logger
from cedtrainscheduler.runtime.utils.metric_util import calculate_task_metrics
from cedtrainscheduler.scheduler.factory import SchedulerFactory
from cedtrainscheduler.scheduler.types.scheduler_context import SchedulerContext

MANAGER_SCHEDULER_INTERVAL = 10
MANAGER_TASK_RECORD_SAVE_INTERVAL = 5


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

        self.task_scheduler_buffer_lock = asyncio.Lock()
        self.task_scheduler_buffer: list[TaskMeta] = []

        self.logger = setup_logger(__name__)

    async def _start(self):
        """实现启动所有服务"""
        # 启动API服务器并追踪任务
        api_server_task = await self.api_server.start(port=self.component_info.component_port)
        self._tasks.append(api_server_task)

        # 启动调度器守护进程并追踪任务
        scheduler_task = asyncio.create_task(self._scheduler_daemon())
        self._tasks.append(scheduler_task)

        # 任务日志保存
        task_record_save_task = asyncio.create_task(self._save_task_record_daemon())
        self._tasks.append(task_record_save_task)

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
            self.logger.exception(f"Scheduler daemon error: {e}")
            raise

    async def _save_task_record_daemon(self):
        try:
            while self._running:
                await self.task_manager.save()
                await asyncio.sleep(MANAGER_TASK_RECORD_SAVE_INTERVAL)
        except asyncio.CancelledError:
            self.logger.info("Task record save daemon cancelled")
        except Exception as e:
            self.logger.exception(f"Task record save daemon error: {e}")
            raise

    async def _schedule(self):
        async with self.task_scheduler_buffer_lock:
            while not self.scheduler.is_queue_full() and len(self.task_scheduler_buffer) > 0:
                scheduler_context = await self._build_scheduler_context()
                self.logger.info(scheduler_context.cluster_manager)
                task_meta = self.task_scheduler_buffer.pop(0)
                self.scheduler.submit_task(scheduler_context, task_meta)

        if len(self.scheduler.task_queue) > 0:
            scheduler_context = await self._build_scheduler_context()
            task_wrap_runtime_info, _ = self.scheduler.schedule(scheduler_context)
            await self.task_manager.add_task_info(task_wrap_runtime_info)
            # gpu_id -> cluster_id
            cluster = await self.cluster_manager.get_cluster_by_gpu_id(task_wrap_runtime_info.schedule_infos[0].gpu_id)
            master_client = await self.cluster_manager.get_master_client_by_cluster_id(cluster.cluster_id)
            sim_data_transfer_time = await self._calculate_data_transfer_time(
                task_wrap_runtime_info, cluster.cluster_id
            )
            await master_client.submit_task(task_wrap_runtime_info, sim_data_transfer_time)
            self.logger.info(
                f"Scheduler scheduled task {task_wrap_runtime_info.task_meta.task_id} to cluster {cluster.cluster_id}"
            )

    async def handle_task_submit(self, task_meta: TaskMeta):
        async with self.task_scheduler_buffer_lock:
            self.task_scheduler_buffer.append(task_meta)

    async def handle_master_register(
        self,
        cluster: Cluster,
        task_infos: dict[str, TaskWrapRuntimeInfo],
        master_info: ComponentInfo,
        task_queue_map: dict[str, list[TaskInst]],
    ) -> bool:
        self.logger.info(f"handle_master_register: {master_info.component_id}")
        await self.cluster_manager.register_master(master_info, cluster)
        for task_info in task_infos.values():
            await self.task_manager.add_task_info(task_info)
        await self.task_manager.extend_task_queue_map(task_queue_map)
        await self.task_manager.save()

    async def handle_metrics(self) -> dict:
        task_metrics = calculate_task_metrics(TASK_RECORD_SAVE_PATH)
        return task_metrics

    async def _calculate_data_transfer_time(
        self, task_wrap_runtime_info: TaskWrapRuntimeInfo, cluster_id: str
    ) -> float:
        cluster_manager = await SchedulerUtils.build_scheduler_cluster_manager(self.cluster_manager, self.task_manager)
        self.file_system_manager.set_file_system(FS_CONFIG_PATH, cluster_manager)
        file_system = self.file_system_manager.get_file_system()
        task_data_info = file_system.get_task_data_info(task_wrap_runtime_info.task_meta.task_name)
        data_transfer_time = file_system.get_data_arrival_time(
            task_data_info,
            cluster_id,
            cluster_manager,
        )
        return data_transfer_time

    async def _build_scheduler_context(self) -> SchedulerContext:
        cluster_manager = await SchedulerUtils.build_scheduler_cluster_manager(self.cluster_manager, self.task_manager)
        task_record = await SchedulerUtils.build_scheduler_task_record(self.task_manager)
        task_queue = self.scheduler.task_queue
        self.file_system_manager.set_file_system(FS_CONFIG_PATH, cluster_manager)
        file_system = self.file_system_manager.get_file_system()
        return SchedulerContext(
            current_time=time.time(),
            cluster_manager=cluster_manager,
            task_record=task_record,
            file_system=file_system,
            task_queue=task_queue,
        )
