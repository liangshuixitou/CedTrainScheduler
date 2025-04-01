import asyncio
import time
from asyncio import Lock
from typing import Optional

from cedtrainscheduler.runtime.components import BaseServer
from cedtrainscheduler.runtime.components import ComponentInfo
from cedtrainscheduler.runtime.components import ComponentType
from cedtrainscheduler.runtime.manager.api_client import MasterManagerClient
from cedtrainscheduler.runtime.master.api_server import MasterAPIServer
from cedtrainscheduler.runtime.master.service import MasterService
from cedtrainscheduler.runtime.types.args import MasterArgs
from cedtrainscheduler.runtime.types.cluster import Cluster
from cedtrainscheduler.runtime.types.cluster import Node
from cedtrainscheduler.runtime.types.task import TaskInst
from cedtrainscheduler.runtime.types.task import TaskInstStatus
from cedtrainscheduler.runtime.types.task import TaskStatus
from cedtrainscheduler.runtime.types.task import TaskWrapRuntimeInfo
from cedtrainscheduler.runtime.utils.ip_util import IPUtil
from cedtrainscheduler.runtime.utils.logger import setup_logger
from cedtrainscheduler.runtime.worker.api_client import MasterWorkerClient

MASTER_HEARTBEAT_INTERVAL = 5


class Master(BaseServer, MasterService):
    """
    Master组件，负责调度和管理任务
    """

    def __init__(self, master_args: MasterArgs):
        super().__init__(master_args.master_info)
        self.master_args = master_args
        self.master_info = self.master_args.master_info
        self.manager_info = self.master_args.manager_info
        self.ip = self.master_info.component_ip
        self.port = self.master_info.component_port

        self.manager_client = MasterManagerClient(self.manager_info.component_ip, self.manager_info.component_port)

        self.cluster_lock = Lock()

        self.worker_manager = WorkerManager(self.master_args)
        self.task_manager = TaskManager()
        self.api_server = MasterAPIServer(self)

        self.logger = setup_logger(__name__)

    async def _start(self):
        api_server_task = await self.api_server.start(port=self.master_info.component_port)
        self._tasks.append(api_server_task)

        heartbeat_task = asyncio.create_task(self._heartbeat_daemon())
        self._tasks.append(heartbeat_task)

    async def _stop(self):
        """停止Master服务"""
        await self.api_server.stop()

    async def _heartbeat_daemon(self):
        try:
            while self._running:
                await self._heartbeat()
                await asyncio.sleep(MASTER_HEARTBEAT_INTERVAL)
        except asyncio.CancelledError:
            self.logger.info("Heartbeat daemon cancelled")
        except Exception as e:
            self.logger.exception(f"Heartbeat daemon error: {e}")
            raise

    async def _heartbeat(self):
        cluster = await self.worker_manager.get_cluster()
        task_record = await self.task_manager.get_task_record()
        task_queue_map = await self.task_manager.get_task_queue_map()
        response = await self.manager_client.register_master(cluster, task_record, self.master_info, task_queue_map)
        self.logger.info(
            f"Master registered to Manager {self.manager_info.component_ip}:{self.manager_info.component_port}:"
            f"Response: {response}"
        )

    async def handle_task_submit(self, task_info: TaskWrapRuntimeInfo):
        task_id = task_info.task_meta.task_id
        self.logger.info(f"handle_task_submit: {task_id}")
        if not task_id:
            return {"status": "error", "message": "Missing task_id"}

        await self.task_manager.add_task(task_info)
        await self.task_manager.update_task_time(task_id, time.time())
        schedule_infos = task_info.schedule_infos
        for inst_id, schedule_info in schedule_infos.items():
            gpu_id = schedule_info.gpu_id
            # Find the worker that has the specified GPU
            worker_ip = await self.worker_manager.get_worker_ip_by_gpu_id(gpu_id)
            if not worker_ip:
                self.logger.error(f"No worker found with GPU {gpu_id} for task {task_id}, inst {inst_id}")
                continue

            # Send the task instance to the worker
            worker_client = await self.worker_manager.get_worker_client_by_ip(worker_ip)
            if not worker_client:
                self.logger.error(f"Worker client not found for worker {worker_ip}")
                continue

            # Prepare instance info for the worker
            inst_data = TaskInst(
                task_id=task_id,
                inst_id=inst_id,
                inst_status=TaskInstStatus.Pending,
            )

            await worker_client.submit_task(inst_data, gpu_id)
            self.logger.info(f"Task {task_id}, instance {inst_id} sent to worker {worker_ip}")

        return {"status": "success", "task_id": task_id}

    async def handle_worker_register(
        self, node: Node, task_insts: list[TaskInst], task_queue_map: dict[str, list[TaskInst]]
    ):
        # register worker
        await self.worker_manager.register_worker(node)

        await self.task_manager.extend_task_queue_map(task_queue_map)

        # record task info
        for task_inst in task_insts:
            await self.task_manager.update_task_inst(task_inst)
            task = await self.task_manager.get_task(task_inst.task_id)

            if task_inst.inst_status == TaskInstStatus.Ready:
                is_task_ready = await self.task_manager.check_task_ready(task_inst.task_id)
                if is_task_ready and task.task_meta.task_status != TaskStatus.Ready:
                    await self.task_manager.update_task_status(task_inst.task_id, TaskStatus.Ready)
                    asyncio.create_task(self._start_task(task))
            elif task_inst.inst_status == TaskInstStatus.Running:
                is_task_running = await self.task_manager.check_task_running(task_inst.task_id)
                if is_task_running and task.task_meta.task_status != TaskStatus.Running:
                    await self.task_manager.update_task_status(task_inst.task_id, TaskStatus.Running)
                    await self.task_manager.update_task_time(task_inst.task_id, time.time())
            elif task_inst.inst_status == TaskInstStatus.Finished:
                is_task_finished = await self.task_manager.check_task_finished(task_inst.task_id)
                if is_task_finished and task.task_meta.task_status != TaskStatus.Finished:
                    await self.task_manager.update_task_status(task_inst.task_id, TaskStatus.Finished)
                    await self.task_manager.update_task_time(task_inst.task_id, time.time())
        return {"status": "success", "message": "Worker registered"}

    async def _start_task(self, task: TaskWrapRuntimeInfo):
        first_worker_ip: Optional[str] = None

        schedule_infos = task.schedule_infos
        for inst_id, schedule_info in schedule_infos.items():
            gpu_id = schedule_info.gpu_id
            # Find the worker that has the specified GPU
            worker_ip = await self.worker_manager.get_worker_ip_by_gpu_id(gpu_id)
            if not worker_ip:
                self.logger.error(
                    f"No worker found with GPU {gpu_id} for task {task.task_meta.task_id}, inst {inst_id}"
                )
                continue

            # Send the task instance to the worker
            worker_client = await self.worker_manager.get_worker_client_by_ip(worker_ip)
            if not worker_client:
                self.logger.error(f"Worker client not found for worker {worker_ip}")
                continue

            if first_worker_ip is None:
                first_worker_ip = worker_ip

            # Prepare instance info for the worker
            inst_data = TaskInst(
                task_id=task.task_meta.task_id,
                inst_id=inst_id,
                inst_status=TaskInstStatus.Pending,
            )

            master_inst_ip = first_worker_ip
            master_inst_port = IPUtil.get_available_port(first_worker_ip)
            await worker_client.start_task_inst(
                task_inst=inst_data,
                gpu_id=gpu_id,
                task_name=task.task_meta.task_name,
                world_size=len(schedule_infos),
                inst_rank=inst_id,
                master_addr=master_inst_ip,
                master_port=master_inst_port,
            )
            self.logger.info(f"Task {task.task_meta.task_id}, instance {inst_id} sent to worker {worker_ip}")


class WorkerManager:
    """Worker管理器"""

    def __init__(self, master_args: MasterArgs):
        # dict[worker_id, worker_info]
        self.worker_record: dict[str, ComponentInfo] = {}
        # dict[worker_ip, worker_client]
        self.worker_client_record: dict[str, MasterWorkerClient] = {}
        self.cluster: Cluster = Cluster(
            cluster_id=master_args.master_info.component_id,
            cluster_name=master_args.cluster_name,
            cluster_type=master_args.cluster_type,
            nodes=dict(),
        )
        self.worker_lock = Lock()

    async def register_worker(self, node: Node):
        async with self.worker_lock:
            node_id = node.node_id
            self.worker_record[node.node_id] = ComponentInfo(
                component_type=ComponentType.WORKER,
                component_id=node.node_id,
                component_ip=node.ip,
                component_port=node.port,
            )
            self.worker_client_record[node.ip] = MasterWorkerClient(node.ip, node.port)
            node_id = node.node_id
            # record node info
            self.cluster.nodes[node_id] = node

    async def remove_worker(self, node_id: str):
        async with self.worker_lock:
            del self.worker_record[node_id]
            del self.worker_client_record[self.worker_record[node_id].component_ip]
            del self.cluster.nodes[node_id]

    async def get_node(self, node_id: str) -> Node:
        async with self.worker_lock:
            return self.cluster.nodes[node_id]

    async def get_cluster(self) -> Cluster:
        async with self.worker_lock:
            return self.cluster

    async def get_worker_client_by_id(self, worker_id: str) -> MasterWorkerClient:
        async with self.worker_lock:
            worker_ip = self.worker_record[worker_id].component_ip
            return self.worker_client_record[worker_ip]

    async def get_worker_client_by_ip(self, worker_ip: str) -> MasterWorkerClient:
        async with self.worker_lock:
            return self.worker_client_record[worker_ip]

    async def get_worker_ip_by_gpu_id(self, gpu_id: str) -> str:
        async with self.worker_lock:
            for node in self.cluster.nodes.values():
                if gpu_id in node.gpus.keys():
                    return node.ip
        return None


class TaskManager:
    def __init__(self):
        self.task_record: dict[str, TaskWrapRuntimeInfo] = {}
        # dict[gpu_id, list[TaskInst]]
        self.task_queue_map: dict[str, list[TaskInst]] = {}
        self.task_lock = Lock()

    async def add_task(self, task: TaskWrapRuntimeInfo):
        async with self.task_lock:
            self.task_record[task.task_meta.task_id] = task

    async def get_task(self, task_id: str) -> TaskWrapRuntimeInfo:
        async with self.task_lock:
            return self.task_record[task_id]

    async def get_task_record(self) -> dict[str, TaskWrapRuntimeInfo]:
        async with self.task_lock:
            return self.task_record

    async def update_task_status(self, task_id: str, task_status: TaskStatus):
        async with self.task_lock:
            self.task_record[task_id].task_meta.task_status = task_status

    async def update_task_time(self, task_id: str, task_time: float):
        async with self.task_lock:
            task = self.task_record[task_id]
            if task.task_meta.task_status == TaskStatus.Finished:
                task.task_end_time = task_time
            elif task.task_meta.task_status == TaskStatus.Running:
                task.task_start_time = task_time
            elif task.task_meta.task_status == TaskStatus.Pending:
                task.task_submit_time = task_time

    async def update_task_inst(self, task_inst: TaskInst):
        async with self.task_lock:
            self.task_record[task_inst.task_id].inst_status[task_inst.inst_id] = task_inst.inst_status

    async def check_task_ready(self, task_id: str) -> bool:
        async with self.task_lock:
            task = self.task_record[task_id]
            for inst_status in task.inst_status.values():
                if inst_status != TaskInstStatus.Ready:
                    return False
            return True

    async def check_task_running(self, task_id: str) -> bool:
        async with self.task_lock:
            task = self.task_record[task_id]
            for inst_status in task.inst_status.values():
                if inst_status != TaskInstStatus.Running:
                    return False
            return True

    async def check_task_finished(self, task_id: str) -> bool:
        async with self.task_lock:
            task = self.task_record[task_id]
            for inst_status in task.inst_status.values():
                if inst_status != TaskInstStatus.Finished:
                    return False
            return True

    async def extend_task_queue_map(self, task_queue_map: dict[str, list[TaskInst]]):
        async with self.task_lock:
            self.task_queue_map.update(task_queue_map)

    async def get_task_queue_map(self) -> dict[str, list[TaskInst]]:
        async with self.task_lock:
            return self.task_queue_map
