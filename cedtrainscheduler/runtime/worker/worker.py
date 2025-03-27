import asyncio
from asyncio import Lock

from cedtrainscheduler.runtime.components import BaseServer
from cedtrainscheduler.runtime.master.api_client import WorkerMasterClient
from cedtrainscheduler.runtime.types.args import WorkerArgs
from cedtrainscheduler.runtime.types.cluster import GPU
from cedtrainscheduler.runtime.types.cluster import Node
from cedtrainscheduler.runtime.types.task import TaskInst
from cedtrainscheduler.runtime.utils.gpu_util import GPUUtil
from cedtrainscheduler.runtime.utils.logger import setup_logger
from cedtrainscheduler.runtime.worker.api_server import WorkerAPIServer
from cedtrainscheduler.runtime.worker.executor import Executor
from cedtrainscheduler.runtime.worker.service import WorkerService

WORKER_HEARTBEAT_INTERVAL = 5


class Worker(BaseServer, WorkerService):
    def __init__(self, worker_args: WorkerArgs):
        super().__init__(worker_args.worker_info)
        self.worker_info = worker_args.worker_info
        self.master_info = worker_args.master_info

        self.node = Node()
        self.node_lock = Lock()

        self.executors: dict[str, Executor] = {}
        self.api_server = WorkerAPIServer(self)

        self.worker_client = WorkerMasterClient(self.master_info.component_ip, self.master_info.component_port)

        self._init_node(worker_args)
        self._init_executor()

        self.logger = setup_logger(__name__)

    def _init_node(self, worker_args: WorkerArgs):
        self.node.node_id = worker_args.worker_info.component_id
        self.node.ip = worker_args.worker_info.component_ip
        self.node.port = worker_args.worker_info.component_port
        self.node.cluster_id = worker_args.cluster_id

        # init_gpu
        self.node.gpus = {}
        gpu_ids = GPUUtil.get_gpus(self.node.node_id)
        for id, gpu_id in enumerate(gpu_ids):
            self.node.gpus[gpu_id] = GPU(
                gpu_id=gpu_id, gpu_type=worker_args.gpu_type, gpu_rank=id, node_id=self.node.node_id
            )

    def _init_executor(self):
        for gpu_id, gpu in self.node.gpus.items():
            self.executors[gpu_id] = Executor(gpu)

    async def _start(self):
        api_server_task = await self.api_server.start()
        self._tasks.append(api_server_task)

        heartbeat_task = asyncio.create_task(self._heartbeat_daemon())
        self._tasks.append(heartbeat_task)

    async def _stop(self):
        """停止Worker服务"""
        await self.api_server.stop()

    async def _heartbeat_daemon(self):
        try:
            while self._running:
                await self._heartbeat()
                await asyncio.sleep(WORKER_HEARTBEAT_INTERVAL)
        except asyncio.CancelledError:
            self.logger.info("Heartbeat daemon cancelled")
        except Exception as e:
            self.logger.error(f"Heartbeat daemon error: {e}")
            raise

    async def _heartbeat(self):
        task_record = []
        for executor in self.executors.values():
            task_record.extend(await executor.get_task_record())

        task_queue_map = {gpu_id: executor.task_queue for gpu_id, executor in self.executors.items()}
        response = await self.worker_client.register_worker(self.node, task_record, task_queue_map)
        self.logger.info(
            f"Worker registered to Master {self.master_info.component_ip}:{self.master_info.component_port}:"
            f"Response: {response}"
        )

    async def handle_task_inst_submit(self, task_inst: TaskInst, gpu_id: str):
        executor = self.executors[gpu_id]
        await executor.append_task(task_inst)

    async def handle_task_inst_start(
        self, task_inst: TaskInst, gpu_id: str, task_name: str, world_size: int, inst_rank: int
    ):
        executor = self.executors[gpu_id]
        await executor.start_task(task_name=task_name, task_inst=task_inst, world_size=world_size, inst_rank=inst_rank)
