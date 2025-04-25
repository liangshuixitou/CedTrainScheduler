import asyncio
from asyncio import Lock

from cedtrainscheduler.runtime.components import BaseServer
from cedtrainscheduler.runtime.master.api_client import WorkerMasterClient
from cedtrainscheduler.runtime.types.args import WorkerArgs
from cedtrainscheduler.runtime.types.cluster import GPU
from cedtrainscheduler.runtime.types.cluster import Node
from cedtrainscheduler.runtime.types.task import TaskInst
from cedtrainscheduler.runtime.types.task import TaskInstStatus
from cedtrainscheduler.runtime.utils.gpu_util import GPUUtil
from cedtrainscheduler.runtime.utils.logger import setup_logger
from cedtrainscheduler.runtime.worker.api_server import WorkerAPIServer
from cedtrainscheduler.runtime.worker.executor import Executor
from cedtrainscheduler.runtime.worker.service import WorkerService
from cedtrainscheduler.runtime.workload.workload import WorkloadType

WORKER_HEARTBEAT_INTERVAL = 5


class Worker(BaseServer, WorkerService):
    def __init__(self, worker_args: WorkerArgs):
        super().__init__(worker_args.worker_info)
        self.worker_info = worker_args.worker_info
        self.master_info = worker_args.master_info

        self.node: Node = None
        self.node_lock = Lock()

        self.executors: dict[str, Executor] = {}
        self.api_server = WorkerAPIServer(self)

        self.worker_client = WorkerMasterClient(self.master_info.component_ip, self.master_info.component_port)

        self.sim_mode: bool = worker_args.sim_gpu_num != 0
        self._init_node(worker_args)
        self._init_executor()

        self.logger = setup_logger(__name__)

    def _init_node(self, worker_args: WorkerArgs):
        self.node = Node(
            node_id=worker_args.worker_info.component_id,
            ip=worker_args.worker_info.component_ip,
            port=worker_args.worker_info.component_port,
            cluster_id=worker_args.master_info.component_id,
            gpus={},
        )

        # init_gpu
        self.node.gpus = {}
        if self.sim_mode:
            gpu_num = worker_args.sim_gpu_num
            gpus = GPUUtil.get_gpus_with_num(self.node.node_id, gpu_num)
        elif worker_args.gpu_ids:
            gpus = GPUUtil.get_gpus_with_ids(self.node.node_id, worker_args.gpu_ids)
        else:
            gpu_num = GPUUtil.get_gpu_count()
            gpus = GPUUtil.get_gpus_with_num(self.node.node_id, gpu_num)
        for gpu_id, gpu_rank in gpus.items():
            self.node.gpus[gpu_id] = GPU(
                gpu_id=gpu_id, gpu_type=worker_args.gpu_type, gpu_rank=gpu_rank, node_id=self.node.node_id
            )

    def _init_executor(self):
        for gpu_id, gpu in self.node.gpus.items():
            self.executors[gpu_id] = Executor(gpu)

    async def _start(self):
        api_server_task = await self.api_server.start(port=self.worker_info.component_port)
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
            self.logger.exception(f"Heartbeat daemon error: {e}")
            raise

    async def _heartbeat(self):
        task_record = []
        for executor in self.executors.values():
            gpu_task_record = await executor.get_task_record()
            task_record.extend(gpu_task_record)
            self.logger.info(f"GPU {executor.gpu.gpu_id} task queue size: {len(executor.task_queue)}")

        task_queue_map = {gpu_id: executor.task_queue for gpu_id, executor in self.executors.items()}

        response = await self.worker_client.register_worker(self.node, task_record, task_queue_map)
        self.logger.info(
            f"Worker registered to Master {self.master_info.component_ip}:{self.master_info.component_port}:"
            f"Response: {response}"
        )

    async def handle_task_inst_submit(self, task_inst: TaskInst, gpu_id: str):
        executor = self.executors[gpu_id]
        await executor.append_task(task_inst)
        self.logger.info(f"Task {task_inst.task_id} instance {task_inst.inst_id} submitted to GPU {gpu_id}")

    async def handle_task_inst_start(
        self,
        task_inst: TaskInst,
        gpu_id: str,
        task_name: str,
        world_size: int,
        inst_rank: int,
        master_addr: str,
        master_port: int,
        plan_runtime: int,
        data_transfer_time: float,
    ):
        executor = self.executors[gpu_id]
        current_task_inst = executor.task_queue[0]
        if current_task_inst.task_id != task_inst.task_id or current_task_inst.inst_id != task_inst.inst_id:
            raise ValueError(f"Task inst mismatch: {current_task_inst} != {task_inst}")
        if current_task_inst.inst_status != TaskInstStatus.Ready:
            self.logger.info(f"Task inst {task_inst.task_id} is not ready, skip start")
            return

        self.logger.info(f"Task {task_inst.task_id} instance {task_inst.inst_id} start on GPU {gpu_id} ")

        if self.sim_mode:
            task_name = WorkloadType.SIM_WORKLOAD

        await executor.start_task(
            task_name=task_name,
            task_inst=task_inst,
            world_size=world_size,
            inst_rank=inst_rank,
            master_addr=master_addr,
            master_port=master_port,
            plan_runtime=plan_runtime,
            data_transfer_time=data_transfer_time,
        )
