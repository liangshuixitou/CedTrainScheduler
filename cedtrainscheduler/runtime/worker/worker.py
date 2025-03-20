import logging
import time
from concurrent.futures import Executor
from threading import Lock
from threading import Thread

from cedtrainscheduler.runtime.master.api_client import WorkerMasterClient
from cedtrainscheduler.runtime.types.args import WorkerArgs
from cedtrainscheduler.runtime.types.cluster import GPU
from cedtrainscheduler.runtime.types.cluster import Node
from cedtrainscheduler.runtime.types.task import TaskInst, TaskWrapRuntimeInfo
from cedtrainscheduler.runtime.utils.gpu_util import GPUUtil
from cedtrainscheduler.runtime.worker.api_server import WorkerAPIServer

WORKER_HEARTBEAT_INTERVAL = 5


class Worker:
    def __init__(self, worker_args: WorkerArgs):
        self.worker_info = worker_args.worker_info
        self.master_info = worker_args.master_info

        self.node = Node()
        self.node_lock = Lock()

        self.executors: dict[str, Executor] = {}
        self.api_server = WorkerAPIServer(self)
        self.logger = logging.getLogger(__name__)

        self.worker_client = WorkerMasterClient(self.master_info.component_ip, self.master_info.component_port)

        self._init_node(worker_args)
        self._init_executor()

        self.heartbeat_thread = None
        self.start()

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

    def start(self):
        """启动Master服务"""
        self.api_server.start(host=self.worker_info.component_ip, port=self.worker_info.component_port)
        self._start_heartbeat_daemon()

    def stop(self):
        """停止Master服务"""
        self.api_server.stop()

    def _start_heartbeat_daemon(self):
        def heartbeat_loop():
            while True:
                self._heartbeat()
                time.sleep(WORKER_HEARTBEAT_INTERVAL)

        heartbeat_thread = Thread(target=heartbeat_loop, daemon=True)
        heartbeat_thread.start()
        self.heartbeat_thread = heartbeat_thread

    def _heartbeat(self):
        with self.node_lock:
            # TODO: 从executor中获取task_record
            task_record = []
            response = self.worker_client.register_worker(self.node, task_record)
            if response:
                self.logger.info(
                    f"Worker registered to Master {self.master_info.component_ip}:"
                    f"{self.master_info.component_port}: {response}"
                )
            else:
                self.logger.error("Worker registration failed")

    async def handle_task_inst_submit(self, task_inst: TaskInst, gpu_id: str):
        with self.task_record_lock:
            self.task_record[gpu_id].append(task_inst)

    async def handle_task_inst_start(
        self, task_inst: TaskInst, gpu_id: str, task_record: dict[str, TaskWrapRuntimeInfo]
    ):
        pass
