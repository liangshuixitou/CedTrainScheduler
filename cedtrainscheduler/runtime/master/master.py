import asyncio
import logging
from asyncio import Lock

from cedtrainscheduler.runtime.components import BaseComponent
from cedtrainscheduler.runtime.components import ComponentInfo
from cedtrainscheduler.runtime.components import ComponentType
from cedtrainscheduler.runtime.master.api_server import MasterAPIServer
from cedtrainscheduler.runtime.types.args import MasterArgs
from cedtrainscheduler.runtime.types.cluster import Cluster
from cedtrainscheduler.runtime.types.cluster import ClusterType
from cedtrainscheduler.runtime.types.cluster import Node
from cedtrainscheduler.runtime.types.task import TaskInst
from cedtrainscheduler.runtime.types.task import TaskInstStatus
from cedtrainscheduler.runtime.types.task import TaskWrapRuntimeInfo
from cedtrainscheduler.runtime.worker.api_client import MasterWorkerClient


class Master(BaseComponent):
    """
    Master组件，负责调度和管理任务
    """

    def __init__(self, master_args: MasterArgs):
        super().__init__(master_args.master_info.component_id)
        self.ip = master_args.master_info.component_ip
        self.port = master_args.master_info.component_port

        self.cluster = Cluster(
            cluster_id=master_args.master_info.component_id,
            cluster_name=master_args.cluster_name,
            cluster_type=master_args.cluster_type,
            nodes=[],
        )
        self.task_record: dict[str, TaskWrapRuntimeInfo] = {}
        self.cluster_lock = Lock()
        self.task_record_lock = Lock()

        self.worker_manager = WorkerManager()
        self.api_server = MasterAPIServer(self)

        self.logger = logging.getLogger(__name__)

    async def start(self):
        """启动Master服务"""
        await self.api_server.start(host=self.ip, port=self.port)

    async def stop(self):
        """停止Master服务"""
        await self.api_server.stop()

    async def handle_task_submit(self, task_info: TaskWrapRuntimeInfo):
        async with self.task_record_lock:
            task_id = task_info.task_meta.task_id
            self.logger.info(f"handle_task_submit: {task_id}")
            if not task_id:
                return {"status": "error", "message": "Missing task_id"}

            self.task_record[task_id] = task_info

        schedule_infos = task_info.schedule_infos
        for inst_id, schedule_info in schedule_infos.items():
            gpu_id = schedule_info.gpu_id
            # Find the worker that has the specified GPU
            worker_ip = self._find_worker_with_gpu(gpu_id)
            if not worker_ip:
                self.logger.error(f"No worker found with GPU {gpu_id} for task {task_id}, inst {inst_id}")
                continue

            # Send the task instance to the worker
            worker_client = self.worker_manager.get_worker_by_ip(worker_ip)
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

    async def handle_worker_register(self, node: Node, task_insts: list[TaskInst]):
        async with self.cluster_lock:
            node_id = node.node_id
            self.logger.info(f"handle_worker_register: {node_id}")
            if not node_id:
                return {"status": "error", "message": "Missing node_id"}

            # record node info
            self.cluster.nodes[node_id] = node

            # register worker
            worker_info = ComponentInfo(
                component_type=ComponentType.Worker,
                component_id=node.node_id,
                component_ip=node.ip,
                component_port=node.port,
            )
            self.worker_manager.remove_worker(node.node_id)
            self.worker_manager.register_worker(worker_info)

            # record task info
            for task_inst in task_insts:
                task = self.task_record[task_inst.task_id]
                task.inst_status[task_inst.inst_id] = task_inst.inst_status

                if task_inst.inst_status != TaskInstStatus.Ready:
                    continue

                is_task_ready = True
                for task_inst in task.task_insts:
                    if task_inst.inst_status != TaskInstStatus.Ready:
                        is_task_ready = False
                        break
                if is_task_ready:
                    self.start_task(task)

            return {"status": "success", "message": "Worker registered"}

    async def start_task(self, task: TaskWrapRuntimeInfo):
        schedule_infos = task.schedule_infos
        for inst_id, schedule_info in schedule_infos.items():
            gpu_id = schedule_info.gpu_id
            # Find the worker that has the specified GPU
            worker_ip = self._find_worker_with_gpu(gpu_id)
            if not worker_ip:
                self.logger.error(
                    f"No worker found with GPU {gpu_id} for task {task.task_meta.task_id}, inst {inst_id}"
                )
                continue

            # Send the task instance to the worker
            worker_client = self.worker_manager.get_worker_by_ip(worker_ip)
            if not worker_client:
                self.logger.error(f"Worker client not found for worker {worker_ip}")
                continue

            # Prepare instance info for the worker
            inst_data = TaskInst(
                task_id=task.task_meta.task_id,
                inst_id=inst_id,
                inst_status=TaskInstStatus.Pending,
            )

            await worker_client.start_task_inst(inst_data, gpu_id, self.task_record)
            self.logger.info(f"Task {task.task_meta.task_id}, instance {inst_id} sent to worker {worker_ip}")

    def _find_worker_with_gpu(self, gpu_id: str) -> str:
        """Find a worker that has the specified GPU"""
        with self.cluster_lock:
            for node in self.cluster.nodes.values():
                if gpu_id in node.gpus.keys():
                    return node.ip
        return None


class WorkerManager:
    """Worker管理器"""

    def __init__(self):
        # dict[worker_id, worker_info]
        self.worker_record: dict[str, ComponentInfo] = {}
        # dict[worker_ip, worker_client]
        self.worker_client_record: dict[str, MasterWorkerClient] = {}
        self.worker_lock = Lock()

    def register_worker(self, worker: ComponentInfo):
        with self.worker_lock:
            self.worker_record[worker.component_id] = worker
            self.worker_client_record[worker.component_ip] = MasterWorkerClient(
                worker.component_ip, worker.component_port
            )

    def remove_worker(self, worker_id: str):
        with self.worker_lock:
            del self.worker_record[worker_id]
            del self.worker_client_record[self.worker_record[worker_id].component_ip]

    def get_worker(self, worker_id: str) -> ComponentInfo:
        with self.worker_lock:
            return self.worker_record[worker_id]

    def get_worker_by_id(self, worker_id: str) -> MasterWorkerClient:
        with self.worker_lock:
            worker_ip = self.worker_record[worker_id].component_ip
            return self.worker_client_record[worker_ip]

    def get_worker_by_ip(self, worker_ip: str) -> MasterWorkerClient:
        with self.worker_lock:
            return self.worker_client_record[worker_ip]


async def main():
    master = Master(
        MasterArgs(
            master_info=ComponentInfo(component_id="master1", component_ip="127.0.0.1", component_port=5000),
            cluster_name="cluster1",
            cluster_type=ClusterType.CLOUD,
        )
    )
    await master.start()


if __name__ == "__main__":
    asyncio.run(main())
