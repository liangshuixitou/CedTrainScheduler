import asyncio
import logging
import time
from asyncio import Lock

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
        self.ip = self.master_info.component_ip
        self.port = self.master_info.component_port

        self.manager_client = MasterManagerClient(self.master_info.component_ip, self.master_info.component_port)

        self.cluster = Cluster(
            cluster_id=self.master_info.component_id,
            cluster_name=self.master_args.cluster_name,
            cluster_type=self.master_args.cluster_type,
            nodes=[],
        )
        self.cluster_lock = Lock()

        self.worker_manager = WorkerManager()
        self.task_manager = TaskManager()
        self.api_server = MasterAPIServer(self)

        self.logger = logging.getLogger(__name__)

    async def _serve(self):
        await self.api_server.start(host=self.master_info.component_ip, port=self.master_info.component_port)
        await self._start_heartbeat_daemon()

    async def stop(self):
        """停止Master服务"""
        await self.api_server.stop()
        await super().stop()

    async def _start_heartbeat_daemon(self):
        async def heartbeat_loop():
            while True:
                self._heartbeat()
                self.logger.info(f"Master heartbeat: {self.cluster}")
                await asyncio.sleep(MASTER_HEARTBEAT_INTERVAL)

        self.heartbeat_thread = asyncio.create_task(heartbeat_loop())

    async def _heartbeat(self):
        task_record = await self.task_manager.get_task_record()
        self.manager_client.register_master(self.cluster, task_record, self.master_info)

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
            worker_ip = self._find_worker_with_gpu(gpu_id)
            if not worker_ip:
                self.logger.error(f"No worker found with GPU {gpu_id} for task {task_id}, inst {inst_id}")
                continue

            # Send the task instance to the worker
            worker_client = self.worker_manager.get_worker_client_by_ip(worker_ip)
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
                self.task_manager.update_task_inst(task_inst)
                task = await self.task_manager.get_task(task_inst.task_id)

                if task_inst.inst_status == TaskInstStatus.Ready:
                    is_task_ready = await self.task_manager.check_task_ready(task_inst.task_id)
                    if is_task_ready:
                        await self.task_manager.update_task_status(task_inst.task_id, TaskStatus.Ready)
                        asyncio.create_task(self._start_task(task))
                elif task_inst.inst_status == TaskInstStatus.Running:
                    is_task_running = await self.task_manager.check_task_running(task_inst.task_id)
                    if is_task_running:
                        await self.task_manager.update_task_status(task_inst.task_id, TaskStatus.Running)
                        await self.task_manager.update_task_time(task_inst.task_id, time.time())
                elif task_inst.inst_status == TaskInstStatus.Finished:
                    is_task_finished = await self.task_manager.check_task_finished(task_inst.task_id)
                    if is_task_finished:
                        await self.task_manager.update_task_status(task_inst.task_id, TaskStatus.Finished)
                        await self.task_manager.update_task_time(task_inst.task_id, time.time())
            return {"status": "success", "message": "Worker registered"}

    async def _start_task(self, task: TaskWrapRuntimeInfo):
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
            worker_client = self.worker_manager.get_worker_client_by_ip(worker_ip)
            if not worker_client:
                self.logger.error(f"Worker client not found for worker {worker_ip}")
                continue

            # Prepare instance info for the worker
            inst_data = TaskInst(
                task_id=task.task_meta.task_id,
                inst_id=inst_id,
                inst_status=TaskInstStatus.Pending,
            )

            await worker_client.start_task_inst(
                task_inst=inst_data,
                gpu_id=gpu_id,
                task_name=task.task_meta.task_name,
                world_size=len(schedule_infos),
                inst_rank=inst_id,
                master_addr=self.ip,  # TODO: 需要修改为master的地址
                master_port=self.port,  # TODO: 需要修改为master的端口
            )
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

    def get_worker_client_by_id(self, worker_id: str) -> MasterWorkerClient:
        with self.worker_lock:
            worker_ip = self.worker_record[worker_id].component_ip
            return self.worker_client_record[worker_ip]

    def get_worker_client_by_ip(self, worker_ip: str) -> MasterWorkerClient:
        with self.worker_lock:
            return self.worker_client_record[worker_ip]


class TaskManager:
    def __init__(self):
        self.task_record: dict[str, TaskWrapRuntimeInfo] = {}
        self.task_lock = Lock()

    async def add_task(self, task: TaskWrapRuntimeInfo):
        with self.task_lock:
            self.task_record[task.task_meta.task_id] = task

    async def get_task(self, task_id: str) -> TaskWrapRuntimeInfo:
        with self.task_lock:
            return self.task_record[task_id]

    async def get_task_record(self) -> dict[str, TaskWrapRuntimeInfo]:
        with self.task_lock:
            return self.task_record

    async def update_task_status(self, task_id: str, task_status: TaskStatus):
        with self.task_lock:
            self.task_record[task_id].task_meta.task_status = task_status

    async def update_task_time(self, task_id: str, task_time: float):
        with self.task_lock:
            task = self.task_record[task_id]
            if task.task_meta.task_status == TaskStatus.Finished:
                task.task_meta.task_end_time = task_time
            elif task.task_meta.task_status == TaskStatus.Running:
                task.task_meta.task_start_time = task_time
            elif task.task_meta.task_status == TaskStatus.Pending:
                task.task_meta.task_submit_time = task_time

    async def update_task_inst(self, task_inst: TaskInst):
        with self.task_lock:
            self.task_record[task_inst.task_id].inst_status[task_inst.inst_id] = task_inst.inst_status

    async def check_task_ready(self, task_id: str) -> bool:
        with self.task_lock:
            task = self.task_record[task_id]
            for inst in task.task_insts:
                if inst.inst_status != TaskInstStatus.Ready:
                    return False
            return True

    async def check_task_running(self, task_id: str) -> bool:
        with self.task_lock:
            task = self.task_record[task_id]
            for inst in task.task_insts:
                if inst.inst_status != TaskInstStatus.Running:
                    return False
            return True


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Start the Master service")
    parser.add_argument("--id", default="master", help="Master component ID")
    parser.add_argument("--ip", default="127.0.0.1", help="Master IP address")
    parser.add_argument("--port", type=int, default=5000, help="Master port")
    parser.add_argument("--manager-id", default="manager", help="Manager component ID")
    parser.add_argument("--manager-ip", default="127.0.0.1", help="Manager IP address")
    parser.add_argument("--manager-port", type=int, default=5001, help="Manager port")
    parser.add_argument("--cluster-name", default="cluster", help="Cluster name")
    parser.add_argument(
        "--cluster-type", default="CLOUD", choices=["CLOUD", "LOCAL"], help="Cluster type (CLOUD or LOCAL)"
    )

    args = parser.parse_args()

    master = Master(
        MasterArgs(
            master_info=ComponentInfo(
                component_id=args.id,
                component_ip=args.ip,
                component_port=args.port,
                component_type=ComponentType.MASTER,
            ),
            manager_info=ComponentInfo(
                component_id=args.manager_id,
                component_ip=args.manager_ip,
                component_port=args.manager_port,
                component_type=ComponentType.MANAGER,
            ),
            cluster_name=args.cluster_name,
            cluster_type=args.cluster_type,
        )
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    await master.start()


if __name__ == "__main__":
    asyncio.run(main())
