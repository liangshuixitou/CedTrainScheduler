import asyncio

from cedtrainscheduler.runtime.types.cluster import GPU
from cedtrainscheduler.runtime.types.task import TaskInst
from cedtrainscheduler.runtime.types.task import TaskInstStatus
from cedtrainscheduler.runtime.utils.logger import setup_logger
from cedtrainscheduler.runtime.worker.backend.backend import Backend


class Executor:
    def __init__(self, gpu: GPU, backend: Backend):
        self.gpu = gpu
        self.backend = backend
        self.task_record: list[TaskInst] = []
        self.task_queue: list[TaskInst] = []
        self.task_record_lock = asyncio.Lock()
        self.logger = setup_logger(__name__)

    async def append_task(self, task: TaskInst):
        async with self.task_record_lock:
            if len(self.task_queue) == 0:
                task.inst_status = TaskInstStatus.Ready
            self.task_record.append(task)
            self.task_queue.append(task)
            self.logger.info(f"[GPU {self.gpu.gpu_id}] append task {task.task_id}, instance {task.inst_id}")

    async def get_task_record(self) -> list[TaskInst]:
        async with self.task_record_lock:
            return self.task_record

    async def task_finished(self, current_task_inst: TaskInst):
        current_task_inst.inst_status = TaskInstStatus.Finished
        async with self.task_record_lock:
            self.task_queue.pop(0)
            if len(self.task_queue) > 0:
                next_task_inst = self.task_queue[0]
                next_task_inst.inst_status = TaskInstStatus.Ready
                self.logger.info(
                    f"[GPU {self.gpu.gpu_id}] task {next_task_inst.task_id}, instance {next_task_inst.inst_id} is ready"
                )

    async def start_task(
        self,
        task_name: str,
        task_inst: TaskInst,
        world_size: int,
        inst_rank: int,
        master_addr: str,
        master_port: int,
        plan_runtime: int,
        data_transfer_time: float,
    ):
        async with self.task_record_lock:
            current_task_inst = self.task_queue[0]

        if current_task_inst.inst_id != task_inst.inst_id or current_task_inst.inst_status != TaskInstStatus.Ready:
            self.logger.error(
                f"[GPU {self.gpu.gpu_id}] Task {task_inst.task_id} instance {task_inst.inst_id} is not ready"
            )
            return

        current_task_inst.inst_status = TaskInstStatus.Running

        try:
            process = await self.backend.execute_task(
                task_name=task_name,
                task_inst=task_inst,
                world_size=world_size,
                inst_rank=inst_rank,
                master_addr=master_addr,
                master_port=master_port,
                plan_runtime=plan_runtime,
                data_transfer_time=data_transfer_time,
            )

            # Start monitoring the task
            asyncio.create_task(self._monitor_task_completion(process, current_task_inst))

        except Exception as e:
            self.logger.error(f"[GPU {self.gpu.gpu_id}] Error starting task {task_inst.task_id}: {str(e)}")
            await self.task_finished(current_task_inst)

    async def _monitor_task_completion(self, process: asyncio.subprocess.Process, task_inst: TaskInst):
        try:
            await self.backend.monitor_task(process, task_inst)
            await self.task_finished(task_inst)
        except Exception as e:
            self.logger.error(f"[GPU {self.gpu.gpu_id}] Error monitoring task {task_inst.task_id}: {str(e)}")
            await self.task_finished(task_inst)
