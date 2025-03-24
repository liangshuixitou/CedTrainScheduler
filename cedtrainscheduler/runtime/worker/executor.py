import asyncio
import logging

from cedtrainscheduler.runtime.types.cluster import GPU
from cedtrainscheduler.runtime.types.task import TaskInst
from cedtrainscheduler.runtime.workload.script import ScriptGenerator


class Executor:
    def __init__(self, gpu: GPU):
        self.gpu = gpu
        self.task_queue: list[TaskInst] = []
        self.task_queue_lock = asyncio.Lock()

        self.logger = logging.getLogger(__name__)


    async def append_task(self, task: TaskInst):
        async with self.task_queue_lock:
            self.task_queue.append(task)

    async def get_task_record(self) -> list[TaskInst]:
        async with self.task_queue_lock:
            return self.task_queue

    async def start_task(
        self,
        task_name: str,
        task_inst: TaskInst,
        world_size: int,
        inst_rank: int,
        master_addr: str,
        master_port: str,
    ):
        script = ScriptGenerator.generate_script(
            gpu_rank=self.gpu.gpu_rank,
            task_name=task_name,
            world_size=world_size,
            inst_rank=inst_rank,
            master_addr=master_addr,
            master_port=master_port,
        )
